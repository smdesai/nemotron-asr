import AVFoundation
import Accelerate
@preconcurrency import CoreML
import Foundation
import SwiftUI

/// watchOS orchestrator for the Nemotron multilingual streaming ASR. Loads the
/// bundled CoreML `2240ms` multilingual models on launch and drives live
/// microphone transcription, publishing everything the SwiftUI view needs.
///
/// Runs the vendored FluidAudio CoreML pipeline (split encoder on the ANE +
/// bare decoder/joint on CPU). The Core AI (.aimodel) path was abandoned for
/// now — the watchOS 27 beta's Core AI compiler has no m11 SoC backend
/// ("Unsupported SoC (m11)"); revisit on future betas.
@MainActor
final class WatchASRManager: ObservableObject {

    enum Phase: Equatable {
        case loading  // loading models on launch
        case ready  // models loaded, awaiting input
        case starting  // activating the microphone session
        case listening  // microphone is live
        case stopping  // draining the final ASR chunk
        case failed(String)
    }

    @Published private(set) var phase: Phase = .loading
    /// Running (streamed) transcript shown in the ScrollView.
    @Published var transcript: String = ""
    /// Language tag emitted by the ASR decoder (for example, "en-US").
    /// Sentiment analysis uses this instead of re-detecting short phrases.
    @Published private(set) var detectedLanguageCode: String?
    /// Status line for the UI.
    @Published private(set) var status: String = "Loading models…"
    /// Smoothed mic input level 0...1 driving the record button's level ring.
    @Published private(set) var micLevel: Float = 0
    /// Wall-clock seconds the pipeline spent inside the last mic block that
    /// processed at least one full encoder chunk — the UI's live "on-device
    /// inference" stat.
    @Published private(set) var lastChunkSeconds: Double?

    private let logger = AppLogger(category: "WatchASR")

    private var shared: SharedNemotronMultilingualModels?
    private var manager: StreamingNemotronMultilingualAsrManager?

    private var micCapture: WatchMicrophoneCapture?
    private var micTask: Task<Void, Never>?

    var isReady: Bool {
        if case .ready = phase { return true }
        return false
    }

    var isListening: Bool {
        if case .listening = phase { return true }
        return false
    }

    var isTransitioning: Bool {
        switch phase {
        case .starting, .stopping: return true
        default: return false
        }
    }

    // MARK: - Model loading (on launch)

    /// Locate the bundled model directory. Models are staged into a
    /// `NemotronWatchModels` folder reference, so they land in the bundle root
    /// under that subdirectory.
    private static func resourceDirectory() -> URL {
        if let url = Bundle.main.url(forResource: "NemotronWatchModels", withExtension: nil) {
            return url
        }
        // Fallback: bundle root (flat copy).
        return Bundle.main.resourceURL ?? Bundle.main.bundleURL
    }

    /// Load the bundled multilingual `2240ms` models. Called once on launch.
    func load() async {
        guard manager == nil else { return }
        // NOTE: Core AI probe disabled — watchOS 27 / m11 exposes only CPU (no ANE)
        // and the CPU compile path crashes in libODIECompiler. Re-enable
        // `WatchCoreAIProbe.run()` (+ stage a shard) to re-test on a future beta.
        phase = .loading
        status = "Loading models…"

        let dir = Self.resourceDirectory()
        let metadata = dir.appendingPathComponent(
            ModelNames.NemotronMultilingualStreaming.metadata)
        guard FileManager.default.fileExists(atPath: metadata.path) else {
            phase = .failed("Bundled models not found.")
            status = "Models missing from bundle."
            return
        }

        // CPU-only base configuration (memory budget); the vendored loader
        // pins the encoder shards to the ANE on watchOS — CPU-only encoding
        // was too slow to keep up with streaming.
        let cpuOnly = MLModelConfigurationUtils.defaultConfiguration(computeUnits: .cpuOnly)

        do {
            let shared = try await StreamingNemotronMultilingualAsrManager.preloadShared(
                from: dir,
                configuration: cpuOnly
            )
            let mgr = StreamingNemotronMultilingualAsrManager()
            try await mgr.loadFromShared(shared)
            await mgr.setLanguage(nil)  // auto-detect

            self.shared = shared
            self.manager = mgr
            phase = .ready
            status = "Ready"
        } catch {
            logger.error("CoreML load failed: \(error)")
            phase = .failed(friendly(error))
            status = "Load failed."
        }
    }

    // MARK: - Microphone streaming

    func start() async {
        guard case .ready = phase, let manager else { return }

        // Permission gate.
        let granted: Bool
        if WatchMicrophoneCapture.permission == .granted {
            granted = true
        } else {
            granted = await WatchMicrophoneCapture.requestPermission()
        }
        guard granted else {
            phase = .failed(
                WatchMicrophoneCapture.CaptureError.permissionDenied.localizedDescription)
            status = "Microphone denied."
            return
        }

        phase = .starting
        status = "Starting microphone…"
        transcript = ""
        detectedLanguageCode = nil

        // Live partials hop to the main actor to update @Published transcript.
        await manager.setPartialCallback { [weak self] text in
            Task { @MainActor in self?.transcript = text }
        }
        await manager.reset()

        let capture = WatchMicrophoneCapture()
        self.micCapture = capture

        do {
            let stream = try await capture.start()
            phase = .listening
            status = "Listening…"
            micTask = Task { [weak self] in
                guard let self else { return }
                do {
                    for await block in stream {
                        if Task.isCancelled { break }
                        self.updateMicLevel(block)
                        let started = Date()
                        _ = try await manager.process(samples: block)
                        let partial = await manager.getPartialTranscript()
                        if !partial.isEmpty {
                            self.transcript = partial
                        }
                        if let language = await manager.detectedLanguage() {
                            self.detectedLanguageCode = language
                        }
                        let elapsed = Date().timeIntervalSince(started)
                        // Mic blocks are ~0.26 s; only blocks that ran a full
                        // encoder chunk take meaningful time. Surface those.
                        if elapsed > 0.2 {
                            self.lastChunkSeconds = elapsed
                        }
                    }
                } catch  where Task.isCancelled || error is CancellationError {
                    return
                } catch {
                    let message = self.friendly(error)
                    self.logger.error("process(samples:) failed: \(error)")
                    self.phase = .failed(message)
                    self.status = message
                }
            }
        } catch {
            logger.error("microphone start failed: \(error)")
            phase = .failed(friendly(error))
            status = friendly(error)
            micCapture = nil
        }
    }

    private func updateMicLevel(_ block: [Float]) {
        guard !block.isEmpty else { return }
        var rms: Float = 0
        vDSP_rmsqv(block, 1, &rms, vDSP_Length(block.count))
        micLevel = min(1, micLevel * 0.7 + min(1, rms * 8) * 0.3)
    }

    func stop() async {
        guard case .listening = phase else { return }
        phase = .stopping
        status = "Finishing…"

        let processingTask = micTask
        processingTask?.cancel()
        micTask = nil
        let capture = micCapture
        micCapture = nil
        await capture?.stop()
        await processingTask?.value
        micLevel = 0

        guard let manager else {
            phase = .ready
            status = "Ready"
            return
        }
        do {
            let finalText = try await manager.finish()
            if !finalText.isEmpty { transcript = finalText }
            detectedLanguageCode = await manager.detectedLanguage()
        } catch {
            // Keep whatever partial we have; nothing fatal on stop.
        }
        phase = .ready
        status = "Ready"
    }

    // MARK: - Helpers

    func clearTranscript() {
        transcript = ""
        detectedLanguageCode = nil
        lastChunkSeconds = nil
    }

    private func friendly(_ error: Error) -> String {
        let raw = error.localizedDescription
        if raw.localizedCaseInsensitiveContains("Apple Silicon") {
            return "This model needs an Apple-Silicon device."
        }
        return raw
    }
}
