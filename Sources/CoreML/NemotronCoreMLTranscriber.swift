import AVFoundation
import Foundation

/// Small public CoreML-only facade over the vendored Nemotron multilingual ASR
/// runtime. It is intentionally UI-free so host apps can embed the package and
/// drive their own command surfaces.
@MainActor
public final class NemotronCoreMLTranscriber {
    public enum Phase: Equatable {
        case idle
        case preparing
        case ready
        case listening
        case failed(String)
    }

    public private(set) var phase: Phase = .idle {
        didSet { onPhaseChange?(phase) }
    }
    public private(set) var transcript: String = "" {
        didSet { onTranscriptChange?(transcript) }
    }
    public private(set) var prepMessage: String = "" {
        didSet { onPreparationMessageChange?(prepMessage) }
    }

    public var onTranscriptChange: ((String) -> Void)?
    public var onPhaseChange: ((Phase) -> Void)?
    public var onPreparationMessageChange: ((String) -> Void)?

    private let languageCode: String?
    private let chunkMs: Int
    private var sharedCache: [String: SharedNemotronMultilingualModels] = [:]
    private var loadedVariantKey: String?
    private var manager: StreamingNemotronMultilingualAsrManager?
    private var micCapture: MicrophoneCapture?
    private var micTask: Task<Void, Never>?

    /// Whether the current/most recent listening session should retain raw
    /// samples for later export.
    private var retainAudio: Bool = false
    /// Raw 16 kHz mono samples captured during a retaining session, in the
    /// order they were produced. Cap enforced in `startListening`'s mic loop.
    private var retainedSamples: [Float] = []
    /// Peak absolute amplitude seen across the retaining session's samples.
    /// Lets hosts distinguish "mic captured silence" (e.g. an idle virtual
    /// input device) from "speech wasn't recognized".
    public private(set) var recordedPeakAmplitude: Float = 0
    /// Hard cap on retained samples: 30 minutes at 16 kHz.
    private static let maxRetainedSamples = 16_000 * 60 * 30

    public init(languageCode: String? = nil, chunkMs: Int = 2240) {
        self.languageCode = languageCode
        self.chunkMs = chunkMs
    }

    public var isPreparing: Bool {
        if case .preparing = phase { return true }
        return false
    }

    public var isListening: Bool {
        if case .listening = phase { return true }
        return false
    }

    public func prepareModelIfNeeded() async throws {
        let key = variantKey(code: languageCode, chunkMs: chunkMs)
        if loadedVariantKey == key, manager != nil {
            if phase == .idle { phase = .ready }
            return
        }

        phase = .preparing
        prepMessage = "Loading speech model…"

        let shared = try await loadShared(code: languageCode, chunkMs: chunkMs, key: key)
        let newManager = manager ?? StreamingNemotronMultilingualAsrManager()
        try await newManager.loadFromShared(shared)
        await newManager.setLanguage(languageCode)
        self.manager = newManager
        self.loadedVariantKey = key
        prepMessage = "Speech model ready"
        phase = .ready
    }

    /// Starts a listening session.
    /// - Parameter retainAudio: When `true`, raw captured samples are kept
    ///   in memory (up to a 30-minute cap) so the session's audio can later
    ///   be exported via `exportRecording(to:)`. Defaults to `false` for
    ///   source compatibility with existing callers.
    public func startListening(retainAudio: Bool = false) async throws {
        guard !isListening else { return }

        let granted: Bool
        if MicrophoneCapture.permission == .granted {
            granted = true
        } else {
            granted = await MicrophoneCapture.requestPermission()
        }
        guard granted else { throw MicrophoneCapture.CaptureError.permissionDenied }

        try await prepareModelIfNeeded()
        guard let manager, phase == .ready else { return }

        transcript = ""
        self.retainAudio = retainAudio
        retainedSamples.removeAll()
        recordedPeakAmplitude = 0
        await manager.setPartialCallback { [weak self] text in
            Task { @MainActor in self?.transcript = text }
        }
        await manager.reset()

        let capture = MicrophoneCapture()
        micCapture = capture
        let stream = try await capture.start()
        phase = .listening
        micTask = Task { [weak self] in
            guard let self else { return }
            do {
                for await block in stream {
                    if Task.isCancelled { break }
                    if self.retainAudio {
                        self.recordedPeakAmplitude = block.reduce(
                            self.recordedPeakAmplitude
                        ) { max($0, abs($1)) }
                        if self.retainedSamples.count < Self.maxRetainedSamples {
                            // Cap retention at 30 minutes of audio; drop further
                            // samples from the retained buffer once reached.
                            self.retainedSamples.append(contentsOf: block)
                        }
                    }
                    _ = try await manager.process(samples: block)
                }
            } catch {
                await MainActor.run {
                    self.phase = .failed(error.localizedDescription)
                }
            }
        }
    }

    @discardableResult
    public func stopListening() async -> String {
        guard isListening else { return transcript.trimmingCharacters(in: .whitespacesAndNewlines) }

        await micCapture?.stop()
        micTask?.cancel()
        micTask = nil
        micCapture = nil

        let finalText: String
        if let manager {
            finalText = (try? await manager.finish()) ?? transcript
        } else {
            finalText = transcript
        }

        if !finalText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            transcript = finalText
        }
        phase = .ready
        return transcript.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func cancelListening() async {
        guard isListening else { return }
        await micCapture?.stop()
        micTask?.cancel()
        micTask = nil
        micCapture = nil
        phase = .ready
    }

    /// Duration, in seconds, of the samples currently retained from the most
    /// recent `startListening(retainAudio: true)` session. Zero if no audio
    /// has been retained.
    public var recordedDuration: TimeInterval {
        Double(retainedSamples.count) / 16_000
    }

    /// Writes the samples retained from the most recent
    /// `startListening(retainAudio: true)` session to `url` as a 16-bit PCM,
    /// mono, 16 kHz WAV file. The retained samples are left untouched, so
    /// this may be called more than once.
    public func exportRecording(to url: URL) throws {
        guard !retainedSamples.isEmpty else {
            throw TranscriberError.noRecordedAudio
        }

        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        )!
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(retainedSamples.count)
            )
        else {
            throw TranscriberError.noRecordedAudio
        }
        buffer.frameLength = buffer.frameCapacity
        retainedSamples.withUnsafeBufferPointer { source in
            buffer.floatChannelData![0].update(from: source.baseAddress!, count: source.count)
        }

        let outputSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let file = try AVAudioFile(forWriting: url, settings: outputSettings)
        try file.write(from: buffer)
    }

    private func loadShared(code: String?, chunkMs: Int, key: String) async throws
        -> SharedNemotronMultilingualModels
    {
        if let cached = sharedCache[key] { return cached }

        let variantDir = try bundledVariantDirectory(code: code, chunkMs: chunkMs)
        let coremlDir = variantDir.appendingPathComponent("coreml", isDirectory: true)
        let shared = try await StreamingNemotronMultilingualAsrManager.preloadShared(
            from: coremlDir,
            commonDirectory: variantDir
        )
        sharedCache[key] = shared
        return shared
    }

    private func bundledVariantDirectory(code: String?, chunkMs: Int) throws -> URL {
        let preferredShip = StreamingNemotronMultilingualAsrManager.languageDirectory(for: code ?? "auto")
        let tier = "\(chunkMs)ms"
        guard let modelsRoot = Bundle.module.resourceURL?.appendingPathComponent("Models") else {
            throw TranscriberError.modelsNotBundled
        }

        let preferred = modelsRoot
            .appendingPathComponent(preferredShip, isDirectory: true)
            .appendingPathComponent(tier, isDirectory: true)
        if FileManager.default.fileExists(atPath: preferred.appendingPathComponent("metadata.json").path) {
            return preferred
        }

        let multilingual = modelsRoot
            .appendingPathComponent("multilingual", isDirectory: true)
            .appendingPathComponent(tier, isDirectory: true)
        if FileManager.default.fileExists(atPath: multilingual.appendingPathComponent("metadata.json").path) {
            return multilingual
        }

        throw TranscriberError.variantNotBundled(ship: preferredShip, tier: tier)
    }

    private func variantKey(code: String?, chunkMs: Int) -> String {
        let ship = StreamingNemotronMultilingualAsrManager.languageDirectory(for: code ?? "auto")
        return "\(ship)/\(chunkMs)"
    }
}

public enum TranscriberError: LocalizedError {
    case modelsNotBundled
    case variantNotBundled(ship: String, tier: String)
    case noRecordedAudio

    public var errorDescription: String? {
        switch self {
        case .modelsNotBundled:
            return "Nemotron ASR models are not bundled in this build."
        case .variantNotBundled(let ship, let tier):
            return "Nemotron ASR model variant \(ship)/\(tier) is not bundled in this build."
        case .noRecordedAudio:
            return
                "No recorded audio is available to export. Start listening with retainAudio: true first."
        }
    }
}
