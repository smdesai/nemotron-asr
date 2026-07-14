import AVFoundation
import Foundation

// AudioConverter is vendored into this target (NemotronWatch/Vendored) — no
// external FluidAudio package.
#if canImport(FluidAudio)
import FluidAudio
#endif

/// Captures microphone audio, resamples it to 16 kHz mono `[Float]`, and
/// delivers it **in order** via an `AsyncStream`. Also reports a smoothed
/// input level for live waveform visualisation.
///
/// Ordering matters for streaming ASR — the realtime audio tap runs on a
/// high-priority audio thread, so we resample synchronously there (cheap,
/// `AudioConverter` is `Sendable`) and yield into a stream whose consumer
/// feeds the ASR actor one buffer at a time.
final class MicrophoneCapture: @unchecked Sendable {
    enum CaptureError: LocalizedError {
        case permissionDenied
        case engineFailed(String)

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Microphone access was denied. Enable it in Settings ▸ Privacy ▸ Microphone."
            case .engineFailed(let msg):
                return "Could not start the microphone: \(msg)"
            }
        }
    }

    private let engine = AVAudioEngine()
    private let converter = AudioConverter()  // defaults to 16 kHz mono Float32
    private var continuation: AsyncStream<[Float]>.Continuation?

    /// Called on the audio thread with a smoothed 0...1 level. Hop to main inside.
    var onLevel: (@Sendable (Float) -> Void)?

    private var smoothedLevel: Float = 0

    /// Request microphone permission (async).
    static func requestPermission() async -> Bool {
        await withCheckedContinuation { cont in
            AVAudioApplication.requestRecordPermission { granted in
                cont.resume(returning: granted)
            }
        }
    }

    static var permission: AVAudioApplication.recordPermission {
        AVAudioApplication.shared.recordPermission
    }

    /// Configure the audio session and start the engine. Returns a stream of
    /// 16 kHz mono sample blocks. Throws if permission is missing or the
    /// engine fails to start.
    func start() async throws -> AsyncStream<[Float]> {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, Error>) in
                session.activate(options: []) { activated, error in
                    if let error {
                        continuation.resume(throwing: error)
                    } else if activated {
                        continuation.resume()
                    } else {
                        continuation.resume(
                            throwing: CaptureError.engineFailed(
                                "The audio session could not be activated."
                            ))
                    }
                }
            }
        } catch {
            throw CaptureError.engineFailed(error.localizedDescription)
        }
        #endif

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        // Guard against a zero/invalid format (can happen if the session
        // didn't activate or no input route exists).
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw CaptureError.engineFailed("No active microphone input was found.")
        }

        let stream = AsyncStream<[Float]> { continuation in
            self.continuation = continuation
        }

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            if let samples = try? self.converter.resampleBuffer(buffer), !samples.isEmpty {
                // Derive the meter level from the resampled samples — these are
                // always non-interleaved Float32, so the level tracks the exact
                // audio being transcribed regardless of the hardware input
                // format (raw `floatChannelData` can be nil for non-float inputs,
                // which previously froze the waveform at zero).
                self.publishLevel(fromSamples: samples)
                self.continuation?.yield(samples)
            } else {
                self.publishLevel(from: buffer)
            }
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            continuation?.finish()
            throw CaptureError.engineFailed(error.localizedDescription)
        }
        return stream
    }

    func stop() async {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        continuation?.finish()
        continuation = nil
        #if os(iOS)
        await withCheckedContinuation {
            (continuation: CheckedContinuation<Void, Never>) in
            AVAudioSession.sharedInstance().deactivate(
                options: [.notifyOthersOnDeactivation]
            ) { _, _ in
                continuation.resume()
            }
        }
        #endif
        onLevel?(0)
    }

    private func publishLevel(from buffer: AVAudioPCMBuffer) {
        guard let channel = buffer.floatChannelData?[0] else { return }
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return }
        var sum: Float = 0
        for i in 0 ..< frames {
            let s = channel[i]
            sum += s * s
        }
        publishLevel(rms: sqrt(sum / Float(frames)))
    }

    /// Level from already-resampled 16 kHz mono samples. Used as the primary
    /// path so the meter never depends on the raw tap's channel layout.
    private func publishLevel(fromSamples samples: [Float]) {
        guard !samples.isEmpty else { return }
        var sum: Float = 0
        for s in samples { sum += s * s }
        publishLevel(rms: sqrt(sum / Float(samples.count)))
    }

    private func publishLevel(rms: Float) {
        // Map RMS to a 0...1 range. Speech RMS is typically small, so apply a
        // high gain to make the meter swing across most of its height — the
        // amplitude (and thus audio changes) is much easier to see.
        let scaled = min(1, max(0, rms * 28))
        // Exponential smoothing for a less jittery meter. Weighted toward the
        // newest sample so swings register quickly.
        smoothedLevel = smoothedLevel * 0.6 + scaled * 0.4
        onLevel?(smoothedLevel)
    }
}
