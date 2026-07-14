import AVFoundation
import Foundation

/// Captures microphone audio on watchOS, resamples it to 16 kHz mono `[Float]`,
/// and delivers it **in order** via an `AsyncStream`. Ported from the iOS
/// `MicrophoneCapture` with the `AVAudioSession` configuration extended to
/// watchOS.
///
/// Ordering matters for streaming ASR — the realtime audio tap runs on a
/// high-priority audio thread, so we resample synchronously there (cheap,
/// `AudioConverter` is `Sendable`) and yield into a stream whose consumer feeds
/// the ASR actor one buffer at a time.
final class WatchMicrophoneCapture: @unchecked Sendable {
    enum CaptureError: LocalizedError {
        case permissionDenied
        case engineFailed(String)

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Microphone access was denied. Enable it in the Watch Settings ▸ Privacy."
            case .engineFailed(let msg):
                return "Could not start the microphone: \(msg)"
            }
        }
    }

    private let engine = AVAudioEngine()
    private let converter = AudioConverter()  // defaults to 16 kHz mono Float32
    private var continuation: AsyncStream<[Float]>.Continuation?

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
    /// 16 kHz mono sample blocks. Throws if the engine fails to start.
    func start() async throws -> AsyncStream<[Float]> {
        #if os(iOS) || os(watchOS)
        let session = AVAudioSession.sharedInstance()
        do {
            // `.measurement` mode + `.duckOthers` mirror the iOS app. If watchOS
            // rejects them at runtime, fall back to a plain record session.
            do {
                try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            } catch {
                try session.setCategory(.record, mode: .default)
            }
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
        // Guard against a zero/invalid format (can happen if the session didn't
        // activate or no input route exists).
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw CaptureError.engineFailed("No active microphone input was found.")
        }

        let stream = AsyncStream<[Float]> { continuation in
            self.continuation = continuation
        }

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            if let samples = try? self.converter.resampleBuffer(buffer), !samples.isEmpty {
                self.continuation?.yield(samples)
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
        #if os(iOS) || os(watchOS)
        await withCheckedContinuation {
            (continuation: CheckedContinuation<Void, Never>) in
            AVAudioSession.sharedInstance().deactivate(
                options: [.notifyOthersOnDeactivation]
            ) { _, _ in
                continuation.resume()
            }
        }
        #endif
    }
}
