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
    /// 0...1 while the model variant is downloading, nil otherwise.
    public private(set) var prepFraction: Double? {
        didSet { onPreparationProgressChange?(prepFraction) }
    }

    public var onTranscriptChange: ((String) -> Void)?
    public var onPhaseChange: ((Phase) -> Void)?
    public var onPreparationMessageChange: ((String) -> Void)?
    public var onPreparationProgressChange: ((Double?) -> Void)?

    private let languageCode: String?
    private let chunkMs: Int
    /// Optional pre-provisioned models root laid out as `<ship>/<tier>ms/...` (for example an
    /// app bundle's `Models` folder). When nil, variants are downloaded from the Hugging Face
    /// Hub into `NemotronModelDownloader.rootDirectory()` on first use.
    private let modelsRoot: URL?
    private let downloader = NemotronModelDownloader()
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

    /// - Parameters:
    ///   - languageCode: Language hint (nil = auto). Latin-script hints prefer the `latin`
    ///     ship when it exists; everything else uses `multilingual`.
    ///   - chunkMs: Streaming chunk tier (560, 1120, 2240 or 4480).
    ///   - modelsRoot: Directory already containing `<ship>/<tier>ms/...` (each tier either
    ///     flat or with a `coreml/` subfolder). Pass nil to download from the Hugging Face Hub.
    public init(languageCode: String? = nil, chunkMs: Int = 2240, modelsRoot: URL? = nil) {
        self.languageCode = languageCode
        self.chunkMs = chunkMs
        self.modelsRoot = modelsRoot
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

        let variantDir = try await resolveVariantDirectory(code: code, chunkMs: chunkMs)
        // Bundled layouts keep the .mlmodelc trees under `coreml/`; the Hub layout is flat.
        // metadata.json / tokenizer.json always sit at the tier root.
        let nested = variantDir.appendingPathComponent("coreml", isDirectory: true)
        let coremlDir = FileManager.default.fileExists(atPath: nested.path) ? nested : variantDir
        prepMessage = "Loading speech model…"
        let shared = try await StreamingNemotronMultilingualAsrManager.preloadShared(
            from: coremlDir,
            commonDirectory: variantDir
        )
        sharedCache[key] = shared
        return shared
    }

    /// Locates `<ship>/<tier>ms` for the requested language: from `modelsRoot` when one was
    /// supplied, otherwise from the Hugging Face download cache (downloading on first use).
    /// Falls back from the language-specific ship to `multilingual` when the former is absent.
    private func resolveVariantDirectory(code: String?, chunkMs: Int) async throws -> URL {
        let preferredShip = StreamingNemotronMultilingualAsrManager.languageDirectory(for: code ?? "auto")
        let tier = "\(chunkMs)ms"
        let ships = preferredShip == "multilingual" ? ["multilingual"] : [preferredShip, "multilingual"]

        if let modelsRoot {
            for ship in ships {
                let dir = modelsRoot
                    .appendingPathComponent(ship, isDirectory: true)
                    .appendingPathComponent(tier, isDirectory: true)
                if FileManager.default.fileExists(atPath: dir.appendingPathComponent("metadata.json").path) {
                    return dir
                }
            }
            throw TranscriberError.variantNotBundled(ship: preferredShip, tier: tier)
        }

        for ship in ships {
            do {
                return try await downloadVariant(ship: ship, chunkMs: chunkMs)
            } catch NemotronModelDownloadError.variantNotAvailable {
                continue
            }
        }
        throw TranscriberError.variantNotBundled(ship: preferredShip, tier: tier)
    }

    private func downloadVariant(ship: String, chunkMs: Int) async throws -> URL {
        if NemotronModelDownloader.isInstalled(ship: ship, chunkMs: chunkMs) {
            return NemotronModelDownloader.variantDirectory(ship: ship, chunkMs: chunkMs)
        }
        prepMessage = "Downloading speech model…"
        prepFraction = 0
        defer { prepFraction = nil }
        return try await downloader.ensureInstalled(ship: ship, chunkMs: chunkMs) { progress in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let fraction = progress.fractionCompleted {
                    self.prepFraction = fraction
                    let mb = Double(progress.bytesTotal) / 1_048_576
                    let size = mb >= 1000 ? String(format: "%.1f GB", mb / 1024) : String(format: "%.0f MB", mb)
                    self.prepMessage = "Downloading speech model… \(Int(fraction * 100))% of \(size)"
                }
            }
        }
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
            return "Nemotron ASR models are not available."
        case .variantNotBundled(let ship, let tier):
            return
                "Nemotron ASR model variant \(ship)/\(tier) is neither available locally nor published in \(NemotronModelDownloader.repoId)."
        case .noRecordedAudio:
            return
                "No recorded audio is available to export. Start listening with retainAudio: true first."
        }
    }
}
