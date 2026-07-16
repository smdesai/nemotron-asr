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

    public func startListening() async throws {
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

    public var errorDescription: String? {
        switch self {
        case .modelsNotBundled:
            return "Nemotron ASR models are not bundled in this build."
        case .variantNotBundled(let ship, let tier):
            return "Nemotron ASR model variant \(ship)/\(tier) is not bundled in this build."
        }
    }
}
