import AVFoundation
import Foundation
import SwiftUI

/// Drives the Nemotron multilingual streaming ASR for both audio-file and
/// microphone input, and publishes everything the UI needs to render.
///
/// Both backends sit behind `ASRSession`, so the file and mic flows are shared:
///   - CoreML: split encoder (pre-encode + 4 shards) + smart-speculative decode.
///   - Core AI: 4 int8 `.aimodel` encoder shards + fused `decoder_joint.aimodel`,
///     with `CoreAIComputePolicy` choosing ANE-preferred or CPU-only per tier.
@MainActor
final class TranscriptionEngine: ObservableObject {

    // MARK: Published UI state

    enum Phase: Equatable {
        case idle  // nothing loaded yet
        case preparing  // loading models
        case ready  // models loaded, awaiting input
        case transcribingFile  // processing an audio file
        case listening  // microphone is live
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle

    /// Running transcript (streamed) or final transcript (at-end).
    @Published var transcript: String = "" {
        didSet {
            // Mirror live transcript into the Live Activity (the controller
            // throttles) — both for the mic and for streamed file transcription.
            // `isListening` is true only for the mic (drives the Stop button).
            if phase == .listening || phase == .transcribingFile {
                liveActivity.update(
                    transcript: transcript,
                    isListening: phase == .listening,
                    language: liveActivityLanguageLabel
                )
            }
        }
    }
    /// Live partial text being appended during streaming.
    @Published private(set) var isStreaming: Bool = false
    /// Language the model auto-detected, if any (friendly name).
    @Published private(set) var detectedLanguage: String?

    /// Model-preparation progress (load + specialize).
    @Published private(set) var prepFraction: Double = 0
    @Published private(set) var prepMessage: String = ""

    /// Audio-file processing progress 0...1 (published in ≥1% steps).
    @Published private(set) var fileProgress: Double = 0

    /// Smoothed mic input level 0...1 for the waveform.
    @Published private(set) var micLevel: Float = 0

    /// Real-time factor of the last completed run (xRT), for a little stat chip.
    @Published private(set) var lastRTFx: Double?
    /// Per-stage CoreML timing of the last file run (prep / encoder / decoder
    /// seconds + chunk count), for BenchmarkRunner. nil on the Core AI path.
    private(set) var lastStageTimes: String?

    /// Core AI encoder residency report (compute types + dtype histogram per
    /// shard), surfaced in Settings for debugging. nil when unavailable.
    @Published private(set) var coreAIResidency: String?

    /// Live Activity start/end result, surfaced in Settings for on-device
    /// diagnosis (whether the activity was requested, denied, or failed).
    @Published private(set) var liveActivityStatus: String = "idle"

    // MARK: Dependencies

    private let settings: AppSettings

    /// Drives the lock-screen / Dynamic Island Live Activity during sessions.
    private let liveActivity = LiveActivityController()

    /// Friendly language label for the Live Activity: the detected language when
    /// available, else the pinned language name (nil while auto-detecting).
    private var liveActivityLanguageLabel: String? {
        detectedLanguage ?? (settings.languageCode == nil ? nil : settings.language.name)
    }

    // MARK: Model state

    /// The prepared pipeline for the current backend + tier.
    private var session: ASRSession?
    /// "<backend>|<ship>/<chunkMs>" that `session` was prepared for.
    private var sessionKey: String?
    /// Language code currently applied to `session` (outer nil = not applied).
    private var appliedLanguageCode: String??
    /// Loaded CoreML model bundles keyed by "<ship>/<chunkMs>". Holds at most the
    /// current variant; emptied when the Core AI backend is active.
    private var sharedCache: [String: SharedNemotronMultilingualModels] = [:]

    private var micCapture: MicrophoneCapture?
    private var micTask: Task<Void, Never>?
    private var isStopping = false

    private var prepareInFlight: Task<Void, Never>?

    init(settings: AppSettings) {
        self.settings = settings
        // The Live Activity's Stop button posts this notification from the app
        // process; end the current mic session in response.
        NotificationCenter.default.addObserver(
            forName: .stopRecordingRequested, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in await self?.stopListening() }
        }
    }

    var isBusy: Bool {
        switch phase {
        case .preparing, .transcribingFile, .listening: return true
        default: return false
        }
    }

    // MARK: - Variant resolution

    /// A resolved `<ship>/<tier>ms` model directory.
    private struct ModelTier {
        let ship: String
        let chunkMs: Int
        let directory: URL
        var variant: String { "\(ship)/\(chunkMs)" }
    }

    /// Root of the `<ship>/<tier>ms/` model tree: the app bundle's `Models/`, or —
    /// for on-device A/B benchmarking without reinstalling — a tree copied into
    /// the app container and named by the `NEMOTRON_MODELS_ROOT` launch env var
    /// (path relative to the container, e.g. `Documents/Models`).
    private static let modelsRoot: URL? = {
        if let rel = ProcessInfo.processInfo.environment["NEMOTRON_MODELS_ROOT"], !rel.isEmpty {
            let url =
                rel.hasPrefix("/")
                ? URL(fileURLWithPath: rel)
                : URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(rel)
            print("[Engine] models root override: \(url.path)")
            return url
        }
        return Bundle.main.resourceURL?.appendingPathComponent("Models")
    }()

    /// Resolve the tier directory for a language + chunk size. The language's
    /// preferred ship ("latin" for en/es/fr/it/pt/de, else "multilingual") is
    /// used when bundled; otherwise falls back to "multilingual". Each tier holds
    /// `metadata.json` + `tokenizer.json` with `coreml/` and `coreai/` subdirs.
    private static func tierDirectory(code: String?, chunkMs: Int) throws -> ModelTier {
        let preferred = StreamingNemotronMultilingualAsrManager.languageDirectory(
            for: code ?? "auto")
        let tierName = "\(chunkMs)ms"
        if let root = modelsRoot {
            for ship in preferred == "multilingual" ? [preferred] : [preferred, "multilingual"] {
                let dir = root.appendingPathComponent(ship, isDirectory: true)
                    .appendingPathComponent(tierName, isDirectory: true)
                if FileManager.default.fileExists(
                    atPath: dir.appendingPathComponent("metadata.json").path)
                {
                    return ModelTier(ship: ship, chunkMs: chunkMs, directory: dir)
                }
            }
        }
        throw EngineError.modelNotBundled(ship: preferred, tier: tierName)
    }

    private static func sessionKey(backend: InferenceBackend, tier: ModelTier) -> String {
        "\(backend.rawValue)|\(tier.variant)"
    }

    /// Session key the current settings resolve to (nil if no tier is bundled).
    private var desiredSessionKey: String? {
        (try? Self.tierDirectory(
            code: settings.languageCode, chunkMs: settings.chunkSize.rawValue))
            .map { Self.sessionKey(backend: settings.backend, tier: $0) }
    }

    // MARK: - Model preparation

    /// Ensure a session is loaded for the current backend + tier. Safe to call
    /// repeatedly — reuses the loaded session and only reloads when the
    /// backend or tier changes (a language change just re-applies the hint).
    func prepareModelIfNeeded() async {
        // Serialize: a second caller arriving mid-load (e.g. the view's .task and
        // BenchmarkRunner at launch) would otherwise start a second full load in
        // parallel. Wait for the in-flight preparation, then re-check.
        while let inFlight = prepareInFlight {
            await inFlight.value
            // Clear a FINISHED task ourselves: awaiting a completed Task returns
            // without suspending, so waiting for its creator to clear it would
            // spin this loop on the main actor and starve the creator.
            if prepareInFlight == inFlight { prepareInFlight = nil }
        }
        let task = Task { await self.prepareModelIfNeededSerialized() }
        prepareInFlight = task
        await task.value
        if prepareInFlight == task { prepareInFlight = nil }
    }

    private func prepareModelIfNeededSerialized() async {
        let backend = settings.backend
        let code = settings.languageCode
        let tier: ModelTier
        do {
            tier = try Self.tierDirectory(code: code, chunkMs: settings.chunkSize.rawValue)
        } catch {
            phase = .failed(friendly(error))
            return
        }
        let key = Self.sessionKey(backend: backend, tier: tier)

        // Already loaded for this backend + tier — just (re)apply the language.
        if key == sessionKey, let session {
            await applyLanguageIfNeeded(code, to: session)
            if phase == .idle { phase = .ready }
            return
        }

        phase = .preparing
        prepFraction = 0
        prepMessage = "Preparing model…"

        // Release the previous pipeline before loading the next: lowers peak
        // memory and frees the inactive backend's models on a backend switch.
        releaseSession()
        if backend == .coreml {
            sharedCache = sharedCache.filter { $0.key == tier.variant }
        } else {
            sharedCache.removeAll()
        }

        do {
            let newSession: ASRSession
            switch backend {
            case .coreml: newSession = try await makeCoreMLSession(tier: tier)
            case .coreai: newSession = try await makeCoreAISession(tier: tier)
            }
            session = newSession
            sessionKey = key
            await applyLanguageIfNeeded(code, to: newSession)
            phase = .ready
            prepFraction = 1
        } catch {
            phase = .failed(backend == .coreai ? "Core AI: \(friendly(error))" : friendly(error))
        }
    }

    private func makeCoreMLSession(tier: ModelTier) async throws -> ASRSession {
        prepMessage = "Loading model…"
        let shared: SharedNemotronMultilingualModels
        if let cached = sharedCache[tier.variant] {
            shared = cached
        } else {
            prepFraction = 0.15
            // CoreML artifacts live in `coreml/`; metadata.json / tokenizer.json
            // stay at the tier root.
            shared = try await StreamingNemotronMultilingualAsrManager.preloadShared(
                from: tier.directory.appendingPathComponent("coreml", isDirectory: true),
                commonDirectory: tier.directory
            )
            sharedCache[tier.variant] = shared
        }
        prepFraction = 0.8
        let manager = StreamingNemotronMultilingualAsrManager()
        try await manager.loadFromShared(shared)
        prepMessage = "Ready"
        return CoreMLSession(manager: manager)
    }

    private func makeCoreAISession(tier: ModelTier) async throws -> ASRSession {
        let dir = tier.directory.appendingPathComponent("coreai", isDirectory: true)
        guard CoreAIAssets.hasEncoder(in: dir) else {
            throw EngineError.coreAINotBundled(ship: tier.ship, tier: "\(tier.chunkMs)ms")
        }
        prepMessage = "Core AI: loading .aimodel shards…"

        // 1. Mel front-end (Swift/vDSP).
        let mel = try MelFrontend(resourceDirectory: dir)
        prepFraction = 0.2

        // 2. Encoder shards + fused decoder/joint, ANE-preferred or CPU-only.
        let policy = CoreAIComputePolicy.decide(tier: tier.variant, chunkMs: tier.chunkMs)
        print("[CoreAI] compute policy for \(tier.variant): \(policy.reason)")
        let runner = CoreAIEncoderRunner(
            coreaiDirectory: dir, cpuOnly: policy.cpuOnly, aneGuardTier: tier.variant)
        do {
            try await runner.load()
            try await runner.loadDecoderJoint(coreaiDirectory: dir)
            prepFraction = 0.5

            coreAIResidency = CoreAIResidencyProbe.report(in: dir)

            // 3. Tokenizer + loop params from the tier's metadata.json.
            let meta = try CoreAIMetadata.load(from: tier.directory)
            let tokenizer = try NemotronMultilingualTokenizer(
                vocabPath: tier.directory.appendingPathComponent("tokenizer.json"),
                langTagTokenIds: Set(meta.langTagTokenIds)
            )
            prepFraction = 0.7

            // 4. Streaming transcriber (mel → encoder → RNN-T greedy decode).
            let transcriber = CoreAIStreamingTranscriber(
                runner: runner,
                mel: mel,
                tokenizer: tokenizer,
                blankIdx: meta.blankIdx,
                promptId: Int32(meta.promptId(for: nil)),
                totalMelFrames: meta.totalMelFrames,
                chunkMelFrames: meta.chunkMelFrames,
                preEncodeCache: meta.preEncodeCache
            )
            prepMessage = "Core AI ready — \(policy.reason)."
            return CoreAISession(runner: runner, transcriber: transcriber, metadata: meta)
        } catch {
            runner.tearDown()
            throw error
        }
    }

    /// Tear down the loaded session (crash-safe order for Core AI).
    private func releaseSession() {
        session?.tearDown()
        session = nil
        sessionKey = nil
        appliedLanguageCode = nil
    }

    private func applyLanguageIfNeeded(_ code: String?, to session: ASRSession) async {
        if appliedLanguageCode == .some(code) { return }
        await session.setLanguage(code)
        appliedLanguageCode = .some(code)
        detectedLanguage = nil
    }

    enum EngineError: LocalizedError {
        case modelNotBundled(ship: String, tier: String)
        case coreAINotBundled(ship: String, tier: String)
        case notReady

        var errorDescription: String? {
            switch self {
            case .modelNotBundled(let ship, let tier):
                return "The \(ship) model for \(tier) isn't bundled in this build."
            case .coreAINotBundled(let ship, let tier):
                return
                    "Core AI models for \(ship) \(tier) aren't bundled (expected coreai/encoder_shard_{0..3}_int8.aimodel)."
            case .notReady:
                return "The speech model isn't loaded."
            }
        }
    }

    /// Called when the user changes backend / chunk size / language. Drops back
    /// to idle if the loaded session no longer matches (the next use reloads),
    /// and releases the loaded models right away on a backend switch.
    func invalidateForSettingsChange() {
        let desired = desiredSessionKey
        guard desired != sessionKey else { return }
        if !isBusy, let sessionKey,
            !sessionKey.hasPrefix("\(settings.backend.rawValue)|")
        {
            releaseSession()
            if settings.backend != .coreml { sharedCache.removeAll() }
        }
        if phase == .ready { phase = .idle }
    }

    // MARK: - File transcription

    func transcribeFile(url: URL) async {
        guard !isBusy else { return }
        await prepareModelIfNeeded()
        guard let session, phase == .ready else { return }

        let streamed = settings.fileMode == .streamed
        transcript = ""
        detectedLanguage = nil
        fileProgress = 0
        lastStageTimes = nil
        isStreaming = streamed
        phase = .transcribingFile
        liveActivity.start(language: liveActivityLanguageLabel, isListening: false)
        liveActivityStatus = liveActivity.lastStatus

        let started = Date()
        var loggedFirstText = false
        do {
            await session.reset()
            let durationHint = try? await audioDuration(url)
            var totalSamples = 0
            for try await block in decodedFileBlocks(url) {
                if let partial = try await session.process(block), streamed {
                    if !loggedFirstText, !partial.isEmpty {
                        loggedFirstText = true
                        print(
                            "[FileTranscribe] first partial after \(String(format: "%.2f", Date().timeIntervalSince(started)))s"
                        )
                    }
                    transcript = partial
                }
                totalSamples += block.count
                if let durationHint, durationHint > 0 {
                    setFileProgress(min(0.99, Double(totalSamples) / (durationHint * 16000.0)))
                }
            }

            transcript = try await session.finish()
            fileProgress = 1
            detectedLanguage = await resolveDetectedLanguage()

            let elapsed = Date().timeIntervalSince(started)
            let duration = Double(totalSamples) / 16000.0
            lastRTFx = elapsed > 0 ? duration / elapsed : nil
            lastStageTimes = await session.stageTimes()

            phase = .ready
            isStreaming = false
            liveActivity.end(finalTranscript: transcript, language: liveActivityLanguageLabel)
        } catch {
            phase = .failed(friendly(error))
            isStreaming = false
            liveActivity.end(finalTranscript: transcript, language: liveActivityLanguageLabel)
        }
    }

    /// Publish file progress only in ≥1% steps (and the final value).
    private func setFileProgress(_ value: Double) {
        if value - fileProgress >= 0.01 || value >= 1 { fileProgress = value }
    }

    private nonisolated func audioDuration(_ url: URL) async throws -> Double {
        try await Task.detached(priority: .userInitiated) {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            let file = try AVAudioFile(forReading: url)
            return Double(file.length) / file.processingFormat.sampleRate
        }.value
    }

    /// Read + resample an audio file block by block (16 kHz mono) on a
    /// background task, so transcription starts before the whole file is decoded.
    private nonisolated func decodedFileBlocks(_ url: URL) -> AsyncThrowingStream<[Float], Error> {
        AsyncThrowingStream { continuation in
            Task.detached(priority: .userInitiated) {
                let scoped = url.startAccessingSecurityScopedResource()
                defer { if scoped { url.stopAccessingSecurityScopedResource() } }

                do {
                    let file = try AVAudioFile(forReading: url)
                    let format = file.processingFormat
                    let converter = AudioConverter()
                    let framesPerRead = AVAudioFrameCount(max(4096, Int(format.sampleRate)))

                    while file.framePosition < file.length {
                        let remaining = AVAudioFrameCount(
                            min(Int64(framesPerRead), file.length - file.framePosition))
                        guard
                            let buffer = AVAudioPCMBuffer(
                                pcmFormat: format, frameCapacity: remaining)
                        else {
                            throw AudioConverterError.failedToCreateBuffer
                        }
                        try file.read(into: buffer)
                        if buffer.frameLength == 0 { break }
                        let samples = try converter.resampleBuffer(buffer)
                        if !samples.isEmpty {
                            continuation.yield(samples)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Microphone transcription

    func startListening() async {
        guard !isBusy else { return }

        // Permission gate.
        let granted: Bool
        if MicrophoneCapture.permission == .granted {
            granted = true
        } else {
            granted = await MicrophoneCapture.requestPermission()
        }
        guard granted else {
            phase = .failed(MicrophoneCapture.CaptureError.permissionDenied.localizedDescription)
            return
        }

        await prepareModelIfNeeded()
        guard let session, phase == .ready else { return }

        transcript = ""
        detectedLanguage = nil
        isStreaming = true
        await session.reset()

        let capture = MicrophoneCapture()
        capture.onLevel = { [weak self] level in
            Task { @MainActor in self?.micLevel = level }
        }
        self.micCapture = capture

        do {
            let stream = try await capture.start()
            phase = .listening
            liveActivity.start(language: liveActivityLanguageLabel)
            liveActivityStatus = liveActivity.lastStatus
            micTask = Task { [weak self] in
                guard let self else { return }
                do {
                    // Ends once `stopListening` stops the capture and every
                    // buffered block has been processed.
                    for await block in stream {
                        if let partial = try await session.process(block) {
                            self.transcript = partial
                        }
                    }
                } catch {
                    self.phase = .failed(self.friendly(error))
                    self.isStreaming = false
                    self.liveActivity.end(
                        finalTranscript: self.transcript, language: self.liveActivityLanguageLabel)
                    if !self.isStopping {
                        await self.micCapture?.stop()
                        self.micCapture = nil
                        self.micLevel = 0
                    }
                }
            }
        } catch {
            phase = .failed(friendly(error))
            isStreaming = false
            micCapture = nil
        }
    }

    func stopListening() async {
        guard phase == .listening, !isStopping else { return }
        isStopping = true
        defer { isStopping = false }

        // Stop the capture first (this finishes the sample stream), then let
        // the mic task drain the blocks already buffered. Cancelling it instead
        // could cut a chunk off mid-encode while `finish()` touches the same
        // streaming caches / decoder state.
        await micCapture?.stop()
        await micTask?.value
        micTask = nil
        micCapture = nil
        micLevel = 0

        // The mic task failed while draining; it already reported the error.
        guard phase == .listening else { return }

        if let session {
            do {
                let finalText = try await session.finish()
                if !finalText.isEmpty { transcript = finalText }
                detectedLanguage = await resolveDetectedLanguage()
            } catch {
                // Keep whatever partial we have; nothing fatal on stop.
            }
        }
        isStreaming = false
        phase = .ready
        liveActivity.end(finalTranscript: transcript, language: liveActivityLanguageLabel)
    }

    // MARK: - Helpers

    private func resolveDetectedLanguage() async -> String? {
        guard settings.languageCode == nil else {
            // User pinned a language — show that.
            return settings.language.name
        }
        guard let code = await session?.detectedLanguageCode() else { return nil }
        return ASRLanguageCatalog.displayName(forDetectedCode: code)
    }

    func clearTranscript() {
        transcript = ""
        detectedLanguage = nil
        fileProgress = 0
        lastRTFx = nil
    }

    /// Retry preparation after a failure (driven by the failure overlay).
    func retryPreparation() async {
        // Drop any half-state so prepare re-runs cleanly.
        releaseSession()
        phase = .idle
        await prepareModelIfNeeded()
    }

    private func friendly(_ error: Error) -> String {
        // CoreML's Apple-silicon requirement is the most common hard failure
        // (e.g. running on an Intel Mac / unsupported target).
        let raw = error.localizedDescription
        if raw.localizedCaseInsensitiveContains("Apple Silicon") {
            return "This model needs an Apple-silicon device. Run on a physical iPhone or iPad."
        }
        return raw
    }
}

// MARK: - Backend sessions

/// One prepared backend pipeline, driven identically by the file and mic flows.
@MainActor
private protocol ASRSession: AnyObject {
    /// Apply a language hint (nil = auto-detect) for the next utterance.
    func setLanguage(_ code: String?) async
    /// Start a new utterance (clears streaming caches + decoder state).
    func reset() async
    /// Feed 16 kHz mono samples (any block size); returns the running
    /// transcript when it changed.
    func process(_ samples: [Float]) async throws -> String?
    /// Flush the buffered tail (zero-padded to a full chunk); final transcript.
    func finish() async throws -> String
    /// Language tag the model emitted this utterance (e.g. "en-US").
    func detectedLanguageCode() async -> String?
    /// Per-stage timing of the last utterance, if the backend records it.
    func stageTimes() async -> String?
    /// Release model resources.
    func tearDown()
}

/// CoreML backend: the vendored streaming manager (split encoder + smart-spec).
@MainActor
private final class CoreMLSession: ASRSession {
    private var manager: StreamingNemotronMultilingualAsrManager?
    private var lastPartial = ""

    init(manager: StreamingNemotronMultilingualAsrManager) {
        self.manager = manager
    }

    private func loaded() throws -> StreamingNemotronMultilingualAsrManager {
        guard let manager else { throw TranscriptionEngine.EngineError.notReady }
        return manager
    }

    func setLanguage(_ code: String?) async { await manager?.setLanguage(code) }

    func reset() async {
        lastPartial = ""
        await manager?.reset()
    }

    func process(_ samples: [Float]) async throws -> String? {
        let manager = try loaded()
        _ = try await manager.process(samples: samples)
        let partial = await manager.getPartialTranscript()
        guard partial != lastPartial else { return nil }
        lastPartial = partial
        return partial
    }

    func finish() async throws -> String { try await loaded().finish() }

    func detectedLanguageCode() async -> String? { await manager?.detectedLanguage() }

    func stageTimes() async -> String? {
        guard let manager else { return nil }
        return String(
            format: "prep %.2fs enc %.2fs dec %.2fs chunks %d",
            Double(await manager.prepNanos) / 1e9, Double(await manager.encNanos) / 1e9,
            Double(await manager.decNanos) / 1e9, await manager.chunkCount)
    }

    func tearDown() { manager = nil }
}

/// Core AI backend: `CoreAIStreamingTranscriber` over `CoreAIEncoderRunner`.
@MainActor
private final class CoreAISession: ASRSession {
    private var runner: CoreAIEncoderRunner?
    private var transcriber: CoreAIStreamingTranscriber?
    private let metadata: CoreAIMetadata
    private var lastPartial = ""

    init(runner: CoreAIEncoderRunner, transcriber: CoreAIStreamingTranscriber, metadata: CoreAIMetadata) {
        self.runner = runner
        self.transcriber = transcriber
        self.metadata = metadata
    }

    private func loaded() throws -> CoreAIStreamingTranscriber {
        guard let transcriber else { throw TranscriptionEngine.EngineError.notReady }
        return transcriber
    }

    func setLanguage(_ code: String?) async {
        transcriber?.promptId = Int32(metadata.promptId(for: code))
    }

    func reset() async {
        lastPartial = ""
        transcriber?.reset()
    }

    func process(_ samples: [Float]) async throws -> String? {
        let transcriber = try loaded()
        guard try await transcriber.stream(samples: samples) else { return nil }
        let partial = transcriber.partialText
        guard partial != lastPartial else { return nil }
        lastPartial = partial
        return partial
    }

    func finish() async throws -> String { try await loaded().finishStreaming() }

    func detectedLanguageCode() async -> String? { transcriber?.detectedLanguage }

    func stageTimes() async -> String? { nil }

    /// Drop the transcriber first (it retains the runner), then release the
    /// runner's functions before its models.
    func tearDown() {
        transcriber = nil
        runner?.tearDown()
        runner = nil
    }
}
