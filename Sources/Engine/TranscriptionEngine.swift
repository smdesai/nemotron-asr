import AVFoundation
import CoreML
// The ASR core is vendored into this target (NemotronWatch/Vendored) — no
// external FluidAudio package. The guard keeps the source usable either way.
#if canImport(FluidAudio)
import FluidAudio
#endif
import Foundation
import SwiftUI

/// Drives the Nemotron multilingual streaming ASR for both audio-file and
/// microphone input, and publishes everything the UI needs to render.
@MainActor
final class TranscriptionEngine: ObservableObject {

    // MARK: Published UI state

    enum Phase: Equatable {
        case idle               // nothing loaded yet
        case preparing          // downloading / compiling / loading models
        case ready              // models loaded, awaiting input
        case transcribingFile   // processing an audio file
        case listening          // microphone is live
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

    /// Model-prep progress (download + compile + load).
    @Published private(set) var prepFraction: Double = 0
    @Published private(set) var prepMessage: String = ""

    /// Audio-file processing progress 0...1.
    @Published private(set) var fileProgress: Double = 0

    /// Smoothed mic input level 0...1 for the waveform.
    @Published private(set) var micLevel: Float = 0

    /// Real-time factor of the last completed run (xRT), for a little stat chip.
    @Published private(set) var lastRTFx: Double?

    /// Core AI on-device residency report (compute types + dtype histogram per
    /// shard). Populated by `prepareCoreAI`; surfaced in Settings for debugging.
    @Published private(set) var coreAIResidency: String?

    /// Live Activity start/end result, surfaced in Settings for on-device
    /// diagnosis (whether the activity was requested, denied, or failed).
    @Published private(set) var liveActivityStatus: String = "idle"

    // MARK: Dependencies

    private let settings: AppSettings

    /// Drives the lock-screen / Dynamic Island Live Activity during mic sessions.
    private let liveActivity = LiveActivityController()

    /// Friendly language label for the Live Activity: the detected language when
    /// available, else the pinned language name (nil while auto-detecting).
    private var liveActivityLanguageLabel: String? {
        detectedLanguage ?? (settings.languageCode == nil ? nil : settings.language.name)
    }

    // MARK: Model state

    /// Cache of loaded shared model bundles, keyed by "<ship>/<chunkMs>".
    private var sharedCache: [String: SharedNemotronMultilingualModels] = [:]
    private var manager: StreamingNemotronMultilingualAsrManager?
    /// The "<ship>/<chunkMs>" key currently loaded into `manager`.
    private var loadedVariantKey: String?
    /// The Core AI variant key (`ship/chunk#code`) currently loaded into
    /// `coreAIRunner`. Repeated prepares for the same variant become a no-op so
    /// the CPU-delegated decoder/joint functions aren't churned (and crashed).
    private var coreAIPreparedKey: String?
    /// Language code currently applied to the manager.
    private var appliedLanguageCode: String??

    private var micCapture: MicrophoneCapture?
    private var micTask: Task<Void, Never>?

    /// Core AI (iOS 27+) encoder runner + mel front-end, held across prepares.
    /// Typed `Any?` so the property exists on the iOS-27-min target without an
    /// availability annotation on the stored declaration.
    private var coreAIRunnerBox: Any?
    private var coreAIMelBox: Any?
    private var coreAITranscriberBox: Any?
    @available(iOS 27.0, *)
    private var coreAIRunner: CoreAIEncoderRunner? {
        get { coreAIRunnerBox as? CoreAIEncoderRunner }
        set { coreAIRunnerBox = newValue }
    }
    private var coreAIMel: MelFrontend? {
        get { coreAIMelBox as? MelFrontend }
        set { coreAIMelBox = newValue }
    }
    @available(iOS 27.0, *)
    private var coreAITranscriber: CoreAIStreamingTranscriber? {
        get { coreAITranscriberBox as? CoreAIStreamingTranscriber }
        set { coreAITranscriberBox = newValue }
    }

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

    /// Resolve the model ship directory ("latin" or "multilingual") for the
    /// currently selected language.
    private func shipDirectory(for code: String?) -> String {
        StreamingNemotronMultilingualAsrManager.languageDirectory(for: code ?? "auto")
    }

    private func variantKey(code: String?, chunkMs: Int) -> String {
        "\(shipDirectory(for: code))/\(chunkMs)"
    }

    // MARK: - Model preparation

    /// Ensure a manager is loaded for the current language ship + chunk size,
    /// loading the bundled CoreML variant if needed. Safe to call repeatedly —
    /// reuses cached bundles and only reloads when the variant changes.
    func prepareModelIfNeeded() async {
        let code = settings.languageCode
        let chunkMs = settings.chunkSize.rawValue

        // Core AI backend (iOS 27+): validate + run the staged .aimodel encoder.
        // This is the experimental path; full streaming transcription via Core AI
        // is wired in the runtime-integration step. For now it confirms the
        // converted assets load and run on-device.
        if settings.backend == .coreai {
            await prepareCoreAI(code: code, chunkMs: chunkMs)
            return
        }

        let key = variantKey(code: code, chunkMs: chunkMs)

        // Already loaded for this variant — just (re)apply the language hint.
        if loadedVariantKey == key, manager != nil {
            await applyLanguageIfNeeded(code)
            if case .preparing = phase {} else if phase == .idle {
                phase = .ready
            }
            return
        }

        phase = .preparing
        prepFraction = 0
        prepMessage = "Preparing model…"

        do {
            let shared = try await loadShared(code: code, chunkMs: chunkMs, key: key)
            let mgr = manager ?? StreamingNemotronMultilingualAsrManager()
            try await mgr.loadFromShared(shared)
            self.manager = mgr
            self.loadedVariantKey = key
            self.appliedLanguageCode = nil
            await applyLanguageIfNeeded(code)
            phase = .ready
            prepFraction = 1
            prepMessage = "Ready"
        } catch {
            phase = .failed(friendly(error))
        }
    }

    private func loadShared(code: String?, chunkMs: Int, key: String) async throws
        -> SharedNemotronMultilingualModels
    {
        if let cached = sharedCache[key] {
            prepMessage = "Loading model…"
            prepFraction = 1
            return cached
        }

        // The HuggingFace repo is gated, so the model variants are bundled into
        // the app under `Models/<ship>/<tier>ms/` and loaded from the bundle —
        // no runtime download. `.mlmodelc` is already compiled, so this is fast.
        prepMessage = "Loading model…"
        prepFraction = 0.15
        let variantDir = try bundledVariantDirectory(code: code, chunkMs: chunkMs)
        // CoreML model artifacts live in the `coreml/` subdirectory; the shared
        // `metadata.json` / `tokenizer.json` stay at the tier root (variantDir).
        let coremlDir = variantDir.appendingPathComponent("coreml", isDirectory: true)

        prepFraction = 0.4
        try await probeShardedEncoderLoadIfPresent(in: coremlDir)
        let shared = try await StreamingNemotronMultilingualAsrManager.preloadShared(
            from: coremlDir,
            commonDirectory: variantDir
        )
        sharedCache[key] = shared
        return shared
    }

    /// Resolve the bundled directory for a given ship + chunk tier.
    private func bundledVariantDirectory(code: String?, chunkMs: Int) throws -> URL {
        let ship = shipDirectory(for: code)            // "latin" or "multilingual"
        let tier = "\(chunkMs)ms"
        guard let modelsRoot = Bundle.main.resourceURL?.appendingPathComponent("Models") else {
            throw EngineError.modelNotBundled(ship: ship, tier: tier)
        }
        let variantDir = modelsRoot
            .appendingPathComponent(ship, isDirectory: true)
            .appendingPathComponent(tier, isDirectory: true)
        let metadata = variantDir.appendingPathComponent("metadata.json")
        guard FileManager.default.fileExists(atPath: metadata.path) else {
            throw EngineError.modelNotBundled(ship: ship, tier: tier)
        }
        return variantDir
    }

    // MARK: - Core AI path (iOS 27+, experimental)

    /// Resolve the bundled `coreai/` subdir for a tier (parallel to the CoreML
    /// `.mlmodelc` set), holding the int8 sharded encoder + decoder/joint + the
    /// mel front-end resources.
    private func coreaiDirectory(code: String?, chunkMs: Int) throws -> URL {
        let ship = shipDirectory(for: code)
        let tier = "\(chunkMs)ms"
        guard let modelsRoot = Bundle.main.resourceURL?.appendingPathComponent("Models") else {
            throw EngineError.modelNotBundled(ship: ship, tier: tier)
        }
        let dir = modelsRoot
            .appendingPathComponent(ship, isDirectory: true)
            .appendingPathComponent(tier, isDirectory: true)
            .appendingPathComponent("coreai", isDirectory: true)
        let shard0 = CoreAIAssets.encoderShardURL(0, in: dir)
        guard FileManager.default.fileExists(atPath: shard0.path) else {
            throw EngineError.coreAINotBundled(ship: ship, tier: tier)
        }
        return dir
    }

    /// Load + run the staged Core AI encoder shards on real mel features.
    /// Reports a verified status; does not (yet) produce a full transcript — that
    /// is the streaming-runtime integration step.
    private func prepareCoreAI(code: String?, chunkMs: Int) async {
        // Idempotent: if a runner is already prepared for this exact variant,
        // reuse it. Rebuilding the runner deallocates the CPU(BNNS)-delegated
        // decoder/joint InferenceFunctions, whose teardown over-releases on the
        // 27.0 beta — so avoid redundant reloads (and re-entrant double-prepares).
        let preparedKey = "\(variantKey(code: code, chunkMs: chunkMs))#\(code ?? "auto")"
        if #available(iOS 27.0, *), coreAIRunner != nil,
           coreAIPreparedKey == preparedKey, phase == .ready {
            return
        }

        phase = .preparing
        prepFraction = 0
        prepMessage = "Core AI: loading .aimodel shards…"

        guard #available(iOS 27.0, *) else {
            phase = .failed("Core AI requires iOS 27 or later.")
            return
        }

        // Tear down any previous runner in a crash-safe order (functions before
        // models) BEFORE building the next one. Drop the transcriber first — it
        // strongly retains the runner.
        coreAITranscriber = nil
        coreAIRunner?.tearDown()
        coreAIRunner = nil
        coreAIMel = nil
        coreAIPreparedKey = nil

        do {
            let dir = try coreaiDirectory(code: code, chunkMs: chunkMs)
            // Shared metadata.json / tokenizer.json live at the tier root (the
            // parent of coreai/); the .aimodel + mel resources stay in coreai/.
            let commonDir = dir.deletingLastPathComponent()

            // 1. Mel front-end (Swift/vDSP) — validated at ~130 dB vs NeMo.
            let mel = try MelFrontend(resourceDirectory: dir)
            prepFraction = 0.2

            // 2. Load the four int8 encoder shards + decoder/joint.
            let runner = CoreAIEncoderRunner(coreaiDirectory: dir)
            try await runner.load()
            try await runner.loadDecoderJoint(coreaiDirectory: dir)
            prepFraction = 0.5
            let fnNames = runner.shardFunctionNames
            print("[CoreAI] loaded \(fnNames.count) encoder shards + decoder/joint")

            // ANE residency probe (console + Settings card).
            let residency = CoreAIResidencyProbe.report(in: dir)
            self.coreAIResidency = residency

            // 3. Tokenizer + loop params from metadata.json (blank idx, lang tags).
            let meta = try CoreAIMetadata.load(from: commonDir)
            let tokenizerURL = commonDir.appendingPathComponent("tokenizer.json")
            let tokenizer = try NemotronMultilingualTokenizer(
                vocabPath: tokenizerURL,
                langTagTokenIds: Set(meta.langTagTokenIds)
            )
            prepFraction = 0.7

            // 4. Build the streaming transcriber (mel → encoder → RNN-T greedy decode).
            let totalMelFrames = meta.totalMelFrames
            let transcriber = CoreAIStreamingTranscriber(
                runner: runner,
                mel: mel,
                tokenizer: tokenizer,
                blankIdx: meta.blankIdx,
                promptId: Int32(promptIdForCode(code, meta: meta)),
                totalMelFrames: totalMelFrames,
                chunkMelFrames: meta.chunkMelFrames,
                preEncodeCache: meta.preEncodeCache
            )

            self.coreAIRunner = runner
            self.coreAIMel = mel
            self.coreAITranscriber = transcriber
            coreAIPreparedKey = preparedKey
            phase = .ready
            prepFraction = 1
            let fp32Note = residency.contains("⚠️")
                ? " (encoder partly GPU — fp32 ops remain)" : " (encoder ANE-clean)"
            prepMessage = "Core AI ready\(fp32Note)."
        } catch {
            phase = .failed("Core AI: \(friendly(error))")
        }
    }

    /// Resolve the language prompt id for the current language code from the
    /// model's prompt_dictionary (default = auto/101 when unset or unknown).
    private func promptIdForCode(_ code: String?, meta: CoreAIMetadata) -> Int {
        guard let code, let id = meta.promptDictionary[code] else {
            return meta.promptDictionary["auto"] ?? 101
        }
        return id
    }

    /// Transcribe an audio file via the Core AI pipeline (mel → encoder →
    /// RNN-T greedy decode through `decoder.aimodel` + `joint.aimodel`).
    private func transcribeFileCoreAI(url: URL) async {
        guard #available(iOS 27.0, *), let transcriber = coreAITranscriber, phase == .ready else {
            transcript = "[Core AI] not ready: \(prepMessage)"
            return
        }
        let streamed = settings.fileMode == .streamed
        transcript = ""
        detectedLanguage = nil
        fileProgress = 0
        isStreaming = streamed
        phase = .transcribingFile
        liveActivity.start(language: liveActivityLanguageLabel, isListening: false)
        liveActivityStatus = liveActivity.lastStatus
        let started = Date()
        do {
            let samples = try await decodeFile(url)
            print("[CoreAI] decoded file: \(samples.count) samples")
            guard !samples.isEmpty else {
                transcript = "[Core AI] decoded 0 samples from file."
                phase = .ready
                liveActivity.end(finalTranscript: transcript, language: liveActivityLanguageLabel)
                return
            }
            // Stream partial text after each chunk when "Stream live" is selected.
            // The transcriber runs on the MainActor-isolated engine, so the
            // callback can publish to @Published transcript directly.
            let onPartial: ((String) -> Void)? = streamed
                ? { [weak self] text in self?.transcript = text }
                : nil
            let text = try await transcriber.transcribe(samples: samples, onPartial: onPartial)
            isStreaming = false
            let elapsed = Date().timeIntervalSince(started)
            let duration = Double(samples.count) / 16000.0
            lastRTFx = elapsed > 0 ? duration / elapsed : nil
            // Surface an explicit note when decode produced no tokens, so the UI
            // isn't silently blank.
            transcript = text.isEmpty
                ? "[Core AI] no tokens emitted (all-blank decode) in \(String(format: "%.1f", elapsed))s."
                : text
            fileProgress = 1
            phase = .ready
            liveActivity.end(finalTranscript: transcript, language: liveActivityLanguageLabel)
        } catch {
            transcript = "[Core AI] error: \(friendly(error))"
            phase = .failed("Core AI transcribe: \(friendly(error))")
            liveActivity.end(finalTranscript: transcript, language: liveActivityLanguageLabel)
        }
    }

    private nonisolated func probeShardedEncoderLoadIfPresent(in coremlDir: URL) async throws {
        let shardURLs = (0..<4).map { coremlDir.appendingPathComponent("encoder_shard_\($0).mlmodelc") }
        guard shardURLs.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) else { return }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        print("[ShardedEncoderProbe] Found 4 sharded encoders; probing cpuAndNeuralEngine load")
        for (idx, url) in shardURLs.enumerated() {
            let started = Date()
            print("[ShardedEncoderProbe] Loading shard \(idx): \(url.lastPathComponent)")
            _ = try await MLModel.load(contentsOf: url, configuration: cfg)
            let elapsed = Date().timeIntervalSince(started)
            print("[ShardedEncoderProbe] Loaded shard \(idx) in \(String(format: "%.2f", elapsed))s")
        }
        print("[ShardedEncoderProbe] Completed sharded encoder ANE load probe")
    }

    enum EngineError: LocalizedError {
        case modelNotBundled(ship: String, tier: String)
        case coreAINotBundled(ship: String, tier: String)

        var errorDescription: String? {
            switch self {
            case .modelNotBundled(let ship, let tier):
                return "The \(ship) model for \(tier) isn't bundled in this build."
            case .coreAINotBundled(let ship, let tier):
                return "Core AI models for \(ship) \(tier) aren't bundled (expected coreai/encoder_shard_0_int8.aimodel)."
            }
        }
    }

    private func applyLanguageIfNeeded(_ code: String?) async {
        guard let manager else { return }
        if appliedLanguageCode == .some(code) { return }
        await manager.setLanguage(code)
        appliedLanguageCode = .some(code)
        detectedLanguage = nil
    }

    /// Force a reload on next use (e.g. after the user changes settings).
    func invalidateForSettingsChange() {
        // If the resolved variant changed, drop the loaded manager so the next
        // prepare reloads. Language-only changes within the same ship are
        // applied lazily via `applyLanguageIfNeeded`.
        let key = variantKey(code: settings.languageCode, chunkMs: settings.chunkSize.rawValue)
        if key != loadedVariantKey || settings.backend == .coreai {
            loadedVariantKey = nil
            if phase == .ready { phase = .idle }
        }
    }

    // MARK: - File transcription

    func transcribeFile(url: URL) async {
        guard !isBusy else { return }
        await prepareModelIfNeeded()
        if settings.backend == .coreai {
            await transcribeFileCoreAI(url: url)
            return
        }
        guard let manager else { return }
        guard phase == .ready else { return }

        let streamed = settings.fileMode == .streamed
        transcript = ""
        detectedLanguage = nil
        fileProgress = 0
        isStreaming = streamed
        phase = .transcribingFile
        liveActivity.start(language: liveActivityLanguageLabel, isListening: false)
        liveActivityStatus = liveActivity.lastStatus

        let firstVisibleStarted = Date()
        var didShowFirstText = false

        // Wire (or clear) the live partial callback.
        if streamed {
            await manager.setPartialCallback { [weak self] text in
                Task { @MainActor in
                    guard let self else { return }
                    if !didShowFirstText, !text.isEmpty {
                        didShowFirstText = true
                        let elapsed = Date().timeIntervalSince(firstVisibleStarted)
                        print("[FileTranscribe] first partial after \(String(format: "%.2f", elapsed))s")
                    }
                    self.transcript = text
                }
            }
        } else {
            await manager.clearPartialCallback()
        }

        let started = Date()
        do {
            await manager.reset()

            let durationHint = try? await audioDuration(url)
            let decodeStarted = Date()
            var totalSamples = 0
            var didLogFirstBlock = false
            for try await block in decodedFileBlocks(url) {
                if Task.isCancelled { break }
                if !didLogFirstBlock {
                    didLogFirstBlock = true
                    print("[FileTranscribe] first decoded block after \(String(format: "%.2f", Date().timeIntervalSince(decodeStarted)))s")
                }
                _ = try await manager.process(samples: block)
                totalSamples += block.count
                if let durationHint, durationHint > 0 {
                    fileProgress = min(0.99, Double(totalSamples) / (durationHint * 16000.0))
                }
            }

            let finalText = try await manager.finish()
            transcript = finalText
            if !didShowFirstText, !finalText.isEmpty {
                didShowFirstText = true
                let elapsed = Date().timeIntervalSince(firstVisibleStarted)
                print("[FileTranscribe] first text at final after \(String(format: "%.2f", elapsed))s")
            }
            fileProgress = 1
            detectedLanguage = await resolveDetectedLanguage()

            let elapsed = Date().timeIntervalSince(started)
            let duration = Double(totalSamples) / 16000.0
            lastRTFx = elapsed > 0 ? duration / elapsed : nil

            phase = .ready
            isStreaming = false
            liveActivity.end(finalTranscript: transcript, language: liveActivityLanguageLabel)
        } catch {
            phase = .failed(friendly(error))
            isStreaming = false
            liveActivity.end(finalTranscript: transcript, language: liveActivityLanguageLabel)
        }
    }

    /// Read + resample an audio file on a background task.
    private nonisolated func decodeFile(_ url: URL) async throws -> [Float] {
        try await Task.detached(priority: .userInitiated) {
            let converter = AudioConverter()
            // Security-scoped access for files vended by the document picker.
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            return try converter.resampleAudioFile(url)
        }.value
    }

    private nonisolated func audioDuration(_ url: URL) async throws -> Double {
        try await Task.detached(priority: .userInitiated) {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            let file = try AVAudioFile(forReading: url)
            return Double(file.length) / file.processingFormat.sampleRate
        }.value
    }

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
                        let remaining = AVAudioFrameCount(min(Int64(framesPerRead), file.length - file.framePosition))
                        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: remaining) else {
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
        if settings.backend == .coreai {
            await startListeningCoreAI()
            return
        }
        guard let manager, phase == .ready else { return }

        transcript = ""
        detectedLanguage = nil
        isStreaming = true

        // Mic streaming always shows live partials.
        await manager.setPartialCallback { [weak self] text in
            Task { @MainActor in self?.transcript = text }
        }
        await manager.reset()

        let capture = MicrophoneCapture()
        capture.onLevel = { [weak self] level in
            Task { @MainActor in self?.micLevel = level }
        }
        self.micCapture = capture

        do {
            let stream = try capture.start()
            phase = .listening
            liveActivity.start(language: liveActivityLanguageLabel)
            liveActivityStatus = liveActivity.lastStatus
            micTask = Task { [weak self] in
                guard let self else { return }
                do {
                    for await block in stream {
                        if Task.isCancelled { break }
                        _ = try await manager.process(samples: block)
                    }
                } catch {
                    await MainActor.run {
                        self.phase = .failed(self.friendly(error))
                        self.liveActivity.end(
                            finalTranscript: self.transcript,
                            language: self.liveActivityLanguageLabel
                        )
                    }
                }
            }
        } catch {
            phase = .failed(friendly(error))
            isStreaming = false
            micCapture = nil
        }
    }

    /// Live microphone transcription on the Core AI backend: mirror of the
    /// CoreML branch below, feeding capture blocks into the transcriber's
    /// chunk buffer and publishing the running partial after each chunk.
    private func startListeningCoreAI() async {
        guard #available(iOS 27.0, *), let transcriber = coreAITranscriber, phase == .ready else {
            transcript = "[Core AI] not ready: \(prepMessage)"
            return
        }

        transcript = ""
        detectedLanguage = nil
        isStreaming = true
        transcriber.reset()

        let capture = MicrophoneCapture()
        capture.onLevel = { [weak self] level in
            Task { @MainActor in self?.micLevel = level }
        }
        self.micCapture = capture

        do {
            let stream = try capture.start()
            phase = .listening
            liveActivity.start(language: liveActivityLanguageLabel)
            liveActivityStatus = liveActivity.lastStatus
            micTask = Task { [weak self] in
                guard let self else { return }
                do {
                    for await block in stream {
                        if Task.isCancelled { break }
                        if try await transcriber.stream(samples: block) {
                            self.transcript = transcriber.partialText
                        }
                    }
                } catch {
                    await MainActor.run {
                        self.phase = .failed(self.friendly(error))
                        self.liveActivity.end(
                            finalTranscript: self.transcript,
                            language: self.liveActivityLanguageLabel
                        )
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
        guard phase == .listening else { return }
        micCapture?.stop()
        micTask?.cancel()
        micTask = nil
        micCapture = nil
        micLevel = 0

        if settings.backend == .coreai {
            if #available(iOS 27.0, *), let transcriber = coreAITranscriber {
                do {
                    let finalText = try await transcriber.finishStreaming()
                    if !finalText.isEmpty { transcript = finalText }
                } catch {
                    // Keep whatever partial we have; nothing fatal on stop.
                }
            }
            isStreaming = false
            phase = .ready
            liveActivity.end(finalTranscript: transcript, language: liveActivityLanguageLabel)
            return
        }

        guard let manager else {
            phase = .ready
            isStreaming = false
            liveActivity.end(finalTranscript: transcript, language: liveActivityLanguageLabel)
            return
        }
        do {
            let finalText = try await manager.finish()
            if !finalText.isEmpty { transcript = finalText }
            detectedLanguage = await resolveDetectedLanguage()
        } catch {
            // Keep whatever partial we have; surface nothing fatal on stop.
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
        guard let manager, let code = await manager.detectedLanguage() else { return nil }
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
        loadedVariantKey = nil
        phase = .idle
        await prepareModelIfNeeded()
    }

    private func friendly(_ error: Error) -> String {
        // CoreML's Apple-Silicon requirement is the most common hard failure
        // (e.g. running on an Intel Mac / unsupported target).
        let raw = error.localizedDescription
        if raw.localizedCaseInsensitiveContains("Apple Silicon") {
            return "This model needs the Apple Neural Engine. Run on a physical Apple-Silicon device."
        }
        return raw
    }
}
