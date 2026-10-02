@preconcurrency import CoreML
import Foundation

/// Callback invoked when new tokens are decoded (for live transcription updates).
/// Fires with the running transcript text only — the language tag, if any,
/// is surfaced via `detectedLanguage()`.
public typealias NemotronMultilingualPartialCallback = @Sendable (String) -> Void

/// Streaming manager for the Nemotron Speech Streaming Multilingual 0.6B
/// RNN-T model over CoreML.
///
/// Models are loaded once via `preloadShared(from:)` and adopted per stream
/// with `loadFromShared(_:)` (see `+Shared`). The encoder is the split
/// pipeline (`encoder_pre_encode` + `encoder_shard_0..3`); the per-chunk
/// encoder takes an extra `prompt_id` int32 [1] input (language hint), and
/// the ~13k vocab includes `<xx-XX>` language-tag pieces that are filtered
/// from the transcript and surfaced via `detectedLanguage()`.
///
/// Decode paths (see `+Pipeline`):
///   - Smart speculative-blank (`joint_noencproj_batched` + bare `decoder` +
///     `native_weights/` for the Swift-side encoder projection).
///   - Standard per-token greedy loop: B2 `decoder_joint_argmax` >
///     B1 `decoder_joint` > bare `decoder` + `joint`.
public actor StreamingNemotronMultilingualAsrManager {
    internal let logger = AppLogger(category: "NemotronMultilingualStreaming")

    // MARK: Models (shared handles adopted from SharedNemotronMultilingualModels)

    internal var preprocessor: MLModel?
    internal var encoderPreEncode: MLModel?
    internal var encoderShards: [MLModel] = []
    internal var decoder: MLModel?
    internal var joint: MLModel?
    /// B1: fused decoder + joint (returns logits + LSTM state).
    internal var decoderJoint: MLModel?
    /// B2: fused decoder + joint + argmax (returns int32 `token_id` + LSTM state).
    internal var decoderJointArgmax: MLModel?
    /// Smart-spec batched joint: `encoder_proj` [1, K, 640] + `decoder`
    /// [1, 640, 1] → logits [1, K, 1, V].
    internal var jointNoEncProjBatched: MLModel?
    /// K (speculative window) read from the batched joint's input shape.
    internal var jointNoEncProjBatchedK: Int = 8
    /// joint.enc weights for the Swift-side encoder projection (smart-spec).
    internal var nativeRnnt: NativeRnntInner?

    internal var tokenizer: NemotronMultilingualTokenizer?

    /// Configuration (loaded from metadata.json).
    public internal(set) var config: NemotronMultilingualStreamingConfig

    // MARK: Streaming state

    /// Audio buffer + read offset. Offset arithmetic (with periodic
    /// compaction) avoids O(N²) memmove from `removeFirst` per chunk.
    private var audioBuffer: [Float] = []
    private var audioBufferOffset: Int = 0

    /// Accumulated token IDs (raw, including any lang-tag tokens).
    internal var accumulatedTokenIds: [Int] = []

    /// Partial-transcript cache, keyed by the token count it was decoded at.
    /// Tokens are append-only between resets, so a count match means the
    /// text is current; every token-clearing site invalidates it.
    private var partialTextTokenCount: Int = -1
    private var partialText: String = ""

    /// First lang-tag piece encountered this session (without angle brackets).
    private var firstDetectedLanguage: String?

    /// Per-shard encoder caches (split encoder).
    internal var shardCacheChannels: [MLMultiArray] = []
    internal var shardCacheTimes: [MLMultiArray] = []
    internal var shardCacheLens: [MLMultiArray] = []

    /// Mel cache (last `preEncodeCache` frames from the previous chunk).
    internal var melCache: MLMultiArray?

    // Decoder LSTM state
    internal var hState: MLMultiArray?
    internal var cState: MLMultiArray?
    internal var lastToken: Int32

    /// Current prompt id (language hint). Defaults to `defaultPromptId` ("auto").
    internal var currentPromptId: Int32

    internal var partialCallback: NemotronMultilingualPartialCallback?

    // MARK: Per-stream reusable buffers (allocated in loadFromShared)

    /// Output backings passed via `MLPredictionOptions.outputBackings` so each
    /// prediction writes into a stable buffer instead of allocating.
    internal var decoderPredictionOptions: MLPredictionOptions?
    internal var jointPredictionOptions: MLPredictionOptions?
    internal var decoderJointPredictionOptions: MLPredictionOptions?
    internal var decoderJointArgmaxPredictionOptions: MLPredictionOptions?
    internal var jointNoEncProjBatchedPredictionOptions: MLPredictionOptions?
    /// [1, encoderDim, 1] fp32 per-frame encoder step, refilled in place.
    internal var encoderStepBuf: MLMultiArray?
    /// [1, T_enc, 640] fp32 Swift-computed encoder_proj (smart-spec).
    internal var encProjReusable: MLMultiArray?
    /// [1, K, 640] fp32 window slice of encProjReusable (smart-spec).
    internal var encProjBatchReusable: MLMultiArray?
    /// [1, 1] int32 token + [1] int32 length (constant 1) decoder inputs.
    internal var tokenInputBuf: MLMultiArray?
    internal var tokenLenBuf: MLMultiArray?
    /// [1, chunkSamples] fp32 audio + [1] int32 length preprocessor inputs.
    internal var audioInputBuf: MLMultiArray?
    internal var audioLenBuf: MLMultiArray?

    // MARK: Counters (reset on `reset()`)

    public internal(set) var prepNanos: UInt64 = 0
    public internal(set) var encNanos: UInt64 = 0
    public internal(set) var decNanos: UInt64 = 0
    public internal(set) var chunkCount: Int = 0
    public internal(set) var vadSkipCount: Int = 0
    /// Smart-spec acceptance counters: windows scanned / all-blank / hit a non-blank.
    public internal(set) var specWindowsTotal: Int = 0
    public internal(set) var specWindowsAllBlank: Int = 0
    public internal(set) var specWindowsHitNonBlank: Int = 0

    /// VAD hangover state: consecutive low-RMS chunks seen.
    internal var vadConsecutiveLowChunks: Int = 0

    public internal(set) var mlConfiguration: MLModelConfiguration

    public init(configuration: MLModelConfiguration? = nil) {
        self.mlConfiguration = configuration ?? MLModelConfigurationUtils.defaultConfiguration()
        self.config = NemotronMultilingualStreamingConfig()
        self.lastToken = Int32(config.blankIdx)
        self.currentPromptId = Int32(config.defaultPromptId)
    }

    /// Set callback for partial transcription updates.
    public func setPartialCallback(_ callback: @escaping NemotronMultilingualPartialCallback) {
        self.partialCallback = callback
    }

    /// Clear callback for at-end transcription modes.
    public func clearPartialCallback() {
        self.partialCallback = nil
    }

    /// Set the language hint by code (e.g. `"en-US"`, `"zh-CN"`, `"auto"`).
    /// Falls back to the model's `default_prompt_id` if the code is unknown.
    public func setLanguage(_ language: String?) async {
        let id = config.promptId(forLanguage: language)
        self.currentPromptId = Int32(id)
        logger.info("Prompt id set to \(id) for language \(language ?? "auto")")
    }

    /// First language-tag piece (e.g. `"en-US"`) emitted by the decoder this
    /// session, or `nil` if no tag has been seen yet.
    public func detectedLanguage() -> String? { firstDetectedLanguage }

    /// Build an `MLPredictionOptions` with pre-allocated output backings.
    /// Outputs that loop back as inputs to the next call (LSTM / encoder
    /// cache state) are skipped — sharing those backings lets the model
    /// overwrite a value before the next call reads it (catastrophic WER).
    private static let loopbackOutputNames: Set<String> = [
        "h_out", "c_out",
        "cache_channel_out", "cache_time_out", "cache_len_out",
        "mel_cache_out",
    ]

    /// Per-model compute-unit override. Reads the named env var; if set to
    /// CPU / CPU_AND_NE / CPU_AND_GPU / ALL, builds a fresh configuration
    /// with that compute unit. Falls back to `base` otherwise.
    internal static func computeUnitOverride(
        name: String,
        base: MLModelConfiguration,
        logger: AppLogger
    ) -> MLModelConfiguration {
        guard let raw = ProcessInfo.processInfo.environment[name]?.uppercased() else {
            return base
        }
        let override: MLComputeUnits
        switch raw {
        case "CPU", "CPU_ONLY": override = .cpuOnly
        case "CPU_AND_NE": override = .cpuAndNeuralEngine
        case "CPU_AND_GPU": override = .cpuAndGPU
        case "ALL": override = .all
        default:
            logger.warning("Unknown value for \(name): \(raw); ignoring")
            return base
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = override
        logger.info("\(name)=\(raw) → MLComputeUnits.\(override.rawValue)")
        return cfg
    }

    internal static func computeUnitsDescription(_ configuration: MLModelConfiguration) -> String {
        switch configuration.computeUnits {
        case .all: return "all"
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        @unknown default: return "unknown(\(configuration.computeUnits.rawValue))"
        }
    }

    internal static func makePredictionOptions(for model: MLModel?) -> MLPredictionOptions? {
        guard let model = model else { return nil }
        var backings: [String: Any] = [:]
        for (name, desc) in model.modelDescription.outputDescriptionsByName {
            if loopbackOutputNames.contains(name) { continue }
            guard let cons = desc.multiArrayConstraint else { continue }
            let shape = cons.shape.map { $0 }
            if shape.contains(where: { $0.intValue <= 0 }) { continue }
            guard let arr = try? MLMultiArray(shape: shape, dataType: cons.dataType) else {
                continue
            }
            backings[name] = arr
        }
        guard !backings.isEmpty else { return nil }
        let options = MLPredictionOptions()
        options.outputBackings = backings
        return options
    }

    /// True when a preprocessor, the full split encoder, and at least one
    /// decode path (B2, B1, or bare decoder + joint) are loaded.
    internal var isReady: Bool {
        preprocessor != nil && encoderPreEncode != nil && encoderShards.count == 4
            && (decoderJointArgmax != nil || decoderJoint != nil
                || (decoder != nil && joint != nil))
    }

    /// Reset all states for a new transcription session.
    /// Preserves the currently selected prompt id and ml configuration.
    public func reset() async {
        StreamingAsrUtils.resetSharedState(
            audioBuffer: &audioBuffer,
            accumulatedTokenIds: &accumulatedTokenIds
        )
        audioBufferOffset = 0
        firstDetectedLanguage = nil
        do {
            try resetStates()
        } catch {
            logger.error("Failed to reset states: \(error.localizedDescription)")
        }
    }

    internal func resetStates() throws {
        shardCacheChannels = []
        shardCacheTimes = []
        shardCacheLens = []
        if encoderPreEncode != nil && encoderShards.count == 4 {
            let perShardChannelShape = [
                config.cacheChannelShape[0],
                max(1, config.cacheChannelShape[1] / 4),
                config.cacheChannelShape[2],
                config.cacheChannelShape[3],
            ]
            let perShardTimeShape = [
                config.cacheTimeShape[0],
                max(1, config.cacheTimeShape[1] / 4),
                config.cacheTimeShape[2],
                config.cacheTimeShape[3],
            ]
            for _ in 0 ..< 4 {
                shardCacheChannels.append(
                    try EncoderCacheManager.createZeroArray(shape: perShardChannelShape))
                shardCacheTimes.append(
                    try EncoderCacheManager.createZeroArray(shape: perShardTimeShape))
                // Seed cache_len with 1 (not 0) so the encoder's slice op never
                // sees a zero-length slice; the caches are zero, so this is one
                // frame of silence preamble.
                let len = try EncoderCacheManager.createZeroArray(shape: [1])
                len[0] = 1
                shardCacheLens.append(len)
            }
        }

        melCache = nil
        partialTextTokenCount = -1
        prepNanos = 0
        encNanos = 0
        decNanos = 0
        chunkCount = 0
        vadSkipCount = 0
        specWindowsTotal = 0
        specWindowsAllBlank = 0
        specWindowsHitNonBlank = 0

        hState = try EncoderCacheManager.createZeroArray(
            shape: [config.decoderLayers, 1, config.decoderHidden])
        cState = try EncoderCacheManager.createZeroArray(
            shape: [config.decoderLayers, 1, config.decoderHidden])
        lastToken = Int32(config.blankIdx)
    }

    /// Process pre-resampled 16 kHz mono Float samples. Returns the empty
    /// string; the partial transcript is delivered via the partial callback
    /// or `getPartialTranscript()`.
    public func process(samples: [Float]) async throws -> String {
        guard isReady else { throw ASRError.notInitialized }

        self.audioBuffer.append(contentsOf: samples)

        let chunkSamples = config.chunkSamples
        while (self.audioBuffer.count - self.audioBufferOffset) >= chunkSamples {
            let chunkStart = self.audioBufferOffset
            let chunk = Array(self.audioBuffer[chunkStart ..< chunkStart + chunkSamples])
            try await processChunk(chunk)
            self.audioBufferOffset += chunkSamples

            // Periodic compaction: one memmove once enough prefix is consumed.
            if self.audioBufferOffset > 16 * chunkSamples {
                self.audioBuffer.removeFirst(self.audioBufferOffset)
                self.audioBufferOffset = 0
            }
        }

        return ""
    }

    /// Finish processing remaining audio (zero-padded to a full chunk) and
    /// return the final transcript text. The detected language is available
    /// via `detectedLanguage()` after this returns.
    public func finish() async throws -> String {
        guard let tokenizer = tokenizer, isReady else {
            throw ASRError.notInitialized
        }

        let remaining = audioBuffer.count - audioBufferOffset
        if remaining > 0 {
            let paddingNeeded = config.chunkSamples - remaining
            if paddingNeeded > 0 {
                audioBuffer.append(contentsOf: Array(repeating: 0.0, count: paddingNeeded))
            }
            let chunkStart = audioBufferOffset
            let chunk = Array(audioBuffer[chunkStart ..< chunkStart + config.chunkSamples])
            try await processChunk(chunk)
            audioBuffer.removeAll()
            audioBufferOffset = 0
        }

        let decoded = tokenizer.decode(ids: accumulatedTokenIds)
        if firstDetectedLanguage == nil {
            firstDetectedLanguage = decoded.detectedLanguage
        }
        accumulatedTokenIds.removeAll()
        partialTextTokenCount = -1

        if chunkCount > 0 {
            let prepMs = Double(prepNanos) / 1_000_000.0
            let encMs = Double(encNanos) / 1_000_000.0
            let decMs = Double(decNanos) / 1_000_000.0
            let totalMs = prepMs + encMs + decMs
            logger.debug(
                "[PROFILE] chunks=\(chunkCount) prep=\(String(format: "%.0f", prepMs))ms "
                    + "enc=\(String(format: "%.0f", encMs))ms "
                    + "dec=\(String(format: "%.0f", decMs))ms "
                    + "total=\(String(format: "%.0f", totalMs))ms "
                    + "per_chunk=\(String(format: "%.2f", totalMs / Double(chunkCount)))ms"
            )
            if specWindowsTotal > 0 {
                let allBlankPct = Double(specWindowsAllBlank) / Double(specWindowsTotal) * 100.0
                let hitPct = Double(specWindowsHitNonBlank) / Double(specWindowsTotal) * 100.0
                logger.debug(
                    "[SPEC] windows=\(specWindowsTotal) "
                        + "all_blank=\(specWindowsAllBlank) "
                        + "(\(String(format: "%.1f", allBlankPct))%) "
                        + "hit_non_blank=\(specWindowsHitNonBlank) "
                        + "(\(String(format: "%.1f", hitPct))%)"
                )
            }
        }

        return decoded.text
    }

    /// Get current partial transcript without finishing.
    public func getPartialTranscript() -> String {
        currentPartialText()
    }

    /// Decoded text of `accumulatedTokenIds`, re-decoded only when the token
    /// count changed since the last call.
    internal func currentPartialText() -> String {
        guard let tokenizer = tokenizer else { return "" }
        if partialTextTokenCount != accumulatedTokenIds.count {
            let decoded = tokenizer.decode(ids: accumulatedTokenIds)
            if firstDetectedLanguage == nil {
                firstDetectedLanguage = decoded.detectedLanguage
            }
            partialText = decoded.text
            partialTextTokenCount = accumulatedTokenIds.count
        }
        return partialText
    }

    /// Record the first language tag observed by the decoder.
    internal func recordDetectedLanguage(_ language: String) {
        if firstDetectedLanguage == nil {
            firstDetectedLanguage = language
        }
    }
}
