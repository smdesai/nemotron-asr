@preconcurrency import CoreML
import Foundation

/// Immutable bundle of CoreML models + tokenizer + config shared across N
/// `StreamingNemotronMultilingualAsrManager` instances. Each manager keeps its
/// own caches / LSTM state / prediction output backings; only the compiled
/// model graphs are shared (MLModel predictions are thread-safe).
public struct SharedNemotronMultilingualModels: Sendable {
    public let preprocessor: MLModel
    /// Split encoder frontend: mel -> hidden. Loaded CPU-only.
    public let encoderPreEncode: MLModel
    /// Split encoder layer shards (exactly four).
    public let encoderShards: [MLModel]
    /// Bare prediction LSTM. Required by smart-spec (for dec_out); optional otherwise.
    public let decoder: MLModel?
    /// Bare joint. Used only by the bare decoder + joint decode path.
    public let joint: MLModel?
    /// B1 fusion (decoder + joint). Loaded only when B2 is absent.
    public let decoderJoint: MLModel?
    /// B2 triple-fusion (decoder + joint + argmax).
    public let decoderJointArgmax: MLModel?
    /// Smart-speculative batched joint.
    public let jointNoEncProjBatched: MLModel?
    public let config: NemotronMultilingualStreamingConfig
    public let tokenizer: NemotronMultilingualTokenizer
    /// `native_weights/` directory (joint.enc weights for the smart-spec
    /// Swift-side encoder projection), if present.
    public let nativeWeightsDir: URL?
    /// MLModelConfiguration used to load these.
    public let mlConfiguration: MLModelConfiguration
}

extension StreamingNemotronMultilingualAsrManager {

    /// Split-encoder model files, all required.
    private static let splitEncoderNames: [String] =
        ["encoder_pre_encode"] + (0 ..< 4).map { "encoder_shard_\($0)" }

    /// Load all CoreML models + tokenizer + config ONCE, producing a bundle
    /// that N managers consume via `loadFromShared(_:)`.
    ///
    /// Model artifacts load from `directory`; `metadata.json` /
    /// `tokenizer.json` load from `commonDirectory` (defaults to `directory`).
    public static func preloadShared(
        from directory: URL,
        commonDirectory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> SharedNemotronMultilingualModels {
        let logger = AppLogger(category: "NemotronMultilingualStreaming")
        let commonDir = commonDirectory ?? directory

        guard SystemInfo.isAppleSilicon else {
            throw ASRError.unsupportedPlatform(
                "Nemotron multilingual int8 streaming models require Apple Silicon (ANE)."
            )
        }

        let mlConfiguration = configuration ?? MLModelConfigurationUtils.defaultConfiguration()
        let cpuOnlyConfiguration = MLModelConfigurationUtils.defaultConfiguration(
            computeUnits: .cpuOnly)
        logger.info("Preloading shared Nemotron multilingual models from \(directory.path)...")

        let metadataPath = commonDir.appendingPathComponent(
            ModelNames.NemotronMultilingualStreaming.metadata)
        guard FileManager.default.fileExists(atPath: metadataPath.path) else {
            throw ASRError.processingFailed("metadata.json not found at \(metadataPath.path).")
        }
        let config = try NemotronMultilingualStreamingConfig(from: metadataPath)
        logger.info(
            "Loaded multilingual config: \(config.chunkMs)ms chunks, vocab=\(config.vocabSize), \(config.numPrompts) prompts"
        )

        // The split encoder (pre_encode + 4 shards) is the only encoder path.
        let missingEncoderFiles = splitEncoderNames.filter { name in
            !FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("\(name).mlmodelc").path)
                && !FileManager.default.fileExists(
                    atPath: directory.appendingPathComponent("\(name).mlpackage").path)
        }
        guard missingEncoderFiles.isEmpty else {
            throw ASRError.processingFailed(
                "Missing split encoder in \(directory.path): "
                    + missingEncoderFiles.map { "\($0).mlmodelc" }.joined(separator: ", ")
                    + ". Required: encoder_pre_encode.mlmodelc + encoder_shard_0..3.mlmodelc.")
        }

        let preprocessor = try await loadShared(
            directory: directory,
            name: ModelNames.NemotronMultilingualStreaming.preprocessor,
            configuration: computeUnitOverride(
                name: "NEMOTRON_PREPROCESSOR_CU", base: cpuOnlyConfiguration, logger: logger),
            logger: logger
        )

        let encoderPreEncode = try await loadShared(
            directory: directory,
            name: "encoder_pre_encode",
            configuration: cpuOnlyConfiguration,
            logger: logger
        )
        // Shard base compute units: watchOS runs the shards on the ANE
        // (CPU-only was too slow to keep up with streaming); iOS pins them to
        // CPU-only (avoids a slow failed ANEF compile); macOS uses the
        // caller's configuration. NEMOTRON_ENCODER_SHARDS_CU overrides.
        #if os(watchOS)
        let shardBaseConfiguration = MLModelConfigurationUtils.defaultConfiguration(
            computeUnits: .cpuAndNeuralEngine)
        #elseif os(iOS)
        let shardBaseConfiguration = cpuOnlyConfiguration
        #else
        let shardBaseConfiguration = mlConfiguration
        #endif
        var encoderShards: [MLModel] = []
        for idx in 0 ..< 4 {
            encoderShards.append(
                try await loadShared(
                    directory: directory,
                    name: "encoder_shard_\(idx)",
                    configuration: computeUnitOverride(
                        name: "NEMOTRON_ENCODER_SHARDS_CU", base: shardBaseConfiguration,
                        logger: logger),
                    logger: logger
                ))
        }
        logger.info("Loaded split encoder path: encoder_pre_encode CPU + 4 encoder shards")

        let decoderCU = computeUnitOverride(
            name: "NEMOTRON_DECODER_CU", base: cpuOnlyConfiguration, logger: logger)
        let jointCU = computeUnitOverride(
            name: "NEMOTRON_JOINT_CU", base: cpuOnlyConfiguration, logger: logger)
        let decoderJointCU = computeUnitOverride(
            name: "NEMOTRON_DECODERJOINT_CU", base: cpuOnlyConfiguration, logger: logger)
        let jointBatchedCU = computeUnitOverride(
            name: "NEMOTRON_JOINT_BATCHED_CU", base: cpuOnlyConfiguration, logger: logger)

        let decoder = try await loadOptionalShared(
            directory: directory, name: ModelNames.NemotronMultilingualStreaming.decoder,
            configuration: decoderCU, logger: logger)
        let joint = try await loadOptionalShared(
            directory: directory, name: ModelNames.NemotronMultilingualStreaming.joint,
            configuration: jointCU, logger: logger)
        // Fused decode precedence: B2 > B1 (B1 is not loaded when B2 exists).
        let decoderJointArgmax = try await loadOptionalShared(
            directory: directory, name: "decoder_joint_argmax",
            configuration: decoderJointCU, logger: logger)
        var decoderJoint: MLModel? = nil
        if decoderJointArgmax == nil {
            decoderJoint = try await loadOptionalShared(
                directory: directory, name: "decoder_joint",
                configuration: decoderJointCU, logger: logger)
        }
        let jointNoEncProjBatched = try await loadOptionalShared(
            directory: directory, name: "joint_noencproj_batched",
            configuration: jointBatchedCU, logger: logger)

        // Standard decode needs B2, B1, or the bare decoder + joint pair.
        let hasStandardPath =
            decoderJointArgmax != nil || decoderJoint != nil || (decoder != nil && joint != nil)
        guard hasStandardPath else {
            throw ASRError.processingFailed(
                "No decode path in \(directory.path): provide decoder_joint_argmax.mlmodelc, "
                    + "decoder_joint.mlmodelc, or both bare decoder.mlmodelc + joint.mlmodelc.")
        }
        // Smart-spec needs the bare decoder (one call per K-frame window for
        // dec_out). Its drain uses the standard step (B2 > B1 > bare), so the
        // bare joint is required only when neither B2 nor B1 is present.
        if jointNoEncProjBatched != nil {
            let drainHasFused = decoderJointArgmax != nil || decoderJoint != nil
            if decoder == nil || (joint == nil && !drainHasFused) {
                throw ASRError.processingFailed(
                    "Smart-spec asset joint_noencproj_batched present but its decode deps are missing "
                        + "— needs decoder.mlmodelc, plus joint.mlmodelc unless "
                        + "decoder_joint_argmax/decoder_joint is present.")
            }
        }
        if decoder == nil && joint == nil {
            logger.info("Lean ship: bare decoder/joint omitted; using fused decode path only.")
        }

        let tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: commonDir.appendingPathComponent(
                ModelNames.NemotronMultilingualStreaming.tokenizer),
            langTagTokenIds: config.langTagTokenIds
        )

        let nativeWeightsDir = directory.appendingPathComponent("native_weights")
        let nativeAvailable = FileManager.default.fileExists(
            atPath: nativeWeightsDir.appendingPathComponent("weights.bin").path)

        logger.info("Shared models preload complete — ready for N consumers")

        return SharedNemotronMultilingualModels(
            preprocessor: preprocessor,
            encoderPreEncode: encoderPreEncode,
            encoderShards: encoderShards,
            decoder: decoder,
            joint: joint,
            decoderJoint: decoderJoint,
            decoderJointArgmax: decoderJointArgmax,
            jointNoEncProjBatched: jointNoEncProjBatched,
            config: config,
            tokenizer: tokenizer,
            nativeWeightsDir: nativeAvailable ? nativeWeightsDir : nil,
            mlConfiguration: mlConfiguration
        )
    }

    /// Initialize this manager from a pre-loaded shared model bundle. Builds
    /// this stream's own state (caches, prediction output backings, step
    /// buffers, encoder-projection weights); only MLModel handles are shared.
    public func loadFromShared(_ shared: SharedNemotronMultilingualModels) async throws {
        self.mlConfiguration = shared.mlConfiguration
        self.config = shared.config
        self.currentPromptId = Int32(config.defaultPromptId)

        self.preprocessor = shared.preprocessor
        self.encoderPreEncode = shared.encoderPreEncode
        self.encoderShards = shared.encoderShards
        self.decoder = shared.decoder
        self.joint = shared.joint
        self.decoderJoint = shared.decoderJoint
        self.decoderJointArgmax = shared.decoderJointArgmax
        self.jointNoEncProjBatched = shared.jointNoEncProjBatched
        self.tokenizer = shared.tokenizer

        if let m = self.jointNoEncProjBatched,
            let constraint = m.modelDescription.inputDescriptionsByName["encoder_proj"]?
                .multiArrayConstraint,
            constraint.shape.count >= 2,
            constraint.shape[1].intValue > 0
        {
            self.jointNoEncProjBatchedK = constraint.shape[1].intValue
        }

        // joint.enc weights for the smart-spec encoder projection. Only the
        // smart-spec path consumes them, so skip them when it can't run.
        self.nativeRnnt = nil
        if let nativeDir = shared.nativeWeightsDir, self.jointNoEncProjBatched != nil {
            #if os(watchOS)
            let started = Date()
            print("[Load] native_weights start")
            #endif
            self.nativeRnnt = NativeRnntInner(directory: nativeDir)
            #if os(watchOS)
            print("[Load] native_weights done \(String(format: "%.2f", Date().timeIntervalSince(started)))s ok=\(self.nativeRnnt != nil)")
            #endif
        }

        try resetStates()

        // Per-stream prediction options (pre-allocated output backings —
        // cannot be shared across streams).
        self.decoderPredictionOptions = Self.makePredictionOptions(for: self.decoder)
        self.jointPredictionOptions = Self.makePredictionOptions(for: self.joint)
        self.decoderJointPredictionOptions = Self.makePredictionOptions(for: self.decoderJoint)
        self.decoderJointArgmaxPredictionOptions = Self.makePredictionOptions(
            for: self.decoderJointArgmax)
        self.jointNoEncProjBatchedPredictionOptions = Self.makePredictionOptions(
            for: self.jointNoEncProjBatched)

        // Per-stream reusable input buffers.
        self.encoderStepBuf = try MLMultiArray(
            shape: [1, NSNumber(value: config.encoderDim), 1], dataType: .float32)
        self.tokenInputBuf = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let tokLen = try MLMultiArray(shape: [1], dataType: .int32)
        tokLen[0] = 1
        self.tokenLenBuf = tokLen
        self.audioInputBuf = try MLMultiArray(
            shape: [1, NSNumber(value: config.chunkSamples)], dataType: .float32)
        self.audioLenBuf = try MLMultiArray(shape: [1], dataType: .int32)
        self.encProjReusable = nil
        self.encProjBatchReusable = nil

        logger.info(
            "Nemotron multilingual manager initialized from shared models (\(config.chunkMs)ms chunks)."
        )
    }

    /// Map a language hint (e.g. "en-US", "zh-CN", "de-DE", "auto") to the
    /// model folder in the HuggingFace repo: `latin` (Latin-script-pruned
    /// vocab for en/es/fr/it/pt/de) or `multilingual` (full vocab, incl.
    /// zh/ja, and "auto").
    public static func languageDirectory(for languageCode: String) -> String {
        let c = languageCode.lowercased()
        let latinPrefixes = ["en", "es", "fr", "it", "pt", "de"]
        if latinPrefixes.contains(where: { c.hasPrefix($0) }) { return "latin" }
        return "multilingual"
    }

    /// Load `<name>.mlmodelc` (or compile `<name>.mlpackage`, not on watchOS);
    /// throws if neither exists.
    private static func loadShared(
        directory: URL,
        name: String,
        configuration: MLModelConfiguration,
        logger: AppLogger
    ) async throws -> MLModel {
        guard
            let model = try await loadOptionalShared(
                directory: directory, name: name, configuration: configuration, logger: logger)
        else {
            throw ASRError.processingFailed(
                "Neither \(name).mlmodelc nor \(name).mlpackage found in \(directory.path)")
        }
        return model
    }

    /// Load `<name>.mlmodelc` (or compile `<name>.mlpackage`, not on watchOS);
    /// returns nil if neither exists.
    private static func loadOptionalShared(
        directory: URL,
        name: String,
        configuration: MLModelConfiguration,
        logger: AppLogger
    ) async throws -> MLModel? {
        let compiledName = "\(name).mlmodelc"
        let compiledURL = directory.appendingPathComponent(compiledName)
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            let started = Date()
            logger.info(
                "Loading shared \(name) from \(compiledName) with computeUnits=\(computeUnitsDescription(configuration))"
            )
            #if os(watchOS)
            // watchOS: AppLogger never reaches stdout, so trace loads with print
            // (visible via `devicectl device process launch --console`).
            print("[Load] \(name) start (\(computeUnitsDescription(configuration)))")
            #endif
            let m = try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
            #if os(watchOS)
            print("[Load] \(name) done \(String(format: "%.2f", Date().timeIntervalSince(started)))s")
            #endif
            logger.info(
                "Loaded shared \(compiledName) in \(String(format: "%.2f", Date().timeIntervalSince(started)))s"
            )
            return m
        }
        #if !os(watchOS)
        // MLModel.compileModel(at:) is unavailable on watchOS (the watch app
        // bundles only pre-compiled .mlmodelc directories).
        let packageName = "\(name).mlpackage"
        let packageURL = directory.appendingPathComponent(packageName)
        if FileManager.default.fileExists(atPath: packageURL.path) {
            let started = Date()
            logger.info(
                "Compiling shared \(name) from \(packageName) with computeUnits=\(computeUnitsDescription(configuration))"
            )
            let tempCompiledURL = try await MLModel.compileModel(at: packageURL)
            let m = try await MLModel.load(contentsOf: tempCompiledURL, configuration: configuration)
            logger.info(
                "Compiled + loaded shared \(packageName) in \(String(format: "%.2f", Date().timeIntervalSince(started)))s"
            )
            return m
        }
        #endif
        return nil
    }
}
