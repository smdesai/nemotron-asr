@preconcurrency import CoreML
import Foundation

/// Immutable bundle of CoreML models + tokenizer + config that can be
/// shared across N independent `StreamingNemotronMultilingualAsrManager`
/// instances. Use this for multi-stream parallel inference where each
/// stream has its own cache/state but reuses the same compiled model
/// graphs to avoid O(N) memory blowup.
///
/// MLModel is thread-safe for `prediction(from:)` calls — multiple
/// streams may dispatch predictions concurrently against the same
/// model object. Per-stream mutable state (caches, hState/cState,
/// melCache, prediction output backings) stays inside the manager
/// actor.
public struct SharedNemotronMultilingualModels: Sendable {
    public let preprocessor: MLModel
    /// Monolithic encoder. Optional: the split-encoder ship (pre_encode + 4
    /// shards) omits it, and the consumer runs the sharded path instead.
    public let encoder: MLModel?
    /// Optional split encoder frontend: mel -> hidden. Loaded CPU-only.
    public let encoderPreEncode: MLModel?
    /// Optional split encoder layer shards. When all four are present, the
    /// manager can run pre_encode on CPU and the 24 Conformer layers on ANE.
    public let encoderShards: [MLModel]
    /// Bare prediction LSTM. Optional: a lean ship may omit it when B1
    /// (`decoderJoint`) covers the standard path and no smart-spec (K=4)
    /// asset is present (the smart-spec path is the only consumer of the
    /// unfused decoder, for `dec_out`).
    public let decoder: MLModel?
    /// Bare joint. Optional: only the smart-spec drain and the hybrid path
    /// use it standalone; the standard path uses B1.
    public let joint: MLModel?
    /// B1 fusion (decoder + joint merged). May be nil.
    public let decoderJoint: MLModel?
    /// B2 triple-fusion (decoder + joint + argmax). May be nil.
    public let decoderJointArgmax: MLModel?
    /// B3+B1 fusion (decoder + joint-without-encproj). May be nil.
    public let decoderJointNoEncProj: MLModel?
    /// Speculative batched joint. May be nil.
    public let jointBatched: MLModel?
    /// Smart-speculative batched joint. May be nil.
    public let jointNoEncProjBatched: MLModel?
    /// True iff the encoder uses MLState for cache (iOS 18+ stateful path).
    public let encoderIsStateful: Bool
    public let config: NemotronMultilingualStreamingConfig
    public let tokenizer: NemotronMultilingualTokenizer
    /// Optional native-Accelerate RNN-T weights blob directory.
    public let nativeWeightsDir: URL?
    /// MLModelConfiguration used to load these. Each manager uses the
    /// same configuration to stay on the same compute units.
    public let mlConfiguration: MLModelConfiguration

    fileprivate init(
        preprocessor: MLModel,
        encoder: MLModel?,
        encoderPreEncode: MLModel?,
        encoderShards: [MLModel],
        decoder: MLModel?,
        joint: MLModel?,
        decoderJoint: MLModel?,
        decoderJointArgmax: MLModel?,
        decoderJointNoEncProj: MLModel?,
        jointBatched: MLModel?,
        jointNoEncProjBatched: MLModel?,
        encoderIsStateful: Bool,
        config: NemotronMultilingualStreamingConfig,
        tokenizer: NemotronMultilingualTokenizer,
        nativeWeightsDir: URL?,
        mlConfiguration: MLModelConfiguration
    ) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.encoderPreEncode = encoderPreEncode
        self.encoderShards = encoderShards
        self.decoder = decoder
        self.joint = joint
        self.decoderJoint = decoderJoint
        self.decoderJointArgmax = decoderJointArgmax
        self.decoderJointNoEncProj = decoderJointNoEncProj
        self.jointBatched = jointBatched
        self.jointNoEncProjBatched = jointNoEncProjBatched
        self.encoderIsStateful = encoderIsStateful
        self.config = config
        self.tokenizer = tokenizer
        self.nativeWeightsDir = nativeWeightsDir
        self.mlConfiguration = mlConfiguration
    }
}

extension StreamingNemotronMultilingualAsrManager {

    /// Load all CoreML models + tokenizer + config ONCE, producing a
    /// shareable bundle that N managers can consume via
    /// `loadFromShared(_:)`. The single load cost is paid once; each
    /// consumer pays only its own per-stream state allocation.
    ///
    /// Memory footprint at N managers:
    /// - With per-manager loadModels(): N × (~1.5 GB models + ~50 MB state)
    /// - With shared+loadFromShared(): 1 × ~1.5 GB models + N × ~50 MB state
    ///
    /// `configuration` defaults to `.cpuAndNeuralEngine` (ANE path).
    public static func preloadShared(
        from directory: URL,
        commonDirectory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> SharedNemotronMultilingualModels {
        let logger = AppLogger(category: "NemotronMultilingualStreaming")
        // Model artifacts load from `directory`; the shared metadata.json /
        // tokenizer.json load from `commonDirectory` when provided (iOS keeps
        // them at the tier root, one level above `coreml/`). Defaults to
        // `directory` so the flat watch layout is unaffected.
        let commonDir = commonDirectory ?? directory

        guard SystemInfo.isAppleSilicon else {
            throw ASRError.unsupportedPlatform(
                "Nemotron multilingual int8 streaming models require Apple Silicon (ANE)."
            )
        }

        let mlConfiguration = configuration ?? MLModelConfigurationUtils.defaultConfiguration()
        let cpuOnlyConfiguration = MLModelConfigurationUtils.defaultConfiguration(computeUnits: .cpuOnly)
        let encoderConfiguration = Self.defaultEncoderConfiguration()
        logger.info("Preloading shared Nemotron multilingual models from \(directory.path)...")

        let metadataPath = commonDir.appendingPathComponent(ModelNames.NemotronMultilingualStreaming.metadata)
        guard FileManager.default.fileExists(atPath: metadataPath.path) else {
            throw ASRError.processingFailed(
                "metadata.json not found at \(metadataPath.path)."
            )
        }
        let config = try NemotronMultilingualStreamingConfig(from: metadataPath)
        logger.info(
            "Loaded multilingual config: \(config.chunkMs)ms chunks, vocab=\(config.vocabSize), \(config.numPrompts) prompts"
        )

        let preprocessor = try await Self.loadShared(
            directory: directory,
            compiledName: ModelNames.NemotronMultilingualStreaming.preprocessorFile,
            packageName: ModelNames.NemotronMultilingualStreaming.preprocessorPackage,
            configuration: Self.computeUnitOverride(
                name: "FLUIDAUDIO_PREPROCESSOR_CU", base: cpuOnlyConfiguration, logger: logger),
            logName: "preprocessor",
            logger: logger
        )

        // Monolithic encoder is OPTIONAL: the 2240ms ship uses the split encoder
        // (encoder_pre_encode + encoder_shard_0..3). Load it only if present;
        // a usable-encoder guard below enforces that at least one path exists.
        let encoder = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: ModelNames.NemotronMultilingualStreaming.encoderFile,
            packageName: ModelNames.NemotronMultilingualStreaming.encoderPackage,
            configuration: Self.computeUnitOverride(name: "FLUIDAUDIO_ENCODER_CU", base: encoderConfiguration, logger: logger),
            logName: "encoder",
            logger: logger
        )
        let encoderIsStateful: Bool
        if #available(macOS 15, iOS 18, *) {
            encoderIsStateful = !(encoder?.modelDescription.stateDescriptionsByName.isEmpty ?? true)
            if encoderIsStateful {
                logger.info("Encoder has MLState — per-stream state will be allocated on consumer init")
            }
        } else {
            encoderIsStateful = false
        }

        let encoderPreEncode = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: "encoder_pre_encode.mlmodelc",
            packageName: "encoder_pre_encode.mlpackage",
            configuration: cpuOnlyConfiguration,
            logName: "encoder_pre_encode",
            logger: logger
        )
        var encoderShards: [MLModel] = []
        // watchOS port: run the encoder shards (the bulk of the compute) on the
        // ANE — CPU-only was too slow to keep up with streaming, so no transcript
        // was produced. iOS keeps the shards pinned to the encoder base config.
        #if os(watchOS)
        let shardBaseConfiguration = MLModelConfigurationUtils.defaultConfiguration(
            computeUnits: .cpuAndNeuralEngine)
        #elseif os(iOS)
        let shardBaseConfiguration = encoderConfiguration
        #else
        let shardBaseConfiguration = mlConfiguration
        #endif
        for idx in 0..<4 {
            guard
                let shard = try await Self.loadOptionalShared(
                    directory: directory,
                    compiledName: "encoder_shard_\(idx).mlmodelc",
                    packageName: "encoder_shard_\(idx).mlpackage",
                    configuration: Self.computeUnitOverride(
                        name: "FLUIDAUDIO_ENCODER_SHARDS_CU", base: shardBaseConfiguration, logger: logger),
                    logName: "encoder_shard_\(idx)",
                    logger: logger
                )
            else {
                encoderShards.removeAll()
                break
            }
            encoderShards.append(shard)
        }
        if encoderPreEncode != nil && encoderShards.count == 4 {
            logger.info("Loaded split encoder path: encoder_pre_encode CPU + 4 encoder shards")
        } else if encoderPreEncode != nil || !encoderShards.isEmpty {
            logger.warning("Incomplete split encoder assets; falling back to monolithic encoder")
            encoderShards.removeAll()
        }

        // Require at least one usable encoder path: monolithic OR the complete
        // split set. (The watch ship is split-only — no encoder.mlmodelc.)
        guard encoder != nil || (encoderPreEncode != nil && encoderShards.count == 4) else {
            throw ASRError.processingFailed(
                "No usable encoder in \(directory.path): provide encoder.mlmodelc or the split "
                    + "encoder (encoder_pre_encode.mlmodelc + encoder_shard_0..3.mlmodelc).")
        }

        // Bare decoder + joint are now OPTIONAL. A lean B1 ship can omit them;
        // they're only consumed by the smart-spec (K=4) path and the hybrid
        // path. The standard decode path uses B1 (`decoderJoint`). A valid
        // decode path is enforced after the fused assets load (below).
        let decoder = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: ModelNames.NemotronMultilingualStreaming.decoderFile,
            packageName: ModelNames.NemotronMultilingualStreaming.decoderPackage,
            configuration: Self.computeUnitOverride(
                name: "FLUIDAUDIO_DECODER_CU", base: cpuOnlyConfiguration, logger: logger),
            logName: "decoder",
            logger: logger
        )

        let joint = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: ModelNames.NemotronMultilingualStreaming.jointFile,
            packageName: ModelNames.NemotronMultilingualStreaming.jointPackage,
            configuration: Self.computeUnitOverride(name: "FLUIDAUDIO_JOINT_CU", base: cpuOnlyConfiguration, logger: logger),
            logName: "joint",
            logger: logger
        )

        // Optional fusion mlpackages (B2 > B3+B1 > B1 priority — same
        // precedence as the per-manager loadModels path)
        let decoderJointArgmax = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: "decoder_joint_argmax.mlmodelc",
            packageName: "decoder_joint_argmax.mlpackage",
            configuration: Self.computeUnitOverride(
                name: "FLUIDAUDIO_DECODERJOINT_CU", base: cpuOnlyConfiguration, logger: logger),
            logName: "decoder_joint_argmax",
            logger: logger
        )
        var decoderJointNoEncProj: MLModel? = nil
        if decoderJointArgmax == nil {
            decoderJointNoEncProj = try await Self.loadOptionalShared(
                directory: directory,
                compiledName: "decoder_joint_noencproj.mlmodelc",
                packageName: "decoder_joint_noencproj.mlpackage",
                configuration: Self.computeUnitOverride(
                    name: "FLUIDAUDIO_DECODERJOINT_CU", base: cpuOnlyConfiguration, logger: logger),
                logName: "decoder_joint_noencproj",
                logger: logger
            )
        }
        var decoderJoint: MLModel? = nil
        if decoderJointArgmax == nil && decoderJointNoEncProj == nil {
            decoderJoint = try await Self.loadOptionalShared(
                directory: directory,
                compiledName: "decoder_joint.mlmodelc",
                packageName: "decoder_joint.mlpackage",
                configuration: Self.computeUnitOverride(
                    name: "FLUIDAUDIO_DECODERJOINT_CU", base: cpuOnlyConfiguration, logger: logger),
                logName: "decoder_joint",
                logger: logger
            )
        }
        let jointBatched = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: "joint_batched.mlmodelc",
            packageName: "joint_batched.mlpackage",
            configuration: Self.computeUnitOverride(
                name: "FLUIDAUDIO_JOINT_BATCHED_CU", base: cpuOnlyConfiguration, logger: logger),
            logName: "joint_batched",
            logger: logger
        )
        let jointNoEncProjBatched = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: "joint_noencproj_batched.mlmodelc",
            packageName: "joint_noencproj_batched.mlpackage",
            configuration: Self.computeUnitOverride(
                name: "FLUIDAUDIO_JOINT_BATCHED_CU", base: cpuOnlyConfiguration, logger: logger),
            logName: "joint_noencproj_batched",
            logger: logger
        )

        // Validate a usable decode path exists now that decoder/joint are
        // optional. Standard path needs a fused decoder_joint (B1/B3/B2) or
        // the bare decoder+joint pair. Smart-spec (K=4) consumes the bare
        // decoder (for dec_out) and bare joint (drain), so if it's present
        // both must be too — otherwise its force-unwraps would crash.
        let hasStandardPath =
            decoderJoint != nil || decoderJointNoEncProj != nil || decoderJointArgmax != nil
            || (decoder != nil && joint != nil)
        guard hasStandardPath else {
            throw ASRError.processingFailed(
                "No decode path in \(directory.path): provide a fused decoder_joint (B1/B3) "
                    + "or both bare decoder.mlmodelc + joint.mlmodelc.")
        }
        if jointNoEncProjBatched != nil && (decoder == nil || joint == nil) {
            throw ASRError.processingFailed(
                "Smart-spec asset joint_noencproj_batched present but bare decoder/joint missing "
                    + "— K=4 needs both. Either add them or remove the smart-spec asset.")
        }
        if decoder == nil && joint == nil {
            logger.info("Lean B1 ship: bare decoder/joint omitted; using fused decode path only.")
        }

        // Tokenizer (shared file — from the common dir)
        let tokenizerURL = commonDir.appendingPathComponent(ModelNames.NemotronMultilingualStreaming.tokenizer)
        let tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: tokenizerURL,
            langTagTokenIds: config.langTagTokenIds
        )

        let nativeWeightsDir = directory.appendingPathComponent("native_weights")
        let nativeAvailable = FileManager.default.fileExists(
            atPath: nativeWeightsDir.appendingPathComponent("weights.bin").path
        )

        logger.info("Shared models preload complete — ready for N consumers")

        return SharedNemotronMultilingualModels(
            preprocessor: preprocessor,
            encoder: encoder,
            encoderPreEncode: encoderPreEncode,
            encoderShards: encoderShards,
            decoder: decoder,
            joint: joint,
            decoderJoint: decoderJoint,
            decoderJointArgmax: decoderJointArgmax,
            decoderJointNoEncProj: decoderJointNoEncProj,
            jointBatched: jointBatched,
            jointNoEncProjBatched: jointNoEncProjBatched,
            encoderIsStateful: encoderIsStateful,
            config: config,
            tokenizer: tokenizer,
            nativeWeightsDir: nativeAvailable ? nativeWeightsDir : nil,
            mlConfiguration: mlConfiguration
        )
    }

    /// Initialize this manager from a pre-loaded shared model bundle.
    /// Each manager builds its OWN per-stream state (caches, MLState
    /// instance, prediction options with output backings, step buffers,
    /// NativeRnntInner) — only the MLModel handles are shared.
    public func loadFromShared(_ shared: SharedNemotronMultilingualModels) async throws {
        // Adopt shared configuration so prediction calls route through
        // the same compute units. Without this, the manager's default
        // MLModelConfiguration may differ from the shared bundle's.
        self.mlConfiguration = shared.mlConfiguration

        self.config = shared.config
        self.lastToken = Int32(config.blankIdx)
        self.currentPromptId = Int32(config.defaultPromptId)

        // Adopt shared MLModel references
        self.preprocessor = shared.preprocessor
        self.encoder = shared.encoder
        self.encoderPreEncode = shared.encoderPreEncode
        self.encoderShards = shared.encoderShards
        self.decoder = shared.decoder
        self.joint = shared.joint
        self.decoderJoint = shared.decoderJoint
        self.decoderJointArgmax = shared.decoderJointArgmax
        self.decoderJointNoEncProj = shared.decoderJointNoEncProj
        self.jointBatched = shared.jointBatched
        self.jointNoEncProjBatched = shared.jointNoEncProjBatched
        self.tokenizer = shared.tokenizer

        if let m = self.jointNoEncProjBatched,
            let constraint = m.modelDescription.inputDescriptionsByName["encoder_proj"]?.multiArrayConstraint,
            constraint.shape.count >= 2
        {
            let kFromModel = constraint.shape[1].intValue
            if kFromModel > 0 {
                self.jointNoEncProjBatchedK = kFromModel
            }
        }

        // Per-stream MLState instance (each stream gets its own).
        // makeState() returns a fresh zero-initialized state.
        if #available(macOS 15, iOS 18, *) {
            if shared.encoderIsStateful, let monolithicEncoder = shared.encoder {
                self.encoderState = monolithicEncoder.makeState()
            }
        }

        // Per-stream NativeRnntInner (has its own LSTM state buffers).
        if let nativeDir = shared.nativeWeightsDir {
            self.nativeRnnt = NativeRnntInner(directory: nativeDir)
        }

        // Per-stream cache/state init
        try resetStates()

        // Per-stream MLPredictionOptions (each contains pre-allocated
        // output buffers — CANNOT be shared across streams).
        self.encoderPredictionOptions = Self.makePredictionOptions(for: self.encoder)
        self.decoderPredictionOptions = Self.makePredictionOptions(for: self.decoder)
        self.jointPredictionOptions = Self.makePredictionOptions(for: self.joint)
        self.decoderJointPredictionOptions = Self.makePredictionOptions(for: self.decoderJoint)
        self.decoderJointArgmaxPredictionOptions = Self.makePredictionOptions(for: self.decoderJointArgmax)
        self.decoderJointNoEncProjPredictionOptions = Self.makePredictionOptions(for: self.decoderJointNoEncProj)
        self.jointNoEncProjBatchedPredictionOptions = Self.makePredictionOptions(for: self.jointNoEncProjBatched)

        // Per-stream inner-loop step buffers
        self.encoderStepBuf = try? MLMultiArray(shape: [1, NSNumber(value: config.encoderDim), 1], dataType: .float32)
        self.encoderProjStepBuf = try? MLMultiArray(shape: [1, 1, NSNumber(value: 640)], dataType: .float32)

        // Per-stream token input buffers
        if let tokInput = try? MLMultiArray(shape: [1, 1], dataType: .int32) {
            self.tokenInputBuf = tokInput
        }
        if let tokLen = try? MLMultiArray(shape: [1], dataType: .int32) {
            tokLen[0] = 1
            self.tokenLenBuf = tokLen
        }

        // Skip warmup — the shared models are already compiled & resident
        // from preloadShared(). The first real chunk pays no cold-start
        // penalty in any consumer.

        logger.info(
            "Nemotron multilingual manager initialized from shared models (\(config.chunkMs)ms chunks)."
        )
    }

    /// Map a language hint (e.g. "en-US", "zh-CN", "de-DE", "auto") to the
    /// model folder in the HuggingFace repo.
    ///
    /// The repo ships two models: `latin` (a Latin-script-pruned vocab shared by
    /// en/es/fr/it/pt/de — smaller, faster joint) and `multilingual` (the full
    /// 13087-token vocab covering every language, incl. zh/ja). Latin-script
    /// language hints route to `latin`; everything else, and "auto", falls back
    /// to the full-vocab `multilingual` model.
    public static func languageDirectory(for languageCode: String) -> String {
        let c = languageCode.lowercased()
        let latinPrefixes = ["en", "es", "fr", "it", "pt", "de"]
        if latinPrefixes.contains(where: { c.hasPrefix($0) }) { return "latin" }
        return "multilingual"
    }

    // `downloadAndPreloadShared(...)` and `downloadVariant(...)` are omitted in
    // the watchOS port. They depended on `DownloadUtils` / `Repo` (HuggingFace
    // download machinery, intentionally not vendored). The watch loads models
    // directly from the app bundle via `preloadShared(from:configuration:)`.

    /// Compile-if-needed + load helper for required model files.
    private static func loadShared(
        directory: URL,
        compiledName: String,
        packageName: String,
        configuration: MLModelConfiguration,
        logName: String,
        logger: AppLogger
    ) async throws -> MLModel {
        let compiledURL = directory.appendingPathComponent(compiledName)
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            let started = Date()
            logger.info(
                "Loading shared \(logName) from \(compiledName) with computeUnits=\(Self.computeUnitsDescription(configuration))"
            )
            let model = try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
            logger.info(
                "Loaded shared \(compiledName) in \(String(format: "%.2f", Date().timeIntervalSince(started)))s"
            )
            return model
        }
        let packageURL = directory.appendingPathComponent(packageName)
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw ASRError.processingFailed(
                "Neither \(compiledName) nor \(packageName) found in \(directory.path)"
            )
        }
        #if os(watchOS)
        // MLModel.compileModel(at:) is unavailable on watchOS. The watch app
        // bundles only pre-compiled .mlmodelc directories, so this branch is
        // never reached at runtime — fail loudly if a package is the only form.
        _ = packageURL
        throw ASRError.processingFailed(
            "On watchOS only pre-compiled .mlmodelc is supported; missing \(compiledName) in \(directory.path)"
        )
        #else
        let started = Date()
        logger.info(
            "Compiling shared \(logName) from \(packageName) with computeUnits=\(Self.computeUnitsDescription(configuration))"
        )
        let tempCompiledURL = try await MLModel.compileModel(at: packageURL)
        logger.info("Loading compiled shared \(logName) from temporary mlmodelc")
        let model = try await MLModel.load(contentsOf: tempCompiledURL, configuration: configuration)
        logger.info(
            "Compiled + loaded shared \(packageName) in \(String(format: "%.2f", Date().timeIntervalSince(started)))s"
        )
        return model
        #endif
    }

    /// Compile-if-needed + load helper for optional fusion bundles.
    /// Returns nil if neither the compiled nor the package form is present.
    private static func loadOptionalShared(
        directory: URL,
        compiledName: String,
        packageName: String,
        configuration: MLModelConfiguration,
        logName: String,
        logger: AppLogger
    ) async throws -> MLModel? {
        let compiledURL = directory.appendingPathComponent(compiledName)
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            let started = Date()
            logger.info(
                "Loading optional shared \(logName) from \(compiledName) with computeUnits=\(Self.computeUnitsDescription(configuration))"
            )
            let m = try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
            logger.info(
                "Loaded shared \(compiledName) in \(String(format: "%.2f", Date().timeIntervalSince(started)))s"
            )
            return m
        }
        #if !os(watchOS)
        let packageURL = directory.appendingPathComponent(packageName)
        if FileManager.default.fileExists(atPath: packageURL.path) {
            let started = Date()
            logger.info(
                "Compiling optional shared \(logName) from \(packageName) with computeUnits=\(Self.computeUnitsDescription(configuration))"
            )
            let tempCompiledURL = try await MLModel.compileModel(at: packageURL)
            logger.info("Loading compiled optional shared \(logName) from temporary mlmodelc")
            let m = try await MLModel.load(contentsOf: tempCompiledURL, configuration: configuration)
            logger.info(
                "Compiled + loaded shared \(packageName) in \(String(format: "%.2f", Date().timeIntervalSince(started)))s"
            )
            return m
        }
        #else
        _ = packageName
        #endif
        return nil
    }

}
