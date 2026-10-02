import CoreAI
import Foundation

/// Resolves staged Core AI asset filenames. The conversion pipeline emits the
/// int8 encoder shards as `encoder_shard_N_int8.aimodel`; earlier drops used
/// `encoder_shard_N.aimodel`. Accept both, preferring the int8 name.
enum CoreAIAssets {
    static func encoderShardURL(_ index: Int, in dir: URL) -> URL {
        let int8 = dir.appendingPathComponent("encoder_shard_\(index)_int8.aimodel")
        if FileManager.default.fileExists(atPath: int8.path) { return int8 }
        return dir.appendingPathComponent("encoder_shard_\(index).aimodel")
    }

    /// True when `dir` holds all four encoder shards.
    static func hasEncoder(in dir: URL) -> Bool {
        (0 ..< 4).allSatisfy {
            FileManager.default.fileExists(atPath: encoderShardURL($0, in: dir).path)
        }
    }
}

/// Decides per tier whether Core AI runs CPU-only instead of ANE-preferred, and
/// remembers Neural Engine run-time failures across launches.
///
/// The ANE failure seen on h18p (iOS 27.2) for the 4480 ms shards is an assertion
/// inside MPSGraph (`ANERegion.mm: ANE inference operation failed ... Code=-19`)
/// that aborts the process — Swift cannot catch it. So the runner sets a
/// per-tier "probe" flag before the first ANE inference and clears it once that
/// chunk succeeds (or throws); a launch that finds the flag still set knows the
/// previous run died there, records the tier as ANE-failed, and uses CPU from
/// then on. Reinstalling the app (or COREAI_FORCE_ANE=YES) resets/overrides this.
enum CoreAIComputePolicy {
    private static func probeKey(_ tier: String) -> String { "coreai.aneProbe.\(tier)" }
    private static func failedKey(_ tier: String) -> String { "coreai.aneFailed.\(tier)" }

    /// Tiers whose ANE plan is known to fail at run time on current devices.
    /// 4480 ms: Code -19 on the first chunk (h18p, iOS 27.2); CPU-only measured
    /// ~20x real time there, the fastest Core AI tier.
    static let knownANEFailingTiers: Set<Int> = [4480]

    /// COREAI_CPU_ONLY=YES (env) or the `coreai.cpuOnly` default forces every
    /// model CPU-only (no ANE, no GPU), regardless of tier.
    static var cpuOnlyForced: Bool {
        ProcessInfo.processInfo.environment["COREAI_CPU_ONLY"]?.uppercased() == "YES"
            || UserDefaults.standard.bool(forKey: "coreai.cpuOnly")
    }

    /// (cpuOnly, reason) for a tier. Also converts a stale probe flag (previous
    /// launch died during the first ANE inference) into a sticky ANE failure.
    static func decide(tier: String, chunkMs: Int) -> (cpuOnly: Bool, reason: String) {
        let d = UserDefaults.standard
        if d.bool(forKey: probeKey(tier)) {
            d.set(true, forKey: failedKey(tier))
            d.removeObject(forKey: probeKey(tier))
        }
        if cpuOnlyForced {
            return (true, "CPU-only (forced by COREAI_CPU_ONLY)")
        }
        if ProcessInfo.processInfo.environment["COREAI_FORCE_ANE"]?.uppercased() == "YES" {
            return (false, "ANE forced (COREAI_FORCE_ANE=YES)")
        }
        if d.bool(forKey: failedKey(tier)) {
            return (true, "CPU-only (a previous launch crashed in the first ANE inference)")
        }
        if knownANEFailingTiers.contains(chunkMs) {
            return (true, "CPU-only (ANE inference fails on the \(chunkMs) ms tier)")
        }
        return (false, "ANE-preferred")
    }

    static func beginANEProbe(tier: String) {
        UserDefaults.standard.set(true, forKey: probeKey(tier))
        UserDefaults.standard.synchronize()  // must hit disk before a possible abort
    }

    static func endANEProbe(tier: String) {
        UserDefaults.standard.removeObject(forKey: probeKey(tier))
    }
}

/// Runs the Nemotron Core AI pipeline's model calls: the four int8 encoder
/// shards (`encoder_shard_{0..3}_int8.aimodel`, hidden state threaded
/// shard0→1→2→3 with per-shard streaming caches) and the fused
/// `decoder_joint.aimodel` (decoder LSTM + batched joint in one dispatch).
///
/// Models load ANE-preferred with a `.cpuOnly` fallback, or CPU-only when
/// `CoreAIComputePolicy` says so. The GPU is never requested: iOS rejects GPU
/// work from a backgrounded app, and the `audio` background mode keeps the
/// mic session alive.
///
/// Core AI API surface (from `apple/coreai-models`): `AIModel(contentsOf:)`,
/// `model.functionNames`, `model.loadFunction(named:)` → `InferenceFunction`,
/// `fn.descriptor`, `NDArray(descriptor:)` + `mutableView(as:)`,
/// `fn.run(inputs:)` → `outputs.remove(name)?.ndArray`.
@MainActor
final class CoreAIEncoderRunner {

    private let shardURLs: [URL]
    private var shards: [AIModel] = []
    /// Cached per-shard InferenceFunctions. Reuse is safe because every output
    /// is consumed (copied out) before the same function runs again.
    private var shardFns: [InferenceFunction] = []

    enum RunnerError: LocalizedError {
        case shardMissing(Int, URL)
        case decoderJointMissing(URL)
        case functionMissing(URL)
        case outputMissing(String)
        case unsupportedScalarType

        var errorDescription: String? {
            switch self {
            case .shardMissing(let i, let u):
                return "Core AI encoder shard \(i) missing: \(u.lastPathComponent)"
            case .decoderJointMissing(let u):
                return "Core AI fused decoder missing: \(u.lastPathComponent) is required"
            case .functionMissing(let u):
                return "Core AI model \(u.lastPathComponent) exposes no function"
            case .outputMissing(let n): return "Core AI model produced no '\(n)' output"
            case .unsupportedScalarType: return "Unsupported NDArray scalar type for model I/O"
            }
        }
    }

    // MARK: - Streaming cache state (threaded chunk→chunk)
    //
    // The encoder is a cache-aware FastConformer: each shard owns 6 of the 24
    // conformer layers and carries its own left-context caches, exposed as I/O
    // (inputs cache_channel / cache_time / cache_len, outputs *_out). They MUST
    // be threaded across chunks — feeding zeros every chunk garbles words at
    // chunk boundaries. Mirrors the CoreML split-encoder path in
    // StreamingNemotronMultilingualAsrManager+Pipeline.swift.
    //
    // Held as persistent NDArrays allocated once per shard from the function's
    // own input descriptors (zero-filled = NeMo's all-zero initial state) and
    // refilled from each run's cache_*_out with a raw same-dtype copy.
    private var shardCacheChannelND = [NDArray?](repeating: nil, count: 4)
    private var shardCacheTimeND = [NDArray?](repeating: nil, count: 4)
    private var shardCacheLen = [Int32](repeating: 0, count: 4)

    /// Load every model CPU-only (CoreAIComputePolicy decided the ANE can't run
    /// this tier, or COREAI_CPU_ONLY forces it). When false, ANE-preferred.
    let cpuOnly: Bool
    /// Tier key for the crash-safe first-inference ANE probe; nil disables it.
    private let aneGuardTier: String?
    private var aneVerified = false

    init(coreaiDirectory dir: URL, cpuOnly: Bool = false, aneGuardTier: String? = nil) {
        let cpu = cpuOnly || CoreAIComputePolicy.cpuOnlyForced
        self.cpuOnly = cpu
        self.aneGuardTier = cpu ? nil : aneGuardTier
        self.shardURLs = (0 ..< 4).map { CoreAIAssets.encoderShardURL($0, in: dir) }
    }

    /// Release Core AI resources in a crash-safe order: the cached
    /// `InferenceFunction`s (and the streaming-cache NDArrays) BEFORE the
    /// `AIModel`s that back them. Releasing a CPU(BNNS)-delegated function after
    /// its backing model is gone over-releases inside the BNNS delegate and
    /// SIGSEGVs on the 27.0 beta. Call this on the main actor before dropping
    /// or replacing a runner.
    func tearDown() {
        decoderJointFn = nil
        shardFns.removeAll()
        shardCacheChannelND = [NDArray?](repeating: nil, count: 4)
        shardCacheTimeND = [NDArray?](repeating: nil, count: 4)
        decoderJointModel = nil
        shards.removeAll()
    }

    /// Reset streaming caches to the all-zero initial state (NeMo's
    /// `get_initial_cache_state`: zeros + cache_len 0). Call at the start of
    /// each utterance, before the first chunk.
    func resetStreamingCaches() {
        for idx in 0 ..< 4 {
            if var buf = shardCacheChannelND[idx] {
                Self.zeroFill(&buf)
                shardCacheChannelND[idx] = buf
            }
            if var buf = shardCacheTimeND[idx] {
                Self.zeroFill(&buf)
                shardCacheTimeND[idx] = buf
            }
        }
        shardCacheLen = [Int32](repeating: 0, count: 4)
    }

    // MARK: - Model loading

    /// App group whose container backs the CPU-only model cache (declared in
    /// the watch target's entitlements; on the watchOS beta the default cache
    /// location failed specialization with a bare ENOENT).
    nonisolated static let appGroupCacheId = "group.com.sdesai.NemotronASR"

    /// Load an `.aimodel` ANE-preferred (ops the ANE can't run fall to the
    /// CPU), retrying CPU-only if that specialization fails. `cpuOnly` skips
    /// straight to the CPU-only load.
    static func loadModel(at url: URL, cpuOnly: Bool = false) async throws -> AIModel {
        if cpuOnly {
            print("[CoreAI] loading \(url.lastPathComponent) .cpuOnly (compute policy)")
            return try await loadCPUOnly(at: url)
        }
        do {
            let model = try await AIModel(
                contentsOf: url,
                options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            print("[CoreAI] loaded \(url.lastPathComponent) (ANE-preferred)")
            return model
        } catch {
            print(
                "[CoreAI] ANE-preferred specialization failed for \(url.lastPathComponent): \(error) — retrying .cpuOnly"
            )
            return try await loadCPUOnly(at: url)
        }
    }

    /// Specialize a model CPU-only. Prefers the app-group model cache (the
    /// watchOS-beta ENOENT workaround), then a plain CPU-only load.
    private static func loadCPUOnly(at url: URL) async throws -> AIModel {
        if let cache = AIModelCache(appGroup: appGroupCacheId) {
            do {
                return try await AIModel.specialize(
                    contentsOf: url, options: .cpuOnly, cache: cache)
            } catch {
                print(
                    "[CoreAI] app-group .cpuOnly failed for \(url.lastPathComponent): \(error) — retrying plain .cpuOnly"
                )
            }
        }
        return try await AIModel(contentsOf: url, options: .cpuOnly)
    }

    /// Load all four encoder shards. Throws if any is missing or unloadable.
    func load() async throws {
        print(
            "[CoreAI] device arch: \(AIModel.deviceArchitectureName), available compute: \(ComputeUnitKind.availableKinds)"
        )
        var loaded: [AIModel] = []
        for (i, url) in shardURLs.enumerated() {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw RunnerError.shardMissing(i, url)
            }
            loaded.append(try await Self.loadModel(at: url, cpuOnly: cpuOnly))
        }
        shards = loaded

        // Cache one InferenceFunction per shard and allocate the persistent
        // streaming-cache buffers from its input descriptors.
        shardFns = []
        for (i, model) in shards.enumerated() {
            guard let fn = try model.loadFunction(named: model.functionNames.first ?? "main") else {
                throw RunnerError.functionMissing(shardURLs[i])
            }
            shardFns.append(fn)
            if fn.descriptor.inputNames.contains("cache_channel") {
                shardCacheChannelND[i] = try makeZeroNDArray(fn, name: "cache_channel")
            }
            if fn.descriptor.inputNames.contains("cache_time") {
                shardCacheTimeND[i] = try makeZeroNDArray(fn, name: "cache_time")
            }
        }
    }

    // MARK: - Encoder

    /// Run one chunk through shard0→1→2→3, threading the streaming caches and
    /// `length` across shards (and storing the updated caches for the next
    /// chunk). `mel` is `[nMels*T]` row-major from `MelFrontend`; returns the
    /// final `encoded` features flat (row-major) + shape.
    func encode(mel: [Float], frames T: Int, promptId: Int32) async throws -> (
        encoded: [Float], shape: [Int]
    ) {
        // Crash-safe ANE probe around the FIRST encoder inference of this runner
        // (see CoreAIComputePolicy): if the ANE aborts the process here, the next
        // launch finds the flag and switches this tier to CPU. A thrown error is
        // not an abort, so it clears the flag.
        let probeTier = aneVerified ? nil : aneGuardTier
        if let tier = probeTier { CoreAIComputePolicy.beginANEProbe(tier: tier) }
        do {
            let result = try await encodeChain(mel: mel, frames: T, promptId: promptId)
            if let tier = probeTier {
                CoreAIComputePolicy.endANEProbe(tier: tier)
                aneVerified = true
            }
            return result
        } catch {
            if let tier = probeTier { CoreAIComputePolicy.endANEProbe(tier: tier) }
            throw error
        }
    }

    private func encodeChain(mel: [Float], frames T: Int, promptId: Int32) async throws -> (
        encoded: [Float], shape: [Int]
    ) {
        // Hidden state threaded shard→shard as an NDArray. When its layout
        // matches the next shard's declared input it is passed straight
        // through (zero copies); otherwise flatten + rebuild (interleaved ANE
        // layouts).
        var dataND: NDArray? = nil
        var shape = [1, MelFrontend.nMels, T]
        var length = Int32(T)
        for (idx, fn) in shardFns.enumerated() {
            let inputName = idx == 0 ? "mel" : "hidden"
            let isLast = idx == shardFns.count - 1
            let outputName = isLast ? "encoded" : "hidden_out"

            // A Core AI function requires EVERY declared input.
            var inputs: [String: NDArray] = [:]
            for name in fn.descriptor.inputNames {
                switch name {
                case inputName:
                    if let nd = dataND {
                        if Self.matchesDeclaredInput(nd, fn: fn, name: name) {
                            inputs[name] = nd
                        } else {
                            inputs[name] = try makeFloatNDArray(
                                fn, name: name, data: try ndArrayToFloats(nd), shape: nd.shape)
                        }
                    } else {
                        inputs[name] = try makeFloatNDArray(fn, name: name, data: mel, shape: shape)
                    }
                case "length":
                    inputs[name] = try makeInt32NDArray(fn, name: name, data: [length], shape: [1])
                case "prompt_id":
                    inputs[name] = try makeInt32NDArray(
                        fn, name: name, data: [promptId], shape: [1])
                case "cache_channel":
                    inputs[name] = try shardCacheChannelND[idx] ?? makeZeroNDArray(fn, name: name)
                case "cache_time":
                    inputs[name] = try shardCacheTimeND[idx] ?? makeZeroNDArray(fn, name: name)
                case "cache_len":
                    inputs[name] = try makeInt32NDArray(
                        fn, name: name, data: [shardCacheLen[idx]], shape: [1])
                default:
                    inputs[name] = try makeZeroNDArray(fn, name: name)
                }
            }

            var outputs = try await fn.run(inputs: inputs)
            guard let out = outputs.remove(outputName)?.ndArray else {
                throw RunnerError.outputMissing(outputName)
            }
            shape = out.shape
            dataND = out

            // Thread length forward (shard 0's pre_encode downsamples 233→~29).
            if let lenOut = outputs.remove("length_out")?.ndArray {
                length = try ndArrayToInt32(lenOut).first ?? length
            }
            // Refill the persistent cache buffers with a raw same-dtype copy:
            // the runtime may reuse a function's output buffers on its next
            // run, so we must own the cache storage we feed back in.
            if let chOut = outputs.remove("cache_channel_out")?.ndArray,
                shardCacheChannelND[idx] != nil
            {
                try Self.copyContents(of: chOut, into: &shardCacheChannelND[idx]!)
            }
            if let tiOut = outputs.remove("cache_time_out")?.ndArray, shardCacheTimeND[idx] != nil {
                try Self.copyContents(of: tiOut, into: &shardCacheTimeND[idx]!)
            }
            if let lnOut = outputs.remove("cache_len_out")?.ndArray {
                shardCacheLen[idx] = try ndArrayToInt32(lnOut).first ?? shardCacheLen[idx]
            }
        }
        guard let encodedND = dataND else { throw RunnerError.outputMissing("encoded") }
        return (try ndArrayToFloats(encodedND), shape)
    }

    // MARK: - Fused decoder + joint (RNN-T inner loop)

    private var decoderJointModel: AIModel?
    private var decoderJointFn: InferenceFunction?

    /// Load the fused `decoder_joint.aimodel` (one dispatch per decoder state).
    func loadDecoderJoint(coreaiDirectory dir: URL) async throws {
        let url = dir.appendingPathComponent("decoder_joint.aimodel")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw RunnerError.decoderJointMissing(url)
        }
        let model = try await Self.loadModel(at: url, cpuOnly: cpuOnly)
        guard let fn = try model.loadFunction(named: model.functionNames.first ?? "main") else {
            throw RunnerError.functionMissing(url)
        }
        decoderJointModel = model
        decoderJointFn = fn
        print("[CoreAI] loaded fused decoder_joint.aimodel")
    }

    /// Build the fused decoder's `encoder` input [1, 1024, tEnc] once per
    /// chunk; every decode step of that chunk reuses it.
    func makeDecoderEncoderInput(_ encoded: [Float], tEnc: Int) throws -> NDArray {
        guard let fn = decoderJointFn else {
            throw RunnerError.functionMissing(URL(fileURLWithPath: "decoder_joint.aimodel"))
        }
        return try makeFloatNDArray(fn, name: "encoder", data: encoded, shape: [1, 1024, tEnc])
    }

    /// Fused decode step (decoder LSTM + batched joint) in ONE dispatch:
    /// (token, h, c, encoder[1,1024,T]) → (logits[T*vocab] row-major, hOut, cOut).
    func decodeJointStep(token: Int32, h: [Float], c: [Float], encoder: NDArray)
        async throws -> (logits: [Float], hOut: [Float], cOut: [Float])
    {
        guard let fn = decoderJointFn else {
            throw RunnerError.functionMissing(URL(fileURLWithPath: "decoder_joint.aimodel"))
        }
        var inputs: [String: NDArray] = [:]
        for name in fn.descriptor.inputNames {
            switch name {
            case "token":
                inputs[name] = try makeInt32NDArray(fn, name: name, data: [token], shape: [1, 1])
            case "token_length":
                inputs[name] = try makeInt32NDArray(fn, name: name, data: [1], shape: [1])
            case "h_in":
                inputs[name] = try makeFloatNDArray(fn, name: name, data: h, shape: [2, 1, 640])
            case "c_in":
                inputs[name] = try makeFloatNDArray(fn, name: name, data: c, shape: [2, 1, 640])
            case "encoder":
                inputs[name] = encoder
            default: inputs[name] = try makeZeroNDArray(fn, name: name)
            }
        }
        var outputs = try await fn.run(inputs: inputs)
        guard let logits = outputs.remove("logits")?.ndArray,
            let hOut = outputs.remove("h_out")?.ndArray,
            let cOut = outputs.remove("c_out")?.ndArray
        else { throw RunnerError.outputMissing("logits/h_out/c_out") }
        return (try ndArrayToFloats(logits), try ndArrayToFloats(hOut), try ndArrayToFloats(cOut))
    }

    // MARK: - NDArray helpers

    private func makeFloatNDArray(_ fn: InferenceFunction, name: String, data: [Float], shape: [Int])
        throws -> NDArray
    {
        guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else {
            throw RunnerError.unsupportedScalarType
        }
        let resolved = nd.resolvingDynamicDimensions(shape)
        var array = NDArray(descriptor: resolved)
        // copyElements respects the array's (possibly interleaved) layout — a flat
        // pointer write would mis-place elements on non-contiguous ANE buffers.
        switch resolved.scalarType {
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.copyElements(fromContentsOf: data)
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: data.map { Float16($0) })
        default:
            throw RunnerError.unsupportedScalarType
        }
        return array
    }

    private func makeInt32NDArray(
        _ fn: InferenceFunction, name: String, data: [Int32], shape: [Int]
    ) throws -> NDArray {
        guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else {
            throw RunnerError.unsupportedScalarType
        }
        var array = NDArray(descriptor: nd.resolvingDynamicDimensions(shape))
        var view = array.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: data)
        return array
    }

    /// Zero-filled NDArray matching a function input's declared descriptor
    /// (shape + scalar type) — used for the all-zero initial caches.
    private func makeZeroNDArray(_ fn: InferenceFunction, name: String) throws -> NDArray {
        guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else {
            throw RunnerError.unsupportedScalarType
        }
        var array = NDArray(descriptor: nd.resolvingDynamicDimensions(nd.shape))
        guard Self.zeroFill(&array) else { throw RunnerError.unsupportedScalarType }
        return array
    }

    /// Zero every element of an NDArray in place. Returns false for an
    /// unsupported scalar type.
    @discardableResult
    private static func zeroFill(_ array: inout NDArray) -> Bool {
        func fill<T: BitwiseCopyable>(_ type: T.Type, _ zero: T) {
            let count = array.shape.reduce(1, *)
            let v = array.mutableView(as: T.self)
            v.withUnsafeMutablePointer { ptr, _, _ in ptr.update(repeating: zero, count: count) }
        }
        switch array.scalarType {
        case .float32: fill(Float.self, 0)
        case .float16: fill(Float16.self, 0)
        case .int32: fill(Int32.self, 0)
        default: return false
        }
        return true
    }

    /// True when an upstream output NDArray can be fed directly as the input
    /// `name`: row-major contiguous with the declared shape + scalar type.
    private static func matchesDeclaredInput(_ array: NDArray, fn: InferenceFunction, name: String)
        -> Bool
    {
        guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else { return false }
        guard nd.shape == array.shape, nd.scalarType == array.scalarType else { return false }
        switch array.scalarType {
        case .float32: return withLayout(array, as: Float.self) { _, l in l.isRowMajorContiguous }
        case .float16: return withLayout(array, as: Float16.self) { _, l in l.isRowMajorContiguous }
        case .int32: return withLayout(array, as: Int32.self) { _, l in l.isRowMajorContiguous }
        default: return false
        }
    }

    /// Copy `src`'s elements into `dst` (same shape + scalar type) without any
    /// dtype conversion; `copyElements` respects `dst`'s layout. fp16 caches
    /// stay fp16 end-to-end.
    private static func copyContents(of src: NDArray, into dst: inout NDArray) throws {
        guard src.shape == dst.shape, src.scalarType == dst.scalarType else {
            throw RunnerError.unsupportedScalarType
        }
        func copy<T: BitwiseCopyable>(_ type: T.Type) {
            let flat = readRowMajor(src, as: T.self) { $0 }
            var view = dst.mutableView(as: T.self)
            view.copyElements(fromContentsOf: flat)
        }
        switch src.scalarType {
        case .float32: copy(Float.self)
        case .float16: copy(Float16.self)
        case .int32: copy(Int32.self)
        default: throw RunnerError.unsupportedScalarType
        }
    }

    /// Read an int32 NDArray (length_out / cache_len_out) to a flat `[Int32]`.
    private func ndArrayToInt32(_ array: NDArray) throws -> [Int32] {
        guard array.scalarType == .int32 else { throw RunnerError.unsupportedScalarType }
        return Self.readRowMajor(array, as: Int32.self) { $0 }
    }

    /// Stride-aware flatten to row-major `[Float]` — REQUIRED because Core AI /
    /// ANE outputs can be non-contiguous (interleaved) layouts; a naive flat
    /// read returns garbage for those (seen as all-blank RNN-T decode).
    private func ndArrayToFloats(_ array: NDArray) throws -> [Float] {
        switch array.scalarType {
        case .float32: return Self.readRowMajor(array, as: Float.self) { $0 }
        case .float16: return Self.readRowMajor(array, as: Float16.self) { Float($0) }
        default: throw RunnerError.unsupportedScalarType
        }
    }

    /// Shape + element strides of an NDArray view.
    private struct Layout {
        let shape: [Int]
        let strides: [Int]
        var count: Int { shape.reduce(1, *) }
        var isRowMajorContiguous: Bool {
            var expected = 1
            for d in (0 ..< shape.count).reversed() {
                if strides[d] != expected { return false }
                expected *= shape[d]
            }
            return true
        }
    }

    private static func withLayout<T: BitwiseCopyable, R>(
        _ array: NDArray, as type: T.Type, _ body: (UnsafePointer<T>, Layout) -> R
    ) -> R {
        array.view(as: T.self).withUnsafePointer { ptr, shape, strides in
            var s: [Int] = []
            var st: [Int] = []
            for d in 0 ..< shape.count {
                s.append(shape[d])
                st.append(strides[d])
            }
            return body(ptr, Layout(shape: s, strides: st))
        }
    }

    /// The single row-major reader: copies `array`'s elements (as `T`) in
    /// row-major order into a new `[U]`, converting each with `convert`. Fast
    /// path for row-major contiguous storage; otherwise walks the indices
    /// through the actual strides (ported from coreai-models `flattenNDArray`).
    private static func readRowMajor<T: BitwiseCopyable, U>(
        _ array: NDArray, as type: T.Type, convert: (T) -> U
    ) -> [U] {
        withLayout(array, as: T.self) { ptr, layout in
            let total = layout.count
            return [U](unsafeUninitializedCapacity: total) { buf, initialized in
                let out = buf.baseAddress!
                if layout.isRowMajorContiguous {
                    for i in 0 ..< total { (out + i).initialize(to: convert(ptr[i])) }
                } else {
                    let rank = layout.shape.count
                    var indices = [Int](repeating: 0, count: rank)
                    for i in 0 ..< total {
                        var offset = 0
                        for d in 0 ..< rank { offset += indices[d] * layout.strides[d] }
                        (out + i).initialize(to: convert(ptr[offset]))
                        var dim = rank - 1
                        while dim >= 0 {
                            indices[dim] += 1
                            if indices[dim] < layout.shape[dim] { break }
                            indices[dim] = 0
                            dim -= 1
                        }
                    }
                }
                initialized = total
            }
        }
    }
}
