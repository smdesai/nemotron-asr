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

    /// Sideload directory for AOT-compiled assets (`coreai-build compile`),
    /// pushed without rebuilding the app:
    ///   xcrun devicectl device copy to --device <id> \
    ///     --source <model>.<arch>.aimodelc \
    ///     --destination Documents/CoreAICompiled/<model>.<arch>.aimodelc \
    ///     --domain-type appDataContainer --domain-identifier <bundle id>
    static var compiledSideloadDirectory: URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            .first?.appendingPathComponent("CoreAICompiled", isDirectory: true)
    }

    /// Resolve an ahead-of-time compiled variant of a source `.aimodel`:
    /// `<base>.<deviceArch>.aimodelc` (the naming `coreai-build compile`
    /// emits), looked up in the sideload directory first, then alongside the
    /// source model in the bundle.
    @available(iOS 27.0, watchOS 27.0, *)
    static func compiledVariant(of source: URL) -> URL? {
        let base = source.deletingPathExtension().lastPathComponent
        let name = "\(base).\(AIModel.deviceArchitectureName).aimodelc"
        var candidates: [URL] = []
        if let sideload = compiledSideloadDirectory {
            candidates.append(sideload.appendingPathComponent(name))
        }
        candidates.append(source.deletingLastPathComponent().appendingPathComponent(name))
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }
}

/// Runs the sharded Core AI encoder (`.aimodel` int8 shards) for the Nemotron
/// streaming ASR path — the Core AI analogue of FluidAudio's CoreML encoder.
///
/// Loads the four `encoder_shard_{0..3}.aimodel` bundles and threads hidden state
/// through them (shard0→1→2→3), as the conversion-side parity harness validated
/// (`shard_encoder_coreai.py`: fp32 chain cosine 1.0, int8 chain 50 dB).
///
/// The Core AI runtime API used here is the verified surface from the
/// `apple/coreai-models` Swift package (`CoreAIDiffusionModelFunction`,
/// `CoreAISequentialEngine`): `AIModel(contentsOf:)`, `model.functionNames`,
/// `model.loadFunction(named:)` → `InferenceFunction`, `fn.descriptor`,
/// `NDArray(descriptor:)` + `mutableView(as:)`, `fn.run(inputs:)` →
/// `outputs.remove(name)?.ndArray`.
///
/// Requires iOS 27 (`CoreAI` framework). Streaming-cache threading + the
/// decoder/joint/greedy loop are completed in the full runtime-integration step;
/// this runner proves the encoder shards load and run on real `MelFrontend` mel.
@available(iOS 27.0, watchOS 27.0, *)
@MainActor
final class CoreAIEncoderRunner {

    private let shardURLs: [URL]
    private var shards: [AIModel] = []
    /// Cached per-shard InferenceFunctions. The reference (coreai-models)
    /// reloaded the function each call citing stale outputs on alternating
    /// calls for stateless graphs — but every output here is consumed (copied
    /// out) before the same function runs again, which is what makes reuse
    /// safe. Set `reuseFunctions = false` to restore the per-chunk reload if
    /// a device ever shows alternating/stale encoder outputs.
    private var shardFns: [InferenceFunction] = []
    var reuseFunctions = true

    enum RunnerError: LocalizedError {
        case shardMissing(Int, URL)
        case functionMissing(URL)
        case outputMissing(String)
        case unsupportedScalarType

        var errorDescription: String? {
            switch self {
            case .shardMissing(let i, let u): return "Core AI encoder shard \(i) missing: \(u.lastPathComponent)"
            case .functionMissing(let u): return "Core AI shard \(u.lastPathComponent) exposes no function"
            case .outputMissing(let n): return "Core AI shard produced no '\(n)' output"
            case .unsupportedScalarType: return "Unsupported NDArray scalar type for shard I/O"
            }
        }
    }

    // MARK: - Streaming cache state (threaded chunk→chunk)
    //
    // The encoder is a cache-aware FastConformer: each shard owns 6 of the 24
    // conformer layers and carries its own left-context caches. The conversion
    // exposes them as I/O (`shard_encoder_coreai.py`): inputs cache_channel /
    // cache_time / cache_len, outputs cache_channel_out / cache_time_out /
    // cache_len_out. These MUST be threaded across chunks — feeding zeros every
    // chunk (cold start) is what garbles words at chunk boundaries. Mirrors the
    // PROVEN CoreML path `runShardedEncoderIfAvailable` in FluidAudio's
    // StreamingNemotronMultilingualAsrManager+Pipeline.swift.
    //
    // Held as persistent NDArrays allocated once per shard from the
    // function's own input descriptors (zero-filled = NeMo's all-zero initial
    // state), passed directly as inputs each chunk, and refilled from the
    // run's cache_*_out outputs with a raw same-dtype copy. This avoids the
    // previous flatten-to-[Float] + rebuild-NDArray round trip (millions of
    // scalar fp16↔fp32 conversions per chunk across the 4 shards).
    private var shardCacheChannelND = [NDArray?](repeating: nil, count: 4)
    private var shardCacheTimeND = [NDArray?](repeating: nil, count: 4)
    private var shardCacheLen = [Int32](repeating: 0, count: 4)

    init(coreaiDirectory dir: URL) {
        self.shardURLs = (0..<4).map { CoreAIAssets.encoderShardURL($0, in: dir) }
    }

    /// Release Core AI resources in a crash-safe order: the cached
    /// `InferenceFunction`s (and the streaming-cache NDArrays) BEFORE the
    /// `AIModel`s that back them. Under the GPU-free plan the decoder/joint
    /// functions are CPU(BNNS)-delegated; releasing a function after its backing
    /// model is already gone over-releases inside the BNNS delegate and SIGSEGVs
    /// on the 27.0 beta (faulting in `BNNSCoreAIDelegate` during
    /// `InferenceFunction` destroy). Call this explicitly on the main actor
    /// before dropping or replacing a runner — after it returns, the eventual
    /// implicit deinit has no Core AI refs left to mis-release.
    func tearDown() {
        // Functions first — they reference into the models / compute delegate.
        decoderFn = nil
        jointFn = nil
        decoderJointFn = nil
        shardFns.removeAll()
        // Then the streaming-cache NDArrays.
        shardCacheChannelND = [NDArray?](repeating: nil, count: 4)
        shardCacheTimeND = [NDArray?](repeating: nil, count: 4)
        // Models last.
        decoderModel = nil
        jointModel = nil
        decoderJointModel = nil
        shards.removeAll()
    }

    /// Reset streaming caches to the all-zero initial state (NeMo's
    /// `get_initial_cache_state`: zeros + cache_len 0). Call at the start of
    /// each utterance, before the first chunk.
    func resetStreamingCaches() {
        for idx in 0..<4 {
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

    /// Load an .aimodel, retrying with explicit compute-unit specialization
    /// when the default fails. On the watchOS 27 beta, `.default`
    /// specialization can fail with a bare POSIX ENOENT — the specializer
    /// appears to route through a compiler path that doesn't exist on the
    /// platform (watchOS has no GPU/Metal stack) — while pinning to the ANE
    /// or CPU uses a path that does.
    /// App group whose container backs the fallback model cache (declared in
    /// the watch target's entitlements).
    static let appGroupCacheId = "group.com.sdesai.NemotronASR"

    static func loadModel(at url: URL) async throws -> AIModel {
        // Ahead-of-time GPU-compiled variant (coreai-build compile
        // --preferred-compute gpu): load with matching GPU specialization
        // options per Apple's AOT guidance ("specify options that match the
        // compute units you used at compile time"), retrying with defaults
        // before falling back to the source .aimodel. To bypass AOT assets
        // (A/B against the default ANE-planned runtime specialization),
        // launch with the env var COREAI_AOT=NO — via devicectl:
        //   DEVICECTL_CHILD_COREAI_AOT=NO xcrun devicectl device process launch …
        let env = ProcessInfo.processInfo.environment

        // ── Background safety: keep the whole pipeline off the GPU ──
        // iOS rejects GPU (Metal) command-buffer submission from a backgrounded
        // app (kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted).
        // The `audio` background mode keeps CPU + ANE alive but never the GPU,
        // and the fp32 decoder/joint otherwise default-plan to the GPU (the int8
        // encoder shards are already ANE-clean). So by DEFAULT load every model
        // GPU-free: ANE-preferred (ops the ANE can't run fall to the CPU), never
        // requesting the GPU. Opt out for foreground-only A/B perf testing with
        // COREAI_GPU_FREE=NO; force the most conservative CPU-only plan with
        // COREAI_CPU_ONLY=YES (guaranteed no GPU, no ANE).
        if env["COREAI_CPU_ONLY"] == "YES" || UserDefaults.standard.bool(forKey: "coreai.cpuOnly") {
            print("[CoreAI] loading \(url.lastPathComponent) .cpuOnly (forced)")
            return try await loadCPUOnly(at: url)
        }
        let gpuFree = env["COREAI_GPU_FREE"]?.uppercased() != "NO"
            && (UserDefaults.standard.object(forKey: "coreai.gpuFree") == nil
                || UserDefaults.standard.bool(forKey: "coreai.gpuFree"))
        if gpuFree {
            do {
                let model = try await AIModel(
                    contentsOf: url,
                    options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
                print("[CoreAI] loaded \(url.lastPathComponent) GPU-free (ANE-preferred)")
                return model
            } catch {
                print("[CoreAI] ANE-preferred specialization failed for \(url.lastPathComponent): \(error) — retrying .cpuOnly")
                return try await loadCPUOnly(at: url)
            }
        }

        var aotEnabled = env["COREAI_AOT"] != "NO"
            && (UserDefaults.standard.object(forKey: "coreai.aot") == nil
                || UserDefaults.standard.bool(forKey: "coreai.aot"))
        // COREAI_AOT_MODELS="encoder" (or "decoder,joint", …): restrict AOT
        // assets to models whose filename starts with one of the given
        // prefixes — lets a single device session mix AOT-GPU and JIT-ANE
        // components to isolate accuracy/perf differences.
        //
        // Default safe-list: decoder,joint only. The int8 encoder shards'
        // GPU plan has a verified accuracy regression on h18p (drops the
        // first chunks of an utterance and garbles content; 77 vs 129 tokens
        // on the same audio) while the fp32 decoder/joint GPU plans matched
        // the ANE output exactly. Set COREAI_AOT_MODELS=encoder,decoder,joint
        // to re-test the encoder on a future OS/compiler.
        if aotEnabled {
            let filter = env["COREAI_AOT_MODELS"] ?? "decoder,joint"
            let base = url.deletingPathExtension().lastPathComponent
            aotEnabled = filter.split(separator: ",").contains {
                base.hasPrefix($0.trimmingCharacters(in: .whitespaces))
            }
        }
        if aotEnabled, let compiled = CoreAIAssets.compiledVariant(of: url) {
            do {
                let model = try await AIModel(
                    contentsOf: compiled,
                    options: SpecializationOptions(preferredComputeUnitKind: .gpu))
                print("[CoreAI] loaded AOT GPU asset \(compiled.lastPathComponent)")
                return model
            } catch {
                print("[CoreAI] AOT asset \(compiled.lastPathComponent) failed with GPU options: \(error) — retrying default options")
                do {
                    let model = try await AIModel(contentsOf: compiled)
                    print("[CoreAI] loaded AOT asset \(compiled.lastPathComponent) (default options)")
                    return model
                } catch {
                    print("[CoreAI] AOT asset \(compiled.lastPathComponent) failed: \(error) — falling back to source .aimodel")
                }
            }
        }
        // A/B switch: COREAI_FORCE_GPU=YES (env) or `-coreai.forceGPU YES`
        // (launch argument) requests runtime (JIT) GPU specialization of the
        // source .aimodel — for comparing GPU vs the default (ANE) plan
        // without AOT assets.
        if env["COREAI_FORCE_GPU"] == "YES" || UserDefaults.standard.bool(forKey: "coreai.forceGPU") {
            do {
                let model = try await AIModel(
                    contentsOf: url,
                    options: SpecializationOptions(preferredComputeUnitKind: .gpu))
                print("[CoreAI] loaded \(url.lastPathComponent) with runtime GPU specialization")
                return model
            } catch {
                print("[CoreAI] runtime GPU specialization failed for \(url.lastPathComponent): \(error) — falling back to default")
            }
        }
        do {
            return try await AIModel(contentsOf: url)
        } catch {
            print("[CoreAI] default specialization failed for \(url.lastPathComponent): \(error) — retrying app-group cache")
            // The DEFAULT AIModelCache location is one suspect for the bare
            // ENOENT on the watchOS beta — retry with a cache rooted in our
            // own app-group container before falling back on compute pins.
            if let cache = AIModelCache(appGroup: appGroupCacheId) {
                do {
                    return try await AIModel.specialize(contentsOf: url, cache: cache)
                } catch {
                    print("[CoreAI] app-group cache specialization failed: \(error) — retrying .cpuOnly in app-group cache")
                    do {
                        return try await AIModel.specialize(contentsOf: url, options: .cpuOnly, cache: cache)
                    } catch {
                        print("[CoreAI] app-group cpuOnly failed: \(error)")
                    }
                }
            } else {
                print("[CoreAI] app-group cache unavailable (\(appGroupCacheId))")
            }
            do {
                return try await AIModel(
                    contentsOf: url,
                    options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            } catch {
                print("[CoreAI] ANE specialization failed for \(url.lastPathComponent): \(error) — retrying .cpuOnly")
                return try await AIModel(contentsOf: url, options: .cpuOnly)
            }
        }
    }

    /// Specialize a model without ever requesting the GPU. Prefers the app-group
    /// model cache (the watchOS-beta ENOENT workaround) then a plain CPU-only
    /// load. Used by the default GPU-free path and the forced CPU-only override.
    private static func loadCPUOnly(at url: URL) async throws -> AIModel {
        if let cache = AIModelCache(appGroup: appGroupCacheId) {
            do {
                return try await AIModel.specialize(contentsOf: url, options: .cpuOnly, cache: cache)
            } catch {
                print("[CoreAI] app-group .cpuOnly failed for \(url.lastPathComponent): \(error) — retrying plain .cpuOnly")
            }
        }
        return try await AIModel(contentsOf: url, options: .cpuOnly)
    }

    /// Load all four shards. Throws if any is missing or unloadable — this is the
    /// on-device check of the staged int8 `.aimodel` assets.
    func load() async throws {
        print("[CoreAI] device arch: \(AIModel.deviceArchitectureName), available compute: \(ComputeUnitKind.availableKinds)")
        var loaded: [AIModel] = []
        for (i, url) in shardURLs.enumerated() {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw RunnerError.shardMissing(i, url)
            }
            loaded.append(try await Self.loadModel(at: url))
        }
        shards = loaded

        // Cache one InferenceFunction per shard and allocate the persistent
        // streaming-cache buffers from its input descriptors (zero-filled =
        // the all-zero initial cache state).
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

    /// Function names of each loaded shard — lets the engine confirm the load and
    /// report the on-device graph structure (e.g. static/NE vs dynamic).
    var shardFunctionNames: [[String]] { shards.map { $0.functionNames } }

    /// Run one chunk through shard0→1→2→3, threading the streaming caches and
    /// `length` across shards (and storing the updated caches for the next
    /// chunk). `mel` is `[nMels*T]` row-major from `MelFrontend`; returns the
    /// final `encoded` features flat + shape.
    ///
    /// Cache threading (the fix for chunk-boundary garbling): each shard reads
    /// its current per-shard cache and emits `cache_*_out`; we feed those back
    /// in on the next chunk. `length` likewise flows shard→shard from each
    /// shard's `length_out` (shard 0 takes the full mel length 233; shards 1-3
    /// take the post-pre-encode length). Matches FluidAudio's proven CoreML
    /// `runShardedEncoderIfAvailable`.
    func encode(mel: [Float], frames T: Int, promptId: Int32) async throws -> (encoded: [Float], shape: [Int]) {
        var data = mel
        // Hidden state threaded shard→shard as an NDArray. When its layout
        // matches the next shard's declared input it is passed straight
        // through (zero copies); otherwise it falls back to flatten + rebuild
        // (handles interleaved ANE layouts).
        var dataND: NDArray? = nil
        var shape = [1, MelFrontend.nMels, T]
        var length = Int32(T)
        for (idx, model) in shards.enumerated() {
            let inputName = idx == 0 ? "mel" : "hidden"
            let fn: InferenceFunction
            if reuseFunctions, idx < shardFns.count {
                fn = shardFns[idx]
            } else {
                // Escape hatch: per-chunk reload (the original reference
                // behavior for stale-output-suspect stateless graphs).
                guard let fresh = try model.loadFunction(named: model.functionNames.first ?? "main") else {
                    throw RunnerError.functionMissing(shardURLs[idx])
                }
                fn = fresh
            }
            let outputName = idx == 3 ? "encoded" : "hidden_out"

            // A Core AI function requires EVERY declared input. Provide the data
            // input + threaded length/caches; the persistent cache buffers are
            // zero-filled at load/reset (NeMo's all-zero initial cache state).
            var inputs: [String: NDArray] = [:]
            for name in fn.descriptor.inputNames {
                switch name {
                case inputName:
                    if let nd = dataND {
                        if Self.matchesDeclaredInput(nd, fn: fn, name: name) {
                            inputs[name] = nd
                        } else {
                            let flat = try ndArrayToFloats(nd)
                            inputs[name] = try makeFloatNDArray(fn, name: name, data: flat, shape: nd.shape)
                        }
                    } else {
                        inputs[name] = try makeFloatNDArray(fn, name: name, data: data, shape: shape)
                    }
                case "length":
                    inputs[name] = try makeInt32NDArray(fn, name: name, data: [length], shape: [1])
                case "prompt_id":
                    inputs[name] = try makeInt32NDArray(fn, name: name, data: [promptId], shape: [1])
                case "cache_channel":
                    if let buf = shardCacheChannelND[idx] {
                        inputs[name] = buf
                    } else {
                        inputs[name] = try makeZeroNDArray(fn, name: name)
                    }
                case "cache_time":
                    if let buf = shardCacheTimeND[idx] {
                        inputs[name] = buf
                    } else {
                        inputs[name] = try makeZeroNDArray(fn, name: name)
                    }
                case "cache_len":
                    inputs[name] = try makeInt32NDArray(fn, name: name, data: [shardCacheLen[idx]], shape: [1])
                default:
                    inputs[name] = try makeZeroNDArray(fn, name: name)
                }
            }

            var outputs = try await fn.run(inputs: inputs)
            guard let out = outputs.remove(outputName)?.ndArray else {
                throw RunnerError.outputMissing(outputName)
            }
            shape = out.shape
            if idx == 3 {
                // Final shard: the decode loop needs [Float].
                data = try ndArrayToFloats(out)
            } else {
                dataND = out
            }

            // Thread length forward (shard 0's pre_encode downsamples 233→~29).
            if let lenOut = outputs.remove("length_out")?.ndArray {
                length = try ndArrayToInt32(lenOut).first ?? length
            }
            // Refill the persistent cache buffers from this run's outputs with
            // a raw same-dtype copy. Copying (vs retaining the output NDArray)
            // matters with cached functions: the runtime may reuse a function's
            // output buffers on its next run, so we must own the cache storage
            // we feed back in.
            if let chOut = outputs.remove("cache_channel_out")?.ndArray, shardCacheChannelND[idx] != nil {
                try Self.copyContents(of: chOut, into: &shardCacheChannelND[idx]!)
            }
            if let tiOut = outputs.remove("cache_time_out")?.ndArray, shardCacheTimeND[idx] != nil {
                try Self.copyContents(of: tiOut, into: &shardCacheTimeND[idx]!)
            }
            if let lnOut = outputs.remove("cache_len_out")?.ndArray {
                shardCacheLen[idx] = try ndArrayToInt32(lnOut).first ?? shardCacheLen[idx]
            }
        }
        return (data, shape)
    }

    // MARK: - Decoder + Joint (RNN-T inner loop)

    private var decoderModel: AIModel?
    private var jointModel: AIModel?
    // Cache the loaded InferenceFunctions — re-loading per token step is the
    // dominant overhead (the reference engine caches these lazily too).
    private var decoderFn: InferenceFunction?
    private var jointFn: InferenceFunction?
    private var decoderJointModel: AIModel?
    private var decoderJointFn: InferenceFunction?

    /// Load the fused `decoder_joint.aimodel` if present (one dispatch/token);
    /// otherwise the separate decoder.aimodel + joint.aimodel.
    func loadDecoderJoint(coreaiDirectory dir: URL) async throws {
        let fusedURL = dir.appendingPathComponent("decoder_joint.aimodel")
        if FileManager.default.fileExists(atPath: fusedURL.path) {
            let fm = try await Self.loadModel(at: fusedURL)
            decoderJointModel = fm
            decoderJointFn = try fm.loadFunction(named: fm.functionNames.first ?? "main")
            print("[CoreAI] loaded fused decoder_joint.aimodel (one dispatch/token)")
            return
        }
        let decURL = dir.appendingPathComponent("decoder.aimodel")
        let jntURL = dir.appendingPathComponent("joint.aimodel")
        guard FileManager.default.fileExists(atPath: decURL.path) else {
            throw RunnerError.shardMissing(-1, decURL)
        }
        guard FileManager.default.fileExists(atPath: jntURL.path) else {
            throw RunnerError.shardMissing(-2, jntURL)
        }
        let dm = try await Self.loadModel(at: decURL)
        let jm = try await Self.loadModel(at: jntURL)
        decoderModel = dm
        jointModel = jm
        decoderFn = try dm.loadFunction(named: dm.functionNames.first ?? "main")
        jointFn = try jm.loadFunction(named: jm.functionNames.first ?? "main")
    }

    /// One decoder step: (token, h[2,1,640], c[2,1,640]) → (decoder_out flat, h_out, c_out).
    /// Shapes match `DecoderWrapper`: token[1,1] int32, token_length[1]=1.
    func decoderStep(token: Int32, h: [Float], c: [Float]) async throws
        -> (decoderOut: [Float], decShape: [Int], hOut: [Float], cOut: [Float])
    {
        guard let fn = decoderFn
        else { throw RunnerError.functionMissing(URL(fileURLWithPath: "decoder")) }

        var inputs: [String: NDArray] = [:]
        for name in fn.descriptor.inputNames {
            switch name {
            case "token": inputs[name] = try makeInt32NDArray(fn, name: name, data: [token], shape: [1, 1])
            case "token_length": inputs[name] = try makeInt32NDArray(fn, name: name, data: [1], shape: [1])
            case "h_in": inputs[name] = try makeFloatNDArray(fn, name: name, data: h, shape: [2, 1, 640])
            case "c_in": inputs[name] = try makeFloatNDArray(fn, name: name, data: c, shape: [2, 1, 640])
            default: inputs[name] = try makeZeroNDArray(fn, name: name)
            }
        }
        var outputs = try await fn.run(inputs: inputs)
        guard let dOut = outputs.remove("decoder_out")?.ndArray,
              let hOut = outputs.remove("h_out")?.ndArray,
              let cOut = outputs.remove("c_out")?.ndArray
        else { throw RunnerError.outputMissing("decoder_out/h_out/c_out") }
        return (try ndArrayToFloats(dOut), dOut.shape, try ndArrayToFloats(hOut), try ndArrayToFloats(cOut))
    }

    /// One joint step: (encoderFrame[1024], decoderStep[640]) → logits[vocab].
    /// Joint expects encoder[1,1024,1] and decoder[1,640,1] (per JointWrapper).
    func jointStep(encoderFrame: [Float], decoderStep: [Float]) async throws -> [Float] {
        guard let fn = jointFn
        else { throw RunnerError.functionMissing(URL(fileURLWithPath: "joint")) }

        var inputs: [String: NDArray] = [:]
        for name in fn.descriptor.inputNames {
            switch name {
            case "encoder": inputs[name] = try makeFloatNDArray(fn, name: name, data: encoderFrame, shape: [1, 1024, 1])
            case "decoder": inputs[name] = try makeFloatNDArray(fn, name: name, data: decoderStep, shape: [1, 640, 1])
            default: inputs[name] = try makeZeroNDArray(fn, name: name)
            }
        }
        var outputs = try await fn.run(inputs: inputs)
        guard let logits = outputs.remove("logits")?.ndArray else {
            throw RunnerError.outputMissing("logits")
        }
        return try ndArrayToFloats(logits)
    }

    /// Batched joint: encoder[1,1024,T] (ALL frames, row-major from `encode`) x
    /// decoder[1,640,1] → logits[1,T,1,vocab] flattened to [T*vocab]. One dispatch
    /// covers every encoder frame for a fixed decoder state, so the greedy loop
    /// scans the returned logits instead of re-dispatching the joint per frame
    /// (~2x fewer joint calls, dispatch overhead amortized over T). Requires the
    /// joint exported with `--joint-frames T` (encoder time dim = T).
    func jointBatched(encoder: [Float], tEnc: Int, decoderStep: [Float]) async throws -> [Float] {
        guard let fn = jointFn
        else { throw RunnerError.functionMissing(URL(fileURLWithPath: "joint")) }
        var inputs: [String: NDArray] = [:]
        for name in fn.descriptor.inputNames {
            switch name {
            case "encoder": inputs[name] = try makeFloatNDArray(fn, name: name, data: encoder, shape: [1, 1024, tEnc])
            case "decoder": inputs[name] = try makeFloatNDArray(fn, name: name, data: decoderStep, shape: [1, 640, 1])
            default: inputs[name] = try makeZeroNDArray(fn, name: name)
            }
        }
        var outputs = try await fn.run(inputs: inputs)
        guard let logits = outputs.remove("logits")?.ndArray else {
            throw RunnerError.outputMissing("logits")
        }
        return try ndArrayToFloats(logits)  // [1,T,1,vocab] row-major -> logits[t*vocab + v]
    }

    /// Fused decode step (decoder LSTM + batched joint) in ONE dispatch:
    /// (token, h, c, encoder[1,1024,T]) -> (logits[T*vocab], hOut, cOut).
    /// Returns nil if the fused `decoder_joint.aimodel` wasn't loaded, so the
    /// caller can fall back to decoderStep + jointBatched.
    func decodeJointStep(token: Int32, h: [Float], c: [Float], encoder: [Float], tEnc: Int)
        async throws -> (logits: [Float], hOut: [Float], cOut: [Float])?
    {
        guard let fn = decoderJointFn else { return nil }
        var inputs: [String: NDArray] = [:]
        for name in fn.descriptor.inputNames {
            switch name {
            case "token": inputs[name] = try makeInt32NDArray(fn, name: name, data: [token], shape: [1, 1])
            case "token_length": inputs[name] = try makeInt32NDArray(fn, name: name, data: [1], shape: [1])
            case "h_in": inputs[name] = try makeFloatNDArray(fn, name: name, data: h, shape: [2, 1, 640])
            case "c_in": inputs[name] = try makeFloatNDArray(fn, name: name, data: c, shape: [2, 1, 640])
            case "encoder": inputs[name] = try makeFloatNDArray(fn, name: name, data: encoder, shape: [1, 1024, tEnc])
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

    // MARK: - NDArray helpers (verified pattern from CoreAIDiffusionModelFunction)

    func makeFloatNDArray(_ fn: InferenceFunction, name: String, data: [Float], shape: [Int]) throws -> NDArray {
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
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: data.map { Float16($0) })
        #endif
        default:
            throw RunnerError.unsupportedScalarType
        }
        return array
    }

    /// True when an upstream output NDArray can be fed directly as the input
    /// `name`: row-major contiguous with the declared shape + scalar type.
    /// Interleaved ANE layouts (or shape/dtype drift) fall back to the
    /// flatten + rebuild path.
    private static func matchesDeclaredInput(_ array: NDArray, fn: InferenceFunction, name: String) -> Bool {
        guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else { return false }
        guard nd.shape == array.shape, nd.scalarType == array.scalarType else { return false }
        return isRowMajorContiguous(array)
    }

    private static func isRowMajorContiguous(_ array: NDArray) -> Bool {
        func check<T: BitwiseCopyable>(_ type: T.Type) -> Bool {
            var contiguous = true
            array.view(as: T.self).withUnsafePointer { _, shape, strides in
                var expected = 1
                for d in (0..<shape.count).reversed() {
                    if strides[d] != expected {
                        contiguous = false
                        return
                    }
                    expected *= shape[d]
                }
            }
            return contiguous
        }
        switch array.scalarType {
        case .float32: return check(Float.self)
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16: return check(Float16.self)
        #endif
        case .int32: return check(Int32.self)
        default: return false
        }
    }

    /// Copy `src`'s elements into `dst` (same shape + scalar type) without any
    /// dtype conversion: a stride-aware read into a same-dtype staging buffer,
    /// then `copyElements` (which respects `dst`'s layout). fp16 caches stay
    /// fp16 end-to-end — no per-element Float16↔Float conversion.
    private static func copyContents(of src: NDArray, into dst: inout NDArray) throws {
        guard src.shape == dst.shape, src.scalarType == dst.scalarType else {
            throw RunnerError.unsupportedScalarType
        }
        switch src.scalarType {
        case .float32: copyTyped(src, &dst, as: Float.self)
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16: copyTyped(src, &dst, as: Float16.self)
        #endif
        case .int32: copyTyped(src, &dst, as: Int32.self)
        default: throw RunnerError.unsupportedScalarType
        }
    }

    private static func copyTyped<T: BitwiseCopyable>(_ src: NDArray, _ dst: inout NDArray, as type: T.Type) {
        let total = src.shape.reduce(1, *)
        let flat = [T](unsafeUninitializedCapacity: total) { buf, initialized in
            src.view(as: T.self).withUnsafePointer { ptr, shape, strides in
                let rank = shape.count
                // Fast path: row-major contiguous.
                var expectedStride = 1
                var isContiguous = true
                for d in (0..<rank).reversed() {
                    if strides[d] != expectedStride {
                        isContiguous = false
                        break
                    }
                    expectedStride *= shape[d]
                }
                if isContiguous {
                    buf.baseAddress!.update(from: ptr, count: total)
                } else {
                    // Slow path: walk row-major indices through actual strides.
                    var indices = [Int](repeating: 0, count: rank)
                    for i in 0..<total {
                        var offset = 0
                        for d in 0..<rank { offset += indices[d] * strides[d] }
                        buf[i] = ptr[offset]
                        var dim = rank - 1
                        while dim >= 0 {
                            indices[dim] += 1
                            if indices[dim] < shape[dim] { break }
                            indices[dim] = 0
                            dim -= 1
                        }
                    }
                }
            }
            initialized = total
        }
        var view = dst.mutableView(as: T.self)
        view.copyElements(fromContentsOf: flat)
    }

    /// Zero every element of an NDArray in place (used to reset the
    /// persistent streaming-cache buffers to the initial state).
    private static func zeroFill(_ array: inout NDArray) {
        let count = array.shape.reduce(1, *)
        switch array.scalarType {
        case .float32:
            var v = array.mutableView(as: Float.self)
            v.withUnsafeMutablePointer { ptr, _, _ in for j in 0..<count { ptr[j] = 0 } }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            var v = array.mutableView(as: Float16.self)
            v.withUnsafeMutablePointer { ptr, _, _ in for j in 0..<count { ptr[j] = 0 } }
        #endif
        case .int32:
            var v = array.mutableView(as: Int32.self)
            v.withUnsafeMutablePointer { ptr, _, _ in for j in 0..<count { ptr[j] = 0 } }
        default:
            break
        }
    }

    /// Build a zero-filled NDArray matching a function input's declared
    /// descriptor (shape + scalar type) — used for the all-zero initial caches.
    private func makeZeroNDArray(_ fn: InferenceFunction, name: String) throws -> NDArray {
        guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else {
            throw RunnerError.unsupportedScalarType
        }
        // Descriptor shape has no dynamic dims for the caches; resolve to itself.
        let resolved = nd.resolvingDynamicDimensions(nd.shape)
        let count = nd.shape.reduce(1, *)   // [Int] — element count
        var array = NDArray(descriptor: resolved)
        switch resolved.scalarType {
        case .float32:
            var v = array.mutableView(as: Float.self)
            v.withUnsafeMutablePointer { ptr, _, _ in for j in 0..<count { ptr[j] = 0 } }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            var v = array.mutableView(as: Float16.self)
            v.withUnsafeMutablePointer { ptr, _, _ in for j in 0..<count { ptr[j] = 0 } }
        #endif
        case .int32:
            var v = array.mutableView(as: Int32.self)
            v.withUnsafeMutablePointer { ptr, _, _ in for j in 0..<count { ptr[j] = 0 } }
        default:
            throw RunnerError.unsupportedScalarType
        }
        return array
    }

    private func makeInt32NDArray(_ fn: InferenceFunction, name: String, data: [Int32], shape: [Int]) throws -> NDArray {
        guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else {
            throw RunnerError.unsupportedScalarType
        }
        let resolved = nd.resolvingDynamicDimensions(shape)
        var array = NDArray(descriptor: resolved)
        var view = array.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: data)
        return array
    }

    /// Read an int32 NDArray (length_out / cache_len_out) to a flat `[Int32]`.
    private func ndArrayToInt32(_ array: NDArray) throws -> [Int32] {
        guard array.scalarType == .int32 else { throw RunnerError.unsupportedScalarType }
        let total = array.shape.reduce(1, *)
        var result = [Int32](repeating: 0, count: total)
        array.view(as: Int32.self).withUnsafePointer { ptr, _, _ in
            for i in 0..<total { result[i] = ptr[i] }
        }
        return result
    }

    private func ndArrayToFloats(_ array: NDArray) throws -> [Float] {
        switch array.scalarType {
        case .float32:
            return Self.flatten(array, as: Float.self)
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            return Self.flatten(array, as: Float16.self)
        #endif
        default:
            throw RunnerError.unsupportedScalarType
        }
    }

    /// Stride-aware flatten to row-major `[Float]` — REQUIRED because Core AI /
    /// ANE outputs can be non-contiguous (interleaved) layouts. A naive flat read
    /// returns garbage for those, which manifested as all-blank RNN-T decode.
    /// Ported from coreai-models `flattenNDArray`.
    private static func flatten<T: BinaryFloatingPoint & BitwiseCopyable>(
        _ array: NDArray, as type: T.Type
    ) -> [Float] {
        let outerShape = array.shape
        let rank = outerShape.count
        let total = outerShape.reduce(1, *)
        var result = [Float](repeating: 0, count: total)
        array.view(as: type).withUnsafePointer { ptr, shape, strides in
            // Fast path: row-major contiguous.
            var expectedStride = 1
            var isContiguous = true
            for d in (0..<rank).reversed() {
                if strides[d] != expectedStride { isContiguous = false; break }
                expectedStride *= shape[d]
            }
            if isContiguous {
                for i in 0..<total { result[i] = Float(ptr[i]) }
                return
            }
            // Slow path: walk row-major indices through actual strides.
            var indices = [Int](repeating: 0, count: rank)
            for i in 0..<total {
                var offset = 0
                for d in 0..<rank { offset += indices[d] * strides[d] }
                result[i] = Float(ptr[offset])
                var dim = rank - 1
                while dim >= 0 {
                    indices[dim] += 1
                    if indices[dim] < shape[dim] { break }
                    indices[dim] = 0
                    dim -= 1
                }
            }
        }
        return result
    }
}
