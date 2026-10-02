import Accelerate
@preconcurrency import CoreML
import Foundation

/// Per-chunk processing pipeline: preprocessor → split encoder
/// (pre_encode + 4 shards) → RNN-T greedy decode (smart speculative-blank
/// or the standard per-token loop). The encoder takes an extra `prompt_id`
/// int32 [1] input with the selected language hint; the decode loop records
/// the first language-tag token it emits via `recordDetectedLanguage(_:)`.
extension StreamingNemotronMultilingualAsrManager {

    // MARK: - Env-derived settings (read once per process)

    /// Smart speculative-blank decode is default-on whenever its assets are
    /// loaded; `NEMOTRON_ENABLE_SMART_SPECULATIVE=0|false|no` opts out.
    nonisolated internal static let smartSpecEnabled: Bool = {
        guard let v = ProcessInfo.processInfo.environment["NEMOTRON_ENABLE_SMART_SPECULATIVE"]?
            .lowercased()
        else { return true }
        return !(v == "0" || v == "false" || v == "no")
    }()

    /// NEMOTRON_VAD_RMS_THRESHOLD ([0,1] linear PCM). 0 (default) = VAD
    /// disabled. 0.003 very conservative, 0.005 conservative, 0.010 aggressive.
    nonisolated internal static let vadRmsThreshold: Float = {
        guard let s = ProcessInfo.processInfo.environment["NEMOTRON_VAD_RMS_THRESHOLD"],
            let v = Float(s), v > 0, v < 1.0
        else { return 0 }
        return v
    }()

    /// NEMOTRON_VAD_HANGOVER_CHUNKS: consecutive low-RMS chunks required
    /// before a skip (default 2 — the first low chunk after speech is always
    /// processed to preserve consonant tails).
    nonisolated internal static let vadHangoverChunks: Int = {
        guard let s = ProcessInfo.processInfo.environment["NEMOTRON_VAD_HANGOVER_CHUNKS"],
            let v = Int(s), v >= 1
        else { return 2 }
        return v
    }()

    // MARK: - Chunk pipeline

    /// Process a single full chunk (`config.chunkSamples`) of audio.
    internal func processChunk(_ samples: [Float]) async throws {
        guard let preprocessor = preprocessor,
            let tokenizer = tokenizer,
            var currentH = hState,
            var currentC = cState,
            let tokenInput = tokenInputBuf,
            let tokenLen = tokenLenBuf,
            let encStep = encoderStepBuf
        else {
            throw ASRError.notInitialized
        }
        var currentToken = lastToken

        self.chunkCount += 1

        // VAD-gated skip with hangover: skip only after N consecutive low-RMS
        // chunks; the first low chunk after speech is always processed.
        if Self.vadRmsThreshold > 0 {
            if Self.isAudioSilent(samples: samples, threshold: Self.vadRmsThreshold) {
                self.vadConsecutiveLowChunks &+= 1
                if self.vadConsecutiveLowChunks >= Self.vadHangoverChunks {
                    self.vadSkipCount &+= 1
                    return
                }
            } else {
                self.vadConsecutiveLowChunks = 0
            }
        }

        // 1. Preprocessor
        let prepStart = DispatchTime.now().uptimeNanoseconds
        let chunkMel = try await runPreprocessor(samples, preprocessor: preprocessor)
        self.prepNanos &+= DispatchTime.now().uptimeNanoseconds &- prepStart

        // 2. Split encoder
        let encStart = DispatchTime.now().uptimeNanoseconds
        let inputMel = try Self.prependMelCachePure(
            melCache: melCache,
            chunkMel: chunkMel,
            totalMelFrames: config.totalMelFrames,
            melFeatures: config.melFeatures,
            preEncodeCache: config.preEncodeCache
        )
        let melLen = try MLMultiArray(shape: [1], dataType: .int32)
        melLen[0] = NSNumber(value: config.totalMelFrames)
        let promptIdArray = try MLMultiArray(shape: [1], dataType: .int32)
        promptIdArray[0] = NSNumber(value: currentPromptId)

        let encoded = try await runShardedEncoder(
            inputMel: inputMel, length: melLen, promptId: promptIdArray)
        self.encNanos &+= DispatchTime.now().uptimeNanoseconds &- encStart
        melCache = try Self.extractMelCachePure(
            chunkMel: chunkMel,
            melFeatures: config.melFeatures,
            preEncodeCache: config.preEncodeCache
        )

        // 3. RNN-T greedy decode
        let decStart = DispatchTime.now().uptimeNanoseconds
        let numEncoderFrames = encoded.shape[2].intValue
        let tokensBefore = accumulatedTokenIds.count

        // Smart speculative-blank path: one decoder call + one batched joint
        // call per K-frame window; blank streaks skip K frames at a time.
        // encoder_proj is computed in Swift (cblas_sgemm with joint.enc
        // weights from native_weights/) so the encoder stays single-output.
        if Self.smartSpecEnabled,
            let jointBatched = jointNoEncProjBatched,
            let decoder = decoder,
            let native = nativeRnnt
        {
            let encProj = try encoderProjection(encoded, native: native)
            try await runSpeculativeBlankDecodeV2(
                encoded: encoded,
                encoderProj: encProj,
                numEncoderFrames: numEncoderFrames,
                decoder: decoder,
                jointBatched: jointBatched,
                tokenInput: tokenInput,
                tokenLen: tokenLen,
                encStep: encStep,
                currentH: &currentH,
                currentC: &currentC,
                currentToken: &currentToken,
                tokenizer: tokenizer
            )
        } else {
            let blankIdx = config.blankIdx
            for t in 0 ..< numEncoderFrames {
                Self.fillEncoderStep(into: encStep, from: encoded, timeIndex: t)
                // Greedy: up to 10 symbols per frame.
                for _ in 0 ..< 10 {
                    let step = try await decodeStep(
                        token: currentToken, h: currentH, c: currentC,
                        encStep: encStep, tokenInput: tokenInput, tokenLen: tokenLen)
                    if step.token == blankIdx { break }
                    emit(step.token, tokenizer: tokenizer)
                    currentToken = Int32(step.token)
                    currentH = step.h
                    currentC = step.c
                }
            }
        }

        self.decNanos &+= DispatchTime.now().uptimeNanoseconds &- decStart

        // Save final decoder state back to actor properties.
        self.lastToken = currentToken
        self.hState = currentH
        self.cState = currentC

        if accumulatedTokenIds.count != tokensBefore, let callback = partialCallback {
            callback(currentPartialText())
        }
    }

    /// Append an emitted (non-blank) token and surface the first language tag.
    private func emit(_ token: Int, tokenizer: NemotronMultilingualTokenizer) {
        accumulatedTokenIds.append(token)
        if config.langTagTokenIds.contains(token), detectedLanguage() == nil,
            let lang = tokenizer.decode(ids: [token]).detectedLanguage
        {
            recordDetectedLanguage(lang)
        }
    }

    private func predict(
        _ model: MLModel, _ features: [String: MLFeatureValue], options: MLPredictionOptions?
    ) async throws -> MLFeatureProvider {
        let input = try MLDictionaryFeatureProvider(dictionary: features)
        if let options = options {
            return try await model.prediction(from: input, options: options)
        }
        return try await model.prediction(from: input)
    }

    /// One greedy decoder + joint step at a single encoder frame. Priority:
    /// B2 (decoder_joint_argmax → token_id) > B1 (decoder_joint → logits) >
    /// bare decoder + joint. Returns the predicted token and the candidate
    /// LSTM state (commit it only for non-blank tokens).
    internal func decodeStep(
        token: Int32,
        h: MLMultiArray,
        c: MLMultiArray,
        encStep: MLMultiArray,
        tokenInput: MLMultiArray,
        tokenLen: MLMultiArray
    ) async throws -> (token: Int, h: MLMultiArray, c: MLMultiArray) {
        tokenInput.dataPointer.assumingMemoryBound(to: Int32.self).pointee = token
        let tokenIn = MLFeatureValue(multiArray: tokenInput)
        let tokenLength = MLFeatureValue(multiArray: tokenLen)
        let hIn = MLFeatureValue(multiArray: h)
        let cIn = MLFeatureValue(multiArray: c)

        if let dja = decoderJointArgmax {
            let out = try await predict(
                dja,
                [
                    "token": tokenIn, "token_length": tokenLength, "h_in": hIn, "c_in": cIn,
                    "encoder": MLFeatureValue(multiArray: encStep),
                ],
                options: decoderJointArgmaxPredictionOptions)
            guard let tokenId = out.featureValue(for: "token_id")?.multiArrayValue,
                let hOut = out.featureValue(for: "h_out")?.multiArrayValue,
                let cOut = out.featureValue(for: "c_out")?.multiArrayValue
            else {
                throw ASRError.processingFailed("Triple-fused decoder_joint_argmax failed")
            }
            return (Int(tokenId[0].int32Value), hOut, cOut)
        }

        if let dj = decoderJoint {
            let out = try await predict(
                dj,
                [
                    "token": tokenIn, "token_length": tokenLength, "h_in": hIn, "c_in": cIn,
                    "encoder": MLFeatureValue(multiArray: encStep),
                ],
                options: decoderJointPredictionOptions)
            guard let logits = out.featureValue(for: "logits")?.multiArrayValue,
                let hOut = out.featureValue(for: "h_out")?.multiArrayValue,
                let cOut = out.featureValue(for: "c_out")?.multiArrayValue
            else {
                throw ASRError.processingFailed("Fused decoder_joint failed")
            }
            return (Self.argmax(logits), hOut, cOut)
        }

        guard let decoder = decoder, let joint = joint else {
            throw ASRError.processingFailed(
                "No decode path: need decoder_joint_argmax (B2), decoder_joint (B1), or bare decoder+joint"
            )
        }
        let decOut = try await predict(
            decoder,
            ["token": tokenIn, "token_length": tokenLength, "h_in": hIn, "c_in": cIn],
            options: decoderPredictionOptions)
        guard let decoderOut = decOut.featureValue(for: "decoder_out")?.multiArrayValue,
            let hOut = decOut.featureValue(for: "h_out")?.multiArrayValue,
            let cOut = decOut.featureValue(for: "c_out")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Decoder failed")
        }
        let jointOut = try await predict(
            joint,
            [
                "encoder": MLFeatureValue(multiArray: encStep),
                "decoder": MLFeatureValue(multiArray: try Self.sliceDecoderOutput(decoderOut)),
            ],
            options: jointPredictionOptions)
        guard let logits = jointOut.featureValue(for: "logits")?.multiArrayValue else {
            throw ASRError.processingFailed("Joint failed")
        }
        return (Self.argmax(logits), hOut, cOut)
    }

    // MARK: - Encoder

    /// Run pre_encode (CPU) + the 4 encoder shards, threading per-shard caches.
    /// Returns `encoded` [1, encoderDim, T] as fp32.
    internal func runShardedEncoder(
        inputMel: MLMultiArray,
        length: MLMultiArray,
        promptId: MLMultiArray
    ) async throws -> MLMultiArray {
        guard let preEncode = encoderPreEncode,
            encoderShards.count == 4,
            shardCacheChannels.count == 4,
            shardCacheTimes.count == 4,
            shardCacheLens.count == 4
        else { throw ASRError.notInitialized }

        let preOutput = try await preEncode.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "mel": MLFeatureValue(multiArray: inputMel),
                "length": MLFeatureValue(multiArray: length),
            ]))
        guard var hidden = preOutput.featureValue(for: "hidden")?.multiArrayValue,
            var hiddenLength = preOutput.featureValue(for: "length_out")?.multiArrayValue
        else {
            throw ASRError.processingFailed("encoder_pre_encode failed to produce hidden")
        }

        let last = encoderShards.count - 1
        for idx in 0 ... last {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "hidden": MLFeatureValue(multiArray: hidden),
                "length": MLFeatureValue(multiArray: hiddenLength),
                "cache_channel": MLFeatureValue(multiArray: shardCacheChannels[idx]),
                "cache_time": MLFeatureValue(multiArray: shardCacheTimes[idx]),
                "cache_len": MLFeatureValue(multiArray: shardCacheLens[idx]),
                "prompt_id": MLFeatureValue(multiArray: promptId),
            ])
            let output = try await encoderShards[idx].prediction(from: input)
            if let ch = output.featureValue(for: "cache_channel_out")?.multiArrayValue {
                shardCacheChannels[idx] = ch
            }
            if let ti = output.featureValue(for: "cache_time_out")?.multiArrayValue {
                shardCacheTimes[idx] = ti
            }
            if let ln = output.featureValue(for: "cache_len_out")?.multiArrayValue {
                shardCacheLens[idx] = ln
            }
            guard let len = output.featureValue(for: "length_out")?.multiArrayValue else {
                throw ASRError.processingFailed("encoder_shard_\(idx) failed to produce length_out")
            }
            hiddenLength = len
            if idx == last {
                guard let encoded = output.featureValue(for: "encoded")?.multiArrayValue else {
                    throw ASRError.processingFailed(
                        "encoder_shard_\(idx) failed to produce encoded")
                }
                return try Self.widenToFloat32(encoded)
            }
            guard let nextHidden = output.featureValue(for: "hidden_out")?.multiArrayValue else {
                throw ASRError.processingFailed("encoder_shard_\(idx) failed to produce hidden_out")
            }
            hidden = nextHidden
        }
        throw ASRError.processingFailed("encoder shards produced no output")
    }

    /// fp16-I/O shard builds keep hidden + caches fp16 end to end. The decode
    /// paths read `encoded` as fp32, so widen just the final [1, D, T] output
    /// once per chunk into a dense fp32 array. fp32 outputs pass through.
    nonisolated internal static func widenToFloat32(_ array: MLMultiArray) throws -> MLMultiArray {
        guard array.dataType == .float16 else { return array }
        let shape = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }
        let out = try MLMultiArray(shape: array.shape, dataType: .float32)
        let src = array.dataPointer.bindMemory(to: UInt16.self, capacity: array.count)
        let dst = out.dataPointer.bindMemory(to: Float.self, capacity: out.count)
        let dstStrides = out.strides.map { $0.intValue }
        precondition(shape.count == 3, "encoded must be rank 3 [1, D, T]")
        for b in 0 ..< shape[0] {
            for d in 0 ..< shape[1] {
                let s = b * strides[0] + d * strides[1]
                let o = b * dstStrides[0] + d * dstStrides[1]
                for t in 0 ..< shape[2] {
                    dst[o + t * dstStrides[2]] = Float(Float16(bitPattern: src[s + t * strides[2]]))
                }
            }
        }
        return out
    }

    /// Fill the reusable `encProjReusable` [1, T, hidden] with
    /// `joint.enc(encoded)` and return it.
    internal func encoderProjection(_ encoded: MLMultiArray, native: NativeRnntInner) throws
        -> MLMultiArray
    {
        let tEnc = encoded.shape[2].intValue
        let buf: MLMultiArray
        if let existing = encProjReusable, existing.shape[1].intValue == tEnc {
            buf = existing
        } else {
            buf = try MLMultiArray(
                shape: [1, NSNumber(value: tEnc), NSNumber(value: native.hidden)],
                dataType: .float32)
            encProjReusable = buf
        }
        Self.computeEncoderProjSwift(encoded: encoded, native: native, outBuf: buf)
        return buf
    }

    /// encoder_proj = encoded^T @ joint.enc.W^T + b on CPU (one sgemm per chunk).
    /// `encoded` is [1, encoderDim, T]; `outBuf` is [1, T, hidden] fp32.
    /// fp32 with unit time-stride (the normal case) feeds BLAS directly as a
    /// transposed operand; anything else is gathered into [T, encoderDim].
    nonisolated internal static func computeEncoderProjSwift(
        encoded: MLMultiArray,
        native: NativeRnntInner,
        outBuf: MLMultiArray
    ) {
        let encoderDim = native.encoderDim
        let hidden = native.hidden
        let tEnc = encoded.shape[2].intValue
        precondition(encoded.shape[1].intValue == encoderDim, "encoded dim mismatch")
        precondition(outBuf.dataType == .float32, "encoder_proj outBuf must be fp32")
        precondition(
            outBuf.shape.count == 3 && outBuf.shape[1].intValue == tEnc
                && outBuf.shape[2].intValue == hidden,
            "encoder_proj outBuf shape mismatch")
        let dst = outBuf.dataPointer.bindMemory(to: Float.self, capacity: outBuf.count)
        let stride1 = encoded.strides[1].intValue
        let stride2 = encoded.strides[2].intValue

        if encoded.dataType == .float32, stride2 == 1, stride1 >= tEnc {
            let src = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
            native.computeEncoderProjBatch(
                encoded: src, T_enc: tEnc, featureMajor: true, lda: stride1, outBuf: dst)
            return
        }

        var rowMajor = [Float](repeating: 0, count: tEnc * encoderDim)
        if encoded.dataType == .float16 {
            let src = encoded.dataPointer.bindMemory(to: UInt16.self, capacity: encoded.count)
            for t in 0 ..< tEnc {
                for d in 0 ..< encoderDim {
                    rowMajor[t * encoderDim + d] = Float(
                        Float16(bitPattern: src[d * stride1 + t * stride2]))
                }
            }
        } else {
            let src = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
            for t in 0 ..< tEnc {
                for d in 0 ..< encoderDim {
                    rowMajor[t * encoderDim + d] = src[d * stride1 + t * stride2]
                }
            }
        }
        rowMajor.withUnsafeBufferPointer { src in
            native.computeEncoderProjBatch(
                encoded: src.baseAddress!, T_enc: tEnc, featureMajor: false, lda: encoderDim,
                outBuf: dst)
        }
    }

    // MARK: - Smart speculative-blank decode

    /// Smart speculative-blank decode (V2).
    ///
    /// Per window starting at frame t: run the decoder once for the current
    /// state, run the batched joint over encoder_proj[t ..< t+K], and scan
    /// for the first non-blank. All-blank → skip K frames. Otherwise emit it
    /// at frame t+k, then drain up to 9 more symbols at that frame with the
    /// standard per-token step, and continue at t+k+1.
    ///
    /// Parity: blank emissions don't advance RNN-T state, so the K blank
    /// predictions made with one dec_out are exactly what the per-frame loop
    /// would produce; the first non-blank used the correct state-at-t.
    internal func runSpeculativeBlankDecodeV2(
        encoded: MLMultiArray,
        encoderProj: MLMultiArray,
        numEncoderFrames: Int,
        decoder: MLModel,
        jointBatched: MLModel,
        tokenInput: MLMultiArray,
        tokenLen: MLMultiArray,
        encStep: MLMultiArray,
        currentH: inout MLMultiArray,
        currentC: inout MLMultiArray,
        currentToken: inout Int32,
        tokenizer: NemotronMultilingualTokenizer
    ) async throws {
        let K = jointNoEncProjBatchedK
        let blankIdx = config.blankIdx
        let projDim = encoderProj.shape[2].intValue

        let batchBuf: MLMultiArray
        if let existing = encProjBatchReusable, existing.shape[1].intValue == K,
            existing.shape[2].intValue == projDim
        {
            batchBuf = existing
        } else {
            batchBuf = try MLMultiArray(
                shape: [1, NSNumber(value: K), NSNumber(value: projDim)], dataType: .float32)
            encProjBatchReusable = batchBuf
        }

        // encoderProj is the fp32 [1, T, projDim] buffer from encoderProjection.
        let srcPtr = encoderProj.dataPointer.bindMemory(to: Float.self, capacity: encoderProj.count)
        let srcStride1 = encoderProj.strides[1].intValue
        let srcStride2 = encoderProj.strides[2].intValue
        let srcRowsContiguous = (srcStride2 == 1 && srcStride1 == projDim)
        let dstPtr = batchBuf.dataPointer.bindMemory(to: Float.self, capacity: batchBuf.count)

        var t = 0
        while t < numEncoderFrames {
            let kActual = min(K, numEncoderFrames - t)

            // 1. Decoder once with the current state → dec_out + candidate state.
            tokenInput.dataPointer.assumingMemoryBound(to: Int32.self).pointee = currentToken
            let decOutput = try await predict(
                decoder,
                [
                    "token": MLFeatureValue(multiArray: tokenInput),
                    "token_length": MLFeatureValue(multiArray: tokenLen),
                    "h_in": MLFeatureValue(multiArray: currentH),
                    "c_in": MLFeatureValue(multiArray: currentC),
                ],
                options: decoderPredictionOptions)
            guard let decOutRaw = decOutput.featureValue(for: "decoder_out")?.multiArrayValue,
                let candidateH = decOutput.featureValue(for: "h_out")?.multiArrayValue,
                let candidateC = decOutput.featureValue(for: "c_out")?.multiArrayValue
            else {
                throw ASRError.processingFailed("Speculative decoder failed")
            }
            let decOut = try Self.sliceDecoderOutput(decOutRaw)

            // 2. encoder_proj[t ..< t+kActual] → window buffer (zero-pad the tail).
            if srcRowsContiguous {
                dstPtr.update(from: srcPtr + t * projDim, count: kActual * projDim)
            } else {
                for k in 0 ..< kActual {
                    for d in 0 ..< projDim {
                        dstPtr[k * projDim + d] = srcPtr[(t + k) * srcStride1 + d * srcStride2]
                    }
                }
            }
            if kActual < K {
                (dstPtr + kActual * projDim).update(repeating: 0, count: (K - kActual) * projDim)
            }

            // 3. Batched joint over the window → logits [1, K, 1, V].
            let jointOutput = try await predict(
                jointBatched,
                [
                    "encoder_proj": MLFeatureValue(multiArray: batchBuf),
                    "decoder": MLFeatureValue(multiArray: decOut),
                ],
                options: jointNoEncProjBatchedPredictionOptions)
            guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw ASRError.processingFailed("Speculative joint_noencproj_batched failed")
            }
            let frameStride = logits.strides[1].intValue
            let vocabStride = logits.strides[3].intValue
            let vocabSize = logits.shape[3].intValue

            // 4. First non-blank within the window.
            var firstNonBlankAt = -1
            var emittedToken = blankIdx
            for kk in 0 ..< kActual {
                let best = Self.argmax(
                    logits, offset: kk * frameStride, count: vocabSize, stride: vocabStride)
                if best != blankIdx {
                    firstNonBlankAt = kk
                    emittedToken = best
                    break
                }
            }

            self.specWindowsTotal &+= 1
            if firstNonBlankAt == -1 {
                self.specWindowsAllBlank &+= 1
                t += kActual
                continue
            }
            self.specWindowsHitNonBlank &+= 1

            emit(emittedToken, tokenizer: tokenizer)
            currentToken = Int32(emittedToken)
            currentH = candidateH
            currentC = candidateC

            // Multi-emission drain at this frame (max 10 symbols per frame in
            // total) via the standard single-frame step (B2 > B1 > bare).
            Self.fillEncoderStep(into: encStep, from: encoded, timeIndex: t + firstNonBlankAt)
            for _ in 0 ..< 9 {
                let step = try await decodeStep(
                    token: currentToken, h: currentH, c: currentC,
                    encStep: encStep, tokenInput: tokenInput, tokenLen: tokenLen)
                if step.token == blankIdx { break }
                emit(step.token, tokenizer: tokenizer)
                currentToken = Int32(step.token)
                currentH = step.h
                currentC = step.c
            }

            t += firstNonBlankAt + 1
        }
    }

    // MARK: - Preprocessor / mel helpers

    /// Run the preprocessor on one chunk, reusing the per-stream audio input
    /// buffers when the sample count matches.
    private func runPreprocessor(_ samples: [Float], preprocessor: MLModel) async throws
        -> MLMultiArray
    {
        let audio: MLMultiArray
        let audioLen: MLMultiArray
        if let buf = audioInputBuf, buf.shape[1].intValue == samples.count,
            let lenBuf = audioLenBuf
        {
            audio = buf
            audioLen = lenBuf
        } else {
            audio = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
            audioLen = try MLMultiArray(shape: [1], dataType: .int32)
        }
        audio.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
            .update(from: samples, count: samples.count)
        audioLen[0] = NSNumber(value: samples.count)

        let output = try await preprocessor.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "audio": MLFeatureValue(multiArray: audio),
                "audio_length": MLFeatureValue(multiArray: audioLen),
            ]))
        guard let mel = output.featureValue(for: "mel")?.multiArrayValue else {
            throw ASRError.processingFailed("Preprocessor failed to produce mel output")
        }
        return mel
    }

    /// Energy-based silence detector (vDSP_rmsqv): true iff RMS < threshold.
    nonisolated internal static func isAudioSilent(samples: [Float], threshold: Float) -> Bool {
        guard !samples.isEmpty else { return true }
        var rms: Float = 0
        samples.withUnsafeBufferPointer { ptr in
            vDSP_rmsqv(ptr.baseAddress!, 1, &rms, vDSP_Length(samples.count))
        }
        return rms < threshold
    }

    /// Last `preEncodeCache` mel frames of `chunkMel` → [1, melFeatures, n] fp32.
    nonisolated internal static func extractMelCachePure(
        chunkMel: MLMultiArray,
        melFeatures: Int,
        preEncodeCache: Int
    ) throws -> MLMultiArray {
        let chunkFrames = chunkMel.shape[2].intValue
        let cacheFrames = min(preEncodeCache, chunkFrames)
        let cache = try MLMultiArray(
            shape: [1, NSNumber(value: melFeatures), NSNumber(value: cacheFrames)],
            dataType: .float32
        )
        let srcPtr = chunkMel.dataPointer.bindMemory(to: Float.self, capacity: chunkMel.count)
        let dstPtr = cache.dataPointer.bindMemory(to: Float.self, capacity: cache.count)
        let srcStride1 = chunkMel.strides[1].intValue
        let srcStride2 = chunkMel.strides[2].intValue
        let dstStride1 = cache.strides[1].intValue
        let dstStride2 = cache.strides[2].intValue
        let startT = chunkFrames - cacheFrames
        for mel in 0 ..< melFeatures {
            for t in 0 ..< cacheFrames {
                dstPtr[mel * dstStride1 + t * dstStride2] =
                    srcPtr[mel * srcStride1 + (startT + t) * srcStride2]
            }
        }
        return cache
    }

    /// [melCache | chunkMel] → [1, melFeatures, totalMelFrames] fp32 (zeros
    /// where no cache exists yet, i.e. the first chunk).
    nonisolated internal static func prependMelCachePure(
        melCache: MLMultiArray?,
        chunkMel: MLMultiArray,
        totalMelFrames: Int,
        melFeatures: Int,
        preEncodeCache: Int
    ) throws -> MLMultiArray {
        let chunkFrames = chunkMel.shape[2].intValue
        let result = try MLMultiArray(
            shape: [1, NSNumber(value: melFeatures), NSNumber(value: totalMelFrames)],
            dataType: .float32
        )
        result.reset(to: 0)
        let resultPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)
        let chunkPtr = chunkMel.dataPointer.bindMemory(to: Float.self, capacity: chunkMel.count)
        let resultStride1 = result.strides[1].intValue
        let resultStride2 = result.strides[2].intValue
        let chunkStride1 = chunkMel.strides[1].intValue
        let chunkStride2 = chunkMel.strides[2].intValue

        if let melCache = melCache {
            let cachePtr = melCache.dataPointer.bindMemory(to: Float.self, capacity: melCache.count)
            let cacheFrames = melCache.shape[2].intValue
            let cacheStride1 = melCache.strides[1].intValue
            let cacheStride2 = melCache.strides[2].intValue
            for mel in 0 ..< melFeatures {
                for t in 0 ..< cacheFrames {
                    resultPtr[mel * resultStride1 + t * resultStride2] =
                        cachePtr[mel * cacheStride1 + t * cacheStride2]
                }
            }
        }
        let copyFrames = min(chunkFrames, totalMelFrames - preEncodeCache)
        for mel in 0 ..< melFeatures {
            for t in 0 ..< copyFrames {
                resultPtr[mel * resultStride1 + (preEncodeCache + t) * resultStride2] =
                    chunkPtr[mel * chunkStride1 + t * chunkStride2]
            }
        }
        return result
    }

    // MARK: - Tensor helpers

    /// Copy one time step of fp32 `encoded` [1, D, T] into `dest` [1, D, 1].
    nonisolated internal static func fillEncoderStep(
        into dest: MLMultiArray, from encoded: MLMultiArray, timeIndex: Int
    ) {
        let dim = encoded.shape[1].intValue
        let src = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
        let dst = dest.dataPointer.bindMemory(to: Float.self, capacity: dest.count)
        // Strided column copy: `dim` rows of 1 element, source row stride = strides[1].
        vDSP_mmov(
            src + timeIndex * encoded.strides[2].intValue, dst, 1, vDSP_Length(dim),
            vDSP_Length(encoded.strides[1].intValue), 1)
    }

    /// decoder_out [1, hidden, U] → [1, hidden, 1] (frame 0). Already
    /// single-frame outputs pass straight through (no copy).
    nonisolated internal static func sliceDecoderOutput(_ decoderOut: MLMultiArray) throws
        -> MLMultiArray
    {
        if decoderOut.shape.count == 3, decoderOut.shape[2].intValue == 1 {
            return decoderOut
        }
        let hidden = decoderOut.shape[1].intValue
        let result = try MLMultiArray(shape: [1, NSNumber(value: hidden), 1], dataType: .float32)
        let srcPtr = decoderOut.dataPointer.bindMemory(to: Float.self, capacity: decoderOut.count)
        let dstPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)
        let stride1 = decoderOut.strides[1].intValue
        for c in 0 ..< hidden {
            dstPtr[c] = srcPtr[c * stride1]
        }
        return result
    }

    /// Argmax over `count` elements of `array` starting at element `offset`
    /// with element `stride` (defaults: the whole array). fp32 unit-stride
    /// uses vDSP_maxvi; fp16 / strided fall back to a scalar scan. Ties
    /// resolve to the lowest index in both paths.
    nonisolated internal static func argmax(
        _ array: MLMultiArray, offset: Int = 0, count: Int? = nil, stride: Int = 1
    ) -> Int {
        let n = count ?? array.count
        guard n > 0 else { return 0 }
        if array.dataType == .float16 {
            let p = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count) + offset
            var bestIdx = 0
            var bestVal = p[0]
            for i in 1 ..< n where p[i * stride] > bestVal {
                bestVal = p[i * stride]
                bestIdx = i
            }
            return bestIdx
        }
        let p = array.dataPointer.bindMemory(to: Float.self, capacity: array.count) + offset
        if stride == 1 {
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(p, 1, &maxVal, &maxIdx, vDSP_Length(n))
            return Int(maxIdx)
        }
        var bestIdx = 0
        var bestVal = p[0]
        for i in 1 ..< n where p[i * stride] > bestVal {
            bestVal = p[i * stride]
            bestIdx = i
        }
        return bestIdx
    }
}
