import Accelerate
import CoreAI
import Foundation

// The tokenizer comes from the FluidAudio package on iOS; the watch target
// compiles the vendored copy (NemotronWatch/Vendored) directly, so the type is
// in-module there and no import is needed.
#if canImport(FluidAudio)
import FluidAudio
#endif

/// RNN-T loop parameters parsed from the bundled `coreai/metadata.json`.
struct CoreAIMetadata {
    let blankIdx: Int
    let vocabSize: Int
    let totalMelFrames: Int
    let chunkMelFrames: Int
    let preEncodeCache: Int
    let langTagTokenIds: [Int]
    let promptDictionary: [String: Int]

    static func load(from dir: URL) throws -> CoreAIMetadata {
        let url = dir.appendingPathComponent("metadata.json")
        let data = try Data(contentsOf: url)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        return CoreAIMetadata(
            blankIdx: obj["blank_idx"] as? Int ?? 13087,
            vocabSize: obj["vocab_size"] as? Int ?? 13087,
            totalMelFrames: obj["total_mel_frames"] as? Int ?? 233,
            chunkMelFrames: obj["chunk_mel_frames"] as? Int ?? 224,
            preEncodeCache: obj["pre_encode_cache"] as? Int ?? 9,
            langTagTokenIds: obj["lang_tag_token_ids"] as? [Int] ?? [],
            promptDictionary: obj["prompt_dictionary"] as? [String: Int] ?? [:]
        )
    }
}

/// Full Core AI streaming RNN-T transcriber: mel → encoder → greedy decode.
///
/// Wires the three Core AI components into a streaming pipeline that mirrors
/// FluidAudio's PROVEN per-token CoreML path (`StreamingNemotronMultilingualAsrManager
/// +Pipeline.swift`, the `decoder + joint` fallback branch) — NOT the native-Swift
/// `NativeRnntInner` path, which is documented KNOWN-BROKEN (single-step parity OK,
/// multi-step WER 100%).
///
/// RNN-T greedy semantics (matched exactly):
///   - per encoder frame, emit up to 10 symbols
///   - blank (id == blankIdx) advances to the next frame WITHOUT committing state
///   - non-blank emits the token, threads (token, h, c) forward, continues at frame
///
/// Decode is via `decoder.aimodel` + `joint.aimodel` (`.aimodel` model calls per token).
/// Tokenizer is FluidAudio's `NemotronMultilingualTokenizer`.
///
/// Requires iOS 27 (`CoreAI`). Streaming encoder caches are not yet threaded across
/// chunks (the encoder runs per-chunk with zero-init caches); for short clips this
/// matches a single-pass decode. Cross-chunk cache threading is the next refinement.
@available(iOS 27.0, watchOS 27.0, *)
@MainActor
final class CoreAIStreamingTranscriber {

    private let runner: CoreAIEncoderRunner
    private let mel: MelFrontend
    private let tokenizer: NemotronMultilingualTokenizer
    private let blankIdx: Int
    private let promptId: Int32
    private let totalMelFrames: Int
    private let chunkMelFrames: Int
    private let preEncodeCache: Int

    // LSTM state (decoder), threaded across the decode loop. [2*640] each.
    private var h: [Float]
    private var c: [Float]
    private var lastToken: Int32
    private var accumulatedTokenIds: [Int] = []

    // Decoder-output memo. The decoder's output depends only on
    // (lastToken, h, c), which change only when a non-blank token is
    // committed — so consecutive blank frames can reuse the cached step
    // instead of re-running decoder.aimodel once per encoder frame.
    // Invalidated on commit and on reset(); valid across chunk boundaries
    // because the decoder state itself persists across chunks.
    private var cachedDecoderStep: (decStep: [Float], hOut: [Float], cOut: [Float])?

    // Mel left-context carried across chunks: the last `preEncodeCache` mel
    // frames of the previous chunk's standalone mel, prepended to the next
    // chunk so the encoder window is [preEncodeCache prev][chunkMelFrames new]
    // = totalMelFrames. Stored as `[nMels * preEncodeCache]` row-major; empty
    // before the first chunk (→ zero left-context).
    private var melCache: [Float] = []

    // Live-microphone buffer: incoming capture blocks accumulate here until a
    // full encoder chunk (`chunkSamples`) of NEW audio is available.
    private var pendingSamples: [Float] = []

    static let decoderHidden = 640
    static let decoderLayers = 2

    init(
        runner: CoreAIEncoderRunner,
        mel: MelFrontend,
        tokenizer: NemotronMultilingualTokenizer,
        blankIdx: Int,
        promptId: Int32,
        totalMelFrames: Int,
        chunkMelFrames: Int,
        preEncodeCache: Int
    ) {
        self.runner = runner
        self.mel = mel
        self.tokenizer = tokenizer
        self.blankIdx = blankIdx
        self.promptId = promptId
        self.totalMelFrames = totalMelFrames
        self.chunkMelFrames = chunkMelFrames
        self.preEncodeCache = preEncodeCache
        self.h = [Float](repeating: 0, count: Self.decoderLayers * Self.decoderHidden)
        self.c = [Float](repeating: 0, count: Self.decoderLayers * Self.decoderHidden)
        self.lastToken = Int32(blankIdx)
    }

    func reset() {
        h = [Float](repeating: 0, count: Self.decoderLayers * Self.decoderHidden)
        c = [Float](repeating: 0, count: Self.decoderLayers * Self.decoderHidden)
        lastToken = Int32(blankIdx)
        accumulatedTokenIds.removeAll(keepingCapacity: true)
        melCache.removeAll(keepingCapacity: true)
        pendingSamples.removeAll(keepingCapacity: true)
        cachedDecoderStep = nil
        runner.resetStreamingCaches()
    }

    /// Samples of NEW audio consumed per streaming chunk
    /// (chunkMelFrames × hop = 35840 ≈ 2.24 s at 16 kHz).
    var chunkSamples: Int { chunkMelFrames * MelFrontend.hop }

    /// Live-microphone entry point: buffer an incoming capture block and run
    /// the encoder + decode for every full chunk of accumulated new audio.
    /// Returns true if at least one chunk was processed (i.e. `partialText`
    /// may have grown). Call `finishStreaming()` at stop to flush the tail.
    func stream(samples: [Float]) async throws -> Bool {
        pendingSamples.append(contentsOf: samples)
        var processed = false
        while pendingSamples.count >= chunkSamples {
            let chunk = Array(pendingSamples.prefix(chunkSamples))
            pendingSamples.removeFirst(chunkSamples)
            try await processChunk(chunk)
            processed = true
        }
        return processed
    }

    /// Flush any buffered partial chunk (zero-padded to a full encoder
    /// window, matching `transcribe`'s tail handling) and return the final
    /// transcript.
    func finishStreaming() async throws -> String {
        if !pendingSamples.isEmpty {
            var chunk = pendingSamples
            pendingSamples.removeAll(keepingCapacity: true)
            if chunk.count < chunkSamples {
                chunk.append(contentsOf: [Float](repeating: 0, count: chunkSamples - chunk.count))
            }
            try await processChunk(chunk)
        }
        return partialText
    }

    /// Transcribe a full audio buffer (16 kHz mono) by chunking into the encoder's
    /// fixed mel window and running encoder + greedy decode per chunk. Returns the
    /// decoded transcript.
    ///
    /// `onPartial`, if provided, is called after each chunk with the running
    /// transcript so the UI can stream text as it is generated. `onProgress`,
    /// if provided, is called after every chunk with the fraction of chunks
    /// processed (0...1), including chunks that emitted no tokens.
    func transcribe(
        samples: [Float],
        onPartial: ((String) -> Void)? = nil,
        onProgress: ((Double) -> Void)? = nil
    ) async throws -> String {
        reset()
        let hop = MelFrontend.hop
        // Cache-aware streaming: advance by `chunkMelFrames` worth of NEW audio
        // each step (224 mel frames × 160 hop = 35840 samples ≈ 2.24s). Each
        // encoder window is built inside processChunk as
        // [preEncodeCache prev mel frames][chunkMelFrames new] = totalMelFrames,
        // and the FastConformer caches carry the rest of the left-context.
        // (The old code advanced disjoint by (totalMelFrames-1)*hop with zero
        // caches — that cold-started every chunk and garbled boundary words.)
        let chunkSamples = chunkMelFrames * hop
        let nChunks = (samples.count + chunkSamples - 1) / max(1, chunkSamples)
        print(
            "[CoreAI-T] transcribe: \(samples.count) samples (\(String(format: "%.1f", Double(samples.count)/16000))s), "
                + "chunkSamples=\(chunkSamples), \(nChunks) chunks, blankIdx=\(blankIdx), promptId=\(promptId)"
        )
        var pos = 0
        var chunkIdx = 0
        while pos < samples.count {
            let end = min(pos + chunkSamples, samples.count)
            var chunk = Array(samples[pos ..< end])
            if chunk.count < chunkSamples {
                chunk.append(contentsOf: [Float](repeating: 0, count: chunkSamples - chunk.count))
            }
            let t0 = Date()
            let before = accumulatedTokenIds.count
            try await processChunk(chunk)
            let dt = Date().timeIntervalSince(t0)
            print(
                "[CoreAI-T] chunk \(chunkIdx)/\(nChunks): +\(accumulatedTokenIds.count - before) tokens "
                    + "(total \(accumulatedTokenIds.count)) in \(String(format: "%.2f", dt))s")
            // Stream the running transcript after each chunk (only when it grew).
            if let onPartial, accumulatedTokenIds.count != before {
                onPartial(tokenizer.decode(ids: accumulatedTokenIds).text)
            }
            pos += chunkSamples
            chunkIdx += 1
            onProgress?(Double(chunkIdx) / Double(max(1, nChunks)))
            if onPartial != nil || onProgress != nil {
                // Yield so the MainActor can paint the partial/progress before
                // the next chunk's compute monopolises the thread.
                await Task.yield()
            }
        }
        let text = tokenizer.decode(ids: accumulatedTokenIds).text
        print("[CoreAI-T] DONE: \(accumulatedTokenIds.count) tokens → \"\(text.prefix(80))\"")
        return text
    }

    /// Run one chunk: mel → encoder → RNN-T greedy decode, appending to the transcript.
    ///
    /// `samples` is one chunk of NEW audio (`chunkMelFrames` worth). The encoder
    /// window is assembled as [preEncodeCache prev mel frames][chunkMelFrames
    /// new] = totalMelFrames; the trailing `preEncodeCache` frames of this
    /// chunk's standalone mel become the next chunk's left-context. Mirrors
    /// FluidAudio's `prependMelCache` / `extractMelCache`.
    func processChunk(_ samples: [Float]) async throws {
        let (m, frames) = mel.melSpectrogram(samples)
        guard frames > 0 else { return }
        let nMels = MelFrontend.nMels
        let window = assembleWindow(chunkMel: m, chunkFrames: frames, nMels: nMels)
        melCache = extractMelCache(chunkMel: m, chunkFrames: frames, nMels: nMels)
        let (encoded, shape) = try await runner.encode(
            mel: window, frames: totalMelFrames, promptId: promptId)
        // encoded is row-major [1, 1024, T_enc]; T_enc = shape[2].
        guard shape.count == 3 else {
            print("[CoreAI-T] unexpected encoder shape \(shape)")
            return
        }
        let tEnc = shape[2]

        // Batched RNN-T greedy decode: run decoder + joint ONCE per decoder state
        // over ALL tEnc frames at once (`encoded` is already row-major [1, D, T]),
        // then scan the returned logits[t]. Consecutive blank frames cost NO extra
        // dispatch — only committing a token (state change) re-runs decoder+joint.
        // ~2x fewer joint dispatches than the old per-frame path, with the async/
        // ANE dispatch overhead amortized over all T frames.
        var t = 0
        var symbolsAtT = 0
        var batchedLogits: [Float]? = nil  // logits[t*vocab + v] for the current state
        var stepHOut = h
        var stepCOut = c
        while t < tEnc {
            if batchedLogits == nil {
                if let fused = try await runner.decodeJointStep(
                    token: lastToken, h: h, c: c, encoder: encoded, tEnc: tEnc)
                {
                    // fused decoder+joint: one dispatch per decoder state
                    batchedLogits = fused.logits
                    stepHOut = fused.hOut
                    stepCOut = fused.cOut
                } else {
                    // fallback: separate decoder + batched joint (two dispatches)
                    let out = try await runner.decoderStep(token: lastToken, h: h, c: c)
                    let decStep = Array(out.decoderOut.prefix(Self.decoderHidden))
                    stepHOut = out.hOut
                    stepCOut = out.cOut
                    batchedLogits = try await runner.jointBatched(
                        encoder: encoded, tEnc: tEnc, decoderStep: decStep)
                }
            }
            let logits = batchedLogits!
            let vocab = logits.count / tEnc
            let frameLogits = Array(logits[(t * vocab) ..< ((t + 1) * vocab)])
            let bestIdx = Self.argmax(frameLogits)

            if bestIdx == blankIdx || symbolsAtT >= 10 {
                t += 1
                symbolsAtT = 0
                continue  // blank / per-frame symbol cap → next frame; reuse batchedLogits
            }
            // non-blank → emit, commit state; state advanced so re-batch at this frame
            accumulatedTokenIds.append(bestIdx)
            lastToken = Int32(bestIdx)
            h = stepHOut
            c = stepCOut
            batchedLogits = nil
            symbolsAtT += 1
        }
    }

    /// Vectorized argmax over the joint logits (vDSP) — runs once per
    /// inner-loop step over the full ~13k vocab.
    private static func argmax(_ values: [Float]) -> Int {
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(values, 1, &maxValue, &maxIndex, vDSP_Length(values.count))
        return Int(maxIndex)
    }

    /// Assemble the `[nMels * totalMelFrames]` encoder window from the carried
    /// mel cache (left-context) + this chunk's new frames. Mirrors FluidAudio's
    /// `prependMelCache`. Mel buffers are `[nMels * T]` row-major (`m[mel*T+t]`).
    private func assembleWindow(chunkMel m: [Float], chunkFrames: Int, nMels: Int) -> [Float] {
        var window = [Float](repeating: 0, count: nMels * totalMelFrames)
        // Left-context: previous chunk's last `preEncodeCache` frames at t=0..
        let cacheFrames = melCache.isEmpty ? 0 : preEncodeCache
        if cacheFrames > 0 {
            for mel in 0 ..< nMels {
                for t in 0 ..< cacheFrames {
                    window[mel * totalMelFrames + t] = melCache[mel * cacheFrames + t]
                }
            }
        }
        // New frames after the cache position (cap so cache + new == window).
        let copyFrames = min(chunkFrames, totalMelFrames - preEncodeCache)
        for mel in 0 ..< nMels {
            for t in 0 ..< copyFrames {
                window[mel * totalMelFrames + (preEncodeCache + t)] = m[mel * chunkFrames + t]
            }
        }
        return window
    }

    /// Extract the trailing `preEncodeCache` frames of this chunk's standalone
    /// mel as the next chunk's left-context. Mirrors FluidAudio's
    /// `extractMelCache`. Returns `[nMels * preEncodeCache]` row-major.
    private func extractMelCache(chunkMel m: [Float], chunkFrames: Int, nMels: Int) -> [Float] {
        let cacheFrames = min(preEncodeCache, chunkFrames)
        var cache = [Float](repeating: 0, count: nMels * cacheFrames)
        let startT = chunkFrames - cacheFrames
        for mel in 0 ..< nMels {
            for t in 0 ..< cacheFrames {
                cache[mel * cacheFrames + t] = m[mel * chunkFrames + (startT + t)]
            }
        }
        return cache
    }

    /// Current running transcript (for streaming partial display).
    var partialText: String {
        tokenizer.decode(ids: accumulatedTokenIds).text
    }
}
