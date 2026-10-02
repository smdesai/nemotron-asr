import Accelerate
import CoreAI
import Foundation

/// RNN-T loop parameters parsed from the tier's `metadata.json`.
struct CoreAIMetadata {
    let blankIdx: Int
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
            totalMelFrames: obj["total_mel_frames"] as? Int ?? 233,
            chunkMelFrames: obj["chunk_mel_frames"] as? Int ?? 224,
            preEncodeCache: obj["pre_encode_cache"] as? Int ?? 9,
            langTagTokenIds: obj["lang_tag_token_ids"] as? [Int] ?? [],
            promptDictionary: obj["prompt_dictionary"] as? [String: Int] ?? [:]
        )
    }

    /// Prompt id for a language code from the model's prompt_dictionary
    /// (auto/101 when unset or unknown).
    func promptId(for code: String?) -> Int {
        if let code, let id = promptDictionary[code] { return id }
        return promptDictionary["auto"] ?? 101
    }
}

/// Core AI streaming RNN-T transcriber: mel (Swift/vDSP) → 4 int8 encoder
/// shards (cache-aware, threaded across chunks) → greedy decode through the
/// fused `decoder_joint.aimodel`.
///
/// RNN-T greedy semantics (matching the CoreML per-token path):
///   - per encoder frame, emit up to 10 symbols
///   - blank (id == blankIdx) advances to the next frame WITHOUT committing state
///   - non-blank emits the token, threads (token, h, c) forward, continues at frame
///
/// Feed audio with `stream(samples:)` (any block size) and flush the tail with
/// `finishStreaming()`; call `reset()` before each utterance.
@MainActor
final class CoreAIStreamingTranscriber {

    private let runner: CoreAIEncoderRunner
    private let mel: MelFrontend
    private let tokenizer: NemotronMultilingualTokenizer
    private let blankIdx: Int
    /// Language prompt id fed to the encoder; changeable between utterances.
    var promptId: Int32
    private let totalMelFrames: Int
    private let chunkMelFrames: Int
    private let preEncodeCache: Int

    // LSTM state (decoder), threaded across the decode loop. [2*640] each.
    private var h: [Float]
    private var c: [Float]
    private var lastToken: Int32
    private var accumulatedTokenIds: [Int] = []

    // Mel left-context carried across chunks: the last `preEncodeCache` mel
    // frames of the previous chunk's standalone mel, prepended to the next
    // chunk so the encoder window is [preEncodeCache prev][chunkMelFrames new]
    // = totalMelFrames. `[nMels * preEncodeCache]` row-major; empty before the
    // first chunk (→ zero left-context).
    private var melCache: [Float] = []

    // Incoming blocks accumulate here until a full encoder chunk
    // (`chunkSamples`) of NEW audio is available.
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
        runner.resetStreamingCaches()
    }

    /// Samples of NEW audio consumed per streaming chunk
    /// (chunkMelFrames × hop, e.g. 35840 ≈ 2.24 s at 16 kHz).
    var chunkSamples: Int { chunkMelFrames * MelFrontend.hop }

    /// Buffer an incoming 16 kHz mono block and run the encoder + decode for
    /// every full chunk of accumulated new audio. Returns true if at least one
    /// chunk was processed (i.e. `partialText` may have grown). Call
    /// `finishStreaming()` at the end to flush the tail.
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

    /// Flush any buffered partial chunk (zero-padded to a full chunk) and
    /// return the final transcript.
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

    /// Run one chunk: mel → encoder → RNN-T greedy decode, appending to the
    /// transcript.
    ///
    /// `samples` is one chunk of NEW audio (`chunkMelFrames` worth). The encoder
    /// window is [preEncodeCache prev mel frames][chunkMelFrames new] =
    /// totalMelFrames; the trailing `preEncodeCache` frames of this chunk's
    /// standalone mel become the next chunk's left-context.
    private func processChunk(_ samples: [Float]) async throws {
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
        // Built once per chunk; every decode step below reuses it.
        let encoderInput = try runner.makeDecoderEncoderInput(encoded, tEnc: tEnc)

        // Batched RNN-T greedy decode: one fused decoder+joint dispatch per
        // decoder state covers ALL tEnc frames; scan the returned logits[t].
        // Consecutive blank frames cost no extra dispatch — only committing a
        // token (state change) re-runs the model.
        var t = 0
        var symbolsAtT = 0
        var batchedLogits: [Float]? = nil  // logits[t*vocab + v] for the current state
        var stepHOut = h
        var stepCOut = c
        while t < tEnc {
            if batchedLogits == nil {
                let step = try await runner.decodeJointStep(
                    token: lastToken, h: h, c: c, encoder: encoderInput)
                batchedLogits = step.logits
                stepHOut = step.hOut
                stepCOut = step.cOut
            }
            let bestIdx = Self.argmax(batchedLogits!, row: t, rows: tEnc)

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

    /// Vectorized argmax (vDSP) over row `row` of a row-major [rows × vocab]
    /// logits buffer, read in place.
    private static func argmax(_ logits: [Float], row: Int, rows: Int) -> Int {
        let vocab = logits.count / rows
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        logits.withUnsafeBufferPointer { buf in
            vDSP_maxvi(buf.baseAddress! + row * vocab, 1, &maxValue, &maxIndex, vDSP_Length(vocab))
        }
        return Int(maxIndex)
    }

    /// Assemble the `[nMels * totalMelFrames]` encoder window from the carried
    /// mel cache (left-context) + this chunk's new frames. Mel buffers are
    /// `[nMels * T]` row-major (`m[mel*T+t]`).
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
    /// mel as the next chunk's left-context. Returns `[nMels * preEncodeCache]`
    /// row-major.
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

    /// First language tag the decoder emitted this utterance (e.g. "en-US").
    var detectedLanguage: String? {
        tokenizer.decode(ids: accumulatedTokenIds).detectedLanguage
    }
}
