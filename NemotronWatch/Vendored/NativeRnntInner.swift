import Accelerate
import Foundation

/// Joint encoder-projection weights (`joint.enc`) for the smart
/// speculative-blank decode path, which computes
/// `encoder_proj = encoded @ joint.enc.W^T + joint.enc.b` on CPU once per
/// chunk instead of using a multi-output encoder (which broke ANE compiles).
///
/// Loaded from `native_weights/` produced by `extract_decoder_joint_weights.py`:
///   weights.bin        : concatenated float16 little-endian tensors
///   weights_index.json : `{"vocab_size": V, "tensors": {name: {offset, shape}}}`
/// Only `joint.enc.weight` [hidden, encoderDim] and `joint.enc.bias` [hidden]
/// are read (upcast to fp32); the blob is memory-mapped, so the other
/// tensors in it are never made resident.
public final class NativeRnntInner: Sendable {
    private let jointEncW: [Float]  // [hidden, encoderDim], row-major
    private let jointEncB: [Float]  // [hidden]

    public let hidden: Int = 640
    public let encoderDim: Int = 1024

    /// Returns nil if the directory lacks the expected files or tensors.
    public init?(directory: URL) {
        let indexURL = directory.appendingPathComponent("weights_index.json")
        let blobURL = directory.appendingPathComponent("weights.bin")
        guard let indexData = try? Data(contentsOf: indexURL),
            let blobData = try? Data(contentsOf: blobURL, options: .alwaysMapped),
            let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
            let tensors = index["tensors"] as? [String: [String: Any]],
            index["vocab_size"] is Int
        else {
            return nil
        }

        // Slice one fp16 tensor out of the blob and upcast it to fp32.
        func loadF32(_ name: String, expectedCount: Int) -> [Float]? {
            guard let info = tensors[name],
                let offsetBytes = info["offset"] as? Int,
                let shape = info["shape"] as? [Int]
            else { return nil }
            let n = shape.reduce(1, *)
            guard n == expectedCount, offsetBytes >= 0, offsetBytes + n * 2 <= blobData.count
            else { return nil }
            var fp32 = [Float](repeating: 0, count: n)
            blobData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                var src = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: raw.baseAddress!.advanced(by: offsetBytes)),
                    height: 1, width: UInt(n), rowBytes: n * 2)
                fp32.withUnsafeMutableBufferPointer { buf in
                    var dst = vImage_Buffer(
                        data: buf.baseAddress!, height: 1, width: UInt(n), rowBytes: n * 4)
                    vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
                }
            }
            return fp32
        }

        guard let w = loadF32("joint.enc.weight", expectedCount: 640 * 1024),
            let b = loadF32("joint.enc.bias", expectedCount: 640)
        else { return nil }
        self.jointEncW = w
        self.jointEncB = b
    }

    /// outBuf[T_enc, hidden] = A[T_enc, encoderDim] @ joint.enc.W^T + joint.enc.b
    ///
    /// - `featureMajor == false`: `encoded` is row-major [T_enc, encoderDim]
    ///   with leading dimension `lda` (>= encoderDim).
    /// - `featureMajor == true`: `encoded` is row-major [encoderDim, T_enc]
    ///   (the encoder's native [1, D, T] layout) with leading dimension
    ///   `lda` (>= T_enc); BLAS transposes it, so no gather copy is needed.
    public func computeEncoderProjBatch(
        encoded: UnsafePointer<Float>,
        T_enc: Int,
        featureMajor: Bool,
        lda: Int,
        outBuf: UnsafeMutablePointer<Float>
    ) {
        jointEncW.withUnsafeBufferPointer { wPtr in
            cblas_sgemm(
                CblasRowMajor, featureMajor ? CblasTrans : CblasNoTrans, CblasTrans,
                Int32(T_enc), Int32(hidden), Int32(encoderDim),
                1.0,
                encoded, Int32(lda),
                wPtr.baseAddress, Int32(encoderDim),
                0.0,
                outBuf, Int32(hidden)
            )
        }
        jointEncB.withUnsafeBufferPointer { bPtr in
            for t in 0 ..< T_enc {
                let row = outBuf.advanced(by: t * hidden)
                vDSP_vadd(row, 1, bPtr.baseAddress!, 1, row, 1, vDSP_Length(hidden))
            }
        }
    }
}
