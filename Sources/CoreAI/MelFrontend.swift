import Accelerate
import Foundation

/// Host-side mel-spectrogram front-end for the Core AI ASR path.
///
/// Core AI's compiler cannot lower the NeMo preprocessor's STFT (`aten._fft_r2c`),
/// so the mel features are computed here in Swift/Accelerate (vDSP) instead of in
/// an `.aimodel`. This is a verbatim port of the implementation validated against
/// the NeMo reference at **130–136 dB PSNR / cosine 1.0** across all four latency
/// tiers (see `coreml/conversion_scripts/MelParity.swift` and
/// `coreml/docs/COREML_TO_COREAI.md`).
///
/// Exact NeMo eval-mode chain (no dither, `normalize="NA"`):
///   1. preemph:  y[0]=x[0]; y[i]=x[i]-0.97*x[i-1]
///   2. STFT center=True: zero-pad n_fft/2 each side; Hann(win=400) centered in n_fft=512
///   3. power = re^2 + im^2  (vDSP packed FFT scaled by 0.5)
///   4. mel = fb[128,257] @ power
///   5. log(mel + 5.9604645e-08)
///   6. zero frames >= seq_len = floor(n_samples/hop)
///
/// Output is `[nMels=128, T]` row-major — the `mel` input the Core AI encoder expects.
final class MelFrontend {

    // Config — must match dump_mel_reference.py / NeMo FilterbankFeatures.
    static let nFFT = 512
    static let hop = 160
    static let winLength = 400
    static let nMels = 128
    static let nFreq = 257  // nFFT/2 + 1
    static let preemph: Float = 0.97
    static let logGuard: Float = 5.960464477539063e-08

    private let fb: [Float]        // [nMels * nFreq] row-major mel filterbank
    private let win512: [Float]    // Hann(winLength) centered in nFFT
    private let fftSetup: FFTSetup
    private let log2n: vDSP_Length

    enum FrontendError: LocalizedError {
        case resourceMissing(String)
        case badResourceSize(String, Int, Int)

        var errorDescription: String? {
            switch self {
            case .resourceMissing(let name):
                return "Mel front-end resource missing: \(name)"
            case .badResourceSize(let name, let got, let want):
                return "Mel front-end resource \(name) has \(got) floats, expected \(want)"
            }
        }
    }

    /// Load the mel filterbank + window from the bundled `coreai/` resources.
    init(resourceDirectory dir: URL) throws {
        let fbURL = dir.appendingPathComponent("mel_filterbank.f32")
        let winURL = dir.appendingPathComponent("mel_window.f32")
        let fbFloats = try Self.readFloats(fbURL, name: "mel_filterbank.f32")
        let winFloats = try Self.readFloats(winURL, name: "mel_window.f32")
        guard fbFloats.count == Self.nMels * Self.nFreq else {
            throw FrontendError.badResourceSize("mel_filterbank.f32", fbFloats.count, Self.nMels * Self.nFreq)
        }
        guard winFloats.count == Self.winLength else {
            throw FrontendError.badResourceSize("mel_window.f32", winFloats.count, Self.winLength)
        }
        self.fb = fbFloats

        // Center the Hann window in the n_fft frame.
        var w = [Float](repeating: 0, count: Self.nFFT)
        let off = (Self.nFFT - Self.winLength) / 2  // 56
        for i in 0..<Self.winLength { w[off + i] = winFloats[i] }
        self.win512 = w

        self.log2n = vDSP_Length(log2(Float(Self.nFFT)))
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            throw FrontendError.resourceMissing("vDSP FFT setup")
        }
        self.fftSetup = setup
    }

    deinit { vDSP_destroy_fftsetup(fftSetup) }

    private static func readFloats(_ url: URL, name: String) throws -> [Float] {
        guard let data = try? Data(contentsOf: url) else {
            throw FrontendError.resourceMissing(name)
        }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// Compute log-mel features for one audio chunk.
    /// - Parameter samples: 16 kHz mono float PCM (already includes any pre-encode
    ///   left-context the caller wants prepended).
    /// - Returns: `(mel, T)` where `mel` is `[nMels * T]` row-major and `T` is the
    ///   number of frames.
    func melSpectrogram(_ samples: [Float]) -> (mel: [Float], frames: Int) {
        let nSamples = samples.count
        let hop = Self.hop, nFFT = Self.nFFT, half = nFFT / 2
        let nMels = Self.nMels, nFreq = Self.nFreq

        // 1. preemphasis
        var pre = [Float](repeating: 0, count: nSamples)
        if nSamples > 0 { pre[0] = samples[0] }
        if nSamples > 1 {
            for i in 1..<nSamples { pre[i] = samples[i] - Self.preemph * samples[i - 1] }
        }

        // center pad n_fft/2 each side
        let pad = nFFT / 2
        var padded = [Float](repeating: 0, count: nSamples + 2 * pad)
        for i in 0..<nSamples { padded[i + pad] = pre[i] }

        // frame count for center=True STFT
        let T = max(0, (padded.count - nFFT) / hop + 1)
        if T == 0 { return ([], 0) }

        // Power spectra for all frames, [T, nFreq] row-major.
        var powerAll = [Float](repeating: 0, count: T * nFreq)
        var frame = [Float](repeating: 0, count: nFFT)
        var realp = [Float](repeating: 0, count: half)
        var imagp = [Float](repeating: 0, count: half)

        for t in 0..<T {
            let startIdx = t * hop
            padded.withUnsafeBufferPointer { pb in
                vDSP_vmul(pb.baseAddress! + startIdx, 1, win512, 1, &frame, 1, vDSP_Length(nFFT))
            }
            let powerBase = t * nFreq
            realp.withUnsafeMutableBufferPointer { rp in
                imagp.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    frame.withUnsafeBytes { raw in
                        raw.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) { cp in
                            vDSP_ctoz(cp, 2, &split, 1, vDSP_Length(half))
                        }
                    }
                    vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                    // vDSP packed real FFT: scale 0.5 to match torch; rp[0]=DC, ip[0]=Nyquist.
                    let scale: Float = 0.5
                    let dc = rp[0] * scale
                    let nyq = ip[0] * scale
                    powerAll[powerBase] = dc * dc
                    powerAll[powerBase + half] = nyq * nyq
                    for k in 1..<half {
                        let re = rp[k] * scale
                        let im = ip[k] * scale
                        powerAll[powerBase + k] = re * re + im * im
                    }
                }
            }
        }

        // mel = fb[nMels, nFreq] @ powerᵀ[nFreq, T] as ONE vDSP matmul, then a
        // single vectorized log(mel + guard) pass — replaces nMels dot products
        // per frame and a scalar log per element.
        var powerT = [Float](repeating: 0, count: nFreq * T)
        vDSP_mtrans(powerAll, 1, &powerT, 1, vDSP_Length(nFreq), vDSP_Length(T))
        var melOut = [Float](repeating: 0, count: nMels * T)
        vDSP_mmul(fb, 1, powerT, 1, &melOut, 1, vDSP_Length(nMels), vDSP_Length(T), vDSP_Length(nFreq))
        melOut.withUnsafeMutableBufferPointer { buf in
            var logGuard = Self.logGuard
            vDSP_vsadd(buf.baseAddress!, 1, &logGuard, buf.baseAddress!, 1, vDSP_Length(nMels * T))
            var elementCount = Int32(nMels * T)
            vvlogf(buf.baseAddress!, buf.baseAddress!, &elementCount)
        }

        // 6. zero trailing frames beyond valid seq_len = floor(n_samples / hop)
        let seqLen = nSamples / hop
        if seqLen < T {
            for t in seqLen..<T {
                for m in 0..<nMels { melOut[m * T + t] = 0 }
            }
        }
        return (melOut, T)
    }
}
