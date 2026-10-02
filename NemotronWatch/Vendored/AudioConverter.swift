@preconcurrency import AVFoundation
import Accelerate
import Foundation
import OSLog
import os

/// Converts audio buffers to the format required by ASR (16kHz, mono, Float32).
///
/// Implementation notes:
/// - Uses `AVAudioConverter` for all sample-rate, sample-format, and channel-count conversions.
/// - Avoids any manual resampling; only raw sample extraction occurs after conversion.
/// - Creates a new converter for each operation (stateless).
final public class AudioConverter: Sendable {
    private let logger = AppLogger(category: "AudioConverter")
    private let targetFormat: AVAudioFormat
    private let debug: Bool

    /// Public initializer so external modules (e.g. CLI) can construct the converter
    /// - Parameters:
    ///   - targetFormat: Target audio format
    ///   - debug: Whether to log debug messages
    public init(targetFormat: AVAudioFormat? = nil, debug: Bool = false) {
        self.debug = debug
        if let format = targetFormat {
            self.targetFormat = format
        } else {
            /// Most audio models expect this format.
            /// Target format for ASR, Speaker diarization model: 16kHz, mono
            self.targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 16000,
                channels: 1,
                interleaved: false
            )!
        }
    }

    // MARK: - Public Resampling Methods

    /// Resample a float array from one sample rate to the target sample rate.
    /// - Parameters:
    ///   - samples: Input audio samples (mono Float32)
    ///   - inputRate: Input sample rate (e.g., 48000)
    /// - Returns: Float array resampled to target sample rate
    private func resample(_ samples: [Float], from inputRate: Double) throws -> [Float] {
        guard !samples.isEmpty else { return [] }

        let outputRate = targetFormat.sampleRate

        // If already at target rate, return as-is
        if inputRate == outputRate {
            return samples
        }

        return try resampleWithAVAudio(samples, from: inputRate, to: outputRate)
    }

    /// Convert a standalone buffer to the target format.
    /// - Parameters:
    ///   - buffer: Input audio buffer (any format)
    /// - Returns: Float array at target sample rate (default 16kHz) mono
    public func resampleBuffer(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        let inputFormat = buffer.format

        // Fast path: if already in target format, just extract samples
        if isTargetFormat(inputFormat) {
            return extractFloatArray(from: buffer)
        }
        return try convertBuffer(buffer, to: targetFormat)
    }

    /// Convert an audio file to target sample rate (default 16kHz) mono Float32 samples.
    /// - Parameters:
    ///   - url: URL of the audio file to read
    /// - Returns: Float array at target sample rate mono
    public func resampleAudioFile(_ url: URL) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let chunkSize = max(4096, Int(format.sampleRate))
        var monoSamples: [Float] = []
        monoSamples.reserveCapacity(Int(audioFile.length))

        while audioFile.framePosition < audioFile.length {
            let remaining = Int(audioFile.length - audioFile.framePosition)
            let framesToRead = AVAudioFrameCount(min(chunkSize, remaining))
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRead)
            else {
                throw AudioConverterError.failedToCreateBuffer
            }
            try audioFile.read(into: buffer)
            if buffer.frameLength == 0 {
                break
            }
            monoSamples.append(contentsOf: try extractMonoFloat32(from: buffer))
        }

        return try resample(monoSamples, from: format.sampleRate)
    }

    // MARK: - Private Helpers

    /// Resample using AVAudioConverter with raw float arrays
    private func resampleWithAVAudio(
        _ samples: [Float], from inputRate: Double, to outputRate: Double
    ) throws -> [Float] {
        // Create input format and buffer
        guard
            let inputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: inputRate,
                channels: 1,
                interleaved: false
            )
        else {
            throw AudioConverterError.failedToCreateSourceFormat
        }

        guard
            let inputBuffer = AVAudioPCMBuffer(
                pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(samples.count))
        else {
            throw AudioConverterError.failedToCreateBuffer
        }
        inputBuffer.frameLength = AVAudioFrameCount(samples.count)

        // Copy samples into buffer
        if let channelData = inputBuffer.floatChannelData {
            _ = samples.withUnsafeBufferPointer { src in
                memcpy(channelData[0], src.baseAddress!, samples.count * MemoryLayout<Float>.stride)
            }
        }

        // Create output format
        guard
            let outputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: outputRate,
                channels: 1,
                interleaved: false
            )
        else {
            throw AudioConverterError.failedToCreateSourceFormat
        }

        return try convertBuffer(inputBuffer, to: outputFormat)
    }

    /// Extract mono Float32 samples from a buffer (mixing channels if needed)
    private func extractMonoFloat32(from buffer: AVAudioPCMBuffer) throws -> [Float] {
        let format = buffer.format
        let frameCount = Int(buffer.frameLength)

        guard frameCount > 0 else { return [] }

        // If already mono Float32 non-interleaved, fast path
        if format.channelCount == 1 && format.commonFormat == .pcmFormatFloat32
            && !format.isInterleaved
        {
            guard let channelData = buffer.floatChannelData else { return [] }
            return Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        }

        // Need to convert to mono Float32
        // Create intermediate format (mono Float32 at original sample rate)
        guard
            let monoFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: format.sampleRate,
                channels: 1,
                interleaved: false
            )
        else {
            throw AudioConverterError.failedToCreateSourceFormat
        }

        // Use AVAudioConverter for channel mixing (same sample rate, no resampling needed)
        guard let converter = AVAudioConverter(from: format, to: monoFormat) else {
            throw AudioConverterError.failedToCreateConverter
        }

        guard
            let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: monoFormat, frameCapacity: buffer.frameCapacity)
        else {
            throw AudioConverterError.failedToCreateBuffer
        }

        let provided = OSAllocatedUnfairLock(initialState: false)
        let inputBlock: AVAudioConverterInputBlock = { _, status in
            let wasProvided = provided.withLock { state -> Bool in
                if state { return true }
                state = true
                return false
            }
            if !wasProvided {
                status.pointee = .haveData
                return buffer
            } else {
                status.pointee = .endOfStream
                return nil
            }
        }

        var error: NSError?
        let status = converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
        guard status != .error else {
            throw AudioConverterError.conversionFailed(error)
        }

        guard let channelData = outputBuffer.floatChannelData else { return [] }
        return Array(
            UnsafeBufferPointer(start: channelData[0], count: Int(outputBuffer.frameLength)))
    }

    /// Convert a buffer to the target format.
    private func convertBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws
        -> [Float]
    {
        let inputFormat = buffer.format

        // For >2 channels, use manual linear resampling since AVAudioConverter has limitations
        if inputFormat.channelCount > 2 {
            return try linearResample(buffer, to: format)
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: format) else {
            throw AudioConverterError.failedToCreateConverter
        }
        configure(converter: converter)

        // Estimate first pass capacity and allocate
        let sampleRateRatio = format.sampleRate / inputFormat.sampleRate
        let estimatedOutputFrames = AVAudioFrameCount(
            (Double(buffer.frameLength) * sampleRateRatio).rounded(.up))

        func makeOutputBuffer(_ capacity: AVAudioFrameCount) throws -> AVAudioPCMBuffer {
            guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
                throw AudioConverterError.failedToCreateBuffer
            }
            return out
        }

        var aggregated: [Float] = []
        aggregated.reserveCapacity(Int(estimatedOutputFrames))

        // AVAudioConverter consumes this input block synchronously within convert(...),
        // but Swift 6 rejects mutation of captured vars in this callback.
        let provided = OSAllocatedUnfairLock(initialState: false)
        let inputBlock: AVAudioConverterInputBlock = { _, status in
            let wasProvided = provided.withLock { state -> Bool in
                if state { return true }
                state = true
                return false
            }
            if !wasProvided {
                status.pointee = .haveData
                return buffer
            } else {
                status.pointee = .endOfStream
                return nil
            }
        }

        var error: NSError?
        let inputSampleCount = Int(buffer.frameLength)

        // First pass: convert main data
        let firstOut = try makeOutputBuffer(estimatedOutputFrames)
        let firstStatus = converter.convert(to: firstOut, error: &error, withInputFrom: inputBlock)
        guard firstStatus != .error else { throw AudioConverterError.conversionFailed(error) }
        if firstOut.frameLength > 0 {
            aggregated.append(contentsOf: extractFloatArray(from: firstOut))
        }

        // Drain remaining frames until EOS
        while true {
            let out = try makeOutputBuffer(4096)
            let status = converter.convert(to: out, error: &error, withInputFrom: inputBlock)
            guard status != .error else { throw AudioConverterError.conversionFailed(error) }
            if out.frameLength > 0 { aggregated.append(contentsOf: extractFloatArray(from: out)) }
            if status == .endOfStream { break }
        }

        let outputSampleCount = aggregated.count
        if debug {
            logger.debug(
                "Audio conversion: \(inputSampleCount) samples → \(outputSampleCount) samples, ratio: \(Double(outputSampleCount)/Double(inputSampleCount))"
            )
        }

        return aggregated
    }

    private func configure(converter: AVAudioConverter) {
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
    }

    /// Check if a format already matches the target output format.
    private func isTargetFormat(_ format: AVAudioFormat) -> Bool {
        return format.sampleRate == targetFormat.sampleRate
            && format.channelCount == targetFormat.channelCount
            && format.commonFormat == targetFormat.commonFormat
            && format.isInterleaved == targetFormat.isInterleaved
    }

    /// Resample high channel count audio (>2 channels) using linear interpolation
    /// AVAudioConverter has limitations with >2 channels, so we handle it via a linear resample. Accuracy may not be as good as AVAudioConverter.
    /// But this is needed for applications like Safari on speaker mode, or for particular hardware devices.
    private func linearResample(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws
        -> [Float]
    {
        let inputFormat = buffer.format
        guard let channelData = buffer.floatChannelData else {
            throw AudioConverterError.failedToCreateBuffer
        }

        let inputFrameCount = Int(buffer.frameLength)
        let channelCount = Int(inputFormat.channelCount)

        // Step 1: Mix down to mono
        var monoSamples = [Float](repeating: 0, count: inputFrameCount)
        let channelWeight = 1.0 / Float(channelCount)

        for frame in 0 ..< inputFrameCount {
            var sum: Float = 0
            for channel in 0 ..< channelCount {
                sum += channelData[channel][frame]
            }
            monoSamples[frame] = sum * channelWeight
        }

        // Step 2: Resample if needed
        let inputSampleRate = inputFormat.sampleRate
        let targetSampleRate = format.sampleRate

        if inputSampleRate == targetSampleRate {
            return monoSamples
        }

        // Linear interpolation resampling
        let resampleRatio = inputSampleRate / targetSampleRate
        let outputFrameCount = Int(Double(inputFrameCount) / resampleRatio)
        var outputSamples = [Float](repeating: 0, count: outputFrameCount)

        for i in 0 ..< outputFrameCount {
            let sourceIndex = Double(i) * resampleRatio
            let index = Int(sourceIndex)
            let fraction = Float(sourceIndex - Double(index))

            if index < inputFrameCount - 1 {
                // Linear interpolation between samples
                outputSamples[i] =
                    monoSamples[index] * (1.0 - fraction) + monoSamples[index + 1] * fraction
            } else if index < inputFrameCount {
                outputSamples[i] = monoSamples[index]
            }
        }

        if debug {
            logger.debug(
                "Manual resampling: \(channelCount) channels → mono, \(inputSampleRate)Hz → \(targetSampleRate)Hz"
            )
        }

        return outputSamples
    }

    /// Extract Float array from PCM buffer
    private func extractFloatArray(from buffer: AVAudioPCMBuffer) -> [Float] {
        // This function assumes mono, non-interleaved Float32 buffers.
        // All multi-channel or interleaved inputs should be converted via AVAudioConverter first.
        guard let channelData = buffer.floatChannelData else { return [] }

        let frameCount = Int(buffer.frameLength)
        if frameCount == 0 { return [] }

        // Enforce mono; converter guarantees this in normal flow.
        assert(buffer.format.channelCount == 1, "extractFloatArray expects mono buffers")

        // Fast copy using vDSP (equivalent to memcpy for contiguous Float32)
        let out = [Float](unsafeUninitializedCapacity: frameCount) { dest, initialized in
            vDSP_mmov(
                channelData[0],
                dest.baseAddress!,
                vDSP_Length(frameCount),
                1,
                1,
                1
            )
            initialized = frameCount
        }
        return out
    }

}

/// Errors that can occur during audio conversion
public enum AudioConverterError: LocalizedError {
    case failedToCreateConverter
    case failedToCreateBuffer
    case conversionFailed(Error?)
    case failedToCreateSourceFormat

    public var errorDescription: String? {
        switch self {
        case .failedToCreateConverter:
            return "Failed to create audio converter"
        case .failedToCreateBuffer:
            return "Failed to create conversion buffer"
        case .conversionFailed(let error):
            return "Audio conversion failed: \(error?.localizedDescription ?? "Unknown error")"
        case .failedToCreateSourceFormat:
            return "Failed to create a source audio format description."
        }
    }
}
