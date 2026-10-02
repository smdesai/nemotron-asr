import Foundation

// Errors surfaced by the streaming Nemotron CoreML pipeline.

// MARK: - Errors

public enum ASRError: Error, LocalizedError {
    case notInitialized
    case invalidAudioData
    case modelLoadFailed
    case processingFailed(String)
    case modelCompilationFailed
    case unsupportedPlatform(String)
    case streamingConversionFailed(Error)
    case fileAccessFailed(URL, Error)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "ASR manager not initialized. Load the models first."
        case .invalidAudioData:
            return "Invalid audio data provided. Must be at least 300ms of 16kHz audio."
        case .modelLoadFailed:
            return "Failed to load Nemotron CoreML models."
        case .processingFailed(let message):
            return "ASR processing failed: \(message)"
        case .modelCompilationFailed:
            return "CoreML model compilation failed after recovery attempts."
        case .unsupportedPlatform(let message):
            return message
        case .streamingConversionFailed(let error):
            return "Streaming audio conversion failed: \(error.localizedDescription)"
        case .fileAccessFailed(let url, let error):
            return "Failed to access audio file at \(url.path): \(error.localizedDescription)"
        }
    }
}
