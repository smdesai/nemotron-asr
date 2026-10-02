import Foundation

/// Shared helpers for the streaming ASR manager.
public enum StreamingAsrUtils {
    /// Reset shared streaming state (audio buffer and accumulated tokens).
    public static func resetSharedState(
        audioBuffer: inout [Float],
        accumulatedTokenIds: inout [Int]
    ) {
        audioBuffer.removeAll()
        accumulatedTokenIds.removeAll()
    }
}
