import CoreML
import Foundation

/// Encoder cache allocation helpers.
public enum EncoderCacheManager {
    /// Create a zero-initialized array with the specified shape and type.
    public static func createZeroArray(
        shape: [Int],
        dataType: MLMultiArrayDataType = .float32
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: shape.map { NSNumber(value: $0) },
            dataType: dataType
        )
        array.reset(to: 0)
        return array
    }
}
