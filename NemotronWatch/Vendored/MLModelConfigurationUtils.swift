@preconcurrency import CoreML
import Foundation

/// Shared `MLModelConfiguration` factory for the streaming Nemotron CoreML pipeline.
public enum MLModelConfigurationUtils {

    /// Create a default `MLModelConfiguration` with low-precision GPU accumulation enabled.
    ///
    /// Note: `MLOptimizationHints` (reshapeFrequency = .infrequent,
    /// specializationStrategy = .fastPrediction) was benchmarked for this model
    /// and regressed RTFx by ~26% (re-specialization lands ops less optimally
    /// for the static-shape graph); don't enable it without re-benching.
    ///
    /// - Parameter computeUnits: Compute units to use (default: `.cpuAndNeuralEngine`).
    public static func defaultConfiguration(
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true
        config.computeUnits = computeUnits
        return config
    }
}
