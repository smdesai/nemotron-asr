@preconcurrency import CoreML
import Foundation

/// Shared utilities for creating `MLModelConfiguration` instances and resolving model directories.
public enum MLModelConfigurationUtils {

    /// Create a default `MLModelConfiguration` with low-precision GPU accumulation enabled.
    ///
    /// - Parameter computeUnits: Compute units to use (default: `.cpuAndNeuralEngine`).
    /// - Returns: Configured `MLModelConfiguration`.
    public static func defaultConfiguration(
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true
        config.computeUnits = computeUnits
        // ANE residency hints (iOS 17.4+ / macOS 14.4+). Tells the scheduler
        // to fold shape work into the ANE program (reshapeFrequency =
        // .infrequent) and bias compile toward ANE residency
        // (specializationStrategy = .fastPrediction). Picks up the
        // "scheduler chose CPU" shape-op residue without touching the model.
        // NOTE: MLOptimizationHints (reshapeFrequency=.infrequent,
        // specializationStrategy=.fastPrediction) was tested for this model
        // and REGRESSED RTFx 126.6 → 93.3 (-26%) on a 1h LS test-clean
        // bench. The hints trigger re-specialization that lands ops less
        // optimally for our static-shape graph. Don't re-enable without
        // re-benching. See /Users/hanweng/Documents/voicelink/results/
        // for the full investigation.
        return config
    }

    // `defaultModelsDirectory(for:)` is omitted from the watch port: it depended
    // on the full `Repo` enum (download/cache machinery) which the watch never
    // uses — models are loaded directly from the app bundle.
}
