import CoreAI
import Foundation

/// One-shot on-device feasibility probe for Core AI on **this watch**.
///
/// The Core AI (.aimodel) path was shelved because the watchOS 27 beta runtime
/// compiler had no backend for the watch SoC — "Unsupported SoC (m11)". This
/// re-tests the current watch's runtime after a reported compiler update:
///   1. prints the device SoC name + available compute units, then
///   2. tries to JIT-specialize a *palettized* encoder shard ANE-preferred
///      (the real target — exercises the m11 ANE backend), then
///   3. CPU-only (bypasses the ANE SoC backend — a viable, if slower, fallback).
///
/// Self-contained: bundles one shard in `NemotronWatchCoreAIProbe/` and does NOT
/// touch `NemotronWatchModels/` (the CoreML pipeline stays the shipping path).
@available(watchOS 27.0, *)
enum WatchCoreAIProbe {
    static let appGroupCacheId = "group.com.sdesai.NemotronASR"

    static func run() async {
        print("[WatchCoreAIProbe] device arch: \(AIModel.deviceArchitectureName), "
            + "available compute: \(ComputeUnitKind.availableKinds)")

        guard let dir = Bundle.main.url(forResource: "NemotronWatchCoreAIProbe", withExtension: nil) else {
            print("[WatchCoreAIProbe] ❌ probe model folder not bundled")
            return
        }
        let url = dir.appendingPathComponent("encoder_shard_0.aimodel")
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("[WatchCoreAIProbe] ❌ \(url.lastPathComponent) not found in bundle")
            return
        }
        print("[WatchCoreAIProbe] probing \(url.lastPathComponent) …")

        // 1) ANE-preferred JIT specialization — the real target (m11 ANE backend).
        do {
            let m = try await AIModel(
                contentsOf: url,
                options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            print("[WatchCoreAIProbe] ✅ ANE-preferred LOAD OK — functions=\(m.functionNames)")
        } catch {
            print("[WatchCoreAIProbe] ❌ ANE-preferred FAILED: \(error)")
        }

        // 2) CPU-only — no ANE SoC backend needed (app-group cache path, then plain).
        do {
            let m: AIModel
            if let cache = AIModelCache(appGroup: appGroupCacheId) {
                m = try await AIModel.specialize(contentsOf: url, options: .cpuOnly, cache: cache)
            } else {
                m = try await AIModel(contentsOf: url, options: .cpuOnly)
            }
            print("[WatchCoreAIProbe] ✅ CPU-only LOAD OK — functions=\(m.functionNames)")
        } catch {
            print("[WatchCoreAIProbe] ❌ CPU-only FAILED: \(error)")
        }
    }
}
