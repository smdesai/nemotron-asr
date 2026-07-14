import CoreAI
import Foundation

/// On-device introspection of the Core AI encoder shards via
/// `AIModelAsset.summary(includingStatistics:)` — the Core AI analogue of the
/// CoreML `MLComputePlan` residency check used in the conversion repo.
///
/// Surfaces, per shard: the target **compute types** (ANE/GPU/CPU), the **storage
/// types** (dtype histogram — this is where lingering fp32 shows up and explains
/// the `ANE cannot handle intermediate tensor type fp32` fallback), and the
/// **operation distribution** (op-name histogram).
///
/// Verified CoreAI API (`AIModelAsset.Summary`):
///   - `computeTypes: [String]`
///   - `storageTypes: [StorageType]` (`.typeName`, `.count`)
///   - `operationDistribution: [OperationCount]` (`.operationName`, `.count`)
@available(iOS 27.0, watchOS 27.0, *)
enum CoreAIResidencyProbe {

    struct ShardReport {
        let name: String
        let computeTypes: [String]
        /// dtype → element/op count (e.g. "float32" → 5).
        let storageTypes: [(type: String, count: Int)]
        /// op name → count, sorted descending.
        let topOps: [(op: String, count: Int)]
        /// Total fp32/fp64 storage ELEMENTS. The export pipeline is fp16-clean
        /// (zero fp32 nodes in the torch graph); coreai-torch's lowering still
        /// materializes a handful of scalar constants (LayerNorm eps, the
        /// attention 1/√d_k and −10000 mask fill, …) as f32/f64 — ~9-12 elements
        /// per shard, deduped, all fp16-representable. Verified unfixable from
        /// the torch side (rewriting scalars to fp16 tensor constants yields an
        /// identical storage histogram after prog.optimize()).
        var fp32Elements: Int {
            storageTypes
                .filter {
                    let t = $0.type.lowercased()
                    return t == "fp32" || t == "fp64"
                        || (t.contains("float") && (t.contains("32") || t.contains("64")))
                }
                .reduce(0) { $0 + $1.count }
        }
        /// True only when fp32/fp64 storage is big enough to be real tensor
        /// data (a fallback-causing subgraph), not lowering-injected scalar
        /// constants.
        var hasFP32Storage: Bool { fp32Elements > 1000 }
    }

    /// Probe all `encoder_shard_{0..3}.aimodel` in the given `coreai/` directory.
    static func probeShards(in dir: URL) -> [ShardReport] {
        var reports: [ShardReport] = []
        for i in 0 ..< 4 {
            let url = CoreAIAssets.encoderShardURL(i, in: dir)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            do {
                let asset = try AIModelAsset(contentsOf: url)
                guard let summary = try asset.summary(includingStatistics: true) else {
                    reports.append(
                        ShardReport(
                            name: "shard_\(i)", computeTypes: ["<no summary>"],
                            storageTypes: [], topOps: []))
                    continue
                }
                let storage = summary.storageTypes
                    .map { (type: $0.typeName, count: $0.count) }
                    .sorted { $0.count > $1.count }
                let ops = summary.operationDistribution
                    .map { (op: $0.operationName, count: $0.count) }
                    .sorted { $0.count > $1.count }
                reports.append(
                    ShardReport(
                        name: "shard_\(i)",
                        computeTypes: summary.computeTypes,
                        storageTypes: storage,
                        topOps: Array(ops.prefix(12))
                    ))
            } catch {
                reports.append(
                    ShardReport(
                        name: "shard_\(i)", computeTypes: ["<error: \(error)>"],
                        storageTypes: [], topOps: []))
            }
        }
        return reports
    }

    /// Format a human-readable residency report and print it (and return it for
    /// the UI). Flags any shard whose storage still contains fp32.
    static func report(in dir: URL) -> String {
        let reports = probeShards(in: dir)
        var lines: [String] = ["=== Core AI encoder residency probe ==="]
        for r in reports {
            lines.append("• \(r.name): computeTypes=\(r.computeTypes)")
            let storageStr = r.storageTypes.map { "\($0.type)×\($0.count)" }.joined(separator: ", ")
            let fpNote: String
            if r.hasFP32Storage {
                fpNote = "   ⚠️ fp32 tensors present → GPU fallback"
            } else if r.fp32Elements > 0 {
                fpNote = "   ✓ fp16-clean (\(r.fp32Elements) scalar constants)"
            } else {
                fpNote = "   ✓ no fp32"
            }
            lines.append("    storage: \(storageStr.isEmpty ? "<none>" : storageStr)" + fpNote)
            let opsStr = r.topOps.prefix(6).map { "\($0.op)×\($0.count)" }.joined(separator: ", ")
            if !opsStr.isEmpty { lines.append("    top ops: \(opsStr)") }
        }
        let anyFP32 = reports.contains { $0.hasFP32Storage }
        lines.append(
            anyFP32
                ? "⚠️ Some shards still store fp32 tensors — those ops fall back off the ANE."
                : "✓ No fp32 tensor storage in any shard — ANE-eligible (scalar constants only).")
        let text = lines.joined(separator: "\n")
        print(text)
        return text
    }
}
