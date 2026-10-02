import CoreAI
import Foundation
import os

/// On-device feasibility + speed probe for Core AI on **this watch**.
///
/// watchOS runs `.aimodel` assets directly (on-device specialization; there is
/// no `coreai-build` AOT backend for watchOS). The watchOS 27 beta failed that
/// specialization ("Unsupported SoC (m11)", then a libODIECompiler crash on the
/// CPU path); this re-tests it on the current OS and, if the shards load, times
/// the real encoder chain:
///   1. device SoC, available compute kinds, RAM headroom;
///   2. shard 0 specialization ANE-preferred, then CPU-only (diagnostic only —
///      released before step 3);
///   3. `CoreAIEncoderRunner.load()` (the iOS loader: ANE-preferred, CPU
///      fallback) on all four shards, then N chunk encodes through
///      shard0→1→2→3 with threaded caches — the exact iOS encode path.
///
/// Gated on the bundle: runs only when a `NemotronWatchCoreAIProbe/` folder
/// (encoder_shard_{0..3}_int8.aimodel) is staged into the app, in which case it
/// REPLACES the CoreML load (both model sets would not fit in RAM together).
/// Results are printed, written to Documents/coreai_probe.txt, and returned for
/// display.
@available(watchOS 27.0, *)
@MainActor
enum WatchCoreAIProbe {
    /// 2240 ms tier: 224 mel frames + 9 pre-encode cache frames per chunk.
    static let melFrames = 233
    static let chunkSeconds = 2.24
    static let timedChunks = 6

    /// The probe asset folder, if this build bundles one (encoder shards and/or a
    /// `diag/` ladder of small models).
    static var bundledDirectory: URL? {
        guard let dir = Bundle.main.url(forResource: "NemotronWatchCoreAIProbe", withExtension: nil)
        else { return nil }
        let shard0 = CoreAIAssets.encoderShardURL(0, in: dir)
        let diag = dir.appendingPathComponent("diag")
        let fm = FileManager.default
        return fm.fileExists(atPath: shard0.path) || fm.fileExists(atPath: diag.path) ? dir : nil
    }

    static func run(dir: URL) async -> String {
        let diag = dir.appendingPathComponent("diag")
        if FileManager.default.fileExists(atPath: diag.path) {
            return await runDiagnostics(dir: diag)
        }
        return await runShards(dir: dir)
    }

    /// Bisect the on-device CPU compile: specialize (.cpuOnly) and run once each
    /// `diag/*.aimodel`, smallest op set first. A crash in the compiler kills the
    /// app, so progress is appended to Documents/coreai_diag2.txt BEFORE each
    /// attempt; on relaunch, a model with BEGIN but no END is reported as
    /// CRASHED and skipped, and the ladder continues. Delete the app (or the
    /// file) to start over.
    static func runDiagnostics(dir: URL) async -> String {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let progress = docs.appendingPathComponent("coreai_diag2.txt")
        var text = (try? String(contentsOf: progress, encoding: .utf8)) ?? ""
        func append(_ s: String) {
            print("[WatchCoreAIProbe] \(s)")
            text += s + "\n"
            try? text.write(to: progress, atomically: true, encoding: .utf8)
        }
        func ms(since t: ContinuousClock.Instant) -> Int {
            let d = ContinuousClock.now - t
            return Int(Double(d.components.seconds) * 1e3 + Double(d.components.attoseconds) / 1e15)
        }
        if text.isEmpty {
            append(
                "arch=\(AIModel.deviceArchitectureName) compute=\(ComputeUnitKind.availableKinds) "
                    + "os=\(ProcessInfo.processInfo.operatingSystemVersionString) "
                    + "avail=\(os_proc_available_memory() / 1_000_000)MB")
        }
        let models = ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? [])
            .filter { $0.hasSuffix(".aimodel") }.sorted()
        // Each model is tried with DEFAULT specialization options (no compute
        // preference — the runtime picks), then .cpuOnly. Keys are "<mode> <name>".
        for mode in ["default", "cpuOnly"] {
        for name in models {
            let key = "\(mode) \(name)"
            if text.contains("END \(key)") || text.contains("CRASHED \(key)") { continue }
            if text.contains("BEGIN \(key)") {
                append("CRASHED \(key) (app died during the previous attempt)")
                continue
            }
            append("BEGIN \(key)")
            let url = dir.appendingPathComponent(name)
            do {
                let t = ContinuousClock.now
                let model: AIModel
                if mode == "default" {
                    model = try await AIModel(contentsOf: url)
                } else if let cache = AIModelCache(appGroup: CoreAIEncoderRunner.appGroupCacheId) {
                    model = try await AIModel.specialize(contentsOf: url, options: .cpuOnly, cache: cache)
                } else {
                    model = try await AIModel(contentsOf: url, options: .cpuOnly)
                }
                let loadMs = ms(since: t)
                guard let fn = try model.loadFunction(named: model.functionNames.first ?? "main") else {
                    append("END \(key) load OK \(loadMs) ms, no function")
                    continue
                }
                var inputs: [String: NDArray] = [:]
                for input in fn.descriptor.inputNames {
                    guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: input) else { continue }
                    var array = NDArray(descriptor: nd.resolvingDynamicDimensions(nd.shape))
                    zeroFill(&array)
                    inputs[input] = array
                }
                let tr = ContinuousClock.now
                _ = try await fn.run(inputs: inputs)
                append("END \(key) OK load \(loadMs) ms, run \(ms(since: tr)) ms")
            } catch {
                append("END \(key) FAILED: \(error)")
            }
        }
        }
        return text
    }

    private static func zeroFill(_ array: inout NDArray) {
        let count = array.shape.reduce(1, *)
        switch array.scalarType {
        case .float32:
            var v = array.mutableView(as: Float.self)
            v.withUnsafeMutablePointer { p, _, _ in for i in 0 ..< count { p[i] = 0 } }
        case .float16:
            var v = array.mutableView(as: Float16.self)
            v.withUnsafeMutablePointer { p, _, _ in for i in 0 ..< count { p[i] = 0 } }
        case .int32:
            var v = array.mutableView(as: Int32.self)
            v.withUnsafeMutablePointer { p, _, _ in for i in 0 ..< count { p[i] = 0 } }
        default:
            break
        }
    }

    static func runShards(dir: URL) async -> String {
        var lines: [String] = []
        func log(_ s: String) {
            print("[WatchCoreAIProbe] \(s)")
            lines.append(s)
        }
        func availMB() -> Int { Int(os_proc_available_memory() / 1_000_000) }
        func ms(since t: ContinuousClock.Instant) -> Double {
            let d = ContinuousClock.now - t
            return Double(d.components.seconds) * 1e3 + Double(d.components.attoseconds) / 1e15
        }

        log("arch=\(AIModel.deviceArchitectureName) compute=\(ComputeUnitKind.availableKinds)")
        log(
            "RAM=\(ProcessInfo.processInfo.physicalMemory / 1_000_000)MB "
                + "avail=\(availMB())MB os=\(ProcessInfo.processInfo.operatingSystemVersionString)")

        // 2) Diagnostic: which specialization paths work for shard 0.
        let shard0 = CoreAIAssets.encoderShardURL(0, in: dir)
        do {
            let t = ContinuousClock.now
            let m = try await AIModel(
                contentsOf: shard0,
                options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            log("shard0 ANE-pref OK \(Int(ms(since: t))) ms fns=\(m.functionNames)")
        } catch {
            log("shard0 ANE-pref FAILED: \(error)")
        }
        do {
            let t = ContinuousClock.now
            let m: AIModel
            if let cache = AIModelCache(appGroup: CoreAIEncoderRunner.appGroupCacheId) {
                m = try await AIModel.specialize(contentsOf: shard0, options: .cpuOnly, cache: cache)
            } else {
                m = try await AIModel(contentsOf: shard0, options: .cpuOnly)
            }
            log("shard0 CPU-only OK \(Int(ms(since: t))) ms fns=\(m.functionNames)")
        } catch {
            log("shard0 CPU-only FAILED: \(error)")
        }

        // 3) Full chain through the iOS runner.
        let runner = CoreAIEncoderRunner(coreaiDirectory: dir)
        defer { runner.tearDown() }
        do {
            let t = ContinuousClock.now
            try await runner.load()
            log("4-shard load \(Int(ms(since: t))) ms avail=\(availMB())MB")
        } catch {
            log("4-shard load FAILED: \(error)")
            return finish(lines)
        }

        let mel = [Float](repeating: 0, count: MelFrontend.nMels * melFrames)
        runner.resetStreamingCaches()
        var times: [Double] = []
        for i in 0 ..< timedChunks {
            do {
                let t = ContinuousClock.now
                let (encoded, shape) = try await runner.encode(
                    mel: mel, frames: melFrames, promptId: 0)
                let dt = ms(since: t)
                times.append(dt)
                let finite = encoded.allSatisfy { $0.isFinite }
                log("chunk \(i): \(Int(dt)) ms shape=\(shape) finite=\(finite) avail=\(availMB())MB")
            } catch {
                log("chunk \(i) FAILED: \(error)")
                return finish(lines)
            }
        }
        // Chunk 0 includes first-run warmup; report the steady state.
        let steady = times.dropFirst().sorted()
        if !steady.isEmpty {
            let median = steady[steady.count / 2]
            log(
                String(
                    format: "encoder median %.0f ms/chunk → encoder-only RTF %.1fx",
                    median, chunkSeconds * 1000 / median))
        }
        return finish(lines)
    }

    private static func finish(_ lines: [String]) -> String {
        let text = lines.joined(separator: "\n")
        if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            try? text.write(
                to: docs.appendingPathComponent("coreai_probe.txt"), atomically: true, encoding: .utf8)
        }
        return text
    }
}
