import Darwin
import Foundation

/// Headless on-device benchmark, driven entirely by launch environment so model
/// builds can be A/B'd from the Mac without touching the UI:
///
///     xcrun devicectl device copy to --device <id> --domain-type appDataContainer \
///         --domain-identifier com.sdesai.NemotronASR \
///         --source clip.wav --destination Documents/bench.wav
///     DEVICECTL_CHILD_NEMOTRON_BENCH=Documents/bench.wav \
///     DEVICECTL_CHILD_NEMOTRON_BENCH_BACKEND=coreai \
///     DEVICECTL_CHILD_NEMOTRON_BENCH_RUNS=3 \
///         xcrun devicectl device process launch --device <id> --terminate-existing \
///         com.sdesai.NemotronASR
///     xcrun devicectl device copy from ... --source Documents/bench_result.txt
///
/// `NEMOTRON_BENCH` is a path relative to the app container (or absolute).
/// `NEMOTRON_BENCH_BACKEND` = coreml | coreai (default: current setting);
/// `NEMOTRON_BENCH_CHUNK_MS` = 560 | 1120 | 2240 | 4480 (default: current).
/// `NEMOTRON_MODELS_ROOT` (read by TranscriptionEngine) points the model tree at a
/// copy in the container, e.g. `Documents/Models`, to A/B builds without reinstalling.
/// The first run includes model specialization; later runs are steady state.
/// The user's backend / chunk settings are restored afterwards.
@MainActor
enum BenchmarkRunner {
    static func runIfRequested(engine: TranscriptionEngine, settings: AppSettings) async {
        let env = ProcessInfo.processInfo.environment
        // NEMOTRON_CLEAR_DOCS=AB,MONO,...: delete those Documents/ subfolders (test
        // model trees pushed with `devicectl device copy to`; devicectl can't delete).
        if let names = env["NEMOTRON_CLEAR_DOCS"], !names.isEmpty {
            let docs = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Documents")
            for name in names.split(separator: ",").map({ $0.trimmingCharacters(in: .whitespaces) })
            where !name.isEmpty && !name.contains("/") && name != ".." {
                let url = docs.appendingPathComponent(name)
                do {
                    try FileManager.default.removeItem(at: url)
                    print("[Bench] cleared Documents/\(name)")
                } catch {
                    print("[Bench] clear Documents/\(name) failed: \(error.localizedDescription)")
                }
            }
        }
        guard let rel = env["NEMOTRON_BENCH"], !rel.isEmpty else { return }
        let home = URL(fileURLWithPath: NSHomeDirectory())
        let audio = rel.hasPrefix("/") ? URL(fileURLWithPath: rel) : home.appendingPathComponent(rel)
        let runs = max(1, Int(env["NEMOTRON_BENCH_RUNS"] ?? "") ?? 3)

        let savedBackend = settings.backend
        let savedChunk = settings.chunkSize
        defer {
            settings.backend = savedBackend
            settings.chunkSize = savedChunk
        }
        if let b = env["NEMOTRON_BENCH_BACKEND"], let backend = InferenceBackend(rawValue: b) {
            settings.backend = backend
        }
        if let c = Int(env["NEMOTRON_BENCH_CHUNK_MS"] ?? ""), let chunk = ChunkSize(rawValue: c) {
            settings.chunkSize = chunk
        }

        var lines = [
            "backend=\(settings.backend.rawValue) chunk=\(settings.chunkSize.rawValue)ms "
                + "audio=\(audio.lastPathComponent) runs=\(runs)"
        ]
        func log(_ s: String) {
            print("[Bench] \(s)")
            lines.append(s)
        }
        print("[Bench] \(lines[0])")
        guard FileManager.default.fileExists(atPath: audio.path) else {
            log("audio not found: \(audio.path)")
            write(lines, home: home)
            return
        }

        log("mem start: \(memoryLine())")
        engine.invalidateForSettingsChange()
        let prepStart = Date()
        await engine.prepareModelIfNeeded()
        log(String(format: "prepare %.2fs phase=%@", Date().timeIntervalSince(prepStart), "\(engine.phase)"))
        log("mem after prepare: \(memoryLine())")

        for i in 0 ..< runs {
            let t = Date()
            await engine.transcribeFile(url: audio)
            let wall = Date().timeIntervalSince(t)
            let rtfx = engine.lastRTFx.map { String(format: "%.1f", $0) } ?? "nil"
            log(
                String(format: "run %d: wall %.2fs RTFx %@ chars %d", i, wall, rtfx, engine.transcript.count)
                    + (engine.lastStageTimes.map { " | \($0)" } ?? ""))
        }
        log("mem after runs: \(memoryLine())")
        log("text: \(engine.transcript.prefix(400))")
        // Full transcript for diffing variants (the log keeps a prefix).
        try? engine.transcript.write(
            to: home.appendingPathComponent("Documents/bench_text.txt"), atomically: true,
            encoding: .utf8)
        write(lines, home: home)
    }

    /// Physical footprint (what jetsam enforces) and its lifetime peak for this
    /// process, plus the remaining headroom before the per-app memory limit.
    static func memoryLine() -> String {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return "task_info failed (\(kr))" }
        let mb = { (v: UInt64) in Int(v / 1_000_000) }
        return "footprint \(mb(info.phys_footprint)) MB, peak \(mb(UInt64(info.ledger_phys_footprint_peak))) MB, "
            + "available \(mb(UInt64(os_proc_available_memory()))) MB"
    }

    private static func write(_ lines: [String], home: URL) {
        let out = home.appendingPathComponent("Documents/bench_result.txt")
        try? lines.joined(separator: "\n").write(to: out, atomically: true, encoding: .utf8)
        print("[Bench] wrote \(out.path)")
    }
}
