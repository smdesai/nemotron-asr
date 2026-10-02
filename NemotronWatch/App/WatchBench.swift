import AVFoundation
import Darwin
import Foundation
import os

/// On-watch CoreML file benchmark for A/B-ing encoder builds (e.g. int8 vs
/// palettized shards staged into NemotronWatchModels/).
///
/// Gated on the bundle: runs only when `NemotronWatchBench/bench.wav` (16 kHz
/// mono) is staged into the app. After the normal CoreML load it streams the
/// file through the same manager the mic path uses, in mic-sized blocks, and
/// reports wall time, RTF, the manager's prep/encoder/decoder split, and memory
/// headroom. Result goes to the console, Documents/watch_bench.txt, and the UI.
@MainActor
enum WatchBench {
    static var bundledAudio: URL? {
        guard
            let dir = Bundle.main.url(forResource: "NemotronWatchBench", withExtension: nil)
        else { return nil }
        let wav = dir.appendingPathComponent("bench.wav")
        return FileManager.default.fileExists(atPath: wav.path) ? wav : nil
    }

    static func run(
        audio: URL, manager: StreamingNemotronMultilingualAsrManager, runs: Int = 3
    ) async -> String {
        var lines: [String] = []
        func log(_ s: String) {
            print("[WatchBench] \(s)")
            lines.append(s)
        }
        do {
            let samples = try readMono16k(audio)
            let seconds = Double(samples.count) / 16000
            log(String(format: "audio %.1fs", seconds))
            log("mem after load: \(memoryLine())")
            let block = 4096  // ~0.26 s, the mic block size
            for r in 0 ..< runs {
                await manager.reset()
                let t = Date()
                var i = 0
                while i < samples.count {
                    let end = min(i + block, samples.count)
                    _ = try await manager.process(samples: Array(samples[i ..< end]))
                    i = end
                }
                let text = try await manager.finish()
                let wall = Date().timeIntervalSince(t)
                let enc = Double(await manager.encNanos) / 1e9
                let dec = Double(await manager.decNanos) / 1e9
                let prep = Double(await manager.prepNanos) / 1e9
                let chunks = await manager.chunkCount
                log(
                    String(
                        format: "run %d: wall %.2fs RTFx %.2f | prep %.2fs enc %.2fs dec %.2fs chunks %d"
                            + " (enc %.0f ms/chunk) chars %d avail %dMB",
                        r, wall, seconds / wall, prep, enc, dec, chunks,
                        chunks > 0 ? enc * 1000 / Double(chunks) : 0, text.count,
                        Int(os_proc_available_memory() / 1_000_000)))
                log("mem after run \(r): \(memoryLine())")
                if r == runs - 1 { log("text: \(text.prefix(300))") }
            }
        } catch {
            log("FAILED: \(error)")
        }
        let text = lines.joined(separator: "\n")
        if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            try? text.write(
                to: docs.appendingPathComponent("watch_bench.txt"), atomically: true, encoding: .utf8)
        }
        return text
    }

    /// Physical footprint (what jetsam enforces), its lifetime peak, and the
    /// remaining headroom — same metric as the iOS BenchmarkRunner.
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

    private static func readMono16k(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
            file.processingFormat.sampleRate == 16000, file.processingFormat.channelCount == 1,
            let buf = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length))
        else { throw NSError(domain: "WatchBench", code: 1, userInfo: [NSLocalizedDescriptionKey: "bench.wav must be 16 kHz mono"]) }
        try file.read(into: buf)
        return Array(UnsafeBufferPointer(start: buf.floatChannelData![0], count: Int(buf.frameLength)))
    }
}
