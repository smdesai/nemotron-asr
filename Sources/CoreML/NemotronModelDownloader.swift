//
//  NemotronModelDownloader.swift
//  NemotronASRCoreML
//
//  Fetches one Nemotron multilingual streaming ASR variant (<ship>/<tier>ms)
//  from the Hugging Face Hub into Application Support. Dependency-free
//  (URLSession only) so the package stays usable on iOS, macOS and watchOS.
//
//  On-device layout mirrors the Hub repo, one tier per directory:
//    Application Support/nemotron-asr/models/<ship>/<tier>ms/
//        ├── metadata.json
//        ├── tokenizer.json
//        ├── preprocessor.mlmodelc/...
//        ├── encoder_pre_encode.mlmodelc/... + encoder_shard_0..3.mlmodelc/...   (split encoder)
//        │   — or — encoder.mlmodelc/...                                        (monolithic)
//        ├── decoder.mlmodelc/, joint.mlmodelc/, decoder_joint*.mlmodelc/       (whatever the tier ships)
//        └── .complete                          (sentinel — written last)
//
//  The `coreai/` subtree (Core AI .aimodel bundles) is never downloaded here.
//

import Foundation

/// Progress snapshot for a variant download.
public struct NemotronModelDownloadProgress: Sendable, Equatable {
    public let bytesCompleted: Int64
    public let bytesTotal: Int64
    public let currentFileIndex: Int
    public let totalFiles: Int
    public let currentFileName: String

    public static let zero = NemotronModelDownloadProgress(
        bytesCompleted: 0, bytesTotal: 0, currentFileIndex: 0, totalFiles: 0, currentFileName: "")

    /// 0...1 once the total size is known, nil before.
    public var fractionCompleted: Double? {
        guard bytesTotal > 0 else { return nil }
        return min(1, Double(bytesCompleted) / Double(bytesTotal))
    }
}

public enum NemotronModelDownloadError: LocalizedError {
    case networkError(String)
    case httpStatus(Int, String)
    case invalidResponse(String)
    /// The repository has no `<ship>/<tier>ms/` directory (or it is empty).
    case variantNotAvailable(ship: String, tier: String)

    public var errorDescription: String? {
        switch self {
        case .networkError(let m): return "Network error: \(m)"
        case .httpStatus(let code, let m): return "HTTP \(code): \(m)"
        case .invalidResponse(let m): return "Invalid response: \(m)"
        case .variantNotAvailable(let ship, let tier):
            return "Nemotron ASR variant \(ship)/\(tier) is not published in \(NemotronModelDownloader.repoId)."
        }
    }
}

/// Downloads Nemotron multilingual streaming ASR variants from the Hugging Face Hub.
public final class NemotronModelDownloader: Sendable {

    // MARK: - Repo coordinates

    public static let repoId = "smdesai/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML"
    public static let revision = "main"

    public init() {}

    // MARK: - Paths

    /// Application Support/nemotron-asr/models — root for all variants. Excluded from
    /// backup; created on demand.
    public static func rootDirectory() -> URL {
        let fm = FileManager.default
        let base =
            fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fm.temporaryDirectory
        let root =
            base
            .appendingPathComponent("nemotron-asr", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
        if !fm.fileExists(atPath: root.path) {
            try? fm.createDirectory(at: root, withIntermediateDirectories: true)
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var mutableRoot = root
            try? mutableRoot.setResourceValues(values)
        }
        return root
    }

    /// `<root>/<ship>/<tier>ms`
    public static func variantDirectory(ship: String, chunkMs: Int) -> URL {
        rootDirectory()
            .appendingPathComponent(ship, isDirectory: true)
            .appendingPathComponent("\(chunkMs)ms", isDirectory: true)
    }

    /// True once a variant has been fully downloaded.
    public static func isInstalled(ship: String, chunkMs: Int) -> Bool {
        let sentinel = variantDirectory(ship: ship, chunkMs: chunkMs).appendingPathComponent(".complete")
        return FileManager.default.fileExists(atPath: sentinel.path)
    }

    // MARK: - Public API

    /// Downloads `<ship>/<chunkMs>ms` if not already installed and returns its directory.
    /// Idempotent and resumable: files already present at their expected size are skipped.
    /// Throws `NemotronModelDownloadError.variantNotAvailable` when the repository does not
    /// publish that ship/tier, so callers can fall back to another ship.
    public func ensureInstalled(
        ship: String,
        chunkMs: Int,
        onProgress: @escaping @Sendable (NemotronModelDownloadProgress) -> Void = { _ in }
    ) async throws -> URL {
        let destRoot = Self.variantDirectory(ship: ship, chunkMs: chunkMs)
        if Self.isInstalled(ship: ship, chunkMs: chunkMs) { return destRoot }

        onProgress(.zero)

        let tier = "\(chunkMs)ms"
        let files = try await fetchFileList(ship: ship, tier: tier)
        guard !files.isEmpty else {
            throw NemotronModelDownloadError.variantNotAvailable(ship: ship, tier: tier)
        }

        let totalBytes = files.reduce(Int64(0)) { $0 + max($1.size, 0) }
        var bytesCompleted: Int64 = 0
        let fm = FileManager.default
        let totalFiles = files.count
        try fm.createDirectory(at: destRoot, withIntermediateDirectories: true)

        for (idx, file) in files.enumerated() {
            try Task.checkCancellation()
            let destURL = destRoot.appendingPathComponent(file.relativePath)
            let fileIndex = idx + 1
            let fileName = file.relativePath

            if let attrs = try? fm.attributesOfItem(atPath: destURL.path),
                let size = attrs[.size] as? Int64, file.size > 0, size == file.size
            {
                bytesCompleted += size
                onProgress(
                    NemotronModelDownloadProgress(
                        bytesCompleted: bytesCompleted, bytesTotal: totalBytes,
                        currentFileIndex: fileIndex, totalFiles: totalFiles,
                        currentFileName: fileName))
                continue
            }

            try fm.createDirectory(
                at: destURL.deletingLastPathComponent(), withIntermediateDirectories: true)

            let url = Self.downloadURL(ship: ship, tier: tier, relativePath: file.relativePath)
            let fileStart = bytesCompleted
            try await downloadFile(from: url, to: destURL) { written in
                onProgress(
                    NemotronModelDownloadProgress(
                        bytesCompleted: fileStart + written, bytesTotal: totalBytes,
                        currentFileIndex: fileIndex, totalFiles: totalFiles,
                        currentFileName: fileName))
            }
            let finalSize =
                (try? fm.attributesOfItem(atPath: destURL.path)[.size] as? Int64) ?? file.size
            bytesCompleted = fileStart + max(finalSize, 0)
            onProgress(
                NemotronModelDownloadProgress(
                    bytesCompleted: bytesCompleted, bytesTotal: totalBytes,
                    currentFileIndex: fileIndex, totalFiles: totalFiles,
                    currentFileName: fileName))
        }

        try Data().write(to: destRoot.appendingPathComponent(".complete"), options: .atomic)
        return destRoot
    }

    // MARK: - Internal

    struct RemoteFile: Sendable {
        let relativePath: String
        let size: Int64
    }

    private static func downloadURL(ship: String, tier: String, relativePath: String) -> URL {
        var u = URL(string: "https://huggingface.co")!
            .appendingPathComponent(repoId)
            .appendingPathComponent("resolve")
            .appendingPathComponent(revision)
            .appendingPathComponent(ship)
            .appendingPathComponent(tier)
        for segment in relativePath.split(separator: "/") {
            u.appendPathComponent(String(segment))
        }
        return u
    }

    private struct TreeEntry: Decodable {
        let type: String
        let path: String
        let size: Int64?
    }

    /// Lists the files of `<ship>/<tier>` that the CoreML runtime needs: `metadata.json`,
    /// `tokenizer.json` and every `*.mlmodelc` tree, minus `coreai/` and minus the monolithic
    /// `encoder.mlmodelc` when the complete split encoder (pre-encode + 4 shards) is present —
    /// the runtime prefers the split path, so the monolithic one would be dead weight.
    /// Returns an empty array when the directory does not exist.
    private func fetchFileList(ship: String, tier: String) async throws -> [RemoteFile] {
        let u = URL(string: "https://huggingface.co/api/models")!
            .appendingPathComponent(Self.repoId)
            .appendingPathComponent("tree")
            .appendingPathComponent(Self.revision)
            .appendingPathComponent(ship)
            .appendingPathComponent(tier)
        var comps = URLComponents(url: u, resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        let apiURL = comps.url!

        var req = URLRequest(url: apiURL)
        req.httpMethod = "GET"
        req.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await urlSession().data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw NemotronModelDownloadError.invalidResponse("Non-HTTP response from \(apiURL)")
        }
        if http.statusCode == 404 { return [] }
        guard (200 ..< 300).contains(http.statusCode) else {
            throw NemotronModelDownloadError.httpStatus(
                http.statusCode, "Failed to list files at \(apiURL.path)")
        }

        let entries: [TreeEntry]
        do {
            entries = try JSONDecoder().decode([TreeEntry].self, from: data)
        } catch {
            throw NemotronModelDownloadError.invalidResponse(
                "Could not parse tree response: \(error.localizedDescription)")
        }

        let prefix = "\(ship)/\(tier)/"
        var all: [RemoteFile] = []
        for entry in entries where entry.type == "file" {
            guard entry.path.hasPrefix(prefix) else { continue }
            let rel = String(entry.path.dropFirst(prefix.count))
            guard !rel.isEmpty, !rel.hasPrefix("coreai/") else { continue }
            let top = rel.split(separator: "/").first.map(String.init) ?? rel
            let wanted = top == "metadata.json" || top == "tokenizer.json" || top.hasSuffix(".mlmodelc")
            guard wanted else { continue }
            all.append(RemoteFile(relativePath: rel, size: entry.size ?? 0))
        }

        let topLevel = Set(all.map { $0.relativePath.split(separator: "/").first.map(String.init) ?? "" })
        let hasSplitEncoder =
            topLevel.contains("encoder_pre_encode.mlmodelc")
            && (0 ..< 4).allSatisfy { topLevel.contains("encoder_shard_\($0).mlmodelc") }
        var out = hasSplitEncoder ? all.filter { !$0.relativePath.hasPrefix("encoder.mlmodelc/") } : all
        out.sort { $0.relativePath < $1.relativePath }
        return out
    }

    private func urlSession() -> URLSession {
        let config = URLSessionConfiguration.default
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 60 * 60
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }

    private func downloadFile(
        from url: URL,
        to destURL: URL,
        onBytes: @escaping @Sendable (Int64) -> Void
    ) async throws {
        let tmpURL = destURL.appendingPathExtension("tmp")
        let fm = FileManager.default
        if fm.fileExists(atPath: tmpURL.path) { try? fm.removeItem(at: tmpURL) }
        if fm.fileExists(atPath: destURL.path) { try? fm.removeItem(at: destURL) }
        fm.createFile(atPath: tmpURL.path, contents: nil)
        guard let handle = try? FileHandle(forWritingTo: tmpURL) else {
            throw NemotronModelDownloadError.networkError("Could not open \(tmpURL.path) for writing")
        }
        defer { try? handle.close() }

        var req = URLRequest(url: url)
        req.httpMethod = "GET"

        let (bytes, response) = try await urlSession().bytes(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw NemotronModelDownloadError.invalidResponse("Non-HTTP response from \(url.absoluteString)")
        }
        guard (200 ..< 300).contains(http.statusCode) else {
            throw NemotronModelDownloadError.httpStatus(
                http.statusCode, "Failed to download \(url.lastPathComponent)")
        }

        var buffer = Data()
        buffer.reserveCapacity(256 * 1024)
        var written: Int64 = 0
        var lastEmit = Date.distantPast
        let emitInterval: TimeInterval = 0.1

        for try await byte in bytes {
            try Task.checkCancellation()
            buffer.append(byte)
            if buffer.count >= 256 * 1024 {
                try handle.write(contentsOf: buffer)
                written += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                let now = Date()
                if now.timeIntervalSince(lastEmit) >= emitInterval {
                    onBytes(written)
                    lastEmit = now
                }
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            written += Int64(buffer.count)
        }
        try handle.close()
        onBytes(written)

        try fm.moveItem(at: tmpURL, to: destURL)
    }
}
