import Foundation

/// Flat `{"id": "piece"}` SentencePiece vocabulary decoder.
public final class Tokenizer: Sendable {
    private let idToToken: [Int: String]

    public init(vocabPath: URL) throws {
        let data = try Data(contentsOf: vocabPath)
        let json = try JSONSerialization.jsonObject(with: data, options: []) as! [String: String]
        var idToToken: [Int: String] = [:]
        idToToken.reserveCapacity(json.count)
        for (key, value) in json {
            if let id = Int(key) {
                idToToken[id] = value
            }
        }
        self.idToToken = idToToken
    }

    public func decode(ids: [Int]) -> String {
        var text = ""
        for id in ids {
            if let token = idToToken[id] {
                text += token
            }
        }
        // Replace SentencePiece word boundary marker with space, then trim
        return text.replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    /// Raw SentencePiece piece for a token id, or `nil` if not in the vocab.
    public func piece(forId id: Int) -> String? {
        idToToken[id]
    }
}
