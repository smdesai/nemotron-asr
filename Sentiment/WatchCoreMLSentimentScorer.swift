import Foundation

struct SentimentEncodedText: Sendable, Equatable {
    let inputIDs: [Int]
    let attentionMask: [Int]
}

enum SentimentTokenizerError: Error {
    case unreadableVocabulary(URL)
    case missingSpecialToken(String)
}

struct SentimentWordPieceTokenizer: Sendable {
    private let vocabulary: [String: Int]
    private let unknownID: Int
    private let classifierID: Int
    private let separatorID: Int
    private let paddingID: Int
    private let specialTokens: Set<String> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    init(vocabulary: [String: Int]) throws {
        self.vocabulary = vocabulary
        guard let unknownID = vocabulary["[UNK]"] else {
            throw SentimentTokenizerError.missingSpecialToken("[UNK]")
        }
        guard let classifierID = vocabulary["[CLS]"] else {
            throw SentimentTokenizerError.missingSpecialToken("[CLS]")
        }
        guard let separatorID = vocabulary["[SEP]"] else {
            throw SentimentTokenizerError.missingSpecialToken("[SEP]")
        }
        guard let paddingID = vocabulary["[PAD]"] else {
            throw SentimentTokenizerError.missingSpecialToken("[PAD]")
        }
        self.unknownID = unknownID
        self.classifierID = classifierID
        self.separatorID = separatorID
        self.paddingID = paddingID
    }

    init(vocabularyURL: URL) throws {
        guard let contents = try? String(contentsOf: vocabularyURL, encoding: .utf8) else {
            throw SentimentTokenizerError.unreadableVocabulary(vocabularyURL)
        }
        var vocabulary: [String: Int] = [:]
        for (index, token) in contents.split(separator: "\n", omittingEmptySubsequences: false)
            .enumerated()
        {
            vocabulary[String(token)] = index
        }
        try self.init(vocabulary: vocabulary)
    }

    func encode(_ text: String, maxLength: Int) -> SentimentEncodedText {
        precondition(maxLength >= 2)
        let contentIDs = basicTokens(text).flatMap(wordPieceIDs).prefix(maxLength - 2)
        var ids = [classifierID]
        ids.append(contentsOf: contentIDs)
        ids.append(separatorID)
        var mask = Array(repeating: 1, count: ids.count)
        if ids.count < maxLength {
            let padding = maxLength - ids.count
            ids.append(contentsOf: repeatElement(paddingID, count: padding))
            mask.append(contentsOf: repeatElement(0, count: padding))
        }
        return SentimentEncodedText(inputIDs: ids, attentionMask: mask)
    }

    private func basicTokens(_ text: String) -> [String] {
        splitOnSpecialTokens(text).flatMap { segment, isSpecial in
            isSpecial ? [segment] : tokenizeOrdinaryText(segment)
        }
    }

    private func splitOnSpecialTokens(_ text: String) -> [(String, Bool)] {
        var segments: [(String, Bool)] = [(text, false)]
        for special in specialTokens.sorted() {
            segments = segments.flatMap { segment, alreadySpecial in
                if alreadySpecial { return [(segment, true)] }
                var result: [(String, Bool)] = []
                var remainder = segment[...]
                while let range = remainder.range(of: special) {
                    if range.lowerBound != remainder.startIndex {
                        result.append((String(remainder[..<range.lowerBound]), false))
                    }
                    result.append((special, true))
                    remainder = remainder[range.upperBound...]
                }
                if !remainder.isEmpty { result.append((String(remainder), false)) }
                return result
            }
        }
        return segments
    }

    private func tokenizeOrdinaryText(_ text: String) -> [String] {
        var cleaned = ""
        for scalar in text.unicodeScalars {
            if scalar.value == 0 || scalar.value == 0xFFFD || isControl(scalar) { continue }
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                cleaned.append(" ")
            } else if isChinese(scalar.value) {
                cleaned.append(" ")
                cleaned.unicodeScalars.append(scalar)
                cleaned.append(" ")
            } else {
                cleaned.unicodeScalars.append(scalar)
            }
        }
        return cleaned.split(whereSeparator: { $0.isWhitespace }).flatMap { token -> [String] in
            let value = String(token)
            var pieces: [String] = []
            var current = ""
            for scalar in value.unicodeScalars {
                if CharacterSet.punctuationCharacters.contains(scalar) {
                    if !current.isEmpty {
                        pieces.append(current)
                        current = ""
                    }
                    pieces.append(String(scalar))
                } else {
                    current.unicodeScalars.append(scalar)
                }
            }
            if !current.isEmpty { pieces.append(current) }
            return pieces
        }
    }

    private func wordPieceIDs(_ token: String) -> [Int] {
        if let exact = vocabulary[token] { return [exact] }
        let scalars = Array(token.unicodeScalars)
        if scalars.count > 100 { return [unknownID] }
        var result: [Int] = []
        var start = 0
        while start < scalars.count {
            var end = scalars.count
            var found: (Int, Int)?
            while start < end {
                let text = String(scalars[start ..< end].map { Character(String($0)) })
                let piece = start == 0 ? text : "##" + text
                if let identifier = vocabulary[piece] {
                    found = (identifier, end)
                    break
                }
                end -= 1
            }
            guard let (identifier, next) = found else { return [unknownID] }
            result.append(identifier)
            start = next
        }
        return result
    }

    private func isControl(_ scalar: UnicodeScalar) -> Bool {
        CharacterSet.controlCharacters.contains(scalar)
            && !CharacterSet.whitespacesAndNewlines.contains(scalar)
    }

    private func isChinese(_ value: UInt32) -> Bool {
        (0x4E00 ... 0x9FFF).contains(value)
            || (0x3400 ... 0x4DBF).contains(value)
            || (0x20000 ... 0x2A6DF).contains(value)
            || (0x2A700 ... 0x2B73F).contains(value)
            || (0x2B740 ... 0x2B81F).contains(value)
            || (0x2B820 ... 0x2CEAF).contains(value)
            || (0xF900 ... 0xFAFF).contains(value)
            || (0x2F800 ... 0x2FA1F).contains(value)
    }
}

private enum SentimentModelClass: Int, CaseIterable {
    case veryNegative
    case negative
    case neutral
    case positive
    case veryPositive

    var score: Double {
        switch self {
        case .veryNegative: -1
        case .negative: -0.5
        case .neutral: 0
        case .positive: 0.5
        case .veryPositive: 1
        }
    }
}

enum SentimentModelScore {
    static func winningClass(_ logits: [Float]) -> Double? {
        let classes = SentimentModelClass.allCases
        guard logits.count == classes.count,
            let classID = logits.indices.max(by: { logits[$0] < logits[$1] }),
            let sentimentClass = SentimentModelClass(rawValue: classID)
        else { return nil }
        return sentimentClass.score
    }
}

#if os(watchOS)
import CoreML

private enum WatchCoreMLSentimentError: Error {
    case missingResource(String)
    case missingLogits
}

final class WatchCoreMLSentimentScorer: @unchecked Sendable {
    private static let sequenceLength = 128
    private static let logger = AppLogger(category: "SentimentCoreML")

    private static let shared: WatchCoreMLSentimentScorer? = {
        do {
            return try WatchCoreMLSentimentScorer()
        } catch {
            logger.error("Failed to load bundled sentiment model: \(error)")
            return nil
        }
    }()

    private let model: MLModel
    private let tokenizer: SentimentWordPieceTokenizer

    static func score(text: String) -> Double? {
        guard let shared else { return nil }
        do {
            return try shared.score(text: text)
        } catch {
            logger.error("Sentiment model prediction failed: \(error)")
            return nil
        }
    }

    private init() throws {
        guard
            let modelURL = Bundle.main.url(
                forResource: "baseline-t128-fp16",
                withExtension: "mlmodelc",
                subdirectory: "NemotronWatchSentiment"
            )
        else {
            throw WatchCoreMLSentimentError.missingResource("baseline-t128-fp16.mlmodelc")
        }
        guard
            let vocabularyURL = Bundle.main.url(
                forResource: "vocab",
                withExtension: "txt",
                subdirectory: "NemotronWatchSentiment"
            )
        else {
            throw WatchCoreMLSentimentError.missingResource("vocab.txt")
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
        tokenizer = try SentimentWordPieceTokenizer(vocabularyURL: vocabularyURL)
    }

    private func score(text: String) throws -> Double? {
        let encoded = tokenizer.encode(text, maxLength: Self.sequenceLength)
        let inputIDs = try MLMultiArray(
            shape: [1, NSNumber(value: Self.sequenceLength)],
            dataType: .int32
        )
        let attentionMask = try MLMultiArray(
            shape: [1, NSNumber(value: Self.sequenceLength)],
            dataType: .int32
        )
        for index in 0 ..< Self.sequenceLength {
            inputIDs[index] = NSNumber(value: encoded.inputIDs[index])
            attentionMask[index] = NSNumber(value: encoded.attentionMask[index])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
        ])
        let output = try model.prediction(from: input)
        guard let values = output.featureValue(for: "logits")?.multiArrayValue else {
            throw WatchCoreMLSentimentError.missingLogits
        }

        let logits = SentimentModelClass.allCases.map { values[$0.rawValue].floatValue }
        return SentimentModelScore.winningClass(logits)
    }
}
#endif
