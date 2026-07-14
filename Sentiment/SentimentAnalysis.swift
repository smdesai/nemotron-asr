import Foundation
import NaturalLanguage

struct SentimentInput: Hashable, Sendable {
    let text: String
    let languageCode: String?
    let isFinal: Bool
}

enum SentimentBand: String, Equatable, Sendable {
    case negative
    case neutral
    case positive
    case unavailable

    init(score: Double) {
        switch score {
        case ...(-0.25): self = .negative
        case 0.25...: self = .positive
        default: self = .neutral
        }
    }
}

enum SentimentSummary: Equatable, Sendable {
    case unavailable
    case negative
    case neutral
    case positive
    case mixed
}

struct SentimentSpan: Identifiable, Equatable, Sendable {
    let id: Int
    let text: String
    let score: Double?
    let band: SentimentBand
    let isProvisional: Bool
}

struct SentimentDocument: Equatable, Sendable {
    static let empty = SentimentDocument(
        spans: [],
        overallScore: nil,
        summary: .unavailable
    )

    let spans: [SentimentSpan]
    let overallScore: Double?
    let summary: SentimentSummary

    var text: String {
        spans.map(\.text).joined()
    }

    func rebased(onto newText: String) -> SentimentDocument {
        guard !newText.isEmpty else { return .empty }

        var preserved: [SentimentSpan] = []
        var cursor = newText.startIndex

        for span in spans where !span.isProvisional {
            guard newText[cursor...].hasPrefix(span.text) else { break }
            preserved.append(
                SentimentSpan(
                    id: preserved.count,
                    text: span.text,
                    score: span.score,
                    band: span.band,
                    isProvisional: false
                )
            )
            cursor = newText.index(cursor, offsetBy: span.text.count)
        }

        if cursor < newText.endIndex {
            preserved.append(
                SentimentSpan(
                    id: preserved.count,
                    text: String(newText[cursor...]),
                    score: nil,
                    band: .unavailable,
                    isProvisional: true
                )
            )
        }

        return Self.make(spans: preserved)
    }

    fileprivate static func make(spans: [SentimentSpan]) -> SentimentDocument {
        let scored = spans.compactMap { span -> (score: Double, weight: Double)? in
            guard let score = span.score else { return nil }
            let weight = Double(max(1, span.text.count))
            return (score, weight)
        }

        guard !scored.isEmpty else {
            return SentimentDocument(spans: spans, overallScore: nil, summary: .unavailable)
        }

        let weightedTotal = scored.reduce(0) { $0 + $1.score * $1.weight }
        let totalWeight = scored.reduce(0) { $0 + $1.weight }
        let overallScore = weightedTotal / totalWeight

        let hasPositive = spans.contains { $0.band == .positive }
        let hasNegative = spans.contains { $0.band == .negative }
        let summary: SentimentSummary
        if hasPositive && hasNegative {
            summary = .mixed
        } else {
            switch SentimentBand(score: overallScore) {
            case .negative: summary = .negative
            case .neutral: summary = .neutral
            case .positive: summary = .positive
            case .unavailable: summary = .unavailable
            }
        }

        return SentimentDocument(
            spans: spans,
            overallScore: overallScore,
            summary: summary
        )
    }
}

actor SentimentAnalyzer {
    typealias ScoreProvider = @Sendable (_ text: String, _ languageCode: String?) -> Double?

    private let scoreProvider: ScoreProvider
    private var scoreCache: [ScoreKey: CachedScore] = [:]

    init() {
        scoreProvider = { text, languageCode in
            let appleScore = NaturalLanguageSentimentScorer.score(
                text: text,
                languageCode: languageCode
            )
            #if os(watchOS)
            // The sentiment tag scheme is present in the watchOS SDK but can
            // return no tag on-device. Keep Apple as the primary scorer and
            // fall back to the bundled multilingual Core ML model.
            return appleScore ?? WatchCoreMLSentimentScorer.score(text: text)
            #else
            return appleScore
            #endif
        }
    }

    init(_ scoreProvider: @escaping ScoreProvider) {
        self.scoreProvider = scoreProvider
    }

    func analyze(_ input: SentimentInput) -> SentimentDocument {
        guard !input.text.isEmpty else { return .empty }

        let segments = Self.segments(in: input.text)
        guard !segments.isEmpty else {
            return SentimentDocument(
                spans: [
                    SentimentSpan(
                        id: 0,
                        text: input.text,
                        score: nil,
                        band: .unavailable,
                        isProvisional: !input.isFinal
                    )
                ],
                overallScore: nil,
                summary: .unavailable
            )
        }

        let spans = segments.enumerated().map { index, segment in
            let isTrailingSegment = index == segments.indices.last
            let isProvisional =
                isTrailingSegment
                && !input.isFinal
                && !Self.hasSentenceTerminator(segment)

            guard !isProvisional else {
                return SentimentSpan(
                    id: index,
                    text: segment,
                    score: nil,
                    band: .unavailable,
                    isProvisional: true
                )
            }

            let score = cachedScore(for: segment, languageCode: input.languageCode)
            return SentimentSpan(
                id: index,
                text: segment,
                score: score,
                band: score.map(SentimentBand.init(score:)) ?? .unavailable,
                isProvisional: false
            )
        }

        return SentimentDocument.make(spans: spans)
    }

    private static func segments(in text: String) -> [String] {
        sentences(in: text).flatMap(splitContrastingClauses)
    }

    private static func splitContrastingClauses(in sentence: String) -> [String] {
        guard !sentence.isEmpty else { return [] }

        var words: [(text: String, range: Range<String.Index>)] = []
        sentence.enumerateSubstrings(
            in: sentence.startIndex ..< sentence.endIndex,
            options: [.byWords, .substringNotRequired]
        ) { _, range, _, _ in
            words.append((sentence[range].lowercased(), range))
        }

        let punctuation = topLevelPunctuation(in: sentence)
        var boundaries: [String.Index] = [sentence.startIndex]
        boundaries.append(contentsOf: punctuation.separatorBoundaries)

        let contrastWords: Set<String> = [
            "although", "but", "however", "though", "whereas", "while", "yet",
        ]
        let restrictiveButPrefixes: Set<String> = [
            "all", "anybody", "anyone", "anything", "everybody", "everyone", "everything",
            "nobody", "none", "nothing", "somebody", "someone",
        ]
        let modalVerbs: Set<String> = [
            "can", "could", "may", "might", "must", "shall", "should", "will", "would",
        ]

        for index in words.indices.dropFirst() {
            let token = words[index]
            guard contrastWords.contains(token.text),
                isTopLevel(token.range.lowerBound, in: sentence)
            else { continue }

            let previousWord = words[index - 1].text
            let nextWord = index + 1 < words.count ? words[index + 1].text : nil
            switch token.text {
            case "but":
                let wordBeforePrevious = index > 1 ? words[index - 2].text : nil
                let followsNoOne = previousWord == "one" && wordBeforePrevious == "no"
                if restrictiveButPrefixes.contains(previousWord)
                    || modalVerbs.contains(previousWord)
                    || followsNoOne
                    || nextWord == "also"
                    || nextWord == "for"
                    || nextWord == "to"
                {
                    continue
                }
            case "however", "though", "while", "yet":
                guard
                    hasTopLevelContrastPunctuation(
                        before: token.range.lowerBound,
                        in: sentence
                    )
                else { continue }
                if token.text == "though", previousWord == "even" { continue }
            default:
                break
            }

            boundaries.append(token.range.lowerBound)
        }

        let leadingContrastWords: Set<String> = ["although", "though", "whereas", "while"]
        let hasLeadingContrast =
            words.first.map { leadingContrastWords.contains($0.text) } == true
            || (words.count > 1 && words[0].text == "even" && words[1].text == "though")
        let explicitMainClauseStarters: Set<String> = [
            "a", "an", "he", "her", "his", "i", "it", "my", "our", "she", "that", "the",
            "their", "there", "these", "they", "this", "those", "we", "you", "your",
        ]
        let leadingClauseComma =
            punctuation.commas.first { comma in
                guard let nextWord = words.first(where: { $0.range.lowerBound > comma }) else {
                    return false
                }
                return explicitMainClauseStarters.contains(nextWord.text)
            } ?? punctuation.commas.first
        if hasLeadingContrast,
            let comma = leadingClauseComma
        {
            var boundary = sentence.index(after: comma)
            while boundary < sentence.endIndex, sentence[boundary].isWhitespace {
                boundary = sentence.index(after: boundary)
            }
            if boundary < sentence.endIndex { boundaries.append(boundary) }
        }
        boundaries.append(sentence.endIndex)

        boundaries = Array(Set(boundaries)).sorted()

        return zip(boundaries, boundaries.dropFirst()).map { lower, upper in
            String(sentence[lower ..< upper])
        }
    }

    private static func topLevelPunctuation(in text: String) -> ClausePunctuation {
        var commas: [String.Index] = []
        var separatorBoundaries: [String.Index] = []
        var parenthesisDepth = 0
        var insideStraightQuote = false
        var insideCurlyQuote = false
        var cursor = text.startIndex

        while cursor < text.endIndex {
            let character = text[cursor]
            let next = text.index(after: cursor)

            if character == "\"" {
                insideStraightQuote.toggle()
                cursor = next
                continue
            }
            if character == "“" {
                insideCurlyQuote = true
                cursor = next
                continue
            }
            if character == "”" {
                insideCurlyQuote = false
                cursor = next
                continue
            }
            guard !insideStraightQuote, !insideCurlyQuote else {
                cursor = next
                continue
            }

            if "([{".contains(character) {
                parenthesisDepth += 1
            } else if ")]}".contains(character) {
                parenthesisDepth = max(0, parenthesisDepth - 1)
            } else if parenthesisDepth == 0 {
                if character == "," { commas.append(cursor) }
                if ";—".contains(character) {
                    var boundary = next
                    while boundary < text.endIndex, text[boundary].isWhitespace {
                        boundary = text.index(after: boundary)
                    }
                    if boundary < text.endIndex { separatorBoundaries.append(boundary) }
                }
            }
            cursor = next
        }

        return ClausePunctuation(
            commas: commas,
            separatorBoundaries: separatorBoundaries
        )
    }

    private static func hasTopLevelContrastPunctuation(
        before index: String.Index,
        in text: String
    ) -> Bool {
        var cursor = index
        while cursor > text.startIndex {
            let previous = text.index(before: cursor)
            if !text[previous].isWhitespace {
                return ",;—".contains(text[previous]) && isTopLevel(previous, in: text)
            }
            cursor = previous
        }
        return false
    }

    private static func isTopLevel(_ index: String.Index, in text: String) -> Bool {
        var parenthesisDepth = 0
        var insideStraightQuote = false
        var insideCurlyQuote = false
        var cursor = text.startIndex

        while cursor < index {
            let character = text[cursor]
            if character == "\"" {
                insideStraightQuote.toggle()
            } else if character == "“" {
                insideCurlyQuote = true
            } else if character == "”" {
                insideCurlyQuote = false
            } else if !insideStraightQuote, !insideCurlyQuote {
                if "([{".contains(character) {
                    parenthesisDepth += 1
                } else if ")]}".contains(character) {
                    parenthesisDepth = max(0, parenthesisDepth - 1)
                }
            }
            cursor = text.index(after: cursor)
        }

        return parenthesisDepth == 0 && !insideStraightQuote && !insideCurlyQuote
    }

    private struct ClausePunctuation {
        let commas: [String.Index]
        let separatorBoundaries: [String.Index]
    }

    private func cachedScore(for sentence: String, languageCode: String?) -> Double? {
        let normalizedText =
            sentence
            .split(whereSeparator: \Character.isWhitespace)
            .joined(separator: " ")
        let key = ScoreKey(text: normalizedText, languageCode: languageCode)

        if let cached = scoreCache[key] { return cached.value }

        let score = scoreProvider(sentence, languageCode)
        scoreCache[key] = score.map(CachedScore.score) ?? .unavailable
        return score
    }

    private static func sentences(in text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text

        var sentences: [String] = []
        var cursor = text.startIndex
        tokenizer.enumerateTokens(in: text.startIndex ..< text.endIndex) { range, _ in
            guard cursor < range.upperBound else { return true }
            sentences.append(String(text[cursor ..< range.upperBound]))
            cursor = range.upperBound
            return true
        }

        if cursor < text.endIndex {
            let remainder = String(text[cursor ..< text.endIndex])
            if sentences.isEmpty {
                sentences.append(remainder)
            } else {
                sentences[sentences.count - 1].append(remainder)
            }
        }

        return sentences
    }

    private static func hasSentenceTerminator(_ sentence: String) -> Bool {
        let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
        let closingCharacters = "\"'”’»)]}"
        for character in trimmed.reversed() {
            if closingCharacters.contains(character) { continue }
            return ".!?…。！？".contains(character)
        }
        return false
    }

    private struct ScoreKey: Hashable {
        let text: String
        let languageCode: String?
    }

    private enum CachedScore {
        case score(Double)
        case unavailable

        var value: Double? {
            switch self {
            case .score(let score): score
            case .unavailable: nil
            }
        }
    }
}

private enum NaturalLanguageSentimentScorer {
    static func score(text: String, languageCode: String?) -> Double? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = trimmed

        let language = languageCode.map(NLLanguage.init(rawValue:)) ?? tagger.dominantLanguage
        if let language {
            let schemes = NLTagger.availableTagSchemes(for: .paragraph, language: language)
            guard schemes.contains(.sentimentScore) else { return nil }
            tagger.setLanguage(language, range: trimmed.startIndex ..< trimmed.endIndex)
        }

        let (tag, _) = tagger.tag(
            at: trimmed.startIndex,
            unit: .paragraph,
            scheme: .sentimentScore
        )
        guard let rawValue = tag?.rawValue, let score = Double(rawValue) else { return nil }
        return min(1, max(-1, score))
    }
}
