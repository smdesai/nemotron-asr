import Foundation
import SwiftUI

@MainActor
final class SentimentHighlighter: ObservableObject {
    @Published private(set) var document: SentimentDocument = .empty
    @Published private(set) var analyzedInput: SentimentInput?

    private let analyzer: SentimentAnalyzer
    private var latestInput: SentimentInput?

    init(analyzer: SentimentAnalyzer = SentimentAnalyzer()) {
        self.analyzer = analyzer
    }

    func update(_ input: SentimentInput?) async {
        latestInput = input
        guard let input else {
            document = .empty
            analyzedInput = nil
            return
        }
        if analyzedInput != input { analyzedInput = nil }

        guard !input.text.isEmpty else {
            document = .empty
            analyzedInput = input
            return
        }

        // Keep already-scored sentence prefixes visible while the mutable tail
        // waits for debounce. A recognizer revision invalidates only the spans
        // at and after the first changed sentence.
        document = document.rebased(onto: input.text)

        if !input.isFinal {
            do {
                try await Task.sleep(for: .milliseconds(500))
            } catch {
                return
            }
        }

        guard !Task.isCancelled else { return }
        let result = await analyzer.analyze(input)
        guard !Task.isCancelled, latestInput == input else { return }
        document = result
        analyzedInput = input
    }
}

struct SentimentPalette {
    let negative: Color
    let neutral: Color
    let positive: Color
    let mixed: Color
    let unavailable: Color
    let provisional: Color

    func color(for span: SentimentSpan) -> Color {
        if span.isProvisional { return provisional }

        let base: Color
        switch span.band {
        case .negative: base = negative
        case .neutral: base = neutral
        case .positive: base = positive
        case .unavailable: base = unavailable
        }

        guard let score = span.score, span.band != .neutral else { return base }
        return base.opacity(0.72 + min(1, abs(score)) * 0.28)
    }
}

struct SentimentPresentation {
    let label: String
    let systemImage: String
    let color: Color
}

extension SentimentDocument {
    func attributedString(fallback: String, palette: SentimentPalette) -> AttributedString {
        let displayedDocument = rebased(onto: fallback)

        var result = AttributedString()
        for span in displayedDocument.spans {
            var text = AttributedString(span.text)
            text.foregroundColor = palette.color(for: span)
            result.append(text)
        }
        return result
    }

    func accessibilityLabel(fallback: String) -> String {
        rebased(onto: fallback).spans.map { span in
            let spokenText = span.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !spokenText.isEmpty, !span.isProvisional else { return span.text }

            switch span.band {
            case .negative: return "Negative sentiment. \(spokenText)"
            case .neutral: return "Neutral sentiment. \(spokenText)"
            case .positive: return "Positive sentiment. \(spokenText)"
            case .unavailable: return spokenText
            }
        }.joined(separator: " ")
    }
}

extension SentimentSummary {
    func presentation(palette: SentimentPalette) -> SentimentPresentation? {
        switch self {
        case .unavailable:
            nil
        case .negative:
            SentimentPresentation(
                label: "Negative", systemImage: "arrow.down.right", color: palette.negative)
        case .neutral:
            SentimentPresentation(label: "Neutral", systemImage: "minus", color: palette.neutral)
        case .positive:
            SentimentPresentation(
                label: "Positive", systemImage: "arrow.up.right", color: palette.positive)
        case .mixed:
            SentimentPresentation(
                label: "Mixed", systemImage: "arrow.left.and.right", color: palette.mixed)
        }
    }
}
