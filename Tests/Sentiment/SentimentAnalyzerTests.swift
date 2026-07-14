import XCTest

final class SentimentAnalyzerTests: XCTestCase {
    func testBandThresholdsKeepAmbiguousScoresNeutral() {
        XCTAssertEqual(SentimentBand(score: -1), .negative)
        XCTAssertEqual(SentimentBand(score: -0.25), .negative)
        XCTAssertEqual(SentimentBand(score: -0.249), .neutral)
        XCTAssertEqual(SentimentBand(score: 0.249), .neutral)
        XCTAssertEqual(SentimentBand(score: 0.25), .positive)
        XCTAssertEqual(SentimentBand(score: 1), .positive)
    }

    func testStreamingAnalysisColorsCompletedSentenceAndLeavesTailProvisional() async {
        let analyzer = SentimentAnalyzer { text, _ in
            text.localizedCaseInsensitiveContains("great") ? 0.8 : -0.8
        }

        let input = SentimentInput(
            text: "This is great. But this is unfinished",
            languageCode: "en",
            isFinal: false
        )
        let document = await analyzer.analyze(input)

        XCTAssertEqual(document.spans.map(\.text).joined(), input.text)
        XCTAssertEqual(document.spans.count, 2)
        XCTAssertEqual(document.spans[0].band, .positive)
        XCTAssertFalse(document.spans[0].isProvisional)
        XCTAssertEqual(document.spans[1].band, .unavailable)
        XCTAssertTrue(document.spans[1].isProvisional)
    }

    func testFinalAnalysisScoresTheTrailingSentence() async {
        let analyzer = SentimentAnalyzer { text, _ in
            text.localizedCaseInsensitiveContains("great") ? 0.8 : -0.8
        }

        let document = await analyzer.analyze(
            SentimentInput(
                text: "This is great. But this is unfinished",
                languageCode: "en",
                isFinal: true
            )
        )

        XCTAssertEqual(document.spans.count, 2)
        XCTAssertEqual(document.spans[0].band, .positive)
        XCTAssertEqual(document.spans[1].band, .negative)
        XCTAssertFalse(document.spans[1].isProvisional)
        XCTAssertEqual(document.summary, .mixed)
    }

    func testMissingLanguageSupportIsUnavailableRatherThanNeutral() async {
        let analyzer = SentimentAnalyzer { _, _ in nil }

        let document = await analyzer.analyze(
            SentimentInput(text: "A complete sentence.", languageCode: "zz", isFinal: true)
        )

        XCTAssertEqual(document.spans.count, 1)
        XCTAssertEqual(document.spans[0].band, .unavailable)
        XCTAssertEqual(document.summary, .unavailable)
    }

    func testRebasingPreservesStableScoredPrefixAndInvalidatesOldSummary() async {
        let analyzer = SentimentAnalyzer { text, _ in
            text.localizedCaseInsensitiveContains("great") ? 0.8 : -0.8
        }
        let original = await analyzer.analyze(
            SentimentInput(
                text: "This is great. An unfinished thought",
                languageCode: "en",
                isFinal: false
            )
        )

        let appended = original.rebased(onto: "This is great. An unfinished thought grows")
        XCTAssertEqual(
            appended.spans.map(\.text).joined(), "This is great. An unfinished thought grows")
        XCTAssertEqual(appended.spans[0].band, .positive)
        XCTAssertTrue(appended.spans.last?.isProvisional == true)
        XCTAssertEqual(appended.summary, .positive)

        let revised = original.rebased(onto: "The recognizer replaced every word")
        XCTAssertEqual(revised.spans.map(\.text).joined(), "The recognizer replaced every word")
        XCTAssertEqual(revised.summary, .unavailable)
    }

    func testQuotedSentenceIsCompleteWhileStreaming() async {
        let analyzer = SentimentAnalyzer { _, _ in 0.8 }

        let document = await analyzer.analyze(
            SentimentInput(text: "\"This is great.\"", languageCode: "en", isFinal: false)
        )

        XCTAssertEqual(document.spans.count, 1)
        XCTAssertFalse(document.spans[0].isProvisional)
        XCTAssertEqual(document.spans[0].band, .positive)
    }

    func testNaturalLanguageScoresShortPositivePhraseWithExplicitEnglish() async {
        let document = await SentimentAnalyzer().analyze(
            SentimentInput(
                text: "I absolutely love this color",
                languageCode: "en",
                isFinal: true
            )
        )

        XCTAssertEqual(document.spans.count, 1)
        XCTAssertEqual(document.spans[0].band, .positive)
        XCTAssertEqual(document.summary, .positive)
    }

    func testWatchModelTokenizerAddsSpecialTokensAndPadding() throws {
        let tokenizer = try SentimentWordPieceTokenizer(vocabulary: [
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "I": 4,
            "hate": 5,
            "the": 6,
            "color": 7,
        ])

        let encoded = tokenizer.encode("I hate the color", maxLength: 8)

        XCTAssertEqual(encoded.inputIDs, [2, 4, 5, 6, 7, 3, 0, 0])
        XCTAssertEqual(encoded.attentionMask, [1, 1, 1, 1, 1, 1, 0, 0])
    }

    func testWatchModelWinningClassMapsToExistingSentimentScale() {
        XCTAssertEqual(
            SentimentModelScore.winningClass([0, 2, 1, 0, 0]),
            -0.5
        )
        XCTAssertEqual(
            SentimentModelScore.winningClass([0, 0, 3, 0, 0]),
            0
        )
        XCTAssertEqual(
            SentimentModelScore.winningClass([0, 0, 0, 0, 4]),
            1
        )
    }

    @MainActor
    func testDisablingHighlighterClearsAnalyzedSentiment() async {
        let input = SentimentInput(text: "I love this.", languageCode: "en", isFinal: true)
        let highlighter = SentimentHighlighter(
            analyzer: SentimentAnalyzer { _, _ in 0.8 }
        )

        await highlighter.update(input)
        XCTAssertEqual(highlighter.document.summary, .positive)
        XCTAssertEqual(highlighter.analyzedInput, input)

        await highlighter.update(nil)
        XCTAssertEqual(highlighter.document, .empty)
        XCTAssertNil(highlighter.analyzedInput)
    }

    @MainActor
    func testDisablingHighlighterRejectsInFlightResult() async {
        let input = SentimentInput(text: "I love this.", languageCode: "en", isFinal: false)
        let highlighter = SentimentHighlighter(
            analyzer: SentimentAnalyzer { _, _ in 0.8 }
        )
        let pendingUpdate = Task { await highlighter.update(input) }

        await Task.yield()
        await highlighter.update(nil)
        await pendingUpdate.value

        XCTAssertEqual(highlighter.document, .empty)
        XCTAssertNil(highlighter.analyzedInput)
    }

    func testContrastingClausesInOneSentenceProduceMixedSentiment() async {
        let analyzer = SentimentAnalyzer { text, _ in
            if text.localizedCaseInsensitiveContains("love") { return 0.8 }
            if text.localizedCaseInsensitiveContains("hate") { return -0.8 }
            return 0
        }
        let input = SentimentInput(
            text: "I love this color, but I hate the price.",
            languageCode: "en",
            isFinal: true
        )

        let document = await analyzer.analyze(input)

        XCTAssertEqual(document.spans.map(\.text).joined(), input.text)
        XCTAssertEqual(document.spans.map(\.band), [.positive, .negative])
        XCTAssertEqual(document.summary, .mixed)
    }

    func testSemicolonAndContrastAdverbProduceTwoExactSpans() async {
        let analyzer = SentimentAnalyzer { text, _ in
            if text.localizedCaseInsensitiveContains("love") { return 0.8 }
            if text.localizedCaseInsensitiveContains("hate") { return -0.8 }
            return 0
        }
        let input = SentimentInput(
            text: "I love the color; however, I hate the price.",
            languageCode: "en",
            isFinal: true
        )

        let document = await analyzer.analyze(input)

        XCTAssertEqual(document.spans.map(\.text).joined(), input.text)
        XCTAssertEqual(document.spans.count, 2)
        XCTAssertEqual(document.spans.map(\.band), [.positive, .negative])
        XCTAssertEqual(document.summary, .mixed)
    }

    func testLeadingContrastClauseSplitsAtItsComma() async {
        let analyzer = SentimentAnalyzer { text, _ in
            if text.localizedCaseInsensitiveContains("love") { return 0.8 }
            if text.localizedCaseInsensitiveContains("hate") { return -0.8 }
            return 0
        }
        let input = SentimentInput(
            text: "Although I love this color, I hate the price.",
            languageCode: "en",
            isFinal: true
        )

        let document = await analyzer.analyze(input)

        XCTAssertEqual(document.spans.map(\.text).joined(), input.text)
        XCTAssertEqual(document.spans.map(\.band), [.positive, .negative])
        XCTAssertEqual(document.summary, .mixed)
    }

    func testStreamingContrastScoresCompletedClauseAndLeavesTailProvisional() async {
        let analyzer = SentimentAnalyzer { text, _ in
            text.localizedCaseInsensitiveContains("great") ? 0.8 : -0.8
        }
        let input = SentimentInput(
            text: "This is great, but this is unfinished",
            languageCode: "en",
            isFinal: false
        )

        let document = await analyzer.analyze(input)

        XCTAssertEqual(document.spans.map(\.text).joined(), input.text)
        XCTAssertEqual(document.spans.count, 2)
        XCTAssertEqual(document.spans[0].band, .positive)
        XCTAssertFalse(document.spans[0].isProvisional)
        XCTAssertEqual(document.spans[1].band, .unavailable)
        XCTAssertTrue(document.spans[1].isProvisional)
        XCTAssertEqual(document.summary, .positive)
    }

    func testOrdinaryConjunctionsAndWordsContainingMarkersRemainOneSpan() async {
        let analyzer = SentimentAnalyzer { _, _ in 0.8 }
        let inputs = [
            "The butter is warm and the bread is fresh.",
            "I have not yet finished the transcription.",
            "I have yet to finish the transcription.",
            "I listened while walking home.",
            "It is anything but bad.",
            "It was good though.",
            "I however agree with the result.",
            "He said \"great; wonderful\" and smiled.",
            "The 2019–2020 period was excellent.",
            "No one but Alice liked it.",
            "I can but try.",
            "There was no choice but to leave.",
        ]

        for text in inputs {
            let document = await analyzer.analyze(
                SentimentInput(text: text, languageCode: "en", isFinal: true)
            )
            XCTAssertEqual(document.spans.map(\.text), [text])
        }
    }

    func testLeadingContrastIgnoresNestedPunctuation() async {
        let analyzer = SentimentAnalyzer { text, _ in
            if text.localizedCaseInsensitiveContains("good") { return 0.8 }
            if text.localizedCaseInsensitiveContains("hate") { return -0.8 }
            return 0
        }
        let input = SentimentInput(
            text: "Although she said \"good, fine\", I hate the outcome.",
            languageCode: "en",
            isFinal: true
        )

        let document = await analyzer.analyze(input)

        XCTAssertEqual(
            document.spans.map(\.text),
            ["Although she said \"good, fine\", ", "I hate the outcome."]
        )
        XCTAssertEqual(document.spans.map(\.band), [.positive, .negative])
        XCTAssertEqual(document.summary, .mixed)
    }

    func testEvenThoughLeadingClauseProducesMixedSentiment() async {
        let analyzer = SentimentAnalyzer { text, _ in
            if text.localizedCaseInsensitiveContains("love") { return 0.8 }
            if text.localizedCaseInsensitiveContains("hate") { return -0.8 }
            return 0
        }
        let input = SentimentInput(
            text: "Even though I love this color, I hate the price.",
            languageCode: "en",
            isFinal: true
        )

        let document = await analyzer.analyze(input)

        XCTAssertEqual(document.spans.map(\.text).joined(), input.text)
        XCTAssertEqual(document.spans.map(\.band), [.positive, .negative])
        XCTAssertEqual(document.summary, .mixed)
    }

    func testLeadingContrastClauseSkipsListCommas() async {
        let analyzer = SentimentAnalyzer { text, _ in
            if text.localizedCaseInsensitiveContains("good") { return 0.8 }
            if text.localizedCaseInsensitiveContains("hate") { return -0.8 }
            return 0
        }
        let input = SentimentInput(
            text: "Although red, blue, and green look good, I hate the price.",
            languageCode: "en",
            isFinal: true
        )

        let document = await analyzer.analyze(input)

        XCTAssertEqual(
            document.spans.map(\.text),
            ["Although red, blue, and green look good, ", "I hate the price."]
        )
        XCTAssertEqual(document.spans.map(\.band), [.positive, .negative])
        XCTAssertEqual(document.summary, .mixed)
    }
}
