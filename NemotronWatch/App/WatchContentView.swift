import SwiftUI

/// watchOS UI for the on-device transcription pipeline: a gradient backdrop,
/// a backend badge (with live per-chunk timing while listening), an
/// auto-scrolling transcript card, a level-reactive record control, and
/// share / clear actions for the finished transcript.
struct WatchContentView: View {

    @StateObject private var manager = WatchASRManager()
    @StateObject private var sentiment = SentimentHighlighter()
    @AppStorage("sentimentAnalysisEnabled") private var sentimentAnalysisEnabled = true

    /// Accent gradient shared with the iOS app's look (indigo → mint).
    private static let accent = LinearGradient(
        colors: [.indigo, .mint], startPoint: .topLeading, endPoint: .bottomTrailing)

    var body: some View {
        ZStack {
            backdrop
            content
        }
        .task { await manager.load() }
        .task(id: sentimentRequest) {
            await sentiment.update(sentimentRequest)
        }
    }

    // MARK: - Phases

    @ViewBuilder
    private var content: some View {
        switch manager.phase {
        case .loading:
            VStack(spacing: 10) {
                ProgressView()
                Text("Loading models…")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                backendBadge
            }
            .padding()

        case .failed(let message):
            VStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title3)
                    .foregroundStyle(.red)
                ScrollView {
                    Text(message)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.leading)
                }
                .frame(maxHeight: 70)
                Button {
                    Task { await manager.load() }
                } label: {
                    Label("Retry", systemImage: "arrow.clockwise")
                        .font(.footnote.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .tint(.indigo)
            }
            .padding(.horizontal, 8)

        default:
            mainView
        }
    }

    // MARK: - Main screen

    private var mainView: some View {
        VStack(spacing: 5) {
            HStack {
                backendBadge
                sentimentToggle
                Spacer()
                if manager.isTransitioning {
                    ProgressView()
                        .controlSize(.mini)
                } else if manager.isListening {
                    listeningIndicator
                } else if let presentation = sentimentSummaryPresentation {
                    sentimentBadge(presentation)
                }
            }

            transcriptCard

            controls
        }
        .padding(.horizontal, 4)
    }

    /// Backend capsule — names the inference stack driving transcription.
    /// (CoreML for now: the watchOS 27 beta's Core AI compiler has no m11
    /// SoC backend, so the Core AI path is parked until a later beta.)
    private var backendBadge: some View {
        HStack(spacing: 3) {
            Image(systemName: "cpu")
                .font(.system(size: 9, weight: .bold))
            Text("CoreML")
                .font(.system(size: 10, weight: .semibold))
        }
        .foregroundStyle(Self.accent)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(Capsule().fill(.white.opacity(0.08)))
        .overlay(Capsule().stroke(.white.opacity(0.12), lineWidth: 0.5))
    }

    /// Animated waveform + live per-chunk on-device inference time.
    private var listeningIndicator: some View {
        HStack(spacing: 4) {
            Image(systemName: "waveform")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.red)
                .symbolEffect(.variableColor.iterative, options: .repeating)
            if let seconds = manager.lastChunkSeconds {
                Text(String(format: "%.1fs", seconds))
                    .font(.system(size: 10, weight: .medium).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var sentimentToggle: some View {
        HStack(spacing: 2) {
            Image(systemName: "face.smiling")
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(sentimentAnalysisEnabled ? .mint : .secondary)
            Toggle("Sentiment analysis", isOn: $sentimentAnalysisEnabled)
                .labelsHidden()
                .controlSize(.mini)
                .tint(.mint)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Sentiment analysis")
        .accessibilityValue(sentimentAnalysisEnabled ? "On" : "Off")
    }

    private var transcriptCard: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Group {
                    if manager.transcript.isEmpty {
                        Text(
                            manager.isListening
                                ? "Listening…"
                                : "Tap the mic and start speaking."
                        )
                        .font(.footnote)
                        .foregroundStyle(.tertiary)
                    } else {
                        Text(sentimentTranscript)
                            .font(.footnote)
                            .accessibilityLabel(sentimentAccessibilityLabel)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
                .id("end")
            }
            .onChange(of: manager.transcript) { _, _ in
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo("end", anchor: .bottom)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(RoundedRectangle(cornerRadius: 10).fill(.white.opacity(0.06)))
    }

    // MARK: - Controls

    private var hasShareableTranscript: Bool {
        !manager.transcript.isEmpty && !manager.isListening && !manager.isTransitioning
    }

    private var sentimentInput: SentimentInput {
        SentimentInput(
            text: manager.transcript,
            // Keep Watch sentiment deterministic while language-specific
            // Natural Language availability is being validated on-device.
            languageCode: "en",
            isFinal: !manager.isListening && !manager.isTransitioning
        )
    }

    private var sentimentRequest: SentimentInput? {
        sentimentAnalysisEnabled ? sentimentInput : nil
    }

    private var sentimentTranscript: AttributedString {
        guard sentimentAnalysisEnabled else {
            return AttributedString(manager.transcript)
        }
        return sentiment.document.attributedString(
            fallback: manager.transcript,
            palette: sentimentPalette
        )
    }

    private var sentimentPalette: SentimentPalette {
        SentimentPalette(
            negative: Color(red: 1, green: 0.48, blue: 0.36),
            neutral: .white,
            positive: .mint,
            mixed: .indigo,
            unavailable: .secondary,
            provisional: .white
        )
    }

    private var sentimentSummaryPresentation: SentimentPresentation? {
        guard sentimentAnalysisEnabled,
            !manager.transcript.isEmpty,
            sentiment.analyzedInput == sentimentInput
        else { return nil }
        return sentiment.document.summary.presentation(palette: sentimentPalette)
            ?? SentimentPresentation(
                label: "Unavailable",
                systemImage: "questionmark",
                color: sentimentPalette.unavailable
            )
    }

    private var sentimentAccessibilityLabel: String {
        guard sentimentAnalysisEnabled else { return manager.transcript }
        return sentiment.document.accessibilityLabel(fallback: manager.transcript)
    }

    private func sentimentBadge(_ presentation: SentimentPresentation) -> some View {
        Label(presentation.label, systemImage: presentation.systemImage)
            .font(.system(size: 9, weight: .semibold))
            .foregroundStyle(presentation.color)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(presentation.color.opacity(0.14), in: Capsule())
            .accessibilityLabel("Overall sentiment: \(presentation.label)")
    }

    private var controls: some View {
        HStack {
            ShareLink(item: manager.transcript) {
                Image(systemName: "square.and.arrow.up")
                    .font(.system(size: 14, weight: .semibold))
                    .frame(width: 34, height: 34)
                    .background(Circle().fill(.white.opacity(0.08)))
            }
            .buttonStyle(.plain)
            .disabled(!hasShareableTranscript)
            .opacity(hasShareableTranscript ? 1 : 0.35)

            Spacer()
            recordButton
            Spacer()

            Button {
                manager.clearTranscript()
            } label: {
                Image(systemName: "trash")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(width: 34, height: 34)
                    .background(Circle().fill(.white.opacity(0.08)))
            }
            .buttonStyle(.plain)
            .disabled(!hasShareableTranscript)
            .opacity(hasShareableTranscript ? 1 : 0.35)
        }
    }

    private var recordButton: some View {
        Button(action: toggle) {
            ZStack {
                if manager.isListening {
                    // Mic-level-reactive halo.
                    Circle()
                        .stroke(Color.red.opacity(0.45), lineWidth: 2)
                        .scaleEffect(1.1 + CGFloat(manager.micLevel) * 0.5)
                        .animation(.easeOut(duration: 0.12), value: manager.micLevel)
                }
                Circle()
                    .fill(
                        manager.isListening ? AnyShapeStyle(Color.red) : AnyShapeStyle(Self.accent))
                if manager.isTransitioning {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: manager.isListening ? "stop.fill" : "mic.fill")
                        .font(.system(size: 17, weight: .bold))
                        .foregroundStyle(.white)
                }
            }
            .frame(width: 46, height: 46)
        }
        .buttonStyle(.plain)
        .disabled(manager.isTransitioning || (!manager.isReady && !manager.isListening))
    }

    private var backdrop: some View {
        LinearGradient(
            colors: [
                Color(red: 0.08, green: 0.08, blue: 0.17),
                Color(red: 0.02, green: 0.02, blue: 0.05),
            ],
            startPoint: .top, endPoint: .bottom
        )
        .ignoresSafeArea()
    }

    private func toggle() {
        Task {
            if manager.isListening {
                await manager.stop()
            } else {
                await manager.start()
            }
        }
    }
}
