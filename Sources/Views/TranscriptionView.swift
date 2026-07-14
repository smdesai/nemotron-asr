import SwiftUI
import UniformTypeIdentifiers

struct TranscriptionView: View {
    @EnvironmentObject var engine: TranscriptionEngine
    @EnvironmentObject var settings: AppSettings
    @Binding var showSettings: Bool
    @StateObject private var sentiment = SentimentHighlighter()

    enum InputMode: String, CaseIterable, Identifiable {
        case microphone, file
        var id: String { rawValue }
        var label: String { self == .microphone ? "Microphone" : "Audio File" }
        var icon: String { self == .microphone ? "mic.fill" : "waveform" }
    }

    @State private var mode: InputMode = .microphone
    @State private var showFileImporter = false

    var body: some View {
        VStack(spacing: 14) {
            header
            statusStrip
            transcriptCard
                .frame(maxHeight: .infinity)  // grow to fill freed space

            // Microphone mode shows the waveform + record button. File mode
            // needs no extra controls — tapping the Audio File tab opens the
            // document picker directly — so the transcript fills the space.
            if mode == .microphone {
                micControls
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
        .safeAreaInset(edge: .bottom) { bottomBar }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.audio, .mpeg4Audio, .wav, .mp3, .aiff],
            allowsMultipleSelection: false
        ) { result in
            handleFileSelection(result)
        }
        .task {
            // Kick off model preparation on first appearance so the first
            // transcription is instant.
            await engine.prepareModelIfNeeded()
        }
        .task(id: sentimentRequest) {
            await sentiment.update(sentimentRequest)
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Nemotron ASR")
                    .font(.system(.title2, design: .rounded).weight(.bold))
                Text("On-device multilingual speech-to-text")
                    .font(.caption)
                    .foregroundStyle(Theme.secondaryText)
            }
            Spacer()
            Button {
                showSettings = true
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.white)
                    .frame(width: 42, height: 42)
                    .background(Circle().fill(Theme.cardFill))
                    .overlay(Circle().stroke(Theme.cardStroke, lineWidth: 1))
            }
        }
    }

    // MARK: Bottom tab bar

    /// Pinned input-mode switcher. Selecting the **Audio File** tab opens the
    /// document picker directly (no separate "Choose Audio File" button).
    private var bottomBar: some View {
        HStack(spacing: 8) {
            ForEach(InputMode.allCases) { m in
                Button {
                    selectMode(m)
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: m.icon)
                            .font(.system(size: 15, weight: .semibold))
                            .symbolVariant(mode == m ? .fill : .none)
                        Text(m.label)
                            .font(.caption.weight(.semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 7)
                    .foregroundStyle(
                        mode == m
                            ? AnyShapeStyle(Theme.brandGradient)
                            : AnyShapeStyle(Theme.secondaryText)
                    )
                    .background(
                        RoundedRectangle(cornerRadius: 13, style: .continuous)
                            .fill(mode == m ? Theme.cardFill : .clear)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(engine.isBusy && mode != m)
            }
        }
        .padding(5)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(
                Theme.cardStroke, lineWidth: 1)
        )
        .padding(.horizontal, 20)
        .padding(.bottom, 6)
    }

    /// Switch tabs. Tapping "Audio File" *is* the action — it always opens the
    /// document picker, so there's no separate "Choose Audio File" button.
    private func selectMode(_ m: InputMode) {
        guard !engine.isBusy else { return }
        withAnimation(.snappy) { mode = m }
        if m == .file {
            showFileImporter = true
        }
    }

    // MARK: Status strip (language + chunk + stats)

    private var statusStrip: some View {
        HStack(spacing: 8) {
            InfoChip(
                systemImage: "globe",
                text: settings.languageCode == nil ? "Auto-detect" : settings.language.name,
                tint: Theme.aurora1
            )
            InfoChip(
                systemImage: "waveform.path", text: settings.chunkSize.label, tint: Theme.aurora2)
            if let rtfx = engine.lastRTFx {
                InfoChip(
                    systemImage: "speedometer", text: String(format: "%.1fx", rtfx),
                    tint: Theme.aurora3)
            }
            if let detected = engine.detectedLanguage, settings.languageCode == nil {
                InfoChip(systemImage: "checkmark.seal.fill", text: detected, tint: Theme.aurora3)
            }
            Spacer(minLength: 0)
        }
    }

    // MARK: Transcript

    private var transcriptCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Transcript", systemImage: "text.quote")
                    .font(.headline)
                if engine.isStreaming {
                    LiveBadge()
                }
                if let presentation = sentimentSummaryPresentation {
                    Label(presentation.label, systemImage: presentation.systemImage)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(presentation.color)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 4)
                        .background(presentation.color.opacity(0.12), in: Capsule())
                        .accessibilityLabel("Overall sentiment: \(presentation.label)")
                }
                Spacer()
                if !engine.transcript.isEmpty {
                    Button {
                        UIPasteboard.general.string = engine.transcript
                    } label: {
                        Image(systemName: "doc.on.doc")
                    }
                    .tint(Theme.secondaryText)
                    Button {
                        engine.clearTranscript()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .tint(Theme.secondaryText)
                }
            }

            ScrollViewReader { proxy in
                ScrollView {
                    Text(sentimentTranscript)
                        .font(.system(.body, design: .rounded))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .accessibilityLabel(sentimentAccessibilityLabel)
                        .id("transcriptEnd")
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .onChange(of: engine.transcript) { _, _ in
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo("transcriptEnd", anchor: .bottom)
                    }
                }
            }

            if engine.phase == .transcribingFile {
                ProgressView(value: engine.fileProgress)
                    .tint(Theme.aurora2)
            }
        }
        .glassCard()
    }

    private var displayTranscript: String {
        if engine.transcript.isEmpty {
            switch mode {
            case .microphone:
                return
                    "Tap the mic and start speaking. Your words appear here live as they're transcribed."
            case .file:
                return
                    "Import an audio file to transcribe it. Choose streamed or final output in Settings."
            }
        }
        return engine.transcript
    }

    private var sentimentInput: SentimentInput {
        SentimentInput(
            text: engine.transcript,
            languageCode: settings.languageCode,
            isFinal: !engine.isStreaming
                && engine.phase != .listening
                && engine.phase != .transcribingFile
        )
    }

    private var sentimentRequest: SentimentInput? {
        settings.sentimentAnalysisEnabled ? sentimentInput : nil
    }

    private var sentimentTranscript: AttributedString {
        guard !engine.transcript.isEmpty else {
            var placeholder = AttributedString(displayTranscript)
            placeholder.foregroundColor = Theme.secondaryText
            return placeholder
        }

        guard settings.sentimentAnalysisEnabled else {
            return AttributedString(engine.transcript)
        }

        return sentiment.document.attributedString(
            fallback: engine.transcript,
            palette: sentimentPalette
        )
    }

    private var sentimentPalette: SentimentPalette {
        SentimentPalette(
            negative: Color(red: 1, green: 0.48, blue: 0.36),
            neutral: .white,
            positive: Theme.aurora2,
            mixed: Theme.aurora1,
            unavailable: Theme.secondaryText,
            provisional: .white
        )
    }

    private var sentimentSummaryPresentation: SentimentPresentation? {
        guard settings.sentimentAnalysisEnabled,
            !engine.transcript.isEmpty,
            sentiment.analyzedInput == sentimentInput
        else { return nil }
        return sentiment.document.summary.presentation(palette: sentimentPalette)
    }

    private var sentimentAccessibilityLabel: String {
        guard settings.sentimentAnalysisEnabled, !engine.transcript.isEmpty else {
            return displayTranscript
        }
        return sentiment.document.accessibilityLabel(fallback: engine.transcript)
    }

    // MARK: Microphone controls

    private var micControls: some View {
        VStack(spacing: 20) {
            WaveformView(level: engine.micLevel, isActive: engine.phase == .listening)
                .frame(height: 90)
                .glassCard(padding: 14)

            RecordButton(
                isRecording: engine.phase == .listening,
                isEnabled: !modelBusyButNotListening
            ) {
                Task {
                    if engine.phase == .listening {
                        await engine.stopListening()
                    } else {
                        await engine.startListening()
                    }
                }
            }

            Text(engine.phase == .listening ? "Listening… tap to stop" : "Tap to start recording")
                .font(.subheadline)
                .foregroundStyle(Theme.secondaryText)
        }
    }

    private var modelBusyButNotListening: Bool {
        engine.isBusy && engine.phase != .listening
    }

    // MARK: File handling

    private func handleFileSelection(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            Task { await engine.transcribeFile(url: url) }
        case .failure:
            break
        }
    }
}

// MARK: - Live badge

private struct LiveBadge: View {
    @State private var pulse = false
    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(Color(hex: 0xFF5E7E))
                .frame(width: 7, height: 7)
                .opacity(pulse ? 0.4 : 1)
            Text("LIVE")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(Color(hex: 0xFF5E7E))
        }
        .padding(.horizontal, 8).padding(.vertical, 3)
        .background(Capsule().fill(Color(hex: 0xFF5E7E).opacity(0.15)))
        .onAppear {
            withAnimation(.easeInOut(duration: 0.7).repeatForever(autoreverses: true)) {
                pulse = true
            }
        }
    }
}

// MARK: - Record button

private struct RecordButton: View {
    var isRecording: Bool
    var isEnabled: Bool
    var action: () -> Void

    @State private var ring = false

    var body: some View {
        Button(action: action) {
            ZStack {
                // Pulsing halo while recording.
                Circle()
                    .stroke(Theme.recordingGradient, lineWidth: 3)
                    .frame(width: 110, height: 110)
                    .scaleEffect(isRecording && ring ? 1.25 : 1.0)
                    .opacity(isRecording ? (ring ? 0 : 0.7) : 0)

                Circle()
                    .fill(
                        isRecording
                            ? AnyShapeStyle(Theme.recordingGradient)
                            : AnyShapeStyle(Theme.brandGradient)
                    )
                    .frame(width: 92, height: 92)
                    .shadow(
                        color: (isRecording ? Color(hex: 0xFF5E7E) : Theme.aurora1).opacity(0.5),
                        radius: 18, y: 6)

                Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                    .font(.system(size: 34, weight: .bold))
                    .foregroundStyle(.white)
                    .contentTransition(.symbolEffect(.replace))
            }
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1 : 0.5)
        .onChange(of: isRecording) { _, rec in
            if rec {
                withAnimation(.easeOut(duration: 1.1).repeatForever(autoreverses: false)) {
                    ring = true
                }
            } else {
                ring = false
            }
        }
    }
}
