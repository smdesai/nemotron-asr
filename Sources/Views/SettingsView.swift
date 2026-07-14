import ActivityKit
import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var engine: TranscriptionEngine
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.backgroundGradient.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 18) {
                        backendCard
                        liveActivityCard
                        sentimentCard
                        languageCard
                        chunkSizeCard
                        fileModeCard
                        aboutCard
                    }
                    .padding(20)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .fontWeight(.semibold)
                }
            }
            .scrollContentBackground(.hidden)
        }
        // Reloading the model variant when the user changes ship/chunk/backend.
        .onChange(of: settings.chunkSize) { _, _ in engine.invalidateForSettingsChange() }
        .onChange(of: settings.languageCode) { _, _ in engine.invalidateForSettingsChange() }
        .onChange(of: settings.backend) { _, _ in engine.invalidateForSettingsChange() }
    }

    // MARK: Backend

    private var backendCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            cardHeader(
                "Inference Backend",
                systemImage: "cpu",
                subtitle: "Choose the on-device runtime. Core AI is experimental and needs iOS 27."
            )

            Picker("Backend", selection: backendBinding) {
                ForEach(InferenceBackend.allCases) { b in
                    Text(b.label).tag(b.id)
                }
            }
            .pickerStyle(.segmented)

            Label(settings.backend.blurb, systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(Theme.secondaryText)

            // Core AI on-device residency report (compute types + dtype histogram).
            if settings.backend == .coreai, let residency = engine.coreAIResidency {
                Text(residency)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(Theme.secondaryText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Theme.cardFill)
                    )
                    .textSelection(.enabled)
            }
        }
        .glassCard()
    }

    private var backendBinding: Binding<String> {
        Binding(
            get: { settings.backend.id },
            set: { settings.backend = InferenceBackend(rawValue: $0) ?? .coreml }
        )
    }

    // MARK: Live Activity (diagnostic)

    /// On-device readout of Live Activity state — avoids needing the console.
    /// `System enabled` reflects `ActivityAuthorizationInfo`; `Last result`
    /// shows what happened the last time recording started.
    private var liveActivityCard: some View {
        let enabled = ActivityAuthorizationInfo().areActivitiesEnabled
        return VStack(alignment: .leading, spacing: 12) {
            cardHeader(
                "Live Activity",
                systemImage: "bolt.badge.clock",
                subtitle:
                    "Shows the running transcript on the lock screen / Dynamic Island while recording in the background."
            )
            statusRow(
                "System enabled", value: enabled ? "Yes" : "No",
                tint: enabled ? Theme.aurora3 : Color(hex: 0xFF5E7E))
            statusRow("Last result", value: engine.liveActivityStatus, tint: Theme.secondaryText)
            if !enabled {
                Label(
                    "Turn on Settings ▸ Nemotron ASR ▸ Live Activities, and Settings ▸ Face ID & Passcode ▸ Live Activities.",
                    systemImage: "exclamationmark.triangle"
                )
                .font(.caption)
                .foregroundStyle(Color(hex: 0xFF5E7E))
            }
        }
        .glassCard()
    }

    private func statusRow(_ label: String, value: String, tint: Color) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.caption)
                .foregroundStyle(Theme.secondaryText)
            Spacer(minLength: 12)
            Text(value)
                .font(.system(.caption, design: .rounded).weight(.semibold))
                .foregroundStyle(tint)
                .multilineTextAlignment(.trailing)
                .textSelection(.enabled)
        }
    }

    // MARK: Sentiment analysis

    private var sentimentCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            cardHeader(
                "Sentiment Analysis",
                systemImage: "face.smiling",
                subtitle: "Color the transcript using on-device sentiment analysis."
            )
            Toggle("Highlight sentiment", isOn: sentimentAnalysisBinding)
                .font(.subheadline.weight(.semibold))
                .tint(Theme.aurora2)
        }
        .glassCard()
    }

    private var sentimentAnalysisBinding: Binding<Bool> {
        Binding(
            get: { settings.sentimentAnalysisEnabled },
            set: { settings.sentimentAnalysisEnabled = $0 }
        )
    }

    // MARK: Language

    private var languageCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            cardHeader(
                "Language", systemImage: "globe",
                subtitle: "Optional — leave on Auto-detect to let the model identify the language.")

            Menu {
                Picker("Language", selection: languageBinding) {
                    ForEach(ASRLanguageCatalog.all) { lang in
                        Text("\(lang.flag)  \(lang.name)").tag(lang.id)
                    }
                }
            } label: {
                HStack {
                    Text(settings.language.flag)
                    Text(settings.language.name)
                        .fontWeight(.semibold)
                    Spacer()
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.secondaryText)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Theme.cardFill)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(
                        Theme.cardStroke, lineWidth: 1))
            }
            .tint(.white)

            if settings.languageCode != nil {
                Label(
                    settings.language.isLatinScript
                        ? "Uses the fast Latin-script model."
                        : "Uses the full multilingual model.",
                    systemImage: "info.circle"
                )
                .font(.caption)
                .foregroundStyle(Theme.secondaryText)
            }
        }
        .glassCard()
    }

    private var languageBinding: Binding<String> {
        Binding(
            get: { settings.language.id },
            set: { newID in
                settings.language =
                    ASRLanguageCatalog.all.first { $0.id == newID } ?? ASRLanguageCatalog.auto
            }
        )
    }

    // MARK: Chunk size

    private var chunkSizeCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            cardHeader(
                "Chunk Size", systemImage: "waveform.path",
                subtitle:
                    "Smaller chunks lower latency; larger chunks suit longer audio. 2.24 s is recommended."
            )

            VStack(spacing: 10) {
                ForEach(ChunkSize.allCases) { size in
                    Button {
                        settings.chunkSize = size
                    } label: {
                        HStack(spacing: 12) {
                            Image(
                                systemName: settings.chunkSize == size
                                    ? "largecircle.fill.circle" : "circle"
                            )
                            .foregroundStyle(
                                settings.chunkSize == size ? Theme.aurora2 : Theme.secondaryText)
                            VStack(alignment: .leading, spacing: 2) {
                                HStack(spacing: 6) {
                                    Text(size.label).fontWeight(.semibold)
                                    if size == .default {
                                        Text("RECOMMENDED")
                                            .font(.system(size: 9, weight: .bold))
                                            .padding(.horizontal, 6).padding(.vertical, 2)
                                            .background(Capsule().fill(Theme.aurora2.opacity(0.2)))
                                            .foregroundStyle(Theme.aurora2)
                                    }
                                }
                                Text(size.blurb)
                                    .font(.caption)
                                    .foregroundStyle(Theme.secondaryText)
                            }
                            Spacer()
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 12)
                        .background(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .fill(
                                    settings.chunkSize == size
                                        ? Theme.aurora2.opacity(0.12) : Theme.cardFill)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .stroke(
                                    settings.chunkSize == size
                                        ? Theme.aurora2.opacity(0.5) : Theme.cardStroke,
                                    lineWidth: 1)
                        )
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.white)
                }
            }
        }
        .glassCard()
    }

    // MARK: File mode

    private var fileModeCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            cardHeader(
                "Audio File Output", systemImage: "doc.text.magnifyingglass",
                subtitle: "How transcription appears when you import an audio file.")

            Picker("File mode", selection: fileModeBinding) {
                ForEach(FileTranscriptionMode.allCases) { mode in
                    Label(mode.label, systemImage: mode.systemImage).tag(mode.id)
                }
            }
            .pickerStyle(.segmented)
        }
        .glassCard()
    }

    private var fileModeBinding: Binding<String> {
        Binding(
            get: { settings.fileMode.id },
            set: { settings.fileMode = FileTranscriptionMode(rawValue: $0) ?? .streamed }
        )
    }

    // MARK: About

    private var aboutCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            cardHeader("About", systemImage: "sparkles", subtitle: nil)
            Text(
                "Powered by NVIDIA Nemotron Speech Streaming Multilingual 0.6B, running fully on-device via CoreML on the Apple Neural Engine."
            )
            .font(.subheadline)
            .foregroundStyle(Theme.secondaryText)
        }
        .glassCard()
    }

    private func cardHeader(_ title: String, systemImage: String, subtitle: String?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(title, systemImage: systemImage)
                .font(.headline)
            if let subtitle {
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(Theme.secondaryText)
            }
        }
    }
}
