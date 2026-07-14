import ActivityKit
import SwiftUI
import WidgetKit

/// Live Activity for the running transcription: a lock-screen banner plus the
/// Dynamic Island presentations. Styling is self-contained (system colors) so
/// the widget target stays free of app dependencies.
struct TranscriptionLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: TranscriptionActivityAttributes.self) { context in
            // Lock screen / banner.
            LockScreenView(state: context.state)
                .activityBackgroundTint(Color.black.opacity(0.55))
                .activitySystemActionForegroundColor(.white)
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Label("Nemotron", systemImage: "waveform")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    if let lang = context.state.languageLabel {
                        Text(lang)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                DynamicIslandExpandedRegion(.bottom) {
                    VStack(alignment: .leading, spacing: 10) {
                        TranscriptText(
                            context.state.transcript, lineLimit: 3,
                            placeholder: context.state.isListening ? "Listening…" : "Transcribing…")
                        if context.state.isListening {
                            StopButton()
                        }
                    }
                }
            } compactLeading: {
                Image(systemName: "mic.fill")
                    .foregroundStyle(context.state.isListening ? .red : .secondary)
            } compactTrailing: {
                Image(systemName: "waveform")
                    .foregroundStyle(.secondary)
            } minimal: {
                Image(systemName: "mic.fill")
                    .foregroundStyle(context.state.isListening ? .red : .secondary)
            }
        }
    }
}

// MARK: - Lock screen

private struct LockScreenView: View {
    let state: TranscriptionActivityAttributes.ContentState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Label("Nemotron ASR", systemImage: "waveform")
                    .font(.caption.weight(.semibold))
                if state.isListening {
                    ListeningDot()
                }
                Spacer(minLength: 0)
                if let lang = state.languageLabel {
                    Text(lang)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            TranscriptText(
                state.transcript, lineLimit: 4,
                placeholder: state.isListening ? "Listening…" : "Transcribing…")
            if state.isListening {
                StopButton()
            }
        }
        .padding()
    }
}

// MARK: - Shared bits

private struct TranscriptText: View {
    let text: String
    let lineLimit: Int
    let placeholder: String

    init(_ text: String, lineLimit: Int, placeholder: String = "Listening…") {
        self.text = text
        self.lineLimit = lineLimit
        self.placeholder = placeholder
    }

    var body: some View {
        Text(text.isEmpty ? placeholder : text)
            .font(.system(.footnote, design: .rounded))
            .foregroundStyle(text.isEmpty ? AnyShapeStyle(.secondary) : AnyShapeStyle(.primary))
            .lineLimit(lineLimit)
            // Keep the *newest* words visible (truncate the start, not the end),
            // and swap content without the per-update fade so it advances
            // smoothly instead of flashing the same leading lines.
            .truncationMode(.head)
            .contentTransition(.identity)
            .multilineTextAlignment(.leading)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct StopButton: View {
    var body: some View {
        Button(intent: StopRecordingIntent()) {
            Label("Stop", systemImage: "stop.fill")
                .font(.caption.weight(.semibold))
                .frame(maxWidth: .infinity)
        }
        .tint(.red)
        .buttonStyle(.borderedProminent)
    }
}

private struct ListeningDot: View {
    var body: some View {
        Circle()
            .fill(.red)
            .frame(width: 7, height: 7)
    }
}
