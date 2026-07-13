import AppIntents
import Foundation

/// "Stop" button backing the Live Activity. As a `LiveActivityIntent` this runs
/// in the **app's** process (the system launches the app in the background if
/// needed), so a `NotificationCenter` post here reaches the engine's observer.
///
/// Kept dependency-free so the type also compiles into the widget extension,
/// where `Button(intent:)` references it.
struct StopRecordingIntent: LiveActivityIntent {
    static var title: LocalizedStringResource = "Stop"
    static var description = IntentDescription("Stop the live transcription.")

    func perform() async throws -> some IntentResult {
        NotificationCenter.default.post(name: .stopRecordingRequested, object: nil)
        return .result()
    }
}

extension Notification.Name {
    /// Posted by `StopRecordingIntent`; observed by `TranscriptionEngine` to end
    /// the current microphone session.
    static let stopRecordingRequested = Notification.Name("com.sdesai.NemotronASR.stopRecordingRequested")
}
