import ActivityKit
import Foundation
import os

/// Owns the lifecycle of the transcription Live Activity: request one when the
/// mic goes live, push throttled transcript updates while listening, and end it
/// when recording stops. App-side only — the widget extension never imports
/// this; it only sees `TranscriptionActivityAttributes`.
@MainActor
final class LiveActivityController {

    /// Unified logging so the lines show in Console.app (and Xcode) even when
    /// the app is detached / backgrounded. Filter Console by the "LiveActivity"
    /// category or subsystem `com.sdesai.NemotronASR`.
    private let log = Logger(subsystem: "com.sdesai.NemotronASR", category: "LiveActivity")

    private var activity: Activity<TranscriptionActivityAttributes>?

    /// Throttle bookkeeping so rapid streaming partials don't spam ActivityKit.
    private var lastPush = Date.distantPast
    private var lastPushedTail = ""

    /// Human-readable result of the most recent start/end, surfaced in Settings
    /// for on-device diagnosis (no console needed).
    private(set) var lastStatus = "idle"

    /// Minimum spacing between pushed updates. Kept modest (~1.5/sec) so the
    /// lock-screen view advances smoothly rather than flickering on every chunk.
    private let minInterval: TimeInterval = 0.65
    /// How much trailing transcript to surface in the activity.
    private let tailLength = 280

    /// Start a fresh activity for a new mic session. No-ops if the user has
    /// Live Activities disabled. Ends any stale activity left from a prior run.
    func start(language: String?, isListening: Bool = true) {
        let info = ActivityAuthorizationInfo()
        log.notice(
            "start() called — areActivitiesEnabled=\(info.areActivitiesEnabled, privacy: .public)")
        guard info.areActivitiesEnabled else {
            lastStatus = "DISABLED — enable Settings ▸ Nemotron ASR ▸ Live Activities"
            log.error(
                "NOT started — Live Activities disabled. Enable Settings ▸ Nemotron ASR ▸ Live Activities (and Settings ▸ Face ID & Passcode ▸ Live Activities for the lock screen)."
            )
            return
        }

        // Clear any activity that survived a crash / prior session.
        for stale in Activity<TranscriptionActivityAttributes>.activities {
            Task { await stale.end(nil, dismissalPolicy: .immediate) }
        }

        let attributes = TranscriptionActivityAttributes(startedAt: Date())
        let initial = TranscriptionActivityAttributes.ContentState(
            transcript: "", isListening: isListening, languageLabel: language
        )
        do {
            let activity = try Activity.request(
                attributes: attributes,
                content: .init(state: initial, staleDate: nil),
                pushType: nil
            )
            self.activity = activity
            lastPush = Date()
            lastPushedTail = ""
            lastStatus = "active ✓ (id \(activity.id.prefix(6)))"
            log.notice(
                "started id=\(activity.id, privacy: .public) (active count: \(Activity<TranscriptionActivityAttributes>.activities.count, privacy: .public))"
            )
        } catch {
            self.activity = nil
            lastStatus = "request FAILED: \(error.localizedDescription)"
            log.error("Activity.request FAILED: \(String(describing: error), privacy: .public)")
        }
    }

    /// Push a (throttled) transcript update. Drops redundant or too-frequent
    /// pushes; the next allowed push always carries the latest tail, and `end`
    /// flushes the final text regardless.
    func update(transcript: String, isListening: Bool, language: String?) {
        guard let activity else { return }
        let tail = Self.tail(of: transcript, max: tailLength)
        if tail == lastPushedTail { return }  // nothing new
        let now = Date()
        if now.timeIntervalSince(lastPush) < minInterval { return }  // rate limit

        lastPush = now
        lastPushedTail = tail
        let state = TranscriptionActivityAttributes.ContentState(
            transcript: tail, isListening: isListening, languageLabel: language
        )
        Task { await activity.update(.init(state: state, staleDate: nil)) }
    }

    /// Finalize and dismiss the activity (shortly after, so the user sees the
    /// final line). Safe to call when nothing is active.
    func end(finalTranscript: String, language: String?) {
        guard let activity else { return }
        log.notice("end() — dismissing id=\(activity.id, privacy: .public)")
        lastStatus = "ended"
        self.activity = nil
        let state = TranscriptionActivityAttributes.ContentState(
            transcript: Self.tail(of: finalTranscript, max: tailLength),
            isListening: false,
            languageLabel: language
        )
        Task {
            await activity.end(
                .init(state: state, staleDate: nil),
                dismissalPolicy: .after(Date().addingTimeInterval(4))
            )
        }
    }

    /// Last `max` characters of the transcript, prefixed with an ellipsis when
    /// truncated so the user can tell it's a tail.
    private static func tail(of transcript: String, max: Int) -> String {
        guard transcript.count > max else { return transcript }
        let start = transcript.index(transcript.endIndex, offsetBy: -max)
        return "…" + transcript[start...]
    }
}
