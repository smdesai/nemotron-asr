import ActivityKit
import Foundation

/// Shared Live Activity contract between the app (which starts/updates/ends the
/// activity) and the widget extension (which renders it). Dependency-free on
/// purpose — it imports only ActivityKit/Foundation so the same file compiles
/// into both targets without dragging in FluidAudio or any app code.
struct TranscriptionActivityAttributes: ActivityAttributes {
    /// The live, mutable part — pushed on every (throttled) transcript update.
    struct ContentState: Codable, Hashable {
        /// Truncated tail of the running transcript (kept small for the lock
        /// screen / Dynamic Island and the ~4 KB ContentState budget).
        var transcript: String
        /// True while the microphone is live; drives the listening indicator
        /// and whether the Stop button is shown.
        var isListening: Bool
        /// Detected or pinned language, friendly name (nil when unknown).
        var languageLabel: String?
    }

    /// Set once when the activity starts.
    var startedAt: Date
}
