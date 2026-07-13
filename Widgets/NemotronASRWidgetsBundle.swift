import SwiftUI
import WidgetKit

/// Entry point for the widget extension. Hosts the single transcription Live
/// Activity — no Home Screen widgets are vended.
@main
struct NemotronASRWidgetsBundle: WidgetBundle {
    var body: some Widget {
        TranscriptionLiveActivity()
    }
}
