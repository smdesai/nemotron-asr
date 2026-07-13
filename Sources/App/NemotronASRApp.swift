import SwiftUI

@main
struct NemotronASRApp: App {
    @StateObject private var settings = AppSettings()
    @StateObject private var engine: TranscriptionEngine

    init() {
        let settings = AppSettings()
        _settings = StateObject(wrappedValue: settings)
        _engine = StateObject(wrappedValue: TranscriptionEngine(settings: settings))
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(settings)
                .environmentObject(engine)
                .preferredColorScheme(.dark)
                .tint(Theme.aurora2)
        }
    }
}
