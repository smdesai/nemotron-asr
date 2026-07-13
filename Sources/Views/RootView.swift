import SwiftUI

/// Top-level shell: animated background + the main transcription screen,
/// with the model-download overlay layered on top during preparation.
struct RootView: View {
    @EnvironmentObject var engine: TranscriptionEngine
    @State private var showSettings = false

    var body: some View {
        ZStack {
            Theme.backgroundGradient.ignoresSafeArea()
            AuroraBackground().ignoresSafeArea()

            TranscriptionView(showSettings: $showSettings)

            if case .preparing = engine.phase {
                Color.black.opacity(0.45).ignoresSafeArea()
                    .transition(.opacity)
                ModelDownloadOverlay(fraction: engine.prepFraction, message: engine.prepMessage)
                    .transition(.scale.combined(with: .opacity))
            }

            if case .failed(let message) = engine.phase {
                Color.black.opacity(0.55).ignoresSafeArea()
                    .transition(.opacity)
                ModelFailureOverlay(message: message) {
                    Task { await engine.retryPreparation() }
                }
                .padding(24)
                .transition(.scale.combined(with: .opacity))
            }
        }
        .animation(.spring(response: 0.4, dampingFraction: 0.85), value: engine.phase)
        .sheet(isPresented: $showSettings) {
            SettingsView()
        }
    }
}

/// Soft moving blobs behind the content for depth. Cheap — two blurred circles.
private struct AuroraBackground: View {
    @State private var animate = false

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Circle()
                    .fill(Theme.aurora1.opacity(0.28))
                    .frame(width: geo.size.width * 0.9)
                    .blur(radius: 90)
                    .offset(
                        x: animate ? -geo.size.width * 0.25 : geo.size.width * 0.2,
                        y: animate ? -geo.size.height * 0.2 : -geo.size.height * 0.32
                    )
                Circle()
                    .fill(Theme.aurora2.opacity(0.22))
                    .frame(width: geo.size.width * 0.8)
                    .blur(radius: 90)
                    .offset(
                        x: animate ? geo.size.width * 0.3 : -geo.size.width * 0.15,
                        y: animate ? geo.size.height * 0.28 : geo.size.height * 0.36
                    )
            }
            .onAppear {
                withAnimation(.easeInOut(duration: 9).repeatForever(autoreverses: true)) {
                    animate = true
                }
            }
        }
    }
}
