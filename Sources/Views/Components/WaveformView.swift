import SwiftUI

/// A lightweight, GPU-friendly bar waveform that reacts to a live 0...1 level.
/// Maintains a rolling history of recent levels so it scrolls like a real
/// level meter while recording.
struct WaveformView: View {
    /// Current input level 0...1.
    var level: Float
    /// Whether audio is actively flowing (drives the idle vs. active look).
    var isActive: Bool

    @State private var history: [CGFloat] = Array(repeating: 0.04, count: 48)

    private let barCount = 48
    /// Steady ~20 Hz tick so the meter scrolls continuously while active, even
    /// if the published level repeats or arrives sparsely. Without this the
    /// bars only move when `level` changes value, which can look frozen.
    private let ticker = Timer.publish(every: 0.05, on: .main, in: .common).autoconnect()

    var body: some View {
        GeometryReader { geo in
            let spacing: CGFloat = 3
            let barWidth = max(2, (geo.size.width - spacing * CGFloat(barCount - 1)) / CGFloat(barCount))
            HStack(alignment: .center, spacing: spacing) {
                ForEach(0..<barCount, id: \.self) { i in
                    Capsule()
                        .fill(isActive ? Theme.recordingGradient : Theme.brandGradient)
                        .frame(
                            width: barWidth,
                            height: max(3, history[i] * geo.size.height)
                        )
                        .opacity(isActive ? 1 : 0.35)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        }
        .onReceive(ticker) { _ in
            // Push the current level on every tick so the meter keeps scrolling
            // while listening; settle to the idle floor when not active.
            // Perceptual emphasis (sqrt) lifts quiet/mid levels so the bars
            // visibly swing instead of hugging the floor — audio changes are
            // easy to see. Idle pushes the bare floor (no emphasis).
            pushLevel(isActive ? sqrt(max(0, CGFloat(level))) : 0.04)
        }
        .onChange(of: isActive) { _, active in
            if !active { decay() }
        }
        .animation(.easeOut(duration: 0.12), value: history)
    }

    private func pushLevel(_ value: CGFloat) {
        // Add a little organic variation so the bars don't all match exactly.
        var next = history
        next.removeFirst()
        let jitter = CGFloat.random(in: 0.85...1.15)
        next.append(min(1, max(0.04, value * jitter)))
        history = next
    }

    private func decay() {
        history = Array(repeating: 0.04, count: barCount)
    }
}
