import SwiftUI

/// Full-screen-ish overlay shown while models download / compile / load.
/// Renders a determinate ring when a fraction is available, plus a phase label.
struct ModelDownloadOverlay: View {
    var fraction: Double
    var message: String

    var body: some View {
        VStack(spacing: 22) {
            ZStack {
                Circle()
                    .stroke(Theme.cardStroke, lineWidth: 8)
                    .frame(width: 96, height: 96)

                Circle()
                    .trim(from: 0, to: max(0.02, fraction))
                    .stroke(
                        Theme.brandGradient,
                        style: StrokeStyle(lineWidth: 8, lineCap: .round)
                    )
                    .frame(width: 96, height: 96)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.3), value: fraction)

                Text("\(Int(fraction * 100))%")
                    .font(.system(.headline, design: .rounded).weight(.bold))
                    .monospacedDigit()
                    .contentTransition(.numericText())
            }

            VStack(spacing: 6) {
                Text("Setting up Nemotron")
                    .font(.headline)
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(Theme.secondaryText)
                    .multilineTextAlignment(.center)
                    .animation(.default, value: message)
            }

            Text("Loading the on-device model onto the Neural Engine.")
                .font(.caption)
                .foregroundStyle(Theme.secondaryText.opacity(0.8))
                .multilineTextAlignment(.center)
        }
        .padding(28)
        .frame(maxWidth: 340)
        .glassCard(padding: 24)
        .shadow(color: .black.opacity(0.4), radius: 30, y: 12)
    }
}

/// Overlay shown when model preparation fails — surfaces the real error and
/// offers a retry instead of silently dropping back to an idle screen.
struct ModelFailureOverlay: View {
    var message: String
    var onRetry: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "exclamationmark.icloud")
                .font(.system(size: 46, weight: .semibold))
                .foregroundStyle(Color(hex: 0xFF5E7E))

            VStack(spacing: 6) {
                Text("Couldn't load the model")
                    .font(.headline)
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(Theme.secondaryText)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Button(action: onRetry) {
                HStack(spacing: 8) {
                    Image(systemName: "arrow.clockwise")
                    Text("Retry").fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 13)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Theme.brandGradient)
                )
                .foregroundStyle(.white)
            }
            .buttonStyle(.plain)
        }
        .padding(26)
        .frame(maxWidth: 340)
        .glassCard(padding: 22)
        .shadow(color: .black.opacity(0.4), radius: 30, y: 12)
    }
}
