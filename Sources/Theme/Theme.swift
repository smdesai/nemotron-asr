import SwiftUI

/// Central design tokens for the app — colors, gradients, and reusable
/// modifiers. Keeping these in one place makes the UI feel cohesive and
/// easy to retheme.
enum Theme {
    // MARK: Brand palette
    static let accent = Color("AccentColor")
    static let aurora1 = Color(hex: 0x6D5CF6)   // violet
    static let aurora2 = Color(hex: 0x2DD4BF)   // teal
    static let aurora3 = Color(hex: 0xF472B6)   // pink

    /// Soft animated-feeling background gradient used app-wide.
    static var backgroundGradient: LinearGradient {
        LinearGradient(
            colors: [
                Color(hex: 0x0B1020),
                Color(hex: 0x141A33),
                Color(hex: 0x1B1130),
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    /// Accent gradient used for primary actions and highlights.
    static var brandGradient: LinearGradient {
        LinearGradient(
            colors: [aurora1, aurora2],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    static var recordingGradient: LinearGradient {
        LinearGradient(
            colors: [Color(hex: 0xFF5E7E), aurora3],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    static let cardStroke = Color.white.opacity(0.08)
    static let cardFill = Color.white.opacity(0.06)
    static let secondaryText = Color.white.opacity(0.6)
}

// MARK: - Card style

struct GlassCard: ViewModifier {
    var padding: CGFloat = 18
    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(Theme.cardFill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(Theme.cardStroke, lineWidth: 1)
            )
    }
}

extension View {
    func glassCard(padding: CGFloat = 18) -> some View {
        modifier(GlassCard(padding: padding))
    }
}

// MARK: - Color hex helper

extension Color {
    init(hex: UInt32, alpha: Double = 1.0) {
        let r = Double((hex >> 16) & 0xFF) / 255.0
        let g = Double((hex >> 8) & 0xFF) / 255.0
        let b = Double(hex & 0xFF) / 255.0
        self.init(.sRGB, red: r, green: g, blue: b, opacity: alpha)
    }
}
