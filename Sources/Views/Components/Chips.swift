import SwiftUI

/// Small pill displaying an icon + label, used for stats and metadata.
struct InfoChip: View {
    var systemImage: String
    var text: String
    var tint: Color = Theme.aurora2

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage)
                .font(.caption2.weight(.semibold))
            Text(text)
                .font(.caption.weight(.medium))
                .lineLimit(1)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Capsule().fill(tint.opacity(0.14)))
    }
}
