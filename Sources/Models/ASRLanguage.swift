import Foundation

/// A language the Nemotron multilingual model supports, paired with the exact
/// code understood by `StreamingNemotronMultilingualAsrManager.setLanguage(_:)`.
///
/// Codes are validated against the model's `prompt_dictionary` (metadata.json).
/// `nil` `code` means "Auto-detect" — the manager uses its `default_prompt_id`
/// and reports the detected language via `detectedLanguage()`.
struct ASRLanguage: Identifiable, Hashable {
    /// Code passed to `setLanguage`; `nil` for auto-detect.
    let code: String?
    /// Human-readable name shown in the picker.
    let name: String
    /// Flag emoji / glyph for a little visual flair.
    let flag: String

    var id: String { code ?? "auto" }

    /// Whether routing this language uses the vocab-pruned "latin" ship
    /// (en/es/fr/it/pt/de) vs. the full "multilingual" ship.
    var isLatinScript: Bool {
        guard let code else { return false }
        let c = code.lowercased()
        return ["en", "es", "fr", "it", "pt", "de"].contains { c.hasPrefix($0) }
    }
}

enum ASRLanguageCatalog {
    /// Auto-detect option — the default selection.
    static let auto = ASRLanguage(code: nil, name: "Auto-detect", flag: "🌐")

    /// The supported languages. Codes match keys present in the model's
    /// prompt_dictionary (verified): e.g. Arabic only exists as `ar-AR`,
    /// Norwegian Bokmål as `nb-NO`, Hebrew as `he-IL`.
    static let all: [ASRLanguage] = [
        auto,
        ASRLanguage(code: "en-US", name: "English (US)", flag: "🇺🇸"),
        ASRLanguage(code: "en-GB", name: "English (UK)", flag: "🇬🇧"),
        ASRLanguage(code: "es-US", name: "Spanish (US)", flag: "🇺🇸"),
        ASRLanguage(code: "es-ES", name: "Spanish (Spain)", flag: "🇪🇸"),
        ASRLanguage(code: "fr-FR", name: "French", flag: "🇫🇷"),
        ASRLanguage(code: "fr-CA", name: "French (Canada)", flag: "🇨🇦"),
        ASRLanguage(code: "it-IT", name: "Italian", flag: "🇮🇹"),
        ASRLanguage(code: "pt-BR", name: "Portuguese (Brazil)", flag: "🇧🇷"),
        ASRLanguage(code: "pt-PT", name: "Portuguese (Portugal)", flag: "🇵🇹"),
        ASRLanguage(code: "ru-RU", name: "Russian", flag: "🇷🇺"),
        ASRLanguage(code: "nl-NL", name: "Dutch", flag: "🇳🇱"),
        ASRLanguage(code: "de-DE", name: "German", flag: "🇩🇪"),
        ASRLanguage(code: "pl-PL", name: "Polish", flag: "🇵🇱"),
        ASRLanguage(code: "cs-CZ", name: "Czech", flag: "🇨🇿"),
        ASRLanguage(code: "ar-AR", name: "Arabic", flag: "🇸🇦"),
        ASRLanguage(code: "hi-IN", name: "Hindi", flag: "🇮🇳"),
        ASRLanguage(code: "ja-JP", name: "Japanese", flag: "🇯🇵"),
        ASRLanguage(code: "ko-KR", name: "Korean", flag: "🇰🇷"),
        ASRLanguage(code: "vi-VN", name: "Vietnamese", flag: "🇻🇳"),
        ASRLanguage(code: "tr-TR", name: "Turkish", flag: "🇹🇷"),
        ASRLanguage(code: "nb-NO", name: "Norwegian Bokmål", flag: "🇳🇴"),
        ASRLanguage(code: "he-IL", name: "Hebrew", flag: "🇮🇱"),
        ASRLanguage(code: "da-DK", name: "Danish", flag: "🇩🇰"),
        ASRLanguage(code: "sv-SE", name: "Swedish", flag: "🇸🇪"),
    ]

    static func language(forCode code: String?) -> ASRLanguage {
        guard let code else { return auto }
        return all.first { $0.code == code } ?? auto
    }

    /// Best-effort friendly name for a code the model reports via
    /// `detectedLanguage()` (which may be a tag like "en-US" or "es-419").
    static func displayName(forDetectedCode code: String) -> String {
        if let match = all.first(where: { $0.code?.caseInsensitiveCompare(code) == .orderedSame }) {
            return match.name
        }
        // Fall back to the system's localized language name for the base code.
        let base =
            code.split(whereSeparator: { $0 == "-" || $0 == "_" }).first.map(String.init) ?? code
        if let localized = Locale.current.localizedString(forLanguageCode: base) {
            return localized.capitalized
        }
        return code
    }
}
