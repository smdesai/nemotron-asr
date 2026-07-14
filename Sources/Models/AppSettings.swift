import Foundation
import SwiftUI

/// Chunk-size tiers supported by the multilingual model. The raw value is the
/// `chunkMs` that selects the bundled `Models/<ship>/<chunkMs>ms/` variant.
enum ChunkSize: Int, CaseIterable, Identifiable {
    case ms560 = 560
    case ms1120 = 1120
    case ms2240 = 2240  // recommended / default
    case ms4480 = 4480

    var id: Int { rawValue }

    var seconds: Double { Double(rawValue) / 1000.0 }

    var label: String {
        switch self {
        case .ms560: return "0.56 s"
        case .ms1120: return "1.12 s"
        case .ms2240: return "2.24 s"
        case .ms4480: return "4.48 s"
        }
    }

    /// Short caption describing the latency/accuracy trade-off.
    var blurb: String {
        switch self {
        case .ms560: return "Lowest latency"
        case .ms1120: return "Low latency"
        case .ms2240: return "Recommended balance"
        case .ms4480: return "Best for long audio"
        }
    }

    static let `default`: ChunkSize = .ms2240
}

/// Which on-device inference backend runs the model.
///
/// - `coreml`: the shipping path via FluidAudio / CoreML `.mlmodelc` (iOS 18+).
/// - `coreai`: the experimental Apple Core AI path via `.aimodel` bundles
///   (iOS 27+). Encoder is the int8 sharded Core AI build; the mel front-end runs
///   in Swift/vDSP (`MelFrontend`). See `coreai/` under each tier dir.
enum InferenceBackend: String, CaseIterable, Identifiable {
    case coreml
    case coreai

    var id: String { rawValue }

    var label: String {
        switch self {
        case .coreml: return "CoreML"
        case .coreai: return "Core AI"
        }
    }

    var blurb: String {
        switch self {
        case .coreml: return "Shipping path — FluidAudio on the Neural Engine."
        case .coreai: return "Experimental — Apple Core AI (.aimodel), requires iOS 27."
        }
    }

    static let `default`: InferenceBackend = .coreml
}

/// How transcription results are surfaced for **audio file** input.
enum FileTranscriptionMode: String, CaseIterable, Identifiable {
    /// Show the transcript only once the whole file finishes.
    case atEnd
    /// Stream the transcript live as the file is processed.
    case streamed

    var id: String { rawValue }

    var label: String {
        switch self {
        case .atEnd: return "Show at end"
        case .streamed: return "Stream live"
        }
    }

    var systemImage: String {
        switch self {
        case .atEnd: return "doc.text"
        case .streamed: return "dot.radiowaves.left.and.right"
        }
    }
}

/// App-wide, persisted user settings. Backed by `@AppStorage` so they survive
/// relaunches.
@MainActor
final class AppSettings: ObservableObject {
    @AppStorage("chunkSizeMs") private var chunkSizeRaw: Int = ChunkSize.default.rawValue
    @AppStorage("languageCode") private var languageCodeStorage: String = ""  // "" == auto
    @AppStorage("fileMode") private var fileModeRaw: String = FileTranscriptionMode.streamed
        .rawValue
    @AppStorage("backend") private var backendRaw: String = InferenceBackend.default.rawValue
    @AppStorage("sentimentAnalysisEnabled") private var sentimentAnalysisEnabledStorage = true

    var backend: InferenceBackend {
        get { InferenceBackend(rawValue: backendRaw) ?? .default }
        set {
            backendRaw = newValue.rawValue
            objectWillChange.send()
        }
    }

    var chunkSize: ChunkSize {
        get { ChunkSize(rawValue: chunkSizeRaw) ?? .default }
        set {
            chunkSizeRaw = newValue.rawValue
            objectWillChange.send()
        }
    }

    /// `nil` means auto-detect.
    var languageCode: String? {
        get { languageCodeStorage.isEmpty ? nil : languageCodeStorage }
        set {
            languageCodeStorage = newValue ?? ""
            objectWillChange.send()
        }
    }

    var language: ASRLanguage {
        get { ASRLanguageCatalog.language(forCode: languageCode) }
        set { languageCode = newValue.code }
    }

    var fileMode: FileTranscriptionMode {
        get { FileTranscriptionMode(rawValue: fileModeRaw) ?? .streamed }
        set {
            fileModeRaw = newValue.rawValue
            objectWillChange.send()
        }
    }

    var sentimentAnalysisEnabled: Bool {
        get { sentimentAnalysisEnabledStorage }
        set {
            sentimentAnalysisEnabledStorage = newValue
            objectWillChange.send()
        }
    }
}
