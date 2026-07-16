// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "NemotronASR",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .watchOS(.v10),
    ],
    products: [
        .library(
            name: "NemotronASRCoreML",
            targets: ["NemotronASRCoreML"]
        )
    ],
    targets: [
        .target(
            name: "NemotronASRCoreML",
            path: ".",
            sources: [
                "Sources/Audio/MicrophoneCapture.swift",
                "Sources/CoreML/NemotronCoreMLTranscriber.swift",
                "Sources/Models/ASRLanguage.swift",
                "NemotronWatch/Vendored/AppLogger.swift",
                "NemotronWatch/Vendored/AsrTypes.swift",
                "NemotronWatch/Vendored/AudioConverter.swift",
                "NemotronWatch/Vendored/EncoderCacheManager.swift",
                "NemotronWatch/Vendored/MLModelConfigurationUtils.swift",
                "NemotronWatch/Vendored/MLMultiArray+Extensions.swift",
                "NemotronWatch/Vendored/ModelNames.swift",
                "NemotronWatch/Vendored/NativeRnntInner.swift",
                "NemotronWatch/Vendored/NemotronMultilingualStreamingConfig.swift",
                "NemotronWatch/Vendored/NemotronMultilingualTokenizer.swift",
                "NemotronWatch/Vendored/StreamingAsrUtils.swift",
                "NemotronWatch/Vendored/StreamingNemotronMultilingualAsrManager.swift",
                "NemotronWatch/Vendored/StreamingNemotronMultilingualAsrManager+Pipeline.swift",
                "NemotronWatch/Vendored/StreamingNemotronMultilingualAsrManager+Shared.swift",
                "NemotronWatch/Vendored/SystemInfo.swift",
                "NemotronWatch/Vendored/Tokenizer.swift",
            ],
            resources: [
                .copy("Resources/Models")
            ]
        )
    ]
)
