import Foundation

// Minimal vendored subset of FluidAudio's ModelNames for the watchOS port.
// Only the Nemotron multilingual streaming file constants are needed; the rest
// of the catalog (Diarizer / VAD / TTS / Qwen3 / Cohere / Sortformer / etc.)
// is intentionally omitted because the watch target never loads those models.

public enum ModelNames {

    /// Nemotron Speech Streaming Multilingual 0.6B model names.
    ///
    /// The multilingual build keeps all four CoreML artifacts at the top level
    /// (no `encoder/` subdirectory), and the encoder takes an extra `prompt_id`
    /// int32 input per chunk. The model is local-path-only (loaded from the app
    /// bundle on the watch — no download).
    public enum NemotronMultilingualStreaming {
        public static let preprocessor = "preprocessor"
        public static let encoder = "encoder"
        public static let decoder = "decoder"
        public static let joint = "joint"
        public static let tokenizer = "tokenizer.json"
        public static let metadata = "metadata.json"

        public static let preprocessorFile = preprocessor + ".mlmodelc"
        public static let encoderFile = encoder + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"
        public static let jointFile = joint + ".mlmodelc"

        public static let preprocessorPackage = preprocessor + ".mlpackage"
        public static let encoderPackage = encoder + ".mlpackage"
        public static let decoderPackage = decoder + ".mlpackage"
        public static let jointPackage = joint + ".mlpackage"
    }
}
