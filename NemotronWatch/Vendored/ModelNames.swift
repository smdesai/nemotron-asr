import Foundation

// Model / resource file names for the streaming Nemotron CoreML pipeline.

public enum ModelNames {

    /// Nemotron Speech Streaming Multilingual 0.6B model names.
    ///
    /// Base names of the CoreML artifacts (`<name>.mlmodelc`) plus the shared
    /// tokenizer / metadata files. The split encoder files
    /// (`encoder_pre_encode`, `encoder_shard_0..3`) and fused decode assets are
    /// named inline at the load site.
    public enum NemotronMultilingualStreaming {
        public static let preprocessor = "preprocessor"
        public static let decoder = "decoder"
        public static let joint = "joint"
        public static let tokenizer = "tokenizer.json"
        public static let metadata = "metadata.json"
    }
}
