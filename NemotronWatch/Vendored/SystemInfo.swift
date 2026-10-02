import Foundation

/// System information utilities (vendored subset: only what the watch/iOS
/// Nemotron pipeline uses).
public enum SystemInfo {
    public static var isAppleSilicon: Bool {
        #if arch(arm64)
        return true
        #else
        return false
        #endif
    }
}
