//
//  Float16Compat.swift
//  NemotronASRCoreML
//
//  The Nemotron CoreML runtime only ever runs on Apple silicon (`preloadShared`
//  refuses anything else), but a universal Xcode build — e.g. a Release
//  configuration where ONLY_ACTIVE_ARCH is off — still compiles an x86_64
//  slice, and Swift marks `Float16` unavailable on macOS x86_64.
//
//  This 2-byte stand-in keeps that dead slice compiling. It is never executed:
//  every entry point traps, and `MemoryLayout<Float16>.size` stays 2 so the
//  pointer arithmetic over fp16 buffers matches the real type's layout.
//

#if !arch(arm64)
    struct Float16: Comparable, Hashable {
        var bitPattern: UInt16

        init(bitPattern: UInt16) {
            fatalError("NemotronASRCoreML: Float16 is unsupported on x86_64; this slice must never run.")
        }

        init(_ value: Float) {
            fatalError("NemotronASRCoreML: Float16 is unsupported on x86_64; this slice must never run.")
        }

        static func < (lhs: Float16, rhs: Float16) -> Bool {
            fatalError("NemotronASRCoreML: Float16 is unsupported on x86_64; this slice must never run.")
        }
    }

    extension Float {
        init(_ half: Float16) {
            fatalError("NemotronASRCoreML: Float16 is unsupported on x86_64; this slice must never run.")
        }
    }
#endif
