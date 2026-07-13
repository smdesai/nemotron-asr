// Compile-time shim for the watchOS 27.0 beta SDK, which omits the Metal
// module that CoreAIRuntime's swiftinterface imports. Only the protocol NAMES
// are needed to typecheck CoreAI's optional Metal-interop NDArray API —
// nothing here is ever linked or instantiated.
#import <Foundation/Foundation.h>
NS_ASSUME_NONNULL_BEGIN
@protocol MTLBuffer <NSObject>
@end
@protocol MTLCommandQueue <NSObject>
@end
NS_ASSUME_NONNULL_END
