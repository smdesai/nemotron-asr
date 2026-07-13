// Compile-time shim for the watchOS 27.0 beta SDK, which omits the IOSurface
// module that CoreAIRuntime's swiftinterface imports. Only the class NAME is
// needed to typecheck CoreAI's optional NDArray(ioSurface:) initializers —
// nothing here is ever linked or instantiated.
#import <Foundation/Foundation.h>
NS_ASSUME_NONNULL_BEGIN
@interface IOSurface : NSObject
@end
NS_ASSUME_NONNULL_END
