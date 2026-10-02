Core AI probe assets for the watch app (contents gitignored). Empty = normal build.

- `encoder_shard_{0..3}_int8.aimodel` → WatchCoreAIProbe.runShards (load + timed encode chain)
- `diag/*.aimodel` → WatchCoreAIProbe.runDiagnostics (crash-safe CPU-compile ladder;
  progress in the app's Documents/coreai_diag.txt)

When either is present the watch app runs the probe INSTEAD of the CoreML pipeline.
