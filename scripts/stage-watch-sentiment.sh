#!/usr/bin/env bash
# Compile and stage the multilingual sentiment model for the watchOS app.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 MODEL.mlpackage vocab.txt"
  exit 2
fi

MODEL="$1"
VOCABULARY="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/NemotronWatchSentiment"

if [ ! -d "$MODEL" ]; then
  echo "ERROR: model package not found at $MODEL"
  exit 1
fi
if [ "$(basename "$MODEL")" != "baseline-t128-fp16.mlpackage" ]; then
  echo "ERROR: expected baseline-t128-fp16.mlpackage, got $(basename "$MODEL")"
  exit 1
fi
if [ ! -f "$VOCABULARY" ]; then
  echo "ERROR: vocabulary not found at $VOCABULARY"
  exit 1
fi

mkdir -p "$DST"
rm -rf "$DST/baseline-t128-fp16.mlmodelc"
xcrun coremlcompiler compile "$MODEL" "$DST" \
  --platform watchOS \
  --deployment-target 27.0 >/dev/null
cp "$VOCABULARY" "$DST/vocab.txt"

echo "Staged watch sentiment resources in $DST"
du -sh "$DST"
