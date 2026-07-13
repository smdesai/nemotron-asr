#!/usr/bin/env bash
# Stage the Nemotron multilingual 2240ms CoreML models into the watch bundle.
#
# The watchOS app loads models from the app bundle (no on-device download) and
# watchOS cannot compile .mlpackage at runtime, so everything is staged as
# pre-compiled .mlmodelc: the split encoder (pre_encode + 4 shards) ships
# compiled in the repo; preprocessor/decoder/joint only exist as .mlpackage in
# the 2240ms drop and are compiled here with coremlcompiler.
#
# (The Core AI .aimodel variant of this script was reverted: the watchOS 27
# beta's Core AI compiler has no m11 SoC backend — "Unsupported SoC (m11)" —
# so the watch runs the proven CoreML pipeline instead.)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/Resources/Models/multilingual/2240ms"
DST="$ROOT/NemotronWatchModels"

if [ ! -d "$SRC" ]; then
  echo "ERROR: source models not found at $SRC"
  exit 1
fi

mkdir -p "$DST"
echo "Staging CoreML models into $DST"

# Clear previously staged content (regenerable copies; sources stay in Resources/).
rm -rf "$DST"/*.aimodel "$DST"/*.mlmodelc "$DST"/mel_filterbank.f32 "$DST"/mel_window.f32

# Pre-compiled split encoder: copy verbatim.
COMPILED=(
  encoder_pre_encode
  encoder_shard_0
  encoder_shard_1
  encoder_shard_2
  encoder_shard_3
)
for m in "${COMPILED[@]}"; do
  if [ ! -d "$SRC/${m}.mlmodelc" ]; then
    echo "ERROR: required model $SRC/${m}.mlmodelc missing"
    exit 1
  fi
  cp -R "$SRC/${m}.mlmodelc" "$DST/"
  echo "  + ${m}.mlmodelc (copied)"
done

# Package-only models: compile to .mlmodelc with coremlcompiler.
PACKAGES=(
  preprocessor
  decoder
  joint
)
for m in "${PACKAGES[@]}"; do
  if [ ! -d "$SRC/${m}.mlpackage" ]; then
    echo "ERROR: required model $SRC/${m}.mlpackage missing"
    exit 1
  fi
  xcrun coremlcompiler compile "$SRC/${m}.mlpackage" "$DST/" > /dev/null
  echo "  + ${m}.mlmodelc (compiled from .mlpackage)"
done

# Companion JSON.
cp "$SRC/metadata.json" "$DST/"
cp "$SRC/tokenizer.json" "$DST/"
echo "  + metadata.json, tokenizer.json"

echo "Total staged size:"
du -sh "$DST"
