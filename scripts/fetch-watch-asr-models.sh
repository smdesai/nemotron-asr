#!/usr/bin/env bash
# Stage the Nemotron multilingual 2240ms CoreML models into the watch bundle
# (NemotronWatchModels/), by delegating to the conversion kit:
#
#   nemotron-asr-conversion/convert_models.sh --watch-only
#
# That builds whatever is missing from the .nemo and stages the watch set:
# the split encoder (encoder_pre_encode + encoder_shard_0..3), preprocessor,
# decoder and the fused decoder_joint_argmax, plus metadata.json/tokenizer.json.
# Deliberately not staged: the bare joint (replaced by the fused decoder) and the
# smart-spec joint_noencproj_batched/native_weights (their ~98 MB in-memory fp32
# weights caused Neural Engine timeouts on the watch). Encoder models whose
# weights and graph are unchanged are kept byte-for-byte, so the watch doesn't
# recompile its Neural Engine programs. Details: the kit's README, "Apple Watch bundle".
#
# The watch runs CoreML only: Core AI can't compile for the watch SoC (m11).
#
# Usage:
#   scripts/fetch-watch-asr-models.sh [extra convert_models.sh args, e.g. --dry-run, --force]
#
# The kit is expected next to this repo (../nemotron-asr-conversion); override with
# NEMOTRON_CONVERSION_DIR=/path/to/nemotron-asr-conversion. Get it with:
#   git clone https://github.com/smdesai/nemotron-asr-conversion.git
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KIT="${NEMOTRON_CONVERSION_DIR:-$ROOT/../nemotron-asr-conversion}"
DST="$ROOT/NemotronWatchModels"

if [ ! -x "$KIT/convert_models.sh" ]; then
  echo "ERROR: conversion kit not found at $KIT (expected convert_models.sh)." >&2
  echo "       git clone https://github.com/smdesai/nemotron-asr-conversion.git \"$KIT\"" >&2
  echo "       or set NEMOTRON_CONVERSION_DIR." >&2
  exit 1
fi

exec "$KIT/convert_models.sh" --watch-only --watch-dir "$DST" "$@"
