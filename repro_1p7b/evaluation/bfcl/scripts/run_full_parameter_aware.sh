#!/usr/bin/env bash
set -euo pipefail
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARAMETER_AWARE_MODEL_PATH=/home/u2024311031/workspace/envfactory_repro_1p7b/repro_1p7b/checkpoints/parameter_aware_sft_8k_1p7b
mkdir -p "$EVAL_DIR/logs/full"
"$EVAL_DIR/scripts/run_bfcl.sh" parameter_aware "$PARAMETER_AWARE_MODEL_PATH" full \
  2>&1 | tee "$EVAL_DIR/logs/full/parameter_aware_full.log"
