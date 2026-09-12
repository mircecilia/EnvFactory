#!/usr/bin/env bash
set -euo pipefail
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/u2024311031/workspace/envfactory_repro_1p7b/repro_1p7b/models/Qwen3-1.7B}"
mkdir -p "$EVAL_DIR/logs/full"
"$EVAL_DIR/scripts/run_bfcl.sh" base "$BASE_MODEL_PATH" full 2>&1 | tee "$EVAL_DIR/logs/full/base_full.log"
