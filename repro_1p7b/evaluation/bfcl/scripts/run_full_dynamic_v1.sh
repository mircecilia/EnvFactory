#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DYNAMIC_MODEL_PATH="${DYNAMIC_MODEL_PATH:-$REPRO_ROOT/checkpoints/graph_frontier_dynamic_v1_8k_1p7b}"
DYNAMIC_RUN_ROOT="$EVAL_DIR/artifacts/full/dynamic_v1"
DYNAMIC_LOG="$EVAL_DIR/logs/full/dynamic_v1_full.log"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != 0 ] && [ "$DRY_RUN" != 1 ]; then
  echo "invalid DRY_RUN value: $DRY_RUN" >&2
  exit 2
fi
if [ -e "$DYNAMIC_RUN_ROOT" ]; then
  echo "refusing to overwrite existing Dynamic v1 BFCL results: $DYNAMIC_RUN_ROOT" >&2
  exit 3
fi

if [ "$DRY_RUN" = 1 ]; then
  require_clean_bfcl
  require_model "$DYNAMIC_MODEL_PATH"
  printf 'DRY_RUN label=dynamic_v1 model=%s result=%s log=%s\n' \
    "$DYNAMIC_MODEL_PATH" "$DYNAMIC_RUN_ROOT" "$DYNAMIC_LOG"
  exit 0
fi

mkdir -p "$EVAL_DIR/logs/full"
"$EVAL_DIR/scripts/run_bfcl.sh" dynamic_v1 "$DYNAMIC_MODEL_PATH" full \
  2>&1 | tee "$DYNAMIC_LOG"
