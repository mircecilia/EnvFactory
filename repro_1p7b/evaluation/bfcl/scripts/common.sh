#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPRO_ROOT="$(cd "$EVAL_DIR/../../.." && pwd)"
BFCL_ROOT="${BFCL_ROOT:-/home/u2024311031/benchmarks/gorilla_bfcl_v1p3/berkeley-function-call-leaderboard}"
BFCL_ENV="${BFCL_ENV:-/home/u2024311031/.conda/envs/envfactory_bfcl_v1p3}"
BFCL_BIN="$BFCL_ENV/bin/bfcl"
BFCL_PYTHON="$BFCL_ENV/bin/python"
BFCL_MODEL_ID="${BFCL_MODEL_ID:-Qwen/Qwen3-1.7B-FC}"
BFCL_BACKEND="${BFCL_BACKEND:-sglang}"
BFCL_TEMPERATURE="${BFCL_TEMPERATURE:-0.7}"
BFCL_GPU_MEMORY_UTILIZATION="${BFCL_GPU_MEMORY_UTILIZATION:-0.75}"
BFCL_PORT="${BFCL_PORT:-1053}"
EXPECTED_BFCL_COMMIT="ea13468e4423454d0c213704fb87cf7cb3990433"
export CUDA_HOME="$BFCL_ENV"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$BFCL_ENV/bin/x86_64-conda-linux-gnu-cc"
export CXX="$BFCL_ENV/bin/x86_64-conda-linux-gnu-c++"
export GCC="$BFCL_ENV/bin/x86_64-conda-linux-gnu-gcc"
export GXX="$BFCL_ENV/bin/x86_64-conda-linux-gnu-g++"
export NVCC_PREPEND_FLAGS="-ccbin=$CXX"

require_clean_bfcl() {
  test "$(git -C "$BFCL_ROOT" rev-parse HEAD)" = "$EXPECTED_BFCL_COMMIT"
  test -z "$(git -C "$BFCL_ROOT" status --porcelain)"
  test -x "$BFCL_BIN"
  test -x "$BFCL_PYTHON"
}

require_model() {
  local model_path="$1"
  test -f "$model_path/config.json"
  test -f "$model_path/tokenizer_config.json"
  test -f "$model_path/tokenizer.json"
  compgen -G "$model_path/*.safetensors" >/dev/null
}
