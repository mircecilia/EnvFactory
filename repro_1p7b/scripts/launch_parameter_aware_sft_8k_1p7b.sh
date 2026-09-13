#!/usr/bin/env bash
set -uo pipefail

WORKTREE=/home/u2024311031/workspace/envfactory_repro_1p7b
OUTPUT_DIR="$WORKTREE/repro_1p7b/checkpoints/parameter_aware_sft_8k_1p7b"
LOG_PATH="$WORKTREE/repro_1p7b/logs/parameter_aware/formal_sft_8k_1p7b.log"
EXIT_PATH="$WORKTREE/repro_1p7b/logs/parameter_aware/formal_sft_8k_1p7b.exit_code"

cd "$WORKTREE"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  echo "Refusing to overwrite an existing formal parameter-aware SFT artifact." >&2
  exit 3
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate /home/u2024311031/.conda/envs/envfactory_sglang_1p7b
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORKTREE/repro_1p7b/logs/parameter_aware"

CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29513 \
  llamafactory-cli train \
  repro_1p7b/configs/llamafactory_sft_parameter_aware_8k_1p7b.yaml \
  2>&1 | tee "$LOG_PATH"
train_rc=${PIPESTATUS[0]}
printf '%s\n' "$train_rc" | tee "$EXIT_PATH"
exit "$train_rc"
