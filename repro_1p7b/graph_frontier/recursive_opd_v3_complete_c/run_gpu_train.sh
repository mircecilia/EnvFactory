#!/usr/bin/env bash
set -euo pipefail

# Complete-C has no inherited/default GPU budget.  A future user decision must
# set every value and provide an exact immutable approval JSON.
: "${COMPLETE_C_CYCLE:?COMPLETE_C_CYCLE is UNDECIDED}"
: "${COMPLETE_C_DATASET:?COMPLETE_C_DATASET is UNDECIDED}"
: "${COMPLETE_C_MODEL:?COMPLETE_C_MODEL is UNDECIDED}"
: "${COMPLETE_C_GPU_STEPS:?COMPLETE_C_GPU_STEPS is UNDECIDED}"
: "${COMPLETE_C_EXPOSURE_BUDGET:?COMPLETE_C_EXPOSURE_BUDGET is UNDECIDED}"
: "${COMPLETE_C_LR:?COMPLETE_C_LR is UNDECIDED}"
: "${COMPLETE_C_SEED:?COMPLETE_C_SEED is UNDECIDED}"
: "${COMPLETE_C_OUTPUT:?COMPLETE_C_OUTPUT is UNDECIDED}"
: "${COMPLETE_C_APPROVAL_MANIFEST:?explicit Complete-C approval manifest is required}"

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
cd "$ROOT"
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false
exec "$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  -m repro_1p7b.graph_frontier.recursive_opd_v3_complete_c.train \
  --cycle "$COMPLETE_C_CYCLE" --dataset "$COMPLETE_C_DATASET" \
  --model "$COMPLETE_C_MODEL" --steps "$COMPLETE_C_GPU_STEPS" \
  --exposure-budget "$COMPLETE_C_EXPOSURE_BUDGET" \
  --learning-rate "$COMPLETE_C_LR" --seed "$COMPLETE_C_SEED" \
  --output "$COMPLETE_C_OUTPUT" --approval-manifest "$COMPLETE_C_APPROVAL_MANIFEST"
