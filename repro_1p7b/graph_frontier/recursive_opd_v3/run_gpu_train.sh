#!/usr/bin/env bash
set -euo pipefail

# Complete-C deliberately has no inherited/default GPU budget.  A future user
# decision must set every value and provide the exact immutable approval JSON.
: "${V3_CYCLE:?V3_CYCLE is UNDECIDED}"
: "${V3_DATASET:?V3_DATASET is UNDECIDED}"
: "${V3_MODEL:?V3_MODEL is UNDECIDED}"
: "${V3_GPU_STEPS:?V3_GPU_STEPS is UNDECIDED}"
: "${V3_EXPOSURE_BUDGET:?V3_EXPOSURE_BUDGET is UNDECIDED}"
: "${V3_LR:?V3_LR is UNDECIDED}"
: "${V3_SEED:?V3_SEED is UNDECIDED}"
: "${V3_OUTPUT:?V3_OUTPUT is UNDECIDED}"
: "${V3_APPROVAL_MANIFEST:?explicit V3 approval manifest is required}"

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
cd "$ROOT"
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false
exec "$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  -m repro_1p7b.graph_frontier.recursive_opd_v3.train \
  --cycle "$V3_CYCLE" --dataset "$V3_DATASET" --model "$V3_MODEL" \
  --steps "$V3_GPU_STEPS" --exposure-budget "$V3_EXPOSURE_BUDGET" \
  --learning-rate "$V3_LR" --seed "$V3_SEED" --output "$V3_OUTPUT" \
  --approval-manifest "$V3_APPROVAL_MANIFEST"
