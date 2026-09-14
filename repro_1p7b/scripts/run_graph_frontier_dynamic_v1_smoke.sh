#!/usr/bin/env bash
set -uo pipefail

WORKTREE=/home/u2024311031/workspace/envfactory_repro_1p7b
TRAIN_ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN_PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
DATA_DIR="$WORKTREE/repro_1p7b/data/graph_frontier_dynamic_v1"
OUTPUT="$WORKTREE/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_smoke"
LOG_DIR="$WORKTREE/repro_1p7b/logs/graph_frontier_dynamic_v1_smoke"
REPORT="$WORKTREE/repro_1p7b/graph_frontier/reports/dynamic_v1_continuity_smoke.json"

cd "$WORKTREE" || exit 2
if [[ -e "$OUTPUT" || -e "$LOG_DIR" || -e "$REPORT" ]]; then
  echo "Refusing to overwrite an existing Dynamic v1 smoke artifact." >&2
  exit 3
fi
mkdir -p "$DATA_DIR" "$LOG_DIR"
"$RUN_PY" -m repro_1p7b.graph_frontier.dynamic_v1 smoke-data \
  --source repro_1p7b/datasets/EnvFactory-SFT-FILTERED/mcp_factory_sft_nips.json \
  --stage1 "$DATA_DIR/smoke_stage1.json" --stage2 "$DATA_DIR/smoke_stage2.json" \
  --report "$LOG_DIR/smoke_data.json" || exit 4

set +u
source /opt/conda/etc/profile.d/conda.sh || exit 7
conda activate "$TRAIN_ENV" || exit 8
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 --master-port=29521 \
  -m repro_1p7b.graph_frontier.train_boundary \
  repro_1p7b/configs/llamafactory_dynamic_v1_smoke_stage1.yaml \
  --stop-after-step 2 2>&1 | tee "$LOG_DIR/stage1.log"
stage1_rc=${PIPESTATUS[0]}
printf '%s\n' "$stage1_rc" > "$LOG_DIR/stage1.exit_code"
[[ "$stage1_rc" -eq 0 ]] || exit 5

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 --master-port=29522 \
  -m repro_1p7b.graph_frontier.train_boundary \
  repro_1p7b/configs/llamafactory_dynamic_v1_smoke_stage2.yaml \
  --resume-from-checkpoint repro_1p7b/checkpoints/graph_frontier_dynamic_v1_smoke/checkpoint-2 \
  --ignore-data-skip 2>&1 | tee "$LOG_DIR/stage2.log"
stage2_rc=${PIPESTATUS[0]}
printf '%s\n' "$stage2_rc" > "$LOG_DIR/stage2.exit_code"
[[ "$stage2_rc" -eq 0 ]] || exit 6

"$RUN_PY" -m repro_1p7b.graph_frontier.continuity_audit smoke \
  --output "$OUTPUT" --report "$REPORT"
