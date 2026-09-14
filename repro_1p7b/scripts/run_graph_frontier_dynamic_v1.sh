#!/usr/bin/env bash
set -uo pipefail

MODE=${1:-start}
[[ "$MODE" == "start" || "$MODE" == "resume" ]] || { echo "usage: $0 [start|resume]" >&2; exit 2; }
WORKTREE=/home/u2024311031/workspace/envfactory_repro_1p7b
TRAIN_ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
SERVER_ENV=/home/u2024311031/.conda/envs/envfactory_bfcl_v1p3
RUN_PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
DATA_DIR="$WORKTREE/repro_1p7b/data/graph_frontier_dynamic_v1"
OUTPUT="$WORKTREE/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b"
LOG_DIR="$WORKTREE/repro_1p7b/logs/graph_frontier_dynamic_v1"
STATUS="$LOG_DIR/status"
DIAG_ROOT="$WORKTREE/repro_1p7b/results/graph_frontier/dynamic_v1_stage1_diagnosis"
MANIFEST="$WORKTREE/repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_20260914.jsonl"
STAGE1="$DATA_DIR/stage1_general.json"
STAGE1_STATS="$WORKTREE/repro_1p7b/graph_frontier/reports/dynamic_v1_stage1_stats.json"
STAGE2="$DATA_DIR/stage2_targeted.json"
CAPABILITY="$DATA_DIR/stage1_capability_map.json"
PLAN="$DATA_DIR/stage2_allocation_plan.json"
STATS="$DATA_DIR/stage2_dataset_stats.json"
VALIDATION="$DATA_DIR/stage2_validation.json"

cd "$WORKTREE" || exit 2

set_status() { mkdir -p "$LOG_DIR"; printf '%s\n' "$1" > "$STATUS"; printf '%s %s\n' "$(date -Is)" "$1" >> "$LOG_DIR/state_history.log"; }
record_rc() { printf '%s\n' "$2" > "$LOG_DIR/$1.exit_code"; }
fail() { printf '%s\n' "$2" > "$LOG_DIR/$1.exit_code"; set_status FAILED; echo "$1 failed with rc=$2" >&2; exit "$2"; }
latest_checkpoint() { find "$OUTPUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -1; }
checkpoint_step() { basename "$1" | sed 's/checkpoint-//'; }

if [[ "$MODE" == "start" ]]; then
  for path in "$OUTPUT" "$LOG_DIR" "$DIAG_ROOT" "$STAGE2" "$CAPABILITY" "$PLAN" "$STATS" "$VALIDATION"; do
    [[ ! -e "$path" ]] || { echo "Refusing to overwrite formal artifact: $path" >&2; exit 3; }
  done
fi
[[ -s "$STAGE1" ]] || { echo "Missing Stage-1 dataset: $STAGE1" >&2; exit 4; }
[[ -s "$STAGE1_STATS" ]] || { echo "Missing Stage-1 stats: $STAGE1_STATS" >&2; exit 4; }
mkdir -p "$LOG_DIR"

"$RUN_PY" -m repro_1p7b.graph_frontier.dynamic_v1 validate-stage1 \
  --stage1 "$STAGE1" --stats "$STAGE1_STATS" --report "$LOG_DIR/stage1_validation.json"
rc=$?; record_rc stage1_validation "$rc"; [[ "$rc" -eq 0 ]] || fail stage1_validation "$rc"

"$RUN_PY" -m repro_1p7b.graph_frontier.continuity_audit config \
  --stage1 repro_1p7b/configs/llamafactory_dynamic_v1_stage1.yaml \
  --stage2 repro_1p7b/configs/llamafactory_dynamic_v1_stage2.yaml \
  --report "$LOG_DIR/config_audit.json"
rc=$?; record_rc config_audit "$rc"; [[ "$rc" -eq 0 ]] || fail config_audit "$rc"

boundary="$OUTPUT/checkpoint-207"
if [[ ! -s "$boundary/trainer_state.json" ]]; then
  set_status STAGE1_TRAINING
  set +u
  source /opt/conda/etc/profile.d/conda.sh || fail conda_source $?
  conda activate "$TRAIN_ENV" || fail conda_activate_train $?
  set -u
  export CUDA_HOME="$CONDA_PREFIX"
  export PATH="$CUDA_HOME/bin:$PATH"
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  resume_args=()
  latest=$(latest_checkpoint)
  if [[ -n "$latest" ]]; then resume_args=(--resume-from-checkpoint "$OUTPUT/$latest"); fi
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 --master-port=29523 \
    -m repro_1p7b.graph_frontier.train_boundary \
    repro_1p7b/configs/llamafactory_dynamic_v1_stage1.yaml --stop-after-step 207 \
    "${resume_args[@]}" 2>&1 | tee -a "$LOG_DIR/stage1.log"
  rc=${PIPESTATUS[0]}; record_rc stage1_train "$rc"; [[ "$rc" -eq 0 ]] || fail stage1_train "$rc"
fi
"$RUN_PY" -m repro_1p7b.graph_frontier.continuity_audit checkpoint \
  --checkpoint "$boundary" --step 207 --horizon 414 --report "$LOG_DIR/stage1_checkpoint_audit.json"
rc=$?; record_rc stage1_checkpoint_audit "$rc"; [[ "$rc" -eq 0 ]] || fail stage1_checkpoint_audit "$rc"
set_status STAGE1_DONE
mkdir -p "$OUTPUT/stage1_boundary_metadata"
for name in trainer_state.json scheduler.pt training_args.bin rng_state_0.pth rng_state_1.pth; do
  [[ -f "$boundary/$name" ]] && cp -p "$boundary/$name" "$OUTPUT/stage1_boundary_metadata/$name"
done
find "$boundary" -maxdepth 2 -type f -printf '%P %s\n' | sort > "$OUTPUT/stage1_boundary_metadata/inventory.txt"

diag_summary="$DIAG_ROOT/stage1/run_summary.json"
diag_valid=0
[[ -s "$diag_summary" ]] && diag_valid=$("$RUN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("valid",0))' "$diag_summary")
if [[ "$diag_valid" -lt 228 ]]; then
  set_status DIAGNOSING
  mkdir -p "$DIAG_ROOT/stage1"; port=1073
  if "$RUN_PY" -c 'import socket,sys; s=socket.socket(); s.settimeout(.3); sys.exit(1 if s.connect_ex(("127.0.0.1",int(sys.argv[1])))==0 else 0)' "$port"; then :; else fail diagnosis_port_busy 7; fi
  export CUDA_HOME="$SERVER_ENV" PATH="$SERVER_ENV/bin:$PATH"
  export CC="$SERVER_ENV/bin/x86_64-conda-linux-gnu-cc" CXX="$SERVER_ENV/bin/x86_64-conda-linux-gnu-c++"
  export GCC="$SERVER_ENV/bin/x86_64-conda-linux-gnu-gcc" GXX="$SERVER_ENV/bin/x86_64-conda-linux-gnu-g++"
  export NVCC_PREPEND_FLAGS="-ccbin=$CXX"
  CUDA_VISIBLE_DEVICES=0 "$SERVER_ENV/bin/python" -m sglang.launch_server \
    --model-path "$boundary" --host 127.0.0.1 --port "$port" --api-key dynamic-v1-stage1 \
    --dtype bfloat16 --tp-size 1 --mem-fraction-static 0.7 --disable-cuda-graph \
    --attention-backend triton --sampling-backend pytorch --context-length 32768 \
    --max-total-tokens 49152 --trust-remote-code > "$DIAG_ROOT/stage1/server.log" 2>&1 &
  server_pid=$!
  ready=0
  for _ in $(seq 1 120); do
    if curl -sf -H 'Authorization: Bearer dynamic-v1-stage1' "http://127.0.0.1:$port/v1/models" >/dev/null; then ready=1; break; fi
    kill -0 "$server_pid" 2>/dev/null || break; sleep 5
  done
  [[ "$ready" -eq 1 ]] || { kill "$server_pid" 2>/dev/null || true; fail diagnosis_server 8; }
  export LITELLM_LOCAL_MODEL_COST_MAP=True SGLANG_BASE_URL="http://127.0.0.1:$port/v1"
  export SGLANG_API_KEY=dynamic-v1-stage1 SGLANG_MODEL="$boundary"
  cleanup_server() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup_server EXIT INT TERM
  "$RUN_PY" -m repro_1p7b.graph_frontier.dynamic_v1 diagnose --manifest "$MANIFEST" \
    --split diagnosis --label stage1 --model-path "$boundary" --output "$DIAG_ROOT/stage1" > "$DIAG_ROOT/stage1/run.log" 2>&1
  rc=$?; cleanup_server; trap - EXIT INT TERM
  record_rc diagnosis "$rc"; [[ "$rc" -eq 0 ]] || fail diagnosis "$rc"
fi
diag_valid=$("$RUN_PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x.get("valid",0) if x.get("split")=="diagnosis" else 0)' "$diag_summary")
[[ "$diag_valid" -ge 228 ]] || fail diagnosis_valid_gate 9

if [[ ! -e "$STAGE2" && ! -e "$CAPABILITY" && ! -e "$PLAN" && ! -e "$STATS" && ! -e "$VALIDATION" ]]; then
  set_status BUILDING_STAGE2
  "$RUN_PY" -m repro_1p7b.graph_frontier.dynamic_v1 capability --manifest "$MANIFEST" \
    --result-root "$DIAG_ROOT" --label stage1 --output "$CAPABILITY"
  rc=$?; record_rc capability_map "$rc"; [[ "$rc" -eq 0 ]] || fail capability_map "$rc"
  "$RUN_PY" -m repro_1p7b.graph_frontier.dynamic_v1 stage2 \
    --source repro_1p7b/datasets/EnvFactory-SFT-FILTERED/mcp_factory_sft_nips.json \
    --features repro_1p7b/results/parameter_aware/baseline_features.jsonl \
    --stage1 "$STAGE1" --capability "$CAPABILITY" --output "$STAGE2" --plan "$PLAN" --stats "$STATS"
  rc=$?; record_rc stage2_build "$rc"; [[ "$rc" -eq 0 ]] || fail stage2_build "$rc"
  "$RUN_PY" -m repro_1p7b.graph_frontier.dynamic_v1 validate \
    --stage1 "$STAGE1" --stage2 "$STAGE2" --plan "$PLAN" --stats "$STATS" --report "$VALIDATION"
  rc=$?; record_rc stage2_validation "$rc"; [[ "$rc" -eq 0 ]] || fail stage2_validation "$rc"
elif [[ ! -s "$VALIDATION" ]] || ! grep -q '"status": "PASS"' "$VALIDATION"; then
  echo "Existing Stage-2 artifacts are incomplete; refusing overwrite." >&2; set_status FAILED; exit 10
fi
set_status STAGE2_READY

final_step=0
[[ -s "$OUTPUT/trainer_state.json" ]] && final_step=$("$RUN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("global_step",0))' "$OUTPUT/trainer_state.json")
if [[ "$final_step" -lt 414 ]]; then
  set_status STAGE2_TRAINING
  set +u
  source /opt/conda/etc/profile.d/conda.sh || fail conda_source $?
  conda activate "$TRAIN_ENV" || fail conda_activate_train $?
  set -u
  export CUDA_HOME="$CONDA_PREFIX"
  export PATH="$CUDA_HOME/bin:$PATH"
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  resume="$boundary"
  latest=$(latest_checkpoint)
  if [[ -n "$latest" ]] && [[ $(checkpoint_step "$latest") -gt 207 ]]; then resume="$OUTPUT/$latest"; fi
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 --master-port=29524 \
    -m repro_1p7b.graph_frontier.train_boundary \
    repro_1p7b/configs/llamafactory_dynamic_v1_stage2.yaml \
    --resume-from-checkpoint "$resume" --ignore-data-skip 2>&1 | tee -a "$LOG_DIR/stage2.log"
  rc=${PIPESTATUS[0]}; record_rc stage2_train "$rc"; [[ "$rc" -eq 0 ]] || fail stage2_train "$rc"
fi
final_step=$("$RUN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("global_step",0))' "$OUTPUT/trainer_state.json")
[[ "$final_step" -eq 414 ]] || fail final_step_gate 11
set_status TRAINING_DONE
record_rc orchestrator 0
