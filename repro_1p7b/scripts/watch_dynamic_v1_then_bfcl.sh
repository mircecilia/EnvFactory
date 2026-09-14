#!/usr/bin/env bash
set -u
set -o pipefail

WORKTREE="${WORKTREE:-/home/u2024311031/workspace/envfactory_repro_1p7b}"
PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-$WORKTREE/repro_1p7b/logs/graph_frontier_dynamic_v1}"
PIPELINE_STATUS_PATH="${PIPELINE_STATUS_PATH:-$PIPELINE_LOG_DIR/status}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$WORKTREE/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b}"
BFCL_RUN_ROOT="${BFCL_RUN_ROOT:-$WORKTREE/repro_1p7b/evaluation/bfcl/artifacts/full/dynamic_v1}"
BFCL_LAUNCHER="${BFCL_LAUNCHER:-$WORKTREE/repro_1p7b/evaluation/bfcl/scripts/run_full_dynamic_v1.sh}"
WATCHER_LOG="${WATCHER_LOG:-$PIPELINE_LOG_DIR/post_train_bfcl.log}"
WATCHER_STATUS_PATH="${WATCHER_STATUS_PATH:-$PIPELINE_LOG_DIR/post_train_bfcl.status}"
WATCHER_EXIT_PATH="${WATCHER_EXIT_PATH:-$PIPELINE_LOG_DIR/post_train_bfcl.exit_code}"
LOCK_PATH="${LOCK_PATH:-$PIPELINE_LOG_DIR/post_train_bfcl.lock}"
POLL_SECONDS="${POLL_SECONDS:-20}"
ORCHESTRATOR_GRACE_SECONDS="${ORCHESTRATOR_GRACE_SECONDS:-60}"
CUDA_TEARDOWN_SECONDS="${CUDA_TEARDOWN_SECONDS:-45}"
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_ORCHESTRATOR_ALIVE="${DRY_RUN_ORCHESTRATOR_ALIVE:-1}"
DRY_RUN_GPU0_IDLE="${DRY_RUN_GPU0_IDLE:-1}"
DRY_RUN_LAUNCHER_RC="${DRY_RUN_LAUNCHER_RC:-0}"

ORCHESTRATOR_PATTERN='[r]epro_1p7b/scripts/run_graph_frontier_dynamic_v1[.]sh (start|resume)'
RELATED_PROCESS_PATTERNS=(
  '[r]epro_1p7b[.]graph_frontier[.]train_boundary.*llamafactory_dynamic_v1_stage[12][.]yaml'
  '[t]orchrun.*master-port=2952(3|4).*graph_frontier[.]train_boundary'
  '[r]epro_1p7b[.]graph_frontier[.]dynamic_v1 (diagnose|capability|stage2|validate)'
  '[s]glang[.]launch_server.*graph_frontier_dynamic_v1_8k_1p7b'
)
FINAL_RC_NAMES=(
  stage1_validation
  config_audit
  stage1_train
  stage1_checkpoint_audit
  diagnosis
  capability_map
  stage2_build
  stage2_validation
  stage2_train
  orchestrator
)

mkdir -p "$(dirname "$WATCHER_LOG")" "$(dirname "$WATCHER_STATUS_PATH")" "$(dirname "$WATCHER_EXIT_PATH")"
touch "$WATCHER_LOG"
exec >>"$WATCHER_LOG" 2>&1

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

set_status() {
  printf '%s\n' "$1" >"$WATCHER_STATUS_PATH"
  log "STATUS $1"
}

finish() {
  local rc="$1"
  printf '%s\n' "$rc" >"$WATCHER_EXIT_PATH"
  exit "$rc"
}

fail_terminal() {
  local status="$1"
  local rc="$2"
  shift 2
  log "$*"
  set_status "$status"
  finish "$rc"
}

read_pipeline_status() {
  if [ -s "$PIPELINE_STATUS_PATH" ]; then
    tr -d '[:space:]' <"$PIPELINE_STATUS_PATH"
  else
    printf '%s' MISSING
  fi
}

rc_failure_reason=""
check_present_exit_codes() {
  local name
  local path
  local value
  rc_failure_reason=""
  for name in "${FINAL_RC_NAMES[@]}"; do
    path="$PIPELINE_LOG_DIR/$name.exit_code"
    [ -f "$path" ] || continue
    value="$(tr -d '[:space:]' <"$path")"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
      rc_failure_reason="$name has invalid exit code: $value"
      return 1
    fi
    if [ "$value" -ne 0 ]; then
      rc_failure_reason="$name failed with exit code $value"
      return 1
    fi
  done
  return 0
}

final_exit_codes_complete() {
  local name
  local value
  for name in "${FINAL_RC_NAMES[@]}"; do
    [ -f "$PIPELINE_LOG_DIR/$name.exit_code" ] || return 1
    value="$(tr -d '[:space:]' <"$PIPELINE_LOG_DIR/$name.exit_code")"
    [[ "$value" =~ ^[0-9]+$ ]] || return 1
    [ "$value" -eq 0 ] || return 1
  done
}

checkpoint_complete() {
  local required
  local weight
  local weights
  for required in config.json tokenizer_config.json tokenizer.json; do
    [ -s "$CHECKPOINT_DIR/$required" ] || return 1
  done
  shopt -s nullglob
  weights=("$CHECKPOINT_DIR"/*.safetensors)
  [ "${#weights[@]}" -gt 0 ] || return 1
  for weight in "${weights[@]}"; do
    [ -s "$weight" ] || return 1
  done
}

orchestrator_running() {
  if [ "$DRY_RUN" = 1 ]; then
    [ "$DRY_RUN_ORCHESTRATOR_ALIVE" = 1 ]
    return
  fi
  pgrep -u "$(id -u)" -f -- "$ORCHESTRATOR_PATTERN" >/dev/null
}

related_process_running() {
  local pattern
  if [ "$DRY_RUN" = 1 ]; then
    [ "$DRY_RUN_ORCHESTRATOR_ALIVE" = 1 ]
    return
  fi
  for pattern in "${RELATED_PROCESS_PATTERNS[@]}"; do
    if pgrep -u "$(id -u)" -f -- "$pattern" >/dev/null; then
      return 0
    fi
  done
  return 1
}

pipeline_activity_running() {
  orchestrator_running || related_process_running
}

gpu0_idle() {
  local output
  local pids
  if [ "$DRY_RUN" = 1 ]; then
    [ "$DRY_RUN_GPU0_IDLE" = 1 ]
    return
  fi
  if ! output="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"; then
    log "nvidia-smi GPU0 query failed; treating GPU0 state as unknown/busy"
    return 1
  fi
  pids="$(printf "%s" "$output" | tr -d "[:space:]")"
  [ -z "$pids" ]
}

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  log "duplicate watcher refused: lock is held at $LOCK_PATH"
  exit 6
fi

for value_name in DRY_RUN DRY_RUN_ORCHESTRATOR_ALIVE DRY_RUN_GPU0_IDLE; do
  value="${!value_name}"
  if [ "$value" != 0 ] && [ "$value" != 1 ]; then
    fail_terminal WATCHER_CONFIG_INVALID 2 "invalid $value_name=$value"
  fi
done
if ! [[ "$DRY_RUN_LAUNCHER_RC" =~ ^[0-9]+$ ]] || [ "$DRY_RUN_LAUNCHER_RC" -gt 255 ]; then
  fail_terminal WATCHER_CONFIG_INVALID 2 "invalid DRY_RUN_LAUNCHER_RC=$DRY_RUN_LAUNCHER_RC"
fi

log "Dynamic status: $PIPELINE_STATUS_PATH"
log "Dynamic final model: $CHECKPOINT_DIR"
log "Dynamic BFCL destination: $BFCL_RUN_ROOT"
log "Dynamic BFCL launcher: $BFCL_LAUNCHER"
set_status WAITING_FOR_DYNAMIC_PIPELINE

dead_since=0
while true; do
  if ! check_present_exit_codes; then
    fail_terminal TRAINING_FAILED 20 "$rc_failure_reason; BFCL will not start"
  fi

  pipeline_status="$(read_pipeline_status)"
  case "$pipeline_status" in
    TRAINING_DONE)
      if final_exit_codes_complete; then
        log "authoritative TRAINING_DONE and all required exit codes are zero"
        break
      fi
      log "TRAINING_DONE observed; waiting for all final zero exit-code files"
      ;;
    FAILED|*_FAILED)
      fail_terminal TRAINING_FAILED 20 "Dynamic pipeline status is $pipeline_status; BFCL will not start"
      ;;
    MISSING|STAGE1_TRAINING|STAGE1_DONE|DIAGNOSING|BUILDING_STAGE2|STAGE2_READY|STAGE2_TRAINING)
      ;;
    *)
      log "unknown non-terminal Dynamic status: $pipeline_status"
      ;;
  esac

  if pipeline_activity_running; then
    dead_since=0
  else
    now="$(date +%s)"
    if [ "$dead_since" -eq 0 ]; then
      dead_since="$now"
      log "no Dynamic orchestrator or related process found; starting ${ORCHESTRATOR_GRACE_SECONDS}s grace period"
    fi
    if [ $((now - dead_since)) -ge "$ORCHESTRATOR_GRACE_SECONDS" ]; then
      fail_terminal ORCHESTRATOR_DIED 21 "Dynamic pipeline is non-terminal but all related processes stayed absent"
    fi
  fi

  if [ "$DRY_RUN" = 1 ]; then
    log "DRY_RUN one-shot decision: wait at Dynamic status $pipeline_status"
    finish 0
  fi
  sleep "$POLL_SECONDS"
done

set_status WAITING_FOR_PROCESS_EXIT
while pipeline_activity_running; do
  if [ "$DRY_RUN" = 1 ]; then
    fail_terminal ORCHESTRATOR_DIED 21 "DRY_RUN success fixture still reports active Dynamic processes"
  fi
  sleep "$POLL_SECONDS"
done
log "all Dynamic orchestrator, train, diagnosis, build, and SGLang processes have exited"

set_status WAITING_FOR_FINAL_CHECKPOINT
if ! checkpoint_complete; then
  fail_terminal CHECKPOINT_INCOMPLETE 4 "final Dynamic model is incomplete or not directly inference-loadable: $CHECKPOINT_DIR"
fi
log "final Dynamic Hugging Face model completeness check passed"

set_status WAITING_FOR_CUDA_TEARDOWN
if [ "$DRY_RUN" = 0 ]; then
  sleep "$CUDA_TEARDOWN_SECONDS"
fi
while ! gpu0_idle; do
  set_status WAITING_FOR_GPU0
  log "GPU0 is occupied; refusing to steal it for BFCL"
  if [ "$DRY_RUN" = 1 ]; then
    fail_terminal GPU0_BUSY 7 "DRY_RUN fixture reports GPU0 busy"
  fi
  sleep "$POLL_SECONDS"
done
log "CUDA teardown elapsed and GPU0 is idle"

if [ -e "$BFCL_RUN_ROOT" ]; then
  fail_terminal REFUSE_EXISTING_RESULTS 5 "refusing to overwrite Dynamic BFCL results: $BFCL_RUN_ROOT"
fi
if [ ! -x "$BFCL_LAUNCHER" ]; then
  fail_terminal BFCL_FAILED 3 "BFCL launcher is missing or not executable: $BFCL_LAUNCHER"
fi

set_status STARTING_BFCL
if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN simulates BFCL launcher exit code $DRY_RUN_LAUNCHER_RC; no benchmark was started"
  bfcl_rc="$DRY_RUN_LAUNCHER_RC"
else
  set_status BFCL_RUNNING
  log "starting Dynamic BFCL launcher"
  "$BFCL_LAUNCHER"
  bfcl_rc=$?
fi

if [ "$bfcl_rc" -eq 0 ]; then
  log "Dynamic BFCL completed successfully"
  set_status DONE
else
  log "Dynamic BFCL failed with exit code $bfcl_rc"
  set_status BFCL_FAILED
fi
finish "$bfcl_rc"
