#!/usr/bin/env bash
set -u
set -o pipefail

WORKTREE="${WORKTREE:-/home/u2024311031/workspace/envfactory_repro_1p7b}"
TRAIN_EXIT_PATH="${TRAIN_EXIT_PATH:-$WORKTREE/repro_1p7b/logs/parameter_aware/formal_sft_8k_1p7b.exit_code}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$WORKTREE/repro_1p7b/checkpoints/parameter_aware_sft_8k_1p7b}"
BFCL_RUN_ROOT="${BFCL_RUN_ROOT:-$WORKTREE/repro_1p7b/evaluation/bfcl/artifacts/full/parameter_aware}"
BFCL_LAUNCHER="${BFCL_LAUNCHER:-$WORKTREE/repro_1p7b/evaluation/bfcl/scripts/run_full_parameter_aware.sh}"
WATCHER_LOG="${WATCHER_LOG:-$WORKTREE/repro_1p7b/logs/parameter_aware/post_train_bfcl_watcher.log}"
STATUS_PATH="${STATUS_PATH:-$WORKTREE/repro_1p7b/logs/parameter_aware/post_train_bfcl.status}"
BFCL_EXIT_PATH="${BFCL_EXIT_PATH:-$WORKTREE/repro_1p7b/logs/parameter_aware/post_train_bfcl.exit_code}"
LOCK_PATH="${LOCK_PATH:-$WORKTREE/repro_1p7b/logs/parameter_aware/post_train_bfcl.lock}"
TRAIN_PROCESS_PATTERN="${TRAIN_PROCESS_PATTERN:-llamafactory_sft_parameter_aware_8k_1p7b.yaml}"
POLL_SECONDS="${POLL_SECONDS:-20}"
CUDA_TEARDOWN_SECONDS="${CUDA_TEARDOWN_SECONDS:-45}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$(dirname "$WATCHER_LOG")" "$(dirname "$STATUS_PATH")" "$(dirname "$BFCL_EXIT_PATH")"
touch "$WATCHER_LOG"
exec >>"$WATCHER_LOG" 2>&1

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

set_status() {
  printf '%s\n' "$1" >"$STATUS_PATH"
  log "STATUS $1"
}

finish() {
  local rc="$1"
  printf '%s\n' "$rc" >"$BFCL_EXIT_PATH"
  exit "$rc"
}

checkpoint_complete() {
  local required
  local weight
  local weights
  for required in config.json tokenizer_config.json tokenizer.json; do
    if [ ! -s "$CHECKPOINT_DIR/$required" ]; then
      return 1
    fi
  done
  shopt -s nullglob
  weights=("$CHECKPOINT_DIR"/*.safetensors)
  if [ "${#weights[@]}" -eq 0 ]; then
    return 1
  fi
  for weight in "${weights[@]}"; do
    if [ ! -s "$weight" ]; then
      return 1
    fi
  done
}

training_is_running() {
  if [ "$DRY_RUN" = 1 ]; then
    return 1
  fi
  pgrep -f -- "$TRAIN_PROCESS_PATTERN" >/dev/null
}

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  log "another post-training BFCL watcher holds $LOCK_PATH; refusing duplicate watcher"
  exit 6
fi

if [ "$DRY_RUN" != 0 ] && [ "$DRY_RUN" != 1 ]; then
  log "invalid DRY_RUN value: $DRY_RUN"
  set_status BFCL_FAILED
  finish 2
fi

set_status WAITING_FOR_TRAIN
log "waiting for training exit code: $TRAIN_EXIT_PATH"
while [ ! -f "$TRAIN_EXIT_PATH" ]; do
  sleep "$POLL_SECONDS"
done

train_rc="$(tr -d '[:space:]' <"$TRAIN_EXIT_PATH")"
if ! [[ "$train_rc" =~ ^[0-9]+$ ]]; then
  log "invalid training exit code content: $train_rc"
  set_status TRAIN_FAILED
  finish 2
fi
log "observed training exit code: $train_rc"
if [ "$train_rc" -ne 0 ]; then
  log "TRAIN_FAILED; BFCL will not be started"
  set_status TRAIN_FAILED
  finish "$train_rc"
fi

if ! checkpoint_complete; then
  log "checkpoint is incomplete: $CHECKPOINT_DIR"
  set_status BFCL_FAILED
  finish 4
fi
log "checkpoint completeness check passed: $CHECKPOINT_DIR"

if training_is_running; then
  log "training exit code is present, but matching training processes remain; waiting for exit"
  while training_is_running; do
    sleep "$POLL_SECONDS"
  done
fi
log "matching training processes have exited"

log "waiting ${CUDA_TEARDOWN_SECONDS}s for CUDA teardown"
sleep "$CUDA_TEARDOWN_SECONDS"

if [ -e "$BFCL_RUN_ROOT" ]; then
  log "REFUSE_EXISTING_RESULTS: $BFCL_RUN_ROOT"
  set_status BFCL_FAILED
  finish 5
fi

set_status STARTING_BFCL
if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN would start BFCL: $BFCL_LAUNCHER"
  set_status DONE
  finish 0
fi
