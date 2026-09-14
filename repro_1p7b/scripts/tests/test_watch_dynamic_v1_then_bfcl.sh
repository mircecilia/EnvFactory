#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WATCHER="$REPO_ROOT/repro_1p7b/scripts/watch_dynamic_v1_then_bfcl.sh"
DYNAMIC_LAUNCHER="$REPO_ROOT/repro_1p7b/evaluation/bfcl/scripts/run_full_dynamic_v1.sh"
RUN_BFCL="$REPO_ROOT/repro_1p7b/evaluation/bfcl/scripts/run_bfcl.sh"
TMP_ROOT="$(mktemp -d /tmp/dynamic-v1-watcher-tests.XXXXXX)"

cleanup() {
  case "$TMP_ROOT" in
    /tmp/dynamic-v1-watcher-tests.*) rm -rf -- "$TMP_ROOT" ;;
    *) echo "unsafe temp cleanup path: $TMP_ROOT" >&2; exit 99 ;;
  esac
}
trap cleanup EXIT

RC_NAMES=(
  stage1_validation config_audit stage1_train stage1_checkpoint_audit
  diagnosis capability_map stage2_build stage2_validation stage2_train orchestrator
)

make_model() {
  local path="$1"
  mkdir -p "$path"
  printf 'x\n' >"$path/config.json"
  printf 'x\n' >"$path/tokenizer_config.json"
  printf 'x\n' >"$path/tokenizer.json"
  printf 'x\n' >"$path/model.safetensors"
}

write_final_zero_exit_codes() {
  local log_dir="$1"
  local name
  for name in "${RC_NAMES[@]}"; do
    printf '0\n' >"$log_dir/$name.exit_code"
  done
}

run_case() {
  local name="$1"
  local pipeline_status="$2"
  local alive="$3"
  local launcher_rc="$4"
  local results_exist="$5"
  local complete_model="$6"
  local failed_rc_name="${7:-}"
  CASE_DIR="$TMP_ROOT/$name"
  local pipeline_log="$CASE_DIR/pipeline"
  local model="$CASE_DIR/model"
  local results="$CASE_DIR/results"
  mkdir -p "$pipeline_log"
  printf '%s\n' "$pipeline_status" >"$pipeline_log/status"
  if [ "$pipeline_status" = TRAINING_DONE ]; then
    write_final_zero_exit_codes "$pipeline_log"
  fi
  if [ -n "$failed_rc_name" ]; then
    printf '9\n' >"$pipeline_log/$failed_rc_name.exit_code"
  fi
  if [ "$complete_model" = 1 ]; then
    make_model "$model"
  fi
  if [ "$results_exist" = 1 ]; then
    mkdir -p "$results"
  fi

  set +e
  PIPELINE_LOG_DIR="$pipeline_log" \
  PIPELINE_STATUS_PATH="$pipeline_log/status" \
  CHECKPOINT_DIR="$model" \
  BFCL_RUN_ROOT="$results" \
  BFCL_LAUNCHER=/bin/true \
  WATCHER_LOG="$CASE_DIR/watcher.log" \
  WATCHER_STATUS_PATH="$CASE_DIR/watcher.status" \
  WATCHER_EXIT_PATH="$CASE_DIR/watcher.exit_code" \
  LOCK_PATH="$CASE_DIR/watcher.lock" \
  POLL_SECONDS=0 \
  ORCHESTRATOR_GRACE_SECONDS=0 \
  CUDA_TEARDOWN_SECONDS=0 \
  DRY_RUN=1 \
  DRY_RUN_ORCHESTRATOR_ALIVE="$alive" \
  DRY_RUN_GPU0_IDLE=1 \
  DRY_RUN_LAUNCHER_RC="$launcher_rc" \
    "$WATCHER"
  CASE_RC=$?
  set -e
}

assert_status() {
  local expected="$1"
  local actual
  actual="$(tr -d '[:space:]' <"$CASE_DIR/watcher.status")"
  [ "$actual" = "$expected" ] || {
    echo "expected watcher status $expected, got $actual in $CASE_DIR" >&2
    exit 1
  }
}

run_case case_a STAGE1_TRAINING 1 0 0 0
[ "$CASE_RC" -eq 0 ]; assert_status WAITING_FOR_DYNAMIC_PIPELINE
echo "PASS Case A: active Stage1 waits"

run_case case_b DIAGNOSING 1 0 0 0
[ "$CASE_RC" -eq 0 ]; assert_status WAITING_FOR_DYNAMIC_PIPELINE
echo "PASS Case B: diagnosis transition waits"

run_case case_c STAGE2_TRAINING 1 0 0 0
[ "$CASE_RC" -eq 0 ]; assert_status WAITING_FOR_DYNAMIC_PIPELINE
echo "PASS Case C: active Stage2 waits"

run_case case_d TRAINING_DONE 0 0 0 1
[ "$CASE_RC" -eq 0 ]; assert_status DONE
grep -q 'no benchmark was started' "$CASE_DIR/watcher.log"
echo "PASS Case D: complete terminal pipeline would launch"

for failure_spec in \
  'stage1_failure:STAGE1_TRAINING:stage1_train' \
  'diagnosis_failure:DIAGNOSING:diagnosis' \
  'stage2_failure:STAGE2_TRAINING:stage2_train'
do
  IFS=: read -r failure_name failure_status failure_rc_name <<<"$failure_spec"
  run_case "$failure_name" "$failure_status" 1 0 0 0 "$failure_rc_name"
  [ "$CASE_RC" -eq 20 ]; assert_status TRAINING_FAILED
done
echo "PASS Case E: explicit Stage1, diagnosis, and Stage2 failures block BFCL"

run_case case_f STAGE2_READY 0 0 0 0
[ "$CASE_RC" -eq 21 ]; assert_status ORCHESTRATOR_DIED
echo "PASS Case F: dead non-terminal orchestrator fails fast"

run_case case_g TRAINING_DONE 0 0 1 1
[ "$CASE_RC" -eq 5 ]; assert_status REFUSE_EXISTING_RESULTS
echo "PASS Case G: existing Dynamic BFCL results are protected"

run_case case_h TRAINING_DONE 0 17 0 1
[ "$CASE_RC" -eq 17 ]; assert_status BFCL_FAILED
[ "$(tr -d '[:space:]' <"$CASE_DIR/watcher.exit_code")" = 17 ]
echo "PASS Case H: BFCL failure and exit code are captured"

run_case case_i TRAINING_DONE 0 0 0 0
[ "$CASE_RC" -eq 4 ]; assert_status CHECKPOINT_INCOMPLETE
echo "PASS Case I: incomplete final model blocks BFCL"

LOCK_CASE="$TMP_ROOT/lock_case"
mkdir -p "$LOCK_CASE/pipeline"
printf 'STAGE1_TRAINING\n' >"$LOCK_CASE/pipeline/status"
exec 8>"$LOCK_CASE/watcher.lock"
flock -n 8
set +e
PIPELINE_LOG_DIR="$LOCK_CASE/pipeline" \
PIPELINE_STATUS_PATH="$LOCK_CASE/pipeline/status" \
WATCHER_LOG="$LOCK_CASE/watcher.log" \
WATCHER_STATUS_PATH="$LOCK_CASE/watcher.status" \
WATCHER_EXIT_PATH="$LOCK_CASE/watcher.exit_code" \
LOCK_PATH="$LOCK_CASE/watcher.lock" \
DRY_RUN=1 \
  "$WATCHER"
lock_rc=$?
set -e
exec 8>&-
[ "$lock_rc" -eq 6 ]
echo "PASS lock: duplicate watcher is refused"

set +e
label_output="$("$RUN_BFCL" dynamic_v1 /nonexistent invalid 2>&1)"
label_rc=$?
set -e
[ "$label_rc" -eq 2 ]
grep -q 'invalid scope' <<<"$label_output"
echo "PASS label: run_bfcl accepts dynamic_v1 before scope validation"

DRY_RUN=1 \
DYNAMIC_MODEL_PATH="$REPO_ROOT/repro_1p7b/models/Qwen3-1.7B" \
  "$DYNAMIC_LAUNCHER" >"$TMP_ROOT/launcher_dry_run.log"
grep -q 'DRY_RUN label=dynamic_v1' "$TMP_ROOT/launcher_dry_run.log"
echo "PASS launcher: read-only dry-run validates BFCL commit and existing model"

bash -n \
  "$WATCHER" \
  "$DYNAMIC_LAUNCHER" \
  "$RUN_BFCL" \
  "$REPO_ROOT/repro_1p7b/scripts/watch_parameter_aware_then_bfcl.sh" \
  "$REPO_ROOT/repro_1p7b/evaluation/bfcl/scripts/run_full_parameter_aware.sh"
echo "PASS compatibility: Dynamic and existing Parameter-Aware scripts parse"
