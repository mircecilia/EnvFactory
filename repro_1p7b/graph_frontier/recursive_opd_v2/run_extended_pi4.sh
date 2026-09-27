#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/recursive_opd_v2_run
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
STATUS=$RUN/extended_pi4_status
cd "$ROOT"
exec 9>"$RUN/.extended_pi4.lock"
flock -n 9 || { echo 'V2 pi4 pipeline already active' >&2; exit 2; }
stage=WAIT_PI2
on_exit() {
  code=$?
  if (( code != 0 )); then
    printf 'FAILED_%s:%s\n' "$stage" "$code" > "$STATUS"
  fi
}
trap on_exit EXIT
export PATH=$ENV/bin:$PATH
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

printf '%s\n' "$stage" > "$STATUS"
for attempt in $(seq 1 17280); do
  if [[ -f $RUN/cycle2/status ]]; then
    current=$(<"$RUN/cycle2/status")
    case "$current" in
      ROLLOUT_COMPLETE) break ;;
      FAILED*) echo "pi2 upstream failed: $current" >&2; exit 3 ;;
    esac
  fi
  if ! tmux has-session -t opd_v2_d1_to_pi2_retry 2>/dev/null; then
    echo 'pi2 upstream tmux disappeared before a complete rollout' >&2
    exit 3
  fi
  sleep 10
done
[[ -f $RUN/cycle2/status && $(<"$RUN/cycle2/status") == ROLLOUT_COMPLETE ]] || {
  echo 'timed out waiting for complete pi2 rollout' >&2
  exit 3
}

for cycle in 2 3; do
  next=$((cycle + 1))
  stage=AUDIT_D$cycle
  printf '%s\n' "$stage" > "$STATUS"
  bash repro_1p7b/graph_frontier/recursive_opd_v2/run_cpu_mine.sh "$cycle" \
    > "$RUN/cycle$cycle/mine_launcher.log" 2>&1
  [[ $(<"$RUN/cycle$cycle/status") == AUDITED ]] || exit 4

  stage=SMOKE_PI$next
  printf '%s\n' "$stage" > "$STATUS"
  bash repro_1p7b/graph_frontier/recursive_opd_v2/run_gpu_smoke.sh "$cycle" \
    > "$RUN/cycle$cycle/smoke_launcher.log" 2>&1
  [[ $(<"$RUN/gpu_smoke/cycle${cycle}_status") == PASS ]] || exit 5

  stage=TRAIN_PI$next
  printf '%s\n' "$stage" > "$STATUS"
  bash repro_1p7b/graph_frontier/recursive_opd_v2/run_gpu_train.sh "$cycle" \
    > "$RUN/cycle$cycle/train_stage_launcher.log" 2>&1
  [[ $(<"$RUN/cycle$cycle/train_status") == TRAINED ]] || exit 6

  stage=ROLLOUT_PI$next
  printf '%s\n' "$stage" > "$STATUS"
  bash repro_1p7b/graph_frontier/recursive_opd_v2/run_gpu_rollout.sh "$next" \
    > "$RUN/cycle$next/rollout_launcher.log" 2>&1
  [[ $(<"$RUN/cycle$next/status") == ROLLOUT_COMPLETE ]] || exit 7
done

stage=AUDIT_D4
printf '%s\n' "$stage" > "$STATUS"
bash repro_1p7b/graph_frontier/recursive_opd_v2/run_cpu_mine.sh 4 \
  > "$RUN/cycle4/mine_launcher.log" 2>&1
[[ $(<"$RUN/cycle4/status") == AUDITED ]] || exit 8

stage=COMPARE_INTERNAL
printf '%s\n' "$stage" > "$STATUS"
mkdir -p "$RUN/comparison"
"$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v2.compare_extended \
  > "$RUN/comparison/four_cycle_launcher.log" 2>&1

stage=EVALUATE_PI4
printf '%s\n' "$stage" > "$STATUS"
bash repro_1p7b/graph_frontier/recursive_opd_v2/run_external_pi4.sh \
  > "$RUN/evaluate_pi4_launcher.log" 2>&1
[[ $(<"$RUN/evaluation/pi4/status") == AUDITED ]] || exit 9

stage=COMPLETE
printf '%s\n' "$stage" > "$STATUS"
echo 'V2 pi4 full rollout, internal audit, Frozen300 and Rich24 complete'
