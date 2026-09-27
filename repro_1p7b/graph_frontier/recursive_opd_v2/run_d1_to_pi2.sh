#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/recursive_opd_v2_run
cd "$ROOT"
exec 9>"$RUN/cycle1/.d1_to_pi2.lock"
flock -n 9 || { echo 'D1-to-pi2 chain already active' >&2; exit 2; }
[[ -d $RUN/cycle1/checkpoint && ! -e $RUN/cycle2/checkpoint ]] || {
  echo 'pi1 checkpoint missing or pi2 checkpoint already exists' >&2; exit 2;
}

for attempt in $(seq 1 1440); do
  status=$(<"$RUN/cycle1/status")
  case "$status" in
    AUDITED) break ;;
    ROLLOUT_COMPLETE|MINING|AUDITING) sleep 10 ;;
    *) echo "D1 gate did not pass: $status" >&2; exit 3 ;;
  esac
done
[[ $(<"$RUN/cycle1/status") == AUDITED ]] || {
  echo 'timed out waiting for audited D1' >&2; exit 3;
}

echo 'D1 AUDITED; starting two-GPU optimizer-free smoke'
bash repro_1p7b/graph_frontier/recursive_opd_v2/run_gpu_smoke.sh 1
[[ $(<"$RUN/gpu_smoke/cycle1_status") == PASS ]] || exit 4
echo 'cycle1 GPU smoke PASS; starting 64-step two-GPU full-parameter training'
bash repro_1p7b/graph_frontier/recursive_opd_v2/run_gpu_train.sh 1
[[ $(<"$RUN/cycle1/train_status") == TRAINED ]] || exit 5
echo 'pi2 checkpoint TRAINED and validated; starting fresh pi2 rollout'
bash repro_1p7b/graph_frontier/recursive_opd_v2/run_gpu_rollout.sh 2
echo 'pi2 rollout complete; CPU mining and audit require a separate gate'
