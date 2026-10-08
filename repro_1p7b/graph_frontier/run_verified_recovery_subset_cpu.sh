#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run
cd "$ROOT"; mkdir -p "$RUN"
exec 9>"$RUN/.subset_lock"; flock -n 9 || exit 2
export PATH=$ENV/bin:$PATH PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1
MOD=repro_1p7b.graph_frontier.verified_recovery_feasibility
ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)_$$
OWNED=()
cleanup(){ for pid in "${OWNED[@]}"; do kill -- "-$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
trap 'printf "SUBSET_FAILED:%s\n" "$LINENO" > "$RUN/status"' ERR
"$ENV/bin/python" -m "$MOD" plan > "$RUN/plan_${ATTEMPT}.log"
printf 'CPU_SUBSET_REPLAY\n' > "$RUN/status"
for shard in 0 1 2 3; do
  setsid "$ENV/bin/python" -m "$MOD" preflight --shard "$shard" > "$RUN/subset_s${shard}_${ATTEMPT}.log" 2>&1 &
  OWNED+=("$!")
done
for pid in "${OWNED[@]}"; do wait "$pid"; done; OWNED=()
"$ENV/bin/python" -m "$MOD" summary > "$RUN/subset_summary_${ATTEMPT}.log"
printf 'SUBSET_READY\n' > "$RUN/status"
