#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/verified_recovery_20261007_run/environment_v3
cd "$ROOT"; mkdir -p "$RUN"
exec 9>"$RUN/.determinism.lock"
flock -n 9 || exit 2
export PATH=$ENV/bin:$PATH PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
MOD=repro_1p7b.graph_frontier.verified_recovery_determinism
ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)_$$
OWNED=()
cleanup() { for pid in "${OWNED[@]}"; do kill -- "-$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
trap 'printf "DETERMINISM_FAILED:%s\n" "$LINENO" > "$RUN/status"' ERR
"$ENV/bin/python" -m "$MOD" plan > "$RUN/plan_${ATTEMPT}.log"
for group in fixed100 random300; do
  replicas=3; [[ $group == fixed100 ]] || replicas=2
  for ((replica=0;replica<replicas;replica++)); do
    printf 'CPU_%s_REPLICA%s\n' "$group" "$replica" > "$RUN/status"
    for shard in 0 1 2 3; do
      setsid "$ENV/bin/python" -m "$MOD" worker --group "$group" --replica "$replica" --shard "$shard" \
        > "$RUN/${group}_r${replica}_s${shard}_${ATTEMPT}.log" 2>&1 &
      OWNED+=("$!")
    done
    for pid in "${OWNED[@]}"; do wait "$pid"; done; OWNED=()
    "$ENV/bin/python" -m "$MOD" summary --group "$group" --replica "$replica" > "$RUN/gate_${group}_r${replica}_${ATTEMPT}.log"
  done
done
"$ENV/bin/python" -m "$MOD" summary > "$RUN/determinism_summary_${ATTEMPT}.log"
printf 'DETERMINISM_PASS\n' > "$RUN/status"
