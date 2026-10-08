#!/usr/bin/env bash
# CPU-only paired collection; NO SGLang server, GPU process, optimizer or training.
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run
cd "$ROOT"; mkdir -p "$RUN"
exec 9>"$RUN/.collection_lock"; flock -n 9 || exit 2
export PATH=$ENV/bin:$PATH PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
MOD=repro_1p7b.graph_frontier.verified_recovery_collect
METRICS=repro_1p7b.graph_frontier.verified_recovery_metrics
ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)_$$
OWNED=()
cleanup(){ for pid in "${OWNED[@]}"; do kill -- "-$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
trap 'printf "COLLECTION_FAILED:%s\n" "$LINENO" > "$RUN/collection_status"' ERR
"$ENV/bin/python" -m unittest repro_1p7b.graph_frontier.tests.test_verified_recovery_feasibility repro_1p7b.graph_frontier.tests.test_verified_recovery_metrics > "$RUN/collection_tests_${ATTEMPT}.log" 2>&1
"$ENV/bin/python" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
assert json.loads((r/'subset_preflight_summary.json').read_text())['status']=='SUBSET_READY'
probe=json.loads((r/'cpu_policy_kv_probe_v2.json').read_text())
assert probe['status']=='CPU_POLICY_PROBE_PASS'
assert probe['samples'][0]['audit']['generated_token_ids']==probe['samples'][1]['audit']['generated_token_ids']
PY
"$ENV/bin/python" -m "$MOD" protocol > "$RUN/collection_protocol_${ATTEMPT}.log"
for mode in smoke4 full399; do
  printf 'CPU_%s\n' "$mode" > "$RUN/collection_status"
  args=(); [[ $mode == full399 ]] || args=(--limit 2)
  for shard in 0 1; do
    setsid "$ENV/bin/python" -m "$MOD" worker --shard "$shard" --threads 4 "${args[@]}" > "$RUN/worker_s${shard}_${mode}_${ATTEMPT}.log" 2>&1 &
    OWNED+=("$!")
  done
  for pid in "${OWNED[@]}"; do wait "$pid"; done; OWNED=()
  args=(); [[ $mode == full399 ]] || args=(--expected 4)
  "$ENV/bin/python" -m "$METRICS" "${args[@]}" > "$RUN/metrics_${mode}_${ATTEMPT}.log"
done
printf 'PAIRED_COLLECTION_COMPLETE_MANUAL_CASE_REVIEW_PENDING\n' > "$RUN/collection_status"
# Every worker owns/closes its MCP clients. No process or model survives this script.
