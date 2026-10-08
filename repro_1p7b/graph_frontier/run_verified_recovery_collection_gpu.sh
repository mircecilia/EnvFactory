#!/usr/bin/env bash
# User-authorized two-GPU INFERENCE ONLY. Same immutable 399-task subset.
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run/gpu_sglang_v3
MODEL=$ROOT/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
cd "$ROOT"; mkdir -p "$RUN"
exec 9>"$RUN/.lock"; flock -n 9 || exit 2
export PATH=$ENV/bin:$PATH PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
MOD=repro_1p7b.graph_frontier.verified_recovery_gpu_v3
ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)_$$
OWNED=(); WORKERS=()
cleanup(){ for pid in "${OWNED[@]}"; do kill -- "-$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
trap 'printf "GPU_COLLECTION_FAILED:%s\n" "$LINENO" > "$RUN/collection_status"' ERR
"$ENV/bin/python" -m unittest repro_1p7b.graph_frontier.tests.test_verified_recovery_feasibility repro_1p7b.graph_frontier.tests.test_verified_recovery_metrics repro_1p7b.graph_frontier.tests.test_verified_recovery_gpu repro_1p7b.graph_frontier.tests.test_verified_recovery_source_order repro_1p7b.graph_frontier.tests.test_verified_recovery_server_order > "$RUN/tests_${ATTEMPT}.log" 2>&1
"$ENV/bin/python" - <<'PY'
import socket,subprocess
import json
from pathlib import Path
diagnostic=Path('repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run/gpu_sglang_v2/diagnostics/github1038_prefix_validation.json')
check=json.loads(diagnostic.read_text())
assert check['status']=='PASS' and len(check['results'])==7 and all(x['PASS'] for x in check['results'])
schema_check=json.loads((diagnostic.parent/'multi_server0656_schema_order.json').read_text())
assert schema_check['status']=='PASS' and schema_check['actual_schema_names']==schema_check['expected_schema_names']
combined=json.loads(Path('repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run/gpu_sglang_v3/diagnostics/source_and_server_regressions.json').read_text())
assert combined['status']=='PASS' and len(combined['results'])==12 and all(x['status']=='PASS' for x in combined['results'])
plan=json.loads(Path('repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run/subset_plan.json').read_text())
assert {'official-rl-1038-86717bd5193b','official-rl-0656-b3222619f7d8'}<=set(plan['task_ids'][:12])
for port in (8260,8261):
    with socket.socket() as s:s.bind(('127.0.0.1',port))
for row in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines():
    assert int(row.strip())<100,'GPU occupied; no preemption allowed'
PY
"$ENV/bin/python" -m "$MOD" protocol > "$RUN/protocol_${ATTEMPT}.log"
printf 'GPU_SERVERS_STARTING\n' > "$RUN/collection_status"
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu setsid "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name dynamic-v1 --dtype bfloat16 \
    --host 127.0.0.1 --port "$((8260+gpu))" --context-length 32768 \
    --mem-fraction-static 0.60 --attention-backend triton \
    --sampling-backend pytorch --disable-cuda-graph \
    --reasoning-parser qwen3 --tool-call-parser qwen25 --api-key placeholder \
    > "$RUN/server_gpu${gpu}_${ATTEMPT}.log" 2>&1 &
  OWNED+=("$!")
done
"$ENV/bin/python" - "${OWNED[@]}" <<'PY'
import os,sys,time,urllib.request
for port,pid in zip((8260,8261),map(int,sys.argv[1:])):
    for _ in range(420):
        os.kill(pid,0)
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
                if r.status==200:print(f'{port} READY',flush=True);break
        except Exception:time.sleep(1)
    else:raise SystemExit(f'{port} readiness timeout')
PY
for mode in smoke12 full399; do
  printf 'GPU_%s\n' "$mode" > "$RUN/collection_status"
  args=(); [[ $mode == full399 ]] || args=(--limit 6)
  WORKERS=()
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES='' VERIFIED_RECOVERY_ENDPOINT="http://127.0.0.1:$((8260+shard))/v1" \
      setsid "$ENV/bin/python" -m "$MOD" worker --shard "$shard" "${args[@]}" \
      > "$RUN/worker_s${shard}_${mode}_${ATTEMPT}.log" 2>&1 &
    WORKERS+=("$!"); OWNED+=("$!")
  done
  # Bash 4.x compatible AND fail-fast on either shard, without wait -p.
  while ((${#WORKERS[@]})); do
    remaining=()
    for pid in "${WORKERS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then remaining+=("$pid"); else wait "$pid"; fi
    done
    WORKERS=("${remaining[@]}")
    if ((${#WORKERS[@]})); then sleep 2; fi
  done
  args=(); [[ $mode == full399 ]] || args=(--expected 12)
  "$ENV/bin/python" -m "$MOD" metrics "${args[@]}" > "$RUN/metrics_${mode}_${ATTEMPT}.log"
done
printf 'PAIRED_COLLECTION_COMPLETE_MANUAL_CASE_REVIEW_PENDING\n' > "$RUN/collection_status"
