#!/usr/bin/env bash
# User explicitly permits evidenced network attrition, silently and without retry.
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
PARENT=$ROOT/repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run/gpu_sglang_v3
RUN=$PARENT/network_skip_v1
MODEL=$ROOT/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
cd "$ROOT"; mkdir -p "$RUN"
exec 9>"$PARENT/.lock"; flock -n 9 || exit 2
export PATH=$ENV/bin:$PATH PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
MOD=repro_1p7b.graph_frontier.verified_recovery_network_skip
ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)_$$
OWNED=(); WORKERS=()
cleanup(){
  for pid in "${OWNED[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  sleep 3
  for pid in "${OWNED[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
trap 'printf "NON_NETWORK_PIPELINE_FAILED:%s\n" "$LINENO" > "$RUN/status"' ERR
"$ENV/bin/python" -m unittest repro_1p7b.graph_frontier.tests.test_verified_recovery_feasibility repro_1p7b.graph_frontier.tests.test_verified_recovery_metrics repro_1p7b.graph_frontier.tests.test_verified_recovery_gpu repro_1p7b.graph_frontier.tests.test_verified_recovery_source_order repro_1p7b.graph_frontier.tests.test_verified_recovery_server_order repro_1p7b.graph_frontier.tests.test_verified_recovery_resume_diagnostics repro_1p7b.graph_frontier.tests.test_verified_recovery_network_skip > "$RUN/tests_${ATTEMPT}.log" 2>&1
"$ENV/bin/python" - <<'PY'
import socket,subprocess,json
from pathlib import Path
p=Path('repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run/gpu_sglang_v3')
smoke=json.loads((p/'smoke12_metrics.json').read_text())
assert smoke['status']=='COMPLETE' and smoke['completed']==12
for port in (8260,8261):
    with socket.socket() as s:s.bind(('127.0.0.1',port))
for line in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines():
    assert int(line.strip())<100,'GPU occupied; no preemption'
PY
if [[ ! -f "$RUN/scope_protocol.json" ]]; then
  "$ENV/bin/python" -m "$MOD" prepare > "$RUN/prepare_${ATTEMPT}.log"
fi
printf 'NETWORK_SKIP_SERVERS_STARTING\n' > "$RUN/status"
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu setsid "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name dynamic-v1 --dtype bfloat16 \
    --host 127.0.0.1 --port "$((8260+gpu))" --context-length 32768 \
    --mem-fraction-static 0.60 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 --api-key placeholder \
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
printf 'COLLECT_AVAILABLE_REMAINING\n' > "$RUN/status"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES='' VERIFIED_RECOVERY_ENDPOINT="http://127.0.0.1:$((8260+shard))/v1" \
    setsid "$ENV/bin/python" -m "$MOD" worker --shard "$shard" \
    > "$RUN/worker_s${shard}_${ATTEMPT}.log" 2>&1 &
  WORKERS+=("$!"); OWNED+=("$!")
done
while ((${#WORKERS[@]})); do
  remaining=()
  for pid in "${WORKERS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then remaining+=("$pid"); else wait "$pid"; fi
  done
  WORKERS=("${remaining[@]}")
  if ((${#WORKERS[@]})); then sleep 2; fi
done
printf 'AVAILABLE_COHORT_SUMMARIZING\n' > "$RUN/status"
"$ENV/bin/python" -m "$MOD" metrics > "$RUN/metrics_${ATTEMPT}.log"
printf 'AVAILABLE_COHORT_COMPLETE_MANUAL_REVIEW_PENDING\n' > "$RUN/status"
