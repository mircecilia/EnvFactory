#!/usr/bin/env bash
# Data-only bounded prototype: no optimizer, no training, no historic writes.
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/research_recovery_20261007_run
cd "$ROOT"
mkdir -p "$RUN"
exec 9>"$RUN/.launcher.lock"
flock -n 9 || { echo 'prototype launcher already running' >&2; exit 2; }
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
MOD=repro_1p7b.graph_frontier.research_recovery_20261007
ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)_$$
OWNED=()
cleanup() {
  for pid in "${OWNED[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
  for pid in "${OWNED[@]}"; do wait "$pid" 2>/dev/null || true; done
}
failed() { printf 'FAILED:%s\n' "$1" > "$RUN/status"; }
trap cleanup EXIT INT TERM
trap 'failed "$LINENO"' ERR
printf 'CPU_PREFLIGHT\n' > "$RUN/status"
"$ENV/bin/python" -m "$MOD" plan > "$RUN/plan_${ATTEMPT}.log" 2>&1
"$ENV/bin/python" -m "$MOD" preflight > "$RUN/preflight_${ATTEMPT}.log" 2>&1
"$ENV/bin/python" - <<'PY'
import socket,subprocess
for port in (8650,8651):
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',port))==0: raise SystemExit(f'port occupied: {port}')
used=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
if len(used)<2 or any(x>=1500 for x in used[:2]): raise SystemExit(f'GPUs busy: {used}')
PY
MODEL=$("$ENV/bin/python" -c "import json;print(json.load(open('$RUN/plan.json'))['model'])")
printf 'STARTING_SERVERS\n' > "$RUN/status"
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu setsid "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name research-recovery-pi0 \
    --host 127.0.0.1 --port "$((8650+gpu))" --context-length 32768 \
    --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "$RUN/server_gpu${gpu}_${ATTEMPT}.log" 2>&1 &
  OWNED+=("$!")
done
"$ENV/bin/python" - "${OWNED[@]}" <<'PY'
import os,sys,time,urllib.request
for port,pid in zip((8650,8651),map(int,sys.argv[1:])):
    for _ in range(420):
        os.kill(pid,0)
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
                if r.status==200: break
        except Exception: time.sleep(1)
    else: raise SystemExit(f'server readiness timeout: {port}')
PY
for mode in smoke full; do
  printf 'COLLECTING_%s\n' "$mode" > "$RUN/status"
  args=(); [[ $mode == full ]] || args=(--limit 2)
  for gpu in 0 1; do
    setsid "$ENV/bin/python" -m "$MOD" worker --shard "$gpu" \
      --endpoint "http://127.0.0.1:$((8650+gpu))/v1" "${args[@]}" \
      > "$RUN/worker_gpu${gpu}_${mode}_${ATTEMPT}.log" 2>&1 &
    OWNED+=("$!")
  done
  wait "${OWNED[2]}"; wait "${OWNED[3]}"
  OWNED=("${OWNED[0]}" "${OWNED[1]}")
  "$ENV/bin/python" -m "$MOD" summary > "$RUN/summary_${mode}_${ATTEMPT}.log"
  "$ENV/bin/python" - "$RUN" "$mode" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);v=json.load(open(run/'summary.json'))
if v['failed'] or v['state_mismatches']: raise SystemExit('Recovery execution/state gate failed')
if sys.argv[2]=='smoke' and v['completed']<4: raise SystemExit('Incomplete 4-task smoke')
if sys.argv[2]=='full' and v['completed']!=100: raise SystemExit('Incomplete 100-task prototype')
PY
done
cleanup; OWNED=()
printf 'PROTOTYPE_COMPLETE\n' > "$RUN/status"
