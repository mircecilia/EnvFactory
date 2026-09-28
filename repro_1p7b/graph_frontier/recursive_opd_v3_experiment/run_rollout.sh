#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN="$ROOT/repro_1p7b/graph_frontier/recursive_opd_v3_experiment_run"
CYCLE=${1:?policy cycle 1..4 required}
[[ "$CYCLE" =~ ^[1-4]$ ]] || exit 2
LOG="$RUN/cycle$CYCLE"
MODEL="$LOG/checkpoint"
cd "$ROOT"
mkdir -p "$LOG"
exec 9>"$LOG/.rollout.lock"
flock -n 9 || { echo 'Complete-C rollout launcher already active' >&2; exit 2; }
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}"
export ENVFACTORY_ROOT="$ROOT" ENVFACTORY_REWARD_MODE=official
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v3_experiment.rollout \
  lock --cycle "$CYCLE" > "$LOG/rollout_lock.log"
"$ENV/bin/python" - <<'PY'
import socket, subprocess
for port in (8550, 8551):
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise SystemExit(f'port {port} occupied')
used = [int(x.strip()) for x in subprocess.check_output(
    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
    text=True).splitlines()]
if len(used) != 2 or any(x >= 1500 for x in used):
    raise SystemExit(f'two idle A100 GPUs required: {used}')
print('GPU/port isolation PASS', used, flush=True)
PY
attempt=$(date -u +%Y%m%dT%H%M%SZ)_$$
SERVER0= SERVER1= WORKER0= WORKER1=
cleanup() {
  for pid in "$WORKER0" "$WORKER1" "$SERVER0" "$SERVER1"; do
    [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
  done
  for pid in "$WORKER0" "$WORKER1" "$SERVER0" "$SERVER1"; do
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM
printf 'STARTING\n' > "$LOG/status"
for gpu in 0 1; do
  port=$((8550 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "recursive-opd-complete-c-pi$CYCLE" \
    --host 127.0.0.1 --port "$port" --context-length 32768 \
    --mem-fraction-static 0.72 --attention-backend triton \
    --sampling-backend pytorch --disable-cuda-graph \
    --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "$LOG/server_gpu${gpu}_${attempt}.log" 2>&1 &
  if [[ "$gpu" == 0 ]]; then SERVER0=$!; else SERVER1=$!; fi
done
"$ENV/bin/python" - "$SERVER0" "$SERVER1" <<'PY'
import os, sys, time, urllib.request
for port, pid in zip((8550, 8551), map(int, sys.argv[1:])):
    for _ in range(420):
        try: os.kill(pid, 0)
        except OSError: raise SystemExit(f'server {port} exited')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as reply:
                if reply.status == 200:
                    print(f'server {port} ready', flush=True)
                    break
        except Exception:
            time.sleep(1)
    else:
        raise SystemExit(f'server {port} readiness timeout')
PY
printf 'COLLECTING\n' > "$LOG/status"
for gpu in 0 1; do
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v3_experiment.rollout \
    worker --cycle "$CYCLE" --shard "$gpu" \
    --endpoint "http://127.0.0.1:$((8550 + gpu))/v1" \
    > "$LOG/worker_gpu${gpu}_${attempt}.log" 2>&1 &
  if [[ "$gpu" == 0 ]]; then WORKER0=$!; else WORKER1=$!; fi
done
set +e
wait "$WORKER0"; RC0=$?
wait "$WORKER1"; RC1=$?
set -e
WORKER0= WORKER1=
if [[ $RC0 -ne 0 || $RC1 -ne 0 ]]; then
  printf 'FAILED_WORKERS:%s:%s\n' "$RC0" "$RC1" > "$LOG/status"
  exit 3
fi
"$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v3_experiment.rollout \
  integrity --cycle "$CYCLE" > "$LOG/rollout_integrity.log"
cleanup
SERVER0= SERVER1=
printf 'ROLLOUT_COMPLETE\n' > "$LOG/status"
