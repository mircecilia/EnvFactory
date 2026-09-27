#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/recursive_opd_v2_run
CYCLE=${1:?fresh V2 policy cycle 1 or 2 required}
[[ $CYCLE = 1 || $CYCLE = 2 ]] || { echo 'only V2 pi1 or pi2 may collect' >&2; exit 2; }
MODEL=$RUN/cycle$CYCLE/checkpoint
LOG=$RUN/cycle$CYCLE
cd "$ROOT"
mkdir -p "$LOG"
exec 9>"$LOG/.rollout.lock"
flock -n 9 || { echo 'V2 rollout launcher already active' >&2; exit 2; }
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

"$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v2.future \
  lock --cycle "$CYCLE" --model "$MODEL" > "$LOG/rollout_lock.log"
"$ENV/bin/python" - <<'PY'
import socket, subprocess
for port in (8450, 8451):
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise SystemExit(f'port {port} occupied')
used = [int(line.strip()) for line in subprocess.check_output(
    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
    text=True).splitlines()]
if len(used) != 2 or any(value >= 1500 for value in used):
    raise SystemExit(f'two isolated GPUs required: {used}')
print('two GPUs and ports isolated', used, flush=True)
PY

attempt=$(date -u +%Y%m%dT%H%M%SZ)_$$
SERVER0= SERVER1= WORKER0= WORKER1=
cleanup() {
  for pid in "$WORKER0" "$WORKER1" "$SERVER0" "$SERVER1"; do
    [[ -z $pid ]] || kill "$pid" 2>/dev/null || true
  done
  for pid in "$WORKER0" "$WORKER1" "$SERVER0" "$SERVER1"; do
    [[ -z $pid ]] || wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM
printf 'STARTING\n' > "$LOG/status"
for gpu in 0 1; do
  port=$((8450 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "recursive-opd-v2-pi$CYCLE" \
    --host 127.0.0.1 --port "$port" --context-length 32768 \
    --mem-fraction-static 0.72 --attention-backend triton \
    --sampling-backend pytorch --disable-cuda-graph \
    --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "$LOG/server_gpu${gpu}_${attempt}.log" 2>&1 &
  if [[ $gpu = 0 ]]; then SERVER0=$!; else SERVER1=$!; fi
done
"$ENV/bin/python" - "$SERVER0" "$SERVER1" <<'PY'
import os, sys, time, urllib.request
for port, pid in zip((8450, 8451), map(int, sys.argv[1:])):
    for _ in range(420):
        try: os.kill(pid, 0)
        except OSError: raise SystemExit(f'V2 server {port} exited')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as reply:
                if reply.status == 200:
                    print(f'V2 server {port} ready', flush=True)
                    break
        except Exception:
            time.sleep(1)
    else:
        raise SystemExit(f'V2 server {port} readiness timeout')
PY
printf 'COLLECTING\n' > "$LOG/status"
for gpu in 0 1; do
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v2.future \
    worker --cycle "$CYCLE" --model "$MODEL" \
    --endpoint "http://127.0.0.1:$((8450 + gpu))/v1" --shard "$gpu" \
    > "$LOG/worker_gpu${gpu}_${attempt}.log" 2>&1 &
  if [[ $gpu = 0 ]]; then WORKER0=$!; else WORKER1=$!; fi
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
cleanup
SERVER0= SERVER1=
"$ENV/bin/python" - "$CYCLE" <<'PY'
import json, sys
from pathlib import Path
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import check
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import RUN
cycle = int(sys.argv[1])
base = RUN / f'cycle{cycle}'
manifest_path = base / 'rollout_manifest.json'
manifest = json.loads(manifest_path.read_text())
protocol = check()
task_ids = protocol['train_task_ids'] + protocol['heldout_task_ids']
errors = []
for task in task_ids:
    episode = json.loads((base / 'rollouts' / f'{task}.json').read_text())
    if (episode['task_id'] != task or episode['cycle'] != cycle
            or episode['model_sha256'] != manifest['model_sha256']
            or episode['manifest_sha256'] != sha256(manifest_path)):
        raise SystemExit(f'fresh V2 rollout identity drift: {task}')
    if episode.get('student_error'):
        errors.append(task)
result = {'status': 'ROLLOUT_COMPLETE', 'cycle': cycle, 'task_count': len(task_ids),
          'student_error_count': len(errors), 'student_error_task_ids': errors,
          'rollout_manifest_sha256': sha256(manifest_path),
          'model_sha256': manifest['model_sha256'],
          'gpu_experiment_base_sha': manifest['gpu_experiment_base_sha']}
output = base / 'rollout_integrity.json'
if output.exists():
    raise SystemExit('refusing to overwrite V2 rollout integrity report')
output.write_text(stable(result) + '\n')
print(stable(result), flush=True)
PY
printf 'ROLLOUT_COMPLETE\n' > "$LOG/status"
