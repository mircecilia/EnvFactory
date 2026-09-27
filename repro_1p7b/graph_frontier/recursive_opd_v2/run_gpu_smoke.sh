#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/recursive_opd_v2_run
CYCLE=${1:-0}
[[ $CYCLE =~ ^[0-3]$ ]] || { echo 'only V2 cycles 0 through 3 may smoke' >&2; exit 2; }
TAG=
if [[ $CYCLE != 0 ]]; then TAG=cycle${CYCLE}_; fi
cd "$ROOT"
mkdir -p "$RUN/gpu_smoke"
exec 9>"$RUN/gpu_smoke/.${TAG}lock"
flock -n 9 || { echo 'V2 smoke already active' >&2; exit 2; }
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

"$ENV/bin/python" - <<'PY'
import subprocess
used = [int(line.strip()) for line in subprocess.check_output(
    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
    text=True).splitlines()]
if len(used) != 2 or any(value >= 1500 for value in used):
    raise SystemExit(f'two isolated GPUs required: {used}')
print('GPU isolation PASS', used, flush=True)
PY
printf 'RUNNING\n' > "$RUN/gpu_smoke/${TAG}status"
if CUDA_VISIBLE_DEVICES=0,1 "$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
    -m repro_1p7b.graph_frontier.recursive_opd_v2.smoke --cycle "$CYCLE" \
    > "$RUN/gpu_smoke/${TAG}launcher.log" 2>&1; then
  printf 'PASS\n' > "$RUN/gpu_smoke/${TAG}status"
else
  printf 'FAILED\n' > "$RUN/gpu_smoke/${TAG}status"
  exit 1
fi
