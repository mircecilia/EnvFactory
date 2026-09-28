#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN="$ROOT/repro_1p7b/graph_frontier/recursive_opd_v3_experiment_run"
CYCLE=${1:?cycle 0..3 required}
[[ "$CYCLE" =~ ^[0-3]$ ]] || exit 2
cd "$ROOT"
mkdir -p "$RUN/cycle$CYCLE"
exec 9>"$RUN/cycle$CYCLE/.smoke.lock"
flock -n 9 || { echo 'Complete-C smoke already active' >&2; exit 2; }
export CUDA_HOME="$ENV" PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
"$ENV/bin/python" - <<'PY'
import subprocess
used = [int(x.strip()) for x in subprocess.check_output(
    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()]
if len(used) != 2 or any(x >= 1500 for x in used):
    raise SystemExit(f'two idle A100 GPUs required: {used}')
print('GPU isolation PASS', used, flush=True)
PY
printf 'RUNNING\n' > "$RUN/cycle$CYCLE/smoke_status"
if CUDA_VISIBLE_DEVICES=0,1 "$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
    -m repro_1p7b.graph_frontier.recursive_opd_v3_experiment.smoke --cycle "$CYCLE" \
    > "$RUN/cycle$CYCLE/smoke_launcher.log" 2>&1; then
  printf 'PASS\n' > "$RUN/cycle$CYCLE/smoke_status"
else
  printf 'FAILED\n' > "$RUN/cycle$CYCLE/smoke_status"
  exit 1
fi
