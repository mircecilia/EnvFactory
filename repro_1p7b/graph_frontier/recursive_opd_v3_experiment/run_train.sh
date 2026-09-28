#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN="$ROOT/repro_1p7b/graph_frontier/recursive_opd_v3_experiment_run"
CYCLE=${1:?cycle 0..3 required}
[[ "$CYCLE" =~ ^[0-3]$ ]] || exit 2
cd "$ROOT"
mkdir -p "$RUN/cycle$CYCLE"
exec 9>"$RUN/cycle$CYCLE/.train.lock"
flock -n 9 || { echo 'Complete-C trainer already active' >&2; exit 2; }
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
[[ ! -e "$RUN/cycle$((CYCLE + 1))/checkpoint" ]] || {
  echo 'refusing to overwrite trained checkpoint' >&2; exit 2;
}
"$ENV/bin/python" - "$CYCLE" <<'PY'
import json, subprocess, sys
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.schedule import build_plan, freeze_plan
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.schedule import RUN
cycle = int(sys.argv[1]); rows, plan = build_plan(cycle); freeze_plan(cycle)
if (len(rows) != plan['total_exposures'] or plan['fresh_anchors_available'] != plan['fresh_anchors_exposed']
        or plan['total_exposures'] != plan['optimizer_steps'] * 2):
    raise SystemExit('exposure plan gate failed')
smoke = json.loads((RUN / f'cycle{cycle}/gpu_smoke.json').read_text())
if (smoke.get('status') != 'PASS' or smoke.get('cycle') != cycle
        or smoke.get('optimizer_updates') != 0
        or smoke.get('input_model_sha256') != plan['input_model_sha256']
        or smoke.get('natural_dataset_sha256') != plan['natural_dataset_sha256']
        or smoke.get('anchor_dataset_sha256') != plan['anchor_dataset_sha256']):
    raise SystemExit('optimizer-free longest-selected-exposure smoke missing/drifted')
used = [int(x.strip()) for x in subprocess.check_output(
    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
    text=True).splitlines()]
if len(used) != 2 or any(x >= 1500 for x in used):
    raise SystemExit(f'two idle A100 GPUs required: {used}')
print(json.dumps({'preflight': 'PASS', 'cycle': cycle, 'sampled': plan['sampled_composition'],
                  'anchor': plan['fresh_anchors_exposed'], 'steps': plan['optimizer_steps'],
                  'gpu_memory_mib_before': used}, sort_keys=True), flush=True)
PY
printf 'RUNNING\n' > "$RUN/cycle$CYCLE/train_status"
attempt=$(date -u +%Y%m%dT%H%M%SZ)_$$
export COMPLETE_C_TRAIN_ATTEMPT="$attempt"
if CUDA_VISIBLE_DEVICES=0,1 "$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
    -m repro_1p7b.graph_frontier.recursive_opd_v3_experiment.train --cycle "$CYCLE" \
    > "$RUN/cycle$CYCLE/train_launcher_${attempt}.log" 2>&1; then
  "$ENV/bin/python" - "$CYCLE" <<'PY'
import json, sys
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.schedule import RUN
cycle=int(sys.argv[1]); p=RUN/f'cycle{cycle+1}/checkpoint'; s=json.loads((p/'trainer_state.json').read_text())
if (s['status'] != 'TRAINED' or s['cycle'] != cycle or s['model_sha256'] != sha256(p/'model.safetensors')
        or s['train_plan_sha256'] != sha256(RUN/f'cycle{cycle}/train_plan.json')):
    raise SystemExit('checkpoint validation failed')
PY
  printf 'TRAINED\n' > "$RUN/cycle$CYCLE/train_status"
else
  printf 'FAILED_TRAIN\n' > "$RUN/cycle$CYCLE/train_status"
  exit 1
fi
