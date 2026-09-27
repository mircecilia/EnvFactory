#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/recursive_opd_v2_run
BASE=$ROOT/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
CYCLE=${1:?cycle 0..3 required}
[[ $CYCLE =~ ^[0-3]$ ]] || { echo 'only V2 cycles 0 through 3 may train' >&2; exit 2; }
NEXT=$((CYCLE + 1))
MODEL=$BASE
DATA=$RUN/d0_context16k_v2
if [[ $CYCLE != 0 ]]; then MODEL=$RUN/cycle$CYCLE/checkpoint; DATA=$RUN/cycle$CYCLE/dataset; fi
OUTPUT=$RUN/cycle$NEXT/checkpoint
cd "$ROOT"
mkdir -p "$RUN/cycle$CYCLE"
exec 9>"$RUN/cycle$CYCLE/.train.lock"
flock -n 9 || { echo 'V2 trainer already active' >&2; exit 2; }
[[ ! -e $OUTPUT ]] || { echo 'refusing to overwrite V2 checkpoint' >&2; exit 2; }
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

"$ENV/bin/python" - "$CYCLE" "$DATA" "$MODEL" "$OUTPUT" <<'PY'
import json, subprocess, sys
from pathlib import Path
from repro_1p7b.graph_frontier.recursive_opd_v2.train import load_plan
cycle, data, model, output = int(sys.argv[1]), *map(Path, sys.argv[2:])
rows, order, plan = load_plan(cycle=cycle, dataset=data, model=model, output=output,
                              steps=64, learning_rate=2e-6, seed=20260927 + cycle,
                              exposure_budget=128)
if len(order) != 128 or plan['unique_exposed_tasks'] != 128:
    raise SystemExit('natural 128-unique exposure gate failed')
smoke_name = 'longest_ddp.json' if cycle == 0 else f'cycle{cycle}_longest_ddp.json'
smoke = json.loads((output.parents[1] / 'gpu_smoke' / smoke_name).read_text())
if (smoke['status'] != 'PASS' or smoke['dataset_sha256'] != plan['dataset_sha256']
        or smoke['model_sha256'] != plan['initialization_sha256']
        or smoke['gpu_experiment_base_sha'] != plan['gpu_experiment_base_sha']):
    raise SystemExit('longest-sample optimizer-free GPU smoke missing/drifted')
used = [int(line.strip()) for line in subprocess.check_output(
    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
    text=True).splitlines()]
if len(used) != 2 or any(value >= 1500 for value in used):
    raise SystemExit(f'two isolated GPUs required: {used}')
print(json.dumps({'preflight': 'PASS', 'cycle': cycle, 'sampled': plan['sampled_class_counts'],
                  'tool': plan['sampled_tool_exposures'], 'eos': plan['sampled_eos_exposures'],
                  'gpu_memory_mib_before': used}, sort_keys=True), flush=True)
PY

printf 'RUNNING\n' > "$RUN/cycle$CYCLE/train_status"
if CUDA_VISIBLE_DEVICES=0,1 "$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
    -m repro_1p7b.graph_frontier.recursive_opd_v2.train \
    --cycle "$CYCLE" --dataset "$DATA" --model "$MODEL" --output "$OUTPUT" \
    --steps 64 --learning-rate 2e-6 --seed "$((20260927 + CYCLE))" --exposure-budget 128 \
    > "$RUN/cycle$CYCLE/train_launcher.log" 2>&1; then
  if "$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v2.validate_checkpoint \
      --cycle "$CYCLE" > "$RUN/cycle$CYCLE/checkpoint_validation.log" 2>&1; then
    printf 'TRAINED\n' > "$RUN/cycle$CYCLE/train_status"
  else
    printf 'FAILED_VALIDATION\n' > "$RUN/cycle$CYCLE/train_status"
    exit 1
  fi
else
  printf 'FAILED_TRAIN\n' > "$RUN/cycle$CYCLE/train_status"
  exit 1
fi
