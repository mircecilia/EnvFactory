#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
EVAL_PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
RUN="$ROOT/repro_1p7b/graph_frontier/recursive_opd_v3_experiment_run"
CYCLE=${1:?external policy cycle 3 or 4 required}
[[ "$CYCLE" == 3 || "$CYCLE" == 4 ]] || exit 2
MODEL="$RUN/cycle$CYCLE/checkpoint"
OUT="$RUN/evaluation/pi$CYCLE"
LABEL="recursive_opd_complete_c_pi$CYCLE"
FROZEN="$ROOT/repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_20260914.jsonl"
FROZEN_BASE="$ROOT/repro_1p7b/results/graph_frontier/confirm_300/dynamic_v1"
RICH_PLAN="$ROOT/repro_1p7b/logs/first_divergence_onpolicy_v1/cycle3_heldout_greedy_plan.json"
RICH_BASE="$ROOT/repro_1p7b/logs/first_divergence_onpolicy_v1/cycle3_heldout_greedy_eval/base"
cd "$ROOT"
mkdir -p "$OUT"
exec 9>"$OUT/.launcher.lock"
flock -n 9 || { echo 'Complete-C external evaluation already active' >&2; exit 2; }
export CUDA_HOME="$ENV" PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}"
export ENVFACTORY_ROOT="$ROOT" ENVFACTORY_REWARD_MODE=official
export MCP_CONFIG_PATH="$ROOT/configs/mcp_server.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1

"$ENV/bin/python" - "$CYCLE" "$MODEL" "$FROZEN" "$FROZEN_BASE" "$RICH_PLAN" "$RICH_BASE" <<'PY'
import json, socket, subprocess, sys
from pathlib import Path
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.rollout import integrity, lock
cycle = int(sys.argv[1]); model, frozen, frozen_base, rich_plan, rich_base = map(Path, sys.argv[2:])
manifest = lock(cycle)
audit = json.loads((model.parent / 'metrics/audit.json').read_text())
if (audit.get('status') != 'AUDITED' or audit.get('cycle') != cycle
        or audit.get('model_sha256') != manifest['model_sha256']
        or integrity(cycle)['task_count'] != 2158):
    raise SystemExit('Complete-C full internal audit/checkpoint gate failed')
if sha256(frozen) != '4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c':
    raise SystemExit('Frozen300 plan hash drift')
if json.loads((frozen_base / 'run_summary.json').read_text())['valid'] != 300:
    raise SystemExit('Frozen300 base missing')
plan = json.loads(rich_plan.read_text()); base = json.loads((rich_base / 'collection_manifest.json').read_text())
if (plan['task_count'] != 24 or plan['train_task_overlap'] != 0
        or base['plan_sha256'] != sha256(rich_plan) or base['task_count'] != 24):
    raise SystemExit('Rich24 heldout/base identity drift')
for port in (8570, 8571, 8580):
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise SystemExit(f'evaluation port {port} busy')
used = [int(x.strip()) for x in subprocess.check_output(
    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()]
if len(used) != 2 or any(x >= 1500 for x in used):
    raise SystemExit(f'two idle evaluation GPUs required: {used}')
print('external plans, full audit, checkpoint and GPU isolation PASS', flush=True)
PY
attempt=$(date -u +%Y%m%dT%H%M%SZ)_$$

if [[ ! -f "$OUT/frozen300/comparison/summary.json" ]]; then
  if [[ -e "$OUT/frozen300" ]]; then
    "$EVAL_PY" - "$OUT/frozen300" "$FROZEN" "$MODEL" "$LABEL" <<'PY'
import json, sys
from pathlib import Path
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256
out, plan, model = map(Path, sys.argv[1:4]); label=sys.argv[4]
tasks={json.loads(line)['task_id'] for line in plan.read_text().splitlines() if line}
for path in out.glob('run_config.shard*.json'):
    value=json.loads(path.read_text())
    if (value.get('model') != label or Path(value.get('model_path','')).resolve() != model.resolve()
            or value.get('manifest_sha256') != sha256(plan)
            or value.get('shard_count') != 2 or value.get('selected_count') != 150):
        raise SystemExit(f'partial Frozen300 shard config drift: {path}')
for path in (out/'tasks').glob('*.result.json'):
    value=json.loads(path.read_text()); task=path.name.removesuffix('.result.json')
    if task not in tasks or value.get('task_id') != task or value.get('model') != label:
        raise SystemExit(f'partial Frozen300 result identity drift: {path}')
print('partial Frozen300 files identity PASS', flush=True)
PY
  fi
  mkdir -p "$OUT/frozen300"
  printf 'FROZEN300\n' > "$OUT/status"
  run_shard() (
    set -euo pipefail
    gpu=$1; port=$((8570 + gpu)); key="complete-c-pi$CYCLE-$gpu"
    server=
    cleanup() { [[ -z "$server" ]] || { kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; }; }
    trap cleanup EXIT INT TERM
    CUDA_VISIBLE_DEVICES="$gpu" "$ENV/bin/python" -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port "$port" --api-key "$key" \
      --dtype bfloat16 --tp-size 1 --mem-fraction-static 0.70 \
      --disable-cuda-graph --attention-backend triton --sampling-backend pytorch \
      --context-length 32768 --max-total-tokens 49152 --trust-remote-code \
      > "$OUT/frozen300/server_shard${gpu}_${attempt}.log" 2>&1 &
    server=$!
    "$EVAL_PY" - "$port" "$key" "$server" <<'PY'
import os, requests, sys, time
port, key, pid = sys.argv[1], sys.argv[2], int(sys.argv[3])
for _ in range(300):
    try: os.kill(pid, 0)
    except OSError: raise SystemExit(f'server {port} died')
    try:
        reply = requests.get(f'http://127.0.0.1:{port}/v1/models',
                             headers={'Authorization': 'Bearer ' + key}, timeout=2)
        if reply.status_code == 200:
            break
    except Exception:
        pass
    time.sleep(1)
else:
    raise SystemExit(f'server {port} readiness timeout')
PY
    export SGLANG_BASE_URL="http://127.0.0.1:$port/v1"
    export SGLANG_API_KEY="$key" SGLANG_MODEL="$MODEL"
    "$EVAL_PY" -m repro_1p7b.graph_frontier.eval_first_divergence_frozen300_shard \
      --label "$LABEL" --model-path "$MODEL" --manifest "$FROZEN" \
      --output-dir "$OUT/frozen300" --shard-index "$gpu" \
      > "$OUT/frozen300/eval_shard${gpu}_${attempt}.log" 2>&1
  )
  run_shard 0 & P0=$!
  run_shard 1 & P1=$!
  set +e
  wait "$P0"; RC0=$?
  wait "$P1"; RC1=$?
  set -e
  if [[ $RC0 -ne 0 || $RC1 -ne 0 ]]; then
    printf 'FAILED_FROZEN_SHARD:%s:%s\n' "$RC0" "$RC1" > "$OUT/status"
    exit 4
  fi
  "$EVAL_PY" -m repro_1p7b.graph_frontier.compare_first_divergence_frozen300 \
    --manifest "$FROZEN" --base "$FROZEN_BASE" --trained "$OUT/frozen300" \
    --output "$OUT/frozen300/comparison" > "$OUT/frozen300/comparison_${attempt}.log" 2>&1 || {
      printf 'FAILED_FROZEN_COMPARE\n' > "$OUT/status"; exit 5;
    }
fi

if [[ ! -f "$OUT/rich24/comparison/metrics.json" ]]; then
  if [[ -e "$OUT/rich24" ]]; then
    "$ENV/bin/python" - "$OUT/rich24/trained" "$RICH_PLAN" "$MODEL" "$LABEL" <<'PY'
import json, sys
from pathlib import Path
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256
out, plan, model = map(Path, sys.argv[1:4]); label=sys.argv[4]
source=json.loads(plan.read_text()); tasks=source['task_ids']
if (out/'collection_manifest.json').exists():
    value=json.loads((out/'collection_manifest.json').read_text())
    if (value.get('plan_sha256') != sha256(plan)
            or value.get('model_weights_sha256') != sha256(model/'model.safetensors')
            or value.get('task_count') != 24 or not value.get('all_rollout_files_present')):
        raise SystemExit('completed Rich24 collection manifest drift')
for path in out.glob('*.r0.json'):
    value=json.loads(path.read_text()); task=path.name.removesuffix('.r0.json')
    if (task not in tasks or value.get('task_id') != task or value.get('model') != label
            or value.get('rollout_index') != 0
            or value.get('sampling_seed') != source['sampling_seed'] + tasks.index(task)*1009):
        raise SystemExit(f'partial Rich24 trajectory identity drift: {path}')
print('partial Rich24 files identity PASS', flush=True)
PY
  fi
  [[ ! -e "$OUT/rich24/comparison" ]] || {
    echo 'partial Rich24 comparison needs explicit audit' >&2; exit 6;
  }
  mkdir -p "$OUT/rich24"
  printf 'RICH24\n' > "$OUT/status"
  server=
  rich_cleanup() { [[ -z "$server" ]] || { kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; }; }
  trap rich_cleanup EXIT INT TERM
  CUDA_VISIBLE_DEVICES=0 "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$LABEL" \
    --host 127.0.0.1 --port 8580 --context-length 32768 \
    --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "$OUT/rich24/server_${attempt}.log" 2>&1 &
  server=$!
  "$ENV/bin/python" - "$server" <<'PY'
import os, sys, time, urllib.request
pid = int(sys.argv[1])
for _ in range(420):
    try: os.kill(pid, 0)
    except OSError: raise SystemExit('Rich24 server died')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8580/health', timeout=2) as reply:
            if reply.status == 200:
                break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit('Rich24 server readiness timeout')
PY
  if [[ ! -e "$OUT/rich24/trained/collection_manifest.json" ]]; then
    "$ENV/bin/python" -m repro_1p7b.graph_frontier.evaluate_first_divergence_ce collect \
      --plan "$RICH_PLAN" --model-path "$MODEL" --served-name "$LABEL" \
      --endpoint http://127.0.0.1:8580/v1 --output "$OUT/rich24/trained" \
      > "$OUT/rich24/collect_${attempt}.log" 2>&1
  fi
  rich_cleanup
  server=
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.evaluate_first_divergence_ce summarize \
    --plan "$RICH_PLAN" --base "$RICH_BASE" --trained "$OUT/rich24/trained" \
    --output "$OUT/rich24/comparison" > "$OUT/rich24/summarize_${attempt}.log" 2>&1 || {
      printf 'FAILED_RICH_SUMMARY\n' > "$OUT/status"; exit 7;
    }
fi
printf 'AUDITED\n' > "$OUT/status"
