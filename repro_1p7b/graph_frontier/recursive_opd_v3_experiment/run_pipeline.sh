#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN="$ROOT/repro_1p7b/graph_frontier/recursive_opd_v3_experiment_run"
CODE="$ROOT/repro_1p7b/graph_frontier/recursive_opd_v3_experiment"
cd "$ROOT"
mkdir -p "$RUN"
exec 8>"$RUN/.pipeline.lock"
flock -n 8 || { echo 'Complete-C pipeline already active' >&2; exit 2; }
export PYTHONPATH="$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
status() { printf '%s\n' "$1" > "$RUN/pipeline_status"; printf '%s\n' "$1"; }
fail() { status "FAILED:$1"; exit 1; }
verify_checkpoint() {
  "$ENV/bin/python" - "$1" <<'PY'
import json, sys
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.schedule import RUN
cycle=int(sys.argv[1]); p=RUN/f'cycle{cycle}/checkpoint'; s=json.loads((p/'trainer_state.json').read_text())
if (s.get('lineage') != 'recursive_opd_v3_complete_c_anchor' or s.get('status') != 'TRAINED'
        or s.get('cycle') != cycle-1 or s.get('model_sha256') != sha256(p/'model.safetensors')
        or s.get('train_plan_sha256') != sha256(RUN/f'cycle{cycle-1}/train_plan.json')):
    raise SystemExit(f'Cpi{cycle} checkpoint/hash gate failed')
print(f'Cpi{cycle} checkpoint PASS', flush=True)
PY
}
verify_audit() {
  "$ENV/bin/python" - "$1" <<'PY'
import json, sys
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.schedule import RUN
cycle=int(sys.argv[1]); base=RUN/f'cycle{cycle}'; a=json.loads((base/'metrics/audit.json').read_text())
if (a.get('status') != 'AUDITED' or a.get('cycle') != cycle or a.get('task_count') != 2158
        or a.get('metrics_sha256') != sha256(base/'metrics/per_task.jsonl')
        or a.get('natural_dataset_sha256') != sha256(base/'natural/dataset.jsonl')
        or a.get('anchor_dataset_sha256') != sha256(base/'anchors/anchors.jsonl')
        or any(v != 'PASS' for v in a.get('gates', {}).values())):
    raise SystemExit(f'Cpi{cycle} fresh data audit failed')
print(f'Cpi{cycle} fresh data audit PASS', flush=True)
PY
}
status 'CYCLE0_PREFLIGHT'
"$ENV/bin/python" - <<'PY' || fail 'D0_PREFLIGHT'
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.schedule import build_plan, freeze_plan
rows, plan=build_plan(0)
if len(rows) != 128 or plan['optimizer_steps'] != 64 or plan['fresh_anchors_exposed'] != 0:
    raise SystemExit('D0 schedule drift')
freeze_plan(0)
print('Complete-C D0 128/64 preflight PASS', flush=True)
PY
if [[ -e "$RUN/cycle1/checkpoint" ]]; then
  verify_checkpoint 1 || fail 'CPI1_CHECKPOINT'
else
  if [[ ! -e "$RUN/cycle0/gpu_smoke.json" ]]; then
    status 'CPI1_GPU_SMOKE'
    bash "$CODE/run_smoke.sh" 0 || fail 'CPI1_GPU_SMOKE'
  fi
  status 'CPI1_TRAIN'
  bash "$CODE/run_train.sh" 0 || fail 'CPI1_TRAIN'
  verify_checkpoint 1 || fail 'CPI1_CHECKPOINT'
fi
for cycle in 1 2 3 4; do
  if [[ -e "$RUN/cycle$cycle/rollout_integrity.json" ]]; then
    "$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v3_experiment.rollout \
      integrity --cycle "$cycle" || fail "CPI${cycle}_ROLLOUT_INTEGRITY"
  else
    status "CPI${cycle}_ROLLOUT"
    bash "$CODE/run_rollout.sh" "$cycle" || fail "CPI${cycle}_ROLLOUT"
  fi
  if [[ -e "$RUN/cycle$cycle/metrics/audit.json" ]]; then
    verify_audit "$cycle" || fail "CPI${cycle}_AUDIT"
  else
    status "CPI${cycle}_DATA_AUDIT"
    "$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v3_experiment.mine \
      --cycle "$cycle" --workers 4 \
      > "$RUN/cycle$cycle/mine_launcher.log" 2>&1 || fail "CPI${cycle}_DATA_AUDIT"
    verify_audit "$cycle" || fail "CPI${cycle}_AUDIT"
  fi
  if [[ "$cycle" -lt 4 ]]; then
    next=$((cycle + 1))
    if [[ -e "$RUN/cycle$next/checkpoint" ]]; then
      verify_checkpoint "$next" || fail "CPI${next}_CHECKPOINT"
    else
      if [[ ! -e "$RUN/cycle$cycle/gpu_smoke.json" ]]; then
        status "CPI${next}_GPU_SMOKE"
        bash "$CODE/run_smoke.sh" "$cycle" || fail "CPI${next}_GPU_SMOKE"
      fi
      status "CPI${next}_TRAIN"
      bash "$CODE/run_train.sh" "$cycle" || fail "CPI${next}_TRAIN"
      verify_checkpoint "$next" || fail "CPI${next}_CHECKPOINT"
    fi
  fi
done
for cycle in 3 4; do
  if [[ -f "$RUN/evaluation/pi$cycle/status" ]] && \
      [[ "$(< "$RUN/evaluation/pi$cycle/status")" == 'AUDITED' ]]; then
    echo "Cpi$cycle external evaluation already audited"
  else
    status "CPI${cycle}_EXTERNAL"
    bash "$CODE/run_external.sh" "$cycle" || fail "CPI${cycle}_EXTERNAL"
  fi
done
status 'COMPARISON_AUDIT'
"$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v3_experiment.compare \
  > "$RUN/comparison.log" 2>&1 || fail 'COMPARISON_AUDIT'
status 'COMPLETE'
