#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/recursive_opd_v2_run
CYCLE=${1:?fresh V2 policy cycle 1 or 2 required}
[[ $CYCLE = 1 || $CYCLE = 2 ]] || { echo 'only D1 or D2 may be mined' >&2; exit 2; }
cd "$ROOT"
LOG=$RUN/cycle$CYCLE
exec 9>"$LOG/.mine.lock"
flock -n 9 || { echo 'V2 CPU miner already active' >&2; exit 2; }
export PATH=$ENV/bin:$PATH
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
[[ $(<"$LOG/status") == ROLLOUT_COMPLETE ]] || { echo 'full rollout gate failed' >&2; exit 2; }
[[ -f $LOG/rollout_integrity.json ]] || { echo 'rollout integrity report missing' >&2; exit 2; }
[[ ! -e $LOG/dataset && ! -e $LOG/metrics ]] || {
  echo 'refusing to overwrite existing V2 Dk/metrics' >&2; exit 2;
}
INTEGRITY=$LOG/rollout_integrity.json
if [[ -f $LOG/rollout_integrity_repaired.json ]]; then
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v2.context_repair \
    verify --cycle "$CYCLE"
  INTEGRITY=$LOG/rollout_integrity_repaired.json
fi
"$ENV/bin/python" - "$INTEGRITY" "$CYCLE" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
if (value.get('status') != 'ROLLOUT_COMPLETE' or value.get('task_count') != 2158
        or value.get('cycle') != int(sys.argv[2])
        or value.get('student_error_count') != 0):
    raise SystemExit('incomplete V2 rollout cannot become official Dk')
print('full rollout integrity PASS', value['task_count'], flush=True)
PY
printf 'MINING\n' > "$LOG/status"
if "$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v2.future \
    mine-build --cycle "$CYCLE" > "$LOG/mine_build.log" 2>&1; then
  if "$ENV/bin/python" - "$LOG/dataset/manifest.json" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
if (manifest.get('status') != 'READY_FOR_GPU_TRAINING'
        or manifest.get('semantic_eligible_count') !=
           manifest.get('trainable_count', -1) + manifest.get('context_overflow_count', -1)
        or any(value != 'PASS' for value in manifest.get('gates', {}).values())):
    raise SystemExit('V2 Dk data gate failed; no further training')
print('Dk READY', manifest['semantic_eligible_count'], manifest['trainable_count'],
      manifest['context_overflow_count'], flush=True)
PY
  then
    :
  else
    printf 'FAILED_DATA_GATE\n' > "$LOG/status"
    exit 1
  fi
  printf 'AUDITING\n' > "$LOG/status"
  if "$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v2.audit_future \
      --cycle "$CYCLE" > "$LOG/independent_audit.log" 2>&1; then
    printf 'AUDITED\n' > "$LOG/status"
  else
    printf 'FAILED_AUDIT\n' > "$LOG/status"
    exit 1
  fi
else
  printf 'FAILED_MINE\n' > "$LOG/status"
  exit 1
fi
