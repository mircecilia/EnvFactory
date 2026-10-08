#!/usr/bin/env bash
# Observational diagnostics only: resume the frozen v3 protocol, never relax it.
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run/gpu_sglang_v3
MODEL=$ROOT/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
cd "$ROOT"
exec 9>"$RUN/.lock"; flock -n 9 || exit 2
export PATH=$ENV/bin:$PATH PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
MOD=repro_1p7b.graph_frontier.verified_recovery_resume_diagnostics
ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)_$$
OWNED=(); WORKERS=()
cleanup(){
  for pid in "${OWNED[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  sleep 3
  for pid in "${OWNED[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
trap 'printf "GPU_RESUME_FAILED:%s\n" "$LINENO" > "$RUN/collection_status"' ERR
"$ENV/bin/python" -m unittest repro_1p7b.graph_frontier.tests.test_verified_recovery_feasibility repro_1p7b.graph_frontier.tests.test_verified_recovery_metrics repro_1p7b.graph_frontier.tests.test_verified_recovery_gpu repro_1p7b.graph_frontier.tests.test_verified_recovery_source_order repro_1p7b.graph_frontier.tests.test_verified_recovery_server_order repro_1p7b.graph_frontier.tests.test_verified_recovery_resume_diagnostics > "$RUN/resume_tests_${ATTEMPT}.log" 2>&1
"$ENV/bin/python" -m "$MOD" check --attempt "$ATTEMPT" > "$RUN/resume_check_${ATTEMPT}.log"
"$ENV/bin/python" - "$ATTEMPT" <<'PY'
import json,socket,subprocess,sys
from pathlib import Path
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import atomic_json
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256
r=Path('repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run/gpu_sglang_v3')
for port in (8260,8261):
    with socket.socket() as s:s.bind(('127.0.0.1',port))
for line in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines():
    assert int(line.strip())<100,'GPU occupied; no preemption'
failed=r/'paired_episodes/official-rl-0056-6fb6805d18ff.json'
value=json.loads(failed.read_text())
assert value['status']=='FAILED' and value['exception']=='AssertionError:paired response drift'
assert value['collection_protocol_sha256']==sha256(r/'collection_protocol.json')
complete={p.name:sha256(p) for p in (r/'paired_episodes').glob('*.json') if json.loads(p.read_text())['status']=='COMPLETE'}
assert len(complete)==367,'unexpected collection state: inspect before attempting'
attempt=r/'resume_diagnostics'/sys.argv[1]
archive=r/'failed_attempts'/sys.argv[1]/failed.name
assert not archive.exists()
atomic_json(attempt/'resume_manifest.json',{'completed_before':complete,'old_failure_sha256':sha256(failed),
    'archived_failure':str(archive),'retried_task_id':value['task_id'],'exact_protocol_sha256':value['collection_protocol_sha256'],
    'retry_count':1,'no_exclusion':True,'no_verifier_change':True})
archive.parent.mkdir(parents=True,exist_ok=True)
failed.rename(archive)  # Original bytes retained, not deleted or overwritten.
PY
printf 'GPU_RESUME_SERVERS_STARTING\n' > "$RUN/collection_status"
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu setsid "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name dynamic-v1 --dtype bfloat16 \
    --host 127.0.0.1 --port "$((8260+gpu))" --context-length 32768 \
    --mem-fraction-static 0.60 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 --api-key placeholder \
    > "$RUN/resume_server_gpu${gpu}_${ATTEMPT}.log" 2>&1 &
  OWNED+=("$!")
done
"$ENV/bin/python" - "${OWNED[@]}" <<'PY'
import os,sys,time,urllib.request
for port,pid in zip((8260,8261),map(int,sys.argv[1:])):
    for _ in range(420):
        os.kill(pid,0)
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
                if r.status==200:print(f'{port} READY',flush=True);break
        except Exception:time.sleep(1)
    else:raise SystemExit(f'{port} readiness timeout')
PY
printf 'GPU_OPENLIBRARY_STRICT_RETRY_ONCE\n' > "$RUN/collection_status"
# Original fixed order: failed task is global index 345, shard-1 position 172.
# The original worker skips all already COMPLETE artifacts under the same lock.
CUDA_VISIBLE_DEVICES='' VERIFIED_RECOVERY_ENDPOINT=http://127.0.0.1:8261/v1 \
  "$ENV/bin/python" -m "$MOD" worker --shard 1 --limit 173 --attempt "$ATTEMPT" \
  > "$RUN/resume_openlibrary_${ATTEMPT}.log" 2>&1
printf 'GPU_RESUMING_REMAINING31\n' > "$RUN/collection_status"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES='' VERIFIED_RECOVERY_ENDPOINT="http://127.0.0.1:$((8260+shard))/v1" \
    setsid "$ENV/bin/python" -m "$MOD" worker --shard "$shard" --attempt "$ATTEMPT" \
    > "$RUN/resume_worker_s${shard}_${ATTEMPT}.log" 2>&1 &
  WORKERS+=("$!"); OWNED+=("$!")
done
while ((${#WORKERS[@]})); do
  remaining=()
  for pid in "${WORKERS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then remaining+=("$pid"); else wait "$pid"; fi
  done
  WORKERS=("${remaining[@]}")
  if ((${#WORKERS[@]})); then sleep 2; fi
done
"$ENV/bin/python" - "$ATTEMPT" <<'PY'
import json,sys
from pathlib import Path
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256
r=Path('repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run/gpu_sglang_v3')
m=json.loads((r/'resume_diagnostics'/sys.argv[1]/'resume_manifest.json').read_text())
assert sha256(r/'collection_protocol.json')==m['exact_protocol_sha256']
for name,digest in m['completed_before'].items():
    assert sha256(r/'paired_episodes'/name)==digest,'pre-existing completed artifact changed'
assert sha256(Path(m['archived_failure']))==m['old_failure_sha256']
print('367 original artifacts + protocol + failure archive unchanged')
PY
"$ENV/bin/python" -m repro_1p7b.graph_frontier.verified_recovery_gpu_v3 metrics > "$RUN/resume_metrics_${ATTEMPT}.log"
printf 'PAIRED_COLLECTION_COMPLETE_MANUAL_CASE_REVIEW_PENDING\n' > "$RUN/collection_status"
