#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <base|sft> <model-path> <smoke|full>" >&2
  exit 2
fi
RUN_LABEL="$1"
MODEL_PATH="$2"
SCOPE="$3"
case "$RUN_LABEL" in base|sft) ;; *) echo "invalid run label: $RUN_LABEL" >&2; exit 2 ;; esac
case "$SCOPE" in smoke|full) ;; *) echo "invalid scope: $SCOPE" >&2; exit 2 ;; esac

require_clean_bfcl
require_model "$MODEL_PATH"

RUN_ROOT="$EVAL_DIR/artifacts/$SCOPE/$RUN_LABEL"
LOG_DIR="$EVAL_DIR/logs/$SCOPE"
mkdir -p "$RUN_ROOT" "$LOG_DIR"
export BFCL_PROJECT_ROOT="$RUN_ROOT"
export VLLM_ENDPOINT=127.0.0.1
export VLLM_PORT="$BFCL_PORT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if ! "$BFCL_PYTHON" -c 'import os,socket; s=socket.socket(); s.settimeout(0.3); raise SystemExit(0 if s.connect_ex(("127.0.0.1",int(os.environ["VLLM_PORT"]))) != 0 else 1)'; then
  echo "port $BFCL_PORT is already in use; refusing to attach to an unknown server" >&2
  exit 1
fi

SERVER_LOG="$LOG_DIR/${RUN_LABEL}_server.log"
SERVER_CMD=(
  "$BFCL_PYTHON" -m sglang.launch_server
  --host 127.0.0.1
  --model-path "$MODEL_PATH"
  --port "$BFCL_PORT"
  --dtype bfloat16
  --tp 1
  --mem-fraction-static "$BFCL_GPU_MEMORY_UTILIZATION"
  --trust-remote-code
  --disable-cuda-graph
  --attention-backend triton
  --sampling-backend pytorch
  --context-length 32768
  --max-total-tokens 65536
)
"${SERVER_CMD[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$BFCL_PYTHON" -c 'import os,time,requests; u=f"http://127.0.0.1:{os.environ[\"VLLM_PORT\"]}/v1/models"; deadline=time.time()+300; last=None
while time.time()<deadline:
    try:
        r=requests.get(u,timeout=2); last=f"{r.status_code} {r.text[:200]}"
        if r.status_code==200: print("server ready"); raise SystemExit(0)
    except Exception as e: last=repr(e)
    time.sleep(2)
raise SystemExit(f"server did not become ready: {last}")'

GENERATE_CMD=(
  "$BFCL_BIN" generate
  --model "$BFCL_MODEL_ID"
  --backend "$BFCL_BACKEND"
  --skip-server-setup
  --num-gpus 1
  --gpu-memory-utilization "$BFCL_GPU_MEMORY_UTILIZATION"
  --temperature "$BFCL_TEMPERATURE"
  --include-input-log
  --allow-overwrite
  --local-model-path "$MODEL_PATH"
)

if [ "$SCOPE" = smoke ]; then
  cp "$EVAL_DIR/configs/smoke_ids.json" "$RUN_ROOT/test_case_ids_to_generate.json"
  "$BFCL_PYTHON" "$EVAL_DIR/scripts/server_smoke.py" --model-path "$MODEL_PATH" --output "$RUN_ROOT/server_smoke.json"
  "${GENERATE_CMD[@]}" --run-ids
else
  "${GENERATE_CMD[@]}" --test-category multi_turn
fi

"$BFCL_BIN" evaluate --model "$BFCL_MODEL_ID" --test-category multi_turn
printf 'BFCL %s %s complete. Results: %s/result Scores: %s/score\n' "$SCOPE" "$RUN_LABEL" "$RUN_ROOT" "$RUN_ROOT"
