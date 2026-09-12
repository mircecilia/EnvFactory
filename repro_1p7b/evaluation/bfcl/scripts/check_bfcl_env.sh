#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_clean_bfcl
"$BFCL_PYTHON" -c 'import bfcl_eval, importlib.metadata as m, sglang, torch, transformers; print("bfcl_eval_path", bfcl_eval.__path__[0]); print("bfcl_eval_version", m.version("bfcl_eval")); print("sglang", sglang.__version__); print("torch", torch.__version__); print("transformers", transformers.__version__)'
"$BFCL_BIN" test-categories | grep -E 'multi_turn|multi_turn_base|multi_turn_miss_func|multi_turn_miss_param|multi_turn_long_context'
"$BFCL_BIN" models | grep -F 'Qwen/Qwen3-1.7B-FC'
printf 'BFCL checkout: %s\n' "$(git -C "$BFCL_ROOT" rev-parse HEAD)"
printf 'BFCL status: clean\n'
