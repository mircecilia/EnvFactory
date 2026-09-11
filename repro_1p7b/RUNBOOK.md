# Runbook

## 2026-09-11 - Phase 0/1 bootstrap and audit

### Remote identity and hardware

`hostname`, `whoami`, `pwd`, and `nvidia-smi` verified host `n1`, user `u2024311031`, home `/home/u2024311031`, and two NVIDIA A100-SXM4-40GB GPUs (40960 MiB each, initially 0 MiB used). The home filesystem had 174T available.

Result: PASS

### Source repository audit

The read-only source is `/home/u2024311031/EnvFactory`, branch `main`, commit `298e4744d329b2f88dc16347cc7734348d40464f`, clean status, remote `git@github.com:LARK-AI-Lab/EnvFactory.git`. It was not modified.

Result: PASS

### Isolated checkout

Created `/home/u2024311031/workspace/envfactory_repro_1p7b` using `git clone --local`, restored the verified upstream origin, and created `repro/envfactory-1p7b`.

Result: PASS

### Isolated Python environment

Created `/home/u2024311031/.conda/envs/envfactory_repro_1p7b` with Python 3.12 via the Tsinghua conda mirror. The upstream unpinned requirements exposed an incompatibility: `openai-agents 0.22.2` requires `openai>=3`, while `litellm 1.100.1` requires `openai<3`. Two resolver processes started by this task were terminated. Dry-run selected `openai-agents 0.20.0`. Verified: `fastmcp 3.1.0`, `openai-agents 0.20.0`, `litellm 1.100.1`, `openai 2.54.0`; `pip check` reported no broken requirements.

Result: PASS with compatibility lock

### Environment and stateful tool smoke

Command: `python repro_1p7b/scripts/environment_tool_smoke.py`

The official CampusCard MCP server exposed six user-facing tools. Loading a scenario succeeded. A real `recharge` call of 25 CNY changed balance 100.0 -> 125.0 and appended transaction `RECHARGE_20260911223000_u1`.

Result: PASS

### ToolGraph/topology smoke

Command: `python repro_1p7b/scripts/toolgraph_smoke.py`

The in-memory official graph/sampler types produced 8 nodes and 8 typed edges. `TopologySampler.sample_prior()` selected `CampusCard-query_balance` as a dependency of `CampusCard-recharge`; seed 42 sampled query_balance -> recharge and validation returned true.

Full `build_tool_graph()` was not faked: it requires embedding and chat endpoints, and `LLMClient` is instantiated at import time.

Result: PASS for graph/sampler core; full build pending backend

### Data processing smoke

Command: `python repro_1p7b/scripts/data_process_smoke.py`

One four-step trajectory produced two SFT samples (one per input/output pair) with fields instruction/input/output/system/history, tool tags, accumulated history, and CampusCard schema in the system prompt. Output: `repro_1p7b/results/smoke/sft_sample.json`.

Result: PASS

### QueryGen readiness

Credential-presence flags were false for CHAT, EMBEDDING, DEEPSEEK, and SGLANG; the source clone has no `.env`; localhost ports 8000 and 30000 were closed. No credential value was read or printed.

Result: BLOCKED on an actual generation backend. No fake QueryGen result was created.

### Training

No training, evaluation, large generation, model download, or dataset download was started.


### Script entry-point correction

The first direct invocation of each smoke script failed before executing project logic because Python placed only `repro_1p7b/scripts` on `sys.path`. Each script now derives the checkout root from `__file__` and prepends it to `sys.path`. `py_compile` and all three direct invocations then passed.

Result: PASS after narrow runner fix
