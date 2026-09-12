# Runbook

## 2026-09-11 - Phase 0/1 bootstrap and audit

### Remote identity and hardware

`hostname`, `whoami`, `pwd`, and `nvidia-smi` verified host `n1`, user `u2024311031`, home `/home/u2024311031`, and two NVIDIA A100-PCIE-40GB GPUs (40960 MiB each, initially 0 MiB used). The home filesystem had 174T available.

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

## 2026-09-12 - Local model, serving, QueryGen, and SFT smoke

### Split core/GPU environments

Installing official `sglang==0.5.9` exposed a hard `openai==2.6.1` pin, incompatible with the core `litellm==1.100.1` / `openai-agents==0.20.0` stack that needs newer OpenAI 2.x. The core environment was restored to OpenAI 2.54.0 and passes `pip check`. SGLang and LlamaFactory run in `envfactory_sglang_1p7b`, which also passes `pip check`.

Importing `src.gen.query_gen` additionally exposed undeclared runtime dependency `ddgs`; `ddgs==9.16.0` and `primp==2.0.0` were installed only in the core environment. Canonical source was not patched.

Result: PASS with role-isolated environments

### Fixed base model and direct GPU smoke

`Qwen/Qwen3-1.7B` was downloaded through `hf-mirror.com` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Twelve files occupy 4,079,450,110 bytes. The first high-parallel Xet transfer failed after about 524 MB; single-worker standard HTTP with Xet disabled completed.

`model_smoke.py` loaded 1,720,574,976 BF16 parameters on one A100, consumed 22 prompt tokens, generated seven tokens with exact text `EnvFactory model smoke passed.`, and observed 3.785 GiB peak CUDA memory. The confirmation run took 5.242 seconds after model load.

Result: PASS

### SGLang serving and project LLM client

Three constrained start attempts identified the missing compiler boundary:

1. default CUDA graph capture required CUDA/nvcc;
2. disabling CUDA graphs still triggered FlashInfer JIT;
3. Triton attention plus PyTorch sampling reached first request, where SGLang rope JIT still required `CUDA_HOME`.

An environment-only CUDA 12.8.61 nvcc/GCC toolchain (218.9 MB download) was then installed. With `CUDA_HOME=$CONDA_PREFIX`, Triton attention, PyTorch sampling, CUDA graphs disabled, 8192 total KV tokens, 4096 context, and 0.2 static memory fraction, SGLang loaded the model in 3.28 GB and allocated 0.88 GB KV cache. The server became ready on `127.0.0.1:30000`.

The repository's unchanged `src.manager.llm_client_manager.LLMClient.inference()` called the OpenAI-compatible endpoint. With Qwen thinking explicitly disabled, the exact response was `EnvFactory SGLang API passed.`.

Result: PASS

### Minimal real QueryGen attempt

A seed-42 two-tool chain (`CampusCard-query_balance -> CampusCard-recharge`) was passed through the real `QueryGenNonConv` Agent/LiteLLM/SGLang path with `pass_k=1`. The first import found the undeclared `ddgs` dependency described above. After installation, ScenarioPlanner succeeded and SchemaGenerator ran.

SchemaGenerator deterministically emitted a JSON-like `statusTextMap` with unquoted numeric keys. `parse_structured_output()` therefore retained `schema` as a string, and the official CampusCard `load_scenario` Pydantic boundary rejected it as not a dictionary. Three built-in retries produced the same invalid schema. No trajectory decision, query, or tool execution was fabricated. Raw generations and the terminated chains are retained under `logs/querygen/` and `results/querygen/`.

Result: BLOCKED by 1.7B structured-output quality/normalization contract; infrastructure and LLM call path reached

### DataProcessing confirmation

After the environment changes, Environment/Tool, ToolGraph/TopologySampler, and DataProcessing scripts were rerun. All passed, and the same one valid four-step trajectory produced two SFT samples with history lengths 0 and 1.

Result: PASS

### One-step full-SFT smoke

LlamaFactory commit `100e9a42c6c09f8f7849b70d60f3da445fb2024b` loaded the fixed Qwen3-1.7B base and the two generated SFT samples. The smoke configuration uses full fine-tuning, BF16, Qwen3 template, cutoff 1024, batch 1, accumulation 1, learning rate 1e-6, cosine schedule, and exactly one optimizer step on GPU 0. It intentionally omits DeepSpeed for this single-process feasibility check.

Observed metrics:

- trainable/all parameters: 1,720,574,976 / 1,720,574,976 (100%)
- train loss: 5.125259876251221
- gradient norm: 222.51739501953125
- train runtime: 6.7239 seconds
- throughput: 0.149 samples/s and 0.149 steps/s
- total FLOPs: 676,516,823,040
- observed process GPU memory: 27,238 MiB
- final checkpoint: 6.5 GiB, ignored by Git

The host kernel is 4.18.0; Accelerate warns that kernels below 5.5 may hang. This one-step run completed, but formal long training needs monitoring and must not treat this smoke as proof of long-run stability.

Result: PASS for one-step feasibility only

### Cleanup

The SGLang server and trainer processes were stopped/exited, port 30000 is closed, and no task-owned GPU process remains.

The formal FILTERED-dataset audit and two-GPU ZeRO-3 gate are recorded in `BASELINE.md` and `results/baseline/`.


## 2026-09-12 - 16K OOM and 8K steady-state recovery

The 16K accumulation-32 pilot completed optimizer step 1 with loss 1.9161 in
115.40 seconds, then rank 1 failed during the next backward pass. GPU 1 had
34.54 GiB in use and 5.13 GiB free when an additional 8.23 GiB allocation was
requested. Only 134.45 MiB was reserved but unallocated, so allocator
fragmentation was not the primary cause. Exit code was 1; both GPUs returned
to 0 MiB afterward.

The fixed-revision tokenized dataset has mean length 8,601.59, median 8,163,
P90 15,055.8, and 1,633 records at the 16,384 cutoff. The earlier one-step,
accumulation-1 gate did not cover post-update steady-state memory.

A separate resource-adapted 8K config was created. Its two-step gate retained
full SFT, BF16, ZeRO-3, accumulation 32, effective batch 64, and seeds 42. It
completed both optimizer steps in 96.8905 seconds with final loss
1.9541229009628296 and observed 34,633 MiB maximum on each GPU. The checkpoint
saved successfully, exit code was 0, and both GPUs returned to 0 MiB.

Result: 16K FAIL (OOM); resource-adapted 8K steady-state gate PASS

## 2026-09-13 - BFCL V3 evaluation bootstrap

Created an independent Python 3.10 BFCL environment at `/home/u2024311031/.conda/envs/envfactory_bfcl_v1p3` through the Tsinghua mirrors. Installed the official Gorilla v1.3 source package plus SGLang 0.5.9 and an environment-local CUDA 12.8.61 nvcc/GCC toolchain. `pip check`, imports, official category enumeration, Qwen3-1.7B-FC registration, fixed commit, and clean checkout checks passed. The display-only `bfcl version` command has an upstream distribution-name mismatch in this non-editable install, so the check reads `bfcl_eval` metadata and the Git commit instead. Smoke is deferred until the formal SFT run releases the GPUs.

## 2026-09-13 - Formal SFT completion and Base BFCL smoke

Formal resource-adapted 8K SFT completed 414/414 steps with exit code 0, epoch 1.0, train loss 1.0137040206199683, and runtime 21463.8971 seconds. The final root checkpoint contains config, generation config, tokenizer files, chat template, trainer state/results, and a 3,441,185,608-byte `model.safetensors`; both GPUs returned to 0 MiB.

Base BFCL generation completed all eight predeclared IDs. The server, ordinary generation, official QwenFC tool-call parser, and `<tool_response>` refill checks passed. Stock v1.3 evaluate rejected the partial result due its 200-row length assertion; the documented in-memory adapter then invoked official `multi_turn_runner` and produced four real score JSON files. Accuracy was 0/2 in every category (0/8 smoke), with no inference errors or top-level empty outputs. This is not a model-quality conclusion.
