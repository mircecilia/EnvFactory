# BFCL V3 evaluation

This directory integrates the unmodified official Berkeley Function Calling Leaderboard evaluator with the EnvFactory 1.7B reproduction. It compares the fixed Qwen3-1.7B Base checkpoint with the formal one-epoch, 8K resource-adapted SFT checkpoint. It does not run post-check NLL, a custom benchmark, RL, MCP-Atlas, tau2-Bench, or VitaBench.

## Benchmark lock

- Official source: `ShishirPatil/gorilla`
- Release/tag: `v1.3`
- Commit: `ea13468e4423454d0c213704fb87cf7cb3990433`
- Clean checkout: `/home/u2024311031/benchmarks/gorilla_bfcl_v1p3`
- Python environment: `/home/u2024311031/.conda/envs/envfactory_bfcl_v1p3` (Python 3.10)
- Official CLI category: `multi_turn`
- Expanded V3 categories: `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, and `multi_turn_long_context`

The four expanded categories are the complete BFCL V3 multi-turn target exposed by the v1.3 CLI. Ground truth, cases, function descriptions, handlers, execution, and scoring remain unchanged.

## Model and handler decision

Both checkpoints use official model id `Qwen/Qwen3-1.7B-FC` with the official `QwenFCHandler`; only `--local-model-path` and the isolated output root differ. This is selected because the Base tokenizer, SFT checkpoint, training contract, and official handler all use the same `<tools>`, `<tool_call>`, and `<tool_response>` representation. No registry or evaluator patch is required.

The handler preserves Qwen3 reasoning text and extracts JSON calls from `<tool_call>`. Since the prompt does not insert the no-thinking marker, the model is treated as thinking-enabled. Temperature is fixed at 0.7 to match the paper's thinking-model evaluation rule. BFCL v1.3 does not expose top_p; it is not overridden. The official handler selects up to 4096 output tokens according to remaining context.

Backend is SGLang 0.5.9 on one A100. This matches the paper backend, the official v1.3 supported path, and the version already validated with this exact local Qwen3 checkpoint. The benchmark environment is nevertheless independent.

## Models

- Base: `/home/u2024311031/workspace/envfactory_repro_1p7b/repro_1p7b/models/Qwen3-1.7B`
- Our SFT: `/home/u2024311031/workspace/envfactory_repro_1p7b/repro_1p7b/checkpoints/baseline_sft_8k_1p7b`

The SFT path is valid only after formal training exits successfully and final weights/tokenizer/config files pass integrity checks. Smoke/gate/intermediate checkpoints are not substitutes.

## Paper reference values

The EnvFactory paper reports for Qwen3-1.7B:

| Model | BFCL V3 single-turn | BFCL V3 multi-turn |
|---|---:|---:|
| Base | 79.48 | 16.75 |
| Our (SFT) | 78.30 | 23.25 |
| Official final (SFT+RL, not equivalent to our SFT) | 78.44 | 28.38 |

Our formal baseline is a resource-adapted one-epoch 8K run, while the paper describes three SFT epochs. Reproduced scores therefore test the pipeline and this baseline, not exact parity with the paper recipe.

## Smoke subset

Eight IDs are predeclared in `configs/smoke_ids.json` before any model result. Seed 20260913 sampled two indices uniformly from each 200-case category. The resulting set spans 2-6 turns, 4-6 expected calls, multiple APIs, missing-function cases, missing-parameter cases, and long-context cases. Smoke scores validate plumbing only and are not evidence that one model is better.

## Commands

Environment and registry check:

```bash
cd /home/u2024311031/workspace/envfactory_repro_1p7b
bash repro_1p7b/evaluation/bfcl/scripts/check_bfcl_env.sh
```

Base smoke (short, Codex may run after GPUs are free):

```bash
CUDA_VISIBLE_DEVICES=0 bash repro_1p7b/evaluation/bfcl/scripts/run_bfcl.sh base   /home/u2024311031/workspace/envfactory_repro_1p7b/repro_1p7b/models/Qwen3-1.7B smoke
```

SFT smoke uses the exact same script and settings with label/path changed.

Full runs must be started manually by the user, one model at a time:

```bash
CUDA_VISIBLE_DEVICES=0 bash repro_1p7b/evaluation/bfcl/scripts/run_full_base.sh
CUDA_VISIBLE_DEVICES=0 bash repro_1p7b/evaluation/bfcl/scripts/run_full_sft.sh
```

Raw result, score, input log, state trace, and server logs are under ignored `artifacts/` and `logs/`. Small summaries are committed under `summaries/`.

## Verified environment

The isolated environment passed imports, `pip check`, official category listing, Qwen3-FC registry lookup, and CUDA compiler checks. Key versions are recorded in `configs/environment-lock.txt`. The official `bfcl version` subcommand is not used because the v1.3 source queries distribution metadata named `bfcl` while its built distribution is `bfcl_eval`; this display-only upstream issue does not affect generate/evaluate.
