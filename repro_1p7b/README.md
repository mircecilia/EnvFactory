# EnvFactory 1.7B Reproduction

## Goal

Establish a clean, reproducible EnvFactory-1.7B baseline before changing any training-data generation or selection strategy.

## Current status

- Remote: `n1`, user `u2024311031`
- Hardware: 2 x NVIDIA A100-SXM4-40GB
- Source baseline: `298e4744d329b2f88dc16347cc7734348d40464f`
- Branch: `repro/envfactory-1p7b`
- Isolated environment: `/home/u2024311031/.conda/envs/envfactory_repro_1p7b`
- Python: 3.12.14
- Environment/Tool smoke: PASS
- In-memory ToolGraph/TopologySampler smoke: PASS
- Data processing smoke: PASS
- QueryGen smoke: BLOCKED pending a configured embedding/chat backend or local OpenAI-compatible server
- SFT smoke/formal training: NOT STARTED

## Official versus ours

| Item | Official | Ours | Reason | Expected impact |
|---|---|---|---|---|
| Base model | Qwen/Qwen3-1.7B | Pending local/cache acquisition | No matching cache found yet | None if identical |
| Released checkpoint | SFT + RL | SFT baseline planned first | Research protocol prioritizes SFT | Not directly equivalent to released checkpoint |
| SFT data | EnvFactory-SFT-FILTERED | Same planned | Hold data fixed | None |
| SFT method | Full fine-tuning, LlamaFactory, ZeRO-3 | Pending smoke | Preserve official objective | None if feasible |
| GPUs | Not specified | 2 x A100 40GB | Available hardware | May require batch/accumulation adaptation |
| Sequence length | Repo config uses 16384 for Qwen3-8B | Unknown until 1.7B recipe is verified | Repo has no 1.7B-specific config | Must be disclosed if adapted |

## Environment setup

Conda was created through the Tsinghua mirror with Python 3.12. Basic requirements were installed through the Tsinghua PyPI mirror. The unpinned upstream requirements resolve to incompatible latest releases, so the verified compatibility lock is in `configs/requirements-compat.txt`.

No SGLang, vLLM, flash-attn, LlamaFactory, model, or dataset has been installed/downloaded yet.

## Reproduce completed smoke tests

```bash
cd /home/u2024311031/workspace/envfactory_repro_1p7b
export PYTHONPATH=.
PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
$PY repro_1p7b/scripts/environment_tool_smoke.py
$PY repro_1p7b/scripts/toolgraph_smoke.py
$PY repro_1p7b/scripts/data_process_smoke.py
```

See `RUNBOOK.md` for observed outputs and `notes/` for the repository and official-recipe audits.
