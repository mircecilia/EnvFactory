# EnvFactory 1.7B Reproduction

## Goal

Establish a clean, reproducible EnvFactory-1.7B baseline before changing any training-data generation or selection strategy.

## Current status

- Remote: `n1`, user `u2024311031`
- Hardware: 2 x NVIDIA A100-SXM4-40GB
- Source baseline: `298e4744d329b2f88dc16347cc7734348d40464f`
- Branch: `repro/envfactory-1p7b`
- Core environment: `/home/u2024311031/.conda/envs/envfactory_repro_1p7b`
- GPU environment: `/home/u2024311031/.conda/envs/envfactory_sglang_1p7b`
- Python: 3.12.14
- Base model: Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Environment/Tool smoke: PASS
- In-memory ToolGraph/TopologySampler smoke: PASS
- Data processing smoke: PASS
- SGLang/OpenAI-compatible project-client smoke: PASS
- QueryGen smoke: BLOCKED at model-generated scenario validation; evidence retained
- One-step 1.7B full-SFT smoke: PASS
- Formal training/evaluation: NOT STARTED

## Official versus ours

| Item | Official | Ours | Reason | Expected impact |
|---|---|---|---|---|
| Base model | Qwen/Qwen3-1.7B | Same, fixed Hub revision | Reproducibility | None |
| Released checkpoint | SFT + RL | SFT baseline planned first | Research protocol prioritizes SFT | Not directly equivalent to released checkpoint |
| SFT data | EnvFactory-SFT-FILTERED | Two locally generated smoke samples only | Do not start formal data run yet | Smoke is not a quality result |
| SFT method | Full fine-tuning, LlamaFactory, ZeRO-3 | Full fine-tuning, one GPU, one optimizer step | Minimal feasibility test | No distributed/quality conclusion |
| GPUs | Not specified | 2 x A100 40GB | Available hardware | May require batch/accumulation adaptation |
| Sequence length | Repo config uses 16384 for Qwen3-8B | 1024 for smoke | Bound compute and memory | Not representative of formal training |

## Environment setup

Conda was created through the Tsinghua mirror with Python 3.12. Python packages use the Tsinghua PyPI mirror. The unpinned upstream requirements resolve to incompatible latest releases, so the verified core lock is in `configs/requirements-compat.txt`.

SGLang 0.5.9 pins OpenAI 2.6.1, which conflicts with the core Agent/LiteLLM environment. Serving/training therefore uses a separate GPU environment; its key package lock is `configs/requirements-gpu.txt`. CUDA 12.8 nvcc/GCC live only in that environment. The NVIDIA CUDA channel has no Tsinghua mirror, so only those CUDA packages came from NVIDIA's official channel. LlamaFactory is an isolated checkout at commit `100e9a42c6c09f8f7849b70d60f3da445fb2024b`.

## Reproduce completed smoke tests

```bash
cd /home/u2024311031/workspace/envfactory_repro_1p7b
export PYTHONPATH=.
PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
$PY repro_1p7b/scripts/environment_tool_smoke.py
$PY repro_1p7b/scripts/toolgraph_smoke.py
$PY repro_1p7b/scripts/data_process_smoke.py
```

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/u2024311031/.conda/envs/envfactory_sglang_1p7b
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 python repro_1p7b/scripts/model_smoke.py
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train repro_1p7b/configs/llamafactory_sft_smoke.yaml
```

See `RUNBOOK.md` for observed outputs and `notes/` for the repository and official-recipe audits.
