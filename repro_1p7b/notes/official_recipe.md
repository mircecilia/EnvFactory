# Official EnvFactory-1.7B recipe audit

Sources: repository README/config at `298e474`, official Hugging Face model card and dataset viewers, and arXiv 2605.18703.

## Verified fields

| Field | Value | Qualification |
|---|---|---|
| Released checkpoint | `LARK-Lab/EnvFactory-1.7B` | Official model card |
| Base model | `Qwen/Qwen3-1.7B` | Model card; model tree shows Qwen3-1.7B-Base -> Qwen3-1.7B -> EnvFactory |
| Released stages | SFT + RL | Explicit model-card statement |
| SFT dataset | `LARK-Lab/EnvFactory-SFT-FILTERED` | Model card/model metadata |
| FILTERED rows | approximately 26.5k | HF viewer/model metadata and repository README |
| ALL rows | 53,412 (53.4k) | HF viewer and repository README |
| RL dataset | `LARK-Lab/EnvFactory-RL`, approximately 3.09k | Model card/README |
| SFT framework | LlamaFactory | README/model card |
| Fine-tuning | full | Model card and repo config |
| Distributed | DeepSpeed ZeRO-3 | Model card and repo config |
| Epochs | 1 | Model card and repo config |
| Learning rate | 1.0e-6 | Model card and repo config |
| Scheduler | cosine | Model card and repo config |
| Per-device batch | 1 | Model card and repo config |
| Gradient accumulation | 32 | Model card and repo config |
| Precision | bf16 | Repo config; artifact is BF16 |
| Warmup ratio | 0.1 | Repo config only |
| Template/history mask | qwen3 / true | Repo config only |
| Repo cutoff length | 16384 | Repo config names Qwen3-8B; not a dedicated 1.7B config |
| RL framework | forked VeRL | README/model card |
| Evaluation | BFCL single/multi, MCP-Atlas pass/mean coverage, tau2-Bench, VitaBench | Model card |

## Source inconsistency

Official prose swaps dataset sizes: the model card says FILTERED has 53.4k, while its metadata/viewer reports 26.5k; the ALL card prose says 26.5k while its viewer reports 53,412. Repository README and live viewers agree: FILTERED ~26.5k, ALL ~53.4k. This audit uses viewer/README counts and records the contradiction.

## Unknown / not specified

- exact optimizer, betas, epsilon, and weight decay
- whether batch/accumulation is per node or global for a stated GPU count
- official 1.7B GPU count/model
- exact 1.7B sequence length
- gradient checkpointing and packing
- all training/shuffle/generation seeds
- LlamaFactory/DeepSpeed versions and commits
- 1.7B checkpoint cadence
- RL reward, hyperparameters, rollout engine, and GPU requirement
- standalone SFT-only 1.7B checkpoint ID
- executable benchmark commands/configs in this checkout

These stay `Unknown / not specified` until primary evidence is found.
