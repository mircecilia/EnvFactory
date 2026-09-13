# BFCL V3 Evaluation Alignment Audit

## Decision

**Status: NO-GO for freezing the present 9.75% run as the project-wide comparison baseline.**

The official scorer completed all 800 tasks and no evaluator, category-selection, or gross parser bug was found. However, the actual resolved Base and SFT inference configurations differ in random_seed while sampling at temperature 0.7. BFCL sends no per-request seed. Consequently this single-run +8/800 result is descriptive, but it does not satisfy the requested "model path is the only difference" protocol lock.

This is separate from the paper discrepancy classification below: that is **Case B (known/possible protocol mismatch with insufficient public detail)**, not evidence that either the paper or this evaluator is wrong.

## Recovered artifacts

- Worktree: /home/u2024311031/workspace/envfactory_repro_1p7b
- Branch: repro/envfactory-1p7b
- Base raw result/score: repro_1p7b/evaluation/bfcl/artifacts/full/base/{result,score}
- SFT raw result/score: repro_1p7b/evaluation/bfcl/artifacts/full/sft/{result,score}
- Aggregate: each run's score/data_multi_turn.csv; data_overall.csv is not applicable because only multi_turn was generated.
- Full logs: repro_1p7b/evaluation/bfcl/logs/full/
- Shared launch driver: repro_1p7b/evaluation/bfcl/scripts/run_bfcl.sh

## Model identities

### Base

- Path: repro_1p7b/models/Qwen3-1.7B
- Family/architecture: Qwen3-1.7B / Qwen3ForCausalLM
- Locked source revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
- The same locked Base is recorded by repro_1p7b/BASELINE.md as the input to the formal SFT run.

### SFT

- Final evaluated checkpoint: repro_1p7b/checkpoints/baseline_sft_8k_1p7b
- This is the final root export, not sft_smoke, a gate, a pilot, or checkpoint-414.
- Trainer state: epoch 1.0, global step 414, max steps 414.
- Dataset: LARK-Lab/EnvFactory-SFT-FILTERED, locked revision ee97a07fd01fe5902ac7ac6358dbbfa5403cc683, actual file count 26,463.
- Training: full SFT, not LoRA.

## Benchmark identity and completeness

- Benchmark checkout: /home/u2024311031/benchmarks/gorilla_bfcl_v1p3/berkeley-function-call-leaderboard
- Commit: ea13468e4423454d0c213704fb87cf7cb3990433
- Checkout status at audit: clean.
- Version/task selection: BFCL V3 Multi-Turn through the official BFCL v1.3 package/checkout.
- Actual benchmark JSONL counts: Base 200, Miss Func 200, Miss Param 200, Long Context 200; total 800.
- Both result sets contain 800 unique shared IDs.
- Missing IDs: 0. Duplicate IDs: 0. Invalid-category IDs: 0.
- Per-category score aggregates match correctness reconstructed from unique failure IDs. Duplicate long-context failure-detail rows do not alter the unique-ID result.

Recomputed scores:

| Category | Base | SFT | Delta |
|---|---:|---:|---:|
| Base | 25/200 = 12.50% | 27/200 = 13.50% | +1.00 pp |
| Miss Func | 15/200 = 7.50% | 22/200 = 11.00% | +3.50 pp |
| Miss Param | 17/200 = 8.50% | 16/200 = 8.00% | -0.50 pp |
| Long Context | 13/200 = 6.50% | 13/200 = 6.50% | 0.00 pp |
| **Overall Multi-Turn** | **70/800 = 8.75%** | **78/800 = 9.75%** | **+1.00 pp** |

## Paper / official alignment

The external reference is the [EnvFactory paper](https://arxiv.org/html/2605.18703) and its [official repository](https://github.com/LARK-AI-Lab/EnvFactory). The benchmark implementation is the [BFCL v1.3 checkout](https://github.com/ShishirPatil/gorilla/tree/v1.3/berkeley-function-call-leaderboard).

| Setting | Paper / official public information | Our evaluation | Match? |
|---|---|---|---|
| BFCL | BFCL V3; exact commit/snapshot not specified | V3 Multi-Turn, BFCL v1.3 commit ea13468e... | Unknown exact snapshot |
| Task set | Multi-Turn result reported; exact IDs/count snapshot not specified | Four official files, 200 each, 800 total | Likely, not provable |
| Model | Qwen3-1.7B family | Qwen3-1.7B locked revision 70d244cc... | Family match; revision unknown |
| Thinking mode | Temperature 0.7 is stated for thinking models; exact enable_thinking control is not stated | Thinking enabled: no no-think marker; 7,570/7,591 Base and 10,900/11,010 SFT assistant steps contain reasoning | Consistent, exact flag unknown |
| Backend | SGLang | SGLang | Yes |
| Temperature | 0.7 for thinking | 0.7 | Yes |
| Top-p | Not specified | Not explicitly passed by BFCL request | Unknown |
| Max generation | Not specified | Official handler dynamically requests at most 4,096 tokens | Unknown |
| Context | Not specified | 40,960 | Unknown |
| Tool parser | Not specified | Official v1.3 QwenFCHandler client-side XML/JSON parser | Unknown |
| Server parser | Not specified | None; expected because the completions endpoint returns manual XML text | Unknown |
| Chat template/system prompt | Not specified | Official v1.3 manual Qwen3 FC template and tool instructions | Unknown |
| Tensor parallel | Default TP=2 | TP=1 | **Known mismatch** |
| Request seed | Not specified | Not passed | Unknown/non-reproducible |
| Resolved server seed | Not specified | Base 668997380; SFT 315606697 | **Internal mismatch** |
| Deterministic inference | Not specified | False | Unknown |
| SFT epochs | Paper appendix says 3; public repo/model-card recipe says 1 | 1 epoch | Public sources conflict; ours is 1 |

The paper reports Qwen3-1.7B Base Multi-Turn 16.75 and SFT 23.25. Public information is insufficient to reconstruct its exact BFCL commit, model revision, prompt/parser implementation, top-p, generation cap, context length, or seed. TP differs, and the public SFT epoch descriptions conflict. These facts reject exact parity claims but do not establish which item caused the absolute-score gap. The Base gap cannot be explained by SFT training differences.

## Actual Base/SFT inference comparison

inspect_eval_alignment.py reads the resolved ServerArgs lines from both server logs and compares selected fields. The same shared driver sets model ID, backend, temperature, dtype, context, TP, attention backend, sampling backend, category, and input logging.

The two observed differences are:

1. model path (expected);
2. server random_seed (not expected under the requested lock).

The BFCL request code passes temperature, prompt, and dynamic max_tokens, but no seed or top_p. With temperature 0.7, the seed difference is material to reproducibility. It is not scorer corruption, but it prevents a strictly controlled one-factor attribution of this small single-run delta.

## Parser and tool-use health

| Metric | Base | SFT |
|---|---:|---:|
| Normal (non-overflow) tasks | 777 | 771 |
| Raw tool-call blocks | 6,570 | 8,856 |
| Mean calls per all 800 tasks | 8.21 | 11.07 |
| JSON parse failures | 5 | 26 |
| Empty tool blocks | 0 | 0 |
| Invalid call schema | 16 | 7 |
| Malformed arguments | 8 | 21 |
| Structurally malformed total | 29 (0.44%) | 54 (0.61%) |
| Unknown/unoffered tool name | 266 | 615 |
| Offered-name structurally valid calls | 6,275 | 8,187 |
| Handler decode-error entries | 23 | 26 |
| Normal tasks with no tool block | 8 | 0 |

Structural/parser failure rates are low and do not support a wholesale adapter or evaluator parser bug. Unknown names and repeated/wrong calls are model behavior: the parser decoded them, but the names were not in the current offered tool set. SFT produces substantially more calls and more unknown-name calls.

## Generation truncation

The result format does not retain provider finish_reason, so exact finish reason cannot be recovered. Using the official handler's dynamic request cap, output count exactly reached that cap:

- Base: 20 requests across 19 tasks; 18 of those tasks were wrong.
- SFT: 116 requests across 110 tasks; 109 of those tasks were wrong.

This is a strong truncation symptom, not proof for every case. The asymmetry is real and consistent with SFT's longer trajectories, but both models used the same generation rule. It may depress SFT performance; it does not explain the lower Base absolute score because the paper's max generation length is not disclosed.

## Long-context overflow

- Both overflow: 20 tasks.
- Base only: 3 tasks.
- SFT only: 9 tasks.
- Neither: 768 tasks.
- Every overflow task is wrong: Base 0/23 correct, SFT 0/29 correct.
- Among nine SFT-only overflows, the full log records more requests before rejection than Base for 5 and no increase for 4.

Rejected result rows discard partial trajectories. More requests is testable from the full log, but added reasoning length, repeated calls, and tool-result length cannot be separated for all nine without guessing.

## Discrepancy classification

**Case B: protocol mismatch / under-specification, impact not identifiable from public evidence.**

No wrong category, missing evaluation, score aggregation error, wrong checkpoint, disabled thinking mode, or systemic parser bug was found. Known differences and unknowns are enough to reject exact paper-parity claims, but not to attribute the entire gap or treat the paper score as ground truth.

## Go / No-Go and recommendation

**NO-GO** applies narrowly to freezing this exact single-run result as the project-wide controlled comparison baseline. It does not invalidate the raw 800/800 evaluation or descriptive paired audit.

Before future Baseline-vs-Ours claims:

1. lock a request/server seed and persist the resolved launch/request config;
2. run a small repeated determinism smoke first;
3. if stable, rerun full Base and baseline SFT with the locked protocol before comparing a new method;
4. keep BFCL commit, model revision, handler, task IDs, temperature, top-p, max-token rule, context length, and evaluator fixed.

No full rerun was started during this audit.
