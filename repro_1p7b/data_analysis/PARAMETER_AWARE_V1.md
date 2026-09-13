# Parameter-Aware Data Sampling v1

## Scope

This experiment reweights only the locked EnvFactory SFT training file. It does not read BFCL tasks and does not modify the baseline dataset, base model, environment, caches, or checkpoints.

The current filtered SFT JSON is already flattened into prefix-level training examples. It does not retain the original ToolGraph dependency edges or a separate raw candidate pool. Therefore the experiment uses the allowed proxy route rather than claiming access to exact dependency labels.

## Proxy

For each sample, the analyzer reconstructs the ordered history plus current training pair. A tool-call argument is marked as internally dependent when its scalar value exactly matches a non-trivial scalar parsed from an earlier `<tool_response>` in that prefix.

To reduce obvious false positives, nulls, booleans, -1/0/1, strings shorter than four characters, and a small low-information stoplist are excluded. Dependency depth is propagated through consecutive tool calls.

This is conservative but imperfect: exact reuse can miss transformations and can count coincidental matches. It is a training-data selection proxy, not an evaluation metric.

## Locked baseline distribution

Input: `repro_1p7b/datasets/EnvFactory-SFT-FILTERED/mcp_factory_sft_nips.json`

- Samples: 26,463
- Character proxy budget: 907,126,404
- Samples with at least one internal-dependency proxy: 18,349 (69.3383%)
- Depth bins: 0 = 8,114; 1 = 6,854; 2 = 4,440; 3+ = 7,055
- Tool calls per sample: mean 8.142, median 7, p90 16
- Dependent calls per sample: mean 3.449, median 2, p90 9
- Maximum proxy dependency depth: mean 1.804, median 1, p90 4

Decision: **GO**. The baseline contains enough positive samples to reweight without external generation, while 30.66% of prefixes have no detected dependency and depth-2+ examples remain a minority. A mild, controlled reweighting can materially change exposure without changing the model or SFT recipe.

## Selector lock

- Seed: `20260913`
- Output sample count: 26,463
- Character-length strata: 512-character bins, preserving the number of draws in every bin
- Weight: `1 + 1.25*min(dependent_calls,3) + 0.75*min(max_dependency_depth,3) + 0.20*min(matched_args,5)`
- Sampling: deterministic weighted resampling with replacement inside each length stratum
- Budget control: same sample count and stratum counts; report exact before/after character ratio
- Diversity disclosure: report unique source count and duplicate draws

The 64-sample smoke completed successfully and changed proxy-positive rate from 81.25% to 89.06%, with a character ratio of 0.999638. This smoke is only an implementation check; the formal full-dataset report determines the achieved shift.

## Reproduction

```bash
python repro_1p7b/data_analysis/parameter_aware_data.py analyze \
  --input repro_1p7b/datasets/EnvFactory-SFT-FILTERED/mcp_factory_sft_nips.json \
  --features repro_1p7b/results/parameter_aware/baseline_features.jsonl \
  --summary repro_1p7b/results/parameter_aware/baseline_distribution.json

python repro_1p7b/data_analysis/parameter_aware_data.py select \
  --input repro_1p7b/datasets/EnvFactory-SFT-FILTERED/mcp_factory_sft_nips.json \
  --features repro_1p7b/results/parameter_aware/baseline_features.jsonl \
  --output repro_1p7b/data/parameter_aware_v1/mcp_factory_sft_parameter_aware.json \
  --report repro_1p7b/results/parameter_aware/formal_selection_report.json \
  --sample-count 26463 --seed 20260913 --bin-width 512
```
