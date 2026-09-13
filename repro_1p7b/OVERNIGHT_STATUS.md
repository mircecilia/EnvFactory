# Overnight Status

Updated: 2026-09-13T19:54:45+08:00

## Current state

PRELAUNCH: parameter-aware full-distribution analysis and 64-sample selector smoke have passed. The formal selector will be launched only after this implementation commit is pushed.

## Completed

- Locked input: `repro_1p7b/datasets/EnvFactory-SFT-FILTERED/mcp_factory_sft_nips.json` (read-only)
- Full analysis: PASS, 26,463 samples
- Baseline dependency proxy: 18,349 / 26,463 = 69.3383%
- Unit tests: PASS (3/3)
- Selector smoke: PASS (64/64 valid)
- Smoke dependency proxy: 81.25% -> 89.06%
- Smoke character budget ratio: 0.999638
- GO/NO-GO: GO

## Formal parameter-aware dataset task

- State: PRELAUNCH
- Planned tmux: `parameter_aware_data_v1`
- Environment: system Python 3.11.5; no packages installed or modified
- Planned log: `repro_1p7b/logs/parameter_aware/formal_selection.log`
- Planned output: `repro_1p7b/data/parameter_aware_v1/mcp_factory_sft_parameter_aware.json`
- Planned report: `repro_1p7b/results/parameter_aware/formal_selection_report.json`

Command:

```bash
python repro_1p7b/data_analysis/parameter_aware_data.py select \
  --input repro_1p7b/datasets/EnvFactory-SFT-FILTERED/mcp_factory_sft_nips.json \
  --features repro_1p7b/results/parameter_aware/baseline_features.jsonl \
  --output repro_1p7b/data/parameter_aware_v1/mcp_factory_sft_parameter_aware.json \
  --report repro_1p7b/results/parameter_aware/formal_selection_report.json \
  --sample-count 26463 --seed 20260913 --bin-width 512
```

Inspection:

```bash
tmux has-session -t parameter_aware_data_v1
tmux capture-pane -pt parameter_aware_data_v1 -S -120
tail -n 120 repro_1p7b/logs/parameter_aware/formal_selection.log
```

## Next gated steps

1. Validate formal JSON and inspect distribution, character budget, and diversity.
2. Add a separate LlamaFactory dataset mapping and full-SFT config that differ from baseline only in dataset/output paths.
3. Recheck GPU ownership, then run a small SFT smoke from the original Qwen3-1.7B base.
4. If smoke passes, commit and push configs before launching formal SFT in a separate tmux session.

Formal SFT is not started yet.
