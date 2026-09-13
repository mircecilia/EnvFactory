# Overnight Status

Updated: 2026-09-13T20:06:17+08:00

## Current state

PRELAUNCH: the formal parameter-aware dataset and SFT smoke are complete and validated. A preflight launch exposed Conda incompatibility with shell nounset before training started; the launcher now retains pipefail without nounset and will be committed before launch.

## Parameter-aware dataset

- State: COMPLETE, validation PASS
- Input (read-only): `repro_1p7b/datasets/EnvFactory-SFT-FILTERED/mcp_factory_sft_nips.json`
- Output: `repro_1p7b/data/parameter_aware_v1/mcp_factory_sft_parameter_aware.json`
- Samples: 26,463
- SHA-256: `a043cdc651363c1dbc68da543262f92807e6940016a2c973b6b7e8131cb9517b`
- Proxy-positive rate: 69.3383% -> 92.4725%
- Mean maximum proxy depth: 1.8036 -> 2.7189
- Character budget ratio: 1.000022 (+0.0022%)
- Unique source samples: 14,797 / 26,463 (55.9158%)
- Duplicate draws: 11,666
- Report: `repro_1p7b/results/parameter_aware/formal_selection_report.json`
- Validation: `repro_1p7b/results/parameter_aware/formal_validation.json`
- Selector tmux `parameter_aware_data_v1`: completed and exited

## SFT smoke

- State: COMPLETE, PASS
- Environment: `/home/u2024311031/.conda/envs/envfactory_sglang_1p7b`
- GPUs: 0,1; both returned to 0 MiB task memory
- Base: `repro_1p7b/models/Qwen3-1.7B`
- Method: full SFT, BF16, DeepSpeed ZeRO-3, cutoff 8,192
- Samples/effective batch/steps: 64 / 64 / 1
- Trainable parameters: 1,720,574,976 (100%)
- Loss: 1.7975304
- Runtime: 55.7397 seconds
- Exit code: 0
- Log: `repro_1p7b/logs/parameter_aware/sft_smoke.log`
- Output: `repro_1p7b/checkpoints/parameter_aware_sft_smoke_8k_1p7b/`

## Formal SFT task

- State: PRELAUNCH
- Preflight: first launch exited before training because Conda activation referenced unset ADDR2LINE; no GPU/output artifact was created
- Planned tmux: `parameter_aware_sft_8k_1p7b`
- Planned pane PID: recorded after launch
- Config: `repro_1p7b/configs/llamafactory_sft_parameter_aware_8k_1p7b.yaml`
- Launcher: `repro_1p7b/scripts/launch_parameter_aware_sft_8k_1p7b.sh`
- Log: `repro_1p7b/logs/parameter_aware/formal_sft_8k_1p7b.log`
- Exit code file: `repro_1p7b/logs/parameter_aware/formal_sft_8k_1p7b.exit_code`
- Output: `repro_1p7b/checkpoints/parameter_aware_sft_8k_1p7b/`
- Expected optimizer steps: 414
- Expected duration: approximately 6 hours based on the completed baseline; this is an estimate

Launch:

```bash
tmux new-session -d -s parameter_aware_sft_8k_1p7b \
  'bash repro_1p7b/scripts/launch_parameter_aware_sft_8k_1p7b.sh'
```

Inspect:

```bash
tmux has-session -t parameter_aware_sft_8k_1p7b
tmux list-panes -t parameter_aware_sft_8k_1p7b -F 'pane_pid=#{pane_pid} command=#{pane_current_command}'
tmux capture-pane -pt parameter_aware_sft_8k_1p7b -S -120
tail -n 120 repro_1p7b/logs/parameter_aware/formal_sft_8k_1p7b.log
nvidia-smi
```

Success requires exit code 0, progress 414/414, final train metrics, and a loadable final checkpoint.
