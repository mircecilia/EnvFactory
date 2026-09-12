# Under-10-hour SFT pilot

This pilot is a capacity and stability run, not the completed baseline. It keeps
the exact formal model, dataset, full-SFT method, BF16 precision, ZeRO-3,
16,384 cutoff, learning rate, scheduler, seeds, and effective batch size. It
changes the training duration to 32 optimizer steps and saves at steps 16 and 32.
At effective batch size 64, it processes 2,048 sample presentations, about 7.7%
of the 26,463-record training set.

The one-step gate took 15.5102 seconds with accumulation 1. Linear scaling for
32 steps with accumulation 32 is about 4.4 hours. Allowing for preprocessing,
checkpointing, sample-length variation, and allocator pressure, the operational
estimate is 5-7 hours. This is an estimate rather than a guarantee.

## Start from a VSCode remote terminal

```bash
cd /home/u2024311031/workspace/envfactory_repro_1p7b
tmux new -s envfactory_sft_pilot_1p7b
```

Inside tmux:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/u2024311031/.conda/envs/envfactory_sglang_1p7b
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
set -o pipefail
mkdir -p repro_1p7b/logs

CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
  llamafactory-cli train repro_1p7b/configs/llamafactory_sft_pilot_1p7b.yaml \
  2>&1 | tee repro_1p7b/logs/baseline_sft_pilot_1p7b.log

train_rc=${PIPESTATUS[0]}
printf '%s\n' "$train_rc" | \
  tee repro_1p7b/logs/baseline_sft_pilot_1p7b.exit_code
```

Detach with Ctrl-B then D. Reattach with:

```bash
tmux attach -t envfactory_sft_pilot_1p7b
```

The run succeeds when the saved exit code is 0, progress reaches 32/32, metrics
are written, and the final checkpoint exists under
`repro_1p7b/checkpoints/baseline_sft_pilot_1p7b/`.
