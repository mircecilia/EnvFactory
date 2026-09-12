# Under-10-hour SFT pilot

This pilot is a capacity and stability run, not the completed baseline. It keeps
the formal model, dataset, full-SFT method, BF16 precision, ZeRO-3, learning
rate, scheduler, seeds, and effective batch size. The cutoff is explicitly
resource-adapted from 16,384 to 8,192, and duration is limited to 32 optimizer
steps with checkpoints at steps 16 and 32. At effective batch size 64, it
processes 2,048 sample presentations, about 7.7% of the 26,463-record set.

The 16K accumulation-32 pilot OOMed after completing its first optimizer step.
The 8K accumulation-32 two-step steady-state gate completed in 96.8905 seconds
(about 48.45 seconds per optimizer step) after one-time preprocessing. Linear
training time for 32 steps is about 25.8 minutes; allowing for startup, logging,
and two checkpoints, the operational budget is under one hour. This is an
estimate rather than a guarantee.

## Start from a VSCode remote terminal

```bash
cd /home/u2024311031/workspace/envfactory_repro_1p7b
tmux new -s envfactory_sft_pilot_8k_1p7b
```

Inside tmux:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/u2024311031/.conda/envs/envfactory_sglang_1p7b
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
set -o pipefail
mkdir -p repro_1p7b/logs

CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
  llamafactory-cli train repro_1p7b/configs/llamafactory_sft_pilot_8k_1p7b.yaml \
  2>&1 | tee repro_1p7b/logs/baseline_sft_pilot_8k_1p7b.log

train_rc=${PIPESTATUS[0]}
printf '%s\n' "$train_rc" | \
  tee repro_1p7b/logs/baseline_sft_pilot_8k_1p7b.exit_code
```

Detach with Ctrl-B then D. Reattach with:

```bash
tmux attach -t envfactory_sft_pilot_8k_1p7b
```

The run succeeds when the saved exit code is 0, progress reaches 32/32, metrics
are written, and the final checkpoint exists under
`repro_1p7b/checkpoints/baseline_sft_pilot_8k_1p7b/`.
