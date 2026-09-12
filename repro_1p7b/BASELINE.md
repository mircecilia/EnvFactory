# Locked 1.7B SFT baseline

## Fixed inputs

- EnvFactory source: `298e4744d329b2f88dc16347cc7734348d40464f`
- Qwen3-1.7B revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- FILTERED dataset revision: `ee97a07fd01fe5902ac7ac6358dbbfa5403cc683`
- LlamaFactory commit: `100e9a42c6c09f8f7849b70d60f3da445fb2024b`
- Seeds: 42 for model/trainer and data

The official dataset was downloaded directly on the remote host. Its released
`mcp_factory_sft_nips.json` is 989,025,444 bytes and contains 26,463 records.
The Hub README prose says 53,400 samples, but the actual fixed-revision file and
Dataset Viewer expose 26,463. This baseline uses the real file count and does not
duplicate samples.

## Dataset audit

Every record contains `instruction`, `input`, `output`, `system`, and
`history`. All 26,463 input fields are empty strings, all history fields are
lists, 1,262 histories are empty, all outputs contain `<think>`, and 12,917
outputs contain `<tool_call>`. Machine-readable evidence is in
`results/baseline/dataset_audit.json`.

## Formal recipe

The config `configs/llamafactory_sft_baseline_1p7b.yaml` locks:

- full SFT with the Qwen3 template and history masking;
- BF16 and DeepSpeed ZeRO-3;
- cutoff length 16,384;
- batch size 1 per GPU and gradient accumulation 32;
- learning rate 1e-6, cosine schedule, warmup ratio 0.1, one epoch;
- two GPUs, giving effective batch size 64.

For 26,463 records, the installed Transformers 4.57.1 formula calculates 414
optimizer steps. The dataset, base model, logs, and checkpoints are Git-ignored.

## Exact gate

The bounded gate retained the formal model, full-SFT method, BF16, ZeRO-3, Qwen3
template, history masking, and 16,384 cutoff. Only the sample bound,
accumulation, reporting, saving, and max-step controls were shortened.

It trained all 1,720,574,976 parameters across both A100-PCIE-40GB GPUs for one
optimizer step. Loss was 2.4957048892974854 and runtime was 15.5102 seconds.
Observed process-level snapshots were 37,338 MiB on GPU 0 and 29,436 MiB on GPU
1. DeepSpeed reported one allocator cache flush under high memory pressure.
Both GPUs returned to zero task memory after completion. Machine-readable
evidence is in `results/baseline/sft_gate_summary.json`.

The 4.18 host kernel remains below Accelerate's recommended 5.5, so the formal
run must be monitored for hangs. Gate status: PASS.

## Formal launch

The long run is prepared but not started until unattended command authority is
transferred.

```bash
cd /home/u2024311031/workspace/envfactory_repro_1p7b
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/u2024311031/.conda/envs/envfactory_sglang_1p7b
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
set -o pipefail
mkdir -p repro_1p7b/logs
CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
  llamafactory-cli train repro_1p7b/configs/llamafactory_sft_baseline_1p7b.yaml \
  2>&1 | tee repro_1p7b/logs/baseline_sft_1p7b.log
```

- GPUs: 0 and 1.
- Log: `repro_1p7b/logs/baseline_sft_1p7b.log`.
- Output: `repro_1p7b/checkpoints/baseline_sft_1p7b/`.
- Expected data: 26,463 records, one epoch, 414 optimizer steps.
- Expected files: model weights/config/tokenizer, trainer state, training args,
  train results, and TensorBoard event data.
- Success: shell exit status zero, progress reaches 414/414, train metrics are
  written, and the final checkpoint loads without error.

After it finishes, report the shell exit status and the last 100 log lines so
