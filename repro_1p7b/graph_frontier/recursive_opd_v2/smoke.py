"""Two-A100 optimizer-free longest-sample V2 DDP forward/backward gate."""
from __future__ import annotations

import json
import os
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import RUN
from repro_1p7b.graph_frontier.recursive_opd_v2.train import load_plan


def run(cycle: int = 0) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM

    rank, local, world = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if cycle not in (0, 1) or world != 2 or local not in (0, 1) or torch.cuda.device_count() != 2:
        raise RuntimeError("V2 smoke requires exactly two GPUs and two ranks")
    model_path = MODEL if cycle == 0 else RUN / "cycle1/checkpoint"
    dataset_path = RUN / "d0_context16k_v2" if cycle == 0 else RUN / "cycle1/dataset"
    output_path = RUN / f"cycle{cycle + 1}/checkpoint"
    seed = 20260927 + cycle
    rows, _, plan = load_plan(
        cycle=cycle, dataset=dataset_path, model=model_path,
        output=output_path, steps=64, learning_rate=2e-6,
        seed=seed, exposure_budget=128,
    )
    index = max(range(len(rows)), key=lambda i: rows[i]["total_tokens"])
    sample = rows[index]
    if sample["total_tokens"] > 16384 or sample["total_tokens"] != len(sample["input_ids"]):
        raise RuntimeError("longest sample violates declared context")
    torch.cuda.set_device(local)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dist.init_process_group("nccl")
    torch.cuda.reset_peak_memory_stats(local)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(f"cuda:{local}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
    ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=local)
    labels = torch.tensor([sample["labels"]], dtype=torch.long, device=local)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = ddp(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite smoke loss")
    loss.backward()  # No optimizer is created and no weights are saved.
    grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("nonfinite smoke gradient norm")
    local_metrics = torch.tensor(
        [float(loss.detach()), float(grad_norm.detach()),
         torch.cuda.max_memory_allocated(local) / (1024 ** 2)],
        dtype=torch.float64, device=local,
    )
    gathered = [torch.zeros_like(local_metrics) for _ in range(world)]
    dist.all_gather(gathered, local_metrics)
    if rank == 0:
        result = {"status": "PASS", "cycle": cycle, "optimizer_updates": 0,
                  "gpu_experiment_base_sha": plan["gpu_experiment_base_sha"],
                  "dataset_sha256": plan["dataset_sha256"],
                  "model_sha256": sha256(model_path / "model.safetensors"),
                  "longest_task_id": sample["task_id"],
                  "longest_total_tokens": sample["total_tokens"],
                  "per_rank_loss": [item[0].item() for item in gathered],
                  "per_rank_grad_norm": [item[1].item() for item in gathered],
                  "per_rank_peak_allocated_mib": [item[2].item() for item in gathered]}
        output = RUN / "gpu_smoke" / (
            "longest_ddp.json" if cycle == 0 else "cycle1_longest_ddp.json")
        if output.exists():
            raise RuntimeError("refusing to overwrite V2 GPU smoke result")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(stable(result) + "\n")
        print(stable(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    run(args.cycle)
