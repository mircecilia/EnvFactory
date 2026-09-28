"""Optimizer-free two-A100 longest-selected-exposure DDP memory smoke."""
from __future__ import annotations

import argparse
import os

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.schedule import RUN, build_plan, model_for


def run(cycle: int) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM

    rank, local, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if cycle not in range(4) or world != 2 or local not in (0, 1) or torch.cuda.device_count() != 2:
        raise RuntimeError("smoke requires two isolated A100 GPUs")
    rows, plan = build_plan(cycle)
    sample = max(rows, key=lambda row: len(row["input_ids"]))
    if len(sample["input_ids"]) > 16384:
        raise RuntimeError("selected sample exceeds declared context")
    model_path = model_for(cycle)
    torch.cuda.set_device(local)
    torch.manual_seed(plan["seed"])
    torch.cuda.manual_seed_all(plan["seed"])
    dist.init_process_group("nccl")
    torch.cuda.reset_peak_memory_stats(local)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True).to(f"cuda:{local}")
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
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("nonfinite smoke grad norm")
    local_metrics = torch.tensor([float(loss.detach()), float(grad_norm.detach()),
                                  torch.cuda.max_memory_allocated(local) / 1048576],
                                 dtype=torch.float64, device=local)
    gathered = [torch.zeros_like(local_metrics) for _ in range(world)]
    dist.all_gather(gathered, local_metrics)
    if rank == 0:
        result = {"status": "PASS", "cycle": cycle, "optimizer_updates": 0,
                  "input_model_sha256": sha256(model_path / "model.safetensors"),
                  "natural_dataset_sha256": plan["natural_dataset_sha256"],
                  "anchor_dataset_sha256": plan["anchor_dataset_sha256"],
                  "longest_task_id": sample["task_id"],
                  "longest_total_tokens": len(sample["input_ids"]),
                  "per_rank_loss": [x[0].item() for x in gathered],
                  "per_rank_grad_norm": [x[1].item() for x in gathered],
                  "per_rank_peak_allocated_mib": [x[2].item() for x in gathered]}
        output = RUN / f"cycle{cycle}/gpu_smoke.json"
        if output.exists():
            raise RuntimeError("refusing to overwrite smoke result")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(stable(result) + "\n")
        print(stable(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    run(parser.parse_args().cycle)
