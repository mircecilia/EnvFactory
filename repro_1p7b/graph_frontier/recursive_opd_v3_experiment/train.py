"""Fail-closed two-A100 Complete-C + fresh-anchor full-parameter trainer."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.schedule import (
    LEARNING_RATE, RUN, build_plan, freeze_plan, model_for,
)


def train(cycle: int) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, local, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if cycle not in range(4) or world != 2 or local not in (0, 1):
        raise RuntimeError("only two-GPU updates Cpi0 through Cpi3 are authorized")
    rows, planned = build_plan(cycle)
    frozen = freeze_plan(cycle)
    if frozen != planned:
        raise RuntimeError("immutable exposure schedule drift")
    checkpoint = RUN / f"cycle{cycle + 1}/checkpoint"
    if checkpoint.exists():
        raise RuntimeError("refusing to overwrite checkpoint")
    attempt = os.environ.get("COMPLETE_C_TRAIN_ATTEMPT", "")
    if not attempt or not all(ch.isalnum() or ch in "_-" for ch in attempt):
        raise RuntimeError("unique training attempt identity required")
    temporary = RUN / f"cycle{cycle + 1}/checkpoint_attempt_{attempt}"
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(local):
        raise RuntimeError("training requires two A100 GPUs")
    torch.cuda.set_device(local)
    torch.manual_seed(planned["seed"])
    torch.cuda.manual_seed_all(planned["seed"])
    dist.init_process_group("nccl")
    log_dir = RUN / f"cycle{cycle}/train_metrics_{attempt}"
    if rank == 0:
        if log_dir.exists():
            raise RuntimeError("refusing to overwrite training metrics")
        log_dir.mkdir(parents=True)
        (log_dir / "plan_sha256.txt").write_text(sha256(RUN / f"cycle{cycle}/train_plan.json") + "\n")
    dist.barrier()
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_for(cycle), dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True).to(f"cuda:{local}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    for step in range(planned["optimizer_steps"]):
        row = rows[step * 2 + rank]
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=local)
        labels = torch.tensor([row["labels"]], dtype=torch.long, device=local)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ddp(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        finite = torch.tensor([int(torch.isfinite(loss).item())], device=local)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite loss at step {step + 1}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        finite = torch.tensor([int(torch.isfinite(norm).item())], device=local)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite gradient norm at step {step + 1}")
        optimizer.step()
        metrics = torch.tensor([float(loss.detach()), float(norm.detach()),
                                torch.cuda.max_memory_allocated(local) / 1048576],
                               dtype=torch.float64, device=local)
        gathered = [torch.zeros_like(metrics) for _ in range(world)]
        dist.all_gather(gathered, metrics)
        if rank == 0:
            with (log_dir / "training_metrics.jsonl").open("a") as stream:
                stream.write(stable({"step": step + 1,
                                     "exposure_tags": [planned["stream"][step * 2 + i]["tag"]
                                                       for i in range(2)],
                                     "loss": [value[0].item() for value in gathered],
                                     "grad_norm": [value[1].item() for value in gathered],
                                     "peak_allocated_mib": [value[2].item() for value in gathered],
                                     "elapsed_seconds": round(time.monotonic() - started, 2)}) + "\n")
    dist.barrier()
    if rank == 0:
        if temporary.exists() or checkpoint.exists():
            raise RuntimeError("refusing to overwrite training checkpoint attempt")
        temporary.mkdir(parents=True)
        model.eval()
        model.save_pretrained(temporary, safe_serialization=True)
        AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True,
                                      local_files_only=True).save_pretrained(temporary)
        torch.save(optimizer.state_dict(), temporary / "optimizer_state.pt")
        result = {"lineage": "recursive_opd_v3_complete_c_anchor", "status": "TRAINED",
                  "cycle": cycle, "model_sha256": sha256(temporary / "model.safetensors"),
                  "input_model_sha256": planned["input_model_sha256"],
                  "tokenizer_sha256": sha256(checkpoint / "tokenizer.json"),
                  "natural_dataset_sha256": planned["natural_dataset_sha256"],
                  "anchor_dataset_sha256": planned["anchor_dataset_sha256"],
                  "train_plan_sha256": sha256(RUN / f"cycle{cycle}/train_plan.json"),
                  "optimizer_steps": planned["optimizer_steps"],
                  "natural_exposures": planned["natural_exposures_including_filler"],
                  "anchor_exposures": planned["fresh_anchors_exposed"],
                  "parity_filler": planned["parity_filler"]}
        (temporary / "trainer_state.json").write_text(stable(result) + "\n")
        temporary.rename(checkpoint)
        (log_dir / "result.json").write_text(stable(result) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    train(parser.parse_args().cycle)
