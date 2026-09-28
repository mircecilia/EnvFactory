"""Fail-closed future two-GPU trainer for audited Complete-C data.

CPU preparation never imports torch through this module and never calls train.
Every budget field and an exact immutable approval manifest are mandatory.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import PROTOCOL, check
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import LIMIT
from repro_1p7b.graph_frontier.recursive_opd_v3_complete_c.prepare import RUN, VERSION
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

APPROVAL_SCHEMA = "recursive_opd_v3_complete_c_training_approval_1"


def load_plan(*, cycle: int, dataset: Path, model: Path, output: Path,
              steps: int, exposure_budget: int, learning_rate: float,
              seed: int, approval_manifest: Path) -> tuple[list[dict], list[int], dict]:
    protocol = check()
    if (cycle not in range(5) or steps < 1 or exposure_budget != 2 * steps
            or learning_rate <= 0 or seed < 0):
        raise RuntimeError("explicit cycle/steps/LR/seed and exposure_budget=2*steps required")
    expected = RUN / f"cycle{cycle}/dataset"
    if dataset.resolve() != expected.resolve():
        raise RuntimeError("dataset must be one audited Complete-C natural union")
    if output.exists() or RUN.resolve() not in output.resolve().parents:
        raise RuntimeError("output must be new and inside isolated Complete-C runtime")
    manifest_path, rows_path = dataset / "manifest.json", dataset / "dataset.jsonl"
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if (manifest.get("schema_version") != VERSION or manifest.get("status") != "READY_CPU_DATA"
            or manifest.get("training_steps") != "UNDECIDED"
            or manifest.get("exposure_budget") != "UNDECIDED"
            or manifest.get("learning_rate") != "UNDECIDED"
            or manifest.get("cycle") != cycle or manifest.get("dataset_sha256") != sha256(rows_path)
            or manifest.get("sample_count") != len(rows) or manifest.get("heldout_training_rows") != 0
            or manifest.get("uncertain_training_rows") != 0 or not manifest.get("gates")
            or any(value != "PASS" for value in manifest["gates"].values())
            or manifest.get("protocol_sha256") != sha256(PROTOCOL)
            or manifest.get("source_policy_sha256") != sha256(model / "model.safetensors")
            or exposure_budget > len(rows)):
        raise RuntimeError("Complete-C dataset/hash/provenance/size gate failed")
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    identities, per_task = set(), collections.Counter()
    for row in rows:
        validate_mask(row)
        identity = row.get("complete_c_sample_identity")
        if (not identity or identity in identities or row.get("task_id") not in train
                or row.get("task_id") in heldout or row.get("pool") != "train"
                or row.get("source_verdict") == "UNCERTAIN"
                or len(row.get("input_ids", [])) > LIMIT):
            raise RuntimeError("Complete-C row split/identity/provenance/context gate failed")
        identities.add(identity)
        per_task[row["task_id"]] += 1
    if any(value > 2 for value in per_task.values()):
        raise RuntimeError("Complete-C task contributes more than two rows")
    approval = json.loads(approval_manifest.read_text())
    expected_approval = {
        "schema_version": APPROVAL_SCHEMA, "approved": True, "cycle": cycle,
        "dataset_sha256": manifest["dataset_sha256"],
        "model_sha256": manifest["source_policy_sha256"], "steps": steps,
        "exposure_budget": exposure_budget, "learning_rate": learning_rate,
        "seed": seed, "output": str(output.resolve()),
    }
    if approval != expected_approval:
        raise RuntimeError("explicit Complete-C user approval does not match all parameters")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    order = order[:exposure_budget]
    selected = collections.Counter(rows[index]["complete_c_source_label"] for index in order)
    plan = {
        "lineage": "recursive_opd_v3_complete_c",
        "schema_version": "recursive_opd_v3_complete_c_ddp_plan_1",
        "cycle": cycle, "steps": steps, "exposure_budget": exposure_budget,
        "learning_rate": learning_rate, "seed": seed, "world_size": 2,
        "full_parameter": True, "precision": "bfloat16", "attention": "sdpa",
        "gradient_checkpointing": True, "optimizer": "AdamW", "scheduler": "constant",
        "weight_decay": 0.0, "grad_clip": 1.0,
        "dataset_sha256": manifest["dataset_sha256"],
        "initialization_sha256": manifest["source_policy_sha256"],
        "approval_manifest_sha256": sha256(approval_manifest),
        "uniform_deterministic_shuffle": True, "class_specific_quota": None,
        "schedule_sample_identities": [rows[index]["complete_c_sample_identity"] for index in order],
        "sampled_class_counts": dict(selected),
        "selected_target_tokens": sum(rows[index]["target_tokens"] for index in order),
    }
    return rows, order, plan


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, local, world = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 2 or local not in (0, 1):
        raise RuntimeError("Complete-C training requires exactly two GPUs")
    rows, order, plan = load_plan(
        cycle=args.cycle, dataset=args.dataset, model=args.model, output=args.output,
        steps=args.steps, exposure_budget=args.exposure_budget,
        learning_rate=args.learning_rate, seed=args.seed,
        approval_manifest=args.approval_manifest)
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(local):
        raise RuntimeError("Complete-C training requires two A100 GPUs")
    torch.cuda.set_device(local)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dist.init_process_group("nccl")
    log_dir = args.output.parent / f"{args.output.name}_train_manifest"
    if log_dir.exists():
        raise RuntimeError("refusing to overwrite Complete-C train manifest")
    if rank == 0:
        log_dir.mkdir(parents=True)
        (log_dir / "plan.json").write_text(stable(plan) + "\n")
    dist.barrier()
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True).to(f"cuda:{local}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.learning_rate, weight_decay=0.0)
    for step in range(args.steps):
        row = rows[order[step * 2 + rank]]
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=local)
        labels = torch.tensor([row["labels"]], dtype=torch.long, device=local)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ddp(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        finite = torch.tensor([int(torch.isfinite(loss).item())], device=local)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite Complete-C loss at step {step + 1}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        finite = torch.tensor([int(torch.isfinite(norm).item())], device=local)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite Complete-C grad norm at step {step + 1}")
        optimizer.step()
        metrics = torch.tensor([float(loss.detach()), float(norm.detach()),
                                torch.cuda.max_memory_allocated(local) / (1024 ** 2)],
                               dtype=torch.float64, device=local)
        gathered = [torch.zeros_like(metrics) for _ in range(world)]
        dist.all_gather(gathered, metrics)
        if rank == 0:
            with (log_dir / "training_metrics.jsonl").open("a") as stream:
                stream.write(stable({
                    "step": step + 1,
                    "rank0_sample": rows[order[step * 2]]["complete_c_sample_identity"],
                    "rank1_sample": rows[order[step * 2 + 1]]["complete_c_sample_identity"],
                    "rank0_loss": gathered[0][0].item(), "rank1_loss": gathered[1][0].item(),
                    "rank0_grad_norm": gathered[0][1].item(),
                    "rank1_grad_norm": gathered[1][1].item(),
                    "rank0_peak_allocated_mib": gathered[0][2].item(),
                    "rank1_peak_allocated_mib": gathered[1][2].item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }) + "\n")
    dist.barrier()
    if rank == 0:
        args.output.mkdir(parents=True)
        model.eval()
        model.save_pretrained(args.output, safe_serialization=True)
        AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, local_files_only=True).save_pretrained(args.output)
        torch.save(optimizer.state_dict(), args.output / "optimizer_state.pt")
        result = {"lineage": "recursive_opd_v3_complete_c", "status": "TRAINED",
                  "cycle": args.cycle, "steps": args.steps,
                  "model_sha256": sha256(args.output / "model.safetensors"),
                  "dataset_sha256": plan["dataset_sha256"],
                  "train_plan_sha256": sha256(log_dir / "plan.json"),
                  "approval_manifest_sha256": plan["approval_manifest_sha256"]}
        (args.output / "trainer_state.json").write_text(stable(result) + "\n")
        (log_dir / "result.json").write_text(stable(result) + "\n")
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete-C GPU entry; all budgets explicit")
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--exposure-budget", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approval-manifest", type=Path, required=True)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
