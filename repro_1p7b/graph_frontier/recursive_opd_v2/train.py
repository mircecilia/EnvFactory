"""Future V2 DDP trainer. Explicit budget flags and READY manifest are mandatory.

This module is prepared for a later GPU session; CPU prep never invokes train().
No V1 artifact path can be an output target.
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
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, PROTOCOL, RUN as V1_RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import RUN, source_policy_gate
from repro_1p7b.graph_frontier.recursive_opd_v2.gpu_protocol import checked_base_sha
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask


def validate_context_manifest(manifest: dict, rows: list[dict], *, cycle: int) -> int:
    """Fail closed on all V2 cycle length exclusions, including forged READY."""
    from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import LIMIT, account_context

    overflow = manifest.get("context_overflow_records")
    if (cycle not in (0, 1, 2)
            or manifest.get("schema_version") != "recursive_opd_v2_context_accounted_1"
            or manifest.get("max_train_context") != LIMIT
            or not isinstance(overflow, list)
            or manifest.get("context_overflow_count") != len(overflow)
            or manifest.get("semantic_eligible_count") != manifest.get("encoded_rows", -1) + len(overflow)
            or manifest.get("trainable_count") != manifest.get("encoded_rows")
            or manifest.get("context_compatible_count") != manifest.get("encoded_rows")
            or manifest.get("encoded_rows") != len(rows)
            or manifest.get("heldout_training_rows") != 0
            or manifest.get("uncertain_training_rows") != 0
            or manifest.get("skipped_uncertain_count", -1) < 0
            or {row.get("task_id") for row in rows} & {row.get("task_id") for row in overflow}
            or any(row.get("train_context_compatible") is not True
                   or row.get("semantic_eligible") is not True
                   or row.get("execution_verified") is not True
                   or len(row.get("input_ids", [])) > LIMIT for row in rows)):
        raise RuntimeError("V2 context-accounted manifest/row gate failed")
    account_context(semantic_eligible=manifest["semantic_eligible_count"],
                    encoded=len(rows), overflow=overflow)
    return len(overflow)


def load_plan(*, cycle: int, dataset: Path, model: Path, output: Path,
              steps: int, learning_rate: float, seed: int, exposure_budget: int) -> tuple[list[dict], list[int], dict]:
    """Pure CPU preflight. Any missing, unapproved, or V1-lineage input fails."""
    protocol = check()
    if (cycle not in (0, 1, 2) or steps < 1 or learning_rate <= 0 or seed < 0
            or exposure_budget != 2 * steps):
        raise RuntimeError("explicit valid cycle/steps/LR/seed/exposure_budget=2*steps required")
    if output.exists() or V1_RUN.resolve() in output.resolve().parents or RUN.resolve() not in output.resolve().parents:
        raise RuntimeError("output must be new and inside isolated V2 run directory")
    manifest_path = dataset / "manifest.json"
    rows_path = dataset / "dataset.jsonl"
    manifest = json.loads(manifest_path.read_text())
    gates = manifest.get("gates")
    if (manifest.get("status") != "READY_FOR_GPU_TRAINING"
            or manifest.get("usage") != f"TRAINABLE_V2_D{cycle}"
            or manifest.get("cycle") != cycle
            or manifest.get("dataset_sha256") != sha256(rows_path)
            or manifest.get("source_pi0_model_sha256" if cycle == 0 else "source_model_sha256")
               != sha256(model / "model.safetensors")
            or manifest.get("protocol_sha256") != sha256(PROTOCOL)
            or not isinstance(gates, dict) or not gates
            or any(value != "PASS" for value in gates.values())):
        raise RuntimeError("V2 dataset READY/lineage/hash/audit gate failed")
    if cycle == 0:
        if (dataset.resolve() != (RUN / "d0_context16k_v2").resolve()
                or model.resolve() != Path(protocol["pi0_model"]).resolve()
                or manifest.get("source_pi0_model_sha256") != protocol["pi0_model_sha256"]
                or manifest.get("tokenizer_sha256") != sha256(MODEL / "tokenizer.json")):
            raise RuntimeError("V2 D0 pi0/tokenizer identity failed")
        from repro_1p7b.graph_frontier.recursive_opd_v2.context_dataset import FORENSICS
        from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import REPORT
        audit_path = REPORT / "recursive_opd_v2_context_token_audit.json"
        if (manifest.get("forensic_sha256") != sha256(FORENSICS) or not audit_path.exists()):
            raise RuntimeError("V2 context forensic or independent CPU audit missing/drifted")
        audit = json.loads(audit_path.read_text())
        required_gates = {"context_accounting", "single_explicit_context_overflow",
                          "case_b_no_server_truncation",
                          "all_context_compatible_rows_encoded", "heldout_zero_rows",
                          "no_uncertain_rows", "assistant_only_loss_mask", "eos_only_single_label"}
        if (audit.get("verdict") != "READY_FOR_GPU_TRAINING"
                or audit.get("candidate_dataset_sha256") != manifest["dataset_sha256"]
                or not required_gates <= gates.keys()
                or any(gates[name] != "PASS" for name in required_gates)
                or not audit.get("gates")
                or any(value != "PASS" for value in audit["gates"].values())):
            raise RuntimeError("V2 context dataset CPU audit gate failed")
    else:
        prior_path = RUN / f"cycle{cycle}/checkpoint/trainer_state.json"
        prior = json.loads(prior_path.read_text())
        if (prior.get("lineage") != "recursive_opd_v2" or prior.get("status") != "TRAINED"
                or prior.get("cycle") != cycle - 1
                or prior.get("model_sha256") != sha256(model / "model.safetensors")
                or model.resolve() != prior_path.parent.resolve()):
            raise RuntimeError("previous V2 checkpoint identity failed")
        if (dataset.resolve() != (RUN / f"cycle{cycle}/dataset").resolve()
                or prior.get("gpu_experiment_base_sha") != checked_base_sha()
                or manifest.get("gpu_experiment_base_sha") != checked_base_sha()):
            raise RuntimeError("V2 fresh-cycle dataset/base identity failed")
        audit_path = RUN / f"cycle{cycle}/metrics/audit.json"
        if not audit_path.exists():
            raise RuntimeError("fresh-cycle independent audit missing")
        audit = json.loads(audit_path.read_text())
        if (audit.get("status") != "AUDITED"
                or audit.get("dataset_sha256") != manifest["dataset_sha256"]
                or audit.get("model_sha256") != manifest["source_model_sha256"]
                or audit.get("task_count") != 2158
                or not audit.get("gates")
                or any(value != "PASS" for value in audit["gates"].values())):
            raise RuntimeError("fresh-cycle independent audit invalid")
        source_policy_gate(cycle, {"cycle": cycle, "protocol_sha256": sha256(PROTOCOL),
                                   "model_sha256": manifest["source_model_sha256"]}, protocol, prior)
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    overflow_count = validate_context_manifest(manifest, rows, cycle=cycle)
    if (len(rows) != manifest.get("sample_count") or len({row["task_id"] for row in rows}) != len(rows)
            or exposure_budget > len(rows) or manifest.get("heldout_training_rows") != 0):
        raise RuntimeError("V2 sample count/uniqueness/exposure gate failed")
    heldout = set(protocol["heldout_task_ids"])
    for row in rows:
        if row.get("task_id") in heldout or row.get("pool") != "train" or row.get("cycle") != cycle:
            raise RuntimeError("heldout or cycle contamination")
        validate_mask(row)
    if overflow_count:
        print(f"WARNING: {overflow_count} execution-verified semantic-eligible sample excluded "
              f"by declared context limit {manifest['max_train_context']}.", flush=True)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    order = order[:exposure_budget]
    selected = collections.Counter(rows[index]["correction_kind"] for index in order)
    eos_exposures = selected["TERMINATE"] + selected["SUCCESS_TERMINAL_POSITIVE"]
    plan = {"lineage": "recursive_opd_v2", "schema_version": "recursive_opd_v2_ddp_plan_1",
            "gpu_experiment_base_sha": checked_base_sha(),
            "cycle": cycle, "steps": steps, "learning_rate": learning_rate,
            "seed": seed, "exposure_budget": exposure_budget, "world_size": 2,
            "full_parameter": True, "precision": "bfloat16", "attention": "sdpa",
            "gradient_checkpointing": True, "optimizer": "AdamW", "scheduler": "constant",
            "weight_decay": 0.0, "grad_clip": 1.0,
            "dataset_sha256": manifest["dataset_sha256"],
            "initialization_sha256": sha256(model / "model.safetensors"),
            "source_rollout_manifest_sha256": manifest["source_rollout_manifest_sha256"],
            "schedule_task_ids": [rows[index]["task_id"] for index in order],
            "sampled_class_counts": dict(selected),
            "sampled_tool_exposures": selected["START"] + selected["ADVANCE"],
            "sampled_eos_exposures": eos_exposures,
            "sampled_eos_fraction": eos_exposures / len(order),
            "unique_exposed_tasks": len({rows[index]["task_id"] for index in order}),
            "selected_target_tokens": sum(rows[index]["target_tokens"] for index in order)}
    return rows, order, plan


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, local, world = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 2 or local not in (0, 1):
        raise RuntimeError("V2 training requires exactly 2 GPUs")
    rows, order, plan = load_plan(cycle=args.cycle, dataset=args.dataset, model=args.model,
                                  output=args.output, steps=args.steps,
                                  learning_rate=args.learning_rate, seed=args.seed,
                                  exposure_budget=args.exposure_budget)
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(local):
        raise RuntimeError("V2 training requires two A100 GPUs")
    torch.cuda.set_device(local)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dist.init_process_group("nccl")
    log_dir = args.output.parent / f"{args.output.name}_train_manifest"
    if log_dir.exists():
        raise RuntimeError("refusing to overwrite V2 train manifest")
    if rank == 0:
        log_dir.mkdir(parents=True)
        (log_dir / "plan.json").write_text(stable(plan) + "\n")
    dist.barrier()
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(f"cuda:{local}")
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
            raise RuntimeError(f"nonfinite loss at step {step + 1}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        finite = torch.tensor([int(torch.isfinite(norm).item())], device=local)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite grad norm at step {step + 1}")
        optimizer.step()
        local_metrics = torch.tensor(
            [float(loss.detach()), float(norm.detach()),
             torch.cuda.max_memory_allocated(local) / (1024 ** 2)],
            dtype=torch.float64, device=local,
        )
        gathered_metrics = [torch.zeros_like(local_metrics) for _ in range(world)]
        dist.all_gather(gathered_metrics, local_metrics)
        if rank == 0:
            with (log_dir / "training_metrics.jsonl").open("a") as stream:
                stream.write(stable({"step": step + 1, "rank0_task": rows[order[step * 2]]["task_id"],
                                     "rank1_task": rows[order[step * 2 + 1]]["task_id"],
                                     "rank0_loss": gathered_metrics[0][0].item(),
                                     "rank1_loss": gathered_metrics[1][0].item(),
                                     "rank0_grad_norm": gathered_metrics[0][1].item(),
                                     "rank1_grad_norm": gathered_metrics[1][1].item(),
                                     "rank0_peak_allocated_mib": gathered_metrics[0][2].item(),
                                     "rank1_peak_allocated_mib": gathered_metrics[1][2].item(),
                                     "learning_rate": optimizer.param_groups[0]["lr"],
                                     "elapsed_seconds": round(time.monotonic() - started, 2)}) + "\n")
    dist.barrier()
    if rank == 0:
        args.output.mkdir(parents=True)
        model.eval()
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
        tokenizer.save_pretrained(args.output)
        torch.save(optimizer.state_dict(), args.output / "optimizer_state.pt")
        result = {"lineage": "recursive_opd_v2", "status": "TRAINED", "cycle": args.cycle,
                  "gpu_experiment_base_sha": plan["gpu_experiment_base_sha"],
                  "steps": args.steps, "model_sha256": sha256(args.output / "model.safetensors"),
                  "dataset_sha256": plan["dataset_sha256"],
                  "train_plan_sha256": sha256(log_dir / "plan.json")}
        (args.output / "trainer_state.json").write_text(stable(result) + "\n")
        (log_dir / "result.json").write_text(stable(result) + "\n")
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 GPU entry; all budget flags explicit; CPU prep must not run")
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exposure-budget", type=int, required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
