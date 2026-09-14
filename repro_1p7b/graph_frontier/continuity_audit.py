#!/usr/bin/env python3
"""Static and runtime gates for Dynamic Graph-Frontier training continuity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def complete_checkpoint(path: Path, step: int) -> dict[str, bool]:
    zero_dir = path / f"global_step{step}"
    return {
        "trainer_state": (path / "trainer_state.json").is_file(),
        "scheduler": (path / "scheduler.pt").is_file(),
        "model": (path / "model.safetensors").is_file(),
        "rng_rank0": (path / "rng_state_0.pth").is_file(),
        "rng_rank1": (path / "rng_state_1.pth").is_file(),
        "optimizer_rank0": (zero_dir / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").is_file(),
        "optimizer_rank1": (zero_dir / "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt").is_file(),
    }


def audit_config(args: argparse.Namespace) -> None:
    one = yaml.safe_load(args.stage1.read_text()); two = yaml.safe_load(args.stage2.read_text())
    checks = {
        "same_output_dir": one["output_dir"] == two["output_dir"],
        "same_model": one["model_name_or_path"] == two["model_name_or_path"] == "repro_1p7b/models/Qwen3-1.7B",
        "fixed_414_horizon": one.get("max_steps") == two.get("max_steps") == 414,
        "stage_counts": one.get("max_samples") == 13232 and two.get("max_samples") == 13231,
        "stage2_ignore_data_skip": two.get("ignore_data_skip") is True,
        "budget_settings": all(
            cfg.get("per_device_train_batch_size") == 1
            and cfg.get("gradient_accumulation_steps") == 32
            and cfg.get("cutoff_len") == 8192
            and cfg.get("learning_rate") == 1.0e-6
            and cfg.get("bf16") is True
            for cfg in (one, two)
        ),
        "scheduler_settings": all(
            cfg.get("lr_scheduler_type") == "cosine" and cfg.get("warmup_ratio") == 0.1
            for cfg in (one, two)
        ),
        "checkpoint_settings": all(
            cfg.get("save_strategy") == "steps" and cfg.get("save_steps") == 30
            and cfg.get("save_total_limit") == 2 and cfg.get("save_only_model") is False
            for cfg in (one, two)
        ),
        "deepspeed_same": one.get("deepspeed") == two.get("deepspeed") == "repro_1p7b/configs/ds_z3_config.json",
        "data_seed_fixed": one.get("data_seed") == two.get("data_seed") == 20260914,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "stage1_count": 13232, "stage2_count": 13231, "total_exposure": 26463,
        "world_size": 2, "global_batch_size": 64, "stage_boundary_step": 207,
        "scheduler_horizon_steps": 414, "expected_total_optimizer_steps": 414,
        "continuity_option": "Option B: native DeepSpeed resume with a fixed-horizon boundary callback",
    }
    write(args.report, report); print(json.dumps(report, indent=2))
    if report["status"] != "PASS": raise SystemExit(1)


def verify_checkpoint(args: argparse.Namespace) -> None:
    checks = complete_checkpoint(args.checkpoint, args.step)
    state = json.loads((args.checkpoint / "trainer_state.json").read_text()) if checks["trainer_state"] else {}
    checks["global_step"] = state.get("global_step") == args.step
    checks["scheduler_horizon"] = state.get("max_steps") == args.horizon
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "checkpoint": str(args.checkpoint)}
    write(args.report, result); print(json.dumps(result, indent=2))
    if result["status"] != "PASS": raise SystemExit(1)


def verify_smoke(args: argparse.Namespace) -> None:
    output = args.output
    begin0 = json.loads((output / "continuity_audit/train_begin_step_0.json").read_text())
    begin2 = json.loads((output / "continuity_audit/train_begin_step_2.json").read_text())
    state2 = json.loads((output / "checkpoint-2/trainer_state.json").read_text())
    state4 = json.loads((output / "checkpoint-4/trainer_state.json").read_text())
    trace = [json.loads(line) for line in (output / "continuity_audit/step_trace.jsonl").read_text().splitlines() if line.strip()]
    steps = [row["global_step"] for row in trace]
    checks = {
        "checkpoint2_complete": all(complete_checkpoint(output / "checkpoint-2", 2).values()),
        "checkpoint4_complete": all(complete_checkpoint(output / "checkpoint-4", 4).values()),
        "global_step_2_to_3_to_4": state2.get("global_step") == 2 and state4.get("global_step") == 4 and steps == [1, 2, 3, 4],
        "fixed_scheduler_horizon": state2.get("max_steps") == state4.get("max_steps") == 4,
        "scheduler_restored": begin2.get("scheduler_last_epoch") == 2,
        "optimizer_restored": (begin2.get("optimizer_state_entries") or 0) > 0,
        "rng_restorable": all((output / "checkpoint-2" / f"rng_state_{rank}.pth").is_file() for rank in (0, 1)),
        "stage2_dataset_changed": begin0.get("tokenized_signature") != begin2.get("tokenized_signature"),
        "stage2_dataset_size": begin2.get("dataset_length") == 4,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "train_begin_stage1": begin0, "train_begin_stage2": begin2, "step_trace": trace,
    }
    write(args.report, result); print(json.dumps(result, indent=2))
    if result["status"] != "PASS": raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("config")
    p.add_argument("--stage1", type=Path, required=True); p.add_argument("--stage2", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True); p.set_defaults(func=audit_config)
    p = sub.add_parser("checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--step", type=int, required=True)
    p.add_argument("--horizon", type=int, required=True); p.add_argument("--report", type=Path, required=True); p.set_defaults(func=verify_checkpoint)
    p = sub.add_parser("smoke")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--report", type=Path, required=True); p.set_defaults(func=verify_smoke)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__": main()
