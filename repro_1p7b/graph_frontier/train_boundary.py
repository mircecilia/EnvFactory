#!/usr/bin/env python3
"""Run LLaMAFactory with a fixed scheduler horizon and an early stage boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from transformers import TrainerCallback


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def dataset_signature(train_dataloader: Any) -> dict[str, Any]:
    dataset = getattr(train_dataloader, "dataset", None)
    length = len(dataset) if dataset is not None and hasattr(dataset, "__len__") else None
    digest = hashlib.sha256()
    inspected = 0
    if dataset is not None and length:
        for index in sorted(set((0, min(1, length - 1), length - 1))):
            row = dataset[index]
            for key in ("input_ids", "labels"):
                value = row.get(key, []) if isinstance(row, dict) else []
                digest.update(key.encode())
                digest.update(json.dumps(list(value), separators=(",", ":")).encode())
            inspected += 1
    return {"dataset_length": length, "inspected_rows": inspected, "tokenized_signature": digest.hexdigest()}


class ContinuityAuditCallback(TrainerCallback):
    def __init__(self, output_dir: Path, stop_after_step: int | None) -> None:
        self.output_dir = output_dir
        self.stop_after_step = stop_after_step
        self.audit_dir = output_dir / "continuity_audit"

    @staticmethod
    def _rank_zero(state: Any) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._rank_zero(state):
            return control
        optimizer = kwargs.get("optimizer")
        inner = getattr(optimizer, "optimizer", optimizer)
        optimizer_state = getattr(inner, "state", {})
        scheduler = kwargs.get("lr_scheduler")
        payload = {
            "event": "train_begin",
            "global_step": int(state.global_step),
            "max_steps": int(state.max_steps),
            "learning_rate": list(scheduler.get_last_lr()) if scheduler is not None else None,
            "scheduler_last_epoch": getattr(scheduler, "last_epoch", None),
            "optimizer_state_entries": len(optimizer_state) if hasattr(optimizer_state, "__len__") else None,
            **dataset_signature(kwargs.get("train_dataloader")),
        }
        atomic_json(self.audit_dir / f"train_begin_step_{state.global_step}.json", payload)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._rank_zero(state):
            scheduler = kwargs.get("lr_scheduler")
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            with (self.audit_dir / "step_trace.jsonl").open("a") as handle:
                handle.write(json.dumps({
                    "global_step": int(state.global_step),
                    "learning_rate": list(scheduler.get_last_lr()) if scheduler is not None else None,
                    "scheduler_last_epoch": getattr(scheduler, "last_epoch", None),
                }) + "\n")
        if self.stop_after_step is not None and state.global_step >= self.stop_after_step:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if self._rank_zero(state) and self.stop_after_step is not None and state.global_step == self.stop_after_step:
            scheduler = kwargs.get("lr_scheduler")
            atomic_json(self.audit_dir / f"boundary_step_{state.global_step}.json", {
                "event": "stage_boundary_saved",
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "scheduler_last_epoch": getattr(scheduler, "last_epoch", None),
                "learning_rate": list(scheduler.get_last_lr()) if scheduler is not None else None,
                "checkpoint": str(self.output_dir / f"checkpoint-{state.global_step}"),
            })
        return control


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--ignore-data-skip", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("Training config must be a mapping")
    if int(config.get("max_steps", -1)) <= 0:
        raise ValueError("A fixed positive max_steps scheduler horizon is required")
    if args.stop_after_step is not None and args.stop_after_step >= int(config["max_steps"]):
        raise ValueError("Boundary must be earlier than the scheduler horizon")
    if args.resume_from_checkpoint:
        config["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.ignore_data_skip:
        config["ignore_data_skip"] = True
    output_dir = Path(config["output_dir"])
    from llamafactory.train.tuner import run_exp
    run_exp(config, callbacks=[ContinuityAuditCallback(output_dir, args.stop_after_step)])


if __name__ == "__main__":
    main()
