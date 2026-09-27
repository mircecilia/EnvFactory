"""CPU-only post-training checkpoint, plan, and 64-step trace validation."""
from __future__ import annotations

import json
import math
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v2.gpu_protocol import checked_base_sha
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import RUN


def validate(cycle: int) -> dict:
    if cycle not in (0, 1):
        raise ValueError("only V2 training cycles 0 and 1 are in scope")
    checkpoint = RUN / f"cycle{cycle + 1}/checkpoint"
    plan_path = RUN / f"cycle{cycle + 1}/checkpoint_train_manifest/plan.json"
    metrics_path = plan_path.parent / "training_metrics.jsonl"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    plan = json.loads(plan_path.read_text())
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    if (state.get("lineage") != "recursive_opd_v2" or state.get("status") != "TRAINED"
            or state.get("cycle") != cycle or state.get("steps") != 64
            or state.get("model_sha256") != sha256(checkpoint / "model.safetensors")
            or state.get("train_plan_sha256") != sha256(plan_path)
            or state.get("dataset_sha256") != plan.get("dataset_sha256")
            or state.get("gpu_experiment_base_sha") != checked_base_sha()
            or plan.get("gpu_experiment_base_sha") != checked_base_sha()
            or plan.get("cycle") != cycle or plan.get("steps") != 64
            or plan.get("seed") != 20260927 + cycle
            or plan.get("learning_rate") != 2e-6 or plan.get("world_size") != 2
            or plan.get("exposure_budget") != 128 or plan.get("full_parameter") is not True
            or plan.get("precision") != "bfloat16"
            or plan.get("unique_exposed_tasks") != 128
            or sum(plan.get("sampled_class_counts", {}).values()) != 128
            or len(metrics) != 64
            or [item.get("step") for item in metrics] != list(range(1, 65))
            or any(item.get("learning_rate") != 2e-6
                   or not all(math.isfinite(item[key]) for key in (
                       "rank0_loss", "rank1_loss", "rank0_grad_norm", "rank1_grad_norm"))
                   for item in metrics)):
        raise RuntimeError("V2 checkpoint/training protocol or finite-metric gate failed")
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        tensor_count = len(weights.keys())
    if tensor_count < 100:
        raise RuntimeError("V2 checkpoint tensor count suspiciously small")
    AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    if tokenizer.eos_token_id != 151645:
        raise RuntimeError("V2 checkpoint tokenizer EOS mismatch")
    result = {"status": "CHECKPOINT_VALIDATED", "cycle": cycle,
              "gpu_experiment_base_sha": checked_base_sha(),
              "model_sha256": state["model_sha256"],
              "config_sha256": sha256(checkpoint / "config.json"),
              "tokenizer_sha256": sha256(checkpoint / "tokenizer.json"),
              "train_plan_sha256": sha256(plan_path),
              "training_metrics_sha256": sha256(metrics_path),
              "tensor_count": tensor_count,
              "steps": 64, "wall_seconds_to_last_step": metrics[-1]["elapsed_seconds"],
              "peak_allocated_mib_by_rank": [max(row[f"rank{rank}_peak_allocated_mib"]
                                                 for row in metrics) for rank in (0, 1)],
              "sampled_class_counts": plan["sampled_class_counts"]}
    output = checkpoint / "checkpoint_validation.json"
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise RuntimeError("existing V2 checkpoint validation report drift")
    else:
        output.write_text(stable(result) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    args = parser.parse_args()
    print(stable(validate(args.cycle)), flush=True)
