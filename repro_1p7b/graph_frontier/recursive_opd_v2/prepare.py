"""CPU-only V2 D0 builder from immutable V1 pi0 rollouts and retrospective audit.

No model inference, training, or new environment execution occurs in this module.
The V1 execution-verified classifications are retained, then checked against the
persisted typed episode and exact student prompt boundaries before encoding.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import (
    encode_first_divergence, encode_terminal_stop,
)
from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for, valid_gold
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import typed_equal
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, OUT, decode
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, PROTOCOL, RUN as V1_RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import GOLD_RUN
from repro_1p7b.graph_frontier.state_verifier import compare_final_states
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

VERSION = "recursive_opd_v2_cpu_prep_1"
ROOT = V1_RUN.parent
RUN = ROOT / "recursive_opd_v2_run"
REPORT = ROOT / "reports"
STATUSES = frozenset({"FIRST_ERROR", "SUCCESS", "UNCERTAIN"})
REGIMES = frozenset({"START", "ADVANCE", "TERMINATE"})


def digest(value: object) -> str:
    return hashlib.sha256(stable(value).encode()).hexdigest()


def source_policy_gate(cycle: int, rollout_manifest: dict, protocol: dict,
                       previous_v2_checkpoint: dict | None = None) -> str:
    """Reject V1 later-policy rollouts as V2 Dk, even if their format matches."""
    if rollout_manifest.get("cycle") != cycle or rollout_manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise RuntimeError("rollout cycle/protocol identity drift")
    if cycle == 0:
        if (rollout_manifest.get("model_sha256") != protocol["pi0_model_sha256"]
                or Path(rollout_manifest.get("model_path", "")).resolve() != Path(protocol["pi0_model"]).resolve()
                or sha256(MODEL / "model.safetensors") != protocol["pi0_model_sha256"]):
            raise RuntimeError("V2 D0 must originate from frozen original Dynamic-v1 pi0")
        return "TRAINABLE_V2_D0"
    if not isinstance(previous_v2_checkpoint, dict):
        raise RuntimeError("V1 pi1/pi2/pi3 rollouts are RETROSPECTIVE_ONLY")
    if (previous_v2_checkpoint.get("lineage") != "recursive_opd_v2"
            or previous_v2_checkpoint.get("status") != "TRAINED"
            or previous_v2_checkpoint.get("cycle") != cycle - 1
            or previous_v2_checkpoint.get("model_sha256") != rollout_manifest.get("model_sha256")):
        raise RuntimeError("V2 cycle source must match the previous V2 checkpoint hash")
    return f"TRAINABLE_V2_D{cycle}"


def load_cycle(cycle: int, protocol: dict) -> tuple[list[dict], dict, dict]:
    base = V1_RUN / f"cycle{cycle}"
    manifest_path = base / "rollout_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("cycle") != cycle or manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise RuntimeError(f"cycle{cycle} rollout manifest drift")
    summary = json.loads((base / "metrics/summary.json").read_text())
    metrics_path = base / "metrics/per_task.jsonl"
    if summary.get("per_task_sha256") != sha256(metrics_path):
        raise RuntimeError(f"cycle{cycle} V1 verifier metrics hash drift")
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    if (len(rows) != len(train) + len(heldout) or len({r["task_id"] for r in rows}) != len(rows)
            or {r["task_id"] for r in rows} != train | heldout):
        raise RuntimeError(f"cycle{cycle} task coverage/identity invalid")
    for row in rows:
        if (row.get("cycle") != cycle or row.get("status") not in STATUSES
                or row.get("pool") != ("train" if row["task_id"] in train else "heldout")):
            raise RuntimeError(f"cycle{cycle} V1 verdict/split drift")
        if row["status"] == "FIRST_ERROR" and row.get("error_regime") not in REGIMES:
            raise RuntimeError("first-error regime missing")
        if row["status"] != "FIRST_ERROR" and row.get("error_regime") is not None:
            raise RuntimeError("non-first-error has correction regime")
    if {r["query_hash"] for r in rows if r["pool"] == "train"} & {
            r["query_hash"] for r in rows if r["pool"] == "heldout"}:
        raise RuntimeError("train/heldout query-hash leakage")
    return rows, manifest, {"manifest_sha256": sha256(manifest_path),
                            "metrics_sha256": sha256(metrics_path)}


def natural_stats(rows: list[dict], cycle: int) -> dict:
    out = {"cycle": cycle, "usage": "TRAINABLE_V2_D0" if cycle == 0 else "RETROSPECTIVE_ONLY"}
    for pool in ("train", "heldout"):
        group = [r for r in rows if r["pool"] == pool]
        counts = collections.Counter(r["status"] for r in group)
        regimes = collections.Counter(r["error_regime"] for r in group if r["status"] == "FIRST_ERROR")
        tool = regimes["START"] + regimes["ADVANCE"]
        eos = regimes["TERMINATE"] + counts["SUCCESS"]
        depth = {}
        for name, subset in (("1", [r for r in group if r["gold_tool_count"] == 1]),
                             ("2", [r for r in group if r["gold_tool_count"] == 2]),
                             ("3+", [r for r in group if r["gold_tool_count"] >= 3])):
            depth[name] = {"tasks": len(subset), "success_positive": sum(r["status"] == "SUCCESS" for r in subset),
                           "terminate_correction": sum(r["error_regime"] == "TERMINATE" for r in subset)}
        out[pool] = {"tasks": len(group), "SUCCESS": counts["SUCCESS"],
                     "FIRST_ERROR": counts["FIRST_ERROR"], "UNCERTAIN": counts["UNCERTAIN"],
                     "START": regimes["START"], "ADVANCE": regimes["ADVANCE"],
                     "TERMINATE": regimes["TERMINATE"],
                     "SUCCESS_TERMINAL_POSITIVE": counts["SUCCESS"],
                     "v1_tool_targets": tool, "v1_eos_targets": regimes["TERMINATE"],
                     "v1_eos_fraction": regimes["TERMINATE"] / counts["FIRST_ERROR"] if counts["FIRST_ERROR"] else 0,
                     "v2_tool_targets": tool, "v2_eos_targets": eos,
                     "v2_natural_rows": tool + eos,
                     "v2_eos_fraction": eos / (tool + eos) if tool + eos else 0,
                     "by_gold_depth": depth}
    return out


def checked_prefix(record: dict, episode: dict) -> list[dict]:
    """Require the exact pre-decision student history, never a static gold path."""
    student, gold = episode["student"], episode["gold"]
    if not student or student.get("status") not in {
            "ROLLOUT_COMPLETE", "ROLLOUT_CONTEXT_OVERFLOW"} or not student.get("steps"):
        raise RuntimeError("student trajectory absent")
    if student["status"] == "ROLLOUT_CONTEXT_OVERFLOW":
        boundary = student.get("context_overflow") or {}
        if (record["status"] != "FIRST_ERROR"
                or student.get("termination_reason") != "CONTEXT_OVERFLOW"
                or boundary.get("step_index") != len(student["steps"])):
            raise RuntimeError("context-boundary rollout has no earlier verified first error")
    if record["status"] == "SUCCESS":
        step = student["steps"][-1]
        if (record["frontier_depth"] != len(gold["actions"])
                or student.get("termination_reason") != "MODEL_FINAL"
                or (step.get("model_message") or {}).get("tool_calls")
                or step.get("structured_calls") or step.get("parse_errors")
                or not typed_equal(step.get("state_before_action"), gold["final_state"])
                or not typed_equal(student.get("final_state"), gold["final_state"])):
            raise RuntimeError("SUCCESS terminal execution/stop gate failed")
    elif record["status"] == "FIRST_ERROR":
        index = record.get("student_step")
        if not isinstance(index, int) or index < 0 or index >= len(student["steps"]):
            raise RuntimeError("FIRST_ERROR step pointer absent")
        step = student["steps"][index]
        if step.get("step_index") != index:
            raise RuntimeError("FIRST_ERROR step index mismatch")
        pointer = record["frontier_depth"]
        if record["error_regime"] == "TERMINATE":
            if (pointer != len(gold["actions"])
                    or not (step.get("model_message") or {}).get("tool_calls")
                    or not typed_equal(step.get("state_before_action"), gold["final_state"])):
                raise RuntimeError("TERMINATE is not verified terminal over-continuation")
        else:
            if (pointer >= len(gold["actions"])
                    or not typed_equal(record.get("gold_next_action"), gold["actions"][pointer])
                    or not typed_equal(step.get("state_before_action"),
                                       gold["initial_state"] if pointer == 0 else gold["states_after"][pointer-1])):
                raise RuntimeError("START/ADVANCE gold-next-action pointer invalid")
    else:
        raise RuntimeError("UNCERTAIN cannot produce a V2 row")
    count, messages = step.get("prompt_message_count"), student["messages"]
    if (not isinstance(count, int) or count < 1 or count >= len(messages)
            or not typed_equal(messages[count], step.get("model_message"))):
        raise RuntimeError("saved student prompt boundary/message mismatch")
    prefix = messages[:count]
    if record["status"] == "SUCCESS":
        if (count != len(messages) - 1 or prefix[-1].get("role") != "tool"
                or any(m is messages[-1] for m in prefix)
                or record.get("termination_reason") != "MODEL_FINAL"):
            raise RuntimeError("SUCCESS prefix includes final assistant or lacks final tool observation")
        # V1 SUCCESS must also satisfy the source final-config verifier.
        # Exact state equality above is stricter than compare_final_states.
    else:
        if not typed_equal(record.get("prompt_messages"), prefix):
            raise RuntimeError("V1 first-error prompt is not the student prefix")
        if record["error_regime"] == "TERMINATE" and prefix[-1].get("role") != "tool":
            raise RuntimeError("TERMINATE prefix lacks terminal tool observation")
    return prefix


def load_verified_episode(record: dict, manifest: dict, manifest_sha: str,
                          source: list[dict], entries: dict[str, dict]) -> tuple[dict, list[dict], str]:
    task = record["task_id"]
    episode = json.loads((V1_RUN / f"cycle{record['cycle']}/rollouts/{task}.json").read_text())
    entry = entries[task]
    gold_source = ((V1_RUN / "gold_cache" / f"{task}.json")
                   if episode.get("gold_source_kind") == "independent_replay_cache"
                   else (GOLD_RUN / "episodes" / f"{task}.json"))
    if (episode.get("task_id") != task or episode.get("cycle") != record["cycle"]
            or episode.get("pool") != record["pool"]
            or episode.get("model_sha256") != manifest["model_sha256"]
            or episode.get("manifest_sha256") != manifest_sha
            or episode.get("gold_source_episode_sha256") != sha256(gold_source)
            or episode.get("student_error")
            or not valid_gold(entry, source[entry["source_row_index"]], episode)):
        raise RuntimeError(f"persisted rollout/gold/verifier identity failed: {task}")
    schema = schema_for(V1_RUN if episode.get("gold_source_kind") == "independent_replay_cache"
                        else GOLD_RUN, episode)
    return episode, schema, episode["gold_source_episode_sha256"]


def encode_row(tokenizer, record: dict, episode: dict, schema: list[dict],
               gold_source_sha: str, *, cycle: int = 0) -> dict:
    prefix = checked_prefix(record, episode)
    task = record["task_id"]
    if record["status"] == "SUCCESS":
        semantics = "SUCCESS_TERMINAL_POSITIVE"
    else:
        semantics = record["error_regime"]
    sample = {"task_id": task, "state_identity": f"{task}:v2:cycle{cycle}:{semantics}:{record['frontier_depth']}",
              "conversation_prefix": prefix}
    if semantics in {"TERMINATE", "SUCCESS_TERMINAL_POSITIVE"}:
        item = encode_terminal_stop(tokenizer, {**sample, "replay_valid": True,
                                                 "final_state_match": True, "tool_schema": schema})
    else:
        item = encode_first_divergence(tokenizer, {**sample, "independent_prefix_replay_valid": True,
                                                   "gold_action": record["gold_next_action"]},
                                       {"task_id": task, "tools": schema})
    validate_mask(item)
    if (item["tool_observation_tokens_in_loss"] or item["historical_assistant_tokens_in_loss"]
            or (semantics in {"TERMINATE", "SUCCESS_TERMINAL_POSITIVE"}) != (item["target_type"] == "terminal_stop")):
        raise RuntimeError("token target semantics/mask drift")
    item.update(source_task_id=task, cycle=cycle, pool="train", source_verdict=record["status"],
                correction_kind=semantics, source_policy_usage=f"TRAINABLE_V2_D{cycle}",
                frontier_depth=record["frontier_depth"], gold_tool_count=record["gold_tool_count"],
                query_hash=record["query_hash"], student_prefix_sha256=digest(prefix),
                student_prompt_message_count=len(prefix),
                gold_source_episode_sha256=gold_source_sha)
    return item


def prepare(output: Path = RUN / "d0") -> dict:
    """Create immutable encoded D0 and retrospective report; fail before writing on any gate."""
    from transformers import AutoTokenizer

    if output.exists():
        raise RuntimeError(f"refusing to overwrite V2 D0: {output}")
    protocol = check()
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    if len(train) != 1726 or len(heldout) != 432 or train & heldout:
        raise RuntimeError("frozen split invalid")
    cycle_rows, manifests, hashes, retrospective = {}, {}, {}, []
    for cycle in range(4):
        rows, manifest, identity = load_cycle(cycle, protocol)
        if cycle == 0:
            source_policy_gate(0, manifest, protocol)
        cycle_rows[cycle], manifests[cycle], hashes[cycle] = rows, manifest, identity
        retrospective.append(natural_stats(rows, cycle))
    source = json.loads(SOURCE.read_text())
    entries = {e["audit_id"]: e for e in json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.eos_token_id != 151645 or sha256(MODEL / "tokenizer.json") != protocol["tokenizer_sha256"]:
        raise RuntimeError("Qwen3 tokenizer/EOS identity drift")
    encoded, gold_sources, encoding_failures = [], {}, []
    for position, record in enumerate(cycle_rows[0]):
        if record["pool"] != "train" or record["status"] == "UNCERTAIN":
            continue
        episode, schema, gold_sha = load_verified_episode(record, manifests[0],
                                                           hashes[0]["manifest_sha256"], source, entries)
        try:
            encoded.append(encode_row(tokenizer, record, episode, schema, gold_sha))
        except ValueError as exc:
            # Do not truncate an exact on-policy prefix or silently drop a
            # SUCCESS retention target. Preserve the candidate data, but the
            # full D0 training gate fails until every eligible row encodes.
            encoding_failures.append({"task_id": record["task_id"], "status": record["status"],
                                      "correction_kind": record.get("error_regime") or "SUCCESS_TERMINAL_POSITIVE",
                                      "reason": str(exc)})
        gold_sources[record["task_id"]] = gold_sha
        if (position + 1) % 250 == 0:
            print(stable({"encoded": len(encoded), "inspected_records": position + 1}), flush=True)
    expected = retrospective[0]["train"]
    count = collections.Counter(row["correction_kind"] for row in encoded)
    if (len(encoded) + len(encoding_failures) != expected["v2_natural_rows"]
            or len({row["task_id"] for row in encoded}) != len(encoded)
            or len({row["task_id"] for row in encoding_failures}) != len(encoding_failures)
            or {row["task_id"] for row in encoded} & heldout
            or any(row["source_verdict"] == "UNCERTAIN" for row in encoded)):
        raise RuntimeError("one-task-one-target/count/heldout data gate failed")
    all_encoded = not encoding_failures and all(count[key] == expected[key] for key in (
        "START", "ADVANCE", "TERMINATE", "SUCCESS_TERMINAL_POSITIVE"))
    output.mkdir(parents=True)
    dataset_path = output / "dataset.jsonl"
    dataset_path.write_text("".join(stable(row) + "\n" for row in encoded))
    split_sha = digest({"train_task_ids": protocol["train_task_ids"],
                        "heldout_task_ids": protocol["heldout_task_ids"]})
    manifest = {"schema_version": VERSION,
                "status": "READY_FOR_GPU_TRAINING" if all_encoded else "NOT_READY",
                "cycle": 0,
                "usage": "TRAINABLE_V2_D0", "builder_version": VERSION,
                "source_pi0_model_sha256": manifests[0]["model_sha256"],
                "source_rollout_manifest_sha256": hashes[0]["manifest_sha256"],
                "source_v1_metrics_sha256": hashes[0]["metrics_sha256"],
                "gold_source_sha256": digest(gold_sources),
                "source_clean_dataset_sha256": sha256(SOURCE),
                "task_split_sha256": split_sha, "tokenizer_sha256": sha256(MODEL / "tokenizer.json"),
                "protocol_sha256": sha256(PROTOCOL), "dataset_sha256": sha256(dataset_path),
                "sample_count": len(encoded), "eligible_task_count": expected["v2_natural_rows"],
                "encoding_failures": encoding_failures, "class_counts": dict(count),
                "tool_targets": count["START"] + count["ADVANCE"],
                "eos_targets": count["TERMINATE"] + count["SUCCESS_TERMINAL_POSITIVE"],
                "natural_eos_fraction": expected["v2_eos_fraction"],
                "heldout_training_rows": 0,
                "gates": {name: "PASS" for name in (
                    "frozen_pi0_source", "v1_verifier_metrics_hashes", "gold_source_identity",
                    "student_exact_predecision_prefix", "success_verified_terminal_and_stop",
                    "terminate_verified_overcontinuation", "gold_next_tool_pointer",
                    "one_task_one_target", "zero_task_overlap", "zero_query_hash_overlap",
                    "heldout_zero_rows", "no_uncertain_rows", "tokenizer_eos_identity",
                    "assistant_only_loss_mask", "eos_only_single_label", "dataset_hash")}}
    manifest["gates"]["all_eligible_rows_encoded"] = "PASS" if all_encoded else "FAIL"
    (output / "manifest.json").write_text(stable(manifest) + "\n")
    report = {"verdict": manifest["status"], "d0": manifest,
              "retrospective": retrospective,
              "retrospective_sources": [{"cycle": c, "usage": "TRAINABLE_V2_D0" if c == 0 else "RETROSPECTIVE_ONLY",
                                         **hashes[c]} for c in range(4)],
              "not_run": ["V2 pi0->pi1 training", "V2 pi1 rollout", "V2 pi1->pi2 training",
                          "V2 pi2 rollout", "V2 pi2->pi3 training", "V2 pi3 rollout",
                          "GPU benchmarks"],
              "unapproved_gpu_parameters": ["optimizer_steps", "learning_rate", "exposure_budget"]}
    REPORT.mkdir(parents=True, exist_ok=True)
    report_path = REPORT / "recursive_opd_v2_cpu_prep.json"
    if report_path.exists():
        raise RuntimeError("refusing to overwrite V2 CPU report")
    report_path.write_text(stable(report) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "prepare-context"))
    args = parser.parse_args()
    if args.command == "prepare":
        answer = prepare()
        print(stable({"verdict": answer["verdict"], "d0": answer["d0"]["class_counts"],
                      "dataset_sha256": answer["d0"]["dataset_sha256"]}), flush=True)
    elif args.command == "prepare-context":
        from repro_1p7b.graph_frontier.recursive_opd_v2.context_dataset import build
        build()


if __name__ == "__main__":
    main()
