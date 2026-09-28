"""Full fresh-cycle verifier, Complete-C pool, and one-cycle anchor builder."""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import encode_assistant_target
from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for, valid_gold
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import OUT, SOURCE
from repro_1p7b.graph_frontier.recursive_opd_v1.mine import classify
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, PROTOCOL, RUN as V1_RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import GOLD_RUN
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import LIMIT, sample_lengths
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import checked_prefix, encode_row
from repro_1p7b.graph_frontier.recursive_opd_v3_complete_c.current_positive import extract_current_continuation
from repro_1p7b.graph_frontier.recursive_opd_v3_fresh_anchor.anchor import (
    exact_overlap, extract_fresh_anchor, fresh_identity,
)
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.rollout import checked, integrity
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.schedule import CPU_RUN, RUN, model_for
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sources() -> tuple[list[dict], dict[str, dict]]:
    raw = json.loads(SOURCE.read_text())
    entries = {r["audit_id"]: r for r in json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    return raw, entries


def _episode(cycle: int, task: str, manifest_sha: str, model_sha: str,
             raw: list[dict], entries: dict[str, dict]) -> tuple[dict, list[dict], str]:
    path = RUN / f"cycle{cycle}/rollouts/{task}.json"
    episode = json.loads(path.read_text())
    entry = entries[task]
    row = raw[entry["source_row_index"]]
    gold_path = (V1_RUN / "gold_cache" / f"{task}.json" if
                 episode.get("gold_source_kind") == "independent_replay_cache" else
                 GOLD_RUN / "episodes" / f"{task}.json")
    if (episode.get("task_id") != task or episode.get("cycle") != cycle
            or episode.get("model_sha256") != model_sha
            or episode.get("manifest_sha256") != manifest_sha
            or episode.get("gold_source_episode_sha256") != sha256(gold_path)
            or episode.get("student_error") or not episode.get("student")
            or not valid_gold(entry, row, episode)):
        raise RuntimeError(f"fresh episode/gold provenance failed: {task}")
    return episode, schema_for(RUN, episode), sha256(path)


def _metrics(records: list[dict]) -> dict:
    answer = {}
    for pool in ("train", "heldout"):
        subset = [r for r in records if r["pool"] == pool]
        answer[pool] = {
            "tasks": len(subset), "success": sum(r["status"] == "SUCCESS" for r in subset),
            "first_action_accuracy": sum(r["frontier_depth"] > 0 for r in subset) / len(subset),
            "start_error": sum(r["error_regime"] == "START" for r in subset),
            "advance_error": sum(r["error_regime"] == "ADVANCE" for r in subset),
            "terminate_error": sum(r["error_regime"] == "TERMINATE" for r in subset),
            "uncertain": sum(r["status"] == "UNCERTAIN" for r in subset),
            "mean_frontier_depth": statistics.mean(r["frontier_depth"] for r in subset),
            "mean_normalized_frontier": statistics.mean(r["normalized_frontier"] for r in subset),
            "verified_terminal_reached": sum(r["frontier_depth"] == r["gold_tool_count"] for r in subset),
            "over_continue": sum(r["error_regime"] == "TERMINATE" for r in subset),
            "premature_stop": sum(r["status"] == "FIRST_ERROR"
                                  and r["error_regime"] in {"START", "ADVANCE"}
                                  and r["termination_reason"] == "MODEL_FINAL" for r in subset),
        }
    return answer


def classify_full(cycle: int, workers: int = 4) -> dict:
    if workers not in range(1, 7):
        raise ValueError("verifier concurrency must be 1..6")
    manifest, manifest_sha = checked(cycle)
    full = integrity(cycle)
    if full["task_count"] != 2158:
        raise RuntimeError("cannot mine a partial rollout")
    protocol = check()
    raw, entries = _sources()
    tasks = protocol["train_task_ids"] + protocol["heldout_task_ids"]
    train = set(protocol["train_task_ids"])
    root = RUN / f"cycle{cycle}/metrics"
    if root.exists():
        summary = json.loads((root / "summary.json").read_text())
        rows_path = root / "per_task.jsonl"
        if (summary["cycle"] != cycle or summary["per_task_sha256"] != sha256(rows_path)
                or summary["rollout_manifest_sha256"] != manifest_sha
                or len(_rows(rows_path)) != 2158):
            raise RuntimeError("persisted fresh metrics drift")
        return summary

    def one(task: str) -> dict:
        episode, schema, _ = _episode(cycle, task, manifest_sha,
                                      manifest["model_sha256"], raw, entries)
        entry = entries[task]
        row = raw[entry["source_row_index"]]
        verdict = asyncio.run(classify(episode, entry, row, schema))
        verdict.update(task_id=task, cycle=cycle,
                       pool="train" if task in train else "heldout",
                       environment=entry["environment"], query_hash=entry["query_hash"],
                       termination_reason=episode["student"]["termination_reason"])
        return verdict

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="complete-c-verifier") as pool:
        records = list(pool.map(one, tasks))
    if (len(records) != 2158 or {r["task_id"] for r in records} != set(tasks)
            or any(r["pool"] != ("train" if r["task_id"] in train else "heldout") for r in records)
            or {r["query_hash"] for r in records if r["pool"] == "train"}
               & {r["query_hash"] for r in records if r["pool"] == "heldout"}):
        raise RuntimeError("full verifier/split/query leakage gate failed")
    root.mkdir(parents=True)
    rows_path = root / "per_task.jsonl"
    rows_path.write_text("".join(stable(r) + "\n" for r in records))
    summary = {"cycle": cycle, "status": "VERIFIED", "metrics": _metrics(records),
               "rollout_manifest_sha256": manifest_sha,
               "rollout_integrity_sha256": sha256(RUN / f"cycle{cycle}/rollout_integrity.json"),
               "per_task_sha256": sha256(rows_path)}
    (root / "summary.json").write_text(stable(summary) + "\n")
    return summary


def _chat_lengths(tokenizer, prefix: list[dict], target: dict,
                  schema: list[dict]) -> tuple[int, int]:
    kwargs = {"tools": schema, "tokenize": True, "enable_thinking": False}
    prompt = tokenizer.apply_chat_template(prefix, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(prefix + [target], add_generation_prompt=False, **kwargs)
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise RuntimeError("assistant target is not an exact suffix")
    return len(prompt), len(full)


def _tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True,
                                               local_files_only=True)
    protocol = check()
    if tokenizer.eos_token_id != 151645 or sha256(MODEL / "tokenizer.json") != protocol["tokenizer_sha256"]:
        raise RuntimeError("Qwen3 tokenizer/EOS drift")
    return tokenizer


def build_natural(cycle: int) -> dict:
    summary = classify_full(cycle)
    manifest, manifest_sha = checked(cycle)
    protocol = check()
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    raw, entries = _sources()
    records = {r["task_id"]: r for r in _rows(RUN / f"cycle{cycle}/metrics/per_task.jsonl")}
    root = RUN / f"cycle{cycle}/natural"
    if root.exists():
        saved = json.loads((root / "manifest.json").read_text())
        if (saved["dataset_sha256"] != sha256(root / "dataset.jsonl")
                or saved["source_policy_sha256"] != manifest["model_sha256"]):
            raise RuntimeError("persisted natural pool drift")
        return saved
    tokenizer = _tokenizer()
    encoded, overflow = [], []
    label_counts = collections.Counter()
    for task in protocol["train_task_ids"]:
        record = records[task]
        if record["status"] == "UNCERTAIN":
            continue
        episode, schema, rollout_sha = _episode(cycle, task, manifest_sha,
                                                manifest["model_sha256"], raw, entries)
        prefix = checked_prefix(record, episode)
        _, full_len = sample_lengths(tokenizer, prefix, schema, record)
        main_label = ("SUCCESS_TERMINAL_POSITIVE" if record["status"] == "SUCCESS"
                      else "CURRENT_FRONTIER_CORRECTION")
        if full_len > LIMIT:
            overflow.append({"task_id": task, "label": main_label, "full_tokens": full_len})
        else:
            item = encode_row(tokenizer, record, episode, schema,
                              episode["gold_source_episode_sha256"], cycle=cycle)
            if len(item["input_ids"]) != full_len:
                raise RuntimeError("main target token length drift")
            item.update(source_policy_usage=f"TRAINABLE_COMPLETE_C_D{cycle}",
                        complete_c_source_label=main_label,
                        complete_c_sample_identity=f"{task}:complete-c:cycle{cycle}:{main_label}",
                        complete_c_schema_version="recursive_opd_v3_complete_c_1",
                        train_context_compatible=True, semantic_eligible=True,
                        execution_verified=True, rollout_sha256=rollout_sha)
            encoded.append(item)
            label_counts[main_label] += 1
        candidate = extract_current_continuation(record, episode)
        if candidate["status"] != "CURRENT_CONTINUATION_ELIGIBLE":
            continue
        _, full_len = _chat_lengths(tokenizer, candidate["conversation_prefix"],
                                    candidate["student_action"], schema)
        label = "CURRENT_VERIFIED_CONTINUATION_POSITIVE"
        if full_len > LIMIT:
            overflow.append({"task_id": task, "label": label, "full_tokens": full_len})
            continue
        item = encode_assistant_target(
            tokenizer, task_id=task,
            state_identity=f"{task}:complete-c:cycle{cycle}:continue:{candidate['turn_index']}",
            target_type="first_divergence_tool",
            conversation_prefix=candidate["conversation_prefix"],
            assistant_target=candidate["student_action"], tool_schema=schema)
        validate_mask(item)
        if len(item["input_ids"]) != full_len:
            raise RuntimeError("continuation token length drift")
        item.update(cycle=cycle, pool="train", query_hash=record["query_hash"],
                    source_verdict="CURRENT_VERIFIED",
                    correction_kind=label, complete_c_source_label=label,
                    complete_c_sample_identity=f"{task}:complete-c:cycle{cycle}:continue:{candidate['turn_index']}",
                    complete_c_schema_version="recursive_opd_v3_complete_c_1",
                    train_context_compatible=True, semantic_eligible=True,
                    execution_verified=True, policy_cycle=cycle,
                    policy_sha256=manifest["model_sha256"], rollout_sha256=rollout_sha,
                    rollout_manifest_sha256=manifest_sha, turn_index=candidate["turn_index"],
                    prefix_sha256=candidate["prefix_sha256"],
                    state_before_sha256=candidate["state_before_sha256"],
                    state_after_sha256=candidate["state_after_sha256"],
                    student_action=candidate["student_action"],
                    student_action_sha256=candidate["student_action_sha256"],
                    verification_status=candidate["verification_status"],
                    progress_pointer_before=candidate["progress_pointer_before"],
                    progress_pointer_after=candidate["progress_pointer_after"],
                    terminal_flag=False)
        encoded.append(item)
        label_counts[label] += 1
    identities = [r["complete_c_sample_identity"] for r in encoded]
    task_counts = collections.Counter(r["task_id"] for r in encoded)
    heldout_queries = {records[task]["query_hash"] for task in heldout}
    if (len(identities) != len(set(identities)) or any(count > 2 for count in task_counts.values())
            or set(task_counts) & heldout or {r["query_hash"] for r in encoded} & heldout_queries
            or any(len(r["input_ids"]) > LIMIT for r in encoded)):
        raise RuntimeError("Complete-C natural identity/limit/leakage gate failed")
    for row in encoded:
        validate_mask(row)
    root.mkdir(parents=True)
    path = root / "dataset.jsonl"
    path.write_text("".join(stable(r) + "\n" for r in encoded))
    eos = sum(r["target_type"] == "terminal_stop" for r in encoded)
    value = {"schema_version": "recursive_opd_v3_complete_c_natural_1",
             "status": "READY_CPU_DATA", "cycle": cycle,
             "source_policy_sha256": manifest["model_sha256"],
             "source_rollout_manifest_sha256": manifest_sha,
             "source_metrics_sha256": sha256(RUN / f"cycle{cycle}/metrics/per_task.jsonl"),
             "protocol_sha256": sha256(PROTOCOL),
             "tokenizer_sha256": sha256(MODEL / "tokenizer.json"),
             "dataset_sha256": sha256(path), "sample_count": len(encoded),
             "class_counts": dict(label_counts),
             "current_frontier": label_counts["CURRENT_FRONTIER_CORRECTION"],
             "success_terminal": label_counts["SUCCESS_TERMINAL_POSITIVE"],
             "current_continuation": label_counts["CURRENT_VERIFIED_CONTINUATION_POSITIVE"],
             "tool_targets": len(encoded) - eos, "eos_targets": eos,
             "eos_fraction": eos / len(encoded) if encoded else 0,
             "context_overflow_count": len(overflow), "context_overflow_records": overflow,
             "heldout_training_rows": 0, "train_heldout_task_overlap": 0,
             "train_heldout_query_hash_overlap": 0,
             "gates": {k: "PASS" for k in ("full_rollout", "execution_verifier",
                                          "current_exact_prefix", "current_continuation",
                                          "mask", "context", "split", "query_overlap",
                                          "hashes")}}
    (root / "manifest.json").write_text(stable(value) + "\n")
    return value


def _previous(cycle: int) -> tuple[dict[str, dict], dict, str, str, Path]:
    if cycle == 1:
        base = V1_RUN / "cycle0"
        manifest_path = base / "rollout_manifest.json"
        metrics_path = base / "metrics/per_task.jsonl"
        dataset_path = CPU_RUN / "cycle0/dataset/dataset.jsonl"
        episodes_root = base / "rollouts"
    else:
        base = RUN / f"cycle{cycle-1}"
        manifest_path = base / "rollout_manifest.json"
        metrics_path = base / "metrics/per_task.jsonl"
        dataset_path = base / "natural/dataset.jsonl"
        episodes_root = base / "rollouts"
    rows = {r["task_id"]: r for r in _rows(metrics_path)}
    manifest = json.loads(manifest_path.read_text())
    metrics_summary = json.loads((base / "metrics/summary.json").read_text())
    dataset_manifest = json.loads((dataset_path.parent / "manifest.json").read_text())
    protocol = check()
    if (len(rows) != 2158 or set(rows) != set(protocol["train_task_ids"] + protocol["heldout_task_ids"])
            or metrics_summary.get("per_task_sha256") != sha256(metrics_path)
            or manifest.get("protocol_sha256") != sha256(PROTOCOL)
            or manifest["model_sha256"] != sha256(model_for(cycle-1) / "model.safetensors")
            or dataset_manifest.get("status") != "READY_CPU_DATA"
            or dataset_manifest.get("dataset_sha256") != sha256(dataset_path)
            or dataset_manifest.get("source_policy_sha256") != manifest["model_sha256"]):
        raise RuntimeError("previous-cycle on-policy source drift")
    if cycle > 1:
        audit = json.loads((base / "metrics/audit.json").read_text())
        if (audit.get("status") != "AUDITED" or audit.get("metrics_sha256") != sha256(metrics_path)
                or audit.get("natural_dataset_sha256") != sha256(dataset_path)):
            raise RuntimeError("previous-cycle audit/hash drift")
    return rows, manifest, sha256(metrics_path), sha256(dataset_path), episodes_root


def build_anchors(cycle: int) -> dict:
    if cycle not in range(1, 5):
        raise ValueError("anchor crossed cycle must be 1..4")
    natural = build_natural(cycle)
    current_manifest, current_manifest_sha = checked(cycle)
    previous_records, previous_manifest, previous_metrics_sha, previous_dataset_sha, previous_root = _previous(cycle)
    protocol = check()
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    raw, entries = _sources()
    current_records = {r["task_id"]: r for r in _rows(RUN / f"cycle{cycle}/metrics/per_task.jsonl")}
    natural_rows = _rows(RUN / f"cycle{cycle}/natural/dataset.jsonl")
    positives = {r["task_id"]: r for r in natural_rows
                 if r["complete_c_source_label"] == "CURRENT_VERIFIED_CONTINUATION_POSITIVE"}
    root = RUN / f"cycle{cycle}/anchors"
    if root.exists():
        saved = json.loads((root / "manifest.json").read_text())
        if saved["anchor_dataset_sha256"] != sha256(root / "anchors.jsonl"):
            raise RuntimeError("persisted fresh anchor pool drift")
        return saved
    tokenizer = _tokenizer()
    anchors, overflow = [], []
    statuses, regimes, depth = collections.Counter(), collections.Counter(), collections.Counter()
    raw_eligible = overlap_count = 0
    source_candidates = [previous_records[task] for task in protocol["train_task_ids"]
                         if previous_records[task]["status"] == "FIRST_ERROR"
                         and previous_records[task].get("error_regime") in {"START", "ADVANCE"}]
    for source_record in source_candidates:
        task = source_record["task_id"]
        source_path = previous_root / f"{task}.json"
        source_episode = json.loads(source_path.read_text())
        if (source_episode.get("task_id") != task
                or source_episode.get("cycle") != cycle - 1
                or source_episode.get("model_sha256") != previous_manifest["model_sha256"]
                or source_episode.get("manifest_sha256")
                   != sha256(previous_root.parent / "rollout_manifest.json")
                or source_episode.get("student_error") or not source_episode.get("student")):
            raise RuntimeError(f"previous-cycle anchor source episode drift: {task}")
        current_record = current_records[task]
        current_episode, schema, current_sha = _episode(
            cycle, task, current_manifest_sha, current_manifest["model_sha256"], raw, entries)
        identity = fresh_identity(
            source_record, source_episode, crossed_cycle=cycle,
            metrics_sha256=previous_metrics_sha, rollout_sha256=sha256(source_path),
            rollout_manifest_sha256=sha256(previous_root.parent / "rollout_manifest.json"),
            source_policy_sha256=previous_manifest["model_sha256"],
            source_dataset_sha256=previous_dataset_sha)
        candidate = extract_fresh_anchor(identity, source_episode, current_record, current_episode)
        statuses[candidate["status"]] += 1
        if candidate["status"] != "FRESH_ANCHOR_ELIGIBLE":
            continue
        raw_eligible += 1
        _, full_len = _chat_lengths(tokenizer, candidate["conversation_prefix"],
                                    candidate["student_action"], schema)
        if full_len > LIMIT:
            overflow.append({"task_id": task, "full_tokens": full_len,
                             "source_error_cycle": cycle-1})
            continue
        item = encode_assistant_target(
            tokenizer, task_id=task,
            state_identity=f"{task}:fresh-anchor:{cycle-1}->{cycle}:{identity['gold_pointer']}",
            target_type="first_divergence_tool", conversation_prefix=candidate["conversation_prefix"],
            assistant_target=candidate["student_action"], tool_schema=schema)
        validate_mask(item)
        if len(item["input_ids"]) != full_len:
            raise RuntimeError("anchor token length drift")
        base = {"task_id": task, "pool": "train", "query_hash": current_record["query_hash"],
                "source_error_cycle": cycle-1, "source_policy_hash": previous_manifest["model_sha256"],
                "source_rollout_hash": sha256(source_path),
                "source_rollout_manifest_hash": sha256(previous_root.parent / "rollout_manifest.json"),
                "source_dataset_hash": previous_dataset_sha,
                "source_frontier_type": identity["frontier_regime"],
                "source_frontier_depth": identity["frontier_depth"],
                "source_state_hash": identity["state_sha256"],
                "crossed_cycle": cycle, "intended_replay_cycle": cycle,
                "expires_after_update_cycle": cycle, "anchor_age": 1,
                "current_policy_hash": current_manifest["model_sha256"],
                "current_rollout_hash": current_sha,
                "current_rollout_manifest_hash": current_manifest_sha,
                "current_prefix_hash": candidate["current_prefix_hash"],
                "current_state_before_hash": candidate["current_state_before_hash"],
                "current_state_after_hash": candidate["current_state_after_hash"],
                "current_student_action": candidate["student_action"],
                "current_action_hash": candidate["current_action_hash"],
                "normalized_action_hash": candidate["normalized_action_hash"],
                "execution_verified": True, "progress_before": candidate["progress_before"],
                "progress_after": candidate["progress_after"],
                "matches_historical_gold": candidate["matches_historical_gold"],
                "encoded_length": full_len, "context_compatible": True}
        overlap = exact_overlap(base, positives.get(task))
        item.update(base, anchor_schema_version="recursive_opd_v3_fresh_anchor_1",
                    anchor_label="FRESH_CROSSED_FRONTIER_ANCHOR",
                    anchor_sample_identity=f"{task}:fresh-anchor:{cycle-1}->{cycle}:{identity['gold_pointer']}",
                    source_verdict="FRESH_CROSSING_VERIFIED",
                    target_provenance="CURRENT_VERIFIED_STUDENT_BEHAVIOR",
                    overlaps_current_positive=overlap)
        anchors.append(item)
        overlap_count += int(overlap)
        regimes[identity["frontier_regime"]] += 1
        depth[str(identity["frontier_depth"])] += 1
    anchor_tasks = {row["task_id"] for row in anchors}
    heldout_queries = {current_records[task]["query_hash"] for task in heldout}
    natural_counts = collections.Counter(row["task_id"] for row in natural_rows)
    if (len(anchor_tasks) != len(anchors) or not anchor_tasks <= train
            or {row["query_hash"] for row in anchors} & heldout_queries
            or any(natural_counts[task] + int(task in anchor_tasks) > 3 for task in train)):
        raise RuntimeError("fresh anchor uniqueness/split/semantic-count gate failed")
    root.mkdir(parents=True)
    path = root / "anchors.jsonl"
    path.write_text("".join(stable(r) + "\n" for r in anchors))
    value = {"schema_version": "recursive_opd_v3_fresh_anchor_on_policy_1",
             "status": "ANCHOR_POOL_READY", "source_error_cycle": cycle-1,
             "crossed_cycle": cycle, "anchor_age": 1,
             "source_frontier_candidates": len(source_candidates),
             "raw_eligible": raw_eligible, "context_compatible": len(anchors),
             "context_overflow_count": len(overflow), "context_overflow_records": overflow,
             "overlap_current_positive": overlap_count,
             "anchor_only": len(anchors) - overlap_count,
             "source_regime_counts": dict(regimes), "source_depth_counts": dict(depth),
             "candidate_status_counts": dict(statuses),
             "source_metrics_sha256": previous_metrics_sha,
             "source_policy_sha256": previous_manifest["model_sha256"],
             "source_rollout_manifest_sha256": sha256(previous_root.parent / "rollout_manifest.json"),
             "source_dataset_sha256": previous_dataset_sha,
             "current_policy_sha256": current_manifest["model_sha256"],
             "current_rollout_manifest_sha256": current_manifest_sha,
             "current_natural_dataset_sha256": natural["dataset_sha256"],
             "anchor_dataset_sha256": sha256(path), "heldout_training_rows": 0,
             "gates": {k: "PASS" for k in ("previous_cycle_only", "start_advance_only",
                                          "current_verified_crossing", "exact_student_target",
                                          "one_anchor_per_task", "max_three_semantic_rows",
                                          "mask", "context", "split", "query_overlap", "hashes")}}
    (root / "manifest.json").write_text(stable(value) + "\n")
    return value


def audit(cycle: int) -> dict:
    summary = classify_full(cycle)
    natural = build_natural(cycle)
    anchor = build_anchors(cycle)
    root = RUN / f"cycle{cycle}/metrics"
    value = {"status": "AUDITED", "cycle": cycle, "task_count": 2158,
             "model_sha256": checked(cycle)[0]["model_sha256"],
             "metrics_sha256": sha256(root / "per_task.jsonl"),
             "natural_dataset_sha256": natural["dataset_sha256"],
             "anchor_dataset_sha256": anchor["anchor_dataset_sha256"],
             "rollout_integrity_sha256": sha256(RUN / f"cycle{cycle}/rollout_integrity.json"),
             "heldout_metrics": summary["metrics"]["heldout"],
             "gates": {"full_rollout": "PASS", "verifier": "PASS",
                       "natural": "PASS", "fresh_anchor": "PASS",
                       "task_split": "PASS", "query_overlap": "PASS"}}
    path = root / "audit.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("independent cycle audit drift")
    else:
        path.write_text(stable(value) + "\n")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    classify_full(args.cycle, args.workers)
    print(stable(audit(args.cycle)), flush=True)


if __name__ == "__main__":
    main()
