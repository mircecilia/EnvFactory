"""CPU-only retrospective Complete-C dataset construction and audit."""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import encode_assistant_target
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import OUT, SOURCE
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, PROTOCOL, check
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import LIMIT
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import digest
from repro_1p7b.graph_frontier.recursive_opd_v3.prepare import (
    REPORT, RUN as HISTORICAL_RUN, length_stats, load_episode, load_policy,
    source_dataset,
)
from repro_1p7b.graph_frontier.recursive_opd_v3_complete_c.current_positive import (
    depth_bucket, extract_current_continuation, sample_count_distribution,
)
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

VERSION = "recursive_opd_v3_complete_c_1"
RUN = HISTORICAL_RUN.parent / "recursive_opd_v3_complete_c_run"
TRAINING_STEPS = "UNDECIDED"
EXPOSURE_BUDGET = "UNDECIDED"
LEARNING_RATE = "UNDECIDED"
LABELS = frozenset({
    "CURRENT_FRONTIER_CORRECTION",
    "SUCCESS_TERMINAL_POSITIVE",
    "CURRENT_VERIFIED_CONTINUATION_POSITIVE",
})
PROVENANCE_KEYS = frozenset({
    "task_id", "policy_cycle", "policy_sha256", "rollout_sha256", "turn_index",
    "prefix_sha256", "state_before_sha256", "state_after_sha256", "student_action",
    "student_action_sha256", "verification_status", "progress_pointer_before",
    "progress_pointer_after", "terminal_flag",
})


def _main_label(row: dict) -> str:
    return ("SUCCESS_TERMINAL_POSITIVE"
            if row.get("correction_kind") == "SUCCESS_TERMINAL_POSITIVE"
            else "CURRENT_FRONTIER_CORRECTION")


def _historical_identities(cycle: int) -> tuple[set[tuple], int]:
    if cycle == 0:
        return set(), 0
    root = HISTORICAL_RUN / f"cycle{cycle}/dataset"
    manifest_path, dataset_path = root / "manifest.json", root / "dataset.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "READY_CPU_DATA"
            or manifest.get("dataset_sha256") != sha256(dataset_path)):
        raise RuntimeError(f"historical V3 D{cycle} identity gate failed")
    identities = set()
    for line in dataset_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("v3_source_label") == "CROSSED_FRONTIER_RETENTION":
            identities.add((row["task_id"], row["student_prefix_sha256"],
                            row["student_action_sha256"], row["current_student_step"]))
    expected = manifest.get("crossed_frontier_retention")
    if expected != len(identities):
        raise RuntimeError(f"historical V3 D{cycle} retention identity drift")
    return identities, expected


def _episode_worker(args: tuple) -> dict:
    cycle, task, record, manifest, meta, source_rows, entries = args
    try:
        episode, schema, rollout_sha = load_episode(
            cycle, task, manifest, meta["rollout_manifest_sha256"], source_rows, entries)
        candidate = extract_current_continuation(record, episode)
        return {"task_id": task, "record": record, "candidate": candidate,
                "schema": schema, "rollout_sha256": rollout_sha}
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return {"task_id": task, "record": record,
                "candidate": {"status": "CONTINUATION_UNCERTAIN",
                              "reason": f"{type(exc).__name__}:{str(exc)[:300]}"}}


def _chat_lengths(tokenizer, prefix: list[dict], target: dict, schema: list[dict]) -> tuple[int, int]:
    kwargs = {"tools": schema, "tokenize": True, "enable_thinking": False}
    prompt = tokenizer.apply_chat_template(prefix, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(prefix + [target], add_generation_prompt=False, **kwargs)
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise RuntimeError("continuation target is not an exact assistant suffix")
    return len(prompt), len(full)


def _reason_bucket(record: dict) -> str:
    reason = (record.get("reason") or "").lower()
    status = (record.get("fd_v2_status") or "").upper()
    if "premature model stop" in reason:
        return "PREMATURE_STOP"
    if "WRONG_ARGUMENT" in status:
        return "WRONG_ARGUMENT"
    if "WRONG_TOOL" in status:
        return "WRONG_TOOL"
    if "execution failed" in reason or "EXECUTION_FAILURE" in status:
        return "EXECUTION_ERROR"
    if record.get("error_regime") == "TERMINATE":
        return "OVER_CONTINUATION"
    return "OTHER_FIRST_ERROR"


def _validate_union(rows: list[dict], train: set[str], heldout: set[str],
                    heldout_query_hashes: set[str]) -> dict:
    identities, tasks, train_query_hashes = set(), collections.Counter(), set()
    for row in rows:
        identity = row.get("complete_c_sample_identity")
        if (not identity or identity in identities or row.get("task_id") not in train
                or row.get("task_id") in heldout or row.get("pool") != "train"
                or row.get("complete_c_source_label") not in LABELS
                or row.get("source_verdict") == "UNCERTAIN"
                or len(row.get("input_ids", [])) > LIMIT):
            raise RuntimeError("Complete-C identity/split/source/context gate failed")
        if row["complete_c_source_label"] == "CURRENT_VERIFIED_CONTINUATION_POSITIVE":
            if (not PROVENANCE_KEYS <= row.keys() or row.get("terminal_flag") is not False
                    or row.get("verification_status")
                    != "CURRENT_EXACT_EXECUTION_VERIFIED_PROGRESS"):
                raise RuntimeError("Complete-C continuation provenance gate failed")
        validate_mask(row)
        identities.add(identity)
        tasks[row["task_id"]] += 1
        train_query_hashes.add(row["query_hash"])
    if any(count > 2 for count in tasks.values()):
        raise RuntimeError("Complete-C task contributes more than two rows")
    if train_query_hashes & heldout_query_hashes:
        raise RuntimeError("Complete-C train/heldout query hash overlap")
    return {"unique_samples": len(identities), "unique_tasks": len(tasks)}


def _first_error_analysis(records: dict[str, dict], continuation_tasks: set[str],
                          train: set[str]) -> dict:
    regimes = {}
    reasons = {}
    for task in train:
        row = records[task]
        if row.get("status") != "FIRST_ERROR":
            continue
        for target, key in ((regimes, row.get("error_regime") or "UNKNOWN"),
                            (reasons, _reason_bucket(row))):
            value = target.setdefault(key, {"with_continuation": 0, "without_continuation": 0})
            value["with_continuation" if task in continuation_tasks
                  else "without_continuation"] += 1
    return {"by_regime": regimes, "by_reason": reasons}


def _premature_regression(records: dict[int, dict[str, dict]],
                          d3_continuations: set[str], historical_d3: set[str],
                          train: set[str]) -> dict:
    tasks = set()
    for task in train:
        before, after = records[3][task], records[4][task]
        if (before.get("status") != "UNCERTAIN"
                and after.get("status") == "FIRST_ERROR"
                and after.get("error_regime") in {"START", "ADVANCE"}
                and after.get("termination_reason") == "MODEL_FINAL"
                and isinstance(before.get("frontier_depth"), int)
                and isinstance(after.get("frontier_depth"), int)
                and before["frontier_depth"] > after["frontier_depth"]):
            tasks.add(task)
    old_regression = json.loads((REPORT / "recursive_opd_v3_cpu_prep.json").read_text())[
        "pi3_crossed_to_pi4_regression"]
    old_tasks = {row["task_id"] for row in old_regression.get("tasks", [])
                 if row.get("pi4_termination_reason") == "MODEL_FINAL"}
    return {
        "definition": "pi3 deeper than pi4; pi4 FIRST_ERROR START/ADVANCE MODEL_FINAL",
        "strict_premature_regression_tasks": len(tasks),
        "complete_c_covered": len(tasks & d3_continuations),
        "complete_c_coverage": len(tasks & d3_continuations) / len(tasks) if tasks else None,
        "historical_matched_in_strict_set": len(tasks & historical_d3),
        "historical_18_subset_total": len(old_tasks),
        "historical_18_subset_complete_c_covered": len(old_tasks & d3_continuations),
    }


def build(*, workers: int = 8) -> dict:
    if RUN.exists():
        raise RuntimeError(f"refusing to overwrite Complete-C runtime artifacts: {RUN}")
    if workers < 1 or workers > 10:
        raise ValueError("Complete-C CPU preparation permits 1..10 workers")
    protocol = check()
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    if len(train) != 1726 or len(heldout) != 432 or train & heldout:
        raise RuntimeError("frozen task split invalid")
    source_rows = json.loads(SOURCE.read_text())
    entries = {row["audit_id"]: row for row in json.loads(
        (OUT / "opd_source_manifest.json").read_text())["rows"]}
    policy_rows, policy_manifests, policy_meta, records = {}, {}, {}, {}
    for cycle in range(5):
        rows, manifest, meta = load_policy(cycle, protocol)
        policy_rows[cycle], policy_manifests[cycle], policy_meta[cycle] = rows, manifest, meta
        records[cycle] = {row["task_id"]: row for row in rows}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    tokenizer_sha = sha256(MODEL / "tokenizer.json")
    if tokenizer.eos_token_id != 151645 or tokenizer_sha != protocol["tokenizer_sha256"]:
        raise RuntimeError("Complete-C tokenizer/EOS identity drift")

    RUN.mkdir(parents=True)
    natural, cycle_continuations, historical_sets = {}, {}, {}
    for cycle in range(5):
        historical_identities, historical_count = _historical_identities(cycle)
        historical_sets[cycle] = {identity[0] for identity in historical_identities}
        jobs = [(cycle, task, records[cycle][task], policy_manifests[cycle], policy_meta[cycle],
                 source_rows, entries) for task in protocol["train_task_ids"]]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="complete-c") as pool:
            outcomes = list(pool.map(_episode_worker, jobs))
        outcome_by_task = {row["task_id"]: row for row in outcomes}
        main_rows, main_manifest, main_path = source_dataset(cycle)
        output = RUN / f"cycle{cycle}/dataset"
        output.mkdir(parents=True)
        dataset_path, audit_path = output / "dataset.jsonl", output / "continuation_audit.jsonl"
        union, overflow, encoding_failures = [], [], []
        semantic_lengths: dict[str, list[int]] = collections.defaultdict(list)
        main_tasks = set()
        for row in main_rows:
            item = dict(row)
            label = _main_label(item)
            item.update(
                complete_c_source_label=label,
                complete_c_sample_identity=f"{item['task_id']}:complete-c:cycle{cycle}:{label}",
                complete_c_schema_version=VERSION,
            )
            union.append(item)
            main_tasks.add(item["task_id"])
            semantic_lengths[label].append(len(item["input_ids"]))
        for item in main_manifest.get("context_overflow_records", []):
            label = ("SUCCESS_TERMINAL_POSITIVE"
                     if item.get("kind") == "SUCCESS_TERMINAL_POSITIVE"
                     else "CURRENT_FRONTIER_CORRECTION")
            semantic_lengths[label].append(item["full_tokens"])

        continuation_tasks, candidate_tasks = set(), set()
        historical_matched = 0
        candidate_status = collections.Counter()
        depth = collections.Counter()
        distance = collections.Counter()
        for task in protocol["train_task_ids"]:
            outcome = outcome_by_task[task]
            candidate = outcome["candidate"]
            candidate_status[candidate.get("status", "MISSING_STATUS")] += 1
            if candidate.get("status") != "CURRENT_CONTINUATION_ELIGIBLE":
                continue
            candidate_tasks.add(task)
            # Historical V3 stored a normalized single-call action hash while
            # Complete-C preserves the whole assistant message.  Compare the
            # common normalized representation; verified parallel actions are
            # necessarily new because historical V3 rejected them.
            historical_action_hash = (digest(candidate["student_action_calls"][0])
                                      if len(candidate["student_action_calls"]) == 1 else None)
            key = (task, candidate["prefix_sha256"], historical_action_hash,
                   candidate["turn_index"])
            matches_historical = key in historical_identities
            historical_matched += int(matches_historical)
            prompt_len, full_len = _chat_lengths(
                tokenizer, candidate["conversation_prefix"], candidate["student_action"],
                outcome["schema"])
            semantic_lengths["CURRENT_VERIFIED_CONTINUATION_POSITIVE"].append(full_len)
            provenance = {
                "task_id": task, "policy_cycle": cycle,
                "policy_sha256": policy_meta[cycle]["source_policy_sha256"],
                "rollout_sha256": outcome["rollout_sha256"],
                "rollout_manifest_sha256": policy_meta[cycle]["rollout_manifest_sha256"],
                "turn_index": candidate["turn_index"],
                "positive_source_turn": candidate["positive_source_turn"],
                "prefix_sha256": candidate["prefix_sha256"],
                "state_before_sha256": candidate["state_before_sha256"],
                "state_after_sha256": candidate["state_after_sha256"],
                "student_action": candidate["student_action"],
                "student_action_sha256": candidate["student_action_sha256"],
                "verification_status": candidate["verification_status"],
                "progress_pointer_before": candidate["progress_pointer_before"],
                "progress_pointer_after": candidate["progress_pointer_after"],
                "distance_to_main_frontier": candidate["distance_to_main_frontier"],
                "terminal_flag": False,
                "parallel_call_count": candidate["parallel_call_count"],
                "matches_historical_frontier": matches_historical,
                "target_provenance": "CURRENT_VERIFIED_STUDENT_BEHAVIOR",
            }
            if not PROVENANCE_KEYS <= provenance.keys():
                raise RuntimeError("internal Complete-C provenance omission")
            if full_len > LIMIT:
                overflow.append({"reason": "CONTEXT_OVERFLOW", "prompt_tokens": prompt_len,
                                 "full_tokens": full_len, **provenance})
                continue
            try:
                encoded = encode_assistant_target(
                    tokenizer, task_id=task,
                    state_identity=(f"{task}:complete-c:cycle{cycle}:continue:"
                                    f"{candidate['turn_index']}"),
                    target_type="first_divergence_tool",
                    conversation_prefix=candidate["conversation_prefix"],
                    assistant_target=candidate["student_action"],
                    tool_schema=outcome["schema"],
                )
                validate_mask(encoded)
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                encoding_failures.append({"task_id": task, "reason": str(exc)[:300]})
                continue
            if len(encoded["input_ids"]) != full_len:
                raise RuntimeError("Complete-C token length differs from independent audit")
            query_hash = records[cycle][task]["query_hash"]
            encoded.update(
                cycle=cycle, pool="train", query_hash=query_hash,
                source_verdict="CURRENT_VERIFIED",
                correction_kind="CURRENT_VERIFIED_CONTINUATION_POSITIVE",
                complete_c_source_label="CURRENT_VERIFIED_CONTINUATION_POSITIVE",
                complete_c_sample_identity=f"{task}:complete-c:cycle{cycle}:continue:{candidate['turn_index']}",
                complete_c_schema_version=VERSION, train_context_compatible=True,
                semantic_eligible=True, execution_verified=True, **provenance,
            )
            union.append(encoded)
            continuation_tasks.add(task)
            depth[depth_bucket(candidate["progress_pointer_before"])] += 1
            distance[str(candidate["distance_to_main_frontier"])] += 1

        heldout_query_hashes = {records[cycle][task]["query_hash"] for task in heldout}
        _validate_union(union, train, heldout, heldout_query_hashes)
        distribution = sample_count_distribution(main_tasks, continuation_tasks, train)
        with dataset_path.open("w") as stream:
            for row in union:
                stream.write(stable(row) + "\n")
        with audit_path.open("w") as stream:
            for outcome in outcomes:
                candidate = dict(outcome["candidate"])
                candidate.pop("conversation_prefix", None)
                candidate.pop("student_action", None)
                candidate.pop("state_before", None)
                candidate.pop("state_after", None)
                stream.write(stable({"task_id": outcome["task_id"],
                                     "record_status": outcome["record"].get("status"),
                                     "candidate": candidate}) + "\n")
        counts = collections.Counter(row["complete_c_source_label"] for row in union)
        tool_targets = main_manifest["tool_targets"] + counts[
            "CURRENT_VERIFIED_CONTINUATION_POSITIVE"]
        eos_targets = main_manifest["eos_targets"]
        success_tasks = {task for task in train if records[cycle][task].get("status") == "SUCCESS"}
        terminal_tasks = {row["task_id"] for row in union
                          if row["complete_c_source_label"] == "SUCCESS_TERMINAL_POSITIVE"}
        context = {label: length_stats(semantic_lengths[label]) for label in sorted(LABELS)}
        context["ALL"] = length_stats(
            [value for values in semantic_lengths.values() for value in values])
        gates = {name: "PASS" for name in (
            "correction_builder", "terminal_builder", "current_continuation_builder",
            "execution_verifier", "provenance", "student_prefix", "student_action_target",
            "parallel_handling", "natural_union", "max_two_rows_per_task",
            "heldout_task_leakage", "heldout_query_leakage", "uncertain_exclusion",
            "context_compatibility", "token_masks", "manifest_hashes",
        )}
        if encoding_failures:
            gates["current_continuation_builder"] = "FAIL"
        manifest = {
            "schema_version": VERSION,
            "status": "READY_CPU_DATA" if not encoding_failures else "NOT_READY",
            "cycle": cycle,
            "training_steps": TRAINING_STEPS,
            "exposure_budget": EXPOSURE_BUDGET,
            "learning_rate": LEARNING_RATE,
            "source_v2_dataset_sha256": sha256(main_path),
            "source_v2_manifest_sha256": sha256(main_path.parent / "manifest.json"),
            "source_rollout_manifest_sha256": policy_meta[cycle]["rollout_manifest_sha256"],
            "source_policy_sha256": policy_meta[cycle]["source_policy_sha256"],
            "protocol_sha256": sha256(PROTOCOL), "tokenizer_sha256": tokenizer_sha,
            "dataset_sha256": sha256(dataset_path), "continuation_audit_sha256": sha256(audit_path),
            "sample_count": len(union), "class_counts": dict(counts),
            "current_frontier": counts["CURRENT_FRONTIER_CORRECTION"],
            "success_terminal": counts["SUCCESS_TERMINAL_POSITIVE"],
            "current_continuation": counts["CURRENT_VERIFIED_CONTINUATION_POSITIVE"],
            "current_continuation_candidates_before_context": len(candidate_tasks),
            "tool_targets": tool_targets, "eos_targets": eos_targets,
            "tool_fraction": tool_targets / len(union) if union else 0,
            "eos_fraction": eos_targets / len(union) if union else 0,
            "tool_eos_ratio": tool_targets / eos_targets if eos_targets else None,
            "v2_sample_count": main_manifest["sample_count"],
            "v2_tool_targets": main_manifest["tool_targets"],
            "v2_eos_targets": eos_targets,
            "v2_eos_fraction": eos_targets / main_manifest["sample_count"],
            "historical_only_continuation": historical_count,
            "historical_frontier_matched": historical_matched,
            "nonhistorical_current_positive": len(candidate_tasks) - historical_matched,
            "continuation_depth_distribution": dict(depth),
            "distance_to_main_frontier_distribution": dict(distance),
            "candidate_status": dict(candidate_status),
            "sample_count_per_task": distribution,
            "success_pair_analysis": {
                "success_tasks": len(success_tasks),
                "terminal_positive": len(success_tasks & terminal_tasks),
                "continuation_positive": len(success_tasks & continuation_tasks),
                "terminal_and_continuation": len(success_tasks & terminal_tasks & continuation_tasks),
            },
            "first_error_analysis": _first_error_analysis(records[cycle], continuation_tasks, train),
            "context_overflow_count": main_manifest.get("context_overflow_count", 0) + len(overflow),
            "main_context_overflow_count": main_manifest.get("context_overflow_count", 0),
            "continuation_context_overflow_count": len(overflow),
            "continuation_context_overflow_records": overflow,
            "continuation_encoding_failures": encoding_failures,
            "context_lengths": context, "heldout_training_rows": 0,
            "train_heldout_task_overlap": 0, "train_heldout_query_hash_overlap": 0,
            "uncertain_training_rows": 0, "gates": gates,
        }
        (output / "manifest.json").write_text(stable(manifest) + "\n")
        natural[str(cycle)] = manifest
        cycle_continuations[cycle] = continuation_tasks
        print(stable({"cycle": cycle, "current_continuation": len(continuation_tasks),
                      "historical": historical_count, "total": len(union),
                      "tool": tool_targets, "eos": eos_targets}), flush=True)

    regression = _premature_regression(
        records, cycle_continuations[3], historical_sets[3], train)
    report = {
        "schema_version": VERSION, "status": "BUILT_PENDING_TESTS", "verdict": "NOT_READY",
        "gpu_used": "NO", "training_steps": TRAINING_STEPS,
        "exposure_budget": EXPOSURE_BUDGET, "learning_rate": LEARNING_RATE,
        "protocol_sha256": sha256(PROTOCOL), "tokenizer_sha256": tokenizer_sha,
        "natural_pools": natural, "premature_regression_coverage": regression,
        "audit_gates": {"data": "PASS" if all(
            pool["status"] == "READY_CPU_DATA"
            and all(value == "PASS" for value in pool["gates"].values())
            for pool in natural.values()) else "FAIL"},
        "not_run": ["V3 GPU training", "V3 fresh rollout", "GRPO",
                    "Static retraining", "BFCL"],
    }
    (RUN / "cpu_prep_draft.json").write_text(stable(report) + "\n")
    return report


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _markdown(report: dict) -> str:
    lines = [
        f"VERDICT = {report['verdict']}", "", "GPU used = NO", "",
        "# Recursive OPD V3 Complete-C CPU preparation", "",
        "All pool counts are train-split, context-compatible rows. No fresh rollout, inference, replay, or training was run.",
        "", "## Natural Complete-C pools", "",
        "| D | CURRENT_FRONTIER | SUCCESS_TERMINAL | CURRENT_CONTINUATION | TOTAL | Tool | EOS | Tool/EOS | EOS fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cycle, value in report["natural_pools"].items():
        lines.append(
            f"| {cycle} | {value['current_frontier']} | {value['success_terminal']} | "
            f"{value['current_continuation']} | {value['sample_count']} | "
            f"{value['tool_targets']} | {value['eos_targets']} | "
            f"{value['tool_eos_ratio']:.4f} | {_pct(value['eos_fraction'])} |")
    lines += ["", "## Historical-only versus Complete-C", "",
              "| D | historical-only | Complete-C | matched historical | new non-historical |",
              "|---:|---:|---:|---:|---:|"]
    for cycle, value in report["natural_pools"].items():
        lines.append(f"| {cycle} | {value['historical_only_continuation']} | "
                     f"{value['current_continuation']} | {value['historical_frontier_matched']} | "
                     f"{value['nonhistorical_current_positive']} |")
    lines += ["", "## Continuation distributions", ""]
    for cycle, value in report["natural_pools"].items():
        lines.append(f"- D{cycle}: depth={value['continuation_depth_distribution']}; "
                     f"distance={value['distance_to_main_frontier_distribution']}; "
                     f"0/1/2 rows={value['sample_count_per_task']}; "
                     f"context overflow={value['context_overflow_count']}.")
    lines += ["", "## Boundary coverage", ""]
    for cycle, value in report["natural_pools"].items():
        lines.append(f"- D{cycle} SUCCESS: {value['success_pair_analysis']}.")
    coverage = report["premature_regression_coverage"]
    lines += ["", "## π3→π4 premature-stop regression", "",
              f"Definition: {coverage['definition']}.",
              f"Complete-C coverage: {coverage['complete_c_covered']}/"
              f"{coverage['strict_premature_regression_tasks']}.",
              f"Historical 18-task subset coverage: "
              f"{coverage['historical_18_subset_complete_c_covered']}/"
              f"{coverage['historical_18_subset_total']}.", "",
              "## Audit", "",
              f"CPU tests: {report['cpu_tests']['summary']}", "",
              "TRAINING STEPS = UNDECIDED", "EXPOSURE BUDGET = UNDECIDED",
              "LEARNING RATE = UNDECIDED", ""]
    return "\n".join(lines)


def finalize() -> dict:
    draft = RUN / "cpu_prep_draft.json"
    if not draft.exists():
        raise RuntimeError("run build before finalize")
    report = json.loads(draft.read_text())
    tests = [
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v1.py",
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v2.py",
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v3.py",
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v3_complete_c.py",
    ]
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(command, text=True, capture_output=True,
                               cwd=HISTORICAL_RUN.parents[2])
    log = RUN / "cpu_tests.log"
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Complete-C/V2/V3 CPU tests failed; see {log}")
    if (report.get("audit_gates", {}).get("data") != "PASS"
            or any(pool.get("status") != "READY_CPU_DATA"
                   or any(value != "PASS" for value in pool.get("gates", {}).values())
                   for pool in report["natural_pools"].values())):
        raise RuntimeError("Complete-C data/provenance gate is not fully PASS")
    summary = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "PASS"
    report.update(status="AUDITED", verdict="READY_FOR_GPU_TRAINING",
                  cpu_tests={"status": "PASS", "summary": summary,
                             "command": command, "log_sha256": sha256(log)})
    REPORT.mkdir(parents=True, exist_ok=True)
    json_path = REPORT / "recursive_opd_v3_complete_c_cpu_prep.json"
    md_path = REPORT / "recursive_opd_v3_complete_c_cpu_prep.md"
    if json_path.exists() or md_path.exists():
        raise RuntimeError("refusing to overwrite Complete-C CPU report")
    json_path.write_text(stable(report) + "\n")
    md_path.write_text(_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "finalize"))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    answer = build(workers=args.workers) if args.command == "build" else finalize()
    print(stable({"status": answer["status"], "verdict": answer["verdict"],
                  "gpu_used": answer["gpu_used"]}), flush=True)


if __name__ == "__main__":
    main()
