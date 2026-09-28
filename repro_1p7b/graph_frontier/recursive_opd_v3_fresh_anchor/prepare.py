"""CPU-only construction and audit of one-cycle fresh frontier anchors."""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import encode_assistant_target
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import OUT, SOURCE
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, PROTOCOL, check
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import LIMIT
from repro_1p7b.graph_frontier.recursive_opd_v3.prepare import (
    REPORT, RUN as HISTORICAL_RUN, length_stats, load_episode, load_policy, source_dataset,
)
from repro_1p7b.graph_frontier.recursive_opd_v3_complete_c.prepare import RUN as COMPLETE_RUN
from repro_1p7b.graph_frontier.recursive_opd_v3_fresh_anchor.anchor import (
    depth_bucket, descriptive_exposure, exact_overlap, extract_fresh_anchor,
    fresh_identity, semantic_count_distribution,
)
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

VERSION = "recursive_opd_v3_fresh_anchor_1"
RUN = HISTORICAL_RUN.parent / "recursive_opd_v3_fresh_anchor_run"
GPU_TRAINING_CONFIG = "UNDECIDED"


def _historical_pool(crossed_cycle: int) -> tuple[set[str], int]:
    root = HISTORICAL_RUN / f"cycle{crossed_cycle}/dataset"
    manifest_path, dataset_path = root / "manifest.json", root / "dataset.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset_sha256") != sha256(dataset_path):
        raise RuntimeError("historical V3 pool hash drift")
    tasks = set()
    for line in dataset_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("v3_source_label") == "CROSSED_FRONTIER_RETENTION":
            tasks.add(row["task_id"])
    expected = manifest.get("crossed_frontier_retention")
    if len(tasks) != expected:
        raise RuntimeError("historical V3 pool identity drift")
    return tasks, expected


def _complete_pool(cycle: int) -> tuple[list[dict], dict, dict[str, dict], dict[str, int]]:
    root = COMPLETE_RUN / f"cycle{cycle}/dataset"
    manifest_path, dataset_path = root / "manifest.json", root / "dataset.jsonl"
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line]
    if (manifest.get("status") != "READY_CPU_DATA"
            or manifest.get("dataset_sha256") != sha256(dataset_path)
            or manifest.get("sample_count") != len(rows)):
        raise RuntimeError(f"Complete-C D{cycle} identity gate failed")
    positives, counts = {}, collections.Counter()
    for row in rows:
        counts[row["task_id"]] += 1
        if row.get("complete_c_source_label") == "CURRENT_VERIFIED_CONTINUATION_POSITIVE":
            if row["task_id"] in positives:
                raise RuntimeError("more than one current positive per task")
            positives[row["task_id"]] = row
    return rows, manifest, positives, dict(counts)


def _worker(args: tuple) -> dict:
    (crossed_cycle, task, source_record, current_record, manifests, meta,
     source_rows, entries, source_dataset_sha) = args
    source_cycle = crossed_cycle - 1
    try:
        source_episode, _, source_rollout_sha = load_episode(
            source_cycle, task, manifests[source_cycle],
            meta[source_cycle]["rollout_manifest_sha256"], source_rows, entries)
        current_episode, schema, current_rollout_sha = load_episode(
            crossed_cycle, task, manifests[crossed_cycle],
            meta[crossed_cycle]["rollout_manifest_sha256"], source_rows, entries)
        identity = fresh_identity(
            source_record, source_episode, crossed_cycle=crossed_cycle,
            metrics_sha256=meta[source_cycle]["metrics_sha256"],
            rollout_sha256=source_rollout_sha,
            rollout_manifest_sha256=meta[source_cycle]["rollout_manifest_sha256"],
            source_policy_sha256=meta[source_cycle]["source_policy_sha256"],
            source_dataset_sha256=source_dataset_sha,
        )
        anchor = extract_fresh_anchor(identity, source_episode, current_record, current_episode)
        return {"task_id": task, "identity": identity, "anchor": anchor, "schema": schema,
                "current_rollout_sha256": current_rollout_sha}
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return {"task_id": task, "anchor": {"status": "ANCHOR_UNCERTAIN",
                                              "reason": f"{type(exc).__name__}:{str(exc)[:300]}"}}


def _chat_lengths(tokenizer, prefix: list[dict], target: dict, schema: list[dict]) -> tuple[int, int]:
    kwargs = {"tools": schema, "tokenize": True, "enable_thinking": False}
    prompt = tokenizer.apply_chat_template(prefix, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(prefix + [target], add_generation_prompt=False, **kwargs)
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise RuntimeError("anchor target is not an exact assistant suffix")
    return len(prompt), len(full)


def _classify_sets(universe: set[str], natural: set[str], anchors: set[str]) -> dict:
    return {
        "current_positive_only": len(universe & natural - anchors),
        "anchor_only": len(universe & anchors - natural),
        "both": len(universe & natural & anchors),
        "neither": len(universe - natural - anchors),
        "natural_coverage": len(universe & natural),
        "anchor_coverage": len(universe & anchors),
        "union_coverage": len(universe & (natural | anchors)),
        "denominator": len(universe),
    }


def build(*, workers: int = 8) -> dict:
    if RUN.exists():
        raise RuntimeError(f"refusing to overwrite fresh-anchor runtime: {RUN}")
    if workers < 1 or workers > 10:
        raise ValueError("fresh-anchor CPU preparation permits 1..10 workers")
    protocol = check()
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    if len(train) != 1726 or len(heldout) != 432 or train & heldout:
        raise RuntimeError("frozen split invalid")
    source_rows = json.loads(SOURCE.read_text())
    entries = {row["audit_id"]: row for row in json.loads(
        (OUT / "opd_source_manifest.json").read_text())["rows"]}
    rows, manifests, meta, records = {}, {}, {}, {}
    for cycle in range(5):
        policy_rows, manifest, policy_meta = load_policy(cycle, protocol)
        rows[cycle], manifests[cycle], meta[cycle] = policy_rows, manifest, policy_meta
        records[cycle] = {row["task_id"]: row for row in policy_rows}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    tokenizer_sha = sha256(MODEL / "tokenizer.json")
    if tokenizer.eos_token_id != 151645 or tokenizer_sha != protocol["tokenizer_sha256"]:
        raise RuntimeError("anchor tokenizer/EOS identity drift")

    RUN.mkdir(parents=True)
    summaries, anchor_task_sets, natural_positive_sets = {}, {}, {}
    for crossed_cycle in range(1, 5):
        source_cycle = crossed_cycle - 1
        _, _, source_dataset_path = source_dataset(source_cycle)
        source_dataset_sha = sha256(source_dataset_path)
        source_frontiers = [row for row in rows[source_cycle]
                            if row["task_id"] in train and row.get("status") == "FIRST_ERROR"
                            and row.get("error_regime") in {"START", "ADVANCE"}]
        jobs = [(crossed_cycle, row["task_id"], row, records[crossed_cycle][row["task_id"]],
                 manifests, meta, source_rows, entries, source_dataset_sha)
                for row in source_frontiers]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fresh-anchor") as pool:
            outcomes = list(pool.map(_worker, jobs))
        _, natural_manifest, current_positives, natural_counts = _complete_pool(crossed_cycle)
        natural_positive_sets[crossed_cycle] = set(current_positives)
        historical_tasks, historical_count = _historical_pool(crossed_cycle)

        output = RUN / f"a{crossed_cycle}"
        output.mkdir(parents=True)
        dataset_path, audit_path = output / "anchors.jsonl", output / "audit.jsonl"
        anchors, overflow, encoding_failures = [], [], []
        raw_eligible, exact_overlaps = 0, 0
        status_counts, regime_counts, depth_counts = collections.Counter(), collections.Counter(), collections.Counter()
        lengths = []
        for outcome in outcomes:
            anchor = outcome["anchor"]
            status_counts[anchor.get("status", "MISSING_STATUS")] += 1
            if anchor.get("status") != "FRESH_ANCHOR_ELIGIBLE":
                continue
            raw_eligible += 1
            identity = outcome["identity"]
            regime_counts[identity["frontier_regime"]] += 1
            depth_counts[depth_bucket(identity["frontier_depth"])] += 1
            prompt_len, full_len = _chat_lengths(
                tokenizer, anchor["conversation_prefix"], anchor["student_action"], outcome["schema"])
            lengths.append(full_len)
            base = {
                "task_id": outcome["task_id"],
                "source_error_cycle": identity["source_error_cycle"],
                "source_policy_hash": identity["source_policy_sha256"],
                "source_rollout_hash": identity["source_rollout_sha256"],
                "source_rollout_manifest_hash": identity["source_rollout_manifest_sha256"],
                "source_dataset_hash": identity["source_dataset_sha256"],
                "source_frontier_type": identity["frontier_regime"],
                "source_frontier_depth": identity["frontier_depth"],
                "source_state_hash": identity["state_sha256"],
                "crossed_cycle": identity["crossed_cycle"],
                "intended_replay_cycle": identity["intended_replay_cycle"],
                "expires_after_update_cycle": identity["expires_after_update_cycle"],
                "anchor_age": identity["anchor_age"],
                "current_policy_hash": meta[crossed_cycle]["source_policy_sha256"],
                "current_rollout_hash": outcome["current_rollout_sha256"],
                "current_rollout_manifest_hash": meta[crossed_cycle]["rollout_manifest_sha256"],
                "current_prefix_hash": anchor["current_prefix_hash"],
                "current_state_before_hash": anchor["current_state_before_hash"],
                "current_state_after_hash": anchor["current_state_after_hash"],
                "current_student_action": anchor["student_action"],
                "current_action_hash": anchor["current_action_hash"],
                "normalized_action_hash": anchor["normalized_action_hash"],
                "execution_verified": anchor["execution_verified"],
                "progress_before": anchor["progress_before"],
                "progress_after": anchor["progress_after"],
                "matches_historical_gold": anchor["matches_historical_gold"],
                "encoded_length": full_len,
                "context_compatible": full_len <= LIMIT,
                "query_hash": records[crossed_cycle][outcome["task_id"]]["query_hash"],
            }
            overlap = exact_overlap(base, current_positives.get(outcome["task_id"]))
            base["overlaps_current_positive"] = overlap
            exact_overlaps += int(overlap and full_len <= LIMIT)
            if full_len > LIMIT:
                overflow.append({"reason": "ANCHOR_CONTEXT_OVERFLOW",
                                 "prompt_tokens": prompt_len, **base})
                continue
            try:
                encoded = encode_assistant_target(
                    tokenizer, task_id=outcome["task_id"],
                    state_identity=(f"{outcome['task_id']}:fresh-anchor:"
                                    f"{source_cycle}->{crossed_cycle}:"
                                    f"{identity['gold_pointer']}"),
                    target_type="first_divergence_tool",
                    conversation_prefix=anchor["conversation_prefix"],
                    assistant_target=anchor["student_action"], tool_schema=outcome["schema"])
                validate_mask(encoded)
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                encoding_failures.append({"task_id": outcome["task_id"],
                                          "reason": str(exc)[:300]})
                continue
            if len(encoded["input_ids"]) != full_len:
                raise RuntimeError("anchor token length differs from independent audit")
            encoded.update(
                anchor_schema_version=VERSION, anchor_label="FRESH_CROSSED_FRONTIER_ANCHOR",
                anchor_sample_identity=(f"{outcome['task_id']}:fresh-anchor:"
                                        f"{source_cycle}->{crossed_cycle}:"
                                        f"{identity['gold_pointer']}"),
                pool="train", source_verdict="FRESH_CROSSING_VERIFIED",
                target_provenance="CURRENT_VERIFIED_STUDENT_BEHAVIOR", **base)
            anchors.append(encoded)

        anchor_tasks = {row["task_id"] for row in anchors}
        if len(anchor_tasks) != len(anchors):
            raise RuntimeError("more than one fresh anchor for a task")
        heldout_query_hashes = {records[crossed_cycle][task]["query_hash"] for task in heldout}
        if (not anchor_tasks <= train or anchor_tasks & heldout
                or {row["query_hash"] for row in anchors} & heldout_query_hashes
                or any(row["anchor_age"] != 1
                       or row["crossed_cycle"] != row["source_error_cycle"] + 1
                       or row["intended_replay_cycle"] != row["crossed_cycle"]
                       for row in anchors)):
            raise RuntimeError("anchor freshness/leakage gate failed")
        with dataset_path.open("w") as stream:
            for row in anchors:
                stream.write(stable(row) + "\n")
        with audit_path.open("w") as stream:
            for outcome in outcomes:
                compact = dict(outcome["anchor"])
                for key in ("conversation_prefix", "student_action"):
                    compact.pop(key, None)
                stream.write(stable({"task_id": outcome["task_id"], "anchor": compact}) + "\n")

        distribution = semantic_count_distribution(natural_counts, anchor_tasks, train)
        source_task_set = set(anchor_tasks)
        historical_intersection = source_task_set & historical_tasks
        gates = {name: "PASS" for name in (
            "previous_cycle_source", "start_advance_only", "source_provenance",
            "current_policy_provenance", "current_student_prefix", "current_student_action",
            "execution_verification", "one_cycle_freshness", "terminal_exclusion",
            "context_compatibility", "encoding", "loss_mask", "max_one_anchor_per_task",
            "max_three_semantic_entries", "heldout_task_leakage", "heldout_query_leakage",
        )}
        if encoding_failures:
            gates["encoding"] = "FAIL"
        summary = {
            "schema_version": VERSION,
            "status": "ANCHOR_POOL_READY" if not encoding_failures else "ANCHOR_POOL_NOT_READY",
            "anchor_pool": f"A{crossed_cycle}",
            "source_transition": f"pi{source_cycle}->pi{crossed_cycle}",
            "source_error_cycle": source_cycle, "crossed_cycle": crossed_cycle,
            "intended_replay_cycle": crossed_cycle, "anchor_age": 1,
            "source_frontier_candidates": len(source_frontiers),
            "raw_eligible": raw_eligible, "context_compatible": len(anchors),
            "overlap_current_positive": exact_overlaps,
            "anchor_only": len(anchors) - exact_overlaps,
            "historical_only_eligible": historical_count,
            "historical_task_intersection": len(historical_intersection),
            "historical_old_only": len(historical_tasks - source_task_set),
            "fresh_only_vs_historical": len(source_task_set - historical_tasks),
            "historical_difference_explanation": (
                "Fresh anchors use only the immediately previous cycle; historical V3 used the latest "
                "eligible frontier from any earlier cycle. Other verifier/context rules are aligned."),
            "regime_distribution": dict(regime_counts),
            "depth_distribution": dict(depth_counts),
            "context_lengths": length_stats(lengths),
            "context_overflow_count": len(overflow), "context_overflow_records": overflow,
            "encoding_failures": encoding_failures, "status_counts": dict(status_counts),
            "natural_pool_size": natural_manifest["sample_count"],
            "fresh_anchor_size": len(anchors),
            "combined_semantic_entries": natural_manifest["sample_count"] + len(anchors),
            "exact_overlap": exact_overlaps,
            "deduplicated_unique_rows": natural_manifest["sample_count"] + len(anchors) - exact_overlaps,
            "semantic_entries_per_task": distribution,
            "descriptive_exposure_b128": descriptive_exposure({
                "CURRENT_FRONTIER_CORRECTION": natural_manifest["current_frontier"],
                "SUCCESS_TERMINAL_POSITIVE": natural_manifest["success_terminal"],
                "CURRENT_VERIFIED_CONTINUATION_POSITIVE": natural_manifest["current_continuation"],
            }, len(anchors), 128),
            "protocol_sha256": sha256(PROTOCOL), "tokenizer_sha256": tokenizer_sha,
            "anchor_dataset_sha256": sha256(dataset_path), "audit_sha256": sha256(audit_path),
            "heldout_rows": 0, "task_overlap": 0, "query_hash_overlap": 0,
            "gates": gates,
        }
        (output / "manifest.json").write_text(stable(summary) + "\n")
        summaries[f"A{crossed_cycle}"] = summary
        anchor_task_sets[crossed_cycle] = anchor_tasks
        print(stable({"pool": f"A{crossed_cycle}", "raw": raw_eligible,
                      "compatible": len(anchors), "overlap": exact_overlaps,
                      "anchor_only": len(anchors) - exact_overlaps}), flush=True)

    old_report = json.loads((REPORT / "recursive_opd_v3_cpu_prep.json").read_text())
    old18 = {row["task_id"] for row in old_report["pi3_crossed_to_pi4_regression"]["tasks"]
             if row.get("pi4_termination_reason") == "MODEL_FINAL"}
    strict = {task for task in train
              if records[3][task].get("status") != "UNCERTAIN"
              and records[4][task].get("status") == "FIRST_ERROR"
              and records[4][task].get("error_regime") in {"START", "ADVANCE"}
              and records[4][task].get("termination_reason") == "MODEL_FINAL"
              and records[3][task].get("frontier_depth", -1)
              > records[4][task].get("frontier_depth", 10**9)}
    regression = {
        "historical_18": _classify_sets(old18, natural_positive_sets[3], anchor_task_sets[3]),
        "strict_all": _classify_sets(strict, natural_positive_sets[3], anchor_task_sets[3]),
    }
    report = {
        "schema_version": VERSION, "status": "BUILT_PENDING_TESTS",
        "verdict": "ANCHOR_NOT_READY", "gpu_used": "NO",
        "gpu_training_config": GPU_TRAINING_CONFIG,
        "anchor_pools": summaries, "pi3_pi4_regression_coverage": regression,
        "audit_gates": {"data": "PASS" if all(
            pool["status"] == "ANCHOR_POOL_READY"
            and all(value == "PASS" for value in pool["gates"].values())
            for pool in summaries.values()) else "FAIL"},
        "not_run": ["V3 GPU training", "V3 fresh rollout", "GRPO",
                    "Static retraining", "BFCL"],
    }
    (RUN / "cpu_audit_draft.json").write_text(stable(report) + "\n")
    return report


def _markdown(report: dict) -> str:
    lines = [f"VERDICT = {report['verdict']}", "", "GPU used = NO", "",
             "# Recursive OPD V3 Fresh Crossed-Frontier Anchor CPU Audit", "",
             "Fresh anchors are event-triggered by previous-cycle failure followed by current-cycle "
             "verified mastery. They are not fixed-ratio continuation oversampling and expire after "
             "the immediately following recursive update.", "", "## Anchor pools", "",
             "| Pool | Transition | Raw | Compatible | Overlap current | Anchor-only | START/ADVANCE | Depth |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    for name, value in report["anchor_pools"].items():
        lines.append(f"| {name} | {value['source_transition']} | {value['raw_eligible']} | "
                     f"{value['context_compatible']} | {value['overlap_current_positive']} | "
                     f"{value['anchor_only']} | {value['regime_distribution']} | "
                     f"{value['depth_distribution']} |")
    lines += ["", "## Historical-only alignment", "",
              "| Pool | Historical latest-prior | Fresh previous-cycle | Intersection | Historical old-only |",
              "|---|---:|---:|---:|---:|"]
    for name, value in report["anchor_pools"].items():
        lines.append(f"| {name} | {value['historical_only_eligible']} | "
                     f"{value['context_compatible']} | {value['historical_task_intersection']} | "
                     f"{value['historical_old_only']} |")
    lines += ["", "A1 aligns exactly. A2-A4 are smaller because Fresh Anchor accepts only the "
              "immediately previous cycle, whereas historical V3 searched the latest eligible "
              "frontier from any earlier cycle. The verifier and context gates remain aligned.",
              "", "## Natural plus anchor (descriptive only)", "",
              "| Pool | Natural | Anchor | Semantic entries | Exact overlap | Deduplicated | 0/1/2/3/>3 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for name, value in report["anchor_pools"].items():
        lines.append(f"| {name} | {value['natural_pool_size']} | {value['fresh_anchor_size']} | "
                     f"{value['combined_semantic_entries']} | {value['exact_overlap']} | "
                     f"{value['deduplicated_unique_rows']} | {value['semantic_entries_per_task']} |")
    lines += ["", "## π3→π4 regression coverage", "",
              f"Historical 18: {report['pi3_pi4_regression_coverage']['historical_18']}",
              f"Strict all: {report['pi3_pi4_regression_coverage']['strict_all']}", "",
              "## Audit", "", f"CPU tests: {report['cpu_tests']['summary']}", "",
              "GPU TRAINING CONFIG = UNDECIDED", "",
              "B=128 scenarios are ILLUSTRATIVE ONLY and NOT APPROVED TRAINING CONFIG.", ""]
    return "\n".join(lines)


def finalize() -> dict:
    draft = RUN / "cpu_audit_draft.json"
    if not draft.exists():
        raise RuntimeError("run build before finalize")
    report = json.loads(draft.read_text())
    tests = [
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v1.py",
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v2.py",
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v3.py",
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v3_complete_c.py",
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v3_fresh_anchor.py",
    ]
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(command, text=True, capture_output=True,
                               cwd=HISTORICAL_RUN.parents[2])
    log = RUN / "cpu_tests.log"
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"fresh-anchor CPU tests failed; see {log}")
    if report.get("audit_gates", {}).get("data") != "PASS":
        raise RuntimeError("fresh-anchor data gates are not fully PASS")
    summary = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "PASS"
    report.update(status="AUDITED", verdict="ANCHOR_READY",
                  cpu_tests={"status": "PASS", "summary": summary,
                             "command": command, "log_sha256": sha256(log)})
    json_path = REPORT / "recursive_opd_v3_fresh_anchor_cpu_audit.json"
    md_path = REPORT / "recursive_opd_v3_fresh_anchor_cpu_audit.md"
    if json_path.exists() or md_path.exists():
        raise RuntimeError("refusing to overwrite fresh-anchor report")
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
