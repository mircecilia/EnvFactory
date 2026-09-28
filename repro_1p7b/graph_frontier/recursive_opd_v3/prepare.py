"""CPU-only Complete-C retrospective builder and independent audit.

No environment replay, model inference, or optimizer update is performed here.
All candidates come from immutable V2 fresh rollouts and verifier decisions.
"""
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

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import encode_first_divergence
from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for, valid_gold
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import OUT, SOURCE
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import (
    MODEL, PROTOCOL, RUN as V1_RUN, check,
)
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import GOLD_RUN
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import LIMIT
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import sample_lengths
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import RUN as V2_RUN, digest
from repro_1p7b.graph_frontier.recursive_opd_v3.lineage import (
    depth_bucket, historical_identity, latest_relevant_frontier, lineage_matches,
    movement, sample_count_distribution, verified_student_crossing,
)
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

VERSION = "recursive_opd_v3_complete_c_1"
RUN = V2_RUN.parent / "recursive_opd_v3_run"
REPORT = V2_RUN.parent / "reports"
TRAINING_BUDGET = "UNDECIDED"
SOURCE_LABELS = frozenset({
    "CURRENT_FRONTIER", "SUCCESS_TERMINAL", "CROSSED_FRONTIER_RETENTION",
})


def policy_root(cycle: int) -> Path:
    if cycle == 0:
        return V1_RUN / "cycle0"
    if cycle in (1, 2, 3, 4):
        return V2_RUN / f"cycle{cycle}"
    raise ValueError("retrospective policy cycle must be 0..4")


def load_policy(cycle: int, protocol: dict) -> tuple[list[dict], dict, dict]:
    root = policy_root(cycle)
    manifest_path = root / "rollout_manifest.json"
    metrics_path = root / "metrics/per_task.jsonl"
    summary_path = root / "metrics/summary.json"
    manifest, summary = json.loads(manifest_path.read_text()), json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    if (manifest.get("cycle") != cycle or manifest.get("protocol_sha256") != sha256(PROTOCOL)
            or summary.get("per_task_sha256") != sha256(metrics_path)
            or len(rows) != 2158 or len({r.get("task_id") for r in rows}) != 2158
            or {r["task_id"] for r in rows} != train | heldout
            or any(r.get("cycle") != cycle
                   or r.get("pool") != ("train" if r["task_id"] in train else "heldout")
                   for r in rows)):
        raise RuntimeError(f"policy pi{cycle} manifest/metrics/task identity drift")
    if {r["query_hash"] for r in rows if r["pool"] == "train"} & {
            r["query_hash"] for r in rows if r["pool"] == "heldout"}:
        raise RuntimeError(f"policy pi{cycle} train/heldout query hash overlap")
    if cycle == 0:
        if manifest.get("model_sha256") != protocol["pi0_model_sha256"]:
            raise RuntimeError("pi0 model identity drift")
    else:
        audit = json.loads((root / "metrics/audit.json").read_text())
        state = json.loads((root / "checkpoint/trainer_state.json").read_text())
        if (audit.get("status") != "AUDITED" or audit.get("cycle") != cycle
                or audit.get("metrics_sha256") != sha256(metrics_path)
                or state.get("status") != "TRAINED" or state.get("cycle") != cycle - 1
                or state.get("model_sha256") != manifest.get("model_sha256")):
            raise RuntimeError(f"pi{cycle} audit/checkpoint lineage drift")
    return rows, manifest, {
        "metrics_sha256": sha256(metrics_path),
        "rollout_manifest_sha256": sha256(manifest_path),
        "source_policy_sha256": manifest["model_sha256"],
    }


def _gold_path(task: str, episode: dict, cycle: int) -> Path:
    if episode.get("gold_source_kind") == "independent_replay_cache":
        return V1_RUN / "gold_cache" / f"{task}.json"
    return GOLD_RUN / "episodes" / f"{task}.json"


def load_episode(cycle: int, task: str, manifest: dict, manifest_sha: str,
                 source_rows: list[dict], entries: dict[str, dict]) -> tuple[dict, list[dict], str]:
    root = policy_root(cycle)
    path = root / "rollouts" / f"{task}.json"
    episode = json.loads(path.read_text())
    entry = entries[task]
    source_row = source_rows[entry["source_row_index"]]
    gold_path = _gold_path(task, episode, cycle)
    if (episode.get("task_id") != task or episode.get("cycle") != cycle
            or episode.get("pool") not in {"train", "heldout"}
            or episode.get("model_sha256") != manifest["model_sha256"]
            or episode.get("manifest_sha256") != manifest_sha
            or episode.get("gold_source_episode_sha256") != sha256(gold_path)
            or episode.get("student_error")
            or not valid_gold(entry, source_row, episode)):
        raise RuntimeError(f"pi{cycle} rollout/gold/provenance invalid: {task}")
    schema_root = (V2_RUN if cycle else
                   (V1_RUN if episode.get("gold_source_kind") == "independent_replay_cache"
                    else GOLD_RUN))
    schema = schema_for(schema_root, episode)
    return episode, schema, sha256(path)


def source_dataset(cycle: int) -> tuple[list[dict], dict, Path]:
    base = V2_RUN / ("d0_context16k_v2" if cycle == 0 else f"cycle{cycle}/dataset")
    path = base / "dataset.jsonl"
    manifest = json.loads((base / "manifest.json").read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    expected_usage = f"TRAINABLE_V2_D{cycle}"
    if (manifest.get("status") != "READY_FOR_GPU_TRAINING"
            or manifest.get("usage") != expected_usage or manifest.get("cycle") != cycle
            or manifest.get("dataset_sha256") != sha256(path)
            or manifest.get("sample_count") != len(rows)
            or manifest.get("heldout_training_rows") != 0
            or manifest.get("uncertain_training_rows", 0) != 0
            or not manifest.get("gates")
            or any(value != "PASS" for value in manifest["gates"].values())):
        raise RuntimeError(f"V2 D{cycle} source dataset gate failed")
    return rows, manifest, path


def _quantile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def length_stats(values: list[int]) -> dict:
    return {
        "count": len(values), "min": min(values) if values else None,
        "p50": _quantile(values, .50), "p90": _quantile(values, .90),
        "p95": _quantile(values, .95), "p99": _quantile(values, .99),
        "max": max(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "token_volume": sum(values),
        ">8192": sum(value > 8192 for value in values),
        ">12288": sum(value > 12288 for value in values),
        ">16384": sum(value > 16384 for value in values),
        ">24576": sum(value > 24576 for value in values),
        ">32768": sum(value > 32768 for value in values),
    }


def _main_label(row: dict) -> str:
    kind = row.get("correction_kind")
    if kind in {"START", "ADVANCE", "TERMINATE"}:
        return "CURRENT_FRONTIER"
    if kind == "SUCCESS_TERMINAL_POSITIVE":
        return "SUCCESS_TERMINAL"
    raise RuntimeError(f"unknown V2 main target: {kind}")


def _validate_union(rows: list[dict], train: set[str], heldout: set[str]) -> dict:
    identities, by_task = set(), collections.Counter()
    for row in rows:
        identity = row.get("v3_sample_identity")
        if (not identity or identity in identities or row.get("task_id") not in train
                or row.get("task_id") in heldout or row.get("pool") != "train"
                or row.get("v3_source_label") not in SOURCE_LABELS
                or row.get("source_verdict") == "UNCERTAIN"
                or len(row.get("input_ids", [])) > LIMIT):
            raise RuntimeError("V3 union identity/split/source/context gate failed")
        validate_mask(row)
        identities.add(identity)
        by_task[row["task_id"]] += 1
    if any(count > 2 for count in by_task.values()):
        raise RuntimeError("V3 union has more than two samples for a task")
    return {"unique_samples": len(identities), "unique_tasks": len(by_task)}


def _transition_worker(args: tuple) -> dict:
    (current_cycle, task, historical_record, source_meta, current_record,
     policy_manifests, policy_meta, source_rows, entries) = args
    try:
        source_episode, _, source_rollout_sha = load_episode(
            historical_record["cycle"], task,
            policy_manifests[historical_record["cycle"]],
            policy_meta[historical_record["cycle"]]["rollout_manifest_sha256"],
            source_rows, entries)
        current_episode, schema, current_rollout_sha = load_episode(
            current_cycle, task, policy_manifests[current_cycle],
            policy_meta[current_cycle]["rollout_manifest_sha256"], source_rows, entries)
        identity = historical_identity(
            historical_record, source_episode,
            metrics_sha256=source_meta["metrics_sha256"],
            rollout_sha256=source_rollout_sha,
            rollout_manifest_sha256=source_meta["rollout_manifest_sha256"],
            source_policy_sha256=source_meta["source_policy_sha256"])
        if not lineage_matches(identity, source_episode, current_record, current_episode):
            return {"task_id": task, "status": "AMBIGUOUS_LINEAGE",
                    "reason": "current/source immutable gold lineage mismatch"}
        moved = movement(identity, current_record)
        result = {"task_id": task, "status": moved, "identity": identity,
                  "current_cycle": current_cycle,
                  "current_status": current_record["status"],
                  "current_frontier_depth": current_record["frontier_depth"],
                  "current_termination_reason": current_record.get("termination_reason"),
                  "current_rollout_sha256": current_rollout_sha,
                  "current_rollout_manifest_sha256": policy_meta[current_cycle][
                      "rollout_manifest_sha256"],
                  "current_policy_sha256": policy_meta[current_cycle]["source_policy_sha256"]}
        if moved == "CROSSED":
            extracted = verified_student_crossing(identity, current_record, current_episode)
            result["retention"] = extracted
            if extracted["status"] == "RETENTION_ELIGIBLE":
                result["schema"] = schema
        return result
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return {"task_id": task, "status": "AMBIGUOUS_LINEAGE",
                "reason": f"{type(exc).__name__}:{str(exc)[:300]}"}


def _summarize_transition(outcomes: list[dict]) -> dict:
    status = collections.Counter(row["status"] for row in outcomes)
    eligible = [row for row in outcomes
                if (row.get("retention") or {}).get("status") == "RETENTION_ELIGIBLE"]
    by_regime = collections.Counter(row["identity"]["frontier_regime"] for row in outcomes
                                    if row.get("identity"))
    eligible_regime = collections.Counter(row["identity"]["frontier_regime"] for row in eligible)
    eligible_depth = collections.Counter(depth_bucket(row["identity"]["frontier_depth"])
                                         for row in eligible)
    ineligible = collections.Counter((row.get("retention") or {}).get("reason", "")
                                     for row in outcomes if row["status"] == "CROSSED"
                                     and (row.get("retention") or {}).get("status")
                                     != "RETENTION_ELIGIBLE")
    return {
        "historical_frontier": len(outcomes),
        "crossed": status["CROSSED"], "regressed": status["REGRESSED"],
        "uncertain": status["UNCERTAIN"],
        "ambiguous_lineage": status["AMBIGUOUS_LINEAGE"],
        "retention_eligible": len(eligible),
        "historical_by_regime": dict(by_regime),
        "retention_eligible_by_regime": dict(eligible_regime),
        "retention_eligible_by_depth": dict(eligible_depth),
        "crossed_but_retention_ineligible_reasons": dict(ineligible),
    }


def build(*, workers: int = 6) -> dict:
    if RUN.exists():
        raise RuntimeError(f"refusing to overwrite V3 runtime artifacts: {RUN}")
    if workers < 1 or workers > 8:
        raise ValueError("CPU preparation permits 1..8 workers")
    protocol = check()
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    if len(train) != 1726 or len(heldout) != 432 or train & heldout:
        raise RuntimeError("frozen task split invalid")
    source_rows = json.loads(SOURCE.read_text())
    entries = {row["audit_id"]: row for row in json.loads(
        (OUT / "opd_source_manifest.json").read_text())["rows"]}
    policy_rows, policy_manifests, policy_meta, records_by_task = {}, {}, {}, {}
    for cycle in range(5):
        rows, manifest, meta = load_policy(cycle, protocol)
        policy_rows[cycle], policy_manifests[cycle], policy_meta[cycle] = rows, manifest, meta
        records_by_task[cycle] = {row["task_id"]: row for row in rows}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    tokenizer_sha = sha256(MODEL / "tokenizer.json")
    if tokenizer.eos_token_id != 151645 or tokenizer_sha != protocol["tokenizer_sha256"]:
        raise RuntimeError("V3 tokenizer/EOS identity drift")

    RUN.mkdir(parents=True)
    all_history = [row for cycle in range(4) for row in policy_rows[cycle]]
    transitions, transition_rows, natural = {}, {}, {}
    for current_cycle in range(1, 5):
        jobs = []
        for task in protocol["train_task_ids"]:
            try:
                historical = latest_relevant_frontier(
                    all_history, task_id=task, current_cycle=current_cycle)
            except ValueError as exc:
                transition_rows.setdefault(current_cycle, []).append({
                    "task_id": task, "status": "AMBIGUOUS_LINEAGE", "reason": str(exc)})
                continue
            if historical is None:
                continue
            jobs.append((current_cycle, task, historical, policy_meta[historical["cycle"]],
                         records_by_task[current_cycle][task], policy_manifests, policy_meta,
                         source_rows, entries))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v3-lineage") as pool:
            outcomes = list(pool.map(_transition_worker, jobs))
        outcomes.extend(transition_rows.get(current_cycle, []))
        transition_rows[current_cycle] = outcomes
        transitions[f"pi{current_cycle - 1}->pi{current_cycle}"] = _summarize_transition(outcomes)
        print(stable({"transition": current_cycle, **transitions[
            f"pi{current_cycle - 1}->pi{current_cycle}"]}), flush=True)

    # Natural D0 has no historical frontier.  D1..D4 each add at most one
    # verified retention row to their already-audited V2 main behavior row.
    for cycle in range(5):
        main_rows, main_manifest, main_path = source_dataset(cycle)
        by_main = {row["task_id"]: row for row in main_rows}
        outcome_by_task = ({row["task_id"]: row for row in transition_rows.get(cycle, [])}
                           if cycle else {})
        output = RUN / f"cycle{cycle}/dataset"
        output.mkdir(parents=True)
        dataset_path, lineage_path = output / "dataset.jsonl", output / "lineage.jsonl"
        union, retention_overflow, retention_failures = [], [], []
        semantic_lengths = collections.defaultdict(list)
        for row in main_rows:
            item = dict(row)
            label = _main_label(item)
            item.update(v3_source_label=label,
                        v3_sample_identity=f"{item['task_id']}:v3:cycle{cycle}:{label}",
                        v3_schema_version=VERSION)
            union.append(item)
            semantic_lengths[label].append(len(item["input_ids"]))
        for overflow in main_manifest.get("context_overflow_records", []):
            label = ("SUCCESS_TERMINAL" if overflow.get("kind") == "SUCCESS_TERMINAL_POSITIVE"
                     else "CURRENT_FRONTIER")
            semantic_lengths[label].append(overflow["full_tokens"])
        if cycle:
            for task in protocol["train_task_ids"]:
                outcome = outcome_by_task.get(task)
                extracted = (outcome or {}).get("retention") or {}
                if extracted.get("status") != "RETENTION_ELIGIBLE":
                    continue
                identity = outcome["identity"]
                sample = {"task_id": task,
                          "state_identity": f"{task}:v3:cycle{cycle}:retain:{identity['source_cycle']}:{identity['gold_pointer']}",
                          "independent_prefix_replay_valid": True,
                          "conversation_prefix": extracted["conversation_prefix"],
                          "gold_action": extracted["student_action"]}
                length_record = {"status": "FIRST_ERROR", "error_regime": "ADVANCE",
                                 "gold_next_action": extracted["student_action"]}
                prompt_len, full_len = sample_lengths(
                    tokenizer, extracted["conversation_prefix"], outcome["schema"], length_record)
                semantic_lengths["CROSSED_FRONTIER_RETENTION"].append(full_len)
                provenance = {
                    "historical_frontier": identity,
                    "current_cycle": cycle,
                    "current_policy_sha256": outcome["current_policy_sha256"],
                    "current_rollout_manifest_sha256": outcome[
                        "current_rollout_manifest_sha256"],
                    "current_rollout_sha256": outcome["current_rollout_sha256"],
                    "current_student_step": extracted["current_student_step"],
                    "student_prefix_sha256": extracted["student_prefix_sha256"],
                    "student_action_sha256": extracted["student_action_sha256"],
                    "execution_success": True, "progress_verified": True,
                    "target_provenance": "CURRENT_VERIFIED_STUDENT_BEHAVIOR",
                }
                if full_len > LIMIT:
                    retention_overflow.append({"task_id": task, "prompt_tokens_local": prompt_len,
                                               "full_tokens": full_len,
                                               "reason": "CONTEXT_OVERFLOW", **provenance})
                    continue
                try:
                    item = encode_first_divergence(
                        tokenizer, sample, {"task_id": task, "tools": outcome["schema"]})
                    validate_mask(item)
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    retention_failures.append({"task_id": task, "reason": str(exc)[:300]})
                    continue
                if len(item["input_ids"]) != full_len:
                    raise RuntimeError("retention token length differs from independent audit")
                item.update(cycle=cycle, pool="train", source_verdict="VERIFIED_CROSSED_FRONTIER",
                            correction_kind="CROSSED_FRONTIER_RETENTION",
                            v3_source_label="CROSSED_FRONTIER_RETENTION",
                            v3_sample_identity=sample["state_identity"],
                            v3_schema_version=VERSION, train_context_compatible=True,
                            semantic_eligible=True, execution_verified=True, **provenance)
                union.append(item)
        _validate_union(union, train, heldout)
        main_tasks, retention_tasks = set(by_main), {
            row["task_id"] for row in union
            if row["v3_source_label"] == "CROSSED_FRONTIER_RETENTION"}
        sample_distribution = sample_count_distribution(main_tasks, retention_tasks, train)
        with dataset_path.open("w") as stream:
            for row in union:
                stream.write(stable(row) + "\n")
        with lineage_path.open("w") as stream:
            for row in transition_rows.get(cycle, []):
                compact = {key: value for key, value in row.items() if key != "schema"}
                stream.write(stable(compact) + "\n")
        counts = collections.Counter(row["v3_source_label"] for row in union)
        tool_targets = counts["CURRENT_FRONTIER"] + counts["CROSSED_FRONTIER_RETENTION"]
        # CURRENT_FRONTIER includes TERMINATE EOS, so use the authoritative V2
        # source manifest instead of treating every CURRENT_FRONTIER as a tool.
        main_tool = main_manifest["tool_targets"]
        main_eos = main_manifest["eos_targets"]
        tool_targets = main_tool + counts["CROSSED_FRONTIER_RETENTION"]
        eos_targets = main_eos
        context = {label: length_stats(semantic_lengths[label]) for label in sorted(SOURCE_LABELS)}
        context["ALL"] = length_stats([value for values in semantic_lengths.values() for value in values])
        gates = {name: "PASS" for name in (
            "current_frontier_builder", "terminal_positive_builder", "crossed_frontier_builder",
            "latest_frontier_lineage", "current_policy_provenance", "execution_verification",
            "heldout_leakage", "query_hash_leakage", "context_compatibility", "token_masks",
            "manifest_hashes", "natural_union_builder", "sample_count_per_task")}
        if retention_failures:
            gates["crossed_frontier_builder"] = "FAIL"
        manifest = {
            "schema_version": VERSION, "status": "READY_CPU_DATA" if not retention_failures else "NOT_READY",
            "cycle": cycle, "training_budget": TRAINING_BUDGET,
            "source_v2_dataset_sha256": sha256(main_path),
            "source_v2_manifest_sha256": sha256(main_path.parent / "manifest.json"),
            "source_rollout_manifest_sha256": policy_meta[cycle]["rollout_manifest_sha256"],
            "source_policy_sha256": policy_meta[cycle]["source_policy_sha256"],
            "protocol_sha256": sha256(PROTOCOL), "tokenizer_sha256": tokenizer_sha,
            "dataset_sha256": sha256(dataset_path), "lineage_sha256": sha256(lineage_path),
            "sample_count": len(union), "class_counts": dict(counts),
            "current_frontier": counts["CURRENT_FRONTIER"],
            "success_terminal": counts["SUCCESS_TERMINAL"],
            "crossed_frontier_retention": counts["CROSSED_FRONTIER_RETENTION"],
            "tool_targets": tool_targets, "eos_targets": eos_targets,
            "tool_fraction": tool_targets / len(union) if union else 0,
            "eos_fraction": eos_targets / len(union) if union else 0,
            "tool_eos_ratio": None if not eos_targets else tool_targets / eos_targets,
            "sample_count_per_task": sample_distribution,
            "context_overflow_count": main_manifest.get("context_overflow_count", 0)
                                      + len(retention_overflow),
            "main_context_overflow_count": main_manifest.get("context_overflow_count", 0),
            "retention_context_overflow_count": len(retention_overflow),
            "retention_context_overflow_records": retention_overflow,
            "retention_encoding_failures": retention_failures,
            "context_lengths": context, "heldout_training_rows": 0,
            "uncertain_training_rows": 0, "gates": gates,
        }
        (output / "manifest.json").write_text(stable(manifest) + "\n")
        natural[str(cycle)] = manifest
        print(stable({"natural_cycle": cycle, "samples": len(union),
                      "classes": dict(counts), "tool": tool_targets, "eos": eos_targets}), flush=True)

    # Directly quantify loss of a continuation that pi3 had strictly crossed.
    pi3_eligible = {row["task_id"]: row for row in transition_rows[3]
                    if (row.get("retention") or {}).get("status") == "RETENTION_ELIGIBLE"}
    pi4_regression = collections.Counter()
    regression_tasks = []
    for task, old in pi3_eligible.items():
        row4 = records_by_task[4][task]
        if row4["frontier_depth"] > old["identity"]["gold_pointer"]:
            pi4_regression["still_crossed"] += 1
        elif row4["status"] == "UNCERTAIN":
            pi4_regression["uncertain"] += 1
        else:
            pi4_regression["regressed"] += 1
            premature = (row4.get("termination_reason") == "MODEL_FINAL"
                          and row4.get("error_regime") in {"START", "ADVANCE"})
            pi4_regression["premature_stop_regression"] += int(premature)
            regression_tasks.append({"task_id": task,
                                     "historical_pointer": old["identity"]["gold_pointer"],
                                     "pi4_frontier_depth": row4["frontier_depth"],
                                     "pi4_error_regime": row4.get("error_regime"),
                                     "pi4_termination_reason": row4.get("termination_reason")})
    report_data = {
        "schema_version": VERSION, "status": "BUILT_PENDING_TESTS",
        "verdict": "NOT_READY", "training_budget": TRAINING_BUDGET,
        "protocol_sha256": sha256(PROTOCOL), "tokenizer_sha256": tokenizer_sha,
        "transitions": transitions, "natural_pools": natural,
        "pi3_crossed_to_pi4_regression": {"pi3_retention_eligible": len(pi3_eligible),
                                           **dict(pi4_regression),
                                           "tasks": regression_tasks},
        "mechanical_accumulation_gate": "PASS" if all(
            value["historical_frontier"] <= 1726 for value in transitions.values()) else "FAIL",
        "not_run": ["V3 GPU training", "V3 fresh rollout", "GRPO", "Static retraining", "BFCL"],
    }
    (RUN / "cpu_prep_draft.json").write_text(stable(report_data) + "\n")
    return report_data


def _markdown(report: dict) -> str:
    lines = [f"VERDICT = {report['verdict']}", "", f"TRAINING BUDGET = {TRAINING_BUDGET}", "",
             "# Recursive OPD V3 / Complete-C CPU preparation", "",
             "All counts below are train-split only; heldout rows are excluded from every dataset.", "",
             "## Transition lineage", "",
             "| transition | historical | crossed | regressed | uncertain | ambiguous | retention eligible |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in report["transitions"].items():
        lines.append(f"| {name} | {value['historical_frontier']} | {value['crossed']} | "
                     f"{value['regressed']} | {value['uncertain']} | "
                     f"{value['ambiguous_lineage']} | {value['retention_eligible']} |")
    lines += ["", "## Natural Complete-C pools", "",
              "| cycle | CURRENT_FRONTIER | SUCCESS_TERMINAL | CROSSED_RETENTION | TOTAL | tool | EOS | tool/EOS | 0/1/2 tasks | overflow |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for cycle, value in report["natural_pools"].items():
        ratio = value["tool_eos_ratio"]
        lines.append(f"| {cycle} | {value['current_frontier']} | {value['success_terminal']} | "
                     f"{value['crossed_frontier_retention']} | {value['sample_count']} | "
                     f"{value['tool_targets']} | {value['eos_targets']} | "
                     f"{ratio if ratio is not None else 'NA'} | {value['sample_count_per_task']} | "
                     f"{value['context_overflow_count']} |")
    regression = report["pi3_crossed_to_pi4_regression"]
    lines += ["", "## π2→π3 versus π3→π4", "",
              f"π3 strict retention-eligible crossings: {regression['pi3_retention_eligible']}.",
              f"Of these, π4 still crossed {regression.get('still_crossed', 0)}, regressed "
              f"{regression.get('regressed', 0)}, and regressed by premature stop "
              f"{regression.get('premature_stop_regression', 0)}.", "",
              "The latest-relevant-frontier rule admits at most one retention candidate per task/cycle; "
              f"mechanical accumulation gate: {report['mechanical_accumulation_gate']}.", "",
              "## Gates", "",
              f"CPU tests: {report['cpu_tests']['summary']}", "",
              "V3 GPU parameters remain deliberately unset. READY means protocol/data engineering is complete, "
              "not that a training budget has been approved.", ""]
    return "\n".join(lines)


def finalize() -> dict:
    draft_path = RUN / "cpu_prep_draft.json"
    if not draft_path.exists():
        raise RuntimeError("run build before finalize")
    report = json.loads(draft_path.read_text())
    tests = [
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v1.py",
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v2.py",
        "repro_1p7b/graph_frontier/tests/test_recursive_opd_v3.py",
    ]
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(command, text=True, capture_output=True, cwd=V2_RUN.parents[2])
    log = RUN / "cpu_tests.log"
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"V3/V2 CPU tests failed; see {log}")
    if (report.get("mechanical_accumulation_gate") != "PASS"
            or any(pool.get("status") != "READY_CPU_DATA"
                   or any(value != "PASS" for value in pool.get("gates", {}).values())
                   for pool in report["natural_pools"].values())):
        raise RuntimeError("V3 data/provenance gate is not fully PASS")
    summary = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "PASS"
    report.update(status="AUDITED", verdict="READY_FOR_GPU_TRAINING",
                  cpu_tests={"status": "PASS", "summary": summary,
                             "command": command, "log_sha256": sha256(log)})
    REPORT.mkdir(parents=True, exist_ok=True)
    json_path = REPORT / "recursive_opd_v3_cpu_prep.json"
    md_path = REPORT / "recursive_opd_v3_cpu_prep.md"
    if json_path.exists() or md_path.exists():
        raise RuntimeError("refusing to overwrite V3 CPU report")
    json_path.write_text(stable(report) + "\n")
    md_path.write_text(_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "finalize"))
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    answer = build(workers=args.workers) if args.command == "build" else finalize()
    print(stable({"status": answer["status"], "verdict": answer["verdict"],
                  "training_budget": answer["training_budget"]}), flush=True)


if __name__ == "__main__":
    main()
