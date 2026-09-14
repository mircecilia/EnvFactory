"""Deterministic v2 analysis for the frozen 300-probe confirmation study."""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from repro_1p7b.graph_frontier.pilot_reanalysis_v2 import (
    LABELS,
    aggregate_model,
    args,
    call_patterns,
    events,
    group_summary,
    internal_funnel,
    mcnemar,
    rate,
    semantic_verify,
    short,
    successful,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260914


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace(",", "")
    return " ".join(text.split())


def semantic_verify_with_text(
    manifest: Mapping[str, Any],
    rollout: Mapping[str, Any],
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    semantic = semantic_verify(manifest, rollout, reference)
    specification = manifest["semantic_verifier"]
    semantic.update(
        verifier_type=specification["verifier_type"],
        checked_fields=specification["checked_fields"],
        reference_source=specification["reference_source"],
        text_match_mode=specification["text_match_mode"],
        ambiguity_flag=specification["ambiguity_flag"],
    )
    supported = (
        specification["semantic_verifier_supported"]
        and semantic["semantic_verifier_supported"]
    )
    semantic["semantic_verifier_supported"] = supported
    if not supported:
        semantic["semantic_success"] = False
        semantic["text_predicate"] = None
        semantic["unsupported_reason"] = (
            specification.get("unsupported_reason")
            or semantic.get("unsupported_reason")
            or "unsupported"
        )
        return semantic
    if specification["verifier_type"] == "state_predicate":
        semantic["text_predicate"] = None
        semantic["unsupported_reason"] = None
        return semantic
    content = rollout.get("final_assistant_content", "unknown")
    normalized = normalize_text(content)
    values = specification.get("text_match_values") or []
    matches = {
        normalize_text(value): normalize_text(value) in normalized for value in values
    }
    semantic["text_match_details"] = matches
    semantic["text_predicate"] = bool(matches) and all(matches.values())
    semantic["semantic_success"] = bool(
        semantic["semantic_success"] and semantic["text_predicate"]
    )
    semantic["unsupported_reason"] = None
    return semantic


def paired_metric(
    left: Mapping[str, bool], right: Mapping[str, bool]
) -> dict[str, Any]:
    shared = sorted(set(left) & set(right))
    base = mcnemar(left, right)
    left_success = sum(left[task_id] for task_id in shared)
    right_success = sum(right[task_id] for task_id in shared)
    return {
        **base,
        "ties_both_success": sum(left[task_id] and right[task_id] for task_id in shared),
        "ties_both_failure": sum(
            not left[task_id] and not right[task_id] for task_id in shared
        ),
        "left_successes": left_success,
        "right_successes": right_success,
        "effect_size_pp_right_minus_left": (
            100 * (right_success - left_success) / len(shared) if shared else None
        ),
    }


def funnel_with_conversions(funnel: Mapping[str, Any]) -> dict[str, Any]:
    stages = [
        ("gold_internal_opportunities", "producer_reached"),
        ("producer_reached", "producer_succeeded"),
        ("producer_succeeded", "consumer_reached_after_successful_producer"),
        (
            "consumer_reached_after_successful_producer",
            "source_and_target_value_inspectable",
        ),
        ("source_and_target_value_inspectable", "correct_propagated"),
        ("correct_propagated", "correct_edges_with_final_semantic_success"),
    ]
    conversions = {}
    for source, target in stages:
        denominator = funnel.get(source, 0)
        conversions[f"{source}_to_{target}"] = (
            funnel.get(target, 0) / denominator if denominator else None
        )
    return {**funnel, "stage_conversion_rates": conversions}


def root_failure(row: Mapping[str, Any]) -> str:
    rollout_events = events(row["rollout"])
    for edge in row["manifest"]["dependency_edges"]:
        producer = edge["producer_tool"]["tool_name"]
        consumer = edge["consumer_tool"]["tool_name"]
        producer_rows = [event for event in rollout_events if event["tool_name"] == producer]
        if not producer_rows:
            return "producer_not_executed"
        successful_producers = [
            event for event in producer_rows if successful(event)
        ]
        if not successful_producers:
            return "producer_failed"
        earliest_producer = min(event.get("step_index", -1) for event in successful_producers)
        consumer_rows = [
            event
            for event in rollout_events
            if event["tool_name"] == consumer
            and event.get("step_index", -1) > earliest_producer
        ]
        if not consumer_rows:
            return "consumer_not_executed"
        check = next(
            (
                item
                for item in row["result"].get("profile", {}).get(
                    "dependency_edge_checks", []
                )
                if item.get("edge_id") == edge["edge_id"]
            ),
            {},
        )
        if check.get("status") == "failed_wrong_value":
            return "wrong_propagated_value"
        if (
            check.get("source_value_available") is not True
            or check.get("target_value_available") is not True
        ):
            return "missing_internal_value"
        if check.get("success") is not True:
            return "dependency_failure"
    semantic = row["semantic"]
    if semantic.get("state_predicate") is False:
        return "semantic_target_failure"
    if (
        semantic.get("observation_predicate") is False
        or semantic.get("text_predicate") is False
    ):
        return "observation_failure"
    if row["calls"]["hit_tool_budget"]:
        return "tool_budget_exhausted"
    if (
        row["calls"]["cycle_pattern_counts"]
        or row["calls"]["repeated_pattern_counts"].get("retry_after_tool_error", 0)
    ):
        return "loop_retry_pathology"
    return "other_failure"


def load_manifest(root: Path) -> tuple[Path, list[dict[str, Any]]]:
    path = root / "frozen" / f"confirm_300_seed_{SEED}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 300:
        raise RuntimeError(f"manifest_count:{len(rows)}")
    return path, rows


def task_rows_for_model(
    root: Path,
    manifests: Sequence[Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    system_errors: Counter[str] = Counter()
    runtime_seconds = 0.0
    completed = 0
    for manifest in manifests:
        task_id = manifest["task_id"]
        result_path = root / label / "tasks" / f"{task_id}.result.json"
        if not result_path.exists():
            system_errors["missing_result"] += 1
            continue
        completed += 1
        result = json.loads(result_path.read_text())
        runtime_seconds += float(result.get("runtime_seconds", 0.0))
        if result.get("valid_capability_probe") is not True:
            system_errors[result.get("system_status", "unknown")] += 1
            continue
        rollout_path = root / label / "tasks" / f"{task_id}.rollout.json"
        if not rollout_path.exists():
            system_errors["missing_rollout"] += 1
            continue
        rollout = json.loads(rollout_path.read_text())
        reference = json.loads((ROOT / manifest["reference_trace_path"]).read_text())
        semantic = semantic_verify_with_text(manifest, rollout, reference)
        funnel = internal_funnel(manifest, rollout, result)
        funnel["correct_edges_with_final_semantic_success"] = (
            funnel.get("correct_propagated", 0)
            if semantic["semantic_success"]
            else 0
        )
        calls = call_patterns(manifest, rollout, result, semantic)
        rows[task_id] = {
            "task_id": task_id,
            "split": manifest["split"],
            "environment": manifest["environment"],
            "template": manifest["template"],
            "dependency_depth": manifest["dependency_depth"],
            "semantic": semantic,
            "reference_path_complete_success": bool(
                result.get("reference_path_complete_success", result.get("task_success"))
            ),
            "internal_task_complete": bool(result.get("internal_param_complete")),
            "edge_complete": bool(result.get("edge_complete")),
            "gold_node_complete": bool(result.get("gold_node_complete")),
            "final_state_success": result.get("final_state_success"),
            "funnel": funnel,
            "calls": calls,
            "manifest": manifest,
            "rollout": rollout,
            "result": result,
        }
    runtime = {
        "completed": completed,
        "valid": len(rows),
        "system_errors": dict(system_errors),
        "runtime_seconds_sum": runtime_seconds,
    }
    return rows, runtime


def metric_maps(
    rows: Mapping[str, Mapping[str, Any]],
    split: str | None = None,
) -> dict[str, dict[str, bool]]:
    selected = {
        task_id: row
        for task_id, row in rows.items()
        if split is None or row["split"] == split
    }
    return {
        "semantic_task_success": {
            task_id: row["semantic"]["semantic_success"]
            for task_id, row in selected.items()
            if row["semantic"]["semantic_verifier_supported"]
        },
        "reference_path_complete_success": {
            task_id: row["reference_path_complete_success"]
            for task_id, row in selected.items()
        },
        "task_level_internal_edge_complete": {
            task_id: row["internal_task_complete"] for task_id, row in selected.items()
        },
    }


def aggregate(
    task_rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    values = list(task_rows.values())
    model = aggregate_model(values)
    model["internal_dependency_funnel"] = funnel_with_conversions(
        model["internal_dependency_funnel"]
    )
    model["final_state_success"] = rate(
        sum(row["final_state_success"] is True for row in values), len(values)
    )
    model["task_level_internal_edge_complete"] = rate(
        sum(row["internal_task_complete"] for row in values), len(values)
    )
    total_calls = sum(len(events(row["rollout"])) for row in values)
    model["efficiency"] = {
        "total_tool_calls": total_calls,
        "calls_per_task": total_calls / len(values) if values else None,
        "redundant_calls": model["call_audit"]["legacy_exact_redundant_calls"],
        "redundant_calls_per_task": model["redundant_calls_per_task"],
        "unexpected_calls": model["call_audit"]["legacy_unexpected_calls"],
        "unexpected_calls_per_task": model["unexpected_calls_per_task"],
        "retry_after_error": model["call_audit"]["repeated_patterns_nonexclusive"].get(
            "retry_after_tool_error", 0
        ),
        "repeated_same_tool_args": model["call_audit"][
            "repeated_patterns_nonexclusive"
        ].get("same_tool_same_args", 0),
        "repeated_block_cycles": model["call_audit"][
            "cycle_patterns_nonexclusive"
        ].get("repeated_block_cycle", 0),
        "tool_budget_exhaustion_tasks": model["call_audit"]["tool_budget_exceeded"][
            "tasks"
        ],
    }
    failures = [
        row for row in values
        if row["semantic"]["semantic_verifier_supported"]
        and not row["semantic"]["semantic_success"]
    ]
    model["root_failure_attribution"] = dict(
        Counter(root_failure(row) for row in failures)
    )
    model["split_breakdown"] = {
        split: group_summary([row for row in values if row["split"] == split])
        for split in ("diagnosis", "heldout")
    }
    return model


def paired_report(
    task_maps: Mapping[str, Mapping[str, Mapping[str, Any]]],
    split: str | None = None,
) -> dict[str, Any]:
    maps = {
        label: metric_maps(task_maps[label], split=split) for label in LABELS
    }
    answer = {}
    for left, right in (
        ("base", "original_sft"),
        ("base", "parameter_aware"),
        ("original_sft", "parameter_aware"),
    ):
        answer[f"{left}_vs_{right}"] = {
            metric: paired_metric(maps[left][metric], maps[right][metric])
            for metric in (
                "semantic_task_success",
                "reference_path_complete_success",
                "task_level_internal_edge_complete",
            )
        }
    return answer


def manifest_summary(manifest_path: Path, manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audit = json.loads((manifest_path.parent / "candidate_audit.json").read_text())
    return {
        "seed": SEED,
        "candidates_generated": audit["candidates_generated"],
        "accepted": len(manifests),
        "rejected": audit["rejected"],
        "rejection_reason_counts": audit["rejection_reason_counts"],
        "split_distribution": dict(Counter(row["split"] for row in manifests)),
        "environment_distribution": dict(
            Counter(row["environment"] for row in manifests)
        ),
        "template_distribution": dict(Counter(row["template"] for row in manifests)),
        "depth_distribution": dict(
            Counter(row["complexity_bucket"] for row in manifests)
        ),
        "gold_internal_edges": sum(
            len(row["dependency_edges"]) for row in manifests
        ),
        "internal_parameter_count": sum(
            len(row["dependency_edges"]) for row in manifests
        ),
        "semantic_verifier_support": sum(
            row["semantic_verifier"]["semantic_verifier_supported"]
            for row in manifests
        ),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def build_report(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path, manifests = load_manifest(root)
    task_maps = {}
    runtime = {}
    models = {}
    for label in LABELS:
        task_maps[label], runtime[label] = task_rows_for_model(root, manifests, label)
        models[label] = aggregate(task_maps[label]) if task_maps[label] else {}
    valid_gate = all(runtime[label]["valid"] >= 285 for label in LABELS)
    paired = paired_report(task_maps) if all(task_maps.values()) else {}
    split_paired = {
        split: paired_report(task_maps, split=split)
        for split in ("diagnosis", "heldout")
    } if all(task_maps.values()) else {}
    conclusions = {
        "validity_gate_passed": valid_gate,
        "interpretation_allowed": valid_gate,
        "pilot_semantic_direction_reproduced": (
            valid_gate
            and models["original_sft"]["semantic_task_success"]["raw_rate"]
            > models["parameter_aware"]["semantic_task_success"]["raw_rate"]
        ),
        "pilot_internal_direction_reproduced": (
            valid_gate
            and models["original_sft"]["internal_end_to_end"]["raw_rate"]
            > models["parameter_aware"]["internal_end_to_end"]["raw_rate"]
        ),
        "pilot_efficiency_direction_reproduced": (
            valid_gate
            and models["parameter_aware"]["efficiency"]["redundant_calls_per_task"]
            > models["original_sft"]["efficiency"]["redundant_calls_per_task"]
        ),
        "bfcl_context": (
            "BFCL Missing Parameter Original=8.0%, PA=15.0%; comparison is "
            "descriptive only and no causal claim is made."
        ),
    }
    report = {
        "schema_version": "graph_frontier_confirm_capability_v1",
        "source_policy": "frozen_manifest_and_completed_rollouts_only",
        "manifest": manifest_summary(manifest_path, manifests),
        "runtime": runtime,
        "validity_gate": {
            "minimum_valid_per_model": 285,
            "passed": valid_gate,
        },
        "models": models,
        "paired_statistics": paired,
        "split_paired_statistics": split_paired,
        "conclusions": conclusions,
        "limitations": [
            "Task-level McNemar tests are primary; edges within a task are clustered.",
            "Deterministic text matching can create false negatives for valid paraphrases.",
            "Unexpected tools can be valid alternative paths and are not necessarily errors.",
            "The diagnosis split is intended for future allocation and is not an unbiased test set.",
            "The 60-task held-out split is a directional sanity check and is not over-interpreted.",
        ],
    }
    failures = {
        "schema_version": "graph_frontier_confirm_failure_patterns_v1",
        "models": {
            label: {
                "call_audit": models[label].get("call_audit", {}),
                "root_failure_attribution": models[label].get(
                    "root_failure_attribution", {}
                ),
                "alternative_valid_path": models[label].get(
                    "alternative_valid_path", {}
                ),
            }
            for label in LABELS
        },
    }
    return report, failures


def fmt(metric: Mapping[str, Any]) -> str:
    if metric.get("raw_rate") is None:
        return f"n/a ({metric.get('successes', 0)}/{metric.get('attempts', 0)})"
    return (
        f"{metric['successes']}/{metric['attempts']}="
        f"{100 * metric['raw_rate']:.1f}%"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    manifest = report["manifest"]
    models = report["models"]
    lines = [
        "# Graph-Frontier 300-Probe Confirmation Study",
        "",
        "## Manifest",
        "",
        f"- Seed: {manifest['seed']}",
        f"- Candidate/accepted/rejected: {manifest['candidates_generated']}/{manifest['accepted']}/{manifest['rejected']}",
        f"- Split: {manifest['split_distribution']}",
        f"- Environment: {manifest['environment_distribution']}",
        f"- Depth: {manifest['depth_distribution']}",
        f"- Gold internal edges: {manifest['gold_internal_edges']}",
        f"- Semantic support: {manifest['semantic_verifier_support']}/300",
        f"- SHA256: {manifest['manifest_sha256']}",
        "",
        "## Runtime",
        "",
        "| Model | Completed | Valid | Runtime seconds | System errors |",
        "|---|---:|---:|---:|---|",
    ]
    for label in LABELS:
        runtime = report["runtime"][label]
        lines.append(
            f"| {label} | {runtime['completed']} | {runtime['valid']} | "
            f"{runtime['runtime_seconds_sum']:.1f} | {runtime['system_errors']} |"
        )
    lines += [
        "",
        "## Core metrics",
        "",
        "| Model | Semantic | Reference path | Internal reach | Conditional propagation | Internal end-to-end | Redundant/task | Unexpected/task |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        model = models[label]
        lines.append(
            f"| {label} | {fmt(model['semantic_task_success'])} | "
            f"{fmt(model['reference_path_complete_success'])} | "
            f"{fmt(model['internal_reach'])} | "
            f"{fmt(model['conditional_propagation_accuracy'])} | "
            f"{fmt(model['internal_end_to_end'])} | "
            f"{model['efficiency']['redundant_calls_per_task']:.2f} | "
            f"{model['efficiency']['unexpected_calls_per_task']:.2f} |"
        )
    lines += ["", "## Depth", ""]
    for depth in ("1", "2", "3+"):
        lines += [
            f"### Depth {depth}",
            "",
            "| Model | Semantic | Reach | Conditional | End-to-end | Redundant/task | Unexpected/task |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label in LABELS:
            bucket = models[label]["depth_breakdown"][depth]
            lines.append(
                f"| {label} | {fmt(bucket['semantic_task_success'])} | "
                f"{fmt(bucket['internal_reach'])} | "
                f"{fmt(bucket['conditional_propagation_accuracy'])} | "
                f"{fmt(bucket['internal_end_to_end'])} | "
                f"{bucket['redundant_calls_per_task']:.2f} | "
                f"{bucket['unexpected_calls_per_task']:.2f} |"
            )
        lines.append("")
    lines += ["## Paired Original vs PA", ""]
    paired = report["paired_statistics"]["original_sft_vs_parameter_aware"]
    for name, metric in paired.items():
        lines.append(
            f"- {name}: PA wins={metric['right_wins']}, losses={metric['right_losses']}, "
            f"effect={metric['effect_size_pp_right_minus_left']:.2f}pp, "
            f"exact p={metric['exact_two_sided_p']:.6g}"
        )
    lines += ["", "## Diagnosis vs held-out", ""]
    for split in ("diagnosis", "heldout"):
        metric = report["split_paired_statistics"][split][
            "original_sft_vs_parameter_aware"
        ]["semantic_task_success"]
        lines.append(
            f"- {split}: PA wins={metric['right_wins']}, losses={metric['right_losses']}, "
            f"effect={metric['effect_size_pp_right_minus_left']:.2f}pp, "
            f"exact p={metric['exact_two_sided_p']:.6g}"
        )
    lines += [
        "",
        "## Conclusions",
        "",
        f"- Validity gate passed: {report['validity_gate']['passed']}",
    ]
    for key, value in report["conclusions"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Limitations", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "repro_1p7b/results/graph_frontier/confirm_300",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=ROOT / "repro_1p7b/graph_frontier/reports",
    )
    args = parser.parse_args()
    report, failures = build_report(args.root)
    args.reports.mkdir(parents=True, exist_ok=True)
    json_path = args.reports / "confirm_300_capability_comparison.json"
    markdown_path = args.reports / "confirm_300_capability_comparison.md"
    failure_path = args.reports / "confirm_300_failure_patterns.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(report))
    failure_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "report": str(json_path),
        "validity_gate": report["validity_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
