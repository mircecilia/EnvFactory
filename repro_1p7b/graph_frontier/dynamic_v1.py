#!/usr/bin/env python3
"""Dynamic Graph-Frontier SFT v1 data and diagnosis pipeline.

Allocation is derived only from the Stage-1 model on the frozen diagnosis split.
The held-out split is rejected at every allocation boundary.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from repro_1p7b.data_analysis import parameter_aware_data as feature_lib

SEED = 20260914
STAGE1_COUNT = 13232
STAGE2_COUNT = 13231
OVERLAP_CAP_RATE = 0.20
UNIQUE_RATIO_FLOOR = 0.90
EXPLORATION_FLOOR = 0.05
BUCKETS = (
    "shallow_general",
    "depth1_internal",
    "depth2_internal",
    "depth3plus_internal",
)
REQUIRED_FIELDS = {"instruction", "input", "output", "system", "history"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_id(sample: Mapping[str, Any]) -> str:
    raw = json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def bucket_name(feature: Mapping[str, Any]) -> str:
    depth = int(feature.get("max_dependency_depth", 0))
    if depth <= 0:
        return "shallow_general"
    if depth == 1:
        return "depth1_internal"
    if depth == 2:
        return "depth2_internal"
    return "depth3plus_internal"


def tool_calls(sample: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    parsed: list[dict[str, Any]] = []
    malformed = 0
    for _, output in feature_lib.pair_sequence(dict(sample)):
        for fragment in feature_lib.TOOL_CALL_RE.findall(output or ""):
            try:
                call = json.loads(fragment)
            except (json.JSONDecodeError, TypeError):
                malformed += 1
                continue
            if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                malformed += 1
                continue
            parsed.append(call)
    return parsed, malformed


def quality_audit(sample: Mapping[str, Any]) -> dict[str, Any]:
    malformed_example = (
        not isinstance(sample, dict)
        or not REQUIRED_FIELDS.issubset(sample)
        or not isinstance(sample.get("history"), list)
    )
    calls, malformed_calls = tool_calls(sample) if not malformed_example else ([], 0)
    if not malformed_example:
        for _, output in feature_lib.pair_sequence(dict(sample)):
            malformed_calls += abs(
                (output or "").count("<tool_call>")
                - (output or "").count("</tool_call>")
            )
    seen: set[str] = set()
    repeated = 0
    signature = []
    for call in calls:
        canonical = json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        repeated += int(canonical in seen)
        seen.add(canonical)
        signature.append(call["name"])
    return {
        "eligible": not malformed_example and malformed_calls == 0 and repeated == 0,
        "malformed_example": malformed_example,
        "malformed_tool_calls": malformed_calls,
        "repeated_exact_tool_calls": repeated,
        "tool_sequence_signature": " -> ".join(signature) if signature else "no_tool_call",
    }


def feature_summary(features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bucket_counts = Counter(bucket_name(row) for row in features)
    total_chars = sum(int(row.get("chars", 0)) for row in features)
    return {
        "bucket_counts": {bucket: bucket_counts[bucket] for bucket in BUCKETS},
        "total_chars": total_chars,
        "token_proxy_chars_div_4": math.ceil(total_chars / 4),
        "internal_dependency_proxy_samples": sum(
            bool(row.get("has_internal_dependency")) for row in features
        ),
        "internal_edge_proxy_total": sum(int(row.get("dependent_calls", 0)) for row in features),
        "tool_chain_length": feature_lib.distribution(
            [int(row.get("tool_calls", 0)) for row in features]
        ),
        "internal_dependency_edges": feature_lib.distribution(
            [int(row.get("dependent_calls", 0)) for row in features]
        ),
    }


def load_source_and_features(source_path: Path, features_path: Path):
    samples = load_json_array(source_path)
    features = load_jsonl(features_path)
    if len(samples) != len(features):
        raise ValueError(f"source/features mismatch: {len(samples)} != {len(features)}")
    for index, feature in enumerate(features):
        if int(feature.get("index", -1)) != index:
            raise ValueError(f"feature index mismatch at {index}")
    return samples, features


def select_unique_positions(samples: Sequence[Mapping[str, Any]], count: int, seed: int):
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    selected: list[int] = []
    hashes: list[str] = []
    used: set[str] = set()
    for pos in order:
        digest = content_id(samples[pos])
        if digest in used:
            continue
        selected.append(pos)
        hashes.append(digest)
        used.add(digest)
        if len(selected) == count:
            return selected, hashes
    raise RuntimeError(f"Only {len(selected)} unique source examples; need {count}")


def build_stage1(args: argparse.Namespace) -> None:
    source, features = load_source_and_features(args.source, args.features)
    positions, hashes = select_unique_positions(source, args.count, args.seed)
    selected = [source[pos] for pos in positions]
    selected_features = [features[pos] for pos in positions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, separators=(",", ":")))
    source_summary = feature_summary(features)
    selected_summary = feature_summary(selected_features)
    stats = {
        "schema_version": "graph_frontier_dynamic_stage1_v1",
        "status": "PASS",
        "seed": args.seed,
        "source": str(args.source),
        "source_count": len(source),
        "sample_count": len(selected),
        "unique_content_count": len(set(hashes)),
        "selection": "deterministic_shuffle_without_replacement",
        "source_sha256": sha256_path(args.source),
        "output_sha256": sha256_path(args.output),
        "source_feature_summary": source_summary,
        "selected_feature_summary": selected_summary,
        "bucket_rate_delta_selected_minus_source": {
            bucket: (
                selected_summary["bucket_counts"][bucket] / len(selected)
                - source_summary["bucket_counts"][bucket] / len(source)
            )
            for bucket in BUCKETS
        },
        **selected_summary,
    }
    write_json(args.stats, stats)
    print(json.dumps(stats, indent=2))


def build_smoke_data(args: argparse.Namespace) -> None:
    source = load_json_array(args.source)
    positions, _ = select_unique_positions(source, args.count * 2, args.seed)
    first = [source[pos] for pos in positions[: args.count]]
    second = [source[pos] for pos in positions[args.count :]]
    args.stage1.parent.mkdir(parents=True, exist_ok=True)
    args.stage1.write_text(json.dumps(first, ensure_ascii=False, separators=(",", ":")))
    args.stage2.write_text(json.dumps(second, ensure_ascii=False, separators=(",", ":")))
    result = {
        "status": "PASS", "stage1_count": len(first), "stage2_count": len(second),
        "cross_stage_overlap": len({content_id(x) for x in first} & {content_id(x) for x in second}),
        "stage1_sha256": sha256_path(args.stage1), "stage2_sha256": sha256_path(args.stage2),
    }
    write_json(args.report, result); print(json.dumps(result, indent=2))


def validate_stage1(args: argparse.Namespace) -> None:
    samples = load_json_array(args.stage1)
    stats = json.loads(args.stats.read_text())
    hashes = [content_id(sample) for sample in samples]
    checks = {
        "exact_count": len(samples) == STAGE1_COUNT,
        "content_unique": len(set(hashes)) == len(samples),
        "stats_pass": stats.get("status") == "PASS",
        "stats_count": stats.get("sample_count") == len(samples),
        "stats_unique_count": stats.get("unique_content_count") == len(set(hashes)),
        "stats_sha256": stats.get("output_sha256") == sha256_path(args.stage1),
        "fixed_seed": stats.get("seed") == SEED,
        "selection_is_unweighted": stats.get("selection") == "deterministic_shuffle_without_replacement",
    }
    report = {
        "schema_version": "graph_frontier_dynamic_stage1_validation_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def metric_payload(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "successes": int(metric["successes"]),
        "attempts": int(metric["attempts"]),
        "raw_rate": metric["raw_rate"],
        "smoothed_rate": metric["smoothed_rate"],
    }


def build_capability_map(args: argparse.Namespace) -> None:
    from repro_1p7b.graph_frontier.confirm_analyze import group_summary, root_failure, task_rows_for_model

    manifests = load_jsonl(args.manifest)
    diagnosis = [row for row in manifests if row.get("split") == "diagnosis"]
    if len(diagnosis) != 240:
        raise RuntimeError(f"diagnosis manifest count must be 240, got {len(diagnosis)}")
    rows, runtime = task_rows_for_model(args.result_root, diagnosis, args.label)
    if runtime["valid"] < args.minimum_valid:
        raise RuntimeError(f"valid diagnosis gate failed: {runtime}")
    buckets: dict[str, Any] = {}
    for depth, bucket in ((1, "depth1_internal"), (2, "depth2_internal"), (3, "depth3plus_internal")):
        selected = [
            row for row in rows.values()
            if min(int(row["dependency_depth"]), 3) == depth
        ]
        summary = group_summary(selected)
        buckets[bucket] = {
            "task_support": len(selected),
            "semantic_success": metric_payload(summary["semantic_task_success"]),
            "internal_reach": metric_payload(summary["internal_reach"]),
            "conditional_propagation": metric_payload(summary["conditional_propagation_accuracy"]),
            "internal_end_to_end": metric_payload(summary["internal_end_to_end"]),
            "root_failure_counts": dict(Counter(root_failure(row) for row in selected)),
        }
    overall = group_summary(list(rows.values()))
    capability = {
        "schema_version": "graph_frontier_stage1_capability_v1",
        "status": "PASS",
        "source_model_label": args.label,
        "consumed_split": "diagnosis",
        "heldout_tasks_consumed": 0,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_path(args.manifest),
        "runtime": runtime,
        "overall_semantic_success": metric_payload(overall["semantic_task_success"]),
        "buckets": buckets,
    }
    write_json(args.output, capability)
    print(json.dumps(capability, indent=2))


def largest_remainder(probabilities: Mapping[str, float], count: int) -> dict[str, int]:
    exact = {bucket: probabilities[bucket] * count for bucket in BUCKETS}
    answer = {bucket: int(math.floor(exact[bucket])) for bucket in BUCKETS}
    order = sorted(BUCKETS, key=lambda b: (-(exact[b] - answer[b]), b))
    for bucket in order[: count - sum(answer.values())]:
        answer[bucket] += 1
    return answer


def priorities(capability: Mapping[str, Any], floor: float) -> dict[str, Any]:
    semantic_global = float(capability["overall_semantic_success"]["smoothed_rate"])
    details: dict[str, Any] = {}
    for bucket in BUCKETS:
        if bucket == "shallow_general":
            reach, conditional, semantic = 0.0, 0.0, semantic_global
            support = int(capability["runtime"]["valid"])
        else:
            row = capability["buckets"][bucket]
            reach = float(row["internal_reach"]["smoothed_rate"])
            conditional = float(row["conditional_propagation"]["smoothed_rate"])
            semantic = float(row["semantic_success"]["smoothed_rate"])
            support = int(row["task_support"])
        propagation_need = reach * (1.0 - conditional)
        semantic_frontier = 4.0 * semantic * (1.0 - semantic)
        raw = 0.70 * propagation_need + 0.30 * semantic_frontier
        details[bucket] = {
            "support": support,
            "reach_smoothed": reach,
            "conditional_smoothed": conditional,
            "semantic_smoothed": semantic,
            "propagation_need": propagation_need,
            "semantic_frontier": semantic_frontier,
            "raw_priority": raw,
        }
    total = sum(row["raw_priority"] for row in details.values())
    if not 0.0 <= floor < 1.0 / len(BUCKETS):
        raise ValueError(f"exploration floor must be in [0, {1.0 / len(BUCKETS)}), got {floor}")
    if total <= 0:
        raise ValueError("Capability map produced zero total frontier priority")
    residual = 1.0 - floor * len(BUCKETS)
    for row in details.values():
        row["normalized_probability"] = floor + residual * row["raw_priority"] / total
    return details


def constrain_allocation(requested, total_capacity, unseen_capacity, count, max_overlap, floor_count, priority_rows):
    allocation = dict(requested)
    while True:
        over = next((b for b in BUCKETS if allocation[b] > total_capacity[b]), None)
        if over is None:
            break
        receivers = [b for b in BUCKETS if allocation[b] < total_capacity[b]]
        if not receivers:
            raise RuntimeError("Insufficient total quality-eligible capacity")
        receiver = max(receivers, key=lambda b: (priority_rows[b]["raw_priority"], b))
        move = min(allocation[over] - total_capacity[over], total_capacity[receiver] - allocation[receiver])
        allocation[over] -= move
        allocation[receiver] += move
    def overlap_required() -> int:
        return sum(max(0, allocation[b] - unseen_capacity[b]) for b in BUCKETS)
    while overlap_required() > max_overlap:
        donors = [b for b in BUCKETS if allocation[b] > max(unseen_capacity[b], floor_count)]
        receivers = [b for b in BUCKETS if allocation[b] < unseen_capacity[b]]
        if not donors or not receivers:
            raise RuntimeError("Cannot satisfy overlap cap and exploration floor")
        donor = max(donors, key=lambda b: allocation[b] - unseen_capacity[b])
        receiver = max(receivers, key=lambda b: (priority_rows[b]["raw_priority"], b))
        move = min(
            overlap_required() - max_overlap,
            allocation[donor] - max(unseen_capacity[donor], floor_count),
            unseen_capacity[receiver] - allocation[receiver],
        )
        allocation[donor] -= move
        allocation[receiver] += move
    if sum(allocation.values()) != count or min(allocation.values()) < floor_count:
        raise RuntimeError(f"invalid constrained allocation: {allocation}")
    return allocation


def build_stage2(args: argparse.Namespace) -> None:
    source, features = load_source_and_features(args.source, args.features)
    stage1 = load_json_array(args.stage1)
    capability = json.loads(args.capability.read_text())
    if capability.get("consumed_split") != "diagnosis" or capability.get("heldout_tasks_consumed") != 0:
        raise RuntimeError("Capability map is not diagnosis-only")
    stage1_hashes = {content_id(sample) for sample in stage1}
    candidates: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    rejection = Counter()
    seen_source_hashes: set[str] = set()
    for pos, (sample, feature) in enumerate(zip(source, features)):
        digest = content_id(sample)
        if digest in seen_source_hashes:
            rejection["duplicate_source_content"] += 1
            continue
        seen_source_hashes.add(digest)
        audit = quality_audit(sample)
        if not audit["eligible"]:
            rejection["malformed_or_repeated_trace"] += 1
            continue
        bucket = bucket_name(feature)
        if bucket != "shallow_general" and int(feature.get("dependent_calls", 0)) <= 0:
            rejection["unresolved_internal_binding_proxy"] += 1
            continue
        candidates[bucket].append((pos, digest, audit))
    priority_rows = priorities(capability, args.exploration_floor)
    probabilities = {b: priority_rows[b]["normalized_probability"] for b in BUCKETS}
    requested = largest_remainder(probabilities, args.count)
    total_capacity = {b: len(candidates[b]) for b in BUCKETS}
    unseen_capacity = {b: sum(digest not in stage1_hashes for _, digest, _ in candidates[b]) for b in BUCKETS}
    max_overlap = math.floor(args.count * args.overlap_cap)
    floor_count = math.floor(args.count * args.exploration_floor)
    allocation = constrain_allocation(
        requested, total_capacity, unseen_capacity, args.count, max_overlap, floor_count, priority_rows
    )
    rng = random.Random(args.seed)
    selected_positions: list[int] = []
    selected_hashes: list[str] = []
    signatures = Counter()
    actual_overlap = 0
    for bucket in BUCKETS:
        unseen = [row for row in candidates[bucket] if row[1] not in stage1_hashes]
        seen = [row for row in candidates[bucket] if row[1] in stage1_hashes]
        rng.shuffle(unseen); rng.shuffle(seen)
        take_unseen = min(allocation[bucket], len(unseen))
        chosen = unseen[:take_unseen] + seen[: allocation[bucket] - take_unseen]
        if len(chosen) != allocation[bucket]:
            raise RuntimeError(f"bucket capacity failure: {bucket}")
        for pos, digest, audit in chosen:
            selected_positions.append(pos); selected_hashes.append(digest)
            signatures[audit["tool_sequence_signature"]] += 1
            actual_overlap += int(digest in stage1_hashes)
    combined = list(zip(selected_positions, selected_hashes))
    rng.shuffle(combined)
    selected_positions = [row[0] for row in combined]
    selected_hashes = [row[1] for row in combined]
    selected = [source[pos] for pos in selected_positions]
    selected_features = [features[pos] for pos in selected_positions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, separators=(",", ":")))
    total_unique = len(stage1_hashes | set(selected_hashes))
    plan = {
        "schema_version": "graph_frontier_dynamic_allocation_v1", "status": "PASS", "seed": args.seed,
        "formula": {
            "propagation_need": "reach_smoothed * (1 - conditional_smoothed)",
            "semantic_frontier": "4 * p_semantic_smoothed * (1 - p_semantic_smoothed)",
            "priority": "0.70 * propagation_need + 0.30 * semantic_frontier",
            "support_smoothing": "Beta(1,1) posterior mean from diagnosis counts",
            "shallow_general_mapping": "zero propagation need plus global semantic frontier",
        },
        "exploration_floor_probability": args.exploration_floor,
        "overlap_cap_rate": args.overlap_cap, "overlap_cap_count": max_overlap,
        "capability_map": capability, "bucket_priority": priority_rows,
        "requested_allocation": requested, "quality_eligible_capacity": total_capacity,
        "unseen_quality_eligible_capacity": unseen_capacity, "target_allocation": allocation,
        "actual_overlap": actual_overlap, "rejection_counts": dict(rejection),
    }
    stats = {
        "schema_version": "graph_frontier_dynamic_stage2_v1", "status": "PASS",
        "sample_count": len(selected), "unique_content_count": len(set(selected_hashes)),
        "cross_stage_overlap": actual_overlap, "cross_stage_overlap_rate": actual_overlap / len(selected),
        "overall_unique_content": total_unique,
        "overall_unique_ratio": total_unique / (len(stage1) + len(selected)),
        "output_sha256": sha256_path(args.output),
        "top_tool_sequence_signatures": dict(signatures.most_common(20)),
        "max_tool_sequence_signature_rate": max(signatures.values(), default=0) / len(selected),
        **feature_summary(selected_features),
    }
    write_json(args.plan, plan); write_json(args.stats, stats)
    print(json.dumps({"plan": plan, "stats": stats}, indent=2))


def validate_datasets(args: argparse.Namespace) -> None:
    stage1 = load_json_array(args.stage1); stage2 = load_json_array(args.stage2)
    plan = json.loads(args.plan.read_text()); stats = json.loads(args.stats.read_text())
    h1 = [content_id(sample) for sample in stage1]; h2 = [content_id(sample) for sample in stage2]
    malformed = [i for i, sample in enumerate(stage2) if not quality_audit(sample)["eligible"]]
    overlap = len(set(h1) & set(h2)); unique_ratio = len(set(h1) | set(h2)) / (len(stage1) + len(stage2))
    stage1_chars = sum(feature_lib.sample_text_size(sample) for sample in stage1)
    stage2_chars = sum(feature_lib.sample_text_size(sample) for sample in stage2)
    per_sample_char_ratio = (
        (stage2_chars / len(stage2)) / (stage1_chars / len(stage1))
        if stage1 and stage2 and stage1_chars else None
    )
    checks = {
        "stage1_exact_count": len(stage1) == STAGE1_COUNT,
        "stage2_exact_count": len(stage2) == STAGE2_COUNT,
        "stage1_unique": len(set(h1)) == len(stage1), "stage2_unique": len(set(h2)) == len(stage2),
        "stage2_no_malformed_or_repeated": not malformed,
        "overlap_cap": overlap <= math.floor(STAGE2_COUNT * OVERLAP_CAP_RATE),
        "overall_unique_ratio": unique_ratio >= UNIQUE_RATIO_FLOOR,
        "allocation_count": sum(plan["target_allocation"].values()) == STAGE2_COUNT,
        "allocation_matches_stats": plan["target_allocation"] == stats["bucket_counts"],
        "no_zero_probability_bucket": all(row["normalized_probability"] > 0 for row in plan["bucket_priority"].values()),
        "no_pathological_signature_domination": stats["max_tool_sequence_signature_rate"] <= 0.25,
        "diagnosis_only": plan["capability_map"]["consumed_split"] == "diagnosis" and plan["capability_map"]["heldout_tasks_consumed"] == 0,
        "char_budget_sanity": per_sample_char_ratio is not None and 0.25 <= per_sample_char_ratio <= 4.0,
    }
    report = {
        "schema_version": "graph_frontier_dynamic_validation_v1",
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "malformed_indices": malformed[:20], "cross_stage_overlap": overlap,
        "overall_unique_ratio": unique_ratio,
        "stage1_total_chars": stage1_chars,
        "stage2_total_chars": stage2_chars,
        "stage2_to_stage1_mean_char_ratio": per_sample_char_ratio,
    }
    write_json(args.report, report); print(json.dumps(report, indent=2))
    if report["status"] != "PASS": raise SystemExit(1)


async def diagnose_async(args: argparse.Namespace) -> dict[str, Any]:
    from repro_1p7b.graph_frontier import pilot_static as pilot
    from repro_1p7b.graph_frontier.confirm_300 import classify_exception, run_one
    from src.manager.mcp_client_manager import MCPManager

    rows = [row for row in load_jsonl(args.manifest) if row.get("split") == args.split]
    expected = 240 if args.split == "diagnosis" else 60
    if len(rows) != expected: raise RuntimeError(f"{args.split} count mismatch: {len(rows)} != {expected}")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "run_config.json", {
        "model": args.label, "model_path": str(args.model_path),
        "manifest": str(args.manifest), "manifest_sha256": sha256_path(args.manifest),
        "consumed_split": args.split, "heldout_tasks_consumed": expected if args.split == "heldout" else 0,
    })
    answers = []; started = time.monotonic(); pilot.register()
    try:
        for index, row in enumerate(rows, 1):
            try:
                result = await run_one(row, args.label, args.output)
            except Exception as exc:
                result = {
                    "schema_version": "graph_frontier_confirm_result_v1", "task_id": row["task_id"],
                    "model": args.label, "split": row["split"], "runtime_seconds": 0.0,
                    "valid_capability_probe": False, "system_status": classify_exception(exc),
                    "profiler_error": None, "fatal_exception": f"{type(exc).__name__}:{exc}",
                }
                write_json(args.output / "tasks" / f"{row['task_id']}.result.json", result)
            answers.append(result)
            print(f"DYNAMIC_DIAGNOSIS_PROGRESS completed={index}/{expected} valid={sum(x.get('valid_capability_probe') is True for x in answers)}", flush=True)
    finally:
        MCPManager.shutdown()
    summary = {
        "model": args.label, "split": args.split, "completed": len(answers),
        "valid": sum(x.get("valid_capability_probe") is True for x in answers),
        "system_error_counts": dict(Counter(x.get("system_status", "unknown") for x in answers if x.get("valid_capability_probe") is not True)),
        "runtime_seconds": time.monotonic() - started,
    }
    write_json(args.output / "run_summary.json", summary)
    return summary


def diagnose(args: argparse.Namespace) -> None:
    print(json.dumps(asyncio.run(diagnose_async(args)), indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(); sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stage1")
    p.add_argument("--source", type=Path, required=True); p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--count", type=int, default=STAGE1_COUNT); p.add_argument("--seed", type=int, default=SEED); p.set_defaults(func=build_stage1)
    p = sub.add_parser("smoke-data")
    p.add_argument("--source", type=Path, required=True); p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--stage2", type=Path, required=True); p.add_argument("--report", type=Path, required=True)
    p.add_argument("--count", type=int, default=4); p.add_argument("--seed", type=int, default=SEED); p.set_defaults(func=build_smoke_data)
    p = sub.add_parser("validate-stage1")
    p.add_argument("--stage1", type=Path, required=True); p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True); p.set_defaults(func=validate_stage1)
    p = sub.add_parser("capability")
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--label", default="stage1"); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--minimum-valid", type=int, default=228); p.set_defaults(func=build_capability_map)
    p = sub.add_parser("stage2")
    p.add_argument("--source", type=Path, required=True); p.add_argument("--features", type=Path, required=True)
    p.add_argument("--stage1", type=Path, required=True); p.add_argument("--capability", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--stats", type=Path, required=True); p.add_argument("--count", type=int, default=STAGE2_COUNT)
    p.add_argument("--seed", type=int, default=SEED); p.add_argument("--overlap-cap", type=float, default=OVERLAP_CAP_RATE)
    p.add_argument("--exploration-floor", type=float, default=EXPLORATION_FLOOR); p.set_defaults(func=build_stage2)
    p = sub.add_parser("validate")
    p.add_argument("--stage1", type=Path, required=True); p.add_argument("--stage2", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True); p.set_defaults(func=validate_datasets)
    p = sub.add_parser("diagnose")
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--split", choices=("diagnosis", "heldout"), default="diagnosis")
    p.add_argument("--label", default="stage1"); p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.set_defaults(func=diagnose)
    return root


def main() -> None:
    args = parser().parse_args(); args.func(args)


if __name__ == "__main__": main()
