"""Pure, fail-closed fresh crossed-frontier anchor verification."""
from __future__ import annotations

from collections import Counter
from typing import Any

from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import typed_equal
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import digest
from repro_1p7b.graph_frontier.recursive_opd_v3.lineage import (
    historical_identity, lineage_matches, movement, verified_student_crossing,
)


def depth_bucket(depth: int) -> str:
    if depth < 0:
        raise ValueError("negative anchor depth")
    return str(depth) if depth < 3 else "3+"


def fresh_identity(source_record: dict, source_episode: dict, *, crossed_cycle: int,
                   metrics_sha256: str, rollout_sha256: str,
                   rollout_manifest_sha256: str, source_policy_sha256: str,
                   source_dataset_sha256: str) -> dict:
    """Create an identity only for the immediately preceding policy error."""
    source_cycle = source_record.get("cycle")
    if not isinstance(source_cycle, int) or crossed_cycle != source_cycle + 1:
        raise ValueError("fresh anchor source must be exactly the previous cycle")
    value = historical_identity(
        source_record, source_episode, metrics_sha256=metrics_sha256,
        rollout_sha256=rollout_sha256,
        rollout_manifest_sha256=rollout_manifest_sha256,
        source_policy_sha256=source_policy_sha256,
    )
    value.update(
        source_error_cycle=source_cycle,
        crossed_cycle=crossed_cycle,
        intended_replay_cycle=crossed_cycle,
        expires_after_update_cycle=crossed_cycle,
        anchor_age=1,
        source_dataset_sha256=source_dataset_sha256,
    )
    return value


def _model_action(step: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    message = step.get("model_message") or {}
    calls = message.get("tool_calls")
    structured = step.get("structured_calls") or []
    if (message.get("role") != "assistant" or message.get("content") not in ("", None)
            or not isinstance(calls, list) or len(calls) != 1 or len(structured) != 1):
        return None
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if (not isinstance(function, dict) or not isinstance(function.get("arguments"), dict)):
        return None
    normalized = {"name": function.get("name"), "arguments": function["arguments"]}
    parsed = {"name": structured[0].get("name"), "arguments": structured[0].get("arguments")}
    if not typed_equal(normalized, parsed):
        return None
    return message, normalized


def extract_fresh_anchor(identity: dict, source_episode: dict, current_record: dict,
                         current_episode: dict) -> dict:
    """Recover one exact current student action crossing a fresh frontier."""
    if identity.get("anchor_age") != 1:
        return {"status": "ANCHOR_UNCERTAIN", "reason": "anchor age is not one cycle"}
    if identity.get("crossed_cycle") != identity.get("source_error_cycle", -2) + 1:
        return {"status": "ANCHOR_UNCERTAIN", "reason": "cycle freshness mismatch"}
    if not lineage_matches(identity, source_episode, current_record, current_episode):
        return {"status": "ANCHOR_UNCERTAIN", "reason": "immutable task/gold lineage mismatch"}
    moved = movement(identity, current_record)
    if moved == "UNCERTAIN":
        return {"status": "ANCHOR_UNCERTAIN", "reason": "current crossing verdict uncertain"}
    if moved != "CROSSED":
        return {"status": "FRONTIER_NOT_CROSSED"}
    verified = verified_student_crossing(identity, current_record, current_episode)
    if verified.get("status") != "RETENTION_ELIGIBLE":
        return {"status": "ANCHOR_UNCERTAIN", "reason": verified.get("reason", verified["status"])}
    steps = (current_episode.get("student") or {}).get("steps") or []
    selected = [step for step in steps if step.get("step_index") == verified["current_student_step"]]
    if len(selected) != 1:
        return {"status": "ANCHOR_UNCERTAIN", "reason": "crossing student turn is not unique"}
    step = selected[0]
    action = _model_action(step)
    if action is None:
        return {"status": "ANCHOR_UNCERTAIN", "reason": "exact current assistant action unavailable"}
    message, normalized = action
    prefix = verified["conversation_prefix"]
    if (digest(prefix) != verified["student_prefix_sha256"]
            or digest(normalized) != verified["student_action_sha256"]
            or not typed_equal(normalized, verified["student_action"])):
        return {"status": "ANCHOR_UNCERTAIN", "reason": "student prefix/action provenance mismatch"}
    return {
        "status": "FRESH_ANCHOR_ELIGIBLE",
        "conversation_prefix": prefix,
        "student_action": message,
        "normalized_student_action": normalized,
        "current_prefix_hash": digest(prefix),
        "current_state_before_hash": digest(step["state_before_action"]),
        "current_state_after_hash": digest(step["state_after_action"]),
        "current_action_hash": digest(message),
        "normalized_action_hash": digest(normalized),
        "current_student_step": step["step_index"],
        "progress_before": identity["gold_pointer"],
        "progress_after": identity["gold_pointer"] + 1,
        "execution_verified": True,
        "terminal_flag": False,
        "matches_historical_gold": digest(normalized) == identity["target_action_sha256"],
    }


def exact_overlap(anchor: dict, current_positive: dict | None) -> bool:
    return bool(current_positive
                and anchor.get("task_id") == current_positive.get("task_id")
                and anchor.get("current_prefix_hash") == current_positive.get("prefix_sha256")
                and anchor.get("current_action_hash") == current_positive.get("student_action_sha256"))


def semantic_count_distribution(natural_counts: dict[str, int], anchor_tasks: set[str],
                                train_tasks: set[str]) -> dict[str, int]:
    if not set(natural_counts) <= train_tasks or not anchor_tasks <= train_tasks:
        raise ValueError("heldout task entered semantic entries")
    values = Counter(natural_counts.get(task, 0) + int(task in anchor_tasks)
                     for task in train_tasks)
    if any(count < 0 or count > 3 for count in values):
        raise RuntimeError("more than three semantic entries for a task")
    return {str(key): values.get(key, 0) for key in range(4)} | {
        ">3": sum(value for key, value in values.items() if key > 3)}


def descriptive_exposure(natural_counts: dict[str, int], anchor_count: int,
                         budget: int) -> dict:
    if budget < 1 or anchor_count < 0 or not natural_counts or sum(natural_counts.values()) < 1:
        raise ValueError("invalid descriptive exposure inputs")
    total = sum(natural_counts.values())
    return {
        "label": "ILLUSTRATIVE ONLY - NOT APPROVED TRAINING CONFIG",
        "natural_budget": budget,
        "natural_expected_category_exposure": {
            key: budget * value / total for key, value in natural_counts.items()},
        "fresh_anchors_guaranteed_once": anchor_count,
        "total_exposure": budget + anchor_count,
    }
