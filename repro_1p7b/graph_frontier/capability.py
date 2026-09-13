"""Aggregate rollout profiles into smoothed capability statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .adapters import UNKNOWN


def beta_smoothed_metric(
    successes: int,
    attempts: int,
    previous_pass_rate: Optional[float] = None,
    hint_with: Optional[Tuple[int, int]] = None,
    hint_without: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    if attempts < 0 or successes < 0 or successes > attempts:
        raise ValueError("Require 0 <= successes <= attempts")
    pass_rate = (successes + 1) / (attempts + 2)
    learning_progress = (
        pass_rate - previous_pass_rate
        if previous_pass_rate is not None
        else None
    )
    hint_regret = None
    hint_evidence: Any = None
    if hint_with is not None and hint_without is not None:
        with_success, with_attempts = hint_with
        without_success, without_attempts = hint_without
        with_rate = (with_success + 1) / (with_attempts + 2)
        without_rate = (without_success + 1) / (without_attempts + 2)
        hint_regret = with_rate - without_rate
        hint_evidence = {
            "with_hint": {
                "successes": with_success,
                "attempts": with_attempts,
                "pass_rate": with_rate,
            },
            "without_hint": {
                "successes": without_success,
                "attempts": without_attempts,
                "pass_rate": without_rate,
            },
        }
    return {
        "successes": successes,
        "attempts": attempts,
        "pass_rate": pass_rate,
        "frontier_score": 4.0 * pass_rate * (1.0 - pass_rate),
        "weakness": 1.0 - pass_rate,
        "learning_progress": learning_progress,
        "hint_regret": hint_regret,
        "hint_evidence": hint_evidence,
        "smoothing": "Beta(1,1)",
    }


def _record(
    counters: Dict[str, list],
    key: str,
    success: Any,
) -> None:
    if isinstance(success, bool):
        counters[key][1] += 1
        counters[key][0] += int(success)


def aggregate_profiles(
    profiles: Iterable[Mapping[str, Any]],
    previous_capability_map: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate known outcomes only; unknown fields do not become attempts."""

    profiles = list(profiles)
    counters: Dict[str, list] = defaultdict(lambda: [0, 0])
    root_causes = Counter()
    max_depth_reached = Counter()
    unknown_counts = Counter()

    hint_terminal: Dict[str, list] = defaultdict(lambda: [0, 0])

    for profile in profiles:
        terminal = profile.get("terminal_success", UNKNOWN)
        _record(counters, "terminal", terminal)
        if not isinstance(terminal, bool):
            unknown_counts["terminal_success"] += 1

        hint_condition = profile.get("hint_condition")
        if hint_condition in {"with_hint", "without_hint"} and isinstance(terminal, bool):
            hint_terminal[hint_condition][1] += 1
            hint_terminal[hint_condition][0] += int(terminal)

        max_depth = profile.get("max_dependency_depth_reached", UNKNOWN)
        if isinstance(max_depth, int):
            max_depth_reached[str(max_depth)] += 1
        else:
            unknown_counts["max_dependency_depth_reached"] += 1

        for check in profile.get("dependency_edge_checks") or []:
            success = check.get("success", UNKNOWN)
            if not isinstance(success, bool):
                unknown_counts["dependency_edge_checks"] += 1
                continue
            depth = check.get("dependency_depth", UNKNOWN)
            if isinstance(depth, int):
                _record(counters, f"depth:{depth}", success)
            _record(counters, f"edge:{check.get('edge_id', UNKNOWN)}", success)
            _record(counters, f"edge_type:{check.get('edge_type', UNKNOWN)}", success)
            if check.get("internal_parameter") is True:
                _record(counters, "internal_parameter_flow", success)

        for event in profile.get("actual_tool_nodes") or []:
            success = event.get("execution_success", UNKNOWN)
            state_changing = event.get("state_changing", UNKNOWN)
            if not isinstance(success, bool):
                unknown_counts["tool_execution_success"] += 1
                continue
            if state_changing is True:
                _record(counters, "tool_class:state_changing", success)
            elif state_changing is False:
                _record(counters, "tool_class:query", success)
            else:
                unknown_counts["tool_state_changing"] += 1

        for root in profile.get("root_cause_failures") or []:
            root_causes[str(root.get("type", UNKNOWN))] += 1

    previous_metrics = (
        previous_capability_map.get("metrics", {})
        if previous_capability_map is not None
        else {}
    )
    metrics = {}
    for key in sorted(counters):
        successes, attempts = counters[key]
        previous = previous_metrics.get(key, {}).get("pass_rate")
        hint_with = hint_without = None
        if key == "terminal" and {"with_hint", "without_hint"}.issubset(hint_terminal):
            hint_with = tuple(hint_terminal["with_hint"])
            hint_without = tuple(hint_terminal["without_hint"])
        metrics[key] = beta_smoothed_metric(
            successes,
            attempts,
            previous_pass_rate=previous,
            hint_with=hint_with,
            hint_without=hint_without,
        )

    return {
        "schema_version": "graph_frontier_capability_map_v1",
        "profile_count": len(profiles),
        "metrics": metrics,
        "dependency_depth_buckets": {
            key.split(":", 1)[1]: metrics[key]
            for key in metrics
            if key.startswith("depth:")
        },
        "edge_type_success": {
            key.split(":", 1)[1]: metrics[key]
            for key in metrics
            if key.startswith("edge_type:")
        },
        "internal_parameter_propagation": metrics.get(
            "internal_parameter_flow",
            beta_smoothed_metric(0, 0),
        ),
        "tool_class_success": {
            key.split(":", 1)[1]: metrics[key]
            for key in metrics
            if key.startswith("tool_class:")
        },
        "root_cause_failure_frequency": dict(sorted(root_causes.items())),
        "max_dependency_depth_reached": dict(sorted(max_depth_reached.items(), key=lambda item: int(item[0]))),
        "unknown_observations": dict(sorted(unknown_counts.items())),
    }
