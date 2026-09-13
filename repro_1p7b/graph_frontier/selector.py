"""Deterministic budget-aware curriculum selector skeleton."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _candidate_signal(
    candidate: Mapping[str, Any],
    capability_map: Mapping[str, Any],
    component: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    metrics = capability_map.get("metrics", {})
    reasons = []
    values = []
    for key in candidate.get("capability_keys") or []:
        metric = metrics.get(key)
        if not isinstance(metric, Mapping):
            reasons.append({"capability_key": key, "status": "missing"})
            continue
        value = metric.get(component)
        if isinstance(value, (int, float)) and math.isfinite(value):
            clipped = min(max(float(value), 0.0), 1.0)
            values.append(clipped)
            reasons.append(
                {
                    "capability_key": key,
                    "status": "used",
                    "component": component,
                    "value": clipped,
                    "attempts": metric.get("attempts"),
                    "pass_rate": metric.get("pass_rate"),
                }
            )
        else:
            reasons.append({"capability_key": key, "status": "unavailable"})
    return (sum(values) / len(values) if values else 0.0), reasons


def _weighted_choice(rng: random.Random, positions: Sequence[int], weights: Sequence[float]) -> int:
    total = sum(weights)
    if total <= 0:
        return positions[rng.randrange(len(positions))]
    threshold = rng.random() * total
    cumulative = 0.0
    for position, weight in zip(positions, weights):
        cumulative += weight
        if threshold <= cumulative:
            return position
    return positions[-1]


def _budget_error(
    chars: int,
    tokens: int,
    target_chars: Optional[int],
    target_tokens: Optional[int],
) -> float:
    error = 0.0
    if target_chars is not None:
        error += abs(chars - target_chars) / max(target_chars, 1)
    if target_tokens is not None:
        error += abs(tokens - target_tokens) / max(target_tokens, 1)
    return error


def select_curriculum(
    capability_map: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    sample_count: int,
    seed: int,
    target_character_budget: Optional[int] = None,
    target_token_budget: Optional[int] = None,
    scoring_component: str = "frontier_score",
    exploration_floor: float = 0.10,
    repeat_cap: int = 1,
    group_cap: Optional[int] = None,
    budget_tolerance: float = 0.02,
    trials: int = 256,
) -> Dict[str, Any]:
    """Select tiny or large candidate lists without materializing training data."""

    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if not 0.0 <= exploration_floor <= 1.0:
        raise ValueError("exploration_floor must be in [0, 1]")
    if repeat_cap < 1:
        raise ValueError("repeat_cap must be at least 1")
    if sample_count > len(candidates) * repeat_cap:
        raise ValueError("repeat_cap makes sample_count infeasible")
    if not candidates and sample_count:
        raise ValueError("candidates cannot be empty")

    prepared = []
    for index, candidate in enumerate(candidates):
        task_id = candidate.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"candidates[{index}].task_id must be a non-empty string")
        chars = candidate.get("character_count", 0)
        tokens = candidate.get("token_count", 0)
        if not isinstance(chars, int) or chars < 0:
            raise ValueError(f"candidates[{index}].character_count must be non-negative")
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError(f"candidates[{index}].token_count must be non-negative")
        signal, reasons = _candidate_signal(candidate, capability_map, scoring_component)
        effective_priority = exploration_floor + (1.0 - exploration_floor) * signal
        prepared.append(
            {
                "index": index,
                "task_id": task_id,
                "character_count": chars,
                "token_count": tokens,
                "group": str(candidate.get("diversity_group", task_id)),
                "signal": signal,
                "effective_priority": effective_priority,
                "reasons": reasons,
            }
        )

    target_char_average = (
        target_character_budget / sample_count
        if target_character_budget is not None and sample_count
        else None
    )
    target_token_average = (
        target_token_budget / sample_count
        if target_token_budget is not None and sample_count
        else None
    )

    best = None
    for trial in range(max(trials, 1)):
        rng = random.Random(seed + 104729 * trial)
        repeats = Counter()
        groups = Counter()
        chosen = []

        for _ in range(sample_count):
            feasible = []
            weights = []
            for position, candidate in enumerate(prepared):
                if repeats[candidate["task_id"]] >= repeat_cap:
                    continue
                if group_cap is not None and groups[candidate["group"]] >= group_cap:
                    continue
                cost_factor = 1.0
                if target_char_average is not None:
                    cost_factor *= math.exp(
                        -abs(candidate["character_count"] - target_char_average)
                        / max(target_char_average, 1.0)
                    )
                if target_token_average is not None:
                    cost_factor *= math.exp(
                        -abs(candidate["token_count"] - target_token_average)
                        / max(target_token_average, 1.0)
                    )
                feasible.append(position)
                weights.append(max(candidate["effective_priority"] * cost_factor, 1e-12))
            if not feasible:
                raise ValueError("group_cap/repeat_cap makes selection infeasible")
            position = _weighted_choice(rng, feasible, weights)
            candidate = prepared[position]
            chosen.append(position)
            repeats[candidate["task_id"]] += 1
            groups[candidate["group"]] += 1

        chars = sum(prepared[position]["character_count"] for position in chosen)
        tokens = sum(prepared[position]["token_count"] for position in chosen)
        error = _budget_error(chars, tokens, target_character_budget, target_token_budget)
        mean_priority = (
            sum(prepared[position]["effective_priority"] for position in chosen)
            / sample_count
            if sample_count
            else 0.0
        )
        objective = (error, -mean_priority, tuple(prepared[position]["task_id"] for position in chosen))
        if best is None or objective < best["objective"]:
            best = {
                "objective": objective,
                "chosen": chosen,
                "chars": chars,
                "tokens": tokens,
                "mean_priority": mean_priority,
                "trial": trial,
                "repeats": repeats,
                "groups": groups,
            }

    assert best is not None
    selected = []
    seen = Counter()
    for draw_index, position in enumerate(best["chosen"]):
        candidate = prepared[position]
        seen[candidate["task_id"]] += 1
        selected.append(
            {
                "draw_index": draw_index,
                "task_id": candidate["task_id"],
                "source_index": candidate["index"],
                "repeat_ordinal": seen[candidate["task_id"]],
                "character_count": candidate["character_count"],
                "token_count": candidate["token_count"],
                "diversity_group": candidate["group"],
                "priority": candidate["effective_priority"],
                "priority_reason": {
                    "scoring_component": scoring_component,
                    "capability_signal": candidate["signal"],
                    "uniform_exploration_floor": exploration_floor,
                    "capabilities": candidate["reasons"],
                },
            }
        )

    char_ratio = (
        best["chars"] / target_character_budget
        if target_character_budget is not None and target_character_budget
        else None
    )
    token_ratio = (
        best["tokens"] / target_token_budget
        if target_token_budget is not None and target_token_budget
        else None
    )
    within_tolerance = all(
        abs(ratio - 1.0) <= budget_tolerance
        for ratio in (char_ratio, token_ratio)
        if ratio is not None
    )

    return {
        "schema_version": "graph_frontier_selection_v1",
        "seed": seed,
        "sample_count": sample_count,
        "scoring_component": scoring_component,
        "exploration_floor": exploration_floor,
        "repeat_cap": repeat_cap,
        "group_cap": group_cap,
        "selected": selected,
        "report": {
            "candidate_count": len(candidates),
            "unique_task_count": len(best["repeats"]),
            "duplicate_draws": sample_count - len(best["repeats"]),
            "max_observed_repeat": max(best["repeats"].values(), default=0),
            "diversity_group_counts": dict(sorted(best["groups"].items())),
            "character_budget": {
                "target": target_character_budget,
                "selected": best["chars"],
                "ratio": char_ratio,
            },
            "token_budget": {
                "target": target_token_budget,
                "selected": best["tokens"],
                "ratio": token_ratio,
            },
            "budget_tolerance": budget_tolerance,
            "within_budget_tolerance": within_tolerance,
            "mean_priority": best["mean_priority"],
            "chosen_trial": best["trial"],
        },
    }
