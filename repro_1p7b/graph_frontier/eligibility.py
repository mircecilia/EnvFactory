"""Probe acceptance gates for Graph-Frontier v1 sidecars."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .adapters import UNKNOWN


def _available(value: Any) -> bool:
    return value is not None and value != UNKNOWN


def evaluate_probe_eligibility(sidecar: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate structural and state-diagnosis readiness without guessing."""

    resolution = sidecar.get("dependency_resolution") or {}
    selected = resolution.get("selected_dependency_count")
    resolved = resolution.get("resolved_dependency_count")
    unresolved = resolution.get("unresolved_dependencies") or []
    cross_turn = sidecar.get("cross_turn_dependencies") or []

    checks = {
        "selected_dependency_trace_present": (
            sidecar.get("selected_dependency_trace_present") is True
        ),
        "all_selected_dependencies_resolved": (
            resolution.get("status") == "resolved"
            and isinstance(selected, int)
            and isinstance(resolved, int)
            and selected == resolved
            and not unresolved
        ),
        "single_turn_structural_dependencies": not cross_turn,
        "raw_tool_call_nonempty": bool(sidecar.get("gold_tool_sequence")),
        "initial_state_available": _available(
            sidecar.get("initial_scenario", UNKNOWN)
        ),
        "expected_final_state_available": _available(
            sidecar.get(
                "expected_final_state",
                sidecar.get("expected_final_scenario", UNKNOWN),
            )
        ),
        "selected_reference_semantics": (
            sidecar.get("dependency_semantics") == "selected_reference"
        ),
    }

    reason_by_check = {
        "selected_dependency_trace_present": "selected_dependency_trace_missing",
        "all_selected_dependencies_resolved": "selected_dependency_resolution_incomplete",
        "single_turn_structural_dependencies": "cross_turn_dependency_not_supported_v1",
        "raw_tool_call_nonempty": "raw_tool_call_empty",
        "initial_state_available": "initial_state_unavailable",
        "selected_reference_semantics": "dependency_semantics_not_selected_reference",
    }
    structural_checks = tuple(reason_by_check)
    reasons = [
        reason_by_check[name]
        for name in structural_checks
        if checks[name] is not True
    ]
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "checks": checks,
        "state_diagnosis_available": checks[
            "expected_final_state_available"
        ],
    }


def is_structurally_diagnosable(sidecar: Mapping[str, Any]) -> bool:
    return bool(evaluate_probe_eligibility(sidecar)["eligible"])
