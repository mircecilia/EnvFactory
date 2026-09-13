"""Join a gold sidecar and typed rollout into the existing profiler input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .adapters import UNKNOWN
from .profiler import profile_rollout
from .rollout_trace import load_rollout_trace


def _load(source: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    with Path(source).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _known_bool(value: Any) -> Any:
    return value if isinstance(value, bool) else UNKNOWN


def sidecar_and_trace_to_bundle(
    gold_sidecar: str | Path | Mapping[str, Any],
    rollout_trace: str | Path | Mapping[str, Any],
) -> Dict[str, Any]:
    """Create a graph-grounded normalized bundle; never infer unknown fields."""

    sidecar = _load(gold_sidecar)
    trace = load_rollout_trace(rollout_trace)
    if sidecar.get("schema_version") != "envfactory_gold_sidecar_v1":
        raise ValueError("unsupported gold sidecar schema_version")
    if trace.get("schema_version") != "envfactory_rollout_trace_v1":
        raise ValueError("unsupported rollout trace schema_version")
    if sidecar.get("task_id") != trace.get("task_id"):
        raise ValueError(f"task_id mismatch: {sidecar.get('task_id')!r} != {trace.get('task_id')!r}")

    expected_tools = [
        node["tool_name"]
        for node in sidecar.get("required_tool_nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("tool_name"), str)
    ]
    edges = []
    for edge in sidecar.get("dependency_edges", []) or []:
        producer = edge.get("producer_tool") or {}
        consumer = edge.get("consumer_tool") or {}
        source = edge.get("producer_output_parameter") or {}
        target = edge.get("consumer_input_parameter") or {}
        edges.append(
            {
                "edge_id": edge.get("edge_id", UNKNOWN),
                "producer_tool": producer.get("tool_name", UNKNOWN),
                "consumer_tool": consumer.get("tool_name", UNKNOWN),
                "edge_type": edge.get("edge_type", UNKNOWN),
                "source_parameter": source.get("parameter_name", UNKNOWN),
                "target_parameter": target.get("parameter_name", UNKNOWN),
                "required": edge.get("required", UNKNOWN),
                "internal_parameter": (
                    not target.get("user_provided")
                    if isinstance(target.get("user_provided"), bool)
                    else UNKNOWN
                ),
                "dependency_depth": edge.get("dependency_depth", UNKNOWN),
            }
        )

    events = []
    for event in trace.get("events", []) or []:
        returned_fields = event.get("returned_fields", UNKNOWN)
        response = event.get("tool_response", UNKNOWN)
        if returned_fields == UNKNOWN and isinstance(response, (Mapping, list)):
            returned_fields = response
        events.append(
            {
                "step_index": event.get("step_index", 0),
                "tool_name": event.get("tool_name", UNKNOWN),
                "arguments": event.get("tool_arguments", UNKNOWN),
                "result": returned_fields,
                "execution_success": _known_bool(event.get("execution_success", UNKNOWN)),
                "state_changing": UNKNOWN,
                "exception": event.get("exception", UNKNOWN),
                "timestamp": event.get("timestamp", UNKNOWN),
            }
        )

    return {
        "schema_version": "graph_frontier_rollout_v1",
        "task_id": sidecar["task_id"],
        "terminal_success": _known_bool(trace.get("terminal_success", UNKNOWN)),
        "state_success": _known_bool(trace.get("verifier_result", UNKNOWN)),
        "hint_condition": UNKNOWN,
        "expected_tool_nodes": expected_tools,
        "dependency_edges": edges,
        "tool_metadata": {name: {"state_changing": UNKNOWN} for name in expected_tools},
        "events": events,
        "initial_state": trace.get("initial_environment_state", sidecar.get("initial_scenario", UNKNOWN)),
        "final_state": trace.get("final_environment_state", UNKNOWN),
        "expected_final_state": sidecar.get("expected_final_scenario", UNKNOWN),
        "adapter_notes": [
            "Joined envfactory_gold_sidecar_v1 with envfactory_rollout_trace_v1.",
            "No success, verifier, response field, or graph field was inferred from human-readable text.",
        ],
    }


def profile_sidecar_and_trace(
    gold_sidecar: str | Path | Mapping[str, Any],
    rollout_trace: str | Path | Mapping[str, Any],
) -> Dict[str, Any]:
    return profile_rollout(sidecar_and_trace_to_bundle(gold_sidecar, rollout_trace))
