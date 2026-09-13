"""Purely programmatic dependency and root-cause profiling."""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .adapters import UNKNOWN


def _known_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return repr(value)


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _path_values(value: Any, path: Any) -> Any:
    """Return all values at a dotted path, traversing lists without guessing keys."""

    if not isinstance(path, str) or not path or path == UNKNOWN:
        return UNKNOWN
    current = [_maybe_json(value)]
    for component in path.split("."):
        next_values = []
        for item in current:
            if isinstance(item, Mapping) and component in item:
                next_values.append(item[component])
            elif isinstance(item, list):
                for child in item:
                    if isinstance(child, Mapping) and component in child:
                        next_values.append(child[component])
        if not next_values:
            return UNKNOWN
        current = next_values

    flattened = []
    for item in current:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _values_match(source_values: Any, target_values: Any) -> Any:
    if source_values == UNKNOWN or target_values == UNKNOWN:
        return UNKNOWN
    source = {_canonical(value) for value in source_values}
    target = {_canonical(value) for value in target_values}
    return bool(source.intersection(target))


def _graph_depths(expected_tools: Any, edges: Any) -> Tuple[Dict[str, int], Dict[str, int]]:
    if expected_tools == UNKNOWN or edges == UNKNOWN:
        return {}, {}
    nodes = list(expected_tools)
    depths = {name: 0 for name in nodes}
    for _ in range(max(len(nodes) - 1, 0)):
        changed = False
        for edge in edges:
            producer = edge["producer_tool"]
            consumer = edge["consumer_tool"]
            if producer not in depths or consumer not in depths:
                continue
            proposed = depths[producer] + 1
            if proposed > depths[consumer]:
                depths[consumer] = proposed
                changed = True
        if not changed:
            break
    edge_depths = {
        edge["edge_id"]: depths.get(edge["consumer_tool"], 0)
        for edge in edges
    }
    return depths, edge_depths


def _root_sort_key(root: Mapping[str, Any]) -> Tuple[int, str]:
    step = root.get("step_index", UNKNOWN)
    return (step if isinstance(step, int) else 10**12, str(root.get("failure_id", "")))


def profile_rollout(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """Profile one normalized rollout without LLM calls or response-text heuristics."""

    task_id = str(bundle.get("task_id", UNKNOWN))
    expected_tools = bundle.get("expected_tool_nodes", UNKNOWN)
    edges = bundle.get("dependency_edges", UNKNOWN)
    tool_metadata = bundle.get("tool_metadata") or {}

    events = []
    for call_index, raw_event in enumerate(
        sorted(bundle.get("events") or [], key=lambda event: event.get("step_index", 0))
    ):
        event = deepcopy(dict(raw_event))
        event["call_index"] = call_index
        event.setdefault("arguments", {})
        event.setdefault("result", UNKNOWN)
        event.setdefault("execution_success", UNKNOWN)
        if event.get("state_changing", UNKNOWN) == UNKNOWN:
            metadata = tool_metadata.get(event.get("tool_name"), {})
            event["state_changing"] = metadata.get("state_changing", UNKNOWN)
        if expected_tools == UNKNOWN:
            event["expected"] = UNKNOWN
        else:
            event["expected"] = event.get("tool_name") in set(expected_tools)
        events.append(event)

    redundant = []
    first_seen = {}
    for event in events:
        signature = (event.get("tool_name"), _canonical(event.get("arguments")))
        if signature in first_seen:
            redundant.append(
                {
                    "call_index": event["call_index"],
                    "step_index": event["step_index"],
                    "tool_name": event["tool_name"],
                    "duplicates_call_index": first_seen[signature],
                    "reason": "exact_same_tool_and_arguments",
                }
            )
        else:
            first_seen[signature] = event["call_index"]

    if expected_tools == UNKNOWN:
        extra_tools: Any = UNKNOWN
        missing_tools: Any = UNKNOWN
    else:
        expected_set = set(expected_tools)
        extra_tools = [
            {
                "call_index": event["call_index"],
                "step_index": event["step_index"],
                "tool_name": event["tool_name"],
            }
            for event in events
            if event["tool_name"] not in expected_set
        ]
        called = {event["tool_name"] for event in events}
        missing_tools = [name for name in expected_tools if name not in called]

    roots: List[Dict[str, Any]] = []
    propagated: List[Dict[str, Any]] = []
    event_root: Dict[int, str] = {}
    root_by_id: Dict[str, Dict[str, Any]] = {}

    def add_root(
        failure_type: str,
        step_index: Any,
        tool_name: Any = UNKNOWN,
        edge_id: Any = UNKNOWN,
        detail: str = "",
    ) -> str:
        failure_id = f"root-{len(roots)}:{failure_type}"
        root = {
            "failure_id": failure_id,
            "type": failure_type,
            "step_index": step_index if isinstance(step_index, int) else UNKNOWN,
            "tool_name": tool_name,
            "edge_id": edge_id,
            "detail": detail,
        }
        roots.append(root)
        root_by_id[failure_id] = root
        return failure_id

    node_depths, edge_depths = _graph_depths(expected_tools, edges)
    edge_checks: List[Dict[str, Any]] = []
    checked_edges = set()

    if edges != UNKNOWN:
        incoming = defaultdict(list)
        for edge in edges:
            incoming[edge["consumer_tool"]].append(edge)

        events_by_tool = defaultdict(list)
        for event in events:
            events_by_tool[event["tool_name"]].append(event)

        for event in events:
            inherited_roots = []
            for edge in incoming.get(event["tool_name"], []):
                edge_id = edge["edge_id"]
                if edge_id in checked_edges:
                    continue
                checked_edges.add(edge_id)
                producers = [
                    producer
                    for producer in events_by_tool.get(edge["producer_tool"], [])
                    if producer["step_index"] < event["step_index"]
                ]
                check = {
                    "edge_id": edge_id,
                    "edge_type": edge.get("edge_type", UNKNOWN),
                    "producer_tool": edge["producer_tool"],
                    "consumer_tool": edge["consumer_tool"],
                    "producer_step": UNKNOWN,
                    "consumer_step": event["step_index"],
                    "dependency_depth": edge_depths.get(edge_id, UNKNOWN),
                    "required": edge.get("required", UNKNOWN),
                    "internal_parameter": edge.get("internal_parameter", UNKNOWN),
                    "status": UNKNOWN,
                    "success": UNKNOWN,
                    "root_cause": False,
                    "propagated_from": UNKNOWN,
                    "source_value_available": UNKNOWN,
                    "target_value_available": UNKNOWN,
                    "source_values": UNKNOWN,
                    "target_values": UNKNOWN,
                }

                if not producers:
                    if edge.get("required") is False:
                        check.update(status="optional_not_observed", success=UNKNOWN)
                    else:
                        root_id = add_root(
                            "missing_producer",
                            event["step_index"],
                            event["tool_name"],
                            edge_id,
                            f"{edge['producer_tool']} was not called before {edge['consumer_tool']}.",
                        )
                        inherited_roots.append(root_id)
                        check.update(
                            status="failed_missing_producer",
                            success=False,
                            root_cause=True,
                        )
                else:
                    producer = producers[-1]
                    check["producer_step"] = producer["step_index"]
                    producer_root = event_root.get(producer["call_index"])
                    if producer_root:
                        inherited_roots.append(producer_root)
                        check.update(
                            status="propagated_upstream_failure",
                            success=False,
                            propagated_from=producer_root,
                        )
                        propagated.append(
                            {
                                "type": "dependency_edge_blocked",
                                "step_index": event["step_index"],
                                "tool_name": event["tool_name"],
                                "edge_id": edge_id,
                                "propagated_from": producer_root,
                            }
                        )
                    elif producer.get("execution_success") is False:
                        producer_root = add_root(
                            "tool_execution_failure",
                            producer["step_index"],
                            producer["tool_name"],
                            UNKNOWN,
                            "Producer tool explicitly failed execution.",
                        )
                        event_root[producer["call_index"]] = producer_root
                        inherited_roots.append(producer_root)
                        check.update(
                            status="propagated_upstream_failure",
                            success=False,
                            propagated_from=producer_root,
                        )
                        propagated.append(
                            {
                                "type": "dependency_edge_blocked",
                                "step_index": event["step_index"],
                                "tool_name": event["tool_name"],
                                "edge_id": edge_id,
                                "propagated_from": producer_root,
                            }
                        )
                    else:
                        source_path = edge.get("source_parameter", UNKNOWN)
                        target_path = edge.get("target_parameter", UNKNOWN)
                        if source_path == UNKNOWN or target_path == UNKNOWN:
                            check.update(status="order_satisfied", success=True)
                        else:
                            source_values = _path_values(producer.get("result"), source_path)
                            target_values = _path_values(event.get("arguments"), target_path)
                            check["source_value_available"] = source_values != UNKNOWN
                            check["target_value_available"] = target_values != UNKNOWN
                            check["source_values"] = deepcopy(source_values)
                            check["target_values"] = deepcopy(target_values)
                            match = _values_match(source_values, target_values)
                            if match is True:
                                check.update(status="satisfied", success=True)
                            elif match is False:
                                root_id = add_root(
                                    "wrong_propagated_value",
                                    event["step_index"],
                                    event["tool_name"],
                                    edge_id,
                                    "Consumer argument did not equal the grounded producer result.",
                                )
                                inherited_roots.append(root_id)
                                check.update(
                                    status="failed_wrong_value",
                                    success=False,
                                    root_cause=True,
                                )
                            else:
                                check.update(status=UNKNOWN, success=UNKNOWN)

                edge_checks.append(check)

            if inherited_roots:
                inherited_roots.sort(
                    key=lambda root_id: _root_sort_key(root_by_id[root_id])
                )
                event_root[event["call_index"]] = inherited_roots[0]

            if event.get("execution_success") is False:
                inherited = event_root.get(event["call_index"])
                if inherited:
                    propagated.append(
                        {
                            "type": "tool_execution_failure",
                            "step_index": event["step_index"],
                            "tool_name": event["tool_name"],
                            "edge_id": UNKNOWN,
                            "propagated_from": inherited,
                        }
                    )
                else:
                    event_root[event["call_index"]] = add_root(
                        "tool_execution_failure",
                        event["step_index"],
                        event["tool_name"],
                        UNKNOWN,
                        "Tool explicitly failed execution with no earlier dependency failure.",
                    )

        for edge in edges:
            if edge["edge_id"] in checked_edges:
                continue
            edge_checks.append(
                {
                    "edge_id": edge["edge_id"],
                    "edge_type": edge.get("edge_type", UNKNOWN),
                    "producer_tool": edge["producer_tool"],
                    "consumer_tool": edge["consumer_tool"],
                    "producer_step": UNKNOWN,
                    "consumer_step": UNKNOWN,
                    "dependency_depth": edge_depths.get(edge["edge_id"], UNKNOWN),
                    "required": edge.get("required", UNKNOWN),
                    "internal_parameter": edge.get("internal_parameter", UNKNOWN),
                    "status": "not_attempted",
                    "success": UNKNOWN,
                    "root_cause": False,
                    "propagated_from": UNKNOWN,
                    "source_value_available": UNKNOWN,
                    "target_value_available": UNKNOWN,
                    "source_values": UNKNOWN,
                    "target_values": UNKNOWN,
                }
            )
    else:
        for event in events:
            if event.get("execution_success") is False:
                event_root[event["call_index"]] = add_root(
                    "tool_execution_failure",
                    event["step_index"],
                    event["tool_name"],
                    UNKNOWN,
                    "Typed executor status reports failure; graph dependencies are unavailable.",
                )

    if missing_tools != UNKNOWN and missing_tools:
        if roots:
            earliest = sorted(roots, key=_root_sort_key)[0]["failure_id"]
            for tool_name in missing_tools:
                propagated.append(
                    {
                        "type": "missing_expected_tool",
                        "step_index": UNKNOWN,
                        "tool_name": tool_name,
                        "edge_id": UNKNOWN,
                        "propagated_from": earliest,
                    }
                )
        else:
            root_id = add_root(
                "missing_expected_tool",
                UNKNOWN,
                missing_tools[0],
                UNKNOWN,
                "Expected tool was never called; no concrete failure step is available.",
            )
            for tool_name in missing_tools[1:]:
                propagated.append(
                    {
                        "type": "missing_expected_tool",
                        "step_index": UNKNOWN,
                        "tool_name": tool_name,
                        "edge_id": UNKNOWN,
                        "propagated_from": root_id,
                    }
                )

    state_success = bundle.get("state_success", UNKNOWN)
    if state_success is False:
        if roots:
            earliest = sorted(roots, key=_root_sort_key)[0]["failure_id"]
            propagated.append(
                {
                    "type": "final_state_failure",
                    "step_index": UNKNOWN,
                    "tool_name": UNKNOWN,
                    "edge_id": UNKNOWN,
                    "propagated_from": earliest,
                }
            )
        else:
            add_root(
                "final_state_failure",
                UNKNOWN,
                UNKNOWN,
                UNKNOWN,
                "Explicit final-state verifier failed after no earlier grounded failure.",
            )

    roots.sort(key=_root_sort_key)
    edge_checks.sort(
        key=lambda check: (
            check["consumer_step"] if isinstance(check["consumer_step"], int) else 10**12,
            check["edge_id"],
        )
    )
    failed_edges = [check for check in edge_checks if check["root_cause"]]
    first_failed_edge: Any = deepcopy(failed_edges[0]) if failed_edges else UNKNOWN
    first_failure_step: Any = roots[0]["step_index"] if roots else UNKNOWN

    if edges == UNKNOWN:
        max_depth_reached: Any = UNKNOWN
        expected_edges: Any = UNKNOWN
    else:
        expected_edges = deepcopy(list(edges))
        reached = []
        for event in events:
            if event.get("execution_success") is not True:
                continue
            if event["call_index"] in event_root:
                continue
            name = event["tool_name"]
            if name in node_depths:
                incoming_checks = [
                    check for check in edge_checks
                    if check["consumer_tool"] == name and check["consumer_step"] == event["step_index"]
                ]
                if all(check["success"] is True for check in incoming_checks):
                    reached.append(node_depths[name])
        max_depth_reached = max(reached) if reached else UNKNOWN

    satisfied_edges = [
        check["edge_id"] for check in edge_checks if check["success"] is True
    ]
    internal_checks = [
        deepcopy(check)
        for check in edge_checks
        if check.get("internal_parameter") is True
    ]

    return {
        "schema_version": "graph_frontier_profile_v1",
        "task_id": task_id,
        "terminal_success": bundle.get("terminal_success", UNKNOWN),
        "state_success": state_success,
        "hint_condition": bundle.get("hint_condition", UNKNOWN),
        "expected_tool_nodes": deepcopy(expected_tools),
        "actual_tool_nodes": events,
        "expected_dependency_edges": expected_edges,
        "satisfied_dependency_edges": satisfied_edges,
        "dependency_edge_checks": edge_checks,
        "internal_parameter_flow_checks": internal_checks,
        "first_failed_edge": first_failed_edge,
        "first_failure_step": first_failure_step,
        "max_dependency_depth_reached": max_depth_reached,
        "redundant_tool_calls": redundant,
        "extra_tool_calls": extra_tools,
        "missing_expected_tools": missing_tools,
        "root_cause_failures": roots,
        "downstream_propagated_failures": propagated,
        "adapter_notes": list(bundle.get("adapter_notes") or []),
    }
