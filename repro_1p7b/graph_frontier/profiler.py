"""Purely programmatic dependency and root-cause profiling."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Tuple

from .adapters import UNKNOWN


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return repr(value)


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _path_values(value: Any, path: Any) -> Any:
    """Return values at a dotted path without flattening the terminal value."""

    if not isinstance(path, str) or not path or path == UNKNOWN:
        return UNKNOWN
    current = [_maybe_json(value)]
    for component in path.split("."):
        next_values = []
        for item in current:
            if isinstance(item, Mapping) and component in item:
                next_values.append(item[component])
            elif isinstance(item, list):
                next_values.extend(
                    child[component]
                    for child in item
                    if isinstance(child, Mapping) and component in child
                )
        if not next_values:
            return UNKNOWN
        current = next_values
    return current


_SCALAR_TYPES = {
    "string", "str", "integer", "int", "number", "float", "boolean", "bool"
}
_COLLECTION_TYPES = {"array", "list", "set", "tuple"}
_OBJECT_TYPES = {"object", "dict", "map"}


def _type_kind(data_type: Any) -> Any:
    if not isinstance(data_type, str) or data_type == UNKNOWN:
        return UNKNOWN
    normalized = data_type.lower().strip()
    if normalized in _SCALAR_TYPES:
        return "scalar"
    if normalized in _COLLECTION_TYPES or normalized.startswith("array"):
        return "collection"
    if normalized in _OBJECT_TYPES:
        return "object"
    return UNKNOWN


def _values_match(
    source_values: Any,
    target_values: Any,
    source_data_type: Any,
    target_data_type: Any,
    value_semantics: Any = UNKNOWN,
) -> Any:
    """Compare one typed flow conservatively; partial overlap is never enough."""

    if source_values == UNKNOWN or target_values == UNKNOWN:
        return UNKNOWN
    if len(source_values) != 1 or len(target_values) != 1:
        return UNKNOWN
    source = source_values[0]
    target = target_values[0]
    source_kind = _type_kind(source_data_type)
    target_kind = _type_kind(target_data_type)
    if source_kind == UNKNOWN or target_kind == UNKNOWN:
        return UNKNOWN
    if source_kind == "scalar" and target_kind == "scalar":
        return type(source) is type(target) and source == target
    if source_kind == "collection" and target_kind == "collection":
        if not isinstance(source, (list, tuple, set)) or not isinstance(
            target, (list, tuple, set)
        ):
            return False
        if (
            isinstance(source, set)
            or isinstance(target, set)
            or value_semantics == "unordered_multiset"
        ):
            return sorted(_canonical(item) for item in source) == sorted(
                _canonical(item) for item in target
            )
        return _canonical(list(source)) == _canonical(list(target))
    if source_kind == "object" and target_kind == "object":
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            return False
        return _canonical(source) == _canonical(target)
    if (
        source_kind == "collection"
        and target_kind == "scalar"
        and value_semantics == "member_selection"
    ):
        if not isinstance(source, (list, tuple, set)):
            return False
        return any(type(item) is type(target) and item == target for item in source)
    return UNKNOWN


def _graph_depths(
    expected_tools: Any, edges: Any
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fallback depth computation with explicit cycle detection."""

    if expected_tools == UNKNOWN or edges == UNKNOWN:
        return {}, {}
    nodes = list(dict.fromkeys(list(expected_tools)))
    for edge in edges:
        for key in ("producer_tool", "consumer_tool"):
            name = edge.get(key)
            if isinstance(name, str) and name not in nodes:
                nodes.append(name)
    successors = defaultdict(list)
    indegree = {node: 0 for node in nodes}
    for edge in edges:
        producer = edge.get("producer_tool")
        consumer = edge.get("consumer_tool")
        if producer not in indegree or consumer not in indegree:
            continue
        if consumer not in successors[producer]:
            successors[producer].append(consumer)
            indegree[consumer] += 1
    queue = deque(node for node in nodes if indegree[node] == 0)
    depths = {node: 0 for node in nodes}
    processed = 0
    while queue:
        producer = queue.popleft()
        processed += 1
        for consumer in successors[producer]:
            depths[consumer] = max(depths[consumer], depths[producer] + 1)
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                queue.append(consumer)
    if processed != len(nodes):
        return (
            {node: UNKNOWN for node in nodes},
            {edge.get("edge_id", UNKNOWN): UNKNOWN for edge in edges},
        )
    return depths, {
        edge.get("edge_id", UNKNOWN): depths.get(
            edge.get("consumer_tool"), UNKNOWN
        )
        for edge in edges
    }


def _root_sort_key(root: Mapping[str, Any]) -> Tuple[int, str]:
    step = root.get("step_index", UNKNOWN)
    return (
        step if isinstance(step, int) else 10**12,
        str(root.get("failure_id", "")),
    )


def profile_rollout(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """Profile one normalized rollout without LLM or response-text heuristics."""

    task_id = str(bundle.get("task_id", UNKNOWN))
    expected_tools = bundle.get("expected_tool_nodes", UNKNOWN)
    edges = bundle.get("dependency_edges", UNKNOWN)
    tool_metadata = bundle.get("tool_metadata") or {}

    events = []
    for call_index, raw_event in enumerate(
        sorted(
            bundle.get("events") or [],
            key=lambda event: event.get("step_index", 0),
        )
    ):
        event = deepcopy(dict(raw_event))
        event["call_index"] = call_index
        event.setdefault("arguments", {})
        event.setdefault("result", UNKNOWN)
        event.setdefault("execution_success", UNKNOWN)
        metadata = tool_metadata.get(event.get("tool_name"), {})
        if event.get("state_changing", UNKNOWN) == UNKNOWN:
            event["state_changing"] = metadata.get("state_changing", UNKNOWN)
        event["expected"] = (
            UNKNOWN
            if expected_tools == UNKNOWN
            else event.get("tool_name") in set(expected_tools)
        )
        events.append(event)

    redundant = []
    first_seen = {}
    for event in events:
        signature = (
            event.get("tool_name"),
            _canonical(event.get("arguments")),
        )
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
        extra_tools = UNKNOWN
        missing_tools = UNKNOWN
        path_adherence = UNKNOWN
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
        reference_calls = [
            event["tool_name"] for event in events if event["tool_name"] in expected_set
        ]
        expected_present = [name for name in expected_tools if name in called]
        path_adherence = (
            not extra_tools
            and not missing_tools
            and not redundant
            and reference_calls == expected_present
        )

    reference_path_divergence = []
    for tool_name in [] if missing_tools == UNKNOWN else missing_tools:
        reference_path_divergence.append(
            {"type": "missing_reference_tool", "tool_name": tool_name}
        )
    for event in [] if extra_tools == UNKNOWN else extra_tools:
        reference_path_divergence.append(
            {
                "type": "extra_nonreference_tool",
                "tool_name": event["tool_name"],
                "step_index": event["step_index"],
            }
        )
    for event in redundant:
        reference_path_divergence.append(
            {
                "type": "repeated_reference_call",
                "tool_name": event["tool_name"],
                "step_index": event["step_index"],
            }
        )

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
            "step_index": (
                step_index if isinstance(step_index, int) else UNKNOWN
            ),
            "tool_name": tool_name,
            "edge_id": edge_id,
            "detail": detail,
        }
        roots.append(root)
        root_by_id[failure_id] = root
        return failure_id

    fallback_node_depths, fallback_edge_depths = _graph_depths(
        expected_tools, edges
    )
    node_depths = {
        name: (
            tool_metadata.get(name, {}).get("dependency_depth")
            if isinstance(
                tool_metadata.get(name, {}).get("dependency_depth"), int
            )
            else fallback_node_depths.get(name, UNKNOWN)
        )
        for name in ([] if expected_tools == UNKNOWN else expected_tools)
    }
    edge_depths = {}
    if edges != UNKNOWN:
        for edge in edges:
            edge_depths[edge["edge_id"]] = (
                edge.get("dependency_depth")
                if isinstance(edge.get("dependency_depth"), int)
                else fallback_edge_depths.get(edge["edge_id"], UNKNOWN)
            )

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
                    for producer in events_by_tool.get(
                        edge["producer_tool"], []
                    )
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
                    "internal_parameter": edge.get(
                        "internal_parameter", UNKNOWN
                    ),
                    "status": UNKNOWN,
                    "success": UNKNOWN,
                    "root_cause": False,
                    "propagated_from": UNKNOWN,
                    "source_value_available": UNKNOWN,
                    "target_value_available": UNKNOWN,
                    "source_values": UNKNOWN,
                    "target_values": UNKNOWN,
                    "value_match": UNKNOWN,
                }
                if not producers:
                    if (
                        edge.get("required") is True
                        and edge.get("internal_parameter") is True
                    ):
                        root_id = add_root(
                            "missing_producer",
                            event["step_index"],
                            event["tool_name"],
                            edge_id,
                            f"{edge['producer_tool']} was not called before "
                            f"{edge['consumer_tool']}.",
                        )
                        inherited_roots.append(root_id)
                        check.update(
                            status="failed_missing_producer",
                            success=False,
                            root_cause=True,
                        )
                    elif (
                        edge.get("required") is False
                        or edge.get("internal_parameter") is False
                    ):
                        check.update(
                            status=(
                                "optional_not_observed"
                                if edge.get("required") is False
                                else "user_provided_no_producer_required"
                            ),
                            success=UNKNOWN,
                        )
                    else:
                        check.update(status=UNKNOWN, success=UNKNOWN)
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
                    else:
                        source_path = edge.get(
                            "source_parameter", UNKNOWN
                        )
                        target_path = edge.get(
                            "target_parameter", UNKNOWN
                        )
                        if source_path == UNKNOWN or target_path == UNKNOWN:
                            check.update(
                                status="order_satisfied", success=True
                            )
                        else:
                            source_values = _path_values(
                                producer.get("result"), source_path
                            )
                            target_values = _path_values(
                                event.get("arguments"), target_path
                            )
                            check["source_value_available"] = (
                                source_values != UNKNOWN
                            )
                            check["target_value_available"] = (
                                target_values != UNKNOWN
                            )
                            check["source_values"] = deepcopy(source_values)
                            check["target_values"] = deepcopy(target_values)
                            match = _values_match(
                                source_values,
                                target_values,
                                edge.get("source_data_type", UNKNOWN),
                                edge.get("target_data_type", UNKNOWN),
                                edge.get("value_semantics", UNKNOWN),
                            )
                            check["value_match"] = match
                            if match is True:
                                check.update(
                                    status="satisfied", success=True
                                )
                            elif match is False:
                                root_id = add_root(
                                    "wrong_propagated_value",
                                    event["step_index"],
                                    event["tool_name"],
                                    edge_id,
                                    "Consumer argument is not an exact typed "
                                    "match for the producer result.",
                                )
                                inherited_roots.append(root_id)
                                check.update(
                                    status="failed_wrong_value",
                                    success=False,
                                    root_cause=True,
                                )
                            else:
                                check.update(
                                    status=UNKNOWN, success=UNKNOWN
                                )
                edge_checks.append(check)

            if inherited_roots:
                inherited_roots.sort(
                    key=lambda root_id: _root_sort_key(
                        root_by_id[root_id]
                    )
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
                        "Tool explicitly failed execution with no earlier "
                        "dependency failure.",
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
                    "dependency_depth": edge_depths.get(
                        edge["edge_id"], UNKNOWN
                    ),
                    "required": edge.get("required", UNKNOWN),
                    "internal_parameter": edge.get(
                        "internal_parameter", UNKNOWN
                    ),
                    "status": "not_attempted",
                    "success": UNKNOWN,
                    "root_cause": False,
                    "propagated_from": UNKNOWN,
                    "source_value_available": UNKNOWN,
                    "target_value_available": UNKNOWN,
                    "source_values": UNKNOWN,
                    "target_values": UNKNOWN,
                    "value_match": UNKNOWN,
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
                    "Typed executor status reports failure; graph unavailable.",
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
                "Explicit or canonical final-state verification failed.",
            )

    roots.sort(key=_root_sort_key)
    edge_checks.sort(
        key=lambda check: (
            check["consumer_step"]
            if isinstance(check["consumer_step"], int)
            else 10**12,
            check["edge_id"],
        )
    )
    failed_edges = [
        check for check in edge_checks if check["root_cause"]
    ]
    first_failed_edge = (
        deepcopy(failed_edges[0]) if failed_edges else UNKNOWN
    )
    first_failure_step = roots[0]["step_index"] if roots else UNKNOWN

    if edges == UNKNOWN:
        max_depth_reached = UNKNOWN
        expected_edges = UNKNOWN
    else:
        expected_edges = deepcopy(list(edges))
        reached = []
        for event in events:
            if event.get("execution_success") is not True:
                continue
            if event["call_index"] in event_root:
                continue
            name = event["tool_name"]
            if not isinstance(node_depths.get(name), int):
                continue
            blocking_checks = [
                check
                for check in edge_checks
                if check["consumer_tool"] == name
                and check["consumer_step"] == event["step_index"]
                and check.get("required") is True
                and check.get("internal_parameter") is True
            ]
            if all(check["success"] is True for check in blocking_checks):
                reached.append(node_depths[name])
        max_depth_reached = max(reached) if reached else UNKNOWN

    satisfied_edges = [
        check["edge_id"]
        for check in edge_checks
        if check["success"] is True
    ]
    structural_roots = deepcopy(roots)
    task_roots = [] if state_success is True else deepcopy(roots)
    task_success = (
        state_success
        if isinstance(state_success, bool)
        else bundle.get("terminal_success", UNKNOWN)
    )

    return {
        "schema_version": "graph_frontier_profile_v1",
        "task_id": task_id,
        "terminal_success": bundle.get("terminal_success", UNKNOWN),
        "state_success": state_success,
        "task_success": task_success,
        "state_verification_source": bundle.get(
            "state_verification_source", UNKNOWN
        ),
        "hint_condition": bundle.get("hint_condition", UNKNOWN),
        "path_adherence": path_adherence,
        "reference_path_divergence": reference_path_divergence,
        "dependency_semantics": bundle.get(
            "dependency_semantics", UNKNOWN
        ),
        "expected_tool_nodes": deepcopy(expected_tools),
        "actual_tool_nodes": events,
        "expected_dependency_edges": expected_edges,
        "satisfied_dependency_edges": satisfied_edges,
        "dependency_edge_checks": edge_checks,
        "internal_parameter_flow_checks": [
            deepcopy(check)
            for check in edge_checks
            if check.get("internal_parameter") is True
        ],
        "first_failed_edge": first_failed_edge,
        "first_failure_step": first_failure_step,
        "max_dependency_depth_reached": max_depth_reached,
        "task_dependency_depth": bundle.get(
            "task_dependency_depth", UNKNOWN
        ),
        "redundant_tool_calls": redundant,
        "extra_tool_calls": extra_tools,
        "missing_expected_tools": missing_tools,
        "structural_root_cause_failures": structural_roots,
        "root_cause_failures": task_roots,
        "downstream_propagated_failures": propagated,
        "trace_quality_metrics": deepcopy(
            bundle.get("trace_quality_metrics", UNKNOWN)
        ),
        "adapter_notes": list(bundle.get("adapter_notes") or []),
    }
