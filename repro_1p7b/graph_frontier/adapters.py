"""Adapters from EnvFactory objects/artifacts to the profiler's normalized bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

UNKNOWN = "unknown"


def _edge_type(data: Mapping[str, Any]) -> str:
    value = data.get("edge_type", UNKNOWN)
    if hasattr(value, "value"):
        value = value.value
    return str(value)


def _is_tool(node: Any) -> bool:
    return hasattr(node, "name") and hasattr(node, "input_schema") and hasattr(node, "output_schema")


def _is_parameter(node: Any) -> bool:
    return hasattr(node, "name") and not _is_tool(node)


def _name(item: Any) -> str:
    return str(item if isinstance(item, str) else item.name)


def _known_flag(value: Any) -> Any:
    return value if isinstance(value, bool) else UNKNOWN


def _parameter_user_provided(parameter: Any) -> Any:
    try:
        return _known_flag(parameter.user_provided)
    except Exception:
        return UNKNOWN


def _iter_nodes(graph: Any) -> List[Any]:
    nodes = graph.nodes
    return list(nodes() if callable(nodes) else nodes)


def _iter_successors(graph: Any, node: Any) -> List[Any]:
    return list(graph.successors(node))


def _iter_predecessors(graph: Any, node: Any) -> List[Any]:
    return list(graph.predecessors(node))


def _edge_id(
    producer: str,
    consumer: str,
    source_parameter: Any,
    target_parameter: Any,
    edge_type: str,
) -> str:
    source = source_parameter if isinstance(source_parameter, str) else "unknown"
    target = target_parameter if isinstance(target_parameter, str) else "unknown"
    return f"{producer}->{consumer}:{source}->{target}:{edge_type}"


def tool_graph_to_spec(
    tool_graph: Any,
    expected_tools: Optional[Sequence[Any]] = None,
    task_id: str = UNKNOWN,
) -> Dict[str, Any]:
    """Project an official EnvFactory ToolGraph using its existing nodes and edges.

    The function intentionally uses duck typing so importing it does not import the
    EnvFactory LLM/embedding stack. Pass the live ToolGraph object at generation time.
    """

    graph = tool_graph.graph
    nodes = _iter_nodes(graph)
    tools = [node for node in nodes if _is_tool(node)]
    tools_by_name = {_name(tool): tool for tool in tools}

    if expected_tools is None:
        expected_names = sorted(tools_by_name)
    else:
        expected_names = [_name(tool) for tool in expected_tools]
    expected_set = set(expected_names)

    edges: List[Dict[str, Any]] = []
    seen_edge_ids = set()
    parameter_pairs = set()

    for consumer_name in expected_names:
        consumer = tools_by_name.get(consumer_name)
        if consumer is None:
            continue
        for input_parameter in _iter_predecessors(graph, consumer):
            if not _is_parameter(input_parameter):
                continue
            input_edge = graph.get_edge_data(input_parameter, consumer) or {}
            if _edge_type(input_edge) != "parameter_to_tool":
                continue

            target_parameter = _name(input_parameter)
            required = _known_flag(input_edge.get("required"))
            user_provided = _parameter_user_provided(input_parameter)
            internal_parameter = (not user_provided) if isinstance(user_provided, bool) else UNKNOWN

            sources = []
            for predecessor in _iter_predecessors(graph, input_parameter):
                predecessor_edge = graph.get_edge_data(predecessor, input_parameter) or {}
                predecessor_type = _edge_type(predecessor_edge)
                if _is_tool(predecessor) and predecessor_type == "tool_to_parameter":
                    sources.append((predecessor, input_parameter, "direct_output"))
                elif _is_parameter(predecessor) and predecessor_type == "parameter_to_parameter":
                    for producer in _iter_predecessors(graph, predecessor):
                        producer_edge = graph.get_edge_data(producer, predecessor) or {}
                        if _is_tool(producer) and _edge_type(producer_edge) == "tool_to_parameter":
                            sources.append((producer, predecessor, "parameter_relation"))

            for producer, output_parameter, provenance in sources:
                producer_name = _name(producer)
                if producer_name not in expected_set:
                    continue
                source_parameter = _name(output_parameter)
                edge_type = "parameter_flow"
                edge_id = _edge_id(
                    producer_name,
                    consumer_name,
                    source_parameter,
                    target_parameter,
                    edge_type,
                )
                if edge_id in seen_edge_ids:
                    continue
                seen_edge_ids.add(edge_id)
                parameter_pairs.add((producer_name, consumer_name))
                edges.append(
                    {
                        "edge_id": edge_id,
                        "producer_tool": producer_name,
                        "consumer_tool": consumer_name,
                        "edge_type": edge_type,
                        "graph_provenance": provenance,
                        "source_parameter": source_parameter,
                        "target_parameter": target_parameter,
                        "required": required,
                        "internal_parameter": internal_parameter,
                    }
                )

    for producer_name in expected_names:
        producer = tools_by_name.get(producer_name)
        if producer is None:
            continue
        for consumer in _iter_successors(graph, producer):
            if not _is_tool(consumer):
                continue
            consumer_name = _name(consumer)
            if consumer_name not in expected_set:
                continue
            edge_data = graph.get_edge_data(producer, consumer) or {}
            if _edge_type(edge_data) != "tool_to_tool":
                continue
            if (producer_name, consumer_name) in parameter_pairs:
                continue
            edge_type = "tool_depend"
            edge_id = _edge_id(producer_name, consumer_name, UNKNOWN, UNKNOWN, edge_type)
            if edge_id in seen_edge_ids:
                continue
            seen_edge_ids.add(edge_id)
            edges.append(
                {
                    "edge_id": edge_id,
                    "producer_tool": producer_name,
                    "consumer_tool": consumer_name,
                    "edge_type": edge_type,
                    "graph_provenance": "tool_depend",
                    "source_parameter": UNKNOWN,
                    "target_parameter": UNKNOWN,
                    "required": UNKNOWN,
                    "internal_parameter": UNKNOWN,
                }
            )

    edges.sort(key=lambda edge: edge["edge_id"])
    return {
        "schema_version": "graph_frontier_graph_v1",
        "task_id": task_id,
        "expected_tool_nodes": expected_names,
        "dependency_edges": edges,
        "tool_metadata": {
            name: {"state_changing": UNKNOWN}
            for name in expected_names
        },
        "adapter_notes": [
            "Projected from the live EnvFactory ToolGraph without rebuilding dependencies.",
            "state_changing is unknown because Tool metadata has no reliable typed field.",
        ],
    }


def _load_artifact(source: Any) -> Dict[str, Any]:
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if isinstance(source, Mapping):
        return dict(source)
    raise TypeError("artifact must be a mapping or JSON path")


def _as_tool_calls(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, Mapping):
        return [dict(content)]
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return []
        return _as_tool_calls(parsed)
    if isinstance(content, list):
        result = []
        for item in content:
            if isinstance(item, Mapping):
                result.append(dict(item))
            elif isinstance(item, str):
                result.extend(_as_tool_calls(item))
        return result
    return []


def _response_items(content: Any) -> List[Any]:
    return content if isinstance(content, list) else [content]


def querygen_artifact_to_bundle(
    source: Any,
    node_index: int = 0,
    pass_k: Optional[int] = None,
    graph_spec: Optional[Mapping[str, Any]] = None,
    task_id: Optional[str] = None,
    terminal_success: Any = UNKNOWN,
    state_success: Any = UNKNOWN,
) -> Dict[str, Any]:
    """Adapt saved ToolQueryChain JSON without inventing missing correctness labels."""

    artifact = _load_artifact(source)
    nodes = artifact.get("nodes") or []
    if not (0 <= node_index < len(nodes)):
        raise IndexError(f"node_index {node_index} is outside {len(nodes)} saved nodes")
    node = nodes[node_index]

    if pass_k is None:
        steps = node.get("steps") or []
        trace_source = "selected steps"
    else:
        traces = node.get("pass_k_trace") or {}
        steps = traces.get(pass_k, traces.get(str(pass_k), [])) or []
        trace_source = f"pass_k_trace[{pass_k}]"

    events: List[Dict[str, Any]] = []
    for position, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("role") != "tool_call":
            continue
        calls = _as_tool_calls(step.get("content"))
        responses: List[Any] = []
        if position + 1 < len(steps):
            response_step = steps[position + 1]
            if isinstance(response_step, Mapping) and response_step.get("role") == "tool_response":
                responses = _response_items(response_step.get("content"))

        for offset, call in enumerate(calls):
            name = call.get("name")
            if not isinstance(name, str):
                continue
            events.append(
                {
                    "step_index": position,
                    "tool_name": name,
                    "arguments": call.get("arguments", {}),
                    "result": responses[offset] if offset < len(responses) else UNKNOWN,
                    "execution_success": UNKNOWN,
                    "state_changing": UNKNOWN,
                }
            )

    graph = dict(graph_spec) if graph_spec is not None else None
    inferred_task_id = task_id or f"seed-{artifact.get('seed', UNKNOWN)}-node-{node_index}"
    notes = [
        f"Adapted from QueryGen {trace_source}.",
        "execution_success is unknown because saved responses have no typed status.",
        "terminal_success/state_success are not inferred from QueryGen decision or response text.",
    ]
    if graph is None:
        expected_nodes: Any = UNKNOWN
        dependency_edges: Any = UNKNOWN
        tool_metadata: Dict[str, Any] = {}
        notes.append("No companion ToolGraph projection was supplied; expected nodes/edges are unknown.")
    else:
        expected_nodes = graph.get("expected_tool_nodes", UNKNOWN)
        dependency_edges = graph.get("dependency_edges", UNKNOWN)
        tool_metadata = graph.get("tool_metadata", {})
        notes.extend(graph.get("adapter_notes", []))

    return {
        "schema_version": "graph_frontier_rollout_v1",
        "task_id": inferred_task_id,
        "terminal_success": _known_flag(terminal_success),
        "state_success": _known_flag(state_success),
        "hint_condition": UNKNOWN,
        "expected_tool_nodes": expected_nodes,
        "dependency_edges": dependency_edges,
        "tool_metadata": tool_metadata,
        "events": events,
        "initial_state": node.get("initial_scenario", UNKNOWN),
        "final_state": node.get("final_scenario", UNKNOWN),
        "adapter_notes": notes,
    }


def normalized_rollout_bundle(
    task_id: str,
    events: Sequence[Mapping[str, Any]],
    graph_spec: Optional[Mapping[str, Any]] = None,
    terminal_success: Any = UNKNOWN,
    state_success: Any = UNKNOWN,
    adapter_notes: Optional[Iterable[str]] = None,
    hint_condition: Any = UNKNOWN,
) -> Dict[str, Any]:
    """Build the preferred typed bundle used by new executable rollout exporters."""

    normalized_events = []
    for index, event in enumerate(events):
        name = event.get("tool_name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"events[{index}].tool_name must be a non-empty string")
        step_index = event.get("step_index", index)
        if not isinstance(step_index, int) or step_index < 0:
            raise ValueError(f"events[{index}].step_index must be a non-negative integer")
        normalized_events.append(
            {
                "step_index": step_index,
                "tool_name": name,
                "arguments": event.get("arguments", {}),
                "result": event.get("result", UNKNOWN),
                "execution_success": _known_flag(event.get("execution_success", UNKNOWN)),
                "state_changing": _known_flag(event.get("state_changing", UNKNOWN)),
            }
        )

    graph = dict(graph_spec) if graph_spec is not None else {}
    return {
        "schema_version": "graph_frontier_rollout_v1",
        "task_id": task_id,
        "terminal_success": _known_flag(terminal_success),
        "state_success": _known_flag(state_success),
        "hint_condition": hint_condition if hint_condition in {"with_hint", "without_hint"} else UNKNOWN,
        "expected_tool_nodes": graph.get("expected_tool_nodes", UNKNOWN),
        "dependency_edges": graph.get("dependency_edges", UNKNOWN),
        "tool_metadata": graph.get("tool_metadata", {}),
        "events": sorted(normalized_events, key=lambda event: event["step_index"]),
        "adapter_notes": list(adapter_notes or []) + list(graph.get("adapter_notes", [])),
    }
