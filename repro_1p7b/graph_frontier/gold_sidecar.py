"""Generation-time export of task-local gold graph sidecars."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .adapters import UNKNOWN, tool_graph_to_spec
from .eligibility import evaluate_probe_eligibility
from .traceable_sampler import (
    has_selected_dependency_trace,
    selected_dependency_trace,
)


def _known_bool(value: Any) -> Any:
    return value if isinstance(value, bool) else UNKNOWN


def _json_value(value: Any) -> Any:
    if value is None:
        return UNKNOWN
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return UNKNOWN
    return value


def _tool_id(tool_name: str) -> str:
    return f"tool::{tool_name}"


def _parameter_id(tool_name: str, role: str, parameter_name: Any) -> str:
    if not isinstance(parameter_name, str) or parameter_name == UNKNOWN:
        return UNKNOWN
    return f"{tool_name}::{role}::{parameter_name}"


def _server_id(tool: Any) -> Any:
    value = getattr(tool, "server", None)
    return value if isinstance(value, str) and value else UNKNOWN


def _parameter_user_provided(parameter: Any) -> Any:
    try:
        return _known_bool(parameter.user_provided)
    except Exception:
        return UNKNOWN


def _parameter_data_type(parameter: Any) -> Any:
    value = getattr(parameter, "data_type", UNKNOWN)
    return value if isinstance(value, str) and value else UNKNOWN


def _required_input(tool_graph: Any, parameter: Any, tool: Any) -> Any:
    try:
        data = tool_graph.graph.get_edge_data(parameter, tool) or {}
        if isinstance(data.get("required"), bool):
            return data["required"]
    except Exception:
        pass
    try:
        required = tool.input_schema.get("required", [])
        name = parameter.name
        base = name.split(".", 1)[0]
        return name in required or base in required
    except Exception:
        return UNKNOWN


def _task_id(seed: Any, turn_index: int, query: Any, tool_names: Sequence[str]) -> str:
    payload = json.dumps(
        {"seed": seed, "turn_index": turn_index, "query": query, "gold_tool_sequence": list(tool_names)},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return "envfactory-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _is_forward_edge(edge: Mapping[str, Any], positions: Mapping[str, List[int]]) -> bool:
    producers = positions.get(str(edge.get("producer_tool")), [])
    consumers = positions.get(str(edge.get("consumer_tool")), [])
    return any(producer < consumer for producer in producers for consumer in consumers)


def _dependency_depths(tool_names: Sequence[str], edges: Sequence[Mapping[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
    unique_names = list(dict.fromkeys(tool_names))
    successors: Dict[str, List[str]] = defaultdict(list)
    indegree = {name: 0 for name in unique_names}
    for edge in edges:
        producer = str(edge["producer_tool"])
        consumer = str(edge["consumer_tool"])
        if producer not in indegree or consumer not in indegree:
            continue
        if consumer not in successors[producer]:
            successors[producer].append(consumer)
            indegree[consumer] += 1

    order_rank = {name: index for index, name in enumerate(unique_names)}
    queue = deque(sorted((name for name, degree in indegree.items() if degree == 0), key=order_rank.get))
    depths = {name: 0 for name in unique_names}
    processed = 0
    while queue:
        node = queue.popleft()
        processed += 1
        for consumer in sorted(successors[node], key=order_rank.get):
            depths[consumer] = max(depths[consumer], depths[node] + 1)
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                queue.append(consumer)

    if processed != len(unique_names):
        return UNKNOWN, {str(edge["edge_id"]): UNKNOWN for edge in edges}
    edge_depths = {str(edge["edge_id"]): depths.get(str(edge["consumer_tool"]), UNKNOWN) for edge in edges}
    return max(depths.values(), default=0), edge_depths


def _tool_parameters(tool_graph: Any, tools: Sequence[Any]) -> List[Dict[str, Any]]:
    parameters = []
    seen = set()
    for tool in tools:
        tool_name = str(tool.name)
        for role, schema_name in (("input", "input_schema"), ("output", "output_schema")):
            schema = getattr(tool, schema_name, {}) or {}
            for parameter in schema.get("parameters", []) or []:
                parameter_name = getattr(parameter, "name", UNKNOWN)
                stable_id = _parameter_id(tool_name, role, parameter_name)
                if stable_id in seen:
                    continue
                seen.add(stable_id)
                required = _required_input(tool_graph, parameter, tool) if role == "input" else UNKNOWN
                parameters.append(
                    {
                        "parameter_id": stable_id,
                        "tool_id": _tool_id(tool_name),
                        "tool_name": tool_name,
                        "role": role,
                        "parameter_name": parameter_name,
                        "data_type": _parameter_data_type(parameter),
                        "user_provided": _parameter_user_provided(parameter),
                        "required": required,
                        "optional": (not required) if isinstance(required, bool) else UNKNOWN,
                    }
                )
    return parameters


def build_gold_sidecar(
    tool_graph: Any,
    tool_chain: Any,
    turn_index: int,
    task_id: Optional[str] = None,
    query_id: Optional[str] = None,
    expected_final_scenario: Any = UNKNOWN,
    trust_node_final_scenario: bool = False,
) -> Dict[str, Any]:
    """Build one task-local sidecar while ToolGraph and raw_tool_call are live."""

    if task_id is not None and (not isinstance(task_id, str) or not task_id):
        raise ValueError("task_id must be a non-empty string when supplied")
    nodes = getattr(tool_chain, "tool_chain", None)
    if nodes is None:
        raise TypeError("tool_chain must expose .tool_chain")
    if not isinstance(turn_index, int) or not 0 <= turn_index < len(nodes):
        raise IndexError(f"turn_index {turn_index} is outside {len(nodes)} turns")
    node = nodes[turn_index]
    raw_tools = list(getattr(node, "raw_tool_call", None) or [])
    if not raw_tools:
        raise ValueError("raw_tool_call is empty; export must run before relying on serialized ToolQueryNode data")

    tool_names = [str(tool.name) for tool in raw_tools]
    unique_tools = list({str(tool.name): tool for tool in raw_tools}.values())
    unique_names = list(dict.fromkeys(tool_names))
    positions: Dict[str, List[int]] = defaultdict(list)
    for index, name in enumerate(tool_names):
        positions[name].append(index)

    trace_present = has_selected_dependency_trace(tool_chain)
    trace = selected_dependency_trace(tool_chain)
    turn_trace = [
        record for record in trace
        if record.get("consumer_tool") in positions
    ]
    selected_producer_names = [
        record.get("selected_producer_tool")
        for record in turn_trace
        if isinstance(record.get("selected_producer_tool"), str)
    ]
    relevant_names = list(dict.fromkeys(unique_names + selected_producer_names))
    graph_spec = tool_graph_to_spec(tool_graph, expected_tools=relevant_names)
    candidate_graph_edges = list(graph_spec["dependency_edges"])
    graph_edges = [
        edge for edge in candidate_graph_edges
        if _is_forward_edge(edge, positions)
    ]
    alternative_dependency_groups = []
    unresolved_dependencies = []
    cross_turn_dependencies = []
    resolved_dependency_count = 0

    if trace_present:
        selected_edges = []
        for index, record in enumerate(turn_trace):
            producer = record.get("selected_producer_tool", UNKNOWN)
            consumer = record.get("consumer_tool", UNKNOWN)
            target = record.get("consumer_input_parameter", UNKNOWN)
            selected_output = record.get(
                "selected_producer_output_parameter", UNKNOWN
            )
            cross_turn = producer not in positions
            candidates = [
                edge for edge in candidate_graph_edges
                if edge["producer_tool"] == producer
                and edge["consumer_tool"] == consumer
                and edge["target_parameter"] == target
            ]
            if selected_output != UNKNOWN:
                candidates = [
                    edge for edge in candidates
                    if edge["source_parameter"] == selected_output
                ]

            if len(candidates) == 1:
                resolution_status = "resolved"
                resolved_dependency_count += 1
                if not cross_turn:
                    selected_edges.append(candidates[0])
            elif len(candidates) > 1:
                resolution_status = "unresolved_ambiguous_source_parameter"
            else:
                resolution_status = "unresolved_missing_graph_edge"

            dependency_record = {
                "trace_index": index,
                "resolution_status": resolution_status,
                "producer_tool": producer,
                "consumer_tool": consumer,
                "consumer_input_parameter": target,
                "selected_producer_output_parameter": selected_output,
                "candidate_edge_ids": [
                    edge.get("edge_id", UNKNOWN) for edge in candidates
                ],
                "provenance": record.get("provenance", UNKNOWN),
            }
            alternative_dependency_groups.append(
                {
                    "group_id": f"or::{consumer}::input::{target}::{index}",
                    "semantics": "or",
                    "consumer_tool": consumer,
                    "consumer_input_parameter": target,
                    "alternatives": record.get("alternatives", UNKNOWN),
                    "selected_producer_tool": producer,
                    "selected_producer_output_parameter": selected_output,
                    "resolution_status": resolution_status,
                    "provenance": record.get("provenance", UNKNOWN),
                }
            )
            if resolution_status != "resolved":
                unresolved_dependencies.append(dependency_record)
            if cross_turn:
                cross_turn_dependencies.append(dependency_record)

        graph_edges = selected_edges
        all_resolved = (
            len(turn_trace) == resolved_dependency_count
            and not unresolved_dependencies
        )
        resolution_status = "resolved" if all_resolved else "unresolved"
        dependency_semantics = (
            "selected_reference"
            if all_resolved and not cross_turn_dependencies
            else "selected_reference_incomplete"
        )
    else:
        grouped = defaultdict(list)
        for edge in graph_edges:
            grouped[(edge["consumer_tool"], edge["target_parameter"])].append(edge)
        if all(len(group) == 1 for group in grouped.values()):
            dependency_semantics = "unique_possible_equals_selected"
        else:
            dependency_semantics = "possible_graph_unselected"
            graph_edges = []
        resolution_status = "trace_missing"

    dependency_resolution = {
        "status": resolution_status,
        "selected_dependency_count": len(turn_trace),
        "resolved_dependency_count": resolved_dependency_count,
        "unresolved_dependency_count": len(unresolved_dependencies),
        "cross_turn_dependency_count": len(cross_turn_dependencies),
        "unresolved_dependencies": unresolved_dependencies,
    }
    dependency_depth, edge_depths = _dependency_depths(tool_names, graph_edges)
    if dependency_semantics == "selected_reference_incomplete":
        dependency_depth = UNKNOWN
        edge_depths = {
            str(edge["edge_id"]): UNKNOWN for edge in graph_edges
        }
    servers = sorted({server for server in (_server_id(tool) for tool in unique_tools) if server != UNKNOWN})

    required_nodes = []
    for name in unique_names:
        tool = next(tool for tool in unique_tools if str(tool.name) == name)
        required_nodes.append(
            {
                "tool_id": _tool_id(name),
                "tool_name": name,
                "server_id": _server_id(tool),
                "dependency_depth": (
                    max(
                        (edge_depths.get(edge["edge_id"], 0) for edge in graph_edges
                         if edge["consumer_tool"] == name),
                        default=0,
                    ) if dependency_depth != UNKNOWN else UNKNOWN
                ),
                "sequence_positions": positions[name],
            }
        )

    dependency_edges = []
    for edge in graph_edges:
        producer = str(edge["producer_tool"])
        consumer = str(edge["consumer_tool"])
        source = edge.get("source_parameter", UNKNOWN)
        target = edge.get("target_parameter", UNKNOWN)
        required = edge.get("required", UNKNOWN)
        dependency_edges.append(
            {
                "edge_id": str(edge["edge_id"]),
                "edge_type": edge.get("edge_type", UNKNOWN),
                "graph_provenance": edge.get("graph_provenance", UNKNOWN),
                "dependency_depth": edge_depths.get(str(edge["edge_id"]), UNKNOWN),
                "producer_tool": {"tool_id": _tool_id(producer), "tool_name": producer},
                "producer_output_parameter": {
                    "parameter_id": _parameter_id(producer, "output", source),
                    "parameter_name": source,
                    "data_type": edge.get("source_data_type", UNKNOWN),
                    "user_provided": edge.get("source_parameter_user_provided", UNKNOWN),
                },
                "consumer_tool": {"tool_id": _tool_id(consumer), "tool_name": consumer},
                "consumer_input_parameter": {
                    "parameter_id": _parameter_id(consumer, "input", target),
                    "parameter_name": target,
                    "data_type": edge.get("target_data_type", UNKNOWN),
                    "user_provided": edge.get("target_parameter_user_provided", UNKNOWN),
                },
                "required": required,
                "internal_parameter": edge.get("internal_parameter", UNKNOWN),
                "value_semantics": UNKNOWN,
                "optional": (not required) if isinstance(required, bool) else UNKNOWN,
            }
        )

    query = getattr(node, "query", None)
    seed = _json_value(getattr(tool_chain, "seed", None))
    assigned_task_id = task_id or _task_id(seed, turn_index, query, tool_names)
    final_scenario = expected_final_scenario
    expected_final_state_source = (
        "explicit_argument" if final_scenario != UNKNOWN else UNKNOWN
    )
    if isinstance(final_scenario, str) and final_scenario == UNKNOWN and trust_node_final_scenario:
        final_scenario = getattr(node, "final_scenario", None)
        expected_final_state_source = "selected_querygen_reference_trajectory"

    sidecar = {
        "schema_version": "envfactory_gold_sidecar_v1",
        "task_id": assigned_task_id,
        "seed": seed,
        "environment_identifiers": servers if servers else UNKNOWN,
        "query": {
            "query_id": query_id or f"turn-{turn_index}",
            "turn_index": turn_index,
            "text": query if isinstance(query, str) and query else UNKNOWN,
        },
        "gold_tool_sequence": [
            {
                "position": index,
                "tool_id": _tool_id(name),
                "tool_name": name,
                "server_id": _server_id(raw_tools[index]),
            }
            for index, name in enumerate(tool_names)
        ],
        "gold_tool_set": sorted(set(tool_names)),
        "selected_dependency_trace_present": trace_present,
        "dependency_semantics": dependency_semantics,
        "dependency_resolution": dependency_resolution,
        "cross_turn_dependency": bool(cross_turn_dependencies),
        "cross_turn_dependencies": cross_turn_dependencies,
        "dependency_depth": dependency_depth,
        "required_tool_nodes": required_nodes,
        "dependency_edges": dependency_edges,
        "alternative_dependency_groups": alternative_dependency_groups,
        "parameters": _tool_parameters(tool_graph, unique_tools),
        "initial_scenario": _json_value(getattr(node, "initial_scenario", None)),
        "expected_final_scenario": _json_value(final_scenario),
        "expected_final_state_source": expected_final_state_source,
        "provenance": {
            "source": "live ToolGraph + live ToolQueryChain/ToolQueryNode",
            "relevant_subgraph_only": True,
            "stable_ids_use_python_object_identity": False,
            "metadata_preservation_point": "generation-time graph metadata must be preserved before ToolQueryNode.save() discards raw_tool_call information",
        },
    }
    eligibility = evaluate_probe_eligibility(sidecar)
    sidecar["probe_eligibility"] = eligibility
    sidecar["structural_diagnosis_eligible"] = eligibility["eligible"]
    return sidecar


def write_gold_sidecar(sidecar: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically write one JSON sidecar after verifying JSON serializability."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(sidecar), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    return output


def _safe_filename(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._") or "task"
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:96]}-{digest}"


def export_chain_sidecars(
    tool_graph: Any,
    tool_chain: Any,
    output_dir: str | Path,
    task_ids: Optional[Sequence[str]] = None,
    expected_final_scenarios: Optional[Sequence[Any]] = None,
) -> List[Path]:
    """Export every live query turn without persisting the full NetworkX graph."""

    nodes = getattr(tool_chain, "tool_chain", None)
    if nodes is None:
        raise TypeError("tool_chain must expose .tool_chain")
    if task_ids is not None and len(task_ids) != len(nodes):
        raise ValueError("task_ids length must match the number of query turns")
    if expected_final_scenarios is not None and len(expected_final_scenarios) != len(nodes):
        raise ValueError("expected_final_scenarios length must match the number of query turns")

    output_dir = Path(output_dir)
    paths = []
    for turn_index in range(len(nodes)):
        sidecar = build_gold_sidecar(
            tool_graph,
            tool_chain,
            turn_index,
            task_id=task_ids[turn_index] if task_ids is not None else None,
            expected_final_scenario=(
                expected_final_scenarios[turn_index]
                if expected_final_scenarios is not None
                else UNKNOWN
            ),
            trust_node_final_scenario=expected_final_scenarios is None,
        )
        paths.append(write_gold_sidecar(sidecar, output_dir / f"{_safe_filename(sidecar['task_id'])}.gold.json"))
    return paths


class GenerationSidecarCallback:
    """Callback for QueryGenNonConv.terminate immediately before chain.save()."""

    def __init__(
        self,
        output_dir: str | Path,
        task_id_factory: Optional[Callable[[Any], Optional[str]]] = None,
        expected_final_scenario_factory: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.task_id_factory = task_id_factory
        self.expected_final_scenario_factory = expected_final_scenario_factory

    def before_save(self, context: Any) -> Path:
        task_id = self.task_id_factory(context) if self.task_id_factory is not None else None
        expected = (
            self.expected_final_scenario_factory(context)
            if self.expected_final_scenario_factory is not None
            else UNKNOWN
        )
        query_id = f"turn-{context.idx}"
        sidecar = build_gold_sidecar(
            context.tool_graph,
            context.tool_chain,
            context.idx,
            task_id=task_id,
            query_id=query_id,
            expected_final_scenario=expected,
            trust_node_final_scenario=self.expected_final_scenario_factory is None,
        )
        return write_gold_sidecar(
            sidecar,
            self.output_dir / f"{_safe_filename(sidecar['task_id'])}.gold.json",
        )
