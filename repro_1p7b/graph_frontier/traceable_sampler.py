"""Non-invasive dependency tracing for EnvFactory topology sampling.

The wrapper temporarily observes _get_priors and choice on an existing sampler.
It delegates every random decision to the original methods exactly once, so
sampling behaviour and RNG consumption are unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .adapters import UNKNOWN

TRACE_ATTRIBUTE = "_graph_frontier_selected_dependencies"


def _name(value: Any) -> Any:
    name = getattr(value, "name", UNKNOWN)
    return name if isinstance(name, str) and name else UNKNOWN


def _data_type(value: Any) -> Any:
    result = getattr(value, "data_type", UNKNOWN)
    return result if isinstance(result, str) and result else UNKNOWN


def _known_bool(value: Any) -> Any:
    return value if isinstance(value, bool) else UNKNOWN


def _candidate_outputs(tool_graph: Any, consumer_parameter: Any, producer: Any) -> List[Any]:
    """Return graph-grounded producer outputs that can feed one input."""

    outputs: List[Any] = []
    for predecessor in tool_graph.graph.predecessors(consumer_parameter):
        if predecessor is producer:
            outputs.append(consumer_parameter)
            continue
        try:
            grandparents = tool_graph.graph.predecessors(predecessor)
        except Exception:
            continue
        if any(grandparent is producer for grandparent in grandparents):
            outputs.append(predecessor)
    unique = []
    seen = set()
    for output in outputs:
        key = (_name(output), _data_type(output))
        if key not in seen:
            unique.append(output)
            seen.add(key)
    return unique


def _alternative(tool_graph: Any, parameter: Any, producer: Any) -> Dict[str, Any]:
    outputs = _candidate_outputs(tool_graph, parameter, producer)
    return {
        "producer_tool": _name(producer),
        "producer_output_parameters": [
            {"parameter_name": _name(output), "data_type": _data_type(output)}
            for output in outputs
        ] or UNKNOWN,
    }


class DependencyTraceRecorder:
    """Observe an existing TopologySampler without importing the core package."""

    def __init__(self, sampler: Any) -> None:
        if not callable(getattr(sampler, "_get_priors", None)):
            raise TypeError("sampler must expose _get_priors")
        if not callable(getattr(sampler, "choice", None)):
            raise TypeError("sampler must expose choice")
        self.sampler = sampler
        self.records: List[Dict[str, Any]] = []
        self._active = False
        self._pending: List[Dict[str, Any]] = []

    def sample(self, tool_graph: Any, **sample_kwargs: Any) -> Any:
        """Call ToolGraph.sample and attach selected-dependency records."""

        if self._active:
            raise RuntimeError("DependencyTraceRecorder is not re-entrant")
        self.records = []
        self._pending = []
        original_get_priors = self.sampler._get_priors
        original_choice = self.sampler.choice
        sampler_dict = getattr(self.sampler, "__dict__", {})
        had_get_override = "_get_priors" in sampler_dict
        had_choice_override = "choice" in sampler_dict
        saved_get_override = sampler_dict.get("_get_priors")
        saved_choice_override = sampler_dict.get("choice")

        def traced_get_priors(graph: Any, parameter: Any, consumer: Any) -> Any:
            candidates = original_get_priors(graph, parameter, consumer)
            edge_data = graph.graph.get_edge_data(parameter, consumer) or {}
            user_provided = _known_bool(getattr(parameter, "user_provided", UNKNOWN))
            self._pending.append(
                {
                    "graph": graph,
                    "consumer": consumer,
                    "parameter": parameter,
                    "required": _known_bool(edge_data.get("required", UNKNOWN)),
                    "user_provided": user_provided,
                    "internal_parameter": (
                        not user_provided if isinstance(user_provided, bool) else UNKNOWN
                    ),
                    "candidates": list(candidates),
                }
            )
            return candidates

        def traced_choice(visited_nodes: Any, candidate_nodes: Any) -> Any:
            context = self._pending.pop()
            selected = original_choice(visited_nodes, candidate_nodes)
            if selected is None:
                return None
            outputs = _candidate_outputs(
                context["graph"], context["parameter"], selected
            )
            selected_output = outputs[0] if len(outputs) == 1 else None
            self.records.append(
                {
                    "consumer_tool": _name(context["consumer"]),
                    "consumer_input_parameter": _name(context["parameter"]),
                    "consumer_input_data_type": _data_type(context["parameter"]),
                    "selected_producer_tool": _name(selected),
                    "selected_producer_output_parameter": (
                        _name(selected_output) if selected_output is not None else UNKNOWN
                    ),
                    "selected_producer_output_data_type": (
                        _data_type(selected_output) if selected_output is not None else UNKNOWN
                    ),
                    "required": context["required"],
                    "user_provided": context["user_provided"],
                    "internal_parameter": context["internal_parameter"],
                    "alternatives": [
                        _alternative(
                            context["graph"], context["parameter"], candidate
                        )
                        for candidate in context["candidates"]
                    ],
                    "alternative_semantics": "or",
                    "provenance": "TopologySampler.sample_prior->_get_priors->choice",
                }
            )
            return selected

        self._active = True
        self.sampler._get_priors = traced_get_priors
        self.sampler.choice = traced_choice
        try:
            chain = tool_graph.sample(self.sampler, **sample_kwargs)
        finally:
            if had_get_override:
                self.sampler._get_priors = saved_get_override
            else:
                del self.sampler._get_priors
            if had_choice_override:
                self.sampler.choice = saved_choice_override
            else:
                del self.sampler.choice
            self._active = False
            self._pending = []
        setattr(chain, TRACE_ATTRIBUTE, deepcopy(self.records))
        return chain


def sample_with_dependency_trace(tool_graph: Any, sampler: Any, **sample_kwargs: Any) -> Any:
    """Convenience entry point returning the normal sampled ToolQueryChain."""

    return DependencyTraceRecorder(sampler).sample(tool_graph, **sample_kwargs)


def has_selected_dependency_trace(tool_chain: Any) -> bool:
    return hasattr(tool_chain, TRACE_ATTRIBUTE)


def selected_dependency_trace(tool_chain: Any) -> List[Dict[str, Any]]:
    value = getattr(tool_chain, TRACE_ATTRIBUTE, [])
    return deepcopy(value) if isinstance(value, list) else []
