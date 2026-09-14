import json
import random
import unittest
from types import SimpleNamespace

try:
    from mcp.types import CallToolResult, TextContent
except ImportError:
    CallToolResult = TextContent = None

from repro_1p7b.graph_frontier.adapters import UNKNOWN, normalized_rollout_bundle
from repro_1p7b.graph_frontier.eligibility import (
    evaluate_probe_eligibility,
    is_structurally_diagnosable,
)
from repro_1p7b.graph_frontier.export_pipeline import profile_sidecar_and_trace
from repro_1p7b.graph_frontier.fastmcp_adapter import (
    fastmcp_execution_success,
    fastmcp_result_fields,
    trace_quality_metrics,
)
from repro_1p7b.graph_frontier.gold_sidecar import build_gold_sidecar
from repro_1p7b.graph_frontier.profiler import profile_rollout
from repro_1p7b.graph_frontier.rollout_trace import TypedRolloutRecorder
from repro_1p7b.graph_frontier.traceable_sampler import (
    TRACE_ATTRIBUTE,
    sample_with_dependency_trace,
)


class Parameter:
    def __init__(self, name, user_provided=False, data_type="string"):
        self.name = name
        self.user_provided = user_provided
        self.data_type = data_type


class Tool:
    def __init__(self, name, inputs=(), outputs=()):
        self.name = name
        self.server = "mock"
        self.input_schema = {"parameters": list(inputs), "required": [p.name for p in inputs]}
        self.output_schema = {"parameters": list(outputs), "required": []}


class Graph:
    def __init__(self, edge_data):
        self.edge_data = edge_data
        self.nodes = list(dict.fromkeys(node for edge in edge_data for node in edge))

    def predecessors(self, node):
        return [source for source, target in self.edge_data if target is node]

    def successors(self, node):
        return [target for source, target in self.edge_data if source is node]

    def get_edge_data(self, source, target):
        return self.edge_data.get((source, target), {})


class Chain:
    def __init__(self, tools, seed):
        self.seed = seed
        self.tool_chain = [
            SimpleNamespace(
                raw_tool_call=tools,
                query="resolve user",
                initial_scenario={"done": False},
                final_scenario={"done": True},
            )
        ]


class FakeTopologySampler:
    def _get_priors(self, graph, parameter, tool):
        return list(graph.producers)

    def choice(self, visited_nodes, candidates):
        return random.choice(candidates) if candidates else None

    def sample_prior(self, graph, node, visited_nodes=()):
        parameter = node.input_schema["parameters"][0]
        prior = self.choice(
            list(visited_nodes), self._get_priors(graph, parameter, node)
        )
        return [prior] if prior is not None else []


class FakeToolGraph:
    def __init__(self):
        out_a = Parameter("user_id")
        out_b = Parameter("user_id")
        in_c = Parameter("user_id")
        self.a = Tool("A", outputs=[out_a])
        self.b = Tool("B", outputs=[out_b])
        self.c = Tool("C", inputs=[in_c])
        self.producers = [self.a, self.b]
        self.graph = Graph(
            {
                (self.a, out_a): {"edge_type": "tool_to_parameter"},
                (out_a, in_c): {"edge_type": "parameter_to_parameter"},
                (self.b, out_b): {"edge_type": "tool_to_parameter"},
                (out_b, in_c): {"edge_type": "parameter_to_parameter"},
                (in_c, self.c): {
                    "edge_type": "parameter_to_tool",
                    "required": True,
                },
            }
        )

    def sample(self, sampler, seed=42, **_kwargs):
        random.seed(seed)
        selected = sampler.sample_prior(self, self.c, visited_nodes=[])[0]
        return Chain([selected, self.c], seed)


def one_edge_bundle(
    source,
    target,
    source_type,
    target_type,
    value_semantics=UNKNOWN,
    state=False,
):
    graph = {
        "expected_tool_nodes": ["P", "C"],
        "dependency_edges": [
            {
                "edge_id": "P-C",
                "producer_tool": "P",
                "consumer_tool": "C",
                "edge_type": "parameter_flow",
                "source_parameter": "value",
                "target_parameter": "value",
                "source_data_type": source_type,
                "target_data_type": target_type,
                "value_semantics": value_semantics,
                "required": True,
                "internal_parameter": True,
            }
        ],
        "tool_metadata": {},
    }
    events = [
        {
            "step_index": 0,
            "tool_name": "P",
            "arguments": {},
            "result": {"value": source},
            "execution_success": True,
        },
        {
            "step_index": 1,
            "tool_name": "C",
            "arguments": {"value": target},
            "result": {},
            "execution_success": True,
        },
    ]
    return normalized_rollout_bundle("typed", events, graph, True, state)


class SamplerAndSidecarTests(unittest.TestCase):
    def test_or_candidates_record_only_selected_reference(self):
        graph = FakeToolGraph()
        baseline = graph.sample(FakeTopologySampler(), seed=19)
        traced = sample_with_dependency_trace(
            graph, FakeTopologySampler(), seed=19
        )
        self.assertEqual(
            [tool.name for tool in baseline.tool_chain[0].raw_tool_call],
            [tool.name for tool in traced.tool_chain[0].raw_tool_call],
        )
        sidecar = build_gold_sidecar(
            graph, traced, 0, task_id="alternative"
        )
        selected = traced.tool_chain[0].raw_tool_call[0].name
        other = "B" if selected == "A" else "A"
        self.assertEqual(sidecar["dependency_semantics"], "selected_reference")
        self.assertEqual(len(sidecar["dependency_edges"]), 1)
        self.assertEqual(
            sidecar["dependency_edges"][0]["producer_tool"]["tool_name"],
            selected,
        )
        group = sidecar["alternative_dependency_groups"][0]
        self.assertEqual(group["semantics"], "or")
        self.assertEqual(
            {item["producer_tool"] for item in group["alternatives"]},
            {"A", "B"},
        )

        recorder = TypedRolloutRecorder("alternative")
        recorder.record_tool_event(
            0, selected, {}, {"user_id": "u1"}, execution_success=True
        )
        recorder.record_tool_event(
            1, "C", {"user_id": "u1"}, {}, execution_success=True
        )
        recorder.finalize(
            "success", {"done": True}, verifier_result=True
        )
        profile = profile_sidecar_and_trace(sidecar, recorder.as_dict())
        self.assertEqual(profile["root_cause_failures"], [])
        self.assertNotIn(
            other,
            [
                root["tool_name"]
                for root in profile["structural_root_cause_failures"]
            ],
        )


class RequiredInternalTests(unittest.TestCase):
    def test_required_user_provided_input_needs_no_producer(self):
        bundle = one_edge_bundle("x", "x", "string", "string", state=UNKNOWN)
        bundle["events"] = [bundle["events"][1]]
        bundle["dependency_edges"][0]["internal_parameter"] = False
        profile = profile_rollout(bundle)
        self.assertEqual(profile["structural_root_cause_failures"], [])
        self.assertEqual(
            profile["dependency_edge_checks"][0]["status"],
            "user_provided_no_producer_required",
        )

    def test_unknown_internal_is_conservative(self):
        bundle = one_edge_bundle("x", "x", "string", "string", state=UNKNOWN)
        bundle["events"] = [bundle["events"][1]]
        bundle["dependency_edges"][0]["internal_parameter"] = UNKNOWN
        profile = profile_rollout(bundle)
        self.assertEqual(profile["structural_root_cause_failures"], [])
        self.assertEqual(
            profile["dependency_edge_checks"][0]["status"], UNKNOWN
        )


class FastMCPAdapterTests(unittest.TestCase):
    @unittest.skipIf(CallToolResult is None, "mcp package is not in this Python")
    def test_installed_mcp_result_types(self):
        result = CallToolResult(
            content=[TextContent(type="text", text='{"fallback": 1}')],
            structuredContent={"typed": 2},
            isError=False,
        )
        self.assertEqual(fastmcp_result_fields(result), {"typed": 2})
        self.assertTrue(fastmcp_execution_success(result))

    def test_structured_content_and_typed_error(self):
        result = SimpleNamespace(
            structuredContent={"id": 7},
            isError=False,
            content=[SimpleNamespace(type="text", text='{"id": 8}')],
        )
        self.assertEqual(fastmcp_result_fields(result), {"id": 7})
        self.assertTrue(fastmcp_execution_success(result))

    def test_snake_case_and_json_text_fallback(self):
        result = {
            "structured_content": None,
            "is_error": True,
            "content": [{"type": "text", "text": '{"id": 9}'}],
        }
        self.assertEqual(fastmcp_result_fields(result), {"id": 9})
        self.assertFalse(fastmcp_execution_success(result))

    def test_non_json_text_stays_unknown(self):
        result = SimpleNamespace(
            structuredContent=None,
            isError=False,
            content=[SimpleNamespace(type="text", text="success: id 9")],
        )
        self.assertEqual(fastmcp_result_fields(result), UNKNOWN)

    def test_trace_quality_rates(self):
        trace = {
            "events": [
                {"returned_fields": {"x": 1}, "execution_success": True},
                {"returned_fields": UNKNOWN, "execution_success": UNKNOWN},
            ]
        }
        self.assertEqual(
            trace_quality_metrics(trace),
            {
                "events": 2,
                "typed_value_recovery_rate": 0.5,
                "typed_execution_status_rate": 0.5,
            },
        )


class StateAndDepthTests(unittest.TestCase):
    def test_correct_alternative_path_is_not_task_failure(self):
        graph = {
            "expected_tool_nodes": ["A", "B", "C"],
            "dependency_edges": [
                {
                    "edge_id": "B-C",
                    "producer_tool": "B",
                    "consumer_tool": "C",
                    "edge_type": "parameter_flow",
                    "source_parameter": "id",
                    "target_parameter": "id",
                    "source_data_type": "string",
                    "target_data_type": "string",
                    "required": True,
                    "internal_parameter": True,
                }
            ],
            "tool_metadata": {},
        }
        events = [
            {"step_index": 0, "tool_name": "A", "arguments": {}, "result": {}, "execution_success": True},
            {"step_index": 1, "tool_name": "X", "arguments": {}, "result": {"id": "ok"}, "execution_success": True},
            {"step_index": 2, "tool_name": "C", "arguments": {"id": "ok"}, "result": {}, "execution_success": True},
        ]
        profile = profile_rollout(
            normalized_rollout_bundle("alt", events, graph, True, True)
        )
        self.assertFalse(profile["path_adherence"])
        self.assertTrue(profile["task_success"])
        self.assertEqual(profile["root_cause_failures"], [])
        self.assertEqual(
            profile["structural_root_cause_failures"][0]["type"],
            "missing_producer",
        )

    def test_canonical_final_state_and_source(self):
        graph = FakeToolGraph()
        chain = sample_with_dependency_trace(
            graph, FakeTopologySampler(), seed=2
        )
        sidecar = build_gold_sidecar(
            graph,
            chain,
            0,
            task_id="canonical",
            trust_node_final_scenario=True,
        )
        recorder = TypedRolloutRecorder("canonical")
        recorder.finalize(
            "unknown", {"done": True}, verifier_result=UNKNOWN
        )
        profile = profile_sidecar_and_trace(sidecar, recorder.as_dict())
        self.assertTrue(profile["state_success"])
        self.assertEqual(
            profile["state_verification_source"],
            "canonical_final_state_comparison",
        )

    def test_sidecar_depth_is_authoritative(self):
        bundle = one_edge_bundle("x", "x", "string", "string", state=True)
        bundle["dependency_edges"][0]["dependency_depth"] = 7
        bundle["tool_metadata"] = {
            "P": {"dependency_depth": 3},
            "C": {"dependency_depth": 7},
        }
        bundle["task_dependency_depth"] = 7
        profile = profile_rollout(bundle)
        self.assertEqual(profile["max_dependency_depth_reached"], 7)
        self.assertEqual(
            profile["dependency_edge_checks"][0]["dependency_depth"], 7
        )

    def test_cycle_depth_fallback_is_unknown(self):
        graph = {
            "expected_tool_nodes": ["A", "B"],
            "dependency_edges": [
                {"edge_id": "A-B", "producer_tool": "A", "consumer_tool": "B", "required": True, "internal_parameter": True, "source_parameter": UNKNOWN, "target_parameter": UNKNOWN},
                {"edge_id": "B-A", "producer_tool": "B", "consumer_tool": "A", "required": True, "internal_parameter": True, "source_parameter": UNKNOWN, "target_parameter": UNKNOWN},
            ],
            "tool_metadata": {},
        }
        events = [
            {"step_index": 0, "tool_name": "A", "arguments": {}, "result": {}, "execution_success": True},
            {"step_index": 1, "tool_name": "B", "arguments": {}, "result": {}, "execution_success": True},
        ]
        profile = profile_rollout(
            normalized_rollout_bundle("cycle", events, graph, True, UNKNOWN)
        )
        self.assertEqual(
            {check["dependency_depth"] for check in profile["dependency_edge_checks"]},
            {UNKNOWN},
        )
        self.assertEqual(profile["max_dependency_depth_reached"], UNKNOWN)

    def test_optional_unknown_incoming_does_not_block_depth(self):
        graph = {
            "expected_tool_nodes": ["C"],
            "dependency_edges": [
                {
                    "edge_id": "optional-C",
                    "producer_tool": "P",
                    "consumer_tool": "C",
                    "required": False,
                    "internal_parameter": UNKNOWN,
                    "source_parameter": "x",
                    "target_parameter": "x",
                }
            ],
            "tool_metadata": {"C": {"dependency_depth": 4}},
        }
        events = [
            {"step_index": 0, "tool_name": "C", "arguments": {}, "result": {}, "execution_success": True}
        ]
        profile = profile_rollout(
            normalized_rollout_bundle("optional-depth", events, graph, True, True)
        )
        self.assertEqual(profile["max_dependency_depth_reached"], 4)


class TypedValueMatchingTests(unittest.TestCase):
    def test_scalar_exact_and_mismatch(self):
        good = profile_rollout(
            one_edge_bundle(456, 456, "integer", "integer", state=True)
        )
        bad = profile_rollout(
            one_edge_bundle(456, 999, "integer", "integer", state=False)
        )
        self.assertTrue(good["dependency_edge_checks"][0]["value_match"])
        self.assertEqual(
            bad["root_cause_failures"][0]["type"], "wrong_propagated_value"
        )

    def test_collection_partial_overlap_is_failure(self):
        profile = profile_rollout(
            one_edge_bundle([1, 2], [2, 3], "array", "array")
        )
        self.assertFalse(profile["dependency_edge_checks"][0]["value_match"])

    def test_dict_structural_equality(self):
        profile = profile_rollout(
            one_edge_bundle(
                {"a": 1, "b": [2]},
                {"b": [2], "a": 1},
                "object",
                "object",
                state=True,
            )
        )
        self.assertTrue(profile["dependency_edge_checks"][0]["value_match"])

    def test_collection_to_scalar_without_semantics_is_unknown(self):
        profile = profile_rollout(
            one_edge_bundle([1, 2], 2, "array", "integer")
        )
        self.assertEqual(
            profile["dependency_edge_checks"][0]["value_match"], UNKNOWN
        )


class ProbeEligibilityTests(unittest.TestCase):
    @staticmethod
    def _record(
        producer="A",
        consumer="C",
        target="user_id",
        output=UNKNOWN,
    ):
        return {
            "consumer_tool": consumer,
            "consumer_input_parameter": target,
            "consumer_input_data_type": "string",
            "selected_producer_tool": producer,
            "selected_producer_output_parameter": output,
            "selected_producer_output_data_type": (
                "string" if output != UNKNOWN else UNKNOWN
            ),
            "required": True,
            "user_provided": False,
            "internal_parameter": True,
            "alternatives": [
                {
                    "producer_tool": producer,
                    "producer_output_parameters": UNKNOWN,
                }
            ],
            "alternative_semantics": "or",
            "provenance": "test_selected_dependency",
        }

    @staticmethod
    def _attach(chain, records):
        setattr(chain, TRACE_ATTRIBUTE, records)
        return chain

    def _unique_sidecar(self, trust_final=False):
        graph = FakeToolGraph()
        chain = sample_with_dependency_trace(
            graph, FakeTopologySampler(), seed=7
        )
        return build_gold_sidecar(
            graph,
            chain,
            0,
            task_id="unique-resolution",
            trust_node_final_scenario=trust_final,
        )

    def _ambiguous_sidecar(self):
        out_user = Parameter("user_id")
        out_owner = Parameter("owner_id")
        input_id = Parameter("id")
        producer = Tool("A", outputs=[out_user, out_owner])
        consumer = Tool("C", inputs=[input_id])
        graph = SimpleNamespace(
            graph=Graph(
                {
                    (producer, out_user): {
                        "edge_type": "tool_to_parameter"
                    },
                    (out_user, input_id): {
                        "edge_type": "parameter_to_parameter"
                    },
                    (producer, out_owner): {
                        "edge_type": "tool_to_parameter"
                    },
                    (out_owner, input_id): {
                        "edge_type": "parameter_to_parameter"
                    },
                    (input_id, consumer): {
                        "edge_type": "parameter_to_tool",
                        "required": True,
                    },
                }
            )
        )
        chain = self._attach(
            Chain([producer, consumer], 11),
            [self._record(target="id")],
        )
        return build_gold_sidecar(
            graph, chain, 0, task_id="ambiguous-resolution"
        )

    def _missing_edge_sidecar(self):
        graph = FakeToolGraph()
        chain = self._attach(
            Chain([graph.a, graph.c], 13),
            [self._record(target="not_in_graph")],
        )
        return build_gold_sidecar(
            graph, chain, 0, task_id="missing-edge-resolution"
        )

    def _cross_turn_sidecar(self):
        graph = FakeToolGraph()
        chain = SimpleNamespace(
            seed=17,
            tool_chain=[
                SimpleNamespace(
                    raw_tool_call=[graph.a],
                    query="lookup",
                    initial_scenario={"done": False},
                    final_scenario={"done": False},
                ),
                SimpleNamespace(
                    raw_tool_call=[graph.c],
                    query="consume",
                    initial_scenario={"done": False},
                    final_scenario={"done": True},
                ),
            ],
        )
        self._attach(
            chain,
            [
                self._record(
                    producer="A",
                    consumer="C",
                    target="user_id",
                    output="user_id",
                )
            ],
        )
        return build_gold_sidecar(
            graph, chain, 1, task_id="cross-turn"
        )

    def test_unique_source_is_resolved(self):
        sidecar = self._unique_sidecar()
        resolution = sidecar["dependency_resolution"]
        self.assertEqual(resolution["status"], "resolved")
        self.assertEqual(resolution["selected_dependency_count"], 1)
        self.assertEqual(resolution["resolved_dependency_count"], 1)
        self.assertEqual(len(sidecar["dependency_edges"]), 1)

    def test_ambiguous_source_is_unresolved(self):
        sidecar = self._ambiguous_sidecar()
        resolution = sidecar["dependency_resolution"]
        self.assertEqual(resolution["status"], "unresolved")
        self.assertEqual(resolution["selected_dependency_count"], 1)
        self.assertEqual(resolution["resolved_dependency_count"], 0)
        self.assertEqual(
            resolution["unresolved_dependencies"][0]["resolution_status"],
            "unresolved_ambiguous_source_parameter",
        )
        self.assertEqual(
            sidecar["dependency_semantics"],
            "selected_reference_incomplete",
        )
        self.assertEqual(sidecar["dependency_depth"], UNKNOWN)

    def test_selected_producer_missing_graph_edge_is_unresolved(self):
        sidecar = self._missing_edge_sidecar()
        unresolved = sidecar["dependency_resolution"][
            "unresolved_dependencies"
        ]
        self.assertEqual(
            unresolved[0]["resolution_status"],
            "unresolved_missing_graph_edge",
        )
        self.assertEqual(sidecar["dependency_edges"], [])

    def test_unresolved_sidecar_rejected_by_helper(self):
        sidecar = self._ambiguous_sidecar()
        self.assertFalse(is_structurally_diagnosable(sidecar))
        self.assertFalse(sidecar["structural_diagnosis_eligible"])
        self.assertIn(
            "selected_dependency_resolution_incomplete",
            evaluate_probe_eligibility(sidecar)["reasons"],
        )

    def test_same_turn_dependency_is_eligible(self):
        sidecar = self._unique_sidecar()
        self.assertFalse(sidecar["cross_turn_dependency"])
        self.assertTrue(sidecar["structural_diagnosis_eligible"])
        self.assertTrue(is_structurally_diagnosable(sidecar))

    def test_cross_turn_dependency_is_ineligible(self):
        sidecar = self._cross_turn_sidecar()
        self.assertTrue(sidecar["cross_turn_dependency"])
        self.assertEqual(
            sidecar["dependency_resolution"]["cross_turn_dependency_count"],
            1,
        )
        self.assertFalse(sidecar["structural_diagnosis_eligible"])

    def test_cross_turn_rejection_reason(self):
        reasons = self._cross_turn_sidecar()["probe_eligibility"]["reasons"]
        self.assertIn(
            "cross_turn_dependency_not_supported_v1", reasons
        )

    def test_state_reference_provenance(self):
        sidecar = self._unique_sidecar(trust_final=True)
        self.assertEqual(
            sidecar["expected_final_state_source"],
            "selected_querygen_reference_trajectory",
        )
        self.assertEqual(
            sidecar["expected_final_state"], {"done": True}
        )
        self.assertEqual(
            sidecar["expected_final_scenario"], {"done": True}
        )

    def test_missing_expected_final_state_keeps_structural_eligibility(self):
        sidecar = self._unique_sidecar(trust_final=False)
        report = evaluate_probe_eligibility(sidecar)
        self.assertEqual(sidecar["expected_final_state"], UNKNOWN)
        self.assertEqual(sidecar["expected_final_scenario"], UNKNOWN)
        self.assertFalse(
            report["checks"]["expected_final_state_available"]
        )
        self.assertFalse(report["state_diagnosis_available"])
        self.assertTrue(report["eligible"])


if __name__ == "__main__":
    unittest.main()
