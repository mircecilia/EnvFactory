import unittest

from repro_1p7b.graph_frontier.adapters import (
    UNKNOWN,
    normalized_rollout_bundle,
    querygen_artifact_to_bundle,
    tool_graph_to_spec,
)
from repro_1p7b.graph_frontier.capability import aggregate_profiles, beta_smoothed_metric
from repro_1p7b.graph_frontier.profiler import profile_rollout


GRAPH = {
    "expected_tool_nodes": ["get_user", "get_order", "cancel_order"],
    "dependency_edges": [
        {
            "edge_id": "user-to-order",
            "producer_tool": "get_user",
            "consumer_tool": "get_order",
            "edge_type": "parameter_flow",
            "source_parameter": "user_id",
            "target_parameter": "user_id",
            "source_data_type": "string",
            "target_data_type": "string",
            "required": True,
            "internal_parameter": True,
        },
        {
            "edge_id": "order-to-cancel",
            "producer_tool": "get_order",
            "consumer_tool": "cancel_order",
            "edge_type": "parameter_flow",
            "source_parameter": "order_id",
            "target_parameter": "order_id",
            "source_data_type": "string",
            "target_data_type": "string",
            "required": True,
            "internal_parameter": True,
        },
    ],
    "tool_metadata": {
        "get_user": {"state_changing": False},
        "get_order": {"state_changing": False},
        "cancel_order": {"state_changing": True},
    },
}


def make_bundle(events=None, terminal=True, state=True):
    if events is None:
        events = [
            {"step_index": 0, "tool_name": "get_user", "arguments": {"email": "a@b"}, "result": {"user_id": "u1"}, "execution_success": True},
            {"step_index": 2, "tool_name": "get_order", "arguments": {"user_id": "u1"}, "result": {"order_id": "o1"}, "execution_success": True},
            {"step_index": 4, "tool_name": "cancel_order", "arguments": {"order_id": "o1"}, "result": {"cancelled": True}, "execution_success": True},
        ]
    return normalized_rollout_bundle("task-1", events, GRAPH, terminal, state)


class ProfilerTests(unittest.TestCase):
    def test_happy_path_and_depth(self):
        profile = profile_rollout(make_bundle())
        self.assertEqual(profile["root_cause_failures"], [])
        self.assertEqual(profile["satisfied_dependency_edges"], ["user-to-order", "order-to-cancel"])
        self.assertEqual(profile["max_dependency_depth_reached"], 2)
        self.assertEqual(profile["first_failed_edge"], UNKNOWN)

    def test_wrong_propagated_value_is_first_root(self):
        bundle = make_bundle(state=False)
        bundle["events"][1]["arguments"]["user_id"] = "WRONG"
        profile = profile_rollout(bundle)
        self.assertEqual(len(profile["root_cause_failures"]), 1)
        self.assertEqual(profile["root_cause_failures"][0]["type"], "wrong_propagated_value")
        self.assertEqual(profile["first_failed_edge"]["edge_id"], "user-to-order")
        self.assertEqual(profile["first_failure_step"], 2)

    def test_missing_producer_is_one_root_and_downstream_is_attributed(self):
        events = [
            {"step_index": 0, "tool_name": "get_order", "arguments": {"user_id": "u1"}, "result": {"order_id": "o1"}, "execution_success": False},
            {"step_index": 2, "tool_name": "cancel_order", "arguments": {"order_id": "o1"}, "result": {}, "execution_success": False},
        ]
        profile = profile_rollout(make_bundle(events, terminal=False, state=False))
        self.assertEqual(len(profile["root_cause_failures"]), 1)
        self.assertEqual(profile["root_cause_failures"][0]["type"], "missing_producer")
        self.assertTrue(all(item["propagated_from"] == profile["root_cause_failures"][0]["failure_id"] for item in profile["downstream_propagated_failures"]))

    def test_first_broken_edge_prevents_duplicate_downstream_roots(self):
        events = [
            {"step_index": 0, "tool_name": "get_user", "arguments": {}, "result": {"user_id": "u1"}, "execution_success": True},
            {"step_index": 2, "tool_name": "get_order", "arguments": {"user_id": "bad"}, "result": {"order_id": "bad-order"}, "execution_success": False},
            {"step_index": 4, "tool_name": "cancel_order", "arguments": {"order_id": "bad-order"}, "result": {}, "execution_success": False},
        ]
        profile = profile_rollout(make_bundle(events, terminal=False, state=False))
        self.assertEqual([r["type"] for r in profile["root_cause_failures"]], ["wrong_propagated_value"])
        self.assertGreaterEqual(len(profile["downstream_propagated_failures"]), 3)

    def test_failed_producer_is_root_not_each_consumer(self):
        events = [
            {"step_index": 0, "tool_name": "get_user", "arguments": {}, "result": {}, "execution_success": False},
            {"step_index": 2, "tool_name": "get_order", "arguments": {"user_id": "u1"}, "result": {}, "execution_success": False},
            {"step_index": 4, "tool_name": "cancel_order", "arguments": {"order_id": "o1"}, "result": {}, "execution_success": False},
        ]
        profile = profile_rollout(make_bundle(events, terminal=False, state=False))
        self.assertEqual([r["type"] for r in profile["root_cause_failures"]], ["tool_execution_failure"])

    def test_final_state_only_failure(self):
        profile = profile_rollout(make_bundle(terminal=False, state=False))
        self.assertEqual(len(profile["root_cause_failures"]), 1)
        self.assertEqual(profile["root_cause_failures"][0]["type"], "final_state_failure")
        self.assertEqual(profile["first_failure_step"], UNKNOWN)

    def test_redundant_call(self):
        bundle = make_bundle()
        bundle["events"].insert(1, dict(bundle["events"][0], step_index=1))
        profile = profile_rollout(bundle)
        self.assertEqual(len(profile["redundant_tool_calls"]), 1)
        self.assertEqual(profile["redundant_tool_calls"][0]["duplicates_call_index"], 0)

    def test_missing_optional_producer_is_not_a_root(self):
        graph = {
            "expected_tool_nodes": ["consumer"],
            "dependency_edges": [{
                "edge_id": "optional-edge", "producer_tool": "optional_lookup", "consumer_tool": "consumer",
                "edge_type": "parameter_flow", "source_parameter": "value", "target_parameter": "value",
                "required": False, "internal_parameter": False,
            }],
            "tool_metadata": {},
        }
        bundle = normalized_rollout_bundle("optional", [{
            "step_index": 0, "tool_name": "consumer", "arguments": {}, "result": {}, "execution_success": True,
        }], graph, True, True)
        profile = profile_rollout(bundle)
        self.assertEqual(profile["root_cause_failures"], [])
        self.assertEqual(profile["dependency_edge_checks"][0]["status"], "optional_not_observed")
        self.assertEqual(profile["dependency_edge_checks"][0]["success"], UNKNOWN)

    def test_missing_reference_tools_are_path_diagnostic_not_root(self):
        profile = profile_rollout(make_bundle(events=[
            {"step_index": 0, "tool_name": "get_user", "arguments": {}, "result": {"user_id": "u1"}, "execution_success": True},
        ], terminal=False, state=False))
        self.assertEqual([root["type"] for root in profile["root_cause_failures"]], ["final_state_failure"])
        self.assertFalse(profile["path_adherence"])
        divergent = {item["tool_name"] for item in profile["reference_path_divergence"]}
        self.assertEqual(divergent, {"get_order", "cancel_order"})


class CapabilityTests(unittest.TestCase):
    def test_beta_smoothing_and_scoring_interfaces(self):
        metric = beta_smoothed_metric(1, 2, previous_pass_rate=0.4, hint_with=(2, 2), hint_without=(0, 2))
        self.assertEqual(metric["pass_rate"], 0.5)
        self.assertEqual(metric["frontier_score"], 1.0)
        self.assertEqual(metric["weakness"], 0.5)
        self.assertAlmostEqual(metric["learning_progress"], 0.1)
        self.assertEqual(metric["hint_regret"], 0.5)
        self.assertEqual(metric["attempts"], 2)

    def test_aggregation_counts_only_known_observations(self):
        good = profile_rollout(make_bundle())
        bad = profile_rollout(make_bundle(terminal=False, state=False))
        unknown = profile_rollout(make_bundle(terminal=UNKNOWN, state=UNKNOWN))
        result = aggregate_profiles([good, bad, unknown])
        self.assertEqual(result["metrics"]["terminal"]["attempts"], 2)
        self.assertEqual(result["metrics"]["terminal"]["pass_rate"], 0.5)
        self.assertEqual(result["unknown_observations"]["terminal_success"], 1)
        self.assertIn("2", result["dependency_depth_buckets"])

    def test_aggregation_exposes_hint_regret_when_labeled(self):
        with_hint = profile_rollout(normalized_rollout_bundle("hint", make_bundle()["events"], GRAPH, True, True, hint_condition="with_hint"))
        without_hint = profile_rollout(normalized_rollout_bundle("no-hint", make_bundle()["events"], GRAPH, False, False, hint_condition="without_hint"))
        metric = aggregate_profiles([with_hint, without_hint])["metrics"]["terminal"]
        self.assertAlmostEqual(metric["hint_regret"], 1 / 3)
        self.assertEqual(metric["hint_evidence"]["with_hint"]["attempts"], 1)


class AdapterTests(unittest.TestCase):
    def test_querygen_adapter_does_not_infer_success(self):
        artifact = {"seed": 7, "nodes": [{"steps": [
            {"role": "tool_call", "content": [{"name": "lookup", "arguments": {"id": 1}}]},
            {"role": "tool_response", "content": [{"value": "ok"}]},
        ]}]}
        bundle = querygen_artifact_to_bundle(artifact)
        self.assertEqual(bundle["terminal_success"], UNKNOWN)
        self.assertEqual(bundle["events"][0]["execution_success"], UNKNOWN)
        self.assertEqual(bundle["expected_tool_nodes"], UNKNOWN)

    def test_toolgraph_adapter_joins_output_to_input(self):
        class Tool:
            def __init__(self, name):
                self.name = name
                self.input_schema = {}
                self.output_schema = {}
        class Parameter:
            def __init__(self, name, user_provided):
                self.name = name
                self.user_provided = user_provided
        class Graph:
            def __init__(self, edges):
                self._edges = edges
                self.nodes = list({node for pair in edges for node in pair})
            def predecessors(self, node):
                return [a for (a, b) in self._edges if b is node]
            def successors(self, node):
                return [b for (a, b) in self._edges if a is node]
            def get_edge_data(self, a, b):
                return self._edges[(a, b)]
        producer, consumer = Tool("producer"), Tool("consumer")
        output = Parameter("record_id", False)
        input_parameter = Parameter("record_id", False)
        edges = {
            (producer, output): {"edge_type": "tool_to_parameter"},
            (output, input_parameter): {"edge_type": "parameter_to_parameter"},
            (input_parameter, consumer): {"edge_type": "parameter_to_tool", "required": True},
        }
        spec = tool_graph_to_spec(type("ToolGraph", (), {"graph": Graph(edges)})(), [producer, consumer])
        self.assertEqual(len(spec["dependency_edges"]), 1)
        edge = spec["dependency_edges"][0]
        self.assertEqual((edge["producer_tool"], edge["consumer_tool"]), ("producer", "consumer"))
        self.assertTrue(edge["internal_parameter"])
        self.assertTrue(edge["required"])


if __name__ == "__main__":
    unittest.main()
