import json
import tempfile
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None

from repro_1p7b.graph_frontier.adapters import UNKNOWN
from repro_1p7b.graph_frontier.export_pipeline import profile_sidecar_and_trace
from repro_1p7b.graph_frontier.gold_sidecar import GenerationSidecarCallback, build_gold_sidecar, write_gold_sidecar
from repro_1p7b.graph_frontier.rollout_trace import TypedRolloutRecorder


class Parameter:
    def __init__(self, name, user_provided, data_type="integer"):
        self.name = name
        self.user_provided = user_provided
        self.data_type = data_type


class Tool:
    def __init__(self, name, inputs=(), outputs=(), required=(), server="MockEnv"):
        self.name = name
        self.server = server
        self.input_schema = {"parameters": list(inputs), "required": list(required)}
        self.output_schema = {"parameters": list(outputs), "required": []}


class Graph:
    def __init__(self, edges):
        self._edges = edges
        self.nodes = list(dict.fromkeys(node for pair in edges for node in pair))

    def predecessors(self, node):
        return [source for source, target in self._edges if target is node]

    def successors(self, node):
        return [target for source, target in self._edges if source is node]

    def get_edge_data(self, source, target):
        return self._edges.get((source, target))


class Node:
    def __init__(self, tools):
        self.raw_tool_call = tools
        self.query = "Cancel the order for the given user"
        self.initial_scenario = {"MockEnv": {"users": [123], "orders": [456]}}
        self.final_scenario = {"MockEnv": {"orders": []}}


class Chain:
    def __init__(self, tools):
        self.seed = 20260913
        self.tool_chain = [Node(tools)]


def mock_source():
    email = Parameter("email", True, "string")
    user_output = Parameter("user_id", False)
    user_input = Parameter("user_id", False)
    order_output = Parameter("order_id", False)
    order_input = Parameter("order_id", False)

    tool_a = Tool("A", inputs=[email], outputs=[user_output], required=["email"])
    tool_b = Tool("B", inputs=[user_input], outputs=[order_output], required=["user_id"])
    tool_c = Tool("C", inputs=[order_input], required=["order_id"])
    edges = {
        (email, tool_a): {"edge_type": "parameter_to_tool", "required": True},
        (tool_a, user_output): {"edge_type": "tool_to_parameter"},
        (user_output, user_input): {"edge_type": "parameter_to_parameter"},
        (user_input, tool_b): {"edge_type": "parameter_to_tool", "required": True},
        (tool_a, tool_b): {"edge_type": "tool_to_tool"},
        (tool_b, order_output): {"edge_type": "tool_to_parameter"},
        (order_output, order_input): {"edge_type": "parameter_to_parameter"},
        (order_input, tool_c): {"edge_type": "parameter_to_tool", "required": True},
        (tool_b, tool_c): {"edge_type": "tool_to_tool"},
    }
    tool_graph = type("ToolGraph", (), {"graph": Graph(edges)})()
    return tool_graph, Chain([tool_a, tool_b, tool_c])


def mock_gold():
    tool_graph, chain = mock_source()
    return build_gold_sidecar(tool_graph, chain, 0, task_id="probe-001")


def recorder_for(events, final_status="failure", verifier=False):
    recorder = TypedRolloutRecorder("probe-001", ["MockEnv"], {"orders": [456]})
    for event in events:
        recorder.record_tool_event(**event)
    recorder.finalize(final_status, {"orders": [456]}, verifier)
    return recorder


def happy_prefix(c_order_id=999, c_success=False):
    return [
        {"step_index": 0, "tool_name": "A", "tool_arguments": {"email": "u@example.test"}, "tool_response": {"user_id": 123}, "execution_success": True},
        {"step_index": 2, "tool_name": "B", "tool_arguments": {"user_id": 123}, "tool_response": {"order_id": 456}, "execution_success": True},
        {"step_index": 4, "tool_name": "C", "tool_arguments": {"order_id": c_order_id}, "tool_response": {"cancelled": c_success}, "execution_success": c_success},
    ]


class ExportPipelineTests(unittest.TestCase):
    def test_end_to_end_parameter_mismatch(self):
        sidecar = mock_gold()
        recorder = recorder_for(happy_prefix())
        trace = recorder.as_dict()
        profile = profile_sidecar_and_trace(sidecar, trace)

        self.assertEqual(sidecar["dependency_depth"], 2)
        pairs = [
            (edge["producer_tool"]["tool_name"], edge["producer_output_parameter"]["parameter_name"], edge["consumer_tool"]["tool_name"], edge["consumer_input_parameter"]["parameter_name"])
            for edge in sidecar["dependency_edges"]
        ]
        self.assertEqual(pairs, [("A", "user_id", "B", "user_id"), ("B", "order_id", "C", "order_id")])
        self.assertEqual(trace["events"][0]["returned_fields"]["user_id"], 123)
        self.assertEqual(trace["events"][1]["returned_fields"]["order_id"], 456)
        self.assertEqual(trace["events"][2]["tool_arguments"]["order_id"], 999)

        checks = {check["edge_id"]: check for check in profile["dependency_edge_checks"]}
        a_to_b = next(check for edge_id, check in checks.items() if edge_id.startswith("A->B"))
        b_to_c = next(check for edge_id, check in checks.items() if edge_id.startswith("B->C"))
        self.assertTrue(a_to_b["success"])
        self.assertEqual(b_to_c["status"], "failed_wrong_value")
        self.assertEqual(b_to_c["source_values"], [456])
        self.assertEqual(b_to_c["target_values"], [999])
        self.assertEqual(profile["first_failed_edge"]["edge_id"], b_to_c["edge_id"])
        self.assertEqual(profile["first_failure_step"], 4)
        self.assertEqual([root["type"] for root in profile["root_cause_failures"]], ["wrong_propagated_value"])
        self.assertTrue(profile["downstream_propagated_failures"])

    def test_json_files_and_schemas(self):
        if jsonschema is None:
            self.skipTest("jsonschema is not installed")
        sidecar = mock_gold()
        recorder = recorder_for(happy_prefix())
        base = Path(__file__).resolve().parents[1]
        gold_schema = json.loads((base / "GOLD_SIDECAR_SCHEMA.json").read_text(encoding="utf-8"))
        eligibility_schema = json.loads(
            (base / "PROBE_ELIGIBILITY_SCHEMA.json").read_text(encoding="utf-8"))
        rollout_schema = json.loads((base / "ROLLOUT_TRACE_SCHEMA.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            gold_path = write_gold_sidecar(sidecar, Path(directory) / "task.gold.json")
            rollout_path = recorder.write(Path(directory) / "task.rollout.json")
            saved_gold = json.loads(gold_path.read_text(encoding="utf-8"))
            saved_rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
        jsonschema.validate(saved_gold, gold_schema)
        jsonschema.validate(saved_gold["probe_eligibility"], eligibility_schema)
        jsonschema.validate(saved_rollout, rollout_schema)
        self.assertEqual(saved_gold["expected_final_state"], UNKNOWN)
        self.assertEqual(saved_gold["expected_final_scenario"], UNKNOWN)
        self.assertIn("before ToolQueryNode.save()", saved_gold["provenance"]["metadata_preservation_point"])
        self.assertTrue(all("0x" not in parameter["parameter_id"] for parameter in saved_gold["parameters"]))

    def test_missing_producer(self):
        trace = recorder_for(happy_prefix()[1:]).as_dict()
        profile = profile_sidecar_and_trace(mock_gold(), trace)
        self.assertEqual(len(profile["root_cause_failures"]), 1)
        self.assertEqual(profile["root_cause_failures"][0]["type"], "missing_producer")

    def test_extra_and_repeated_tool(self):
        events = happy_prefix(c_order_id=456, c_success=True)
        events.insert(1, dict(events[0], step_index=1))
        events.append({"step_index": 6, "tool_name": "X", "tool_arguments": {"q": 1}, "tool_response": {"ok": True}, "execution_success": True})
        profile = profile_sidecar_and_trace(mock_gold(), recorder_for(events, "success", True).as_dict())
        self.assertEqual(len(profile["redundant_tool_calls"]), 1)
        self.assertEqual([event["tool_name"] for event in profile["extra_tool_calls"]], ["X"])
        self.assertEqual(profile["root_cause_failures"], [])

    def test_final_state_only_and_unknown_final_state(self):
        events = happy_prefix(c_order_id=456, c_success=True)
        final_only = profile_sidecar_and_trace(mock_gold(), recorder_for(events, "failure", False).as_dict())
        self.assertEqual([root["type"] for root in final_only["root_cause_failures"]], ["final_state_failure"])

        unknown = recorder_for(events, "unknown", UNKNOWN).as_dict()
        unknown_profile = profile_sidecar_and_trace(mock_gold(), unknown)
        self.assertEqual(unknown_profile["state_success"], UNKNOWN)
        self.assertEqual(unknown_profile["terminal_success"], UNKNOWN)
        self.assertEqual(unknown_profile["root_cause_failures"], [])

    def test_generation_callback_writes_before_save_sidecar(self):
        tool_graph, chain = mock_source()
        context = type("Context", (), {"tool_graph": tool_graph, "tool_chain": chain, "idx": 0})()
        with tempfile.TemporaryDirectory() as directory:
            path = GenerationSidecarCallback(directory, task_id_factory=lambda _context: "probe-hook").before_save(context)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["task_id"], "probe-hook")
        self.assertEqual(saved["query"]["query_id"], "turn-0")
        self.assertEqual(len(saved["dependency_edges"]), 2)
        self.assertEqual(
            saved["expected_final_state_source"],
            "selected_querygen_reference_trajectory",
        )
        self.assertEqual(saved["expected_final_scenario"], chain.tool_chain[0].final_scenario)

    def test_sync_wrapper_marks_exception_without_text_heuristic(self):
        recorder = TypedRolloutRecorder("probe-001")
        def broken(_name, _arguments):
            raise RuntimeError("typed failure")
        with self.assertRaises(RuntimeError):
            recorder.record_sync_call(0, "A", {}, broken)
        event = recorder.as_dict()["events"][0]
        self.assertFalse(event["execution_success"])
        self.assertEqual(event["exception"]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
