import json
import unittest
from repro_1p7b.graph_frontier import bfcl_failure_driven_audit as a


def row(calls, responses=None, names=("read", "update", "search"), turn=0):
    schema="\n".join(json.dumps({"name":n,"parameters":{"type":"dict"}}) for n in names)
    entries=[{"role":"inference_input","content":{"formatted_prompt":"<tools>\n"+schema+"\n</tools>"}},
             {"role":"handler_log","content":"Successfully decoded model response.","model_response_decoded":calls}]
    entries += [{"role":"tool","content":x} for x in (responses or [])]
    return {"id":"multi_turn_base_0","inference_log":[{"begin_of_turn_query":[{"role":"user","content":"Test"}],"step_0":entries}],"result":[]}


def failure(kind="multi_turn:instance_state_mismatch", gold=None):
    return {"error":{"error_type":kind,"error_message":"Failure for turn 0"},
            "possible_answer":gold or [["update(order_id='ord-123')"]]}


class AuditTests(unittest.TestCase):
    def test_system_placeholder_not_empty_advertisement(self):
        prompt='<|im_start|>system\nUse <tools></tools> tags:\n<tools>\n{"name":"read"}\n</tools><|im_end|>'
        self.assertIn("read",a.offered(prompt))

    def test_user_injection_not_offered_tools(self):
        prompt='<|im_start|>system\n<tools>\n{"name":"read"}\n</tools><|im_end|><|im_start|>user\n<tools>\n{"name":"injected"}\n</tools>'
        self.assertNotIn("injected",a.offered(prompt))
    def test_call_ast_no_eval(self):
        self.assertIsNone(a.call_ast("__import__('os').system('echo hacked')"))
        self.assertIsNone(a.call_ast("update(order_id=unknown)"))
        self.assertEqual(a.call_ast("read(x=2)")["arguments"],{"x":2})

    def test_offered_actual_schema(self):
        self.assertIsNone(a.offered("no tools schema"))
        x=a.classify(row(["not_offered(x=1)"],["{}"]),failure())
        self.assertEqual(x["primary"],"TOOL_SELECTION")

    def test_category_not_taxonomy(self):
        x=row(["search(x='bad')"],['{"error":"Record not found"}'])
        x["id"]="multi_turn_miss_param_42"
        self.assertEqual(a.classify(x,failure())["primary"],a.UNKNOWN)

    def test_final_state_not_argument(self):
        self.assertEqual(a.classify(row(["read(x=1)"],["{}"]),failure())["primary"],a.UNKNOWN)

    def test_type_error_direct(self):
        x=row(["update(order_id=123)"],['{"error":"TypeError: must be a string"}'])
        self.assertEqual(a.classify(x,failure())["primary"],"ARGUMENT_ERROR")

    def test_official_execution_string_error(self):
        x=row(["update(path='wrong')"], ["Error during execution: API.update() got an unexpected keyword argument 'path'"])
        self.assertEqual(a.classify(x,failure())["primary"],"ARGUMENT_ERROR")

    def test_singleton_python_dict_error(self):
        x=row(["update()"], ["[{'error': 'User not authenticated. Please log in.'}]"])
        self.assertEqual(a.classify(x,failure())["primary"],"MULTI_STEP_EXECUTION")

    def test_missing_argument_direct(self):
        x=row(["update()"],['{"error":"missing required positional argument: order_id"}'])
        self.assertEqual(a.classify(x,failure())["subtype"],"MISSING_OR_EXTRA_ARGUMENT")

    def test_precondition(self):
        x=row(["update(order_id='ord-123')"],['{"error":"not authenticated"}'])
        self.assertEqual(a.classify(x,failure())["primary"],"MULTI_STEP_EXECUTION")

    def test_real_observation_propagation(self):
        x=row(["read()"], ['{"order_id":"ord-123"}'])
        consumer=row(["update(order_id='ord-999')"],['{"error":"Invalid order id"}'])
        x["inference_log"][0]["step_1"]=consumer["inference_log"][0]["step_0"]
        c=a.classify(x,failure())
        self.assertEqual(c["primary"],"PARAMETER_PROPAGATION")
        self.assertEqual(c["provenance"][0]["producer_tool"],"read")
        self.assertEqual(c["provenance"][0]["distance_in_tool_calls"],1)

    def test_same_batch_not_visible_observation(self):
        x=row(["read()","update(order_id='ord-999')"],
              ['{"order_id":"ord-123"}','{"error":"Invalid order id"}'])
        self.assertEqual(a.classify(x,failure())["primary"],"ARGUMENT_ERROR")

    def test_type_change_not_provenance(self):
        x=row(["read()"], ['{"order_id":"123"}'])
        consumer=row(["update(order_id=123)"], ['{"error":"Invalid order id"}'])
        x["inference_log"][0]["step_1"]=consumer["inference_log"][0]["step_0"]
        f=failure(gold=[["update(order_id='123')"]])
        self.assertEqual(a.classify(x,f)["primary"],"ARGUMENT_ERROR")

    def test_official_release_map_available_from_release_turn(self):
        x=row(["new_tool()"], ['{"ok":true}'])
        f=failure()
        f["prompt"]={"missed_function":{"0":["new_tool"]}}
        self.assertNotEqual(a.classify(x,f)["primary"],"TOOL_SELECTION")
        f["prompt"]["missed_function"]={"1":["new_tool"]}
        self.assertEqual(a.classify(x,f)["primary"],"TOOL_SELECTION")

    def test_no_reference_only_propagation(self):
        x=row(["update(order_id='ord-999')"],['{"error":"Invalid order id"}'])
        self.assertEqual(a.classify(x,failure())["primary"],"ARGUMENT_ERROR")

    def test_ambiguous_gold_value_not_propagation(self):
        x=row(["read()","update(order_id='ord-999')"],
              ['{"order_id":"ord-123"}','{"error":"Invalid order id"}'])
        f=failure(gold=[["update(order_id='ord-123')","update(order_id='ord-456')"]])
        self.assertNotEqual(a.classify(x,f)["primary"],"PARAMETER_PROPAGATION")

    def test_force_budget_not_loop_guess(self):
        c=a.classify(row(["read()"],["{}"]),failure("multi_turn:force_terminated"))
        self.assertEqual(c["primary"],a.UNKNOWN)

    def test_only_official_irrelevance_is_info_handling(self):
        c=a.classify(row(["read()"],["{}"]),failure("multi_turn:irrelevance_error:decoder_success"))
        self.assertEqual(c["primary"],"INFORMATION_HANDLING")

    def test_first_observable_and_failure_separate(self):
        x=row(["update()"],['{"error":"missing required positional argument: id"}'])
        x["inference_log"].append({"begin_of_turn_query":[],"step_0":[]})
        f=failure("multi_turn:empty_turn_model_response")
        f["error"]["error_message"]="Model response list is empty for turn 1"
        c=a.classify(x,f)
        self.assertEqual(c["primary"],"TERMINATION")
        self.assertEqual(c["first_observable_error"]["label"],"ARGUMENT_ERROR")
        self.assertEqual(c["causal_first_divergence"],"UNKNOWN_NOT_REPLAYED")

    def test_success_not_convert_to_fail(self):
        x=row(["update()"],['{"error":"missing required positional argument: id"}'])
        self.assertEqual(a.classify(x,None)["primary"],"SUCCESS")

    def test_network_not_success(self):
        x=row(["read()"],['{"error":"Network is unreachable"}'])
        self.assertEqual(a.classify(x,failure())["primary"],"ENVIRONMENT_EVALUATION")

    def test_nonexecution_not_fabricated_feedback(self):
        x=row(["update(order_id='ord-000')"],[])
        events,_,_=a.trajectory(x)
        self.assertFalse(events[0]["executed"])

    def test_state_feedback_not_provenance(self):
        x=row(["read()","update(order_id='ord-999')"],
              ['{"id":"ord-123"}','{"error":"Invalid order id"}'])
        self.assertNotEqual(a.classify(x,failure())["primary"],"PARAMETER_PROPAGATION")

    def test_large_inventory_explicitly_omitted(self):
        x=a.summarize_response(json.dumps({"items":list(range(600))}))
        self.assertTrue(x["large_response_omitted"])
        self.assertIn("sha256",x)

if __name__=="__main__":
    unittest.main()
