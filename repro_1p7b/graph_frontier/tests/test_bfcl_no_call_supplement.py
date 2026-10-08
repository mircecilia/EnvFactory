import unittest
from repro_1p7b.graph_frontier.bfcl_no_call_supplement import constraints
class NoCallTests(unittest.TestCase):
    def test_nested_actual_call(self):
        x=constraints({"possible_answer":[[]],"model_result_decoded":[[["read()"]]]})
        self.assertEqual(x["violated_turns"],[0])
    def test_decode_failure_not_invented_call(self):
        x=constraints({"possible_answer":[[]],"model_result_decoded":[["Decode failed"]]})
        self.assertEqual(x["violated_turns"],[])
    def test_missing_turn_not_zero_violation(self):
        x=constraints({"possible_answer":[[]],"model_result_decoded":[]})
        self.assertEqual(x["unobserved_turns"],[0])
    def test_category_not_cause(self):
        x=constraints({"id":"multi_turn_miss_param_1","possible_answer":[["read()"]],"model_result_decoded":[[["read()"]]]})
        self.assertEqual(x["eligible_turns"],[])
