#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("parameter_aware_data.py")
SPEC = importlib.util.spec_from_file_location("parameter_aware_data", MODULE_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class ParameterAwareDataTest(unittest.TestCase):
    def test_dependency_depth_across_tool_responses(self):
        sample = {
            "instruction": '<tool_response>{"detail":{"owner_id":"owner-777"}}</tool_response>',
            "input": "",
            "output": '<tool_call>{"name":"fetch_owner","arguments":{"owner":"owner-777","active":true}}</tool_call>',
            "system": "tools",
            "history": [
                ["Find Alice", '<tool_call>{"name":"search","arguments":{"query":"Alice"}}</tool_call>'],
                [
                    '<tool_response>{"id":"abc-123","status":"success"}</tool_response>',
                    '<tool_call>{"name":"fetch","arguments":{"record_id":"abc-123"}}</tool_call>',
                ],
            ],
        }
        feature = MOD.extract_features(sample, 0)
        self.assertEqual(feature["tool_calls"], 3)
        self.assertEqual(feature["dependent_calls"], 2)
        self.assertEqual(feature["matched_args"], 2)
        self.assertEqual(feature["max_dependency_depth"], 2)

    def test_low_information_scalars_are_excluded(self):
        sample = {
            "instruction": '<tool_response>{"currency":"USD","count":1,"ok":true}</tool_response>',
            "input": "",
            "output": '<tool_call>{"name":"noop","arguments":{"currency":"USD","count":1,"ok":true}}</tool_call>',
            "system": "tools",
            "history": [],
        }
        feature = MOD.extract_features(sample, 0)
        self.assertFalse(feature["has_internal_dependency"])
        self.assertEqual(feature["matched_args"], 0)

    def test_length_bins_and_seed_are_deterministic(self):
        features = [
            {"chars": 100, "dependent_calls": 0, "max_dependency_depth": 0, "matched_args": 0},
            {"chars": 120, "dependent_calls": 1, "max_dependency_depth": 1, "matched_args": 1},
            {"chars": 700, "dependent_calls": 0, "max_dependency_depth": 0, "matched_args": 0},
            {"chars": 740, "dependent_calls": 2, "max_dependency_depth": 2, "matched_args": 2},
        ]
        first = MOD.choose_indices(features, 4, 42, 512)
        second = MOD.choose_indices(features, 4, 42, 512)
        self.assertEqual(first, second)
        self.assertEqual(sum(features[i]["chars"] < 512 for i in first), 2)
        self.assertEqual(sum(features[i]["chars"] >= 512 for i in first), 2)


if __name__ == "__main__":
    unittest.main()
