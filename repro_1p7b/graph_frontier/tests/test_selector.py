import unittest

from repro_1p7b.graph_frontier.selector import select_curriculum


CAPABILITY = {
    "metrics": {
        "depth:1": {"frontier_score": 0.9, "weakness": 0.7, "attempts": 8, "pass_rate": 0.3},
        "depth:2": {"frontier_score": 0.4, "weakness": 0.9, "attempts": 3, "pass_rate": 0.1},
    }
}
CANDIDATES = [
    {"task_id": "a", "character_count": 100, "token_count": 10, "diversity_group": "g1", "capability_keys": ["depth:1"]},
    {"task_id": "b", "character_count": 200, "token_count": 20, "diversity_group": "g2", "capability_keys": ["depth:2"]},
    {"task_id": "c", "character_count": 300, "token_count": 30, "diversity_group": "g3", "capability_keys": ["missing"]},
]


class SelectorTests(unittest.TestCase):
    def test_deterministic_selection_and_reasons(self):
        kwargs = dict(
            capability_map=CAPABILITY,
            candidates=CANDIDATES,
            sample_count=2,
            seed=31415,
            target_character_budget=300,
            target_token_budget=30,
            repeat_cap=1,
            trials=64,
        )
        first = select_curriculum(**kwargs)
        second = select_curriculum(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["report"]["max_observed_repeat"], 1)
        self.assertTrue(first["report"]["within_budget_tolerance"])
        self.assertTrue(all("priority_reason" in item for item in first["selected"]))

    def test_repeat_cap_is_never_exceeded(self):
        result = select_curriculum(
            CAPABILITY,
            CANDIDATES,
            sample_count=5,
            seed=9,
            repeat_cap=2,
            exploration_floor=0.2,
            trials=32,
        )
        self.assertLessEqual(result["report"]["max_observed_repeat"], 2)
        self.assertEqual(len(result["selected"]), 5)

    def test_group_cap(self):
        candidates = [dict(CANDIDATES[0], task_id="a2"), *CANDIDATES]
        result = select_curriculum(
            CAPABILITY,
            candidates,
            sample_count=3,
            seed=5,
            repeat_cap=1,
            group_cap=1,
            trials=32,
        )
        self.assertTrue(all(value <= 1 for value in result["report"]["diversity_group_counts"].values()))

    def test_exploration_floor_keeps_unknown_candidate_selectable(self):
        result = select_curriculum(
            CAPABILITY,
            [CANDIDATES[2]],
            sample_count=1,
            seed=1,
            repeat_cap=1,
            exploration_floor=0.15,
        )
        self.assertEqual(result["selected"][0]["priority"], 0.15)
        self.assertEqual(result["selected"][0]["priority_reason"]["capabilities"][0]["status"], "missing")

    def test_infeasible_repeat_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "repeat_cap"):
            select_curriculum(CAPABILITY, CANDIDATES, sample_count=4, seed=0, repeat_cap=1)


if __name__ == "__main__":
    unittest.main()
