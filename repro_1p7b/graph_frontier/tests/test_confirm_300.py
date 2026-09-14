from repro_1p7b.graph_frontier.confirm_300 import (
    EXPECTED_DEPTH,
    TARGET_COUNTS,
    assign_splits,
)
from repro_1p7b.graph_frontier.confirm_analyze import (
    normalize_text,
    paired_metric,
    semantic_verify_with_text,
)


def test_target_counts_and_depth_quota():
    assert sum(TARGET_COUNTS.values()) == 300
    by_depth = {"1": 0, "2": 0, "3+": 0}
    for template, count in TARGET_COUNTS.items():
        by_depth[EXPECTED_DEPTH[template]] += count
    assert by_depth == {"1": 120, "2": 110, "3+": 70}


def test_split_is_exact_and_deterministic():
    left = [{"task_id": f"task-{index:03d}"} for index in range(300)]
    right = list(reversed([dict(row) for row in left]))
    assign_splits(left)
    assign_splits(right)
    assert sum(row["split"] == "heldout" for row in left) == 60
    assert sum(row["split"] == "diagnosis" for row in left) == 240
    assert {row["task_id"]: row["split"] for row in left} == {
        row["task_id"]: row["split"] for row in right
    }


def test_normalized_text_and_paired_exact_counts():
    assert normalize_text("  AAPL, 150.25  ") == "aapl 150.25"
    result = paired_metric(
        {"a": True, "b": True, "c": False, "d": False},
        {"a": True, "b": False, "c": True, "d": False},
    )
    assert result["right_wins"] == 1
    assert result["right_losses"] == 1
    assert result["ties_both_success"] == 1
    assert result["ties_both_failure"] == 1
    assert result["effect_size_pp_right_minus_left"] == 0.0
    assert result["exact_two_sided_p"] == 1.0


def test_read_only_semantic_requires_typed_observation_and_final_text():
    manifest = {
        "template": "symbol_info",
        "environment": "TradingBot",
        "semantic_verifier": {
            "semantic_verifier_supported": True,
            "verifier_type": "typed_observation_predicate",
            "checked_fields": [
                "typed observation predicate",
                "final_assistant_content",
            ],
            "reference_source": "typed reference trace and canonical final state",
            "text_match_mode": "nfkc_casefold_all_normalized_values",
            "text_match_values": ["AAPL", 150.25],
            "ambiguity_flag": False,
            "unsupported_reason": None,
        },
    }
    reference = [
        {
            "tool_name": "TradingBot-get_stock_info",
            "arguments": {"symbol": "AAPL"},
            "result": {"symbol": "AAPL", "price": 150.25},
        }
    ]
    rollout = {
        "events": [
            {
                "tool_name": "TradingBot-get_stock_info",
                "tool_arguments": {"symbol": "AAPL"},
                "returned_fields": {"symbol": "AAPL", "price": 150.25},
                "execution_success": True,
                "step_index": 0,
            }
        ],
        "terminal_success": True,
        "final_status": "success",
        "final_assistant_content": "AAPL is currently trading at $150.25.",
    }
    semantic = semantic_verify_with_text(manifest, rollout, reference)
    assert semantic["observation_predicate"] is True
    assert semantic["text_predicate"] is True
    assert semantic["semantic_success"] is True

    rollout["final_assistant_content"] = "I found the requested quote."
    semantic = semantic_verify_with_text(manifest, rollout, reference)
    assert semantic["observation_predicate"] is True
    assert semantic["text_predicate"] is False
    assert semantic["semantic_success"] is False
