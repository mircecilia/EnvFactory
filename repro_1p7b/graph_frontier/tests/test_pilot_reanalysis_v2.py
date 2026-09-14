from repro_1p7b.graph_frontier.pilot_reanalysis_v2 import (
    call_patterns,
    internal_funnel,
    semantic_verify,
)


def event(tool, step, args=None, result=None, success=True):
    return {
        "tool_name": tool,
        "step_index": step,
        "tool_arguments": args or {},
        "returned_fields": {} if result is None else result,
        "execution_success": success,
    }


def test_read_only_typed_observation_and_empty_reference_support():
    manifest = {"template": "symbol_info", "environment": "TradingBot"}
    reference = [{
        "tool_name": "TradingBot-get_stock_info",
        "arguments": {"symbol": "AAPL"},
        "result": {"symbol": "AAPL", "price": 10.0},
    }]
    rollout = {
        "events": [event(
            "TradingBot-get_stock_info", 0, {"symbol": "AAPL"},
            {"symbol": "AAPL", "price": 10.0},
        )],
        "terminal_success": True,
        "final_status": "success",
    }
    answer = semantic_verify(manifest, rollout, reference)
    assert answer["semantic_success"] is True
    assert answer["ambiguity_flag"] is True

    reference[0]["result"] = {}
    answer = semantic_verify(manifest, rollout, reference)
    assert answer["semantic_verifier_supported"] is False
    assert answer["unsupported_reason"] == "empty_reference_observation"


def test_mutation_semantics_allow_extra_nonreference_tool():
    manifest = {"template": "city_save", "environment": "Weather"}
    reference = [{
        "tool_name": "Weather-save_location",
        "arguments": {
            "alias": "x", "latitude": 1.0, "longitude": 2.0, "name": "X",
        },
        "result": "ok",
    }]
    rollout = {
        "events": [
            event("Weather-unrelated_lookup", 0),
            event(
                "Weather-save_location", 1,
                {"alias": "x", "latitude": 1.0, "longitude": 2.0, "name": "X"},
            ),
        ],
        "final_environment_state": {
            "Weather": {"saved_locations": {
                "x": {
                    "alias": "x", "latitude": 1.0, "longitude": 2.0,
                    "name": "X", "timezone": "UTC",
                }
            }}
        },
        "terminal_success": True,
        "final_status": "success",
    }
    answer = semantic_verify(manifest, rollout, reference)
    assert answer["semantic_success"] is True
    assert answer["completion_step"] == 1


def test_internal_funnel_has_fixed_gold_denominator():
    edge = {
        "edge_id": "A-B",
        "producer_tool": {"tool_name": "S-A"},
        "consumer_tool": {"tool_name": "S-B"},
    }
    other = {
        "edge_id": "A-C",
        "producer_tool": {"tool_name": "S-A"},
        "consumer_tool": {"tool_name": "S-C"},
    }
    manifest = {"dependency_edges": [edge, other]}
    rollout = {"events": [
        event("S-A", 0, result={"x": 1}),
        event("S-B", 1, {"x": 1}),
    ]}
    profiler_result = {"profile": {"dependency_edge_checks": [{
        "edge_id": "A-B",
        "source_value_available": True,
        "target_value_available": True,
        "success": True,
    }]}}
    funnel = internal_funnel(manifest, rollout, profiler_result)
    assert funnel["gold_internal_opportunities"] == 2
    assert funnel["consumer_reached"] == 1
    assert funnel["correct_propagated"] == 1


def test_call_patterns_find_repeat_cycle_and_post_success():
    manifest = {
        "gold_tools": ["S-A", "S-B"],
        "dependency_edges": [{
            "producer_tool": {"tool_name": "S-A"},
            "consumer_tool": {"tool_name": "S-B"},
        }],
    }
    rollout = {"events": [
        event("S-A", 0, {"x": 1}),
        event("S-B", 1, {"x": 1}),
        event("S-A", 2, {"x": 1}),
        event("S-X", 3),
    ]}
    profiler_result = {"profile": {
        "dependency_edge_checks": [],
        "redundant_tool_calls": [{}],
        "extra_tool_calls": [{}],
    }}
    patterns = call_patterns(
        manifest, rollout, profiler_result, {"completion_step": 1}
    )
    assert patterns["repeated_pattern_counts"]["same_tool_same_args"] == 1
    assert patterns["cycle_pattern_counts"]["a_b_a"] == 1
    assert patterns["post_success_extra_calls"] == 2
    assert patterns["unexpected_pattern_counts"]["post_success_extra_tool"] == 1
