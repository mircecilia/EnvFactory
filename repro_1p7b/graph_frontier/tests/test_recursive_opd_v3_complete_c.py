from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import digest
from repro_1p7b.graph_frontier.recursive_opd_v3.prepare import length_stats
from repro_1p7b.graph_frontier.recursive_opd_v3_complete_c.current_positive import (
    depth_bucket, extract_current_continuation, sample_count_distribution,
)
from repro_1p7b.graph_frontier.recursive_opd_v3_complete_c.prepare import _main_label


def gold():
    return {
        "initial_state": {"value": 0},
        "actions": [
            {"name": "tool-a", "arguments": {"value": 1}},
            {"name": "tool-b", "arguments": {"value": 2}},
            {"name": "tool-c", "arguments": {"value": 3}},
        ],
        "observations": [{"ok": 1}, {"ok": 2}, {"ok": 3}],
        "states_after": [{"value": 1}, {"value": 2}, {"value": 3}],
        "final_state": {"value": 3},
    }


def make_tool_step(index, pointer, count=1, *, exact=True):
    value_pointer = pointer if exact else min(pointer + 1, 2)
    calls = []
    events = []
    tool_calls = []
    for offset in range(count):
        action = gold()["actions"][value_pointer + offset]
        call = {"name": action["name"], "arguments": action["arguments"],
                "id": f"call-{index}-{offset}", "position": offset}
        calls.append(call)
        tool_calls.append({"id": call["id"], "type": "function",
                           "function": {"name": call["name"],
                                        "arguments": call["arguments"]}})
        expected = pointer + offset
        events.append({"tool_name": call["name"], "tool_arguments": call["arguments"],
                       "execution_success": True, "exception": None,
                       "tool_response": gold()["observations"][expected]})
    return {
        "step_index": index, "parse_errors": [], "structured_calls": calls,
        "typed_events": events,
        "model_message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
        "state_before_action": (gold()["initial_state"] if pointer == 0
                                else gold()["states_after"][pointer - 1]),
        "state_after_action": gold()["states_after"][pointer + count - 1],
    }


def make_stop_step(index, pointer):
    return {
        "step_index": index, "parse_errors": [], "structured_calls": [], "typed_events": [],
        "model_message": {"role": "assistant", "content": "", "tool_calls": []},
        "state_before_action": (gold()["initial_state"] if pointer == 0
                                else gold()["states_after"][pointer - 1]),
        "state_after_action": (gold()["initial_state"] if pointer == 0
                               else gold()["states_after"][pointer - 1]),
    }


def episode(steps):
    messages = [{"role": "user", "content": "query"}]
    for step in steps:
        step["prompt_message_count"] = len(messages)
        messages.append(step["model_message"])
        for call in step["structured_calls"]:
            messages.append({"role": "tool", "name": call["name"],
                             "tool_call_id": call["id"], "content": "{}"})
    return {"task_id": "task-1", "gold": gold(),
            "student": {"status": "ROLLOUT_COMPLETE", "steps": steps,
                        "messages": messages, "termination_reason": "MODEL_FINAL"}}


def first_error(pointer, step_index, regime="ADVANCE", reason="wrong tool"):
    return {"task_id": "task-1", "status": "FIRST_ERROR", "frontier_depth": pointer,
            "student_step": step_index, "error_regime": regime, "reason": reason,
            "termination_reason": "MODEL_FINAL"}


def success():
    return {"task_id": "task-1", "status": "SUCCESS", "frontier_depth": 3,
            "student_step": None, "error_regime": None, "termination_reason": "MODEL_FINAL"}


def test_v2_main_labels_are_unchanged_semantically():
    assert _main_label({"correction_kind": "ADVANCE"}) == "CURRENT_FRONTIER_CORRECTION"
    assert _main_label({"correction_kind": "SUCCESS_TERMINAL_POSITIVE"}) == \
        "SUCCESS_TERMINAL_POSITIVE"


def test_first_action_error_has_no_continuation():
    out = extract_current_continuation(first_error(0, 0, "START"), episode([make_stop_step(0, 0)]))
    assert out["status"] == "NO_VERIFIED_PRETERMINAL_CONTINUATION"


def test_one_correct_step_then_error_has_one_student_positive():
    steps = [make_tool_step(0, 0), make_stop_step(1, 1)]
    out = extract_current_continuation(first_error(1, 1), episode(steps))
    assert out["status"] == "CURRENT_CONTINUATION_ELIGIBLE"
    assert out["student_action"] == steps[0]["model_message"]
    assert out["progress_pointer_before"] == 0 and out["progress_pointer_after"] == 1


def test_multistep_prefix_selects_only_nearest_continuation():
    steps = [make_tool_step(0, 0), make_tool_step(1, 1), make_stop_step(2, 2)]
    out = extract_current_continuation(first_error(2, 2), episode(steps))
    assert out["turn_index"] == 1
    assert out["student_action"] == steps[1]["model_message"]
    assert out["distance_to_main_frontier"] == 1


def test_success_trajectory_yields_nearest_preterminal_action():
    steps = [make_tool_step(0, 0), make_tool_step(1, 1), make_tool_step(2, 2)]
    out = extract_current_continuation(success(), episode(steps))
    assert out["turn_index"] == 2 and out["progress_pointer_before"] == 2
    assert out["terminal_flag"] is False


def test_premature_stop_keeps_prior_verified_continuation():
    steps = [make_tool_step(0, 0), make_tool_step(1, 1), make_stop_step(2, 2)]
    row = first_error(2, 2, reason="premature model stop")
    assert extract_current_continuation(row, episode(steps))["turn_index"] == 1


def test_over_continuation_keeps_tool_before_terminal():
    steps = [make_tool_step(0, 0), make_tool_step(1, 1), make_tool_step(2, 2),
             make_stop_step(3, 3)]
    row = first_error(3, 3, regime="TERMINATE", reason="tool after terminal")
    assert extract_current_continuation(row, episode(steps))["turn_index"] == 2


def test_verified_parallel_action_is_retained_whole_and_not_reordered():
    steps = [make_tool_step(0, 0, count=2), make_stop_step(1, 2)]
    out = extract_current_continuation(first_error(2, 1), episode(steps))
    assert out["status"] == "CURRENT_CONTINUATION_ELIGIBLE"
    assert out["parallel_call_count"] == 2
    assert out["student_action"] == steps[0]["model_message"]


def test_uncertain_parallel_or_alternative_is_skipped_fail_closed():
    steps = [make_tool_step(0, 0, count=1, exact=False), make_stop_step(1, 0)]
    out = extract_current_continuation(first_error(0, 1, "START"), episode(steps))
    assert out["status"] == "CONTINUATION_UNCERTAIN"


def test_gold_only_state_without_student_visit_is_rejected():
    out = extract_current_continuation(first_error(1, 1), episode([]))
    assert out["status"] == "CONTINUATION_UNCERTAIN"


def test_exact_prefix_stops_before_own_action_and_observation():
    steps = [make_tool_step(0, 0), make_tool_step(1, 1), make_stop_step(2, 2)]
    out = extract_current_continuation(first_error(2, 2), episode(steps))
    assert out["conversation_prefix"][-1]["role"] == "tool"
    assert steps[1]["model_message"] not in out["conversation_prefix"]
    assert len(out["conversation_prefix"]) == steps[1]["prompt_message_count"]


def test_student_action_and_all_required_hash_provenance_are_exact():
    steps = [make_tool_step(0, 0), make_stop_step(1, 1)]
    out = extract_current_continuation(first_error(1, 1), episode(steps))
    assert out["student_action_sha256"] == digest(steps[0]["model_message"])
    assert out["prefix_sha256"] == digest(out["conversation_prefix"])
    assert out["state_before_sha256"] == digest(gold()["initial_state"])
    assert out["state_after_sha256"] == digest(gold()["states_after"][0])


def test_ambiguous_turn_blocks_fallback_to_an_older_positive():
    steps = [make_tool_step(0, 0), make_tool_step(1, 1, exact=False), make_stop_step(2, 1)]
    out = extract_current_continuation(first_error(1, 2), episode(steps))
    assert out["status"] == "CONTINUATION_UNCERTAIN"


def test_authoritative_pointer_mismatch_is_rejected():
    steps = [make_tool_step(0, 0), make_stop_step(1, 1)]
    out = extract_current_continuation(first_error(2, 1), episode(steps))
    assert out["status"] == "CONTINUATION_UNCERTAIN"


def test_max_two_rows_and_heldout_exclusion():
    assert sample_count_distribution({"a", "b"}, {"b", "c"}, {"a", "b", "c"}) == {
        "0": 0, "1": 2, "2": 1}
    with pytest.raises(ValueError, match="heldout"):
        sample_count_distribution({"heldout"}, set(), {"train"})


def test_depth_distance_and_context_overflow_accounting_buckets():
    assert [depth_bucket(x) for x in (0, 1, 2, 3, 7)] == [
        "START-adjacent", "1", "2", "3+", "3+"]
    stats = length_stats([8192, 8193, 12289, 16385, 24577, 32769])
    assert stats[">8192"] == 5 and stats[">16384"] == 3 and stats[">32768"] == 1


def test_training_launcher_has_no_implicit_budget():
    launcher = Path(__file__).parents[1] / "recursive_opd_v3_complete_c/run_gpu_train.sh"
    result = subprocess.run(["bash", str(launcher)], text=True, capture_output=True)
    assert result.returncode != 0
    assert "UNDECIDED" in result.stderr
