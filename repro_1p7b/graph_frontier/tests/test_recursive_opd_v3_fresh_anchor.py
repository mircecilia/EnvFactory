from __future__ import annotations

import pytest

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import encode_assistant_target
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import digest
from repro_1p7b.graph_frontier.recursive_opd_v3.prepare import length_stats
from repro_1p7b.graph_frontier.recursive_opd_v3_fresh_anchor.anchor import (
    descriptive_exposure, exact_overlap, extract_fresh_anchor, fresh_identity,
    semantic_count_distribution,
)


def gold():
    return {
        "initial_state": {"value": 0},
        "actions": [{"name": "a", "arguments": {"x": 1}},
                    {"name": "b", "arguments": {"x": 2}}],
        "observations": [{"ok": 1}, {"ok": 2}],
        "states_after": [{"value": 1}, {"value": 2}],
        "final_state": {"value": 2},
    }


def record(cycle, pointer, regime="START", status="FIRST_ERROR"):
    return {"task_id": "task", "query_hash": "query", "cycle": cycle, "pool": "train",
            "status": status, "error_regime": regime, "frontier_depth": pointer,
            "gold_next_action": gold()["actions"][pointer] if status == "FIRST_ERROR" else None,
            "termination_reason": "MODEL_FINAL"}


def step(index, pointer):
    action = gold()["actions"][pointer]
    message = {"role": "assistant", "content": "", "tool_calls": [{
        "id": f"call-{index}", "type": "function",
        "function": {"name": action["name"], "arguments": action["arguments"]}}]}
    return {"step_index": index, "prompt_message_count": 1 + index * 2,
            "model_message": message, "parse_errors": [],
            "structured_calls": [{"name": action["name"], "arguments": action["arguments"],
                                  "id": f"call-{index}", "position": 0}],
            "typed_events": [{"tool_name": action["name"], "tool_arguments": action["arguments"],
                              "execution_success": True, "exception": None,
                              "tool_response": gold()["observations"][pointer]}],
            "state_before_action": (gold()["initial_state"] if pointer == 0
                                    else gold()["states_after"][pointer - 1]),
            "state_after_action": gold()["states_after"][pointer]}


def episode(cycle, steps):
    messages = [{"role": "user", "content": "query"}]
    for item in steps:
        messages.append(item["model_message"])
        messages.append({"role": "tool", "name": item["structured_calls"][0]["name"],
                         "tool_call_id": item["structured_calls"][0]["id"], "content": "{}"})
    return {"task_id": "task", "cycle": cycle, "pool": "train",
            "gold_source_episode_sha256": "gold-sha", "gold": gold(),
            "student": {"status": "ROLLOUT_COMPLETE", "steps": steps,
                        "messages": messages, "termination_reason": "MODEL_FINAL"}}


def identity(source_cycle=0, crossed_cycle=1, pointer=0, regime="START"):
    return fresh_identity(
        record(source_cycle, pointer, regime), episode(source_cycle, []),
        crossed_cycle=crossed_cycle, metrics_sha256="metrics", rollout_sha256="source-rollout",
        rollout_manifest_sha256="source-manifest", source_policy_sha256="source-policy",
        source_dataset_sha256="source-dataset")


def crossed_record(cycle=1, depth=1, status="FIRST_ERROR"):
    regime = "ADVANCE" if status == "FIRST_ERROR" and depth < 2 else None
    return {**record(cycle, min(depth, 1), regime or "ADVANCE", status),
            "frontier_depth": depth, "error_regime": regime}


def test_previous_cycle_start_crossing_becomes_anchor():
    out = extract_fresh_anchor(identity(), episode(0, []), crossed_record(), episode(1, [step(0, 0)]))
    assert out["status"] == "FRESH_ANCHOR_ELIGIBLE"
    assert out["progress_before"] == 0 and out["progress_after"] == 1


def test_previous_cycle_advance_crossing_becomes_anchor():
    current = crossed_record(cycle=2, depth=2, status="SUCCESS")
    out = extract_fresh_anchor(identity(1, 2, 1, "ADVANCE"), episode(1, []), current,
                               episode(2, [step(0, 0), step(1, 1)]))
    assert out["status"] == "FRESH_ANCHOR_ELIGIBLE"
    assert out["current_student_step"] == 1


def test_terminate_frontier_never_creates_anchor():
    with pytest.raises(ValueError):
        identity(0, 1, 1, "TERMINATE")


def test_frontier_not_crossed_creates_no_anchor():
    out = extract_fresh_anchor(identity(), episode(0, []), record(1, 0), episode(1, []))
    assert out["status"] == "FRONTIER_NOT_CROSSED"


def test_uncertain_crossing_is_skipped():
    current = {**record(1, 0), "status": "UNCERTAIN"}
    assert extract_fresh_anchor(identity(), episode(0, []), current,
                                episode(1, []))["status"] == "ANCHOR_UNCERTAIN"


def test_student_action_not_gold_copy_is_preserved_with_provenance():
    current_episode = episode(1, [step(0, 0)])
    out = extract_fresh_anchor(identity(), episode(0, []), crossed_record(), current_episode)
    assert out["student_action"] == current_episode["student"]["steps"][0]["model_message"]
    assert out["current_action_hash"] == digest(out["student_action"])
    assert out["matches_historical_gold"] is True


def test_gold_only_target_without_current_visit_is_rejected():
    out = extract_fresh_anchor(identity(), episode(0, []), crossed_record(), episode(1, []))
    assert out["status"] == "ANCHOR_UNCERTAIN"


def test_anchor_age_and_update_cycle_are_exactly_one():
    value = identity()
    assert value["anchor_age"] == 1
    assert value["crossed_cycle"] == value["source_error_cycle"] + 1
    assert value["intended_replay_cycle"] == value["crossed_cycle"]
    assert value["expires_after_update_cycle"] == value["crossed_cycle"]


@pytest.mark.parametrize("source,crossed", [(0, 2), (1, 1), (2, 1)])
def test_older_same_or_future_cycle_frontier_is_rejected(source, crossed):
    with pytest.raises(ValueError, match="previous cycle"):
        identity(source, crossed)


def test_exact_current_positive_overlap_detection():
    anchor = {"task_id": "t", "current_prefix_hash": "p", "current_action_hash": "a"}
    current = {"task_id": "t", "prefix_sha256": "p", "student_action_sha256": "a"}
    assert exact_overlap(anchor, current) is True


def test_anchor_only_detection():
    anchor = {"task_id": "t", "current_prefix_hash": "p", "current_action_hash": "a"}
    current = {"task_id": "t", "prefix_sha256": "different", "student_action_sha256": "a"}
    assert exact_overlap(anchor, current) is False
    assert exact_overlap(anchor, None) is False


def test_max_three_semantic_entries_and_one_anchor_per_task():
    assert semantic_count_distribution({"a": 2, "b": 1}, {"a", "c"}, {"a", "b", "c"}) == {
        "0": 0, "1": 2, "2": 0, "3": 1, ">3": 0}
    with pytest.raises(ValueError, match="heldout"):
        semantic_count_distribution({"heldout": 1}, set(), {"train"})


def test_context_overflow_accounting_has_all_thresholds():
    stats = length_stats([100, 8193, 12289, 16385, 24577, 32769])
    assert stats[">8192"] == 5 and stats[">16384"] == 3 and stats[">32768"] == 1


def test_prefix_and_state_hashes_are_current_exact_values():
    current_episode = episode(1, [step(0, 0)])
    out = extract_fresh_anchor(identity(), episode(0, []), crossed_record(), current_episode)
    assert out["current_prefix_hash"] == digest([{"role": "user", "content": "query"}])
    assert out["current_state_before_hash"] == digest(gold()["initial_state"])
    assert out["current_state_after_hash"] == digest(gold()["states_after"][0])


class FakeTokenizer:
    eos_token_id = 151645

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        return [10, 11] if add_generation_prompt else [10, 11, 12, 13]

    def decode(self, ids, *, skip_special_tokens=False):
        return "<tool_call>" if ids else ""


def test_anchor_tool_target_loss_mask_only_supervises_suffix():
    encoded = encode_assistant_target(
        FakeTokenizer(), task_id="t", state_identity="s", target_type="first_divergence_tool",
        conversation_prefix=[{"role": "user", "content": "q"}],
        assistant_target={"role": "assistant", "content": "", "tool_calls": [{
            "id": "c", "type": "function", "function": {"name": "a", "arguments": {}}}]},
        tool_schema=[{"type": "function", "function": {"name": "a"}}])
    assert encoded["labels"] == [-100, -100, 12, 13]
    assert encoded["historical_assistant_tokens_in_loss"] == 0
    assert encoded["tool_observation_tokens_in_loss"] == 0


def test_descriptive_exposure_accounting_is_not_a_budget_decision():
    out = descriptive_exposure({"frontier": 75, "terminal": 25}, 9, 128)
    assert out["total_exposure"] == 137
    assert out["natural_expected_category_exposure"] == {"frontier": 96.0, "terminal": 32.0}
    assert "NOT APPROVED" in out["label"]
