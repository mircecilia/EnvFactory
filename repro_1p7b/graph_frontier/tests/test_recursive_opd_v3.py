from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import digest
from repro_1p7b.graph_frontier.recursive_opd_v3.lineage import (
    depth_bucket, historical_identity, latest_relevant_frontier, lineage_matches,
    movement, sample_count_distribution, verified_student_crossing,
)
from repro_1p7b.graph_frontier.recursive_opd_v3.prepare import length_stats


def gold():
    return {
        "initial_state": {"value": 0},
        "actions": [
            {"name": "tool-a", "arguments": {"value": 1}},
            {"name": "tool-b", "arguments": {"value": 2}},
        ],
        "observations": [{"ok": 1}, {"ok": 2}],
        "states_after": [{"value": 1}, {"value": 2}],
        "final_state": {"value": 2},
    }


def record(cycle=0, pointer=0, regime="START", status="FIRST_ERROR"):
    action = gold()["actions"][pointer] if status == "FIRST_ERROR" else None
    return {"task_id": "task-1", "query_hash": "query-1", "cycle": cycle,
            "pool": "train", "status": status, "error_regime": regime,
            "frontier_depth": pointer, "gold_next_action": action,
            "gold_tool_count": 2, "termination_reason": "MODEL_FINAL"}


def step(index, pointer, *, calls=None, before=None, after=None, response=None):
    calls = calls if calls is not None else [{"name": gold()["actions"][pointer]["name"],
                                               "arguments": gold()["actions"][pointer]["arguments"],
                                               "id": f"call-{index}", "position": 0}]
    message = {"role": "assistant", "content": "", "tool_calls": [
        {"id": call.get("id", "call"), "type": "function",
         "function": {"name": call["name"], "arguments": call["arguments"]}}
        for call in calls]}
    return {"step_index": index, "prompt_message_count": 1 + index * 2,
            "model_message": message, "parse_errors": [], "structured_calls": calls,
            "typed_events": [{"tool_name": call["name"], "tool_arguments": call["arguments"],
                              "execution_success": True, "exception": None,
                              "tool_response": (response if response is not None
                                                else gold()["observations"][pointer + offset])}
                             for offset, call in enumerate(calls)],
            "state_before_action": before if before is not None else (
                gold()["initial_state"] if pointer == 0 else gold()["states_after"][pointer - 1]),
            "state_after_action": after if after is not None else
                gold()["states_after"][pointer + len(calls) - 1]}


def episode(steps=None):
    steps = steps if steps is not None else [step(0, 0), step(1, 1)]
    messages = [{"role": "user", "content": "query"}]
    for item in steps:
        messages.append(item["model_message"])
        messages.append({"role": "tool", "name": item["structured_calls"][0]["name"],
                         "tool_call_id": item["structured_calls"][0].get("id", "call"),
                         "content": "{}"})
    return {"task_id": "task-1", "cycle": 1, "pool": "train",
            "gold_source_episode_sha256": "gold-sha", "gold": gold(),
            "student": {"status": "ROLLOUT_COMPLETE", "steps": steps,
                        "messages": messages, "termination_reason": "MODEL_FINAL"}}


def identity(pointer=0, regime="START"):
    source = episode([])
    source["cycle"] = 0
    return historical_identity(record(pointer=pointer, regime=regime), source,
                               metrics_sha256="metrics", rollout_sha256="rollout",
                               rollout_manifest_sha256="manifest", source_policy_sha256="policy")


def test_historical_frontier_identity_is_stable_and_complete():
    value = identity()
    assert value["state_sha256"] == digest(gold()["initial_state"])
    assert value["target_action_sha256"] == digest(gold()["actions"][0])
    assert value["source_policy_sha256"] == "policy"


def test_latest_relevant_frontier_does_not_accumulate_ancestors():
    rows = [record(cycle=0), record(cycle=1, pointer=1, regime="ADVANCE")]
    assert latest_relevant_frontier(rows, task_id="task-1", current_cycle=2)["cycle"] == 1


def test_duplicate_latest_frontier_is_ambiguous():
    rows = [record(cycle=1), record(cycle=1)]
    with pytest.raises(ValueError, match="AMBIGUOUS_LINEAGE"):
        latest_relevant_frontier(rows, task_id="task-1", current_cycle=2)


def test_terminate_correction_is_not_continuation_lineage():
    row = record(pointer=1, regime="TERMINATE")
    assert latest_relevant_frontier([row], task_id="task-1", current_cycle=2) is None
    with pytest.raises(ValueError):
        historical_identity(row, episode([]), metrics_sha256="m", rollout_sha256="r",
                            rollout_manifest_sha256="rm", source_policy_sha256="p")


def test_start_crossing_uses_current_student_action_and_prefix():
    current = record(cycle=1, pointer=1, regime="ADVANCE")
    current_episode = episode()
    result = verified_student_crossing(identity(), current, current_episode)
    assert result["status"] == "RETENTION_ELIGIBLE"
    assert result["student_action"] == gold()["actions"][0]
    assert result["conversation_prefix"] == [{"role": "user", "content": "query"}]


def test_advance_crossing_is_supported():
    current = {**record(cycle=2, pointer=2, regime=None, status="SUCCESS"),
               "frontier_depth": 2}
    result = verified_student_crossing(identity(1, "ADVANCE"), current, episode())
    assert result["status"] == "RETENTION_ELIGIBLE"
    assert result["current_student_step"] == 1


def test_regressed_and_uncertain_frontier_detection():
    assert movement(identity(), record(cycle=1, pointer=0)) == "REGRESSED"
    uncertain = {**record(cycle=1), "status": "UNCERTAIN", "error_regime": None}
    assert movement(identity(), uncertain) == "UNCERTAIN"


def test_multicall_crossing_is_not_fabricated_into_single_call_target():
    calls = [{"name": "tool-a", "arguments": {"value": 1}, "id": "a", "position": 0},
             {"name": "tool-b", "arguments": {"value": 2}, "id": "b", "position": 1}]
    current_episode = episode([step(0, 0, calls=calls, after=gold()["states_after"][1])])
    current = {**record(cycle=1, pointer=2, regime=None, status="SUCCESS"),
               "frontier_depth": 2}
    result = verified_student_crossing(identity(), current, current_episode)
    assert result["status"] == "AMBIGUOUS_LINEAGE"


def test_off_gold_or_gold_only_state_is_rejected():
    current = record(cycle=1, pointer=1, regime="ADVANCE")
    bad = episode([step(0, 0, before={"value": 999})])
    assert verified_student_crossing(identity(), current, bad)["status"] == "UNCERTAIN_RETENTION"
    assert verified_student_crossing(identity(), current, episode([]))["status"] == "UNCERTAIN_RETENTION"


def test_lineage_requires_same_query_task_and_gold_source():
    source, current = episode([]), episode()
    source["cycle"] = 0
    assert lineage_matches(identity(), source, record(cycle=1, pointer=1, regime="ADVANCE"), current)
    current["gold_source_episode_sha256"] = "different"
    assert not lineage_matches(identity(), source, record(cycle=1, pointer=1, regime="ADVANCE"), current)


def test_max_two_natural_samples_per_task_and_heldout_rejection():
    assert sample_count_distribution({"a", "b"}, {"b", "c"}, {"a", "b", "c"}) == {
        "0": 0, "1": 2, "2": 1}
    with pytest.raises(ValueError):
        sample_count_distribution({"heldout"}, set(), {"train"})


def test_depth_and_context_statistics_cover_required_buckets():
    assert [depth_bucket(value) for value in (0, 1, 2, 3, 9)] == ["0", "1", "2", "3+", "3+"]
    stats = length_stats([100, 9000, 13000, 17000, 25000, 33000])
    assert stats[">8192"] == 5 and stats[">16384"] == 3 and stats[">32768"] == 1


def test_gpu_launcher_refuses_undecided_budget():
    launcher = Path(__file__).parents[1] / "recursive_opd_v3/run_gpu_train.sh"
    result = subprocess.run(["bash", str(launcher)], text=True, capture_output=True)
    assert result.returncode != 0
    assert "UNDECIDED" in result.stderr
