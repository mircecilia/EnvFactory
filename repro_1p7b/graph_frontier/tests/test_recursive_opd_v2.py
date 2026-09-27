"""CPU-only regression tests for exact on-policy success EOS and lineage gates."""
from __future__ import annotations

import copy
import asyncio
import json

import pytest
import requests

from repro_1p7b.graph_frontier.recursive_opd_v2 import prepare
from repro_1p7b.graph_frontier.recursive_opd_v2 import future
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import (
    account_context, context_class, distribution,
)
from repro_1p7b.graph_frontier.recursive_opd_v2.train import validate_context_manifest
from repro_1p7b.graph_frontier.recursive_opd_v2.compare import (
    outcome, paired, pool_stats,
)
from repro_1p7b.graph_frontier.recursive_opd_v2.context_repair import context_rejection
from repro_1p7b.graph_frontier.recursive_opd_v1.mine import classify


class FakeTokenizer:
    eos_token_id = 151645

    def apply_chat_template(self, messages, *, add_generation_prompt, **_):
        # A deterministic synthetic template with exact prefix/suffix behavior.
        prefix = [10] + [20 + i for i in range(len(messages))] + [30]
        if add_generation_prompt:
            return prefix
        target = messages[-1]
        prompt = [10] + [20 + i for i in range(len(messages) - 1)] + [30]
        return prompt + ([111, 112] if target.get("tool_calls") else [self.eos_token_id])

    def decode(self, ids, *, skip_special_tokens):
        if ids == [111, 112]:
            return "<tool_call>{}</tool_call>"
        if ids in ([], [151645]):
            return ""
        raise AssertionError(ids)


def fixture():
    action = {"name": "S-A", "arguments": {"x": 1}}
    initial, final = {"S": {"v": 0}}, {"S": {"v": 1}}
    user = {"role": "user", "content": "q"}
    tool_call = {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
        "function": {"name": "S-A", "arguments": {"x": 1}}}]}
    observation = {"role": "tool", "name": "S-A", "tool_call_id": "c", "content": "result"}
    final_message = {"role": "assistant", "content": "done"}
    steps = [
        {"step_index": 0, "prompt_message_count": 1, "model_message": tool_call,
         "state_before_action": initial, "structured_calls": [{"name": "S-A", "arguments": {"x": 1}}]},
        {"step_index": 1, "prompt_message_count": 3, "model_message": final_message,
         "state_before_action": final, "structured_calls": [], "parse_errors": []},
    ]
    episode = {"gold": {"actions": [action], "initial_state": initial,
                        "states_after": [final], "final_state": final},
               "student": {"status": "ROLLOUT_COMPLETE", "messages": [user, tool_call, observation, final_message],
                           "steps": steps, "termination_reason": "MODEL_FINAL", "final_state": final}}
    success = {"status": "SUCCESS", "error_regime": None, "frontier_depth": 1,
               "gold_tool_count": 1, "task_id": "task", "query_hash": "qhash",
               "termination_reason": "MODEL_FINAL"}
    schema = [{"type": "function", "function": {"name": "S-A", "parameters": {"type": "object"}}}]
    return episode, success, schema


def test_success_uses_exact_student_prefix_and_only_eos_label():
    episode, record, schema = fixture()
    prefix = prepare.checked_prefix(record, episode)
    assert prefix == episode["student"]["messages"][:-1]
    assert prefix[-1]["role"] == "tool"
    assert episode["student"]["messages"][-1] not in prefix
    row = prepare.encode_row(FakeTokenizer(), record, episode, schema, "goldsha")
    assert row["correction_kind"] == "SUCCESS_TERMINAL_POSITIVE"
    assert row["source_verdict"] == "SUCCESS"
    assert row["target_type"] == "terminal_stop"
    assert row["labels"].count(151645) == 1
    assert sum(label != -100 for label in row["labels"]) == 1
    assert row["labels"][:row["prompt_tokens"]] == [-100] * row["prompt_tokens"]


def test_success_rejects_including_generated_final_answer():
    episode, record, _ = fixture()
    episode["student"]["steps"][-1]["prompt_message_count"] = 4
    with pytest.raises(RuntimeError, match="boundary|prefix"):
        prepare.checked_prefix(record, episode)


def test_success_rejects_unverified_terminal_and_continuation():
    episode, record, _ = fixture()
    episode["student"]["steps"][-1]["state_before_action"] = {"S": {"v": 9}}
    with pytest.raises(RuntimeError, match="SUCCESS terminal"):
        prepare.checked_prefix(record, episode)
    episode, record, _ = fixture()
    episode["student"]["steps"][-1]["model_message"]["tool_calls"] = [{"id": "extra"}]
    with pytest.raises(RuntimeError, match="SUCCESS terminal"):
        prepare.checked_prefix(record, episode)


def test_terminate_correction_distinct_from_success_positive():
    episode, record, schema = fixture()
    extra = {"role": "assistant", "tool_calls": [{"id": "extra", "function": {"name": "S-A", "arguments": {"x": 1}}}]}
    episode["student"]["messages"][-1] = extra
    episode["student"]["steps"][-1]["model_message"] = extra
    record.update(status="FIRST_ERROR", error_regime="TERMINATE", student_step=1,
                  prompt_messages=episode["student"]["messages"][:3])
    prefix = prepare.checked_prefix(record, episode)
    assert prefix[-1]["role"] == "tool"
    row = prepare.encode_row(FakeTokenizer(), record, episode, schema, "goldsha")
    assert row["correction_kind"] == "TERMINATE"
    assert row["source_verdict"] == "FIRST_ERROR"
    assert row["target_type"] == "terminal_stop"


def test_first_error_gold_pointer_and_student_prefix():
    episode, record, schema = fixture()
    record.update(status="FIRST_ERROR", error_regime="START", student_step=0,
                  frontier_depth=0, gold_next_action=episode["gold"]["actions"][0],
                  prompt_messages=episode["student"]["messages"][:1])
    row = prepare.encode_row(FakeTokenizer(), record, episode, schema, "goldsha")
    assert row["target_type"] == "first_divergence_tool"
    assert row["correction_kind"] == "START"
    bad = copy.deepcopy(record)
    bad["gold_next_action"]["arguments"]["x"] = 999
    with pytest.raises(RuntimeError, match="gold-next-action"):
        prepare.checked_prefix(bad, episode)


def test_uncertain_never_encodes():
    episode, record, _ = fixture()
    record["status"] = "UNCERTAIN"
    with pytest.raises(RuntimeError, match="UNCERTAIN"):
        prepare.checked_prefix(record, episode)


@pytest.mark.parametrize("message,expected_input", [
    ("Requested token count exceeds the model's maximum context length of 32768 tokens. "
     "You requested a total of 32826 tokens: 31802 tokens from the input messages "
     "and 1024 tokens for the completion.", 31802),
    ("The input (181198 tokens) is longer than the model's context length "
     "(32768 tokens).", 181198),
])
def test_only_explicit_fixed_context_400_is_typed(message, expected_input):
    response = requests.Response()
    response.status_code = 400
    response._content = json.dumps({"message": message}).encode()
    error = requests.HTTPError("bad request", response=response)
    proof = context_rejection(error)
    assert proof["context_limit_tokens"] == 32768
    assert proof["requested_input_tokens"] == expected_input
    response._content = b'{"message":"invalid tool schema"}'
    assert context_rejection(error) is None
    response.status_code = 500
    assert context_rejection(error) is None


def test_context_boundary_after_gold_prefix_is_uncertain_not_terminal_success():
    episode, _, schema = fixture()
    episode["gold"]["observations"] = ["result"]
    episode["student"]["steps"] = episode["student"]["steps"][:1]
    episode["student"]["steps"][0].update(
        typed_events=[{"execution_success": True, "tool_response": "result"}],
        state_after_action=episode["gold"]["final_state"], parse_errors=[])
    episode["student"]["messages"] = episode["student"]["messages"][:3]
    episode["student"].update(status="ROLLOUT_CONTEXT_OVERFLOW",
                              termination_reason="CONTEXT_OVERFLOW")
    row = {"extra_info": {"mcp_factory_kwargs": {
        "final_config": episode["gold"]["final_state"], "mcp_servers": ["S"]}}}
    verdict = asyncio.run(classify(episode, {"audit_id": "task"}, row, schema))
    assert verdict["status"] == "UNCERTAIN"


def test_context_boundary_preserves_only_pre_boundary_first_error():
    episode, record, _ = fixture()
    episode["student"].update(
        status="ROLLOUT_CONTEXT_OVERFLOW", termination_reason="CONTEXT_OVERFLOW",
        context_overflow={"step_index": len(episode["student"]["steps"])})
    record.update(status="FIRST_ERROR", error_regime="START", student_step=0,
                  frontier_depth=0, gold_next_action=episode["gold"]["actions"][0],
                  prompt_messages=episode["student"]["messages"][:1])
    assert prepare.checked_prefix(record, episode) == episode["student"]["messages"][:1]
    record["status"] = "SUCCESS"
    with pytest.raises(RuntimeError, match="earlier verified first error"):
        prepare.checked_prefix(record, episode)
    record["status"] = "FIRST_ERROR"
    episode["student"]["context_overflow"]["step_index"] = 0
    with pytest.raises(RuntimeError, match="earlier verified first error"):
        prepare.checked_prefix(record, episode)


def test_v1_later_policy_fails_closed(monkeypatch):
    monkeypatch.setattr(prepare, "sha256", lambda _: "protocol_hash")
    protocol = {"pi0_model_sha256": "pi0", "pi0_model": "/tmp/pi0"}
    manifest = {"cycle": 1, "protocol_sha256": "protocol_hash", "model_sha256": "v1_pi1"}
    with pytest.raises(RuntimeError, match="RETROSPECTIVE_ONLY"):
        prepare.source_policy_gate(1, manifest, protocol)
    with pytest.raises(RuntimeError, match="previous V2"):
        prepare.source_policy_gate(1, manifest, protocol,
            {"lineage": "recursive_opd_v1", "cycle": 0, "status": "TRAINED", "model_sha256": "v1_pi1"})
    assert prepare.source_policy_gate(1, manifest, protocol,
        {"lineage": "recursive_opd_v2", "cycle": 0, "status": "TRAINED",
         "model_sha256": "v1_pi1"}) == "TRAINABLE_V2_D1"


def test_future_rollout_rejects_v1_checkpoint_lineage(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(future, "RUN", tmp_path)
    monkeypatch.setattr(future, "sha256", lambda _: "fixed_model_hash")
    checkpoint = tmp_path / "cycle1/checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"mock")
    state = {"lineage": "recursive_opd_v1", "status": "TRAINED", "cycle": 0,
             "model_sha256": "fixed_model_hash"}
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="V2-trained checkpoint"):
        future.previous_v2(1, checkpoint)
    state["lineage"] = "recursive_opd_v2"
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    assert future.previous_v2(1, checkpoint)["lineage"] == "recursive_opd_v2"


def test_context_boundary_is_label_agnostic():
    assert context_class(16383) == "CONTEXT_COMPATIBLE"
    assert context_class(16384) == "CONTEXT_COMPATIBLE"
    assert context_class(16385) == "CONTEXT_OVERFLOW"
    assert distribution([100, 16384, 26038])["above"]["16384"] == 1


@pytest.mark.parametrize("length,passes", [(16383, True), (16384, True), (16385, False)])
def test_success_eos_encoding_at_context_boundary(length, passes):
    class SizedTerminalTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, *, add_generation_prompt, **_):
            prompt = [10] * (length - 1)
            return prompt if add_generation_prompt else prompt + [self.eos_token_id]

    episode, record, schema = fixture()
    if not passes:
        with pytest.raises(ValueError, match="context exceeds 16k"):
            prepare.encode_row(SizedTerminalTokenizer(), record, episode, schema, "goldsha")
    else:
        row = prepare.encode_row(SizedTerminalTokenizer(), record, episode, schema, "goldsha")
        assert row["total_tokens"] == length
        assert row["target_tokens"] == 1
        assert sum(label != -100 for label in row["labels"]) == 1


def context_fixture():
    row = {"task_id": "t1", "input_ids": [1] * 3,
           "semantic_eligible": True, "execution_verified": True,
           "train_context_compatible": True}
    overflow = {"task_id": "t2", "full_tokens": 26038, "reason": "CONTEXT_OVERFLOW",
                "semantic_eligible": True, "execution_verified": True,
                "train_context_compatible": False}
    manifest = {"schema_version": "recursive_opd_v2_context_accounted_1",
                "max_train_context": 16384, "semantic_eligible_count": 2,
                "context_compatible_count": 1, "trainable_count": 1, "encoded_rows": 1,
                "context_overflow_count": 1, "context_overflow_records": [overflow],
                "heldout_training_rows": 0, "uncertain_training_rows": 0,
                "skipped_uncertain_count": 1}
    return manifest, [row]


def test_explicit_overflow_accounting_and_trainer_allow():
    manifest, rows = context_fixture()
    account_context(semantic_eligible=2, encoded=1,
                    overflow=manifest["context_overflow_records"])
    assert validate_context_manifest(manifest, rows, cycle=0) == 1


@pytest.mark.parametrize("change", [
    lambda m, r: m.update(semantic_eligible_count=1),
    lambda m, r: m.update(context_overflow_records=[]),
    lambda m, r: m.update(heldout_training_rows=1),
    lambda m, r: m.update(uncertain_training_rows=1),
    lambda m, r: r[0].update(train_context_compatible=False),
    lambda m, r: r[0].update(input_ids=[1] * 16385),
    lambda m, r: m["context_overflow_records"][0].update(reason="MANUAL_DROP"),
])
def test_trainer_rejects_context_manifest_drift(change):
    manifest, rows = context_fixture()
    change(manifest, rows)
    with pytest.raises(RuntimeError, match="context|accounting"):
        validate_context_manifest(manifest, rows, cycle=0)


def test_non_overflow_dataset_at_exact_limit_passes():
    manifest, rows = context_fixture()
    rows[0]["input_ids"] = [1] * 16384
    manifest.update(semantic_eligible_count=1, context_overflow_count=0,
                    context_overflow_records=[])
    assert validate_context_manifest(manifest, rows, cycle=0) == 0


def test_later_recursive_cycle_keeps_identical_context_gate():
    manifest, rows = context_fixture()
    assert validate_context_manifest(manifest, rows, cycle=3) == 1


def test_context_repair_accepts_variable_failed_count_only_with_unique_ids(tmp_path, monkeypatch):
    from repro_1p7b.graph_frontier.recursive_opd_v2 import context_repair

    monkeypatch.setattr(context_repair, "RUN", tmp_path)
    root = tmp_path / "cycle4"
    root.mkdir()
    original = {"status": "ROLLOUT_COMPLETE", "cycle": 4, "task_count": 2158,
                "student_error_count": 2, "student_error_task_ids": ["a", "b"]}
    path = root / "rollout_integrity.json"
    path.write_text(json.dumps(original))
    assert context_repair.original_errors(4)[0]["student_error_count"] == 2
    original["student_error_task_ids"] = ["a", "a"]
    path.write_text(json.dumps(original))
    with pytest.raises(RuntimeError, match="immutable original integrity"):
        context_repair.original_errors(4)


def test_internal_comparison_counts_terminal_vs_premature_stop():
    base = {"pool": "heldout", "gold_tool_count": 2}
    rows = {
        "a": {**base, "status": "SUCCESS", "frontier_depth": 2,
              "error_regime": None, "reason": "gold progress and termination verified"},
        "b": {**base, "status": "FIRST_ERROR", "frontier_depth": 2,
              "error_regime": "TERMINATE", "reason": "tool call after verified terminal state"},
        "c": {**base, "status": "FIRST_ERROR", "frontier_depth": 1,
              "error_regime": "ADVANCE", "reason": "premature model stop"},
    }
    metrics = pool_stats(rows, ["a", "b", "c"])
    assert metrics["success"] == 1
    assert metrics["reached_verified_terminal"] == 2
    assert metrics["over_continuation"] == 1
    assert metrics["premature_stop"] == 1
    assert metrics["first_action_correct"] == 3
    assert outcome(rows["b"]) == "TERMINATE"
    next_rows = {key: {**row, "frontier_depth": 2, "status": "SUCCESS",
                       "error_regime": None} for key, row in rows.items()}
    movement = paired(rows, next_rows, ["a", "b", "c"])
    assert movement["movement"] == {"FORWARD": 1, "SAME": 2, "BACKWARD": 0}
    assert movement["migration"]["TERMINATE"]["SUCCESS"] == 1
