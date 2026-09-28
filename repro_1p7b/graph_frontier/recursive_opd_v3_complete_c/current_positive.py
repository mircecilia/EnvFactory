"""Strict mining of current-policy verified continuation positives.

This module never executes an environment or a model.  It consumes the exact
typed trajectory saved by Recursive OPD V2 and admits a student action only
when the persisted verifier evidence proves an on-gold state transition.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import typed_equal
from repro_1p7b.graph_frontier.recursive_opd_v1.mine import exact_parallel_progress
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import digest


def depth_bucket(depth: int) -> str:
    if depth < 0:
        raise ValueError("negative continuation depth")
    if depth == 0:
        return "START-adjacent"
    return str(depth) if depth < 3 else "3+"


def _normalized_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": call.get("name"), "arguments": call.get("arguments")}
            for call in calls]


def _message_calls(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    if (message.get("role") != "assistant"
            or message.get("content") not in ("", None)
            or not isinstance(message.get("tool_calls"), list)
            or not message["tool_calls"]):
        return None
    answer = []
    for call in message["tool_calls"]:
        function = call.get("function") if isinstance(call, dict) else None
        if (not isinstance(function, dict) or not isinstance(function.get("name"), str)
                or not isinstance(function.get("arguments"), dict)):
            return None
        answer.append({"name": function["name"], "arguments": function["arguments"]})
    return answer


def _exact_prefix(student: dict[str, Any], step: dict[str, Any]) -> list[dict[str, Any]] | None:
    messages, count = student.get("messages"), step.get("prompt_message_count")
    if (not isinstance(messages, list) or not isinstance(count, int)
            or count < 1 or count >= len(messages)
            or not typed_equal(messages[count], step.get("model_message"))):
        return None
    return messages[:count]


def extract_current_continuation(record: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    """Return the nearest strict current continuation before the MAIN boundary.

    The scan is fail-closed.  If any earlier assistant decision cannot be
    independently verified as exact progress, an older positive is not chosen
    through that ambiguous turn.  A complete exact parallel assistant action is
    retained as one target; it is never split or reordered.
    """
    if record.get("status") not in {"FIRST_ERROR", "SUCCESS"}:
        return {"status": "CONTINUATION_UNCERTAIN", "reason": "main verdict uncertain"}
    gold, student = episode.get("gold") or {}, episode.get("student") or {}
    actions, observations = gold.get("actions") or [], gold.get("observations") or []
    states = gold.get("states_after") or []
    steps = student.get("steps")
    if (not actions or len(observations) != len(actions) or len(states) != len(actions)
            or student.get("status") not in {"ROLLOUT_COMPLETE", "ROLLOUT_CONTEXT_OVERFLOW"}
            or not isinstance(steps, list)):
        return {"status": "CONTINUATION_UNCERTAIN", "reason": "typed trajectory absent"}
    main_step = record.get("student_step") if record.get("status") == "FIRST_ERROR" else None
    if record.get("status") == "FIRST_ERROR" and not isinstance(main_step, int):
        return {"status": "CONTINUATION_UNCERTAIN", "reason": "main decision boundary absent"}

    pointer = 0
    verified: list[dict[str, Any]] = []
    for sequence_index, step in enumerate(steps):
        step_index = step.get("step_index")
        if step_index != sequence_index:
            return {"status": "CONTINUATION_UNCERTAIN", "reason": "non-contiguous step indices"}
        if main_step is not None and step_index == main_step:
            break
        # A terminal/failing decision is the MAIN boundary, never a positive.
        if pointer == len(actions):
            break
        before = gold.get("initial_state") if pointer == 0 else states[pointer - 1]
        if not typed_equal(step.get("state_before_action"), before):
            return {"status": "CONTINUATION_UNCERTAIN", "reason": "off-gold state before candidate"}
        consumed = exact_parallel_progress(step, gold, pointer)
        prefix = _exact_prefix(student, step)
        model_message = step.get("model_message") or {}
        structured = step.get("structured_calls") or []
        message_calls = _message_calls(model_message)
        if (not consumed or prefix is None or message_calls is None
                or not typed_equal(message_calls, _normalized_calls(structured))):
            return {"status": "CONTINUATION_UNCERTAIN",
                    "reason": "prior decision not strict exact verified progress"}
        after_pointer = pointer + consumed
        candidate = {
            "conversation_prefix": prefix,
            "student_action": model_message,
            "student_action_calls": message_calls,
            "turn_index": step_index,
            "positive_source_turn": step_index,
            "progress_pointer_before": pointer,
            "progress_pointer_after": after_pointer,
            "distance_to_main_frontier": 1,
            "state_before": step["state_before_action"],
            "state_after": step["state_after_action"],
            "prefix_sha256": digest(prefix),
            "state_before_sha256": digest(step["state_before_action"]),
            "state_after_sha256": digest(step["state_after_action"]),
            "student_action_sha256": digest(model_message),
            "parallel_call_count": consumed,
            "verification_status": "CURRENT_EXACT_EXECUTION_VERIFIED_PROGRESS",
            "terminal_flag": False,
        }
        verified.append(candidate)
        pointer = after_pointer

    expected_pointer = record.get("frontier_depth")
    if pointer != expected_pointer:
        return {"status": "CONTINUATION_UNCERTAIN",
                "reason": "strict progress pointer differs from authoritative frontier",
                "strict_pointer": pointer, "authoritative_pointer": expected_pointer}
    if not verified:
        return {"status": "NO_VERIFIED_PRETERMINAL_CONTINUATION",
                "reason": "no verified tool decision before main boundary"}
    candidate = verified[-1]
    candidate["status"] = "CURRENT_CONTINUATION_ELIGIBLE"
    return candidate


def sample_count_distribution(main_tasks: set[str], continuation_tasks: set[str],
                              train_tasks: set[str]) -> dict[str, int]:
    if not main_tasks <= train_tasks or not continuation_tasks <= train_tasks:
        raise ValueError("heldout task entered Complete-C pool")
    counts = Counter(int(task in main_tasks) + int(task in continuation_tasks)
                     for task in train_tasks)
    if any(value not in (0, 1, 2) for value in counts):
        raise RuntimeError("more than two rows per task")
    return {str(key): counts.get(key, 0) for key in (0, 1, 2)}
