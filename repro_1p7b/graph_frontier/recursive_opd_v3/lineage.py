"""Pure lineage and crossed-frontier verification for Recursive OPD V3.

This module deliberately does not execute environments or models.  It consumes
the typed, execution-verified artifacts produced by Recursive OPD V2 and fails
closed whenever a current student action cannot be tied to one unambiguous
historical pre-terminal frontier.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import typed_equal
from repro_1p7b.graph_frontier.recursive_opd_v1.mine import exact_parallel_progress
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import digest

PRETERMINAL_REGIMES = frozenset({"START", "ADVANCE"})


def depth_bucket(depth: int) -> str:
    if depth < 0:
        raise ValueError("negative frontier depth")
    return str(depth) if depth < 3 else "3+"


def normalize_action(call: dict) -> dict:
    return {"name": call.get("name"), "arguments": call.get("arguments")}


def historical_identity(record: dict, episode: dict, *, metrics_sha256: str,
                        rollout_sha256: str, rollout_manifest_sha256: str,
                        source_policy_sha256: str) -> dict:
    """Create a stable serialized identity for one correction frontier."""
    if (record.get("status") != "FIRST_ERROR"
            or record.get("error_regime") not in PRETERMINAL_REGIMES):
        raise ValueError("only START/ADVANCE FIRST_ERROR can be continuation lineage")
    pointer = record.get("frontier_depth")
    gold = episode.get("gold") or {}
    actions, states = gold.get("actions") or [], gold.get("states_after") or []
    if (not isinstance(pointer, int) or pointer < 0 or pointer >= len(actions)
            or len(states) != len(actions)
            or (pointer == 0) != (record["error_regime"] == "START")
            or not typed_equal(record.get("gold_next_action"), actions[pointer])):
        raise ValueError("historical gold pointer/regime/action is invalid")
    before = gold.get("initial_state") if pointer == 0 else states[pointer - 1]
    return {
        "task_id": record["task_id"],
        "query_hash": record["query_hash"],
        "source_cycle": record["cycle"],
        "frontier_regime": record["error_regime"],
        "gold_pointer": pointer,
        "frontier_depth": pointer,
        "gold_tool_count": len(actions),
        "state_sha256": digest(before),
        "target_action_sha256": digest(actions[pointer]),
        "target_tool_name": actions[pointer]["name"],
        "target_arguments": actions[pointer]["arguments"],
        "source_metrics_sha256": metrics_sha256,
        "source_rollout_sha256": rollout_sha256,
        "source_rollout_manifest_sha256": rollout_manifest_sha256,
        "source_policy_sha256": source_policy_sha256,
        "gold_source_episode_sha256": episode["gold_source_episode_sha256"],
    }


def latest_relevant_frontier(records: Iterable[dict], *, task_id: str,
                             current_cycle: int) -> dict | None:
    """Select at most one latest historical START/ADVANCE correction.

    Duplicate records for the same task/cycle are an ambiguous lineage and are
    rejected rather than broken by list order.
    """
    candidates = [row for row in records
                  if row.get("task_id") == task_id
                  and isinstance(row.get("cycle"), int)
                  and row["cycle"] < current_cycle
                  and row.get("status") == "FIRST_ERROR"
                  and row.get("error_regime") in PRETERMINAL_REGIMES]
    if not candidates:
        return None
    latest_cycle = max(row["cycle"] for row in candidates)
    latest = [row for row in candidates if row["cycle"] == latest_cycle]
    if len(latest) != 1:
        raise ValueError("AMBIGUOUS_LINEAGE: duplicate latest frontier")
    return latest[0]


def lineage_matches(identity: dict, source_episode: dict, current_record: dict,
                    current_episode: dict) -> bool:
    """Require the current trajectory to use the same immutable task/gold."""
    source_gold, current_gold = source_episode.get("gold"), current_episode.get("gold")
    return bool(
        current_record.get("task_id") == identity.get("task_id")
        and current_record.get("query_hash") == identity.get("query_hash")
        and current_record.get("cycle", -1) > identity.get("source_cycle", 10**9)
        and current_episode.get("task_id") == identity.get("task_id")
        and current_episode.get("gold_source_episode_sha256")
            == identity.get("gold_source_episode_sha256")
        and source_gold is not None and current_gold is not None
        and typed_equal(source_gold.get("initial_state"), current_gold.get("initial_state"))
        and typed_equal(source_gold.get("actions"), current_gold.get("actions"))
        and typed_equal(source_gold.get("observations"), current_gold.get("observations"))
        and typed_equal(source_gold.get("states_after"), current_gold.get("states_after"))
        and typed_equal(source_gold.get("final_state"), current_gold.get("final_state"))
    )


def movement(identity: dict, current_record: dict) -> str:
    """Classify verifier movement relative to one historical frontier."""
    if current_record.get("status") == "UNCERTAIN":
        return "UNCERTAIN"
    depth = current_record.get("frontier_depth")
    if not isinstance(depth, int):
        return "UNCERTAIN"
    return "CROSSED" if depth > identity["gold_pointer"] else "REGRESSED"


def _prompt_before(step: dict, student: dict) -> list[dict] | None:
    count, messages = step.get("prompt_message_count"), student.get("messages")
    if (not isinstance(count, int) or not isinstance(messages, list)
            or count < 1 or count >= len(messages)
            or not typed_equal(messages[count], step.get("model_message"))):
        return None
    return messages[:count]


def verified_student_crossing(identity: dict, current_record: dict,
                              current_episode: dict) -> dict:
    """Extract one exact, current-policy, execution-verified continuation.

    Movement may be verifier-confirmed even when the crossing occurred in a
    parallel/alternative assistant message.  V3 v1 only materializes a target
    when the crossing itself is one exact single call, because extracting one
    call from a multi-call assistant message would fabricate a behavior the
    student did not actually emit.
    """
    if movement(identity, current_record) != "CROSSED":
        return {"status": "NOT_CROSSED"}
    if current_record.get("status") == "UNCERTAIN":
        return {"status": "UNCERTAIN_RETENTION", "reason": "current verdict uncertain"}
    gold, student = current_episode.get("gold") or {}, current_episode.get("student") or {}
    if (student.get("status") not in {"ROLLOUT_COMPLETE", "ROLLOUT_CONTEXT_OVERFLOW"}
            or not isinstance(student.get("steps"), list)):
        return {"status": "UNCERTAIN_RETENTION", "reason": "current trajectory absent"}
    pointer, wanted = 0, identity["gold_pointer"]
    actions = gold.get("actions") or []
    states = gold.get("states_after") or []
    observations = gold.get("observations") or []
    if wanted >= len(actions) or len(states) != len(actions) or len(observations) != len(actions):
        return {"status": "AMBIGUOUS_LINEAGE", "reason": "current gold shape invalid"}
    for step in student["steps"]:
        before = gold.get("initial_state") if pointer == 0 else states[pointer - 1]
        if not typed_equal(step.get("state_before_action"), before):
            return {"status": "UNCERTAIN_RETENTION", "reason": "off-gold current state"}
        consumed = exact_parallel_progress(step, gold, pointer)
        if not consumed:
            return {"status": "UNCERTAIN_RETENTION",
                    "reason": "crossing depends on alternative or nonexact action"}
        if pointer <= wanted < pointer + consumed:
            calls, events = step.get("structured_calls") or [], step.get("typed_events") or []
            if consumed != 1 or len(calls) != 1 or len(events) != 1:
                return {"status": "AMBIGUOUS_LINEAGE",
                        "reason": "crossing emitted in multi-call assistant message"}
            prefix = _prompt_before(step, student)
            action = normalize_action(calls[0])
            event = events[0]
            if (prefix is None or step.get("parse_errors")
                    or event.get("execution_success") is not True
                    or not typed_equal(action, actions[wanted])
                    or digest(step.get("state_before_action")) != identity["state_sha256"]
                    or digest(action) != identity["target_action_sha256"]
                    or not typed_equal(event.get("tool_response"), observations[wanted])
                    or not typed_equal(step.get("state_after_action"), states[wanted])):
                return {"status": "UNCERTAIN_RETENTION",
                        "reason": "single-call execution/progress evidence mismatch"}
            return {
                "status": "RETENTION_ELIGIBLE",
                "conversation_prefix": prefix,
                "student_action": action,
                "student_action_sha256": digest(action),
                "student_prefix_sha256": digest(prefix),
                "current_student_step": step["step_index"],
                "current_prompt_message_count": len(prefix),
                "execution_success": True,
                "progress_verified": True,
            }
        pointer += consumed
    return {"status": "UNCERTAIN_RETENTION", "reason": "crossing step not found"}


def sample_count_distribution(main_tasks: set[str], retention_tasks: set[str],
                              all_train_tasks: set[str]) -> dict[str, int]:
    if not main_tasks <= all_train_tasks or not retention_tasks <= all_train_tasks:
        raise ValueError("heldout task entered natural V3 pool")
    counts = Counter(int(task in main_tasks) + int(task in retention_tasks)
                     for task in all_train_tasks)
    if any(value not in (0, 1, 2) for value in counts):
        raise RuntimeError("more than two samples per task")
    return {str(key): counts.get(key, 0) for key in (0, 1, 2)}
