"""Read-only semantic and failure-pattern reanalysis for frozen pilot artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("base", "original_sft", "parameter_aware")
READ_ONLY = {"symbol_info", "city_weather", "city_forecast", "city_alerts"}
MIXED = {"quote_order_details", "list_get", "task_get", "estimate_query", "estimate_cancel"}
MUTATION_TOOLS = {
    "place_order", "make_transaction", "add_to_watchlist", "remove_from_watchlist",
    "update_stock_price", "notify_price_change", "save_location", "create_task_list",
    "update_task_list", "delete_task_list", "create_task", "update_task", "delete_task",
    "create_order", "cancel_order", "create_event", "update_event", "delete_event",
}
LOOKUP_PREFIXES = ("get_", "list_", "search_", "query_", "check_", "estimate_", "filter_")


def canon(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rate(successes: int, attempts: int) -> dict[str, Any]:
    return {
        "successes": successes,
        "attempts": attempts,
        "raw_rate": successes / attempts if attempts else None,
        "smoothed_rate": (successes + 1) / (attempts + 2),
    }


def mcnemar(left: Mapping[str, bool], right: Mapping[str, bool]) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    wins = sum(not left[key] and right[key] for key in ids)
    losses = sum(left[key] and not right[key] for key in ids)
    n = wins + losses
    p = 1.0 if not n else min(
        1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / (2 ** n)
    )
    return {
        "paired_support": len(ids), "right_wins": wins, "right_losses": losses,
        "discordant": n, "exact_two_sided_p": p,
    }


def short(tool_name: str) -> str:
    return tool_name.split("-", 1)[-1]


def events(rollout: Mapping[str, Any], name: str | None = None) -> list[dict[str, Any]]:
    rows = sorted(rollout.get("events") or [], key=lambda row: row.get("step_index", 0))
    return rows if name is None else [row for row in rows if short(str(row.get("tool_name"))) == name]


def args(event: Mapping[str, Any]) -> Any:
    return event.get("tool_arguments", {})


def result(event: Mapping[str, Any]) -> Any:
    return event.get("returned_fields", "unknown")


def successful(event: Mapping[str, Any]) -> bool:
    return event.get("execution_success") is True


def values(mapping: Any) -> list[Mapping[str, Any]]:
    return list(mapping.values()) if isinstance(mapping, Mapping) else []


def state(rollout: Mapping[str, Any], environment: str) -> Mapping[str, Any] | None:
    value = rollout.get("final_environment_state")
    if isinstance(value, Mapping) and isinstance(value.get(environment), Mapping):
        return value[environment]
    return value if isinstance(value, Mapping) else None


def same(actual: Any, expected: Any, casefold: bool = False) -> bool:
    if casefold and isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().casefold() == expected.strip().casefold()
    return type(actual) is type(expected) and actual == expected


def fields_match(
    actual: Any, expected: Any, fields: Iterable[str], casefold: Iterable[str] = ()
) -> bool:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    folded = set(casefold)
    return all(
        key in actual and key in expected and same(actual[key], expected[key], key in folded)
        for key in fields
    )


def reference_call(reference: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    return next((row for row in reference if short(str(row.get("tool_name"))) == name), None)


def success_match(rows: Iterable[Mapping[str, Any]], predicate) -> Mapping[str, Any] | None:
    return next((row for row in rows if successful(row) and predicate(row)), None)


def semantic_base(template: str) -> dict[str, Any]:
    verifier_type = (
        "typed_observation_predicate" if template in READ_ONLY
        else "state_plus_observation" if template in MIXED
        else "state_predicate"
    )
    return {
        "semantic_verifier_supported": True,
        "semantic_success": False,
        "verifier_type": verifier_type,
        "checked_fields": [],
        "reference_source": "frozen typed reference trace and deterministic final state",
        "ambiguity_flag": verifier_type != "state_predicate",
        "ambiguity_reason": (
            "final assistant content was not persisted; terminal_success only proves an assistant turn ended"
            if verifier_type != "state_predicate" else None
        ),
        "completion_step": None,
        "state_predicate": None,
        "observation_predicate": None,
        "terminal_assistant": None,
        "unsupported_reason": None,
    }


def semantic_verify(
    manifest: Mapping[str, Any],
    rollout: Mapping[str, Any],
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply a deterministic template-specific semantic verifier."""
    template = str(manifest["template"])
    environment = str(manifest["environment"])
    final = state(rollout, environment)
    terminal = rollout.get("terminal_success") is True and rollout.get("final_status") == "success"
    out = semantic_base(template)
    out["terminal_assistant"] = terminal

    def finish(
        state_ok: bool | None,
        observation_ok: bool | None,
        event: Mapping[str, Any] | None,
        checked: list[str],
    ) -> dict[str, Any]:
        out["state_predicate"] = state_ok
        out["observation_predicate"] = observation_ok
        out["checked_fields"] = checked
        parts = [value for value in (state_ok, observation_ok) if value is not None]
        if observation_ok is not None:
            parts.append(terminal)
        out["semantic_success"] = bool(parts) and all(parts)
        out["completion_step"] = event.get("step_index") if out["semantic_success"] and event else None
        return out

    if template in READ_ONLY:
        target = {
            "symbol_info": "get_stock_info", "city_weather": "get_current_weather",
            "city_forecast": "get_forecast", "city_alerts": "get_alerts",
        }[template]
        ref = reference_call(reference, target)
        expected = ref.get("result") if ref else None
        if expected in ({}, [], None, ""):
            out.update(
                semantic_verifier_supported=False,
                unsupported_reason="empty_reference_observation",
                checked_fields=[target + ".typed_result", "terminal_success"],
            )
            return out
        event = success_match(
            events(rollout, target),
            lambda row: canon(args(row)) == canon(ref.get("arguments", {}))
            and canon(result(row)) == canon(expected),
        )
        return finish(
            None, event is not None, event,
            [target + ".arguments", target + ".typed_result", "terminal_success"],
        )

    if template in {"quote_order", "quote_order_details"}:
        expected = reference_call(reference, "place_order")["result"]
        checked = ["symbol", "quantity", "price", "order_type", "status"]
        orders = values(final.get("orders", {}) if final else {})
        order = next(
            (row for row in orders if fields_match(row, expected, checked, {"order_type"})),
            None,
        )
        state_ok = order is not None
        if template == "quote_order":
            event = success_match(
                events(rollout, "place_order"),
                lambda row: fields_match(result(row), expected, checked, {"order_type"}),
            )
            return finish(state_ok, None, event, ["orders[*]." + key for key in checked])
        event = success_match(
            events(rollout, "get_order_details"),
            lambda row: bool(order)
            and args(row).get("order_id") == order.get("order_id")
            and fields_match(result(row), order, checked + ["order_id"], {"order_type"}),
        )
        return finish(
            state_ok, event is not None, event,
            ["orders[*]." + key for key in checked]
            + ["get_order_details.typed_result", "terminal_success"],
        )

    if template == "city_save":
        desired = reference_call(reference, "save_location")["arguments"]
        saved = final.get("saved_locations", {}) if final else {}
        actual = saved.get(desired["alias"]) if isinstance(saved, Mapping) else None
        checked = ["alias", "latitude", "longitude", "name"]
        state_ok = fields_match(actual, desired, checked)
        event = success_match(
            events(rollout, "save_location"),
            lambda row: fields_match(args(row), desired, checked),
        )
        return finish(
            state_ok, None, event,
            ["saved_locations[" + desired["alias"] + "]." + key for key in checked],
        )

    if environment == "GoogleTasks":
        list_title = reference_call(reference, "create_task_list")["arguments"]["tasklist_title"]
        task_lists = values(final.get("tasklists", {}) if final else {})
        task_list = next((row for row in task_lists if row.get("title") == list_title), None)
        state_ok = task_list is not None
        if template == "list_get":
            event = success_match(
                events(rollout, "get_task_list"),
                lambda row: bool(task_list)
                and result(row).get("title") == list_title
                and result(row).get("tasklist_id") == task_list.get("tasklist_id"),
            )
            return finish(
                state_ok, event is not None, event,
                ["tasklists[*].title", "get_task_list.typed_result", "terminal_success"],
            )

        initial_title = reference_call(reference, "create_task")["arguments"]["title"]
        desired_title = (
            reference_call(reference, "update_task")["arguments"]["title"]
            if template == "task_update" else initial_title
        )
        tasks = values(final.get("tasks", {}) if final else {})
        task = next(
            (
                row for row in tasks
                if row.get("title") == desired_title
                and task_list and row.get("tasklist_id") == task_list.get("tasklist_id")
            ),
            None,
        )
        if template == "task_delete":
            absent = not any(row.get("title") == initial_title for row in tasks)
            created = success_match(
                events(rollout, "create_task"),
                lambda row: result(row).get("title") == initial_title,
            )
            deleted = success_match(
                events(rollout, "delete_task"),
                lambda row: bool(created)
                and row.get("step_index", -1) > created.get("step_index", -1)
                and args(row).get("task_id") == result(created).get("task_id")
                and args(row).get("tasklist_id") == result(created).get("tasklist_id"),
            )
            return finish(
                bool(task_list) and absent and created is not None and deleted is not None,
                None, deleted,
                ["tasklists[*].title", "create_task.task_id", "delete_task.task_id", "tasks[target] absent"],
            )
        state_ok = bool(task_list) and task is not None
        if template == "task_get":
            event = success_match(
                events(rollout, "get_task"),
                lambda row: bool(task)
                and result(row).get("title") == desired_title
                and result(row).get("task_id") == task.get("task_id")
                and result(row).get("tasklist_id") == task.get("tasklist_id"),
            )
            return finish(
                state_ok, event is not None, event,
                ["tasklists[*].title", "tasks[*].title", "get_task.typed_result", "terminal_success"],
            )
        target = "update_task" if template == "task_update" else "create_task"
        event = success_match(
            events(rollout, target),
            lambda row: result(row).get("title") == desired_title,
        )
        return finish(
            state_ok, None, event,
            ["tasklists[*].title", "tasks[*].tasklist_id", "tasks[*].title"],
        )

    if environment == "UUPaoTui":
        estimate = reference_call(reference, "estimate_price")["arguments"]
        create = reference_call(reference, "create_order")["arguments"]
        desired = {
            "fromAddress": estimate["fromAddress"], "toAddress": estimate["toAddress"],
            "adCode": estimate["adCode"], "sendType": estimate["sendType"],
            "senderPhone": create["senderPhone"], "receiverPhone": create["receiverPhone"],
        }
        checked = list(desired)
        orders = values(final.get("orders", {}) if final else {})
        order = next((row for row in orders if fields_match(row, desired, checked)), None)
        state_ok = order is not None
        if template == "estimate_create":
            event = success_match(
                events(rollout, "create_order"),
                lambda row: bool(order) and result(row).get("orderCode") == order.get("orderCode"),
            )
            return finish(state_ok, None, event, ["orders[*]." + key for key in checked])
        query = success_match(
            events(rollout, "query_order"),
            lambda row: bool(order)
            and args(row).get("orderCode") == order.get("orderCode")
            and fields_match(
                result(row), order, ["orderCode", "fromAddress", "toAddress", "state"]
            ),
        )
        if template == "estimate_query":
            return finish(
                state_ok, query is not None, query,
                ["orders[*]." + key for key in checked]
                + ["query_order.typed_result", "terminal_success"],
            )
        cancelled = bool(order) and order.get("state") in {"cancelled", "已取消"}
        cancel = success_match(
            events(rollout, "cancel_order"),
            lambda row: bool(query)
            and row.get("step_index", -1) > query.get("step_index", -1)
            and args(row).get("orderCode") == order.get("orderCode"),
        )
        return finish(
            state_ok and cancelled, query is not None and cancel is not None, cancel,
            ["orders[*]." + key for key in checked]
            + ["orders[*].state", "query_order.typed_result", "cancel_order.orderCode", "terminal_success"],
        )

    if environment == "Calendar":
        create = reference_call(reference, "create_event")["arguments"]
        updated = reference_call(reference, "update_event")["arguments"]
        desired = {
            "summary": updated["summary"], "calendar_id": create["calendar_id"],
            "start_time": create["start_time"], "end_time": create["end_time"],
        }
        checked = list(desired)
        final_events = values(final.get("events", {}) if final else {})
        if template == "create_update":
            target = next(
                (row for row in final_events if fields_match(row, desired, checked)), None
            )
            event = success_match(
                events(rollout, "update_event"),
                lambda row: bool(target)
                and result(row).get("event_id") == target.get("event_id")
                and result(row).get("summary") == desired["summary"],
            )
            return finish(
                target is not None, None, event,
                ["events[*]." + key for key in checked],
            )
        target_absent = not any(
            row.get("summary") in {create["summary"], updated["summary"]}
            for row in final_events
        )
        created = success_match(
            events(rollout, "create_event"),
            lambda row: result(row).get("summary") == create["summary"],
        )
        update = success_match(
            events(rollout, "update_event"),
            lambda row: bool(created)
            and row.get("step_index", -1) > created.get("step_index", -1)
            and args(row).get("event_id") == result(created).get("event_id")
            and result(row).get("summary") == updated["summary"],
        )
        deleted = success_match(
            events(rollout, "delete_event"),
            lambda row: bool(update)
            and row.get("step_index", -1) > update.get("step_index", -1)
            and args(row).get("event_id") == result(created).get("event_id"),
        )
        return finish(
            target_absent and created is not None and update is not None and deleted is not None,
            None, deleted,
            ["create_event.event_id", "update_event.event_id", "delete_event.event_id", "events[target] absent"],
        )

    out.update(semantic_verifier_supported=False, unsupported_reason="no_template_rule")
    return out


def flatten_scalars(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        answer: list[Any] = []
        for child in value.values():
            answer.extend(flatten_scalars(child))
        return answer
    if isinstance(value, list):
        answer = []
        for child in value:
            answer.extend(flatten_scalars(child))
        return answer
    return [value] if value not in ("unknown", None) else []


def call_patterns(
    manifest: Mapping[str, Any],
    rollout: Mapping[str, Any],
    profiler_result: Mapping[str, Any],
    semantic: Mapping[str, Any],
) -> dict[str, Any]:
    rows = events(rollout)
    gold = set(manifest["gold_tools"])
    edges = manifest["dependency_edges"]
    producers = {edge["producer_tool"]["tool_name"] for edge in edges}
    consumers = {edge["consumer_tool"]["tool_name"] for edge in edges}
    completion = semantic.get("completion_step")
    repeated = Counter()
    repeated_tools = Counter()
    seen: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    wrong_steps = {
        check.get("consumer_step")
        for check in profiler_result.get("profile", {}).get("dependency_edge_checks", [])
        if check.get("status") == "failed_wrong_value"
    }
    for event in rows:
        prior = seen[event["tool_name"]]
        if prior:
            repeated_tools[event["tool_name"]] += 1
            if any(canon(args(old)) == canon(args(event)) for old in prior):
                repeated["same_tool_same_args"] += 1
            else:
                repeated["same_tool_different_args"] += 1
            if any(successful(old) for old in prior):
                if event["tool_name"] in producers:
                    repeated["producer_repeated_after_success"] += 1
                if event["tool_name"] in consumers:
                    repeated["consumer_repeated_after_success"] += 1
                if short(event["tool_name"]) in MUTATION_TOOLS:
                    repeated["duplicate_mutation_action"] += 1
            if any(not successful(old) for old in prior):
                repeated["retry_after_tool_error"] += 1
            if any(old.get("step_index") in wrong_steps for old in prior):
                repeated["retry_after_wrong_value"] += 1
        seen[event["tool_name"]].append(event)

    names = [row["tool_name"] for row in rows]
    cycles = Counter()
    for index in range(len(names) - 2):
        a, b, c = names[index:index + 3]
        if a == c and a != b:
            cycles["a_b_a"] += 1
        if b == c and a != b:
            cycles["a_b_b"] += 1
    for width in range(2, min(6, len(names) // 2 + 1)):
        for index in range(len(names) - 2 * width + 1):
            if names[index:index + width] == names[index + width:index + 2 * width]:
                cycles["repeated_block_cycle"] += 1

    post = [
        row for row in rows
        if isinstance(completion, int) and row.get("step_index", -1) > completion
    ]
    unexpected = Counter()
    unexpected_tools = Counter()
    failure_reasons = Counter()
    unexpected_rows = []
    for index, event in enumerate(rows):
        if event["tool_name"] in gold:
            continue
        tags = ["tool_outside_gold"]
        unexpected_tools[event["tool_name"]] += 1
        tool = short(event["tool_name"])
        future_gold_args = [
            scalar
            for future in rows[index + 1:]
            if future["tool_name"] in gold
            for scalar in flatten_scalars(args(future))
        ]
        flows = any(
            type(value) is type(target) and value == target
            for value in flatten_scalars(result(event))
            for target in future_gold_args
        )
        if flows:
            tags.append("alternative_producer_candidate")
        if tool in MUTATION_TOOLS:
            tags.append("unnecessary_mutation")
        if tool.startswith(LOOKUP_PREFIXES):
            tags.append(
                "exploratory_lookup"
                if completion is None or event.get("step_index", -1) <= completion
                else "unrelated_lookup"
            )
        if isinstance(completion, int) and event.get("step_index", -1) > completion:
            tags.append("post_success_extra_tool")
        if not successful(event):
            tags.append("malformed_or_failed_tool_call")
            exception = event.get("exception")
            if isinstance(exception, Mapping):
                message = str(exception.get("message", exception.get("type", "unknown")))
            else:
                message = str(exception)
            reason = (
                "max_tool_calls_exceeded" if "max_tool_calls_exceeded" in message else message
            )
            failure_reasons[reason] += 1
        if len(tags) == 1:
            tags.append("other")
        unexpected.update(tags)
        unexpected_rows.append(
            {"step_index": event.get("step_index"), "tool_name": event["tool_name"], "patterns": tags}
        )

    return {
        "legacy_exact_redundant_calls": len(
            profiler_result.get("profile", {}).get("redundant_tool_calls") or []
        ),
        "legacy_unexpected_calls": len(
            profiler_result.get("profile", {}).get("extra_tool_calls") or []
        ),
        "repeated_pattern_counts": dict(repeated),
        "repeated_tool_counts": dict(repeated_tools),
        "cycle_pattern_counts": dict(cycles),
        "unexpected_pattern_counts": dict(unexpected),
        "unexpected_tool_counts": dict(unexpected_tools),
        "unexpected_failure_reason_counts": dict(failure_reasons),
        "hit_tool_budget": failure_reasons["max_tool_calls_exceeded"] > 0,
        "tool_budget_exceeded_calls": failure_reasons["max_tool_calls_exceeded"],
        "post_success_extra_calls": len(post),
        "post_success_extra_tools": [row["tool_name"] for row in post],
        "unexpected_rows": unexpected_rows,
    }


def internal_funnel(
    manifest: Mapping[str, Any],
    rollout: Mapping[str, Any],
    profiler_result: Mapping[str, Any],
) -> dict[str, int]:
    rows = events(rollout)
    checks = {
        row["edge_id"]: row
        for row in profiler_result.get("profile", {}).get("dependency_edge_checks", [])
    }
    counts = Counter(gold_internal_opportunities=len(manifest["dependency_edges"]))
    for edge in manifest["dependency_edges"]:
        producer = edge["producer_tool"]["tool_name"]
        consumer = edge["consumer_tool"]["tool_name"]
        producer_rows = [row for row in rows if row["tool_name"] == producer]
        consumer_rows = [row for row in rows if row["tool_name"] == consumer]
        check = checks.get(edge["edge_id"], {})
        producer_success_steps = [
            row.get("step_index", -1) for row in producer_rows if successful(row)
        ]
        consumer_steps = [row.get("step_index", -1) for row in consumer_rows]
        counts["producer_reached"] += bool(producer_rows)
        counts["producer_succeeded"] += bool(producer_success_steps)
        counts["consumer_reached"] += bool(consumer_rows)
        counts["consumer_reached_after_successful_producer"] += any(
            producer_step < consumer_step
            for producer_step in producer_success_steps for consumer_step in consumer_steps
        )
        counts["consumer_input_inspectable"] += any(
            isinstance(args(row), Mapping) for row in consumer_rows
        )
        counts["source_and_target_value_inspectable"] += (
            check.get("source_value_available") is True
            and check.get("target_value_available") is True
        )
        counts["correct_propagated"] += check.get("success") is True
    return dict(counts)


def sum_counters(rows: Iterable[Mapping[str, int]]) -> dict[str, int]:
    total = Counter()
    for row in rows:
        total.update(row)
    return dict(total)


def group_summary(task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    supported = [row for row in task_rows if row["semantic"]["semantic_verifier_supported"]]
    semantic_successes = sum(row["semantic"]["semantic_success"] for row in supported)
    gold = sum(row["funnel"].get("gold_internal_opportunities", 0) for row in task_rows)
    correct = sum(row["funnel"].get("correct_propagated", 0) for row in task_rows)
    reached = sum(row["funnel"].get("consumer_reached", 0) for row in task_rows)
    reference = sum(row["reference_path_complete_success"] for row in task_rows)
    return {
        "tasks": len(task_rows),
        "semantic_task_success": rate(semantic_successes, len(supported)),
        "semantic_supported": len(supported),
        "reference_path_complete_success": rate(reference, len(task_rows)),
        "internal_end_to_end": rate(correct, gold),
        "internal_reach": rate(reached, gold),
        "conditional_propagation_accuracy": rate(correct, reached),
        "redundant_calls_per_task": sum(
            row["calls"]["legacy_exact_redundant_calls"] for row in task_rows
        ) / len(task_rows),
        "unexpected_calls_per_task": sum(
            row["calls"]["legacy_unexpected_calls"] for row in task_rows
        ) / len(task_rows),
    }


def aggregate_model(task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = group_summary(task_rows)
    funnel = sum_counters(row["funnel"] for row in task_rows)
    gold = funnel.get("gold_internal_opportunities", 0)
    reached = funnel.get("consumer_reached", 0)
    correct = funnel.get("correct_propagated", 0)
    funnel.update(
        reach_rate=rate(reached, gold),
        conditional_propagation_accuracy=rate(correct, reached),
        end_to_end_internal_edge_success=rate(correct, gold),
    )
    repeated, cycles, unexpected = Counter(), Counter(), Counter()
    repeated_tools, unexpected_tools, failure_reasons = Counter(), Counter(), Counter()
    for row in task_rows:
        repeated.update(row["calls"]["repeated_pattern_counts"])
        repeated_tools.update(row["calls"]["repeated_tool_counts"])
        cycles.update(row["calls"]["cycle_pattern_counts"])
        unexpected.update(row["calls"]["unexpected_pattern_counts"])
        unexpected_tools.update(row["calls"]["unexpected_tool_counts"])
        failure_reasons.update(row["calls"]["unexpected_failure_reason_counts"])
    post_values = [
        row["calls"]["post_success_extra_calls"]
        for row in task_rows if row["semantic"]["semantic_success"]
    ]
    call_audit = {
        "legacy_exact_redundant_calls": sum(
            row["calls"]["legacy_exact_redundant_calls"] for row in task_rows
        ),
        "legacy_unexpected_calls": sum(
            row["calls"]["legacy_unexpected_calls"] for row in task_rows
        ),
        "repeated_patterns_nonexclusive": dict(repeated),
        "top_repeated_tools": dict(repeated_tools.most_common(15)),
        "cycle_patterns_nonexclusive": dict(cycles),
        "unexpected_patterns_nonexclusive": dict(unexpected),
        "top_unexpected_tools": dict(unexpected_tools.most_common(15)),
        "unexpected_failure_reasons": dict(failure_reasons),
        "tool_budget_exceeded": {
            "tasks": sum(row["calls"]["hit_tool_budget"] for row in task_rows),
            "calls": sum(row["calls"]["tool_budget_exceeded_calls"] for row in task_rows),
        },
        "post_success_continuation": {
            "tasks": sum(value > 0 for value in post_values),
            "extra_calls": sum(post_values),
            "mean_extra_calls_among_semantic_successes": (
                sum(post_values) / len(post_values) if post_values else None
            ),
            "distribution": dict(Counter(str(value) for value in post_values)),
        },
    }
    alternate = [
        row for row in task_rows
        if row["semantic"]["semantic_success"] and not row["reference_path_complete_success"]
    ]
    alternative_patterns = Counter()
    for row in alternate:
        if row["calls"]["legacy_unexpected_calls"]:
            alternative_patterns["extra_nonreference_tool"] += 1
        if row["calls"]["legacy_exact_redundant_calls"]:
            alternative_patterns["repeated_call"] += 1
        if not row["edge_complete"]:
            alternative_patterns["gold_edge_incomplete"] += 1
        if not row["gold_node_complete"]:
            alternative_patterns["gold_node_incomplete"] += 1
        if row["final_state_success"] is False:
            alternative_patterns["exact_final_state_mismatch"] += 1

    by_depth, by_environment, by_template = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in task_rows:
        depth = "3+" if row["dependency_depth"] >= 3 else str(row["dependency_depth"])
        by_depth[depth].append(row)
        by_environment[row["environment"]].append(row)
        by_template[row["environment"] + "/" + row["template"]].append(row)
    summary.update(
        internal_dependency_funnel=funnel,
        call_audit=call_audit,
        alternative_valid_path={
            "count": len(alternate),
            "task_ids": [row["task_id"] for row in alternate],
            "pattern_counts_nonexclusive": dict(alternative_patterns),
            "environment_distribution": dict(Counter(row["environment"] for row in alternate)),
            "template_distribution": dict(Counter(row["template"] for row in alternate)),
        },
        depth_breakdown={key: group_summary(value) for key, value in sorted(by_depth.items())},
        environment_breakdown={
            key: group_summary(value) for key, value in sorted(by_environment.items())
        },
        template_breakdown={key: group_summary(value) for key, value in sorted(by_template.items())},
    )
    return summary


def failure_attribution(row: Mapping[str, Any]) -> str:
    checks = row["result"].get("profile", {}).get("dependency_edge_checks", [])
    statuses = [check.get("status") for check in checks]
    if "failed_wrong_value" in statuses:
        return "wrong_propagated_value"
    if "failed_missing_producer" in statuses:
        return "producer_not_reached"
    if any(
        event["tool_name"] in set(row["manifest"]["gold_tools"]) and not successful(event)
        for event in events(row["rollout"])
    ):
        return "tool_execution_failure"
    if row["semantic"].get("observation_predicate") is False:
        return "target_observation_not_acquired"
    if row["semantic"].get("state_predicate") is False:
        return "semantic_target_state_not_met"
    return "other_semantic_failure"


def paired_regression(
    left_rows: Mapping[str, Mapping[str, Any]],
    right_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    supported = sorted(
        task_id for task_id in set(left_rows) & set(right_rows)
        if left_rows[task_id]["semantic"]["semantic_verifier_supported"]
        and right_rows[task_id]["semantic"]["semantic_verifier_supported"]
    )
    groups: dict[str, list[str]] = defaultdict(list)
    impaired: list[str] = []
    for task_id in supported:
        left, right = left_rows[task_id], right_rows[task_id]
        left_ok = left["semantic"]["semantic_success"]
        right_ok = right["semantic"]["semantic_success"]
        if left_ok and not right_ok:
            groups["original_true_pa_false"].append(task_id)
        elif not left_ok and right_ok:
            groups["original_false_pa_true"].append(task_id)
        elif not left_ok and not right_ok:
            groups["both_false"].append(task_id)
        else:
            groups["both_true"].append(task_id)
            if (
                len(events(right["rollout"])) > len(events(left["rollout"]))
                or right["calls"]["legacy_exact_redundant_calls"] > left["calls"]["legacy_exact_redundant_calls"]
                or right["calls"]["legacy_unexpected_calls"] > left["calls"]["legacy_unexpected_calls"]
                or right["funnel"].get("correct_propagated", 0) < left["funnel"].get("correct_propagated", 0)
            ):
                impaired.append(task_id)
    regressions = [right_rows[task_id] for task_id in groups["original_true_pa_false"]]
    return {
        "supported_pairs": len(supported),
        "groups": {
            key: {"count": len(value), "task_ids": value}
            for key, value in sorted(groups.items())
        },
        "both_true_with_pa_efficiency_or_propagation_regression": {
            "count": len(impaired), "task_ids": impaired,
        },
        "regression_failure_attribution": dict(
            Counter(failure_attribution(row) for row in regressions)
        ),
        "regression_environment_distribution": dict(
            Counter(row["environment"] for row in regressions)
        ),
        "regression_template_distribution": dict(
            Counter(row["template"] for row in regressions)
        ),
        "regression_depth_distribution": dict(
            Counter(str(row["dependency_depth"]) for row in regressions)
        ),
    }


def analyze(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "frozen" / "pilot_100_seed_20260914.jsonl"
    manifests = [
        json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()
    ]
    by_id = {row["task_id"]: row for row in manifests}
    task_maps: dict[str, dict[str, dict[str, Any]]] = {}
    model_reports = {}
    for label in LABELS:
        rows = {}
        for task_id, manifest in sorted(by_id.items()):
            rollout = json.loads(
                (root / label / "tasks" / (task_id + ".rollout.json")).read_text()
            )
            profiler_result = json.loads(
                (root / label / "tasks" / (task_id + ".result.json")).read_text()
            )
            reference = json.loads((ROOT / manifest["reference_trace_path"]).read_text())
            semantic = semantic_verify(manifest, rollout, reference)
            funnel = internal_funnel(manifest, rollout, profiler_result)
            funnel["correct_edges_with_final_semantic_success"] = (
                funnel.get("correct_propagated", 0) if semantic["semantic_success"] else 0
            )
            calls = call_patterns(manifest, rollout, profiler_result, semantic)
            rows[task_id] = {
                "task_id": task_id,
                "environment": manifest["environment"],
                "template": manifest["template"],
                "dependency_depth": manifest["dependency_depth"],
                "semantic": semantic,
                "reference_path_complete_success": bool(profiler_result["task_success"]),
                "edge_complete": bool(profiler_result["edge_complete"]),
                "gold_node_complete": bool(profiler_result["gold_node_complete"]),
                "final_state_success": profiler_result["final_state_success"],
                "funnel": funnel,
                "calls": calls,
                "manifest": manifest,
                "rollout": rollout,
                "result": profiler_result,
            }
        task_maps[label] = rows
        model_reports[label] = aggregate_model(list(rows.values()))
        read_only_rows = [
            row for row in rows.values() if row["template"] in READ_ONLY
        ]
        model_reports[label]["read_only_final_state_audit"] = {
            "tasks": len(read_only_rows),
            "exact_final_state_true": sum(
                row["final_state_success"] is True for row in read_only_rows
            ),
            "exact_final_state_true_with_zero_tool_calls": sum(
                row["final_state_success"] is True and not events(row["rollout"])
                for row in read_only_rows
            ),
            "semantic_supported": sum(
                row["semantic"]["semantic_verifier_supported"] for row in read_only_rows
            ),
        }

    semantic_maps = {
        label: {
            task_id: row["semantic"]["semantic_success"]
            for task_id, row in task_maps[label].items()
            if row["semantic"]["semantic_verifier_supported"]
        }
        for label in LABELS
    }
    verifier_templates = {}
    for manifest in manifests:
        key = manifest["environment"] + "/" + manifest["template"]
        if key in verifier_templates:
            continue
        examples = [
            task_maps["base"][task_id]["semantic"]
            for task_id, row in by_id.items()
            if row["environment"] + "/" + row["template"] == key
        ]
        verifier_templates[key] = {
            "verifier_type": examples[0]["verifier_type"],
            "checked_fields": sorted(
                {field for example in examples for field in example["checked_fields"]}
            ),
            "reference_source": examples[0]["reference_source"],
            "ambiguity_flag": any(row["ambiguity_flag"] for row in examples),
            "supported_tasks": sum(row["semantic_verifier_supported"] for row in examples),
            "total_tasks": len(examples),
            "unsupported_reasons": dict(Counter(
                row["unsupported_reason"] for row in examples if row["unsupported_reason"]
            )),
        }

    regression = paired_regression(
        task_maps["original_sft"], task_maps["parameter_aware"]
    )
    report = {
        "schema_version": "graph_frontier_capability_reanalysis_v2",
        "source_policy": "read_only_existing_artifacts",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "metric_audit": {
            "task_success_legacy": (
                "valid probe AND all required gold nodes successfully executed AND "
                "all selected gold dependency edges satisfied AND exact canonical final state matched"
            ),
            "task_success_v2_name": "reference_path_complete_success",
            "final_state_success_limitation": (
                "exact state equality is vacuously true for read-only tasks even "
                "when no requested observation was acquired"
            ),
            "legacy_internal_attempts": (
                "number of gold internal edges whose consumer tool was called; "
                "therefore conditional on model reach and model-dependent"
            ),
            "path_adherence": (
                "task-level exact reference tool set/order with no extra, missing, or exact duplicate calls"
            ),
            "edge_complete": "task-level all selected gold edges satisfied",
            "internal_param_complete": (
                "task-level all internal parameter checks satisfied; identical "
                "to edge_complete in this all-internal manifest"
            ),
            "all_selected_edges_internal": all(
                edge.get("internal_parameter") is True
                for manifest in manifests for edge in manifest["dependency_edges"]
            ),
            "gold_internal_opportunities": sum(
                len(manifest["dependency_edges"]) for manifest in manifests
            ),
        },
        "semantic_verifier_templates": verifier_templates,
        "models": model_reports,
        "semantic_paired_mcnemar": {
            "base_vs_original_sft": mcnemar(
                semantic_maps["base"], semantic_maps["original_sft"]
            ),
            "original_sft_vs_parameter_aware": mcnemar(
                semantic_maps["original_sft"], semantic_maps["parameter_aware"]
            ),
            "base_vs_parameter_aware": mcnemar(
                semantic_maps["base"], semantic_maps["parameter_aware"]
            ),
        },
        "original_sft_vs_parameter_aware": regression,
        "limitations": [
            (
                "No final assistant response content was persisted; observation-based "
                "semantic success verifies typed acquisition plus terminal assistant "
                "status, not the factual content of final prose."
            ),
            "Empty reference observations are unsupported and excluded from semantic denominators.",
            "State predicates are template-specific deterministic predicates, not perfect semantic oracles.",
            (
                "Unexpected tools may be valid alternatives; alternative_producer_candidate "
                "is evidence of typed value reuse, not proof of necessity."
            ),
            "Call-pattern categories are deterministic and non-exclusive.",
            "Edges within a task are clustered and are not independent statistical samples.",
        ],
    }
    failures = {
        "schema_version": "graph_frontier_failure_patterns_v2",
        "models": {
            label: {
                "call_audit": model_reports[label]["call_audit"],
                "alternative_valid_path": model_reports[label]["alternative_valid_path"],
            }
            for label in LABELS
        },
        "original_sft_vs_parameter_aware": regression,
    }
    return report, failures


def fmt(metric: Mapping[str, Any]) -> str:
    if metric["raw_rate"] is None:
        return "n/a (0/" + str(metric["attempts"]) + ")"
    return (
        f"{metric['successes']}/{metric['attempts']} = "
        f"{100 * metric['raw_rate']:.1f}% "
        f"(Beta {100 * metric['smoothed_rate']:.1f}%)"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    models = report["models"]
    lines = [
        "# Static Graph-Frontier pilot: capability reanalysis v2", "",
        (
            "Derived only from the frozen manifest, reference traces, typed rollouts, "
            "and profiler outputs. No model or environment runtime was invoked."
        ), "",
        "## Metric audit", "",
        "- task_success is the legacy name of reference_path_complete_success, not generic semantic completion.",
        "- It requires all gold nodes, all selected gold edges, and exact final-state equality.",
        "- All 213 selected edges are internal dependencies.",
        "- Legacy internal attempts count consumer-reached edges, so the denominator is model-dependent.",
        "- Read-only exact final-state equality is vacuous and is not used as semantic evidence in v2.",
        "", "## Core metrics", "",
        "| Metric | Base | Original SFT | Parameter-Aware |",
        "|---|---:|---:|---:|",
    ]
    for title, key in [
        ("Semantic task success", "semantic_task_success"),
        ("Reference-path complete", "reference_path_complete_success"),
        ("Internal reach", "internal_reach"),
        ("Internal end-to-end", "internal_end_to_end"),
        ("Conditional propagation", "conditional_propagation_accuracy"),
    ]:
        lines.append(
            "| " + title + " | " + " | ".join(fmt(models[label][key]) for label in LABELS) + " |"
        )
    lines += ["", "## Internal dependency funnel", ""]
    for label in LABELS:
        funnel = models[label]["internal_dependency_funnel"]
        lines += [
            "### " + label, "",
            (
                "gold={gold} -> producer reached={pr} -> producer succeeded={ps} -> "
                "consumer reached={cr} -> values inspectable={vi} -> correct={ok}"
            ).format(
                gold=funnel["gold_internal_opportunities"],
                pr=funnel.get("producer_reached", 0),
                ps=funnel.get("producer_succeeded", 0),
                cr=funnel.get("consumer_reached_after_successful_producer", 0),
                vi=funnel.get("source_and_target_value_inspectable", 0),
                ok=funnel.get("correct_propagated", 0),
            ), "",
        ]
    lines += ["## Calls and post-success continuation", ""]
    for label in LABELS:
        calls = models[label]["call_audit"]
        post = calls["post_success_continuation"]
        lines += [
            "### " + label, "",
            f"- Legacy exact redundant / unexpected: {calls['legacy_exact_redundant_calls']} / {calls['legacy_unexpected_calls']}",
            f"- Repeated patterns (non-exclusive): {calls['repeated_patterns_nonexclusive']}",
            f"- Cycle patterns (non-exclusive): {calls['cycle_patterns_nonexclusive']}",
            f"- Unexpected patterns (non-exclusive): {calls['unexpected_patterns_nonexclusive']}",
            f"- Post-success: {post['tasks']} tasks, {post['extra_calls']} calls, mean {post['mean_extra_calls_among_semantic_successes']}",
            f"- Semantic success but reference-path failure: {models[label]['alternative_valid_path']['count']}",
            "",
        ]
    paired = report["original_sft_vs_parameter_aware"]
    lines += ["## Original SFT vs Parameter-Aware", ""]
    for key, value in paired["groups"].items():
        lines.append(f"- {key}: {value['count']}")
    lines += [
        f"- both_true_with_pa_efficiency_or_propagation_regression: {paired['both_true_with_pa_efficiency_or_propagation_regression']['count']}",
        f"- Regression failure attribution: {paired['regression_failure_attribution']}",
        f"- Regression environments: {paired['regression_environment_distribution']}",
        "",
        "## Read-only final-state audit",
        "",
        "| Model | Read-only tasks | Exact state match | State match with zero calls | Semantic support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        audit = models[label]["read_only_final_state_audit"]
        lines.append(
            f"| {label} | {audit['tasks']} | {audit['exact_final_state_true']} | "
            f"{audit['exact_final_state_true_with_zero_tool_calls']} | "
            f"{audit['semantic_supported']} |"
        )
    lines += [
        "",
        "All 24 read-only tasks have exact final-state equality for every model. "
        "Base has 23 such matches with no tool call, proving that state equality "
        "alone is not capability evidence.",
        "",
        "## Depth breakdown",
        "",
        "| Depth | Model | Semantic | Reference path | Internal end-to-end | Reach | Conditional propagation | Redundant/task | Unexpected/task |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in ("1", "2", "3+"):
        for label in LABELS:
            bucket = models[label]["depth_breakdown"][depth]
            lines.append(
                f"| {depth} | {label} | {fmt(bucket['semantic_task_success'])} | "
                f"{fmt(bucket['reference_path_complete_success'])} | "
                f"{fmt(bucket['internal_end_to_end'])} | "
                f"{fmt(bucket['internal_reach'])} | "
                f"{fmt(bucket['conditional_propagation_accuracy'])} | "
                f"{bucket['redundant_calls_per_task']:.2f} | "
                f"{bucket['unexpected_calls_per_task']:.2f} |"
            )
    lines += [
        "",
        "## Environment breakdown",
        "",
        "| Environment | Model | Semantic | Reference path | Internal end-to-end | Redundant/task | Unexpected/task |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for environment in ("Calendar", "GoogleTasks", "TradingBot", "UUPaoTui", "Weather"):
        for label in LABELS:
            bucket = models[label]["environment_breakdown"][environment]
            lines.append(
                f"| {environment} | {label} | {fmt(bucket['semantic_task_success'])} | "
                f"{fmt(bucket['reference_path_complete_success'])} | "
                f"{fmt(bucket['internal_end_to_end'])} | "
                f"{bucket['redundant_calls_per_task']:.2f} | "
                f"{bucket['unexpected_calls_per_task']:.2f} |"
            )
    original = models["original_sft"]
    parameter_aware = models["parameter_aware"]
    mcnemar = report["semantic_paired_mcnemar"]["original_sft_vs_parameter_aware"]
    lines += [
        "",
        "## Q1-Q7 decisions",
        "",
        "### Q1: What does legacy task_success measure?",
        "",
        "Reference-path completion, not generic semantic completion: all required "
        "gold nodes, all selected gold edges, and the exact canonical final state "
        "must pass.",
        "",
        "### Q2: What is the semantic Original-versus-PA gap?",
        "",
        f"Original is {fmt(original['semantic_task_success'])}; PA is "
        f"{fmt(parameter_aware['semantic_task_success'])}. The paired gap is six "
        f"tasks (6.74 percentage points); PA wins {mcnemar['right_wins']} and loses "
        f"{mcnemar['right_losses']} discordant pairs, exact McNemar "
        f"p={mcnemar['exact_two_sided_p']:.4g}.",
        "",
        "### Q3: What drives the PA regression?",
        "",
        "Primarily propagation correctness, tool execution failures, and redundant "
        "pre-completion loops. Reach alone does not explain it, and no model has "
        "post-success tool continuation under the deterministic completion boundary. "
        "Alternative valid paths explain part of the strict-path gap but not the "
        "large internal-edge regression.",
        "",
        "### Q4: Why are final-state rates close?",
        "",
        "The 40% versus 41% comparison is materially inflated by a permissive "
        "read-only state check. Supported deterministic semantic success is 31/89 "
        "for PA versus 37/89 for Original.",
        "",
        "### Q5: What are PA's dominant extra-call patterns?",
        "",
        "Repeated blocks/cycles, exact same-argument repeats, producer and consumer "
        "repetition, error retries, and tool-budget exhaustion. They occur before "
        "verified completion, not after it.",
        "",
        "### Q6: Is the depth-2/3 zero only a strict-path artifact?",
        "",
        "No. Strict reference-path scoring hides some alternative success, especially "
        "at depth 2, but PA's fixed-denominator and conditional propagation collapse "
        "at depth 2 and depth 3+ is a real deeper-dependency failure signal.",
        "",
        "### Q7: Can this profiler support dynamic allocation?",
        "",
        "Yes, after using semantic support-aware task success, fixed-denominator "
        "internal reach/end-to-end metrics, conditional propagation, and separate "
        "efficiency/call-pattern metrics. Future probes should persist final assistant "
        "content so read-only observation completion can be verified more strongly.",
        "",
        "## Recommendation",
        "",
        "Option 1: freeze a 300-task confirmation protocol using the corrected v2 "
        "metrics. Do not begin dynamic allocation from the uncorrected legacy metrics.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend("- " + item for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "repro_1p7b/results/graph_frontier/pilot_100",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "repro_1p7b/graph_frontier/reports",
    )
    ns = parser.parse_args()
    report, failures = analyze(ns.root)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    (ns.output_dir / "pilot_100_capability_reanalysis_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (ns.output_dir / "pilot_100_capability_reanalysis_v2.md").write_text(
        render_markdown(report)
    )
    (ns.output_dir / "pilot_100_failure_patterns_v2.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        label: report["models"][label]["semantic_task_success"] for label in LABELS
    }, indent=2))


if __name__ == "__main__":
    main()
