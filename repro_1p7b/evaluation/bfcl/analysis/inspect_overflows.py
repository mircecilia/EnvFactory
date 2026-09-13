#!/usr/bin/env python3
"""Audit context overflows without rerunning inference."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from paired_outcomes import EVAL_DIR, REPORT_DIR, load_jsonl, load_score_failures, result_files

OVERFLOW_RE = re.compile(
    r"The input \((\d+) tokens\) is longer than the model's context length \((\d+) tokens\)"
)
LOG_STEP_RE = re.compile(r"^ID: (base|long_context|miss_func|miss_param)_(\d+), Turn: (\d+), Step: (\d+)$", re.MULTILINE)


def load_rows(label: str) -> dict[str, dict]:
    rows = {}
    for path in result_files(label):
        for row in load_jsonl(path):
            rows[row["id"]] = row
    return rows


def overflow_info(row: dict) -> dict | None:
    text = f"{row.get('result', '')}\n{row.get('traceback', '')}"
    match = OVERFLOW_RE.search(text)
    if not match:
        return None
    return {
        "input_tokens": int(match.group(1)),
        "context_tokens": int(match.group(2)),
        "excess_tokens": int(match.group(1)) - int(match.group(2)),
    }


def log_summary(row: dict) -> dict:
    calls = 0
    max_input = 0
    observed_steps = 0
    for item in row.get("inference_log", []):
        if not isinstance(item, dict):
            continue
        for key, step in item.items():
            if not key.startswith("step_") or not isinstance(step, list):
                continue
            observed_steps += 1
            for entry in step:
                if not isinstance(entry, dict):
                    continue
                role = entry.get("role")
                content = entry.get("content")
                if role == "assistant" and isinstance(content, str):
                    calls += content.count("<tool_call>")
    stack = [row.get("input_token_count", [])]
    while stack:
        value = stack.pop()
        if isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, int):
            max_input = max(max_input, value)
    return {
        "tool_call_blocks": calls,
        "observed_requests": observed_steps,
        "max_logged_input_tokens": max_input,
    }


def logged_request_counts(label: str) -> Counter:
    text = (EVAL_DIR / "logs" / "full" / f"{label}_full.log").read_text(
        encoding="utf-8", errors="replace"
    )
    counts = Counter()
    for category, index, _turn, _step in LOG_STEP_RE.findall(text):
        counts[f"multi_turn_{category}_{index}"] += 1
    return counts


def success_table(rows: dict[str, dict], fail_ids: set[str], overflow_ids: set[str]) -> dict:
    all_ids = set(rows)
    correct_ids = all_ids - fail_ids
    return {
        "overflow_correct": len(overflow_ids & correct_ids),
        "overflow_wrong": len(overflow_ids & fail_ids),
        "non_overflow_correct": len((all_ids - overflow_ids) & correct_ids),
        "non_overflow_wrong": len((all_ids - overflow_ids) & fail_ids),
    }


def main() -> None:
    base_rows = load_rows("base")
    sft_rows = load_rows("sft")
    base_failure_rows, _ = load_score_failures("base")
    sft_failure_rows, _ = load_score_failures("sft")
    base_fail, sft_fail = set(base_failure_rows), set(sft_failure_rows)
    base_info = {task_id: info for task_id, row in base_rows.items() if (info := overflow_info(row))}
    sft_info = {task_id: info for task_id, row in sft_rows.items() if (info := overflow_info(row))}
    base_set, sft_set = set(base_info), set(sft_info)
    base_requests = logged_request_counts("base")
    sft_requests = logged_request_counts("sft")

    groups = {
        "both": sorted(base_set & sft_set),
        "base_only": sorted(base_set - sft_set),
        "sft_only": sorted(sft_set - base_set),
        "neither": sorted((set(base_rows) & set(sft_rows)) - base_set - sft_set),
    }
    sft_only_details = []
    for task_id in groups["sft_only"]:
        base_log = log_summary(base_rows[task_id])
        sft_only_details.append({
            "id": task_id,
            "base_completed_request_count": base_requests[task_id],
            "sft_requests_before_rejection": sft_requests[task_id],
            "request_count_delta": sft_requests[task_id] - base_requests[task_id],
            "base_normal_result": base_log,
            "sft_overflow": sft_info[task_id],
            "reasoning_or_tool_result_length_attribution": "unavailable",
        })

    more_requests = sum(
        row["sft_requests_before_rejection"] > row["base_completed_request_count"]
        for row in sft_only_details
    )
    output = {
        "definition": "Server rejected input longer than the model context window.",
        "counts": {key: len(value) for key, value in groups.items()},
        "groups": groups,
        "success_cross_tab": {
            "base": success_table(base_rows, base_fail, base_set),
            "sft": success_table(sft_rows, sft_fail, sft_set),
        },
        "base_overflows": base_info,
        "sft_overflows": sft_info,
        "sft_only_details": sft_only_details,
        "sft_only_summary": {
            "more_requests_than_base": more_requests,
            "not_more_requests_than_base": len(sft_only_details) - more_requests,
        },
        "limitations": [
            "Official overflow result rows replace the partial trajectory with an error and traceback.",
            "The full log preserves request counts and the rejected input length, but not partial response bodies.",
            "More requests is therefore testable; reasoning length, repeated calls, and tool-result length are not separable for rejected trajectories.",
            "Overflow membership is associated with failure here; it does not by itself isolate a causal SFT effect.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "overflows.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["counts"], sort_keys=True))
    print(json.dumps(output["success_cross_tab"], sort_keys=True))
    print(json.dumps(output["sft_only_summary"], sort_keys=True))


if __name__ == "__main__":
    main()
