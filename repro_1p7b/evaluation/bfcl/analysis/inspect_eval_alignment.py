#!/usr/bin/env python3
"""Inspect BFCL result alignment, parser health, tool names, and token-limit symptoms."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from paired_outcomes import REPORT_DIR, load_jsonl, load_score_failures, result_files

CONTEXT_LENGTH = 40960
MAX_NEW_TOKENS = 4096
TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)


def flatten_ints(value):
    if isinstance(value, int):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from flatten_ints(item)


def iter_steps(row: dict):
    for turn in row.get("inference_log", []):
        if not isinstance(turn, dict):
            continue
        for key in sorted(turn):
            if key.startswith("step_") and isinstance(turn[key], list):
                yield turn[key]


def parse_allowed_tools(prompt: str) -> set[str]:
    names = set()
    for match in TOOLS_RE.finditer(prompt):
        for line in match.group(1).splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                spec = json.loads(line)
            except json.JSONDecodeError:
                continue
            function = spec.get("function", spec) if isinstance(spec, dict) else {}
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str):
                names.add(name)
    return names


def inspect_label(label: str) -> dict:
    counts = Counter()
    invalid_examples = []
    unknown_examples = []
    truncation_examples = []
    input_counts = []
    output_counts = []
    cap_hit_ids = set()

    for path in result_files(label):
        for row in load_jsonl(path):
            counts["tasks"] += 1
            task_id = row["id"]
            if "traceback" in row:
                counts["overflow_tasks"] += 1
                continue
            counts["normal_tasks"] += 1
            task_has_schema_call = False
            task_has_offered_call = False
            task_tool_blocks = 0
            for step in iter_steps(row):
                counts["steps"] += 1
                assistant_text = ""
                prompt = ""
                handler_entries = []
                for entry in step:
                    if not isinstance(entry, dict):
                        continue
                    role = entry.get("role")
                    content = entry.get("content")
                    if role == "assistant":
                        counts["assistant_entries"] += 1
                        if isinstance(content, str):
                            assistant_text = content
                        reasoning = entry.get("reasoning_content")
                        if isinstance(reasoning, str) and reasoning.strip():
                            counts["assistant_entries_with_reasoning"] += 1
                    elif role == "inference_input" and isinstance(content, dict):
                        formatted = content.get("formatted_prompt")
                        if isinstance(formatted, str):
                            prompt = formatted
                    elif role == "handler_log":
                        handler_entries.append(content)

                allowed = parse_allowed_tools(prompt)
                blocks = TOOL_BLOCK_RE.findall(assistant_text)
                task_tool_blocks += len(blocks)
                counts["raw_tool_blocks"] += len(blocks)
                if not assistant_text.strip():
                    counts["empty_assistant_steps"] += 1
                elif not blocks:
                    counts["natural_language_or_unclosed_steps"] += 1

                for raw in blocks:
                    if not raw.strip():
                        counts["empty_tool_blocks"] += 1
                        continue
                    try:
                        call = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        counts["invalid_json_tool_blocks"] += 1
                        if len(invalid_examples) < 10:
                            invalid_examples.append({"id": task_id, "raw": raw[:500]})
                        continue
                    counts["json_parseable_tool_blocks"] += 1
                    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                        counts["invalid_schema_tool_blocks"] += 1
                        if len(invalid_examples) < 10:
                            invalid_examples.append({"id": task_id, "raw": raw[:500]})
                        continue
                    if not isinstance(call.get("arguments"), dict):
                        counts["malformed_arguments_tool_blocks"] += 1
                        if len(invalid_examples) < 10:
                            invalid_examples.append({"id": task_id, "raw": raw[:500]})
                        continue
                    counts["schema_valid_tool_blocks"] += 1
                    task_has_schema_call = True
                    name = call["name"]
                    if name in allowed:
                        counts["offered_name_tool_blocks"] += 1
                        task_has_offered_call = True
                    else:
                        counts["unknown_name_tool_blocks"] += 1
                        if len(unknown_examples) < 10:
                            unknown_examples.append({
                                "id": task_id,
                                "name": name,
                                "allowed_sample": sorted(allowed)[:20],
                            })

                for content in handler_entries:
                    rendered = json.dumps(content, ensure_ascii=False) if not isinstance(content, str) else content
                    lowered = rendered.lower()
                    if "error" in lowered or "invalid" in lowered or "failed" in lowered:
                        counts["handler_error_entries"] += 1
                    if content in (None, "", [], {}):
                        counts["empty_handler_entries"] += 1

            if task_tool_blocks:
                counts["tasks_with_tool_block"] += 1
            if task_has_schema_call:
                counts["tasks_with_schema_valid_call"] += 1
            if task_has_offered_call:
                counts["tasks_with_offered_name_call"] += 1

            inputs = list(flatten_ints(row.get("input_token_count", [])))
            outputs = list(flatten_ints(row.get("output_token_count", [])))
            input_counts.extend(inputs)
            output_counts.extend(outputs)
            for index, output_tokens in enumerate(outputs):
                input_tokens = inputs[index] if index < len(inputs) else None
                if input_tokens is None:
                    continue
                requested = 1000 if input_tokens + 2 > CONTEXT_LENGTH else min(
                    MAX_NEW_TOKENS, CONTEXT_LENGTH - input_tokens - 2
                )
                if output_tokens >= requested:
                    counts["generation_cap_hits"] += 1
                    cap_hit_ids.add(task_id)
                    if len(truncation_examples) < 20:
                        truncation_examples.append({
                            "id": task_id,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "computed_request_cap": requested,
                        })
                elif output_tokens >= max(1, requested - 8):
                    counts["generation_near_cap"] += 1

    for key in (
        "empty_tool_blocks",
        "invalid_json_tool_blocks",
        "invalid_schema_tool_blocks",
        "malformed_arguments_tool_blocks",
        "handler_error_entries",
        "empty_handler_entries",
        "generation_cap_hits",
        "generation_near_cap",
        "empty_assistant_steps",
        "assistant_entries_with_reasoning",
    ):
        counts.setdefault(key, 0)
    c = dict(counts)
    normal = c.get("normal_tasks", 0)
    calls = c.get("raw_tool_blocks", 0)
    return {
        "counts": c,
        "rates": {
            "overflow_task_rate": c.get("overflow_tasks", 0) / c["tasks"],
            "tasks_with_schema_valid_call_rate_among_normal": c.get("tasks_with_schema_valid_call", 0) / normal,
            "tasks_with_offered_name_call_rate_among_normal": c.get("tasks_with_offered_name_call", 0) / normal,
            "json_parseable_rate_per_tool_block": c.get("json_parseable_tool_blocks", 0) / calls if calls else None,
            "offered_name_rate_per_schema_valid_block": (
                c.get("offered_name_tool_blocks", 0) / c.get("schema_valid_tool_blocks", 1)
            ),
            "mean_tool_blocks_per_normal_task": calls / normal,
            "mean_tool_blocks_per_all_tasks": calls / c["tasks"],
            "normal_tasks_without_tool_block_rate": (normal - c.get("tasks_with_tool_block", 0)) / normal,
        },
        "token_counts": {
            "input_observations": len(input_counts),
            "output_observations": len(output_counts),
            "max_input_tokens": max(input_counts, default=None),
            "max_output_tokens": max(output_counts, default=None),
            "finish_reason_available": False,
            "generation_cap_hit_task_count": len(cap_hit_ids),
            "generation_cap_hit_task_ids": sorted(cap_hit_ids),
        },
        "invalid_examples": invalid_examples,
        "unknown_name_examples": unknown_examples,
        "truncation_examples": truncation_examples,
        "limitations": [
            "BFCL result JSONL does not persist provider finish_reason.",
            "Cap-hit detection uses the handler's dynamic max-token rule and is a truncation symptom, not proof.",
            "Natural-language and unclosed-tool steps are combined because some malformed outputs omit a closing tag.",
        ],
    }


def parse_server_config(label: str) -> dict:
    log_path = Path(__file__).resolve().parent.parent / "logs" / "full" / f"{label}_server.log"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    line = next(
        (line for line in text.splitlines() if "server_args=ServerArgs(" in line),
        "",
    )
    patterns = {
        "model_path": r"model_path='([^']+)'",
        "dtype": r"dtype='([^']+)'",
        "context_length": r"context_length=(\d+)",
        "max_total_tokens": r"max_total_tokens=(\d+)",
        "tp_size": r"tp_size=(\d+)",
        "random_seed": r"random_seed=(\d+)",
        "chat_template": r"chat_template=([^,]+)",
        "reasoning_parser": r"reasoning_parser=([^,]+)",
        "tool_call_parser": r"tool_call_parser=([^,]+)",
        "sampling_defaults": r"sampling_defaults=([^,]+)",
        "attention_backend": r"attention_backend=([^,]+)",
        "sampling_backend": r"sampling_backend=([^,]+)",
        "enable_deterministic_inference": r"enable_deterministic_inference=([^,]+)",
    }
    config = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        config[key] = match.group(1) if match else None
    config.update({
        "bfcl_model_id": "Qwen/Qwen3-1.7B-FC",
        "bfcl_backend": "sglang",
        "temperature": "0.7",
        "top_p_explicit": False,
        "request_seed_explicit": False,
        "test_category": "multi_turn",
        "include_input_log": True,
    })
    return config


def compare_run_configs() -> dict:
    configs = {label: parse_server_config(label) for label in ("base", "sft")}
    fields = sorted(set(configs["base"]) | set(configs["sft"]))
    differences = {
        field: {"base": configs["base"].get(field), "sft": configs["sft"].get(field)}
        for field in fields
        if configs["base"].get(field) != configs["sft"].get(field)
    }
    return {
        "base": configs["base"],
        "sft": configs["sft"],
        "differences": differences,
        "expected_model_difference_only": set(differences) == {"model_path"},
        "evidence": "Resolved SGLang ServerArgs logs plus the shared BFCL driver.",
        "limitation": "The BFCL request did not pass a seed; the two resolved server random seeds differ.",
    }

def main() -> None:
    output = {
        "base": inspect_label("base"),
        "sft": inspect_label("sft"),
        "run_config_comparison": compare_run_configs(),
        "constants": {
            "context_length": CONTEXT_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    }
    for label in ("base", "sft"):
        failure_ids = set(load_score_failures(label)[0])
        cap_ids = set(output[label]["token_counts"]["generation_cap_hit_task_ids"])
        output[label]["token_counts"]["generation_cap_hit_correct"] = len(cap_ids - failure_ids)
        output[label]["token_counts"]["generation_cap_hit_wrong"] = len(cap_ids & failure_ids)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "eval_health.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    for label in ("base", "sft"):
        print(label, json.dumps(output[label]["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
