"""V2-only, fail-closed repair for fixed-context rollout HTTP 400s.

The original failed episodes and integrity report remain immutable. A retry
uses the same model, task order, seed, sampling and 32K server context. If the
server rejects a turn for context length, the executed prefix is retained as a
typed, non-successful rollout; the miner must still verify any earlier error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import requests

from repro_1p7b.graph_frontier.collect_preference_rollouts import generate_action
from repro_1p7b.graph_frontier.cycle3_official_rl.audit import valid_gold
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import (
    IsolatedEpisode, atomic_json, typed_equal,
)
from repro_1p7b.graph_frontier.official_rl_audit_static import (
    OUT, SOURCE, decode, query_from_row,
)
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import CONFIG, check
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import parse_calls
from repro_1p7b.graph_frontier.recursive_opd_v2.future import checked_manifest, gold_for
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import RUN


def context_rejection(error: requests.HTTPError) -> dict | None:
    """Accept only SGLang's explicit context-length HTTP 400, not arbitrary 400s."""
    response = error.response
    if response is None or response.status_code != 400:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    message = body.get("message") if isinstance(body, dict) else None
    if not isinstance(message, str) or "context length" not in message.lower():
        return None
    limit = re.search(r"(?:maximum context length of\s*|context length \()(\d+)", message)
    if not limit or int(limit.group(1)) != 32768:
        return None
    input_match = (re.search(r"(\d+) tokens from the input messages", message)
                   or re.search(r"input \((\d+) tokens\)", message))
    total_match = re.search(r"total of (\d+) tokens", message)
    return {"http_status": 400, "server_message": message,
            "context_limit_tokens": 32768,
            "requested_input_tokens": int(input_match.group(1)) if input_match else None,
            "requested_total_tokens": int(total_match.group(1)) if total_match else None,
            "requested_max_completion_tokens": CONFIG["max_tokens_per_turn"]}


async def collect_with_context_boundary(entry: dict, row: dict, model_name: str,
                                        endpoint: str, schemas: list[dict]) -> dict:
    """Exact V1 collector loop except for a typed fixed-context boundary."""
    factory = row["extra_info"]["mcp_factory_kwargs"]
    env = IsolatedEpisode(entry["audit_id"], decode(factory["mcp_servers"]),
                          decode(factory["initial_config"]))
    await env.open()
    started = time.monotonic()
    try:
        if not typed_equal(env.schemas, schemas):
            raise RuntimeError("gold/student schema drift")
        messages = [{"role": "user", "content": query_from_row(row)}]
        steps, executed = [], 0
        termination = "MAX_TURNS"
        overflow = None
        for index in range(CONFIG["max_turns"]):
            before = await env.save()
            seed = CONFIG["seed"] + entry["source_row_index"] * 1009 + index
            try:
                message, parsed, usage, finish_reason = generate_action(
                    endpoint, model_name, messages, schemas, seed=seed,
                    temperature=CONFIG["temperature"],
                    max_tokens=CONFIG["max_tokens_per_turn"])
            except requests.HTTPError as error:
                overflow = context_rejection(error)
                if overflow is None:
                    raise
                overflow.update(step_index=index, prompt_message_count=len(messages),
                                state_before_action=before)
                termination = "CONTEXT_OVERFLOW"
                break
            prompt_count = len(messages)
            messages.append(message)
            calls, parse_errors = parse_calls(message)
            results = []
            if not (message.get("tool_calls") or []):
                termination = "MODEL_FINAL" if finish_reason != "length" else "GENERATION_LENGTH"
            else:
                async def execute(call: dict) -> dict:
                    if executed + call["position"] >= CONFIG["tool_call_limit"]:
                        return {"tool_name": call["name"], "tool_arguments": call["arguments"],
                                "tool_response": "unknown", "execution_success": False,
                                "exception": {"type": "ToolCallLimit", "message": "fixed call budget reached"}}
                    response, success, exception = await env.call(call["name"], call["arguments"])
                    return {"tool_name": call["name"], "tool_arguments": call["arguments"],
                            "tool_response": response, "execution_success": success,
                            "exception": exception}
                results = await asyncio.gather(*(execute(call) for call in calls))
                executed += len(calls)
                by_position = {call["position"]: result for call, result in zip(calls, results)}
                bad = {error["position"]: error for error in parse_errors}
                for position, raw in enumerate(message["tool_calls"]):
                    if position in by_position:
                        value = (by_position[position]["tool_response"]
                                 if by_position[position]["execution_success"]
                                 else {"error": by_position[position]["exception"]})
                        call_name = calls[next(i for i, call in enumerate(calls)
                                               if call["position"] == position)]["name"]
                    else:
                        value = {"error": bad[position]}
                        call_name = str((raw.get("function") or {}).get("name") or "unknown")
                    messages.append({"role": "tool", "tool_call_id": str(raw.get("id") or f"call_{position}"),
                                     "name": call_name, "content": stable(value)})
            after = await env.save()
            steps.append({"step_index": index, "prompt_message_count": prompt_count,
                          "state_before_action": before, "raw_assistant_output": message,
                          "model_message": message, "parsed_action": parsed,
                          "structured_calls": calls, "parse_errors": parse_errors,
                          "typed_events": results, "state_after_action": after,
                          "usage": usage, "finish_reason": finish_reason})
            if termination != "MAX_TURNS":
                break
        return {"status": "ROLLOUT_CONTEXT_OVERFLOW" if overflow else "ROLLOUT_COMPLETE",
                "model_source": model_name, "messages": messages, "steps": steps,
                "final_state": await env.save(), "termination_reason": termination,
                "context_overflow": overflow,
                "turn_budget_exhausted": termination == "MAX_TURNS",
                "runtime_seconds": time.monotonic() - started}
    finally:
        await env.close()


def original_errors(cycle: int) -> tuple[dict, Path]:
    path = RUN / f"cycle{cycle}/rollout_integrity.json"
    value = json.loads(path.read_text())
    if (value.get("status") != "ROLLOUT_COMPLETE" or value.get("cycle") != cycle
            or value.get("task_count") != 2158 or value.get("student_error_count") != 10
            or len(set(value.get("student_error_task_ids", []))) != 10):
        raise RuntimeError("unexpected immutable original integrity report")
    return value, path


def repair_one(cycle: int, task: str, endpoint: str) -> dict:
    if cycle != 1:
        raise ValueError("this repair applies only to the observed V2 pi1 batch")
    original, original_path = original_errors(cycle)
    if task not in original["student_error_task_ids"]:
        raise RuntimeError("task not listed among the ten immutable failures")
    manifest, manifest_sha = checked_manifest(cycle)
    base = RUN / f"cycle{cycle}"
    path = base / "rollouts" / f"{task}.json"
    archived = base / "failed_original" / f"{task}.json"
    old = json.loads(path.read_text())
    if old.get("repair"):
        if not archived.exists() or old["repair"]["original_episode_sha256"] != sha256(archived):
            raise RuntimeError("existing repair provenance mismatch")
        return {"task_id": task, "status": "ALREADY_REPAIRED"}
    if (old.get("student") is not None or (old.get("student_error") or {}).get("type") != "HTTPError"
            or old.get("task_id") != task or old.get("cycle") != cycle
            or old.get("model_sha256") != manifest["model_sha256"]
            or old.get("manifest_sha256") != manifest_sha
            or old.get("gpu_experiment_base_sha") != manifest["gpu_experiment_base_sha"]):
        raise RuntimeError("original failed episode identity drift")
    source = json.loads(SOURCE.read_text())
    entries = {item["audit_id"]: item for item in
               json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    entry = entries[task]
    row = source[entry["source_row_index"]]
    gold, schemas, gold_path, _ = gold_for(task, entry, row)
    if (old.get("gold_source_episode_sha256") != sha256(gold_path)
            or not valid_gold(entry, row, old)):
        raise RuntimeError("original gold/source identity drift")
    student = asyncio.run(collect_with_context_boundary(
        entry, row, manifest["served_model"], endpoint, schemas))
    if student["status"] not in {"ROLLOUT_COMPLETE", "ROLLOUT_CONTEXT_OVERFLOW"}:
        raise RuntimeError("retry did not terminate in an auditable state")
    if student["status"] == "ROLLOUT_COMPLETE" and student["context_overflow"] is not None:
        raise RuntimeError("contradictory retry status")
    archived.parent.mkdir(parents=True, exist_ok=True)
    if archived.exists():
        if sha256(archived) != sha256(path):
            raise RuntimeError("original archive differs from failed episode")
    else:
        os.link(path, archived)
    replacement = {**old, "student": student, "student_error": None,
                   "repair": {"kind": "FIXED_CONTEXT_REPLAY", "original_episode_sha256": sha256(archived),
                              "original_rollout_integrity_sha256": sha256(original_path),
                              "sampling_manifest_sha256": manifest_sha,
                              "server_context_limit_tokens": 32768}}
    atomic_json(path, replacement)
    return {"task_id": task, "status": student["status"],
            "steps": len(student["steps"]), "original_episode_sha256": sha256(archived)}


def repaired_integrity(cycle: int, *, write: bool) -> dict:
    original, original_path = original_errors(cycle)
    manifest, manifest_sha = checked_manifest(cycle)
    base = RUN / f"cycle{cycle}"
    task_ids = check()["train_task_ids"] + check()["heldout_task_ids"]
    original_ids = set(original["student_error_task_ids"])
    if len(task_ids) != 2158:
        raise RuntimeError("task plan drift")
    overflow_ids, archive_hashes = [], {}
    for task in task_ids:
        path = base / "rollouts" / f"{task}.json"
        episode = json.loads(path.read_text())
        student = episode.get("student") or {}
        if (episode.get("task_id") != task or episode.get("cycle") != cycle
                or episode.get("model_sha256") != manifest["model_sha256"]
                or episode.get("manifest_sha256") != manifest_sha
                or episode.get("student_error") is not None
                or student.get("status") not in {"ROLLOUT_COMPLETE", "ROLLOUT_CONTEXT_OVERFLOW"}):
            raise RuntimeError(f"unrepaired or drifted episode:{task}")
        if task in original_ids:
            archived = base / "failed_original" / f"{task}.json"
            repair = episode.get("repair") or {}
            if (not archived.exists() or repair.get("original_episode_sha256") != sha256(archived)
                    or repair.get("original_rollout_integrity_sha256") != sha256(original_path)
                    or repair.get("sampling_manifest_sha256") != manifest_sha
                    or (json.loads(archived.read_text()).get("student_error") or {}).get("type") != "HTTPError"):
                raise RuntimeError(f"failed original archival proof:{task}")
            archive_hashes[task] = sha256(archived)
        elif episode.get("repair"):
            raise RuntimeError(f"unlisted task unexpectedly repaired:{task}")
        if student["status"] == "ROLLOUT_CONTEXT_OVERFLOW":
            proof = student.get("context_overflow") or {}
            if (task not in original_ids or student.get("termination_reason") != "CONTEXT_OVERFLOW"
                    or proof.get("context_limit_tokens") != 32768
                    or proof.get("http_status") != 400
                    or "context length" not in proof.get("server_message", "").lower()):
                raise RuntimeError(f"invalid typed context boundary:{task}")
            overflow_ids.append(task)
    value = {"status": "ROLLOUT_COMPLETE", "schema_version": "v2_repaired_context_integrity_1",
             "cycle": cycle, "task_count": len(task_ids), "student_error_count": 0,
             "student_context_overflow_count": len(overflow_ids),
             "student_context_overflow_task_ids": overflow_ids,
             "repaired_task_ids": sorted(original_ids),
             "original_error_archive_sha256": archive_hashes,
             "original_rollout_integrity_sha256": sha256(original_path),
             "rollout_manifest_sha256": manifest_sha,
             "model_sha256": manifest["model_sha256"],
             "gpu_experiment_base_sha": manifest["gpu_experiment_base_sha"]}
    path = base / "rollout_integrity_repaired.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("repaired integrity report drift")
    elif write:
        atomic_json(path, value)
    else:
        raise RuntimeError("repaired integrity report not frozen")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("repair-one", "freeze", "verify"))
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--task-id")
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    if args.command == "repair-one":
        if not args.task_id or not args.endpoint:
            parser.error("repair-one needs --task-id and --endpoint")
        result = repair_one(args.cycle, args.task_id, args.endpoint)
    else:
        result = repaired_integrity(args.cycle, write=args.command == "freeze")
    print(stable(result), flush=True)


if __name__ == "__main__":
    main()
