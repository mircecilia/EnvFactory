"""Reproduce one failed V2 rollout without modifying its persisted artifact."""
from __future__ import annotations

import argparse
import asyncio
import json

import requests

from repro_1p7b.graph_frontier.official_rl_audit_static import OUT, SOURCE
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import collect_one
from repro_1p7b.graph_frontier.recursive_opd_v2.future import checked_manifest, gold_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()

    manifest, _ = checked_manifest(args.cycle)
    entries = {
        item["audit_id"]: item
        for item in json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]
    }
    entry = entries[args.task_id]
    row = json.loads(SOURCE.read_text())[entry["source_row_index"]]
    gold, schemas, _, _ = gold_for(args.task_id, entry, row)

    original_post = requests.post

    def inspect_post(*positional: object, **keyword: object) -> requests.Response:
        response = original_post(*positional, **keyword)
        if response.status_code >= 400:
            payload = keyword.get("json")
            messages = payload.get("messages", []) if isinstance(payload, dict) else []
            print(json.dumps({
                "http_status": response.status_code,
                "response_body": response.text[:3000],
                "message_count": len(messages),
                "message_chars": sum(len(json.dumps(message)) for message in messages),
                "tool_count": len(payload.get("tools", [])) if isinstance(payload, dict) else 0,
            }, ensure_ascii=False, sort_keys=True), flush=True)
        return response

    requests.post = inspect_post
    try:
        result = asyncio.run(collect_one(
            entry, row, manifest["served_model"], args.endpoint,
            {"gold": gold}, schemas,
        ))
        print(json.dumps({"status": result["status"], "steps": len(result["steps"]),
                          "termination_reason": result["termination_reason"]}), flush=True)
    except requests.HTTPError as exc:
        print(json.dumps({"error_type": type(exc).__name__, "task_id": args.task_id}), flush=True)
    finally:
        requests.post = original_post


if __name__ == "__main__":
    main()
