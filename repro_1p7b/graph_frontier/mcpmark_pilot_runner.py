"""Run one arm of the frozen MCPMark Filesystem paired pilot.

The official MCPMark pipeline owns task execution, environment setup and
verification. This wrapper only fixes task order, isolation, per-task
persistence, and conservative resume behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_SHA256 = "fc4f2061622451d12ece56ad80299f526a85d3a715d8b67686e26f942e5359ba"
INFRA_MARKERS = (
    "Manual MCP loop failed",
    "LLM call failed",
    "Proceeding with verification despite agent failure",
    "MCP server failed",
    "Connection error",
)


def verified_meta(task_dir: Path, task_name: str, model: str) -> dict | None:
    matches = list(task_dir.rglob("meta.json"))
    if len(matches) != 1:
        return None
    meta = json.loads(matches[0].read_text(encoding="utf-8"))
    if meta.get("task_name") != task_name or meta.get("model_name") != model:
        return None
    result = meta.get("execution_result")
    if not isinstance(result, dict) or "success" not in result:
        return None
    return meta


def write_status(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(args: argparse.Namespace) -> None:
    manifest_bytes = args.manifest.read_bytes()
    actual_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_sha != MANIFEST_SHA256:
        raise SystemExit(f"frozen manifest SHA mismatch: {actual_sha}")
    manifest = json.loads(manifest_bytes)
    if manifest.get("suite") != "standard" or manifest.get("mcp") != "filesystem":
        raise SystemExit("unexpected benchmark suite")
    if len(manifest["tasks"]) != 10:
        raise SystemExit("expected exactly 10 frozen tasks")
    if not args.benchmark.joinpath("pipeline.py").is_file():
        raise SystemExit("official MCPMark pipeline.py missing")
    if not args.root.is_dir():
        raise SystemExit("isolated FILESYSTEM_TEST_ROOT missing")
    args.output.mkdir(parents=True, exist_ok=True)
    args.tmp.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "pilot10_arm_status.json"
    model = "openai/mcpmark-" + args.arm
    rows = []
    for task in manifest["tasks"]:
        index = task["index"]
        category, task_id = task["category"], task["task_id"]
        task_name = f"{category}__{task_id}"
        task_dir = args.output / f"task_{index:02d}_{category}_{task_id}"
        log = task_dir / "pipeline.log"
        task_dir.mkdir(parents=True, exist_ok=True)
        meta = verified_meta(task_dir, task_name, model)
        if meta is None:
            if log.exists():
                raise SystemExit(f"unverified prior attempt; inspect before retry: {log}")
            env = os.environ.copy()
            env.update(
                OPENAI_BASE_URL=f"http://127.0.0.1:{args.port}/v1",
                OPENAI_API_KEY="local",
                FILESYSTEM_TEST_ROOT=str(args.root),
                TMPDIR=str(args.tmp),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )
            command = [
                sys.executable,
                "pipeline.py",
                "--mcp", "filesystem",
                "--models", model,
                "--agent", "mcpmark",
                "--tasks", f"{category}/{task_id}",
                "--task-suite", "standard",
                "--k", "1",
                "--timeout", str(args.timeout),
                "--exp-name", "pilot",
                "--output-dir", str(task_dir),
            ]
            print(f"[{args.arm}] task {index + 1}/10 {category}/{task_id}", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                code = subprocess.call(command, cwd=args.benchmark, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if code:
                raise SystemExit(f"official pipeline exit {code}: {log}")
            meta = verified_meta(task_dir, task_name, model)
            if meta is None:
                raise SystemExit(f"official verifier result missing: {log}")
        log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        infra_failure = any(marker in log_text for marker in INFRA_MARKERS)
        verification_error = meta["execution_result"].get("verification_error") or ""
        infra_failure |= "Traceback (most recent call last)" in verification_error
        row = {
            "index": index,
            "category": category,
            "task_id": task_id,
            "model": model,
            "status": "INFRA_FAILURE" if infra_failure else "VERIFIED",
            "pass": None if infra_failure else bool(meta["execution_result"]["success"]),
            "turns": meta.get("turn_count"),
            "runtime_seconds": meta.get("task_execution_time"),
            "meta": str(next(task_dir.rglob("meta.json"))),
            "log": str(log),
        }
        rows.append(row)
        write_status(status_path, {
            "arm": args.arm,
            "manifest_sha256": MANIFEST_SHA256,
            "completed": len(rows),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "tasks": rows,
        })
        print(f"[{args.arm}] {row['status']} {index + 1}/10 pass={row['pass']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("grpo", "cpi3"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tmp", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
