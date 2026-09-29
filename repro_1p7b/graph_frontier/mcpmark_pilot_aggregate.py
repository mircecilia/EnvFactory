"""Aggregate only official, paired MCPMark verifier outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from repro_1p7b.graph_frontier.mcpmark_pilot_runner import MANIFEST_SHA256, verified_meta


MODEL_HASHES = {
    "grpo": "d871a96525ae69a40b57484022b8cde9ec7dd4576e6d7f7ac29cddab42727a40",
    "cpi3": "1e1022b12a21a730ce6038f1c257c7d75e3ec7e119314ce80e663088c78e3790",
}
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


def read_arm(task: dict, arm: str, runtime: Path) -> dict | None:
    index, category, task_id = task["index"], task["category"], task["task_id"]
    task_dir = runtime / f"{arm}_results" / "pilot10" / f"task_{index:02d}_{category}_{task_id}"
    meta = verified_meta(task_dir, f"{category}__{task_id}", f"openai/mcpmark-{arm}")
    if meta is None:
        return None
    meta_path = next(task_dir.rglob("meta.json"))
    messages_path = meta_path.with_name("messages.json")
    messages = json.loads(messages_path.read_text(encoding="utf-8"))
    log = (task_dir / "pipeline.log").read_text(encoding="utf-8", errors="replace")
    if "Manual MCP loop failed" in log or "LLM call failed" in log:
        return {"status": "INFRA_FAILURE", "log": str(task_dir / "pipeline.log")}
    verification_error = meta["execution_result"].get("verification_error") or ""
    if "Traceback (most recent call last)" in verification_error:
        return {
            "status": "INFRA_FAILURE",
            "reason": "official verifier raised an exception",
            "verification_error": verification_error,
            "log": str(task_dir / "pipeline.log"),
        }
    reason_match = re.search(r"Task ended with the finish reason from messages being '([^']+)'", log)
    verifier = meta["execution_result"].get("verification_output") or ""
    failures = [line.strip() for line in verifier.splitlines() if line.lstrip().startswith(chr(0x274C))]
    return {
        "status": "MODEL_PASS" if meta["execution_result"]["success"] else "MODEL_FAIL",
        "pass": bool(meta["execution_result"]["success"]),
        "turns": meta.get("turn_count"),
        "tool_calls": sum(msg.get("type") == "function_call" for msg in messages),
        "runtime_seconds": meta.get("task_execution_time"),
        "termination_reason": reason_match.group(1) if reason_match else "unknown",
        "verifier_error": meta["execution_result"].get("verification_error"),
        "verifier_failures": failures[:8],
        "raw_meta": str(meta_path),
        "raw_messages": str(messages_path),
        "raw_log": str(task_dir / "pipeline.log"),
    }


def mean_metric(rows: list[dict], key: str) -> float | None:
    vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return round(statistics.mean(vals), 3) if vals else None


def aggregate(manifest: Path, runtime: Path) -> dict:
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen manifest hash mismatch")
    frozen = json.loads(raw)
    rows = []
    paired = []
    for task in frozen["tasks"]:
        row = {
            "index": task["index"],
            "category": task["category"],
            "task_id": task["task_id"],
            "grpo": read_arm(task, "grpo", runtime),
            "cpi3": read_arm(task, "cpi3", runtime),
        }
        rows.append(row)
        if all(row[arm] and row[arm]["status"] in ("MODEL_PASS", "MODEL_FAIL") for arm in ("grpo", "cpi3")):
            paired.append(row)
    grpo_pass = sum(row["grpo"]["pass"] for row in paired)
    cpi3_pass = sum(row["cpi3"]["pass"] for row in paired)
    counts = {
        "grpo_wins": sum(row["grpo"]["pass"] and not row["cpi3"]["pass"] for row in paired),
        "cpi3_wins": sum(row["cpi3"]["pass"] and not row["grpo"]["pass"] for row in paired),
        "tie_pass": sum(row["grpo"]["pass"] and row["cpi3"]["pass"] for row in paired),
        "tie_fail": sum(not row["grpo"]["pass"] and not row["cpi3"]["pass"] for row in paired),
    }
    return {
        "label": "PRELIMINARY PAIRED PILOT — NOT FULL MCPMARK SCORE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {"commit": frozen["mcpmark_commit"], "suite": "Verified standard", "mcp": "filesystem", "k": 1, "agent": "MCPMarkAgent", "manifest_sha256": MANIFEST_SHA256},
        "protocol": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_output_tokens_transport_cap": 8192, "context_length": 40960, "task_timeout_seconds": 3600, "tool_parser": "qwen25"},
        "models": {
            "grpo": {"checkpoint": "repro_1p7b/checkpoints_eval/standard_rl_v2_control_pilot128_step64_hf", "model_sha256": MODEL_HASHES["grpo"], "tokenizer_sha256": TOKENIZER_SHA256},
            "cpi3": {"checkpoint": "repro_1p7b/graph_frontier/recursive_opd_v3_experiment_run/cycle3/checkpoint", "model_sha256": MODEL_HASHES["cpi3"], "tokenizer_sha256": TOKENIZER_SHA256},
        },
        "paired_completed": len(paired),
        "grpo_pass": grpo_pass,
        "cpi3_pass": cpi3_pass,
        "grpo_accuracy": grpo_pass / len(paired) if paired else None,
        "cpi3_accuracy": cpi3_pass / len(paired) if paired else None,
        **counts,
        "diagnostics": {
            arm: {key: mean_metric([row[arm] for row in paired], key) for key in ("turns", "tool_calls", "runtime_seconds")}
            for arm in ("grpo", "cpi3")
        },
        "tasks": rows,
    }


def markdown(result: dict) -> str:
    n = result["paired_completed"]
    lines = [
        "# MCPMark Verified Filesystem paired pilot",
        "",
        "**PRELIMINARY PAIRED PILOT — NOT FULL MCPMARK SCORE.**",
        "",
        f"Official commit: `{result['benchmark']['commit']}`; standard suite, Filesystem, k=1, MCPMarkAgent.",
        f"Frozen manifest SHA256: `{MANIFEST_SHA256}`. Two independent mutable roots; same task order and protocol.",
        "Official agent temperature=1.0; both arms use the identical 8192-token transport cap to fit the checkpoints' native 40960-token context.",
        "",
        f"Paired completed: **{n}/10**. GRPO: **{result['grpo_pass']}/{n}**; Cπ3: **{result['cpi3_pass']}/{n}**.",
        f"Cπ3 wins {result['cpi3_wins']}; GRPO wins {result['grpo_wins']}; tie-pass {result['tie_pass']}; tie-fail {result['tie_fail']}.",
        "",
        "| Task | GRPO | Cπ3 | Relation |",
        "|---|---|---|---|",
    ]
    for row in result["tasks"]:
        a, b = row["grpo"], row["cpi3"]
        va = "PENDING" if a is None else ("INFRA" if a["status"] == "INFRA_FAILURE" else ("PASS" if a["pass"] else "FAIL"))
        vb = "PENDING" if b is None else ("INFRA" if b["status"] == "INFRA_FAILURE" else ("PASS" if b["pass"] else "FAIL"))
        relation = "unpaired pending"
        if va in ("PASS", "FAIL") and vb in ("PASS", "FAIL"):
            relation = "Cπ3 win" if vb == "PASS" and va == "FAIL" else "GRPO win" if va == "PASS" and vb == "FAIL" else "tie-pass" if va == "PASS" else "tie-fail"
        lines.append(f"| {row['category']}/{row['task_id']} | {va} | {vb} | {relation} |")
    lines += ["", "Failure classification across all ten attempted tasks:", ""]
    for arm in ("grpo", "cpi3"):
        model_fail = sum(row[arm] is not None and row[arm]["status"] == "MODEL_FAIL" for row in result["tasks"])
        infra = [
            f"{row['category']}/{row['task_id']}"
            for row in result["tasks"]
            if row[arm] is not None and row[arm]["status"] == "INFRA_FAILURE"
        ]
        lines.append(f"- {arm}: MODEL_FAIL={model_fail}; INFRA_FAILURE={len(infra)} ({', '.join(infra) or 'none'}); TIMEOUT=0.")
    lines += ["", "No directional signal in the valid paired subset.", ""]
    lines += ["", "Diagnostics on paired tasks only:", ""]
    for arm in ("grpo", "cpi3"):
        d = result["diagnostics"][arm]
        lines.append(f"- {arm}: mean turns {d['turns']}; mean tool calls {d['tool_calls']}; mean runtime {d['runtime_seconds']} s.")
    lines += ["", "Representative official verifier failures:", ""]
    shown = 0
    for row in result["tasks"]:
        for arm in ("grpo", "cpi3"):
            value = row[arm]
            if value and value["status"] == "MODEL_FAIL" and value["verifier_failures"]:
                lines.append(f"- {row['category']}/{row['task_id']} ({arm}): {value['verifier_failures'][0]}")
                shown += 1
                if shown >= 3:
                    break
        if shown >= 3:
            break
    lines += ["", "This small, category-stratified pilot cannot support a final MCPMark claim. Raw official meta/messages/log paths are indexed in the companion JSON.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.manifest, args.runtime)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    stem = args.report_dir / "mcpmark_grpo_vs_cpi3_pilot"
    for extension in (".json", ".md"):
        if stem.with_suffix(extension).exists():
            raise SystemExit(f"refusing to overwrite existing report: {stem.with_suffix(extension)}")
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(f"paired={result['paired_completed']} grpo={result['grpo_pass']} cpi3={result['cpi3_pass']}")


if __name__ == "__main__":
    main()
