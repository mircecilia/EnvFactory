"""Run a deterministic multi-case real Graph-Frontier pipeline smoke.

This is a pipeline-stability probe with an explicit execution instruction. It is
not an autonomous model-capability evaluation. Runtime artifacts belong in /tmp.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from repro_1p7b.graph_frontier.real_probe_smoke import run_probe
from src.manager.mcp_client_manager import MCPManager


CASES: tuple[dict[str, Any], ...] = (
    {
        "user_id": "student-001",
        "balance": 20.0,
        "amount": 50.0,
        "payment_method": "bank_card",
    },
    {
        "user_id": "alice-42",
        "balance": 0.0,
        "amount": 1.0,
        "payment_method": "alipay",
    },
    {
        "user_id": "CARD-9007",
        "balance": 1.5,
        "amount": 5.5,
        "payment_method": "wechat",
    },
    {
        "user_id": "u_0004",
        "balance": 99.9,
        "amount": 25.0,
        "payment_method": "bank_card",
    },
    {
        "user_id": "s-2026-005",
        "balance": 250.0,
        "amount": 88.0,
        "payment_method": "alipay",
    },
    {
        "user_id": "mixedCase06",
        "balance": 1000.0,
        "amount": 100.0,
        "payment_method": "wechat",
    },
    {
        "user_id": "000007",
        "balance": 7.0,
        "amount": 250.0,
        "payment_method": "bank_card",
    },
    {
        "user_id": "hyphen-user-08",
        "balance": 12.34,
        "amount": 0.5,
        "payment_method": "alipay",
    },
    {
        "user_id": "CAPS_USER_09",
        "balance": 500.0,
        "amount": 75.25,
        "payment_method": "wechat",
    },
    {
        "user_id": "student-010",
        "balance": 33.0,
        "amount": 333.0,
        "payment_method": "bank_card",
    },
    {
        "user_id": "edge_11",
        "balance": 0.01,
        "amount": 0.01,
        "payment_method": "alipay",
    },
    {
        "user_id": "final-user-12",
        "balance": 876.5,
        "amount": 123.45,
        "payment_method": "wechat",
    },
)


def _mean_metric(summaries: list[dict[str, Any]], name: str) -> float | str:
    values = [
        summary.get("trace_quality_metrics", {}).get(name)
        for summary in summaries
        if isinstance(summary.get("trace_quality_metrics", {}).get(name), (int, float))
    ]
    if not values:
        return "unknown"
    return sum(values) / len(values)


def _write_batch_summary(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    requested: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    passed = sum(summary.get("status") == "PASS" for summary in summaries)
    completed = len(summaries) + len(failures)
    result = {
        "status": "PASS" if completed == requested and passed == requested else "FAIL",
        "probe_kind": "explicit_execution_pipeline_stability",
        "capability_claim": False,
        "requested_cases": requested,
        "completed_cases": completed,
        "passed_cases": passed,
        "failed_cases": requested - passed,
        "elapsed_seconds": elapsed_seconds,
        "coverage": {
            "distinct_user_ids": len({case["user_id"] for case in CASES[:requested]}),
            "payment_methods": sorted(
                {case["payment_method"] for case in CASES[:requested]}
            ),
            "initial_balance_range": [
                min(case["balance"] for case in CASES[:requested]),
                max(case["balance"] for case in CASES[:requested]),
            ],
            "recharge_amount_range": [
                min(case["amount"] for case in CASES[:requested]),
                max(case["amount"] for case in CASES[:requested]),
            ],
        },
        "aggregate_trace_quality": {
            "events": sum(
                summary.get("trace_quality_metrics", {}).get("events", 0)
                for summary in summaries
            ),
            "typed_value_recovery_rate": _mean_metric(
                summaries, "typed_value_recovery_rate"
            ),
            "typed_execution_status_rate": _mean_metric(
                summaries, "typed_execution_status_rate"
            ),
        },
        "case_summaries": summaries,
        "exceptions": failures,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "batch_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/envfactory_graph_frontier_mini_probe"),
    )
    parser.add_argument("--limit", type=int, default=len(CASES))
    args = parser.parse_args()
    if not 1 <= args.limit <= len(CASES):
        raise ValueError(f"--limit must be between 1 and {len(CASES)}")

    missing = [
        name
        for name in ("SGLANG_BASE_URL", "SGLANG_API_KEY", "SGLANG_MODEL")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(f"missing environment variables: {missing}")

    registration = asyncio.run_coroutine_threadsafe(
        MCPManager.register_mcp_server_async(
            "CampusCard",
            "envs/tools/CampusCard.py",
            False,
        ),
        MCPManager._loop,
    )
    registration.result(timeout=60)

    started = time.monotonic()
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(CASES[: args.limit], start=1):
            task_id = f"graph-frontier-mini-{index:03d}"
            print(
                f"CASE_START {index}/{args.limit} {task_id} "
                f"user_id={case['user_id']} amount={case['amount']} "
                f"payment_method={case['payment_method']}",
                flush=True,
            )
            try:
                summary = asyncio.run(
                    run_probe(
                        args.output_dir / task_id,
                        task_id=task_id,
                        user_id=case["user_id"],
                        balance=case["balance"],
                        amount=case["amount"],
                        payment_method=case["payment_method"],
                        seed=41 + index,
                    )
                )
                summaries.append(summary)
                print(
                    f"CASE_END {index}/{args.limit} {task_id} "
                    f"status={summary['status']}",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "task_id": task_id,
                    "case": case,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }
                failures.append(failure)
                print(
                    f"CASE_END {index}/{args.limit} {task_id} "
                    f"status=EXCEPTION error={exc!r}",
                    flush=True,
                )
            _write_batch_summary(
                args.output_dir,
                summaries,
                failures,
                args.limit,
                time.monotonic() - started,
            )
    finally:
        MCPManager.shutdown()

    result = _write_batch_summary(
        args.output_dir,
        summaries,
        failures,
        args.limit,
        time.monotonic() - started,
    )
    print("GRAPH_FRONTIER_MINI_PROBE=" + json.dumps(result, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
