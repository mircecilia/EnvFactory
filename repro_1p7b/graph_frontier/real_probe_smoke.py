"""One-task real Graph-Frontier smoke using a caller-owned SGLang endpoint.

This probe deliberately bypasses scenario generation by supplying one valid,
fixed CampusCard scenario. It still exercises the real QueryGen query/solve
path, FastMCP tool execution, generation-time gold export, typed rollout export,
and profiler without modifying core src/.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import jsonschema
from agents import ModelSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Core graph imports initialize the legacy LLMClient even though this probe uses
# only the SGLang agent path. Match the existing CPU graph smoke placeholders.
os.environ.setdefault("CHAT_API_KEY", "local-smoke-placeholder")
os.environ.setdefault("CHAT_URL", "http://127.0.0.1:1/v1")
os.environ.setdefault("CHAT_MODEL", "unused")
os.environ.setdefault("EMBEDDING_API_KEY", "local-smoke-placeholder")
os.environ.setdefault("EMBEDDING_URL", "http://127.0.0.1:1")
os.environ.setdefault("EMBEDDING_MODEL", "unused")

from repro_1p7b.graph_frontier.adapters import UNKNOWN
from repro_1p7b.graph_frontier.export_pipeline import profile_sidecar_and_trace
from repro_1p7b.graph_frontier.fastmcp_adapter import (
    fastmcp_execution_success,
    fastmcp_result_fields,
)
from repro_1p7b.graph_frontier.gold_sidecar import GenerationSidecarCallback
from repro_1p7b.graph_frontier.rollout_trace import TypedRolloutRecorder
from repro_1p7b.graph_frontier.traceable_sampler import sample_with_dependency_trace
from src.gen.query_gen import QueryGenConfig, QueryGenContext, QueryGenState
from src.gen.query_gen.query_gen_non_conv import QueryGenNonConv
from src.graph.tool_chain import ToolQueryNode
from src.manager.mcp_client_manager import MCPManager


def make_scenario(user_id: str, balance: float) -> dict[str, Any]:
    return {
        "CampusCard": {
            "accounts": {
                user_id: {
                    "userId": user_id,
                    "name": "Graph Frontier Probe User",
                    "password": "not-used-by-this-probe",
                    "balance": balance,
                    "currency": "CNY",
                    "status": 1,
                    "phone": None,
                    "email": None,
                    "address": None,
                }
            },
            "transactions": {user_id: []},
            "statusTextMap": {
                1: "normal",
                2: "lost",
                3: "system frozen",
                4: "closed",
                5: "pre-closed",
                6: "manually frozen",
            },
            "current_time": "2026-09-14 10:00:00",
        }
    }


class ProbeQueryGenNonConv(QueryGenNonConv):
    def __init__(self, *args: Any, sidecar_callback: GenerationSidecarCallback, **kwargs: Any) -> None:
        self.sidecar_callback = sidecar_callback
        self.gold_path: Path | None = None
        super().__init__(*args, **kwargs)

    async def terminate(self, context: QueryGenContext) -> None:
        self.gold_path = self.sidecar_callback.before_save(context)
        await super().terminate(context)


def _load_smoke_graph(
    scenario: dict[str, Any],
    user_id: str,
    seed: int,
) -> tuple[Any, Any]:
    module = runpy.run_path(str(REPO_ROOT / "repro_1p7b/scripts/toolgraph_smoke.py"))
    graph = module["graph"]
    chain = sample_with_dependency_trace(
        graph,
        module["sampler"],
        max_nodes=2,
        start_node=module["recharge"],
        seed=seed,
    )
    names = [tool.name for tool in chain.init_tool_chain]
    expected = ["CampusCard-query_balance", "CampusCard-recharge"]
    if names != expected:
        raise RuntimeError(f"unexpected sampled chain: {names!r}")
    chain.tool_chain = [
        ToolQueryNode(
            raw_tool_call=list(chain.init_tool_chain),
            initial_scenario=json.loads(json.dumps(scenario)),
        )
    ]
    chain.scenario = (
        f"The user wants to inspect the balance of campus card {user_id} and then "
        "recharge it using a supported payment method."
    )
    return graph, chain


def _install_typed_trace(recorder: TypedRolloutRecorder):
    original = MCPManager._call_tool_async
    step_indices = itertools.count()
    controls = {"load_scenario", "save_scenario"}

    async def traced(tool_name: str, tool_args: Any, client: Any, client_id: str) -> str:
        short_name = tool_name.split("-", 1)[-1]
        args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
        if short_name in controls:
            return await original(tool_name, tool_args, client, client_id)

        server_name = client_id.split("-", 1)[0]

        async def execute(_name: str, _args: Any) -> Any:
            return await client.call_tool(short_name, args)

        async def execute_and_record() -> Any:
            return await recorder.record_async_call(
                next(step_indices),
                tool_name,
                args,
                execute,
                success_from_response=fastmcp_execution_success,
                fields_from_response=fastmcp_result_fields,
                environment_id=server_name,
                server_id=server_name,
            )

        if server_name in MCPManager.stateless_clients:
            async with MCPManager._stateless_lock:
                result = await execute_and_record()
        else:
            async with client:
                result = await execute_and_record()
        return ",".join(
            item.text for item in result.content if hasattr(item, "text")
        )

    MCPManager._call_tool_async = traced
    return original


def _validate(path: Path, schema_name: str) -> None:
    schema = json.loads(
        (REPO_ROOT / "repro_1p7b/graph_frontier" / schema_name).read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), schema)


async def run_probe(
    output_dir: Path,
    *,
    task_id: str = "graph-frontier-real-smoke-001",
    user_id: str = "student-001",
    balance: float = 20.0,
    amount: float = 50.0,
    payment_method: str = "bank_card",
    seed: int = 42,
) -> dict[str, Any]:
    if payment_method not in {"alipay", "wechat", "bank_card"}:
        raise ValueError(f"unsupported payment method: {payment_method}")
    if amount <= 0:
        raise ValueError("amount must be positive")
    scenario = make_scenario(user_id, balance)
    graph, chain = _load_smoke_graph(scenario, user_id, seed)
    node = chain.tool_chain[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    querygen_dir = output_dir / "querygen"
    gold_dir = output_dir / "gold"
    callback = GenerationSidecarCallback(
        gold_dir,
        task_id_factory=lambda _context: task_id,
    )
    recorder = TypedRolloutRecorder(
        task_id,
        environment_identifiers=["CampusCard"],
        initial_environment_state=node.initial_scenario,
        auto_timestamp=True,
    )
    config = QueryGenConfig(
        model_name="sglang",
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        pass_k=1,
        max_iterations=4,
        max_retry=3,
        max_solve_iterations=4,
        max_refine_iterations=0,
        enable_split_turns=False,
        enable_query_refinement=False,
        enable_user_interaction=False,
        enable_user_tool_use=False,
        enable_user_verification=False,
        enable_filteration=False,
        enable_log_thinking_content=False,
        save_folder=str(querygen_dir),
        log_folder=str(output_dir / "querygen_logs"),
    )
    generator = ProbeQueryGenNonConv(
        graph,
        config,
        sidecar_callback=callback,
    )
    settings = ModelSettings(
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for agent_name in (
        "query_generator",
        "query_solver",
        "solution_selector",
    ):
        getattr(generator, agent_name).model_settings = settings

    context = QueryGenContext(
        config=config,
        tool_graph=graph,
        tool_chain=chain,
        idx=0,
        conversation_id="graphfrontier" + "".join(
            char for char in task_id if char.isalnum()
        ),
        user_tools={},
    )
    generator.context_manager.add_prompt(
        generator.query_generator.name,
        f"{context.conversation_id}{context.idx}",
        (
            "Generate a natural request that explicitly includes campus card ID "
            f"{user_id}, asks to check its balance first, and then recharge "
            f"exactly {amount:g} CNY using {payment_method}. "
            "Preserve all three literal values."
        ),
    )
    original_call = _install_typed_trace(recorder)
    try:
        state = QueryGenState.Generating
        for _attempt in range(config.max_retry):
            state = await generator.generate(context)
            if state == QueryGenState.Solving:
                break
        if state != QueryGenState.Solving:
            raise RuntimeError(f"query generation did not reach Solving: {state}")
        generator.context_manager.add_prompt(
            f"{generator.query_solver.name}_0",
            f"{context.conversation_id}{context.idx}0",
            (
                f"{node.query}\nThis is an execution smoke: call query_balance first, "
                "then call recharge with the returned userId, "
                f"amount {amount:g}, and paymentMethod {payment_method}. "
                "Do not emit a natural-language response "
                "until both tool calls have completed."
            ),
        )
        await generator.solve_and_select(context)
        await generator.terminate(context)
    finally:
        MCPManager._call_tool_async = original_call

    if generator.gold_path is None:
        raise RuntimeError("generation callback did not produce a gold sidecar")

    recorder.trace["initial_environment_state"] = json.loads(
        json.dumps(node.initial_scenario)
    )
    final_status = (
        "success" if node.decision is True
        else "failure" if node.decision is False
        else UNKNOWN
    )
    recorder.finalize(
        final_status=final_status,
        final_environment_state=(
            node.final_scenario if node.final_scenario is not None else UNKNOWN
        ),
        verifier_result=UNKNOWN,
        verifier_details={
            "status": UNKNOWN,
            "reason": "no_independent_runtime_verifier",
            "expected_state_provenance": "selected_querygen_reference_trajectory",
        },
    )
    rollout_path = recorder.write(output_dir / f"{task_id}.rollout.json")
    profile = profile_sidecar_and_trace(generator.gold_path, rollout_path)
    profile_path = output_dir / f"{task_id}.profile.json"
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _validate(generator.gold_path, "GOLD_SIDECAR_SCHEMA.json")
    _validate(rollout_path, "ROLLOUT_TRACE_SCHEMA.json")
    _validate(profile_path, "profile.schema.json")

    sidecar = json.loads(generator.gold_path.read_text(encoding="utf-8"))
    trace = recorder.as_dict()
    resolution = sidecar["dependency_resolution"]
    quality = trace["trace_quality_metrics"]
    edge_checks = profile["internal_parameter_flow_checks"]
    typed_edges = bool(edge_checks) and all(
        check.get("source_value_available") is True
        and check.get("target_value_available") is True
        for check in edge_checks
    )
    checks = {
        "dependency_semantics_selected_reference": (
            sidecar["dependency_semantics"] == "selected_reference"
        ),
        "selected_equals_resolved": (
            resolution["selected_dependency_count"]
            == resolution["resolved_dependency_count"]
        ),
        "structural_diagnosis_eligible": (
            sidecar["structural_diagnosis_eligible"] is True
        ),
        "no_cross_turn_dependency": (
            resolution["cross_turn_dependency_count"] == 0
        ),
        "typed_execution_status_rate_1": (
            quality["typed_execution_status_rate"] == 1.0
        ),
        "dependency_edge_present": bool(sidecar["dependency_edges"]),
        "typed_dependency_values_present": typed_edges,
        "state_result_available": (
            isinstance(profile["state_success"], bool)
            or (
                profile["state_success"] == UNKNOWN
                and recorder.trace["verifier_details"]["reason"]
                == "no_independent_runtime_verifier"
            )
        ),
        "no_unresolved_dependency": (
            resolution["unresolved_dependency_count"] == 0
        ),
        "profile_contract_present": all(
            key in profile
            for key in (
                "path_adherence",
                "structural_root_cause_failures",
                "task_success",
                "max_dependency_depth_reached",
            )
        ),
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "task_id": task_id,
        "case": {
            "user_id": user_id,
            "initial_balance": balance,
            "amount": amount,
            "payment_method": payment_method,
            "seed": seed,
        },
        "query": node.query,
        "selected_querygen_decision": node.decision,
        "selected_querygen_accuracy": node.accuracy,
        "gold_path": str(generator.gold_path),
        "rollout_path": str(rollout_path),
        "profile_path": str(profile_path),
        "gold_tool_sequence": [
            item["tool_name"] for item in sidecar["gold_tool_sequence"]
        ],
        "dependency_resolution": resolution,
        "trace_quality_metrics": quality,
        "state_success": profile["state_success"],
        "task_success": profile["task_success"],
        "path_adherence": profile["path_adherence"],
        "max_dependency_depth_reached": profile["max_dependency_depth_reached"],
        "checks": checks,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/envfactory_graph_frontier_real_smoke"),
    )
    parser.add_argument("--task-id", default="graph-frontier-real-smoke-001")
    parser.add_argument("--user-id", default="student-001")
    parser.add_argument("--balance", type=float, default=20.0)
    parser.add_argument("--amount", type=float, default=50.0)
    parser.add_argument(
        "--payment-method",
        choices=("alipay", "wechat", "bank_card"),
        default="bank_card",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    required = ("SGLANG_BASE_URL", "SGLANG_API_KEY", "SGLANG_MODEL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing environment variables: {missing}")

    future = asyncio.run_coroutine_threadsafe(
        MCPManager.register_mcp_server_async(
            "CampusCard",
            "envs/tools/CampusCard.py",
            False,
        ),
        MCPManager._loop,
    )
    future.result(timeout=60)
    try:
        summary = asyncio.run(
            run_probe(
                args.output_dir,
                task_id=args.task_id,
                user_id=args.user_id,
                balance=args.balance,
                amount=args.amount,
                payment_method=args.payment_method,
                seed=args.seed,
            )
        )
        print("GRAPH_FRONTIER_REAL_SMOKE=" + json.dumps(summary, ensure_ascii=False))
        if summary["status"] != "PASS":
            raise SystemExit(2)
    finally:
        MCPManager.shutdown()


if __name__ == "__main__":
    main()
