import asyncio
import json
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agents import ModelSettings
from src.manager.mcp_client_manager import MCPManager


def main() -> None:
    required = ("SGLANG_BASE_URL", "SGLANG_API_KEY", "SGLANG_MODEL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}")

    module = runpy.run_path(str(REPO_ROOT / "repro_1p7b/scripts/toolgraph_smoke.py"))
    graph, chain = module["graph"], module["chain"]
    future = asyncio.run_coroutine_threadsafe(
        MCPManager.register_mcp_server_async(
            "CampusCard", "envs/tools/CampusCard.py", False
        ),
        MCPManager._loop,
    )
    future.result(timeout=30)

    from src.gen.query_gen import QueryGen, QueryGenConfig

    config = QueryGenConfig(
        model_name="sglang",
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        pass_k=1,
        max_iterations=5,
        max_retry=3,
        max_solve_iterations=3,
        max_refine_iterations=0,
        enable_split_turns=False,
        enable_query_refinement=False,
        enable_user_interaction=False,
        enable_user_tool_use=False,
        enable_user_verification=False,
        enable_filteration=False,
        enable_log_thinking_content=False,
        save_folder="repro_1p7b/results/querygen",
        log_folder="repro_1p7b/logs/querygen",
    )
    query_gen = QueryGen(graph, config)
    settings = ModelSettings(
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    agent_names = (
        "scenario_planner",
        "schema_generator",
        "query_generator",
        "query_refiner",
        "query_solver",
        "solution_selector",
        "user_tool_classifier",
    )
    for name in agent_names:
        getattr(query_gen._impl, name).model_settings = settings

    try:
        result = asyncio.run(query_gen.gen(chain))
        node = result[0] if len(result) else None
        summary = {
            "status": "PASS" if getattr(node, "decision", None) else "BLOCKED",
            "turns": len(result),
            "scenario_generated": bool(result.scenario),
            "decision": getattr(node, "decision", None),
            "query": getattr(node, "query", None),
            "steps": len(getattr(node, "steps", []) or []),
            "accuracy": getattr(node, "accuracy", None),
        }
        print("QUERYGEN_SMOKE=" + json.dumps(summary, ensure_ascii=False))
        if summary["status"] != "PASS":
            raise SystemExit(2)
    finally:
        MCPManager.shutdown()


if __name__ == "__main__":
    main()
