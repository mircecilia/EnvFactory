"""Frozen 300-probe Graph-Frontier confirmation study. No training code."""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from agents import ModelSettings
    from repro_1p7b.graph_frontier import pilot_static as pilot
    from src.gen.query_gen import QueryGenConfig, QueryGenContext
    from src.gen.query_gen.query_gen_non_conv import QueryGenNonConv
    from src.manager.mcp_client_manager import MCPManager
except ModuleNotFoundError:
    ModelSettings = None
    pilot = None
    QueryGenConfig = None
    QueryGenContext = None
    QueryGenNonConv = None
    MCPManager = None
from repro_1p7b.graph_frontier.export_pipeline import profile_sidecar_and_trace
from repro_1p7b.graph_frontier.gold_sidecar import write_gold_sidecar
from repro_1p7b.graph_frontier.pilot_reanalysis_v2 import MIXED, READ_ONLY, reference_call
from repro_1p7b.graph_frontier.rollout_trace import TypedRolloutRecorder
from repro_1p7b.graph_frontier.state_verifier import compare_final_states

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260914
COUNT = 300
PATHS = {
    "base": "repro_1p7b/models/Qwen3-1.7B",
    "original_sft": "repro_1p7b/checkpoints/baseline_sft_8k_1p7b",
    "parameter_aware": "repro_1p7b/checkpoints/parameter_aware_sft_8k_1p7b",
    "dynamic_v1": "repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b",
}
PROTOCOL = {
    "temperature": 0.0, "top_p": 1.0, "presence_penalty": 0.0,
    "max_new_tokens": 1024, "max_tool_calls": 8, "max_tool_rounds": 4,
    "max_turns": 1, "probe_timeout_seconds": 120,
    "retry_policy": "no outer retry",
    "stop_condition": "first no-tool response or four rounds",
    "environment_reset": "fresh client id and load_scenario",
    "thinking": False,
}
TARGET_COUNTS = {
    "symbol_info": 14,
    "quote_order": 19,
    "quote_order_details": 35,
    "city_weather": 13,
    "city_forecast": 13,
    "city_alerts": 13,
    "city_save": 13,
    "list_get": 13,
    "list_task": 13,
    "task_get": 18,
    "task_update": 18,
    "task_delete": 18,
    "estimate_create": 14,
    "estimate_query": 19,
    "estimate_cancel": 35,
    "create_update": 14,
    "create_update_delete": 18,
}
EXPECTED_DEPTH = {
    "symbol_info": "1",
    "quote_order": "2",
    "quote_order_details": "3+",
    "city_weather": "1",
    "city_forecast": "1",
    "city_alerts": "1",
    "city_save": "1",
    "list_get": "1",
    "list_task": "1",
    "task_get": "2",
    "task_update": "2",
    "task_delete": "2",
    "estimate_create": "1",
    "estimate_query": "2",
    "estimate_cancel": "3+",
    "create_update": "1",
    "create_update_delete": "2",
}
OBSERVATION_TOOL = {
    "symbol_info": "get_stock_info",
    "city_weather": "get_current_weather",
    "city_forecast": "get_forecast",
    "city_alerts": "get_alerts",
    "quote_order_details": "get_order_details",
    "list_get": "get_task_list",
    "task_get": "get_task",
    "estimate_query": "query_order",
    "estimate_cancel": "cancel_order",
}


def stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coord_key(location: Mapping[str, Any]) -> str:
    return f"{location['latitude']:.4f},{location['longitude']:.4f}"


def weather_candidate(template: str, variant: int) -> dict[str, Any]:
    candidates: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    source_key = {
        "city_weather": "current_weather",
        "city_forecast": "forecasts",
        "city_alerts": "alerts",
    }.get(template)
    for scenario in pilot.scenarios("Weather"):
        for city, location in sorted(scenario["city_database"].items()):
            key = coord_key(location)
            if source_key is None:
                candidates.append((scenario, city, location))
            elif source_key == "alerts":
                if scenario[source_key].get(key):
                    candidates.append((scenario, city, location))
            elif key in scenario[source_key]:
                candidates.append((scenario, city, location))
    if not candidates:
        raise RuntimeError(f"no_supported_weather_state:{template}")
    scenario, _, location = candidates[variant % len(candidates)]
    state = copy.deepcopy(scenario)
    city = f"GF300 {template.replace('_', ' ')} {variant:03d}"
    state["city_database"][city] = copy.deepcopy(location)
    calls = [{"city": city}]
    if template == "city_weather":
        calls.append({"latitude": "@0.location.latitude", "longitude": "@0.location.longitude"})
        query = f"Find {city}'s coordinates and report the current weather there."
    elif template == "city_forecast":
        days = 2 + variant % 3
        calls.append({
            "latitude": "@0.location.latitude",
            "longitude": "@0.location.longitude",
            "days": days,
        })
        query = f"Find {city}'s coordinates and give its {days}-day forecast."
    elif template == "city_alerts":
        calls.append({
            "latitude": "@0.location.latitude",
            "longitude": "@0.location.longitude",
            "active_only": True,
        })
        query = f"Find {city}'s coordinates and report active weather alerts there."
    else:
        alias = f"gf300-{variant:03d}"
        calls.append({
            "alias": alias,
            "latitude": "@0.location.latitude",
            "longitude": "@0.location.longitude",
            "name": f"{city} saved",
        })
        query = f"Find {city}'s coordinates and save them as alias {alias}, named '{city} saved'."
    return {"initial_state": state, "query": query, "call_args": calls}


def make_candidate(spec: Mapping[str, Any], variant: int) -> dict[str, Any]:
    candidate = pilot.make(dict(spec), variant)
    template = str(candidate["template"])
    if candidate["server"] == "Weather":
        candidate.update(weather_candidate(template, variant))
    elif candidate["server"] == "TradingBot":
        first = candidate["call_args"][0]["company_name"]
        symbol = candidate["initial_state"]["company_symbols"][first]
        alias = f"GF300 Company {variant:03d}"
        candidate["initial_state"]["company_symbols"][alias] = symbol
        candidate["call_args"][0]["company_name"] = alias
        if template == "symbol_info":
            candidate["query"] = f"Look up {alias}'s ticker and current quote."
        else:
            quantity = 1 + variant % 3
            candidate["call_args"][2]["quantity"] = quantity
            candidate["query"] = (
                f"Look up {alias}'s ticker and current quote, then buy {quantity} "
                "share(s) at that quoted price"
            )
            if template == "quote_order_details":
                candidate["query"] += ", and retrieve the created order details"
            candidate["query"] += "."
    candidate["candidate_id"] = "confirm300-" + candidate["candidate_id"]
    return candidate


def text_values(template: str, candidate: Mapping[str, Any], reference: Sequence[Mapping[str, Any]]) -> list[Any]:
    tool = OBSERVATION_TOOL.get(template)
    row = reference_call(reference, tool) if tool else None
    result = row.get("result") if row else None
    if template == "symbol_info":
        return [result.get("symbol"), result.get("price")]
    if template == "city_weather":
        return [candidate["call_args"][0]["city"], result.get("temperature")]
    if template == "city_forecast":
        return [
            candidate["call_args"][0]["city"],
            result.get("temperature"),
            result.get("weather_conditions"),
        ]
    if template == "city_alerts":
        alerts = result.get("alerts", []) if isinstance(result, Mapping) else []
        return [candidate["call_args"][0]["city"], alerts[0].get("event"), alerts[0].get("severity")] if alerts else []
    if template == "quote_order_details":
        return [result.get("order_id"), result.get("symbol"), result.get("status")]
    if template == "list_get":
        return [result.get("tasklist_id"), result.get("title")]
    if template == "task_get":
        return [result.get("task_id"), result.get("title")]
    if template == "estimate_query":
        return [result.get("orderCode"), result.get("state")]
    if template == "estimate_cancel":
        return [result.get("orderCode"), "cancel"]
    return []


def semantic_spec(candidate: Mapping[str, Any], reference: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    template = str(candidate["template"])
    verifier_type = (
        "typed_observation_predicate"
        if template in READ_ONLY
        else "state_plus_observation"
        if template in MIXED
        else "state_predicate"
    )
    required = text_values(template, candidate, reference)
    supported = verifier_type == "state_predicate" or (
        bool(required) and all(value not in (None, "", {}, []) for value in required)
    )
    return {
        "semantic_verifier_supported": supported,
        "verifier_type": verifier_type,
        "checked_fields": (
            ["deterministic target state predicate"]
            if verifier_type == "state_predicate"
            else ["typed observation predicate", "final_assistant_content"]
        ),
        "reference_source": "typed reference trace and canonical final state",
        "text_match_mode": "nfkc_casefold_all_normalized_values" if required else "not_applicable",
        "text_match_values": required,
        "ambiguity_flag": any(len(str(value)) < 3 for value in required),
        "unsupported_reason": None if supported else "no_reliable_deterministic_text_target",
    }


def assign_splits(rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SEED}:heldout:{row['task_id']}".encode()
        ).hexdigest(),
    )
    heldout = {row["task_id"] for row in ranked[:60]}
    for row in rows:
        row["split"] = "heldout" if row["task_id"] in heldout else "diagnosis"


def compact_record(row: Mapping[str, Any]) -> dict[str, Any]:
    hidden = {"dependency_edges"}
    return {
        key: value for key, value in row.items() if key not in hidden
    } | {
        "dependency_edge_count": len(row["dependency_edges"]),
        "gold_sidecar_sha256": sha256_path(ROOT / row["gold_sidecar_path"]),
    }


def validate_models() -> dict[str, Any]:
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    answer = {}
    for label, relative in PATHS.items():
        model = ROOT / relative
        missing = [name for name in required if not (model / name).is_file()]
        weights = sorted(model.glob("*.safetensors"))
        if missing or not weights:
            raise RuntimeError(f"invalid_model:{label}:missing={missing}:weights={len(weights)}")
        answer[label] = {
            "path": relative,
            "config": required[0],
            "tokenizer": [required[1], required[2]],
            "safetensors": [path.name for path in weights],
        }
    return answer


def freeze(root: Path, compact: Path, summary_path: Path) -> dict[str, Any]:
    frozen = root / "frozen"
    manifest_path = frozen / f"confirm_300_seed_{SEED}.jsonl"
    if manifest_path.exists():
        raise RuntimeError(f"frozen_manifest_already_exists:{manifest_path}")
    for name in ("gold", "states", "reference"):
        (frozen / name).mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    spec_by_template = {row["template"]: row for row in pilot.specs()}
    for template, target in TARGET_COUNTS.items():
        spec = spec_by_template[template]
        kept = 0
        variant = 100
        while kept < target and variant < 1000:
            candidate = make_candidate(spec, variant)
            audit_row = {
                "candidate_id": candidate["candidate_id"],
                "eligible": False,
                "rejection_reason": "unknown",
            }
            try:
                reference, final = pilot.reference(candidate)
                graph, chain, sidecar = pilot.graph(candidate, final)
                eligibility = sidecar["probe_eligibility"]
                reasons = eligibility.get(
                    "rejection_reasons", eligibility.get("reasons", [])
                )
                if sidecar["probe_eligibility"]["eligible"] is not True:
                    raise RuntimeError("sidecar:" + ",".join(reasons))
                semantic = semantic_spec(candidate, reference)
                if not semantic["semantic_verifier_supported"]:
                    raise RuntimeError("semantic_verifier_unsupported")
                initial_hash = pilot.digest(candidate["initial_state"])
                final_hash = pilot.digest(final)
                initial_path = frozen / "states" / f"{initial_hash}.json"
                final_path = frozen / "states" / f"{final_hash}.json"
                if not initial_path.exists():
                    initial_path.write_text(
                        json.dumps(candidate["initial_state"], ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                    )
                if not final_path.exists():
                    final_path.write_text(
                        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                    )
                gold_path = frozen / "gold" / f"{candidate['candidate_id']}.gold.json"
                reference_path = frozen / "reference" / f"{candidate['candidate_id']}.json"
                write_gold_sidecar(sidecar, gold_path)
                reference_path.write_text(
                    json.dumps(reference, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                )
                depth = int(sidecar["dependency_depth"])
                bucket = "3+" if depth >= 3 else str(depth)
                if bucket != EXPECTED_DEPTH[template]:
                    raise RuntimeError(f"unexpected_depth:{depth}")
                record = {
                    "schema_version": "graph_frontier_confirm_manifest_v1",
                    "task_id": candidate["candidate_id"],
                    "seed": SEED,
                    "eligible": True,
                    "environment": candidate["server"],
                    "environment_source": f"envs/tools/{candidate['server']}.py",
                    "environment_source_sha256": sha256_path(ROOT / f"envs/tools/{candidate['server']}.py"),
                    "scenario_source": "deterministic executable EnvFactory state",
                    "initial_state_sha256": initial_hash,
                    "expected_final_state_sha256": final_hash,
                    "initial_state_path": str(initial_path.relative_to(ROOT)),
                    "expected_final_state_path": str(final_path.relative_to(ROOT)),
                    "gold_sidecar_path": str(gold_path.relative_to(ROOT)),
                    "reference_trace_path": str(reference_path.relative_to(ROOT)),
                    "query": candidate["query"],
                    "gold_tools": [f"{candidate['server']}-{name}" for name in candidate["tools"]],
                    "dependency_edges": sidecar["dependency_edges"],
                    "dependency_depth": depth,
                    "complexity_bucket": bucket,
                    "task_kind": candidate["kind"],
                    "template": template,
                    "variant": variant,
                    "semantic_verifier": semantic,
                    "reference_metadata": {
                        "execution": "real FastMCP typed reference",
                        "call_count": len(reference),
                        "reset_roundtrip": True,
                    },
                }
                accepted.append(record)
                audit_row.update(eligible=True, rejection_reason=None)
                kept += 1
            except Exception as exc:
                reason = f"{type(exc).__name__}:{exc}"
                audit_row["rejection_reason"] = reason
                rejection_counts[reason] += 1
            audit.append(audit_row)
            variant += 1
        if kept != target:
            raise RuntimeError(f"template_quota_unfilled:{template}:{kept}/{target}")
    if len(accepted) != COUNT:
        raise RuntimeError(f"accepted_count:{len(accepted)}")
    assign_splits(accepted)
    accepted.sort(key=lambda row: row["task_id"])
    manifest_path.write_text("".join(stable(row) + "\n" for row in accepted))
    compact.parent.mkdir(parents=True, exist_ok=True)
    compact.write_text("".join(stable(compact_record(row)) + "\n" for row in accepted))
    (frozen / "candidate_audit.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "candidates_generated": len(audit),
                "accepted": len(accepted),
                "rejected": len(audit) - len(accepted),
                "rejection_reason_counts": dict(rejection_counts),
                "candidates": audit,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "schema_version": "graph_frontier_confirm_manifest_summary_v1",
        "seed": SEED,
        "candidates_generated": len(audit),
        "accepted": len(accepted),
        "rejected": len(audit) - len(accepted),
        "rejection_reason_counts": dict(rejection_counts),
        "split_distribution": dict(Counter(row["split"] for row in accepted)),
        "environment_distribution": dict(Counter(row["environment"] for row in accepted)),
        "template_distribution": dict(Counter(row["template"] for row in accepted)),
        "depth_distribution": dict(Counter(row["complexity_bucket"] for row in accepted)),
        "task_kind_distribution": dict(Counter(row["task_kind"] for row in accepted)),
        "gold_internal_edges": sum(len(row["dependency_edges"]) for row in accepted),
        "internal_parameter_count": sum(len(row["dependency_edges"]) for row in accepted),
        "semantic_verifier_support": sum(
            row["semantic_verifier"]["semantic_verifier_supported"] for row in accepted
        ),
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "compact_manifest_path": str(compact.relative_to(ROOT)),
        "manifest_sha256": sha256_path(manifest_path),
        "model_artifacts": validate_models(),
        "inference_protocol": PROTOCOL,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (frozen / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_frozen(root: Path, compact: Path, summary_path: Path) -> dict[str, Any]:
    manifest_path = root / "frozen" / f"confirm_300_seed_{SEED}.jsonl"
    rows = load_jsonl(manifest_path)
    compact_rows = load_jsonl(compact)
    summary = json.loads(summary_path.read_text())
    errors = []
    if len(rows) != COUNT:
        errors.append(f"manifest_count:{len(rows)}")
    if len(compact_rows) != COUNT:
        errors.append(f"compact_count:{len(compact_rows)}")
    if len({row["task_id"] for row in rows}) != COUNT:
        errors.append("duplicate_task_id")
    if len({row["query"] for row in rows}) != COUNT:
        errors.append("duplicate_query")
    if Counter(row["split"] for row in rows) != Counter({"diagnosis": 240, "heldout": 60}):
        errors.append("split_distribution")
    if Counter(row["complexity_bucket"] for row in rows) != Counter(
        {"1": 120, "2": 110, "3+": 70}
    ):
        errors.append("depth_distribution")
    if Counter(row["template"] for row in rows) != Counter(TARGET_COUNTS):
        errors.append("template_distribution")
    if not all(
        row["semantic_verifier"]["semantic_verifier_supported"] for row in rows
    ):
        errors.append("semantic_support")
    compact_ids = {row["task_id"] for row in compact_rows}
    if compact_ids != {row["task_id"] for row in rows}:
        errors.append("compact_task_ids")
    actual_sha = sha256_path(manifest_path)
    if actual_sha != summary.get("manifest_sha256"):
        errors.append("manifest_sha256")
    for row in rows:
        initial_path = ROOT / row["initial_state_path"]
        final_path = ROOT / row["expected_final_state_path"]
        gold_path = ROOT / row["gold_sidecar_path"]
        reference_path = ROOT / row["reference_trace_path"]
        if not all(path.is_file() for path in (initial_path, final_path, gold_path, reference_path)):
            errors.append(f"missing_artifact:{row['task_id']}")
            continue
        if pilot.digest(json.loads(initial_path.read_text())) != row["initial_state_sha256"]:
            errors.append(f"initial_hash:{row['task_id']}")
        if pilot.digest(json.loads(final_path.read_text())) != row["expected_final_state_sha256"]:
            errors.append(f"final_hash:{row['task_id']}")
        sidecar = json.loads(gold_path.read_text())
        if sidecar["task_id"] != row["task_id"]:
            errors.append(f"sidecar_task_id:{row['task_id']}")
        if len(sidecar["dependency_edges"]) != len(row["dependency_edges"]):
            errors.append(f"sidecar_edges:{row['task_id']}")
        reference = json.loads(reference_path.read_text())
        if len(reference) != row["reference_metadata"]["call_count"]:
            errors.append(f"reference_count:{row['task_id']}")
    if errors:
        raise RuntimeError("frozen_validation_failed:" + ",".join(errors[:20]))
    return {
        "valid": True,
        "manifest_count": len(rows),
        "compact_count": len(compact_rows),
        "semantic_support": sum(
            row["semantic_verifier"]["semantic_verifier_supported"] for row in rows
        ),
        "manifest_sha256": actual_sha,
    }


def reconstruct(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], Any, Any, Mapping[str, Any]]:
    initial = json.loads((ROOT / manifest["initial_state_path"]).read_text())
    expected = json.loads((ROOT / manifest["expected_final_state_path"]).read_text())
    sidecar = json.loads((ROOT / manifest["gold_sidecar_path"]).read_text())
    tools = [name.split("-", 1)[1] for name in manifest["gold_tools"]]
    dependencies = []
    for edge in sidecar["dependency_edges"]:
        dependencies.append((
            tools.index(edge["producer_tool"]["tool_name"].split("-", 1)[1]),
            edge["producer_output_parameter"]["parameter_name"],
            tools.index(edge["consumer_tool"]["tool_name"].split("-", 1)[1]),
            edge["consumer_input_parameter"]["parameter_name"],
        ))
    candidate = {
        "server": manifest["environment"],
        "tools": tools,
        "deps": dependencies,
        "initial_state": initial,
        "query": manifest["query"],
        "candidate_id": manifest["task_id"],
    }
    graph, chain, _ = pilot.graph(candidate, expected)
    return candidate, graph, chain, sidecar


def classify_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}:{exc}".casefold()
    if "timeout" in text:
        return "inference_timeout"
    if "scenario" in text or "environment" in text or "mcp" in text:
        return "environment_error"
    return "model_server_error"


async def run_one(manifest: Mapping[str, Any], label: str, output: Path) -> dict[str, Any]:
    tasks = output / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    result_path = tasks / f"{manifest['task_id']}.result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    candidate, graph, chain, sidecar = reconstruct(manifest)
    node = chain.tool_chain[0]
    recorder = TypedRolloutRecorder(
        manifest["task_id"],
        [manifest["environment"]],
        {manifest["environment"]: candidate["initial_state"]},
        auto_timestamp=True,
    )
    config = QueryGenConfig(
        model_name="sglang",
        temperature=PROTOCOL["temperature"],
        top_p=PROTOCOL["top_p"],
        presence_penalty=PROTOCOL["presence_penalty"],
        pass_k=1,
        max_iterations=1,
        max_retry=1,
        max_solve_iterations=PROTOCOL["max_tool_rounds"],
        max_refine_iterations=0,
        enable_split_turns=False,
        enable_query_refinement=False,
        enable_user_interaction=False,
        enable_user_tool_use=False,
        enable_user_verification=False,
        enable_filteration=False,
        enable_log_thinking_content=False,
        save_folder=str(output / "querygen"),
        log_folder=str(output / "querygen_logs"),
    )
    generator = QueryGenNonConv(graph, config)
    generator.query_solver.model_settings = ModelSettings(
        temperature=PROTOCOL["temperature"],
        top_p=PROTOCOL["top_p"],
        presence_penalty=PROTOCOL["presence_penalty"],
        max_tokens=PROTOCOL["max_new_tokens"],
        extra_body={"chat_template_kwargs": {"enable_thinking": PROTOCOL["thinking"]}},
    )
    context = QueryGenContext(
        config=config,
        tool_graph=graph,
        tool_chain=chain,
        idx=0,
        conversation_id="gfc" + hashlib.sha256((label + manifest["task_id"]).encode()).hexdigest()[:20],
        k=0,
        user_tools={},
    )
    original_hook = pilot.hook(recorder, PROTOCOL["max_tool_calls"])
    status = None
    started = time.monotonic()
    try:
        await asyncio.wait_for(generator.solve(context), timeout=PROTOCOL["probe_timeout_seconds"])
    except asyncio.TimeoutError:
        status = "inference_timeout"
    except Exception as exc:
        status = classify_exception(exc)
        recorder.trace["trace_notes"].append(f"{type(exc).__name__}:{exc}")
    finally:
        MCPManager._call_tool_async = original_hook
    steps = node.pass_k_trace.get(0, [])
    final_step = steps[-1] if steps and steps[-1].get("role") == "assistant" else None
    final_content = final_step.get("content") if final_step else None
    if status is None and len(steps) == 1:
        status = "model_server_error"
    finish_reason = status or ("stop" if final_step else "max_tool_rounds")
    final_state = node.pass_k_scenario.get(0, "unknown")
    state_ok = compare_final_states(
        final_state,
        {manifest["environment"]: json.loads((ROOT / manifest["expected_final_state_path"]).read_text())},
    )
    recorder.finalize(
        "success" if final_step else "failure",
        final_state,
        state_ok,
        {"source": "canonical_reference_state", "system_status": status or "none"},
    )
    recorder.trace["final_assistant_content"] = final_content if final_content is not None else "unknown"
    recorder.trace["final_assistant_finish_reason"] = finish_reason
    recorder.trace["assistant_trace"] = steps
    rollout_path = tasks / f"{manifest['task_id']}.rollout.json"
    recorder.write(rollout_path)
    try:
        profile = profile_sidecar_and_trace(sidecar, recorder.as_dict())
        profiler_error = None
    except Exception as exc:
        profile = {}
        profiler_error = f"{type(exc).__name__}:{exc}"
        status = "profiler_error"
    events = recorder.trace["events"]
    successful_tools = {
        event["tool_name"] for event in events if event["execution_success"] is True
    }
    nodes_complete = all(name in successful_tools for name in manifest["gold_tools"])
    checks = profile.get("dependency_edge_checks", [])
    known = [row for row in checks if isinstance(row.get("success"), bool)]
    edge_complete = len(known) == len(sidecar["dependency_edges"]) and all(row["success"] for row in known)
    internal = [
        row for row in profile.get("internal_parameter_flow_checks", [])
        if isinstance(row.get("success"), bool)
    ]
    internal_complete = len(internal) == len(sidecar["dependency_edges"]) and all(
        row["success"] for row in internal
    )
    valid = status is None
    result = {
        "schema_version": "graph_frontier_confirm_result_v1",
        "task_id": manifest["task_id"],
        "model": label,
        "split": manifest["split"],
        "runtime_seconds": time.monotonic() - started,
        "valid_capability_probe": valid,
        "system_status": status or "none",
        "profiler_error": profiler_error,
        "task_success": bool(valid and nodes_complete and edge_complete and state_ok is True),
        "reference_path_complete_success": bool(
            valid and nodes_complete and edge_complete and state_ok is True
        ),
        "final_state_success": state_ok,
        "path_adherence": profile.get("path_adherence", "unknown"),
        "gold_node_complete": nodes_complete,
        "edge_complete": edge_complete,
        "internal_param_complete": internal_complete,
        "tool_call_count": len(events),
        "unexpected_calls": len(profile.get("extra_tool_calls") or []),
        "redundant_calls": len(profile.get("redundant_tool_calls") or []),
        "final_assistant_content": final_content if final_content is not None else "unknown",
        "final_assistant_finish_reason": finish_reason,
        "profile": profile,
        "rollout_path": str(rollout_path.resolve().relative_to(ROOT)),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def select_shard_rows(
    rows: list[dict[str, Any]], shard_count: int, shard_index: int
) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("shard_count_must_be_positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index_out_of_range")
    return [row for index, row in enumerate(rows) if index % shard_count == shard_index]


async def run(
    manifest_path: Path,
    label: str,
    output: Path,
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    rows = load_jsonl(manifest_path)
    if len(rows) != COUNT:
        raise RuntimeError(f"manifest_count:{len(rows)}")
    selected_rows = select_shard_rows(rows, shard_count, shard_index)
    output.mkdir(parents=True, exist_ok=True)
    suffix = "" if shard_count == 1 else f".shard{shard_index}"
    (output / f"run_config{suffix}.json").write_text(
        json.dumps(
            {
                "model": label,
                "model_path": PATHS[label],
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_path(manifest_path),
                "inference": PROTOCOL,
                "shard_count": shard_count,
                "shard_index": shard_index,
                "selected_count": len(selected_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    answers = []
    started = time.monotonic()
    for index, row in enumerate(selected_rows, 1):
        try:
            result = await run_one(row, label, output)
        except Exception as exc:
            result = {
                "schema_version": "graph_frontier_confirm_result_v1",
                "task_id": row["task_id"],
                "model": label,
                "split": row["split"],
                "runtime_seconds": 0.0,
                "valid_capability_probe": False,
                "system_status": classify_exception(exc),
                "profiler_error": None,
                "fatal_exception": f"{type(exc).__name__}:{exc}",
            }
            task_path = output / "tasks" / f"{row['task_id']}.result.json"
            task_path.parent.mkdir(parents=True, exist_ok=True)
            task_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        answers.append(result)
        print(
            f"CONFIRM_PROGRESS model={label} shard={shard_index}/{shard_count} "
            f"completed={index}/{len(selected_rows)} "
            f"valid={sum(item.get('valid_capability_probe') is True for item in answers)} "
            f"reference={sum(item.get('reference_path_complete_success') is True for item in answers)}",
            flush=True,
        )
    summary = {
        "model": label,
        "completed": len(answers),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "valid": sum(item.get("valid_capability_probe") is True for item in answers),
        "system_error_counts": dict(Counter(
            item.get("system_status", "unknown")
            for item in answers
            if item.get("valid_capability_probe") is not True
        )),
        "reference_path_complete_success": sum(
            item.get("reference_path_complete_success") is True for item in answers
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    (output / f"run_summary{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "repro_1p7b/results/graph_frontier/confirm_300",
    )
    freeze_parser.add_argument(
        "--compact",
        type=Path,
        default=ROOT / f"repro_1p7b/graph_frontier/manifests/confirm_300_seed_{SEED}.jsonl",
    )
    freeze_parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "repro_1p7b/graph_frontier/reports/confirm_300_manifest_summary.json",
    )
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "repro_1p7b/results/graph_frontier/confirm_300",
    )
    validate_parser.add_argument(
        "--compact",
        type=Path,
        default=ROOT / f"repro_1p7b/graph_frontier/manifests/confirm_300_seed_{SEED}.jsonl",
    )
    validate_parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "repro_1p7b/graph_frontier/reports/confirm_300_manifest_summary.json",
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / f"repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_{SEED}.jsonl",
    )
    run_parser.add_argument("--model", choices=tuple(PATHS), required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--shard-count", type=int, default=1)
    run_parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    validate_models()
    if args.command == "validate":
        print(json.dumps(
            validate_frozen(args.root, args.compact, args.summary),
            ensure_ascii=False,
            indent=2,
        ))
        return
    pilot.register()
    try:
        answer = (
            freeze(args.root, args.compact, args.summary)
            if args.command == "freeze"
            else asyncio.run(
                run(
                    args.manifest,
                    args.model,
                    args.output_dir,
                    args.shard_count,
                    args.shard_index,
                )
            )
        )
        print(json.dumps(answer, ensure_ascii=False, indent=2))
    finally:
        MCPManager.shutdown()


if __name__ == "__main__":
    main()
