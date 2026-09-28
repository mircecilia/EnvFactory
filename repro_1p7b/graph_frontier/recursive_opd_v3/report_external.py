"""Create the immutable matched π0/π3/π4 external-evaluation report."""
from __future__ import annotations

import json
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import RUN, REPORT


def aggregate_frozen_tasks(directory: Path) -> dict:
    rows = [json.loads(path.read_text()) for path in sorted(directory.glob("*.result.json"))]
    if len(rows) != 300 or any(row.get("valid_capability_probe") is not True for row in rows):
        raise RuntimeError(f"expected 300 valid Frozen results: {directory}")
    expected_edges = reached_edges = propagated_edges = 0
    for row in rows:
        checks = (row.get("profile") or {}).get("dependency_edge_checks") or []
        expected_edges += len(checks)
        reached = [check for check in checks if isinstance(check.get("consumer_step"), int)]
        reached_edges += len(reached)
        propagated_edges += sum(check.get("success") is True for check in reached)
    return {
        "task_success": sum(row["task_success"] is True for row in rows),
        "reference_path_complete_success": sum(
            row["reference_path_complete_success"] is True for row in rows),
        "final_state_success": sum(row["final_state_success"] is True for row in rows),
        "edge_complete": sum(row["edge_complete"] is True for row in rows),
        "gold_node_complete": sum(row["gold_node_complete"] is True for row in rows),
        "internal_param_complete": sum(row["internal_param_complete"] is True for row in rows),
        "tool_call_count": sum(row["tool_call_count"] for row in rows),
        "unexpected_calls": sum(row["unexpected_calls"] for row in rows),
        "redundant_calls": sum(row["redundant_calls"] for row in rows),
        "consumer_reached_edges": reached_edges, "expected_dependency_edges": expected_edges,
        "consumer_reach_rate": reached_edges / expected_edges if expected_edges else None,
        "conditional_propagation_success_edges": propagated_edges,
        "conditional_propagation_rate": propagated_edges / reached_edges if reached_edges else None,
        "consumer_reach_definition": "dependency edge has integer consumer_step",
        "conditional_propagation_definition": "successful edge among consumer-reached edges",
    }


def build() -> dict:
    pi3, pi4 = RUN / "evaluation/pi3", RUN / "evaluation/pi4"
    if (pi3 / "status").read_text().strip() != "AUDITED" or (
            pi4 / "status").read_text().strip() != "AUDITED":
        raise RuntimeError("π3 and π4 external evaluations must both be AUDITED")
    frozen3 = json.loads((pi3 / "frozen300/comparison/summary.json").read_text())
    frozen4 = json.loads((pi4 / "frozen300/comparison/summary.json").read_text())
    rich3 = json.loads((pi3 / "rich24/comparison/metrics.json").read_text())
    rich4 = json.loads((pi4 / "rich24/comparison/metrics.json").read_text())
    validation3 = json.loads((RUN / "cycle3/checkpoint/checkpoint_validation.json").read_text())
    validation4 = json.loads((RUN / "cycle4/checkpoint/checkpoint_validation.json").read_text())
    if (frozen3["valid_probe_count"] != 300 or frozen4["valid_probe_count"] != 300
            or frozen3["manifest_sha256"] != frozen4["manifest_sha256"]
            or frozen3["base"] != frozen4["base"]
            or rich3["task_count"] != 24 or rich4["task_count"] != 24
            or rich3["plan_sha256"] != rich4["plan_sha256"]
            or rich3["base"] != rich4["base"]
            or rich3["trained_model_weights_sha256"] != validation3["model_sha256"]
            or rich4["trained_model_weights_sha256"] != validation4["model_sha256"]):
        raise RuntimeError("π3/π4 matched external identity gate failed")
    base_results = RUN.parents[2] / "repro_1p7b/results/graph_frontier/confirm_300/dynamic_v1/tasks"
    detailed = {
        "pi0": aggregate_frozen_tasks(base_results),
        "v2_pi3": aggregate_frozen_tasks(pi3 / "frozen300/tasks"),
        "v2_pi4": aggregate_frozen_tasks(pi4 / "frozen300/tasks"),
    }
    core = {
        "pi0": {"frozen300": detailed["pi0"],
                "rich24": {"official_reward": rich3["base_mean_official_reward"],
                            "semantic_success": rich3["base"]["semantic_success"],
                            "tool_events": rich3["base"]["tool_events"]}},
        "v2_pi3": {"frozen300": detailed["v2_pi3"],
                   "rich24": {"official_reward": rich3["trained_mean_official_reward"],
                               "semantic_success": rich3["trained"]["semantic_success"],
                               "tool_events": rich3["trained"]["tool_events"],
                               "paired_reward_delta": rich3["paired_official_reward_delta"],
                               "paired_reward_delta_ci95": rich3[
                                   "paired_official_reward_delta_ci95"]}},
        "v2_pi4": {"frozen300": detailed["v2_pi4"],
                   "rich24": {"official_reward": rich4["trained_mean_official_reward"],
                               "semantic_success": rich4["trained"]["semantic_success"],
                               "tool_events": rich4["trained"]["tool_events"],
                               "paired_reward_delta": rich4["paired_official_reward_delta"],
                               "paired_reward_delta_ci95": rich4[
                                   "paired_official_reward_delta_ci95"]}},
    }
    report = {
        "status": "COMPLETE", "schema_version": "recursive_opd_v2_pi3_external_eval_1",
        "v2_pi3_checkpoint_sha256": validation3["model_sha256"],
        "v2_pi3_tokenizer_sha256": validation3["tokenizer_sha256"],
        "v2_pi3_train_plan_sha256": validation3["train_plan_sha256"],
        "frozen300_manifest_sha256": frozen3["manifest_sha256"],
        "rich24_plan_sha256": rich3["plan_sha256"], "core": core,
        "pi3_frozen_paired_vs_pi0": frozen3["paired"],
        "pi4_frozen_paired_vs_pi0": frozen4["paired"],
        "structural_continuation": {
            "pi3_minus_pi4": {
                key: detailed["v2_pi3"][key] - detailed["v2_pi4"][key]
                for key in ("task_success", "final_state_success", "edge_complete",
                            "gold_node_complete", "internal_param_complete", "tool_call_count")},
            "conclusion": "pi3 preserves dependency and structural continuation far better than pi4",
            "rich24_caveat": "pi3 over-continues on Rich24 and has lower official reward than pi0/pi4",
        },
        "source_hashes": {
            "pi3_frozen_summary": sha256(pi3 / "frozen300/comparison/summary.json"),
            "pi3_rich_metrics": sha256(pi3 / "rich24/comparison/metrics.json"),
            "pi4_frozen_summary": sha256(pi4 / "frozen300/comparison/summary.json"),
            "pi4_rich_metrics": sha256(pi4 / "rich24/comparison/metrics.json"),
        },
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    json_path = REPORT / "recursive_opd_v2_pi3_external_eval.json"
    md_path = REPORT / "recursive_opd_v2_pi3_external_eval.md"
    if json_path.exists() or md_path.exists():
        raise RuntimeError("refusing to overwrite π3 external evaluation report")
    json_path.write_text(stable(report) + "\n")
    labels = (("π0", "pi0"), ("V2 π3", "v2_pi3"), ("V2 π4", "v2_pi4"))
    lines = ["V2 π3 EXTERNAL EVAL = COMPLETE", "", "# Matched external evaluation", "",
             "| metric | π0 | V2 π3 | V2 π4 |", "|---|---:|---:|---:|"]
    metrics = [
        ("Frozen task/reference success", "task_success"),
        ("Frozen final-state success", "final_state_success"),
        ("Frozen dependency edge complete", "edge_complete"),
        ("Frozen gold node complete", "gold_node_complete"),
        ("Frozen internal E2E", "internal_param_complete"),
        ("Frozen tool calls", "tool_call_count"),
        ("Frozen unexpected calls", "unexpected_calls"),
        ("Frozen redundant calls", "redundant_calls"),
    ]
    for label, key in metrics:
        lines.append("| " + label + " | " + " | ".join(
            str(core[name]["frozen300"][key]) for _, name in labels) + " |")
    for label, key in (("Rich24 official reward", "official_reward"),
                       ("Rich24 semantic success", "semantic_success"),
                       ("Rich24 tool events", "tool_events")):
        lines.append("| " + label + " | " + " | ".join(
            str(core[name]["rich24"][key]) for _, name in labels) + " |")
    lines += ["", "Consumer reach is the number of expected dependency edges whose consumer was "
              "actually reached; conditional propagation is edge success conditioned on that reach.", "",
              f"π3 consumer reach: {detailed['v2_pi3']['consumer_reached_edges']}/"
              f"{detailed['v2_pi3']['expected_dependency_edges']}; conditional propagation: "
              f"{detailed['v2_pi3']['conditional_propagation_success_edges']}/"
              f"{detailed['v2_pi3']['consumer_reached_edges']}.", "",
              "π3 clearly retains structural continuation better than π4 on Frozen300 "
              f"(edge {detailed['v2_pi3']['edge_complete']} vs {detailed['v2_pi4']['edge_complete']}; "
              f"gold node {detailed['v2_pi3']['gold_node_complete']} vs "
              f"{detailed['v2_pi4']['gold_node_complete']}).  The tradeoff remains visible on Rich24: "
              f"π3 makes {core['v2_pi3']['rich24']['tool_events']} tool events and scores "
              f"{core['v2_pi3']['rich24']['official_reward']}, versus π4's "
              f"{core['v2_pi4']['rich24']['tool_events']} and "
              f"{core['v2_pi4']['rich24']['official_reward']}.", ""]
    md_path.write_text("\n".join(lines))
    return report


if __name__ == "__main__":
    print(stable(build()), flush=True)
