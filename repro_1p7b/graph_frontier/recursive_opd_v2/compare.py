"""Task-paired internal comparison: pi0, V1 pi1/pi2, V2 pi1'/pi2'."""
from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import RUN as V1_RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v2.gpu_protocol import checked_base_sha
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import RUN


def outcome(row: dict) -> str:
    if row["status"] == "SUCCESS":
        return "SUCCESS"
    if row["status"] == "UNCERTAIN":
        return "UNCERTAIN"
    return row["error_regime"]


def reached_terminal(row: dict) -> bool:
    return (row["frontier_depth"] == row["gold_tool_count"]
            and outcome(row) in {"SUCCESS", "TERMINATE"})


def premature_stop(row: dict) -> bool:
    return row["status"] == "FIRST_ERROR" and row.get("reason") == "premature model stop"


def load_metrics(base: Path, cycle: int) -> tuple[dict[str, dict], str]:
    metrics = base / f"cycle{cycle}/metrics/per_task.jsonl"
    summary = json.loads((metrics.parent / "summary.json").read_text())
    if summary.get("per_task_sha256") != sha256(metrics):
        raise RuntimeError("policy metrics hash drift")
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line]
    if len(rows) != 2158 or len({row["task_id"] for row in rows}) != 2158:
        raise RuntimeError("policy metrics not a complete 2158-task reservoir")
    return {row["task_id"]: row for row in rows}, sha256(metrics)


def pool_stats(rows: dict[str, dict], tasks: list[str]) -> dict:
    group = [rows[task] for task in tasks]
    counts = collections.Counter(outcome(row) for row in group)
    return {"tasks": len(tasks), "success": counts["SUCCESS"],
            "first_action_correct": sum(row["frontier_depth"] >= 1 for row in group),
            "reached_verified_terminal": sum(reached_terminal(row) for row in group),
            "premature_stop": sum(premature_stop(row) for row in group),
            "over_continuation": counts["TERMINATE"],
            "mean_frontier_depth": round(statistics.mean(row["frontier_depth"] for row in group), 6),
            "mean_normalized_frontier": round(statistics.mean(
                row["frontier_depth"] / row["gold_tool_count"] for row in group), 8),
            "outcome_counts": dict(counts)}


def paired(old: dict[str, dict], new: dict[str, dict], tasks: list[str]) -> dict:
    movement = collections.Counter()
    migration = collections.defaultdict(collections.Counter)
    for task in tasks:
        before, after = old[task], new[task]
        if (before["gold_tool_count"] != after["gold_tool_count"]
                or before["pool"] != after["pool"]):
            raise RuntimeError(f"paired gold/task drift: {task}")
        delta = after["frontier_depth"] - before["frontier_depth"]
        movement["FORWARD" if delta > 0 else "BACKWARD" if delta < 0 else "SAME"] += 1
        migration[outcome(before)][outcome(after)] += 1
    return {"movement": {key: movement[key] for key in ("FORWARD", "SAME", "BACKWARD")},
            "migration": {key: dict(value) for key, value in migration.items()}}


def retention(pi0: dict[str, dict], policies: dict[str, dict[str, dict]], tasks: list[str]) -> dict:
    baseline_success = [task for task in tasks if pi0[task]["status"] == "SUCCESS"]
    if len(baseline_success) != 89:
        raise RuntimeError("pi0 frozen 89 heldout successes drifted")
    result = {}
    for name, records in policies.items():
        counts = collections.Counter(outcome(records[task]) for task in baseline_success)
        reasons = collections.Counter(records[task].get("reason") for task in baseline_success
                                      if records[task]["status"] != "SUCCESS")
        result[name] = {"baseline_tasks": 89, "still_success": counts["SUCCESS"],
                        "regressed": 89 - counts["SUCCESS"],
                        "regression_outcome": {k: v for k, v in counts.items() if k != "SUCCESS"},
                        "regression_reason": dict(reasons),
                        "over_continuation": counts["TERMINATE"]}
    return result


def compare() -> dict:
    protocol = check()
    policies = {}
    hashes = {}
    for name, base, cycle in (("pi0", V1_RUN, 0), ("v1_pi1", V1_RUN, 1),
                              ("v1_pi2", V1_RUN, 2), ("v2_pi1", RUN, 1),
                              ("v2_pi2", RUN, 2)):
        if base == RUN:
            audit = json.loads((RUN / f"cycle{cycle}/metrics/audit.json").read_text())
            if audit.get("status") != "AUDITED" or audit.get("gpu_experiment_base_sha") != checked_base_sha():
                raise RuntimeError(f"V2 pi{cycle} independent audit missing")
        policies[name], hashes[name] = load_metrics(base, cycle)
    train, heldout = protocol["train_task_ids"], protocol["heldout_task_ids"]
    for name, records in policies.items():
        if set(records) != set(train + heldout):
            raise RuntimeError(f"policy {name} task identity drift")
    summary = {name: {"train": pool_stats(rows, train), "heldout": pool_stats(rows, heldout)}
               for name, rows in policies.items()}
    transitions = {}
    for old, new in (("pi0", "v2_pi1"), ("v2_pi1", "v2_pi2"),
                     ("pi0", "v1_pi1"), ("v1_pi1", "v1_pi2")):
        transitions[f"{old}->{new}"] = {
            "train": paired(policies[old], policies[new], train),
            "heldout": paired(policies[old], policies[new], heldout)}
    value = {"schema_version": "recursive_opd_v2_two_cycle_comparison_1",
             "gpu_experiment_base_sha": checked_base_sha(),
             "metrics_sha256": hashes, "policy_summary": summary,
             "paired_transitions": transitions,
             "pi0_heldout_success_retention": retention(policies["pi0"], {
                 name: policies[name] for name in ("v1_pi1", "v1_pi2", "v2_pi1", "v2_pi2")}, heldout),
             "not_run": ["pi3_training", "Frozen300", "Rich24", "BFCL", "GRPO", "Static_retraining"]}
    output = RUN / "comparison/two_cycle.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite V2 two-cycle comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stable(value) + "\n")
    return value


if __name__ == "__main__":
    value = compare()
    print(stable(value), flush=True)
