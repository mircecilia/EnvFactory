"""Task-paired V2/Complete-C comparison and fresh-anchor retention audit."""
from __future__ import annotations

import collections
import json

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import RUN as V1_RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v2.compare import (
    load_metrics, outcome, paired, pool_stats, retention,
)
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import RUN as V2_RUN
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.schedule import CPU_RUN, RUN


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _anchor_trace(cycle: int, policies: dict) -> dict:
    path = RUN / f"cycle{cycle}/anchors/anchors.jsonl"
    manifest = json.loads((path.parent / "manifest.json").read_text())
    anchors = _rows(path)
    if manifest["anchor_dataset_sha256"] != sha256(path) or manifest["context_compatible"] != len(anchors):
        raise RuntimeError(f"A{cycle} source hash/count drift")
    if cycle == 4:
        return {"available": len(anchors), "next_policy": "NOT_RUN_CPI5",
                "retained": None, "regressed": None, "uncertain": None}
    counts = collections.Counter()
    next_policy = policies[f"Cpi{cycle + 1}"]
    for anchor in anchors:
        task = anchor["task_id"]
        row = next_policy[task]
        if (anchor["anchor_age"] != 1 or anchor["crossed_cycle"] != cycle
                or anchor["source_error_cycle"] != cycle - 1
                or row["gold_tool_count"] <= anchor["source_frontier_depth"]):
            raise RuntimeError(f"A{cycle} retention lineage drift: {task}")
        if row["status"] == "UNCERTAIN":
            counts["uncertain"] += 1
        elif row["frontier_depth"] > anchor["source_frontier_depth"]:
            counts["retained"] += 1
        else:
            counts["regressed"] += 1
    if sum(counts.values()) != len(anchors):
        raise RuntimeError("anchor retention accounting drift")
    return {"available": len(anchors), "next_policy": f"Cpi{cycle + 1}",
            "retained": counts["retained"], "regressed": counts["regressed"],
            "uncertain": counts["uncertain"]}


def build() -> dict:
    protocol = check()
    train, heldout = protocol["train_task_ids"], protocol["heldout_task_ids"]
    policies, hashes = {}, {}
    sources = [("pi0", V1_RUN, 0)]
    sources += [(f"V2pi{i}", V2_RUN, i) for i in range(1, 5)]
    sources += [(f"Cpi{i}", RUN, i) for i in range(1, 5)]
    for name, root, cycle in sources:
        rows, digest = load_metrics(root, cycle)
        if set(rows) != set(train + heldout):
            raise RuntimeError(f"{name} task identity drift")
        if name.startswith("Cpi"):
            audit = json.loads((root / f"cycle{cycle}/metrics/audit.json").read_text())
            if (audit.get("status") != "AUDITED" or audit.get("metrics_sha256") != digest
                    or audit.get("task_count") != 2158):
                raise RuntimeError(f"{name} fresh audit missing")
        policies[name], hashes[name] = rows, digest
    core = {name: {"train": pool_stats(rows, train), "heldout": pool_stats(rows, heldout)}
            for name, rows in policies.items()}
    transitions = {}
    for i in range(1, 5):
        before = policies["pi0" if i == 1 else f"Cpi{i-1}"]
        after = policies[f"Cpi{i}"]
        transitions[f"Cpi{i-1}->Cpi{i}"] = {
            "train": paired(before, after, train), "heldout": paired(before, after, heldout)}
    retained = retention(policies["pi0"],
                         {f"Cpi{i}": policies[f"Cpi{i}"] for i in range(1, 5)}, heldout)
    baseline_success = [task for task in heldout if policies["pi0"][task]["status"] == "SUCCESS"]
    for i in range(1, 5):
        breakdown = collections.Counter()
        for task in baseline_success:
            row = policies[f"Cpi{i}"][task]
            if row["status"] == "SUCCESS":
                continue
            if outcome(row) == "TERMINATE":
                breakdown["over_continuation"] += 1
            elif row["status"] == "FIRST_ERROR" and row.get("reason") == "premature model stop":
                breakdown["premature_stop"] += 1
            else:
                breakdown["other"] += 1
        if sum(breakdown.values()) != retained[f"Cpi{i}"]["regressed"]:
            raise RuntimeError("pi0-success retention accounting drift")
        retained[f"Cpi{i}"]["regression_breakdown"] = dict(breakdown)
    natural, anchors, plans, models = {}, {}, {}, {"pi0": protocol["pi0_model_sha256"]}
    for i in range(5):
        droot = CPU_RUN / "cycle0/dataset" if i == 0 else RUN / f"cycle{i}/natural"
        npath = droot / "dataset.jsonl"
        natural[f"D{i}"] = json.loads((droot / "manifest.json").read_text())
        if natural[f"D{i}"]["dataset_sha256"] != sha256(npath):
            raise RuntimeError(f"D{i} dataset drift")
        if i:
            aroot = RUN / f"cycle{i}/anchors"
            anchors[f"A{i}"] = {**json.loads((aroot / "manifest.json").read_text()),
                                "retention_trace": _anchor_trace(i, policies)}
            checkpoint = RUN / f"cycle{i}/checkpoint"
            state = json.loads((checkpoint / "trainer_state.json").read_text())
            if state["model_sha256"] != sha256(checkpoint / "model.safetensors"):
                raise RuntimeError(f"Cpi{i} checkpoint drift")
            models[f"Cpi{i}"] = state["model_sha256"]
        if i < 4:
            path = RUN / f"cycle{i}/train_plan.json"
            plans[f"cycle{i}"] = {**json.loads(path.read_text()), "sha256": sha256(path)}
    external = {}
    for i in (3, 4):
        cbase = RUN / f"evaluation/pi{i}"
        vbase = V2_RUN / f"evaluation/pi{i}"
        if (cbase / "status").read_text().strip() != "AUDITED":
            raise RuntimeError(f"Cpi{i} external audit missing")
        frozen_c = json.loads((cbase / "frozen300/comparison/summary.json").read_text())
        frozen_v = json.loads((vbase / "frozen300/comparison/summary.json").read_text())
        rich_c = json.loads((cbase / "rich24/comparison/metrics.json").read_text())
        rich_v = json.loads((vbase / "rich24/comparison/metrics.json").read_text())
        if (frozen_c["task_count"] != 300 or frozen_c["valid_probe_count"] != 300
                or frozen_c["manifest_sha256"] != frozen_v["manifest_sha256"]
                or frozen_c["base"] != frozen_v["base"] or rich_c["task_count"] != 24
                or rich_c["plan_sha256"] != rich_v["plan_sha256"]
                or rich_c["base_model_weights_sha256"] != rich_v["base_model_weights_sha256"]):
            raise RuntimeError("external V2/Cpi paired protocol or base drift")
        external[f"pi{i}"] = {
            "Frozen300": {"V2": frozen_v["trained"], "CompleteC": frozen_c["trained"],
                          "base": frozen_c["base"]},
            "Rich24": {"V2": rich_v["trained"], "CompleteC": rich_c["trained"],
                       "V2_reward": rich_v["trained_mean_official_reward"],
                       "CompleteC_reward": rich_c["trained_mean_official_reward"],
                       "base_reward": rich_c["base_mean_official_reward"]}}
    return {"schema_version": "recursive_opd_v3_complete_c_anchor_comparison_1",
            "status": "COMPLETE", "training_compute_strictly_matched_to_v2": False,
            "metrics_sha256": hashes, "policy_summary": core,
            "paired_transitions": transitions, "pi0_success_retention": retained,
            "natural": natural, "anchors": anchors, "train_plans": plans,
            "model_sha256": models, "external": external,
            "not_run": ["Cpi5", "GRPO", "Static retraining", "BFCL", "git push"]}


if __name__ == "__main__":
    value = build()
    path = RUN / "comparison.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("immutable final comparison drift")
    else:
        path.write_text(stable(value) + "\n")
    print(stable({"status": value["status"], "comparison_sha256": sha256(path)}), flush=True)
