"""Task-paired internal π0/V1/V2 comparison through fresh V2 π4."""
from __future__ import annotations

import json

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import RUN as V1_RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v2.compare import (
    load_metrics, paired, pool_stats, retention,
)
from repro_1p7b.graph_frontier.recursive_opd_v2.gpu_protocol import checked_base_sha
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import RUN


def compare() -> dict:
    protocol = check()
    train, heldout = protocol["train_task_ids"], protocol["heldout_task_ids"]
    policies, hashes = {}, {}
    sources = [("pi0", V1_RUN, 0)]
    sources += [(f"v1_pi{cycle}", V1_RUN, cycle) for cycle in range(1, 4)]
    sources += [(f"v2_pi{cycle}", RUN, cycle) for cycle in range(1, 5)]
    for name, root, cycle in sources:
        if root == RUN:
            audit_path = RUN / f"cycle{cycle}/metrics/audit.json"
            audit = json.loads(audit_path.read_text())
            if (audit.get("status") != "AUDITED" or audit.get("cycle") != cycle
                    or audit.get("gpu_experiment_base_sha") != checked_base_sha()
                    or audit.get("metrics_sha256") != sha256(
                        RUN / f"cycle{cycle}/metrics/per_task.jsonl")):
                raise RuntimeError(f"V2 pi{cycle} independent audit missing or drifted")
        policies[name], hashes[name] = load_metrics(root, cycle)
        if set(policies[name]) != set(train + heldout):
            raise RuntimeError(f"{name} task identity drift")
    summary = {name: {"train": pool_stats(rows, train),
                      "heldout": pool_stats(rows, heldout)}
               for name, rows in policies.items()}
    transitions = {}
    for old, new in [("pi0", "v2_pi1"),
                     *((f"v2_pi{cycle}", f"v2_pi{cycle + 1}") for cycle in range(1, 4)),
                     ("pi0", "v1_pi1"), ("v1_pi1", "v1_pi2"),
                     ("v1_pi2", "v1_pi3")]:
        transitions[f"{old}->{new}"] = {
            "train": paired(policies[old], policies[new], train),
            "heldout": paired(policies[old], policies[new], heldout),
        }
    value = {
        "schema_version": "recursive_opd_v2_four_cycle_comparison_1",
        "gpu_experiment_base_sha": checked_base_sha(),
        "metrics_sha256": hashes,
        "policy_summary": summary,
        "paired_transitions": transitions,
        "pi0_heldout_success_retention": retention(
            policies["pi0"],
            {name: policies[name] for name in policies if name != "pi0"}, heldout),
    }
    output = RUN / "comparison/four_cycle.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text()) != value:
            raise RuntimeError("refusing to overwrite V2 four-cycle comparison")
    else:
        output.write_text(stable(value) + "\n")
    return value


if __name__ == "__main__":
    print(stable(compare()), flush=True)
