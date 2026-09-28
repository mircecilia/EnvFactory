"""Immutable two-channel training schedule for one recursive update."""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import OUT
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, PROTOCOL, check
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import LIMIT
from repro_1p7b.graph_frontier.recursive_opd_v3_complete_c.prepare import RUN as CPU_RUN
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

RUN = PROTOCOL.parent.parent / "recursive_opd_v3_experiment_run"
NATURAL_EXPOSURES = 128
LEARNING_RATE = 2e-6
SEED_BASE = 20260927
TAGS = {"CURRENT_FRONTIER_CORRECTION": "NATURAL_FRONTIER",
        "SUCCESS_TERMINAL_POSITIVE": "NATURAL_TERMINAL",
        "CURRENT_VERIFIED_CONTINUATION_POSITIVE": "NATURAL_CONTINUATION"}


def model_for(cycle: int) -> Path:
    if cycle == 0:
        return MODEL
    if cycle in range(1, 5):
        return RUN / f"cycle{cycle}/checkpoint"
    raise ValueError("only Cpi0 through Cpi4 exist")


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _natural(cycle: int) -> tuple[list[dict], dict, Path]:
    directory = (CPU_RUN / "cycle0/dataset" if cycle == 0 else
                 RUN / f"cycle{cycle}/natural")
    manifest_path, rows_path = directory / "manifest.json", directory / "dataset.jsonl"
    manifest, rows = json.loads(manifest_path.read_text()), _read_rows(rows_path)
    if (manifest.get("status") != "READY_CPU_DATA" or manifest.get("cycle") != cycle
            or manifest.get("dataset_sha256") != sha256(rows_path)
            or manifest.get("sample_count") != len(rows)
            or manifest.get("source_policy_sha256") != sha256(model_for(cycle) / "model.safetensors")
            or manifest.get("protocol_sha256") != sha256(PROTOCOL)
            or manifest.get("tokenizer_sha256") != sha256(MODEL / "tokenizer.json")
            or not manifest.get("gates") or any(v != "PASS" for v in manifest["gates"].values())):
        raise RuntimeError(f"C-D{cycle} natural provenance/hash/gate failed")
    rollout_manifest = (PROTOCOL.parent / "cycle0/rollout_manifest.json" if cycle == 0 else
                        RUN / f"cycle{cycle}/rollout_manifest.json")
    if manifest["source_rollout_manifest_sha256"] != sha256(rollout_manifest):
        raise RuntimeError(f"C-D{cycle} source rollout manifest drift")
    return rows, manifest, rows_path


def _anchors(cycle: int) -> tuple[list[dict], dict | None, Path | None]:
    if cycle == 0:
        return [], None, None
    directory = RUN / f"cycle{cycle}/anchors"
    manifest_path, rows_path = directory / "manifest.json", directory / "anchors.jsonl"
    manifest, rows = json.loads(manifest_path.read_text()), _read_rows(rows_path)
    if (manifest.get("status") != "ANCHOR_POOL_READY" or manifest.get("crossed_cycle") != cycle
            or manifest.get("source_error_cycle") != cycle - 1
            or manifest.get("anchor_age") != 1
            or manifest.get("anchor_dataset_sha256") != sha256(rows_path)
            or manifest.get("context_compatible") != len(rows)
            or manifest.get("current_policy_sha256") != sha256(model_for(cycle) / "model.safetensors")
            or not manifest.get("gates") or any(v != "PASS" for v in manifest["gates"].values())):
        raise RuntimeError(f"C-A{cycle} freshness/provenance/hash/gate failed")
    return rows, manifest, rows_path


def exposure_stream(natural: list[dict], anchors: list[dict], seed: int) -> tuple[list[dict], list[int], int]:
    """Interleave 128 unstratified natural rows and every anchor exactly once."""
    if len(natural) < NATURAL_EXPOSURES + 1:
        raise RuntimeError("insufficient natural population for potential DDP parity filler")
    natural_order = list(range(len(natural)))
    random.Random(seed).shuffle(natural_order)
    stream = [{"tag": TAGS[natural[index]["complete_c_source_label"]],
               "source": "natural", "index": index,
               "identity": natural[index]["complete_c_sample_identity"]}
              for index in natural_order[:NATURAL_EXPOSURES]]
    parity = (NATURAL_EXPOSURES + len(anchors)) % 2
    if parity:
        index = natural_order[NATURAL_EXPOSURES]
        stream.append({"tag": "DDP_PARITY_FILLER", "source": "natural", "index": index,
                       "identity": natural[index]["complete_c_sample_identity"]})
    stream += [{"tag": "FRESH_ANCHOR", "source": "anchor", "index": index,
                "identity": row["anchor_sample_identity"]}
               for index, row in enumerate(anchors)]
    random.Random(seed + 100000).shuffle(stream)
    if (len(stream) % 2 or len({x["identity"] for x in stream if x["source"] == "anchor"})
            != len(anchors) or collections.Counter(x["tag"] for x in stream)["FRESH_ANCHOR"]
            != len(anchors)):
        raise RuntimeError("exposure parity/anchor-once gate failed")
    return stream, natural_order, parity


def build_plan(cycle: int) -> tuple[list[dict], dict]:
    if cycle not in range(4):
        raise ValueError("only Cpi0->Cpi4 updates are authorized")
    protocol = check()
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    if len(train) != 1726 or len(heldout) != 432 or train & heldout:
        raise RuntimeError("frozen split drift")
    entries = {row["audit_id"]: row for row in json.loads(
        (OUT / "opd_source_manifest.json").read_text())["rows"]}
    heldout_queries = {entries[task]["query_hash"] for task in heldout}
    model = model_for(cycle)
    model_sha = sha256(model / "model.safetensors")
    if cycle == 0 and model_sha != protocol["pi0_model_sha256"]:
        raise RuntimeError("original Dynamic-v1 model hash drift")
    if cycle:
        state = json.loads((model / "trainer_state.json").read_text())
        if (state.get("lineage") != "recursive_opd_v3_complete_c_anchor"
                or state.get("status") != "TRAINED" or state.get("cycle") != cycle - 1
                or state.get("model_sha256") != model_sha):
            raise RuntimeError("previous Complete-C checkpoint lineage drift")
    natural, n_manifest, n_path = _natural(cycle)
    anchors, a_manifest, a_path = _anchors(cycle)
    if len(natural) < 129 or n_manifest["heldout_training_rows"] != 0:
        raise RuntimeError("natural population or heldout gate failed")
    identities, per_task = set(), collections.Counter()
    for row in natural:
        key = row.get("complete_c_sample_identity")
        if (not key or key in identities or row.get("task_id") not in train
                or row.get("task_id") in heldout or row.get("pool") != "train"
                or row.get("query_hash") != entries[row["task_id"]]["query_hash"]
                or row.get("query_hash") in heldout_queries
                or row.get("complete_c_source_label") not in TAGS
                or len(row.get("input_ids", [])) > LIMIT):
            raise RuntimeError("natural row identity/split/context failed")
        validate_mask(row)
        identities.add(key)
        per_task[row["task_id"]] += 1
    if any(count > 2 for count in per_task.values()):
        raise RuntimeError("natural rows exceed two per task")
    anchor_ids = set()
    for row in anchors:
        key = row.get("anchor_sample_identity")
        if (not key or key in anchor_ids or row.get("task_id") not in train
                or row.get("pool") != "train" or row.get("anchor_age") != 1
                or row.get("query_hash") != entries[row["task_id"]]["query_hash"]
                or row.get("query_hash") in heldout_queries
                or row.get("source_error_cycle") != cycle - 1
                or row.get("crossed_cycle") != cycle
                or row.get("current_policy_hash") != model_sha
                or len(row.get("input_ids", [])) > LIMIT):
            raise RuntimeError("fresh anchor row provenance/split/context failed")
        validate_mask(row)
        anchor_ids.add(key)
        per_task[row["task_id"]] += 1
    if any(count > 3 for count in per_task.values()):
        raise RuntimeError("semantic rows exceed three per task")
    seed = SEED_BASE + cycle
    stream, natural_order, parity = exposure_stream(natural, anchors, seed)
    category_counts = collections.Counter(x["tag"] for x in stream)
    population_counts = collections.Counter(r["complete_c_source_label"] for r in natural)
    main_sampled = collections.Counter(
        natural[index]["complete_c_source_label"] for index in natural_order[:NATURAL_EXPOSURES])
    selected_rows = [natural[x["index"]] if x["source"] == "natural" else anchors[x["index"]]
                     for x in stream]
    plan = {
        "schema_version": "recursive_opd_v3_complete_c_anchor_schedule_1",
        "cycle": cycle, "input_model_sha256": model_sha,
        "input_tokenizer_sha256": sha256(MODEL / "tokenizer.json"),
        "source_rollout_manifest_sha256": n_manifest["source_rollout_manifest_sha256"],
        "natural_dataset_sha256": sha256(n_path),
        "anchor_dataset_sha256": sha256(a_path) if a_path else None,
        "protocol_sha256": sha256(PROTOCOL), "seed": seed,
        "natural_population": dict(population_counts),
        "sampled_natural_main_composition": dict(main_sampled),
        "sampled_composition": dict(category_counts),
        "natural_main_exposures": NATURAL_EXPOSURES,
        "natural_exposures_including_filler": NATURAL_EXPOSURES + parity,
        "fresh_anchors_available": len(anchors), "fresh_anchors_exposed": len(anchors),
        "parity_filler": parity, "total_exposures": len(stream),
        "optimizer_steps": len(stream) // 2, "learning_rate": LEARNING_RATE,
        "sampled_tool_targets": sum(r["target_type"] == "first_divergence_tool"
                                    for r in selected_rows),
        "sampled_eos_targets": sum(r["target_type"] == "terminal_stop"
                                   for r in selected_rows),
        "sampled_natural_main_tool_targets": sum(
            natural[index]["target_type"] == "first_divergence_tool"
            for index in natural_order[:NATURAL_EXPOSURES]),
        "sampled_natural_main_eos_targets": sum(
            natural[index]["target_type"] == "terminal_stop"
            for index in natural_order[:NATURAL_EXPOSURES]),
        "stream": stream,
    }
    return selected_rows, plan


def freeze_plan(cycle: int) -> dict:
    _, plan = build_plan(cycle)
    path = RUN / f"cycle{cycle}/train_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise RuntimeError("immutable schedule drift")
    else:
        path.write_text(stable(plan) + "\n")
    return plan
