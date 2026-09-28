"""Fresh, resumable Cpi1..Cpi4 trajectories under the frozen V1 protocol."""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
from pathlib import Path

import requests

from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for, valid_gold
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import atomic_json, immutable_json
from repro_1p7b.graph_frontier.official_rl_audit_static import OUT, SOURCE
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import CONFIG, MODEL, PROTOCOL, check
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import collect_one
from repro_1p7b.graph_frontier.recursive_opd_v2.context_repair import (
    collect_with_context_boundary, context_rejection,
)
from repro_1p7b.graph_frontier.recursive_opd_v2.future import gold_for
from repro_1p7b.graph_frontier.recursive_opd_v3_experiment.schedule import RUN, model_for


def lock(cycle: int) -> dict:
    if cycle not in range(1, 5):
        raise ValueError("only Cpi1..Cpi4 full rollouts are authorized")
    protocol = check()
    model = model_for(cycle)
    state = json.loads((model / "trainer_state.json").read_text())
    model_hash = sha256(model / "model.safetensors")
    if (state.get("lineage") != "recursive_opd_v3_complete_c_anchor"
            or state.get("status") != "TRAINED" or state.get("cycle") != cycle - 1
            or state.get("model_sha256") != model_hash
            or state.get("tokenizer_sha256") != protocol["tokenizer_sha256"]
            or sha256(model / "tokenizer.json") != protocol["tokenizer_sha256"]):
        raise RuntimeError("checkpoint/tokenizer/policy-chain identity failed")
    plan_path = RUN / f"cycle{cycle - 1}/train_plan.json"
    plan = json.loads(plan_path.read_text())
    if (state["train_plan_sha256"] != sha256(plan_path)
            or plan["input_model_sha256"] != state["input_model_sha256"]
            or plan["natural_dataset_sha256"] != state["natural_dataset_sha256"]
            or plan["anchor_dataset_sha256"] != state["anchor_dataset_sha256"]):
        raise RuntimeError("previous exposure schedule/checkpoint drift")
    value = {"schema_version": "recursive_opd_v3_complete_c_anchor_rollout_1",
             "cycle": cycle, "lineage": "recursive_opd_v3_complete_c_anchor",
             "model_path": str(model.resolve()), "model_sha256": model_hash,
             "tokenizer_sha256": protocol["tokenizer_sha256"],
             "previous_train_plan_sha256": sha256(plan_path),
             "protocol_sha256": sha256(PROTOCOL), "config": CONFIG,
             "served_model": f"recursive-opd-complete-c-pi{cycle}",
             "task_count": 2158, "train_count": 1726, "heldout_count": 432}
    path = RUN / f"cycle{cycle}/rollout_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("immutable rollout manifest drift")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stable(value) + "\n")
    return value


def checked(cycle: int) -> tuple[dict, str]:
    value = lock(cycle)
    path = RUN / f"cycle{cycle}/rollout_manifest.json"
    if json.loads(path.read_text()) != value:
        raise RuntimeError("rollout manifest changed")
    return value, sha256(path)


def worker(cycle: int, shard: int, endpoint: str, *, limit: int | None = None) -> dict:
    if shard not in (0, 1):
        raise ValueError("two fixed rollout shards required")
    manifest, manifest_sha = checked(cycle)
    protocol = check()
    source = json.loads(SOURCE.read_text())
    entries = {row["audit_id"]: row for row in json.loads(
        (OUT / "opd_source_manifest.json").read_text())["rows"]}
    tasks = protocol["train_task_ids"] + protocol["heldout_task_ids"]
    train = set(protocol["train_task_ids"])
    selected = [task for position, task in enumerate(tasks) if position % 2 == shard]
    if limit is not None:
        selected = selected[:limit]
    counts = collections.Counter()
    for task in selected:
        path = RUN / f"cycle{cycle}/rollouts/{task}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if (old.get("task_id") != task or old.get("cycle") != cycle
                    or old.get("model_sha256") != manifest["model_sha256"]
                    or old.get("manifest_sha256") != manifest_sha
                    or old.get("student_error") or not old.get("student")):
                raise RuntimeError(f"persisted rollout identity/completion drift: {task}")
            counts["skipped"] += 1
            continue
        entry = entries[task]
        row = source[entry["source_row_index"]]
        gold, schema, gold_path, gold_kind = gold_for(task, entry, row)
        if not valid_gold(entry, row, {"gold": gold, "schema_sha256": immutable_json(RUN / "schemas", schema)}):
            raise RuntimeError(f"authoritative gold failed: {task}")
        schema_sha = immutable_json(RUN / "schemas", schema)
        try:
            student = asyncio.run(collect_one(entry, row, manifest["served_model"],
                                              endpoint, {"gold": gold}, schema))
        except requests.HTTPError as exc:
            if context_rejection(exc) is None:
                raise
            student = asyncio.run(collect_with_context_boundary(
                entry, row, manifest["served_model"], endpoint, schema))
            counts["context_boundary"] += 1
        if student.get("status") not in {"ROLLOUT_COMPLETE", "ROLLOUT_CONTEXT_OVERFLOW"}:
            raise RuntimeError(f"student rollout incomplete: {task}")
        episode = {"task_id": task, "source_row_index": entry["source_row_index"],
                   "pool": "train" if task in train else "heldout", "cycle": cycle,
                   "model_sha256": manifest["model_sha256"],
                   "manifest_sha256": manifest_sha,
                   "gold_source_episode_sha256": sha256(gold_path),
                   "gold_source_kind": gold_kind, "schema_sha256": schema_sha,
                   "gold": gold, "student": student, "student_error": None}
        atomic_json(path, episode)
        counts["completed"] += 1
        if (counts["completed"] + counts["skipped"]) % 50 == 0:
            print(stable({"cycle": cycle, "shard": shard, "counts": dict(counts)}), flush=True)
    return dict(counts)


def integrity(cycle: int) -> dict:
    manifest, manifest_sha = checked(cycle)
    protocol = check()
    source = json.loads(SOURCE.read_text())
    entries = {row["audit_id"]: row for row in json.loads(
        (OUT / "opd_source_manifest.json").read_text())["rows"]}
    tasks = protocol["train_task_ids"] + protocol["heldout_task_ids"]
    train = set(protocol["train_task_ids"])
    digests = []
    for task in tasks:
        path = RUN / f"cycle{cycle}/rollouts/{task}.json"
        episode = json.loads(path.read_text())
        entry = entries[task]
        row = source[entry["source_row_index"]]
        if (episode.get("task_id") != task or episode.get("cycle") != cycle
                or episode.get("pool") != ("train" if task in train else "heldout")
                or episode.get("model_sha256") != manifest["model_sha256"]
                or episode.get("manifest_sha256") != manifest_sha
                or episode.get("student_error") or not episode.get("student")
                or not valid_gold(entry, row, episode)):
            raise RuntimeError(f"rollout/gold integrity failed: {task}")
        schema_for(RUN, episode)
        digests.append((task, sha256(path)))
    value = {"status": "ROLLOUT_COMPLETE", "cycle": cycle,
             "task_count": len(tasks), "train_count": len(train), "heldout_count": 432,
             "model_sha256": manifest["model_sha256"],
             "rollout_manifest_sha256": manifest_sha,
             "rollouts_sha256": sha256_text(stable(digests))}
    path = RUN / f"cycle{cycle}/rollout_integrity.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("immutable rollout integrity drift")
    else:
        path.write_text(stable(value) + "\n")
    return value


def sha256_text(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    lock_parser = sub.add_parser("lock")
    lock_parser.add_argument("--cycle", type=int, required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--cycle", type=int, required=True)
    worker_parser.add_argument("--shard", type=int, required=True)
    worker_parser.add_argument("--endpoint", required=True)
    worker_parser.add_argument("--limit", type=int)
    check_parser = sub.add_parser("integrity")
    check_parser.add_argument("--cycle", type=int, required=True)
    args = parser.parse_args()
    result = (lock(args.cycle) if args.command == "lock" else
              worker(args.cycle, args.shard, args.endpoint, limit=args.limit)
              if args.command == "worker" else integrity(args.cycle))
    print(stable(result), flush=True)


if __name__ == "__main__":
    main()
