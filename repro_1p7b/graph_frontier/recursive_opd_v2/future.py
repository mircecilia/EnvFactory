"""Prepared V2-only future on-policy rollout/mining adapters; never run in CPU prep.

The collector reuses the V1 executable student collector and frozen gold source.
The miner reuses V1 classify/FD-v2, but requires a V2 checkpoint lineage lock.
No V1 pi1/pi2/pi3 rollout can satisfy these interfaces.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for, valid_gold
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import atomic_json, immutable_json
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, OUT, query_from_row
from repro_1p7b.graph_frontier.recursive_opd_v1.mine import classify, result
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import CONFIG, MODEL, PROTOCOL, RUN as V1_RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import GOLD_RUN, collect_one
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import (
    LIMIT, account_context, context_class, sample_lengths,
)
from repro_1p7b.graph_frontier.recursive_opd_v2.gpu_protocol import checked_base_sha
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import (
    RUN, checked_prefix, digest, encode_row, source_policy_gate,
)


def previous_v2(cycle: int, model: Path) -> dict:
    if cycle not in (1, 2, 3, 4):
        raise ValueError("future V2 policy cycle must be 1..4")
    checkpoint = RUN / f"cycle{cycle}/checkpoint"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if (model.resolve() != checkpoint.resolve()
            or state.get("lineage") != "recursive_opd_v2"
            or state.get("status") != "TRAINED"
            or state.get("cycle") != cycle - 1
            or state.get("model_sha256") != sha256(model / "model.safetensors")):
        raise RuntimeError("future rollout requires the prior V2-trained checkpoint")
    return state


def lock(cycle: int, model: Path) -> dict:
    """Immutable V2 policy identity; called before any future GPU collector."""
    protocol = check()
    prior = previous_v2(cycle, model)
    value = {"schema_version": "recursive_opd_v2_rollout_1", "cycle": cycle,
             "gpu_experiment_base_sha": checked_base_sha(),
             "model_path": str(model.resolve()), "model_sha256": prior["model_sha256"],
             "previous_v2_train_plan_sha256": prior["train_plan_sha256"],
             "protocol_sha256": sha256(PROTOCOL), "config": CONFIG,
             "served_model": f"recursive-opd-v2-pi{cycle}", "task_count": 2158,
             "train_count": 1726, "heldout_count": 432}
    source_policy_gate(cycle, value, protocol, prior)
    path = RUN / f"cycle{cycle}/rollout_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("immutable V2 rollout manifest drift")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stable(value) + "\n")
    return value


def checked_manifest(cycle: int) -> tuple[dict, str]:
    path = RUN / f"cycle{cycle}/rollout_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "recursive_opd_v2_rollout_1":
        raise RuntimeError("not a V2 rollout manifest")
    expected = lock(cycle, Path(manifest["model_path"]))
    if manifest != expected:
        raise RuntimeError("V2 rollout identity drift")
    return manifest, sha256(path)


def gold_for(task: str, entry: dict, source_row: dict) -> tuple[dict, list[dict], Path, str]:
    """Use immutable historical gold, or the V1 independently replayed cache."""
    historical = GOLD_RUN / "episodes" / f"{task}.json"
    gold_episode = json.loads(historical.read_text())
    if valid_gold(entry, source_row, gold_episode):
        return gold_episode["gold"], schema_for(GOLD_RUN, gold_episode), historical, "historical"
    cache = V1_RUN / "gold_cache" / f"{task}.json"
    restored = json.loads(cache.read_text())
    candidate = {**gold_episode, "gold": restored["gold"]}
    if (restored.get("task_id") != task or not valid_gold(entry, source_row, candidate)
            or not isinstance(restored.get("schemas"), list)):
        raise RuntimeError(f"independent gold cache invalid: {task}")
    return restored["gold"], restored["schemas"], cache, "independent_replay_cache"


def collect_worker(cycle: int, model: Path, endpoint: str, shard: int) -> dict:
    """Future GPU-backed collector; prepared only, not invoked in CPU phase."""
    if shard not in (0, 1):
        raise ValueError("exactly two V2 rollout shards expected")
    manifest, manifest_sha = checked_manifest(cycle)
    if (model.resolve() != Path(manifest["model_path"]).resolve()
            or sha256(model / "model.safetensors") != manifest["model_sha256"]):
        raise RuntimeError("collector model does not match V2 manifest")
    protocol = check()
    source = json.loads(SOURCE.read_text())
    entries = {row["audit_id"]: row for row in json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    tasks = protocol["train_task_ids"] + protocol["heldout_task_ids"]
    counts = collections.Counter()
    for position, task in enumerate(tasks):
        if position % 2 != shard:
            continue
        path = RUN / f"cycle{cycle}/rollouts" / f"{task}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if (old.get("task_id") != task or old.get("cycle") != cycle
                    or old.get("model_sha256") != manifest["model_sha256"]
                    or old.get("manifest_sha256") != manifest_sha
                    or old.get("gpu_experiment_base_sha") != manifest["gpu_experiment_base_sha"]):
                raise RuntimeError(f"persisted V2 rollout identity drift: {task}")
            counts["skipped"] += 1
            continue
        entry = entries[task]
        row = source[entry["source_row_index"]]
        gold, schema, gold_path, gold_kind = gold_for(task, entry, row)
        schema_sha = immutable_json(RUN / "schemas", schema)
        payload = {"task_id": task, "source_row_index": entry["source_row_index"],
                   "pool": "train" if task in set(protocol["train_task_ids"]) else "heldout",
                   "cycle": cycle, "model_sha256": manifest["model_sha256"],
                   "manifest_sha256": manifest_sha, "gold_source_episode_sha256": sha256(gold_path),
                   "gpu_experiment_base_sha": manifest["gpu_experiment_base_sha"],
                   "gold_source_kind": gold_kind, "schema_sha256": schema_sha,
                   "gold": gold, "student": None, "student_error": None}
        try:
            payload["student"] = asyncio.run(collect_one(entry, row, manifest["served_model"],
                                                           endpoint, {"gold": gold}, schema))
        except Exception as exc:
            payload["student_error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
            counts["failed"] += 1
        atomic_json(path, payload)
        counts["completed"] += 1
        if (counts["completed"] + counts["skipped"]) % 50 == 0:
            print(stable({"cycle": cycle, "shard": shard,
                          "completed": counts["completed"], "skipped": counts["skipped"],
                          "failed": counts["failed"]}), flush=True)
    return dict(counts)


def mine_build(cycle: int) -> dict:
    """After a fresh V2 rollout: classify by V1 verifier, then add success EOS.

    This is a future CPU command; V1 later-cycle episodes are never read here.
    """
    from transformers import AutoTokenizer

    if cycle not in (1, 2, 3, 4):
        raise ValueError("this experiment mines only V2 D1 through D4")

    manifest, manifest_sha = checked_manifest(cycle)
    repaired_path = RUN / f"cycle{cycle}/rollout_integrity_repaired.json"
    repaired = None
    if repaired_path.exists():
        from repro_1p7b.graph_frontier.recursive_opd_v2.context_repair import repaired_integrity
        repaired = repaired_integrity(cycle, write=False)
    protocol = check()
    source = json.loads(SOURCE.read_text())
    entries = {row["audit_id"]: row for row in json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.eos_token_id != 151645 or sha256(MODEL / "tokenizer.json") != protocol["tokenizer_sha256"]:
        raise RuntimeError("V2 tokenizer/EOS drift")
    task_ids = protocol["train_task_ids"] + protocol["heldout_task_ids"]
    if (RUN / f"cycle{cycle}/dataset").exists() or (RUN / f"cycle{cycle}/metrics").exists():
        raise RuntimeError("refusing to overwrite future V2 mining outputs")
    records, encoded, overflow, failures, gold_sources = [], [], [], [], {}
    train_ids = set(protocol["train_task_ids"])

    def classify_task(task: str) -> tuple[dict, list[dict], dict]:
        # Each verifier opens its own isolated MCP subprocesses. Only this
        # independent, I/O-bound phase runs concurrently; tokenization and
        # dataset writes remain in fixed manifest order on the main thread.
        episode = json.loads((RUN / f"cycle{cycle}/rollouts/{task}.json").read_text())
        entry = entries[task]
        row = source[entry["source_row_index"]]
        gold_path = ((V1_RUN / "gold_cache" / f"{task}.json")
                     if episode.get("gold_source_kind") == "independent_replay_cache"
                     else (GOLD_RUN / "episodes" / f"{task}.json"))
        if (episode.get("task_id") != task or episode.get("cycle") != cycle
                or episode.get("model_sha256") != manifest["model_sha256"]
                or episode.get("manifest_sha256") != manifest_sha
                or episode.get("gpu_experiment_base_sha") != manifest["gpu_experiment_base_sha"]
                or episode.get("gold_source_episode_sha256") != sha256(gold_path)
                or not valid_gold(entry, row, episode)):
            raise RuntimeError(f"V2 rollout/gold identity invalid: {task}")
        schema = schema_for(RUN, episode)
        if episode.get("student_error") or not episode.get("student"):
            verdict = result("UNCERTAIN", 0, len(episode["gold"]["actions"]),
                             reason="student runtime failure")
        else:
            try:
                verdict = asyncio.run(classify(episode, entry, row, schema))
            except Exception as exc:
                verdict = result("UNCERTAIN", 0, len(episode["gold"]["actions"]),
                                 reason=f"verifier exception:{type(exc).__name__}:{str(exc)[:200]}")
        verdict.update(task_id=task, cycle=cycle,
                       pool="train" if task in train_ids else "heldout",
                       query_hash=entry["query_hash"], environment=entry["environment"],
                       termination_reason=(episode.get("student") or {}).get("termination_reason"))
        return episode, schema, verdict

    def ordered_classifications():
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="v2-verifier") as pool:
            yield from pool.map(classify_task, task_ids)

    for task, (episode, schema, verdict) in zip(task_ids, ordered_classifications()):
        records.append(verdict)
        if verdict["pool"] == "train" and verdict["status"] != "UNCERTAIN":
            prefix = checked_prefix(verdict, episode)
            prompt_len, full_len = sample_lengths(tokenizer, prefix, schema, verdict)
            if context_class(full_len, LIMIT) == "CONTEXT_OVERFLOW":
                overflow.append({"task_id": task, "query_hash": verdict["query_hash"],
                                 "kind": verdict["error_regime"] if verdict["status"] == "FIRST_ERROR"
                                 else "SUCCESS_TERMINAL_POSITIVE",
                                 "semantic_eligible": True, "execution_verified": True,
                                 "train_context_compatible": False, "reason": "CONTEXT_OVERFLOW",
                                 "prompt_tokens_local": prompt_len, "full_tokens": full_len,
                                 "gold_depth": verdict["gold_tool_count"]})
                gold_sources[task] = episode["gold_source_episode_sha256"]
                continue
            try:
                item = encode_row(tokenizer, verdict, episode, schema,
                                  episode["gold_source_episode_sha256"], cycle=cycle)
                if item["total_tokens"] != full_len or item["total_tokens"] > LIMIT:
                    raise RuntimeError("fresh-cycle encoded length differs from exact-prefix audit")
                item.update(semantic_eligible=True, execution_verified=True,
                            train_context_compatible=True)
                encoded.append(item)
            except ValueError as exc:
                failures.append({"task_id": task, "reason": str(exc), "status": verdict["status"]})
            gold_sources[task] = episode["gold_source_episode_sha256"]
        if len(records) % 100 == 0:
            print(stable({"cycle": cycle, "mined": len(records), "total": len(task_ids)}), flush=True)
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    skipped_ids = set(repaired.get("skipped_student_error_task_ids", [])) if repaired else set()
    if (len(records) != 2158 or len({r["task_id"] for r in records}) != 2158
            or len({r["task_id"] for r in encoded}) != len(encoded)
            or len({r["task_id"] for r in overflow}) != len(overflow)
            or {r["task_id"] for r in encoded} & heldout
            or {r["task_id"] for r in overflow} & heldout
            or {r["query_hash"] for r in records if r["pool"] == "train"}
               & {r["query_hash"] for r in records if r["pool"] == "heldout"}):
        raise RuntimeError("future V2 task/split/unique/query gate failed")
    if (len(skipped_ids) != (repaired["student_error_count"] if repaired else 0)
            or any(r["status"] != "UNCERTAIN" for r in records if r["task_id"] in skipped_ids)
            or any(r["task_id"] in skipped_ids for r in encoded + overflow)):
        raise RuntimeError("isolated student-error skips entered V2 training data")
    metrics_dir = RUN / f"cycle{cycle}/metrics"
    dataset_dir = RUN / f"cycle{cycle}/dataset"
    metrics_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)
    metrics_path = metrics_dir / "per_task.jsonl"
    dataset_path = dataset_dir / "dataset.jsonl"
    metrics_path.write_text("".join(stable(r) + "\n" for r in records))
    dataset_path.write_text("".join(stable(r) + "\n" for r in encoded))
    kinds = collections.Counter(r["correction_kind"] for r in encoded)
    eligible = sum(r["pool"] == "train" and r["status"] != "UNCERTAIN" for r in records)
    accounting = len(encoded) + len(overflow) + len(failures) == eligible
    if not accounting:
        raise RuntimeError("fresh-cycle semantic eligibility accounting drift")
    ready = not failures
    if ready:
        account_context(semantic_eligible=eligible, encoded=len(encoded), overflow=overflow)
    value = {"schema_version": "recursive_opd_v2_context_accounted_1",
             "status": "READY_FOR_GPU_TRAINING" if ready else "NOT_READY",
             "usage": f"TRAINABLE_V2_D{cycle}", "cycle": cycle,
             "gpu_experiment_base_sha": manifest["gpu_experiment_base_sha"],
             "context_protocol": "only exact-prefix, context-compatible samples are trainable",
             "max_train_context": LIMIT,
             "collection_context_overflow_count":
                 repaired["student_context_overflow_count"] if repaired else 0,
             "collection_skipped_student_error_count": len(skipped_ids),
             "collection_skipped_student_error_task_ids": sorted(skipped_ids),
             "collection_integrity_sha256": sha256(repaired_path) if repaired else
                 sha256(RUN / f"cycle{cycle}/rollout_integrity.json"),
             "source_model_sha256": manifest["model_sha256"],
             "source_rollout_manifest_sha256": manifest_sha,
             "source_v1_metrics_sha256": None, "gold_source_sha256": digest(gold_sources),
             "task_split_sha256": digest({"train_task_ids": protocol["train_task_ids"],
                                           "heldout_task_ids": protocol["heldout_task_ids"]}),
             "tokenizer_sha256": sha256(MODEL / "tokenizer.json"),
             "protocol_sha256": sha256(PROTOCOL), "metrics_sha256": sha256(metrics_path),
             "dataset_sha256": sha256(dataset_path), "sample_count": len(encoded),
             "eligible_task_count": eligible, "semantic_eligible_count": eligible,
             "context_compatible_count": len(encoded), "trainable_count": len(encoded),
             "encoded_rows": len(encoded), "context_overflow_count": len(overflow),
             "context_overflow_records": overflow,
             "skipped_uncertain_count": sum(r["pool"] == "train" and r["status"] == "UNCERTAIN" for r in records),
             "heldout_training_rows": 0, "uncertain_training_rows": 0,
             "class_counts": dict(kinds),
             "tool_targets": kinds["START"] + kinds["ADVANCE"],
             "eos_targets": kinds["TERMINATE"] + kinds["SUCCESS_TERMINAL_POSITIVE"],
             "natural_eos_fraction": (kinds["TERMINATE"] + kinds["SUCCESS_TERMINAL_POSITIVE"])
                                     / len(encoded) if encoded else 0,
             "encoding_failures": failures,
             "gates": {"v2_checkpoint_lineage": "PASS", "rollout_gold_identity": "PASS",
                       "one_task_one_target": "PASS", "zero_query_hash_overlap": "PASS",
                       "heldout_zero_rows": "PASS", "no_uncertain_rows": "PASS",
                       "student_exact_predecision_prefix": "PASS", "gold_next_tool_pointer": "PASS",
                       "success_verified_terminal_and_stop": "PASS",
                       "terminate_verified_overcontinuation": "PASS",
                       "assistant_only_loss_mask": "PASS", "eos_only_single_label": "PASS",
                       "context_accounting": "PASS" if ready else "FAIL",
                       "all_context_compatible_rows_encoded": "PASS" if ready else "FAIL"}}
    (dataset_dir / "manifest.json").write_text(stable(value) + "\n")
    (metrics_dir / "summary.json").write_text(stable({"cycle": cycle, "per_task_sha256": sha256(metrics_path),
                                                       "usage": f"V2_PI{cycle}_FRESH"}) + "\n")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("lock", "worker", "mine-build"))
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.command == "lock":
        if args.model is None:
            parser.error("--model required")
        answer = lock(args.cycle, args.model)
    elif args.command == "worker":
        if args.model is None or args.endpoint is None or args.shard is None:
            parser.error("--model, --endpoint and --shard required")
        answer = collect_worker(args.cycle, args.model, args.endpoint, args.shard)
    else:
        answer = mine_build(args.cycle)
    print(stable(answer), flush=True)


if __name__ == "__main__":
    main()
