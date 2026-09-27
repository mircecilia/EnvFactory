"""Independent CPU audit of one complete fresh V2 rollout and its Dk data."""
from __future__ import annotations

import collections
import json

from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for, valid_gold
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, OUT
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, RUN as V1_RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import GOLD_RUN
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import (
    LIMIT, account_context, context_class, sample_lengths,
)
from repro_1p7b.graph_frontier.recursive_opd_v2.future import checked_manifest
from repro_1p7b.graph_frontier.recursive_opd_v2.gpu_protocol import checked_base_sha
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import (
    RUN, checked_prefix, digest, encode_row,
)
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask


def audit(cycle: int) -> dict:
    from transformers import AutoTokenizer

    if cycle not in (1, 2):
        raise ValueError("only V2 fresh cycles 1 and 2 are in scope")
    protocol = check()
    rollout_manifest, rollout_hash = checked_manifest(cycle)
    base_sha = checked_base_sha()
    if rollout_manifest["gpu_experiment_base_sha"] != base_sha:
        raise RuntimeError("GPU experiment base SHA drift")
    base = RUN / f"cycle{cycle}"
    metrics_path = base / "metrics/per_task.jsonl"
    summary = json.loads((base / "metrics/summary.json").read_text())
    dataset_path = base / "dataset/dataset.jsonl"
    manifest = json.loads((base / "dataset/manifest.json").read_text())
    if (summary.get("per_task_sha256") != sha256(metrics_path)
            or manifest.get("metrics_sha256") != sha256(metrics_path)
            or manifest.get("dataset_sha256") != sha256(dataset_path)
            or manifest.get("status") != "READY_FOR_GPU_TRAINING"
            or manifest.get("source_model_sha256") != rollout_manifest["model_sha256"]
            or manifest.get("source_rollout_manifest_sha256") != rollout_hash
            or manifest.get("gpu_experiment_base_sha") != base_sha):
        raise RuntimeError("fresh-cycle manifest/hash/data gate failed")
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line]
    expected = protocol["train_task_ids"] + protocol["heldout_task_ids"]
    train_tasks = set(protocol["train_task_ids"])
    if (len(records) != 2158 or {record["task_id"] for record in records} != set(expected)
            or len({record["task_id"] for record in records}) != 2158
            or len({row["task_id"] for row in rows}) != len(rows)):
        raise RuntimeError("fresh-cycle complete task/unique gate failed")
    by_record = {record["task_id"]: record for record in records}
    by_row = {row["task_id"]: row for row in rows}
    overflow = {row["task_id"]: row for row in manifest["context_overflow_records"]}
    account_context(semantic_eligible=manifest["semantic_eligible_count"],
                    encoded=len(rows), overflow=list(overflow.values()))
    if (len(overflow) != manifest["context_overflow_count"]
            or manifest["trainable_count"] != len(rows)
            or manifest["encoded_rows"] != len(rows)
            or manifest["heldout_training_rows"] != 0
            or manifest["uncertain_training_rows"] != 0
            or manifest["skipped_uncertain_count"] != sum(
                record["task_id"] in train_tasks and record["status"] == "UNCERTAIN"
                for record in records)
            or any(value != "PASS" for value in manifest["gates"].values())):
        raise RuntimeError("fresh-cycle context/row gate failed")
    source = json.loads(SOURCE.read_text())
    entries = {entry["audit_id"]: entry for entry in
               json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.eos_token_id != 151645 or sha256(MODEL / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise RuntimeError("V2 tokenizer/EOS drift")
    verified_prefixes = full_reencoded = 0
    status_by_pool = collections.defaultdict(collections.Counter)
    regimes_by_pool = collections.defaultdict(collections.Counter)
    for position, task in enumerate(expected):
        record = by_record[task]
        episode_path = base / "rollouts" / f"{task}.json"
        episode = json.loads(episode_path.read_text())
        entry = entries[task]
        source_row = source[entry["source_row_index"]]
        gold_path = ((V1_RUN / "gold_cache" / f"{task}.json")
                     if episode.get("gold_source_kind") == "independent_replay_cache"
                     else (GOLD_RUN / "episodes" / f"{task}.json"))
        pool = "train" if task in train_tasks else "heldout"
        if (record.get("cycle") != cycle or record.get("pool") != pool
                or record.get("query_hash") != entry["query_hash"]
                or episode.get("cycle") != cycle or episode.get("pool") != pool
                or episode.get("task_id") != task
                or episode.get("model_sha256") != rollout_manifest["model_sha256"]
                or episode.get("manifest_sha256") != rollout_hash
                or episode.get("gpu_experiment_base_sha") != base_sha
                or episode.get("gold_source_episode_sha256") != sha256(gold_path)
                or not valid_gold(entry, source_row, episode)):
            raise RuntimeError(f"fresh-cycle episode/source identity failed: {task}")
        status_by_pool[pool][record["status"]] += 1
        if record["status"] == "FIRST_ERROR":
            regimes_by_pool[pool][record["error_regime"]] += 1
        if record["status"] == "UNCERTAIN":
            if task in by_row or task in overflow:
                raise RuntimeError("UNCERTAIN entered training dataset")
            continue
        schema = schema_for(RUN, episode)
        prefix = checked_prefix(record, episode)
        verified_prefixes += 1
        if pool == "heldout":
            if task in by_row or task in overflow:
                raise RuntimeError("heldout entered training dataset")
            continue
        prompt_len, full_len = sample_lengths(tokenizer, prefix, schema, record)
        if context_class(full_len, LIMIT) == "CONTEXT_OVERFLOW":
            item = overflow.get(task)
            if (item is None or task in by_row or item["full_tokens"] != full_len
                    or item["prompt_tokens_local"] != prompt_len
                    or item["reason"] != "CONTEXT_OVERFLOW"):
                raise RuntimeError("overflow record is not the unique exact-prefix length classification")
            continue
        row = by_row.get(task)
        if row is None or task in overflow:
            raise RuntimeError("context-compatible row missing from Dk")
        validate_mask(row)
        fresh = encode_row(tokenizer, record, episode, schema,
                           episode["gold_source_episode_sha256"], cycle=cycle)
        if (fresh["input_ids"] != row["input_ids"] or fresh["labels"] != row["labels"]
                or fresh["student_prefix_sha256"] != row["student_prefix_sha256"]
                or len(row["input_ids"]) != full_len
                or row.get("train_context_compatible") is not True):
            raise RuntimeError("full Dk token-level re-encoding mismatch")
        full_reencoded += 1
        if (position + 1) % 250 == 0:
            print(stable({"cycle": cycle, "audited": position + 1,
                          "total": len(expected)}), flush=True)
    if (full_reencoded != len(rows) or verified_prefixes != sum(
            record["status"] != "UNCERTAIN" for record in records)):
        raise RuntimeError("fresh-cycle full token/prefix audit incomplete")
    if {record["query_hash"] for record in records if record["pool"] == "train"} & {
            record["query_hash"] for record in records if record["pool"] == "heldout"}:
        raise RuntimeError("fresh-cycle train/heldout query-hash overlap")
    result = {"status": "AUDITED", "cycle": cycle, "task_count": 2158,
              "gpu_experiment_base_sha": base_sha,
              "model_sha256": rollout_manifest["model_sha256"],
              "rollout_manifest_sha256": rollout_hash,
              "metrics_sha256": sha256(metrics_path),
              "dataset_sha256": sha256(dataset_path),
              "semantic_eligible_count": manifest["semantic_eligible_count"],
              "trainable_count": len(rows), "context_overflow_count": len(overflow),
              "verified_prefixes": verified_prefixes,
              "full_token_reencoded_rows": full_reencoded,
              "status_by_pool": {pool: dict(count) for pool, count in status_by_pool.items()},
              "regime_by_pool": {pool: dict(count) for pool, count in regimes_by_pool.items()},
              "gates": {name: "PASS" for name in (
                  "all_2158_episodes_and_hashes", "frozen_policy_lineage",
                  "train_heldout_disjoint", "all_verified_student_prefixes",
                  "one_task_one_target", "zero_uncertain_training_rows",
                  "full_token_level_reencoding", "eos_only_and_assistant_mask",
                  "label_agnostic_context_classification", "explicit_overflow_accounting")}}
    path = base / "metrics/audit.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite fresh-cycle independent audit")
    path.write_text(stable(result) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    args = parser.parse_args()
    print(stable(audit(args.cycle)), flush=True)
