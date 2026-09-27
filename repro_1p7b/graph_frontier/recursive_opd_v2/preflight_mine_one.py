"""One saved V2 episode through the unchanged executable verifier; no model call."""
from __future__ import annotations

import argparse
import asyncio
import json

from transformers import AutoTokenizer

from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, OUT
from repro_1p7b.graph_frontier.recursive_opd_v1.mine import classify
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import sample_lengths
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import RUN, checked_prefix, encode_row


def preflight(cycle: int, task_id: str) -> dict:
    if cycle not in (1, 2):
        raise ValueError("only fresh V2 cycles may be checked")
    episode = json.loads((RUN / f"cycle{cycle}/rollouts/{task_id}.json").read_text())
    source = json.loads(SOURCE.read_text())
    entries = {entry["audit_id"]: entry for entry in
               json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    entry = entries[task_id]
    schema = schema_for(RUN, episode)
    verdict = asyncio.run(classify(episode, entry, source[entry["source_row_index"]], schema))
    verdict.update(task_id=task_id, cycle=cycle, pool=episode["pool"],
                   query_hash=entry["query_hash"],
                   termination_reason=(episode.get("student") or {}).get("termination_reason"))
    result = {"task_id": task_id, "status": verdict["status"],
              "error_regime": verdict["error_regime"], "reason": verdict["reason"]}
    if verdict["status"] != "UNCERTAIN":
        prefix = checked_prefix(verdict, episode)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
        prompt, full = sample_lengths(tokenizer, prefix, schema, verdict)
        result.update(prompt_tokens=prompt, full_tokens=full)
        if full <= 16384:
            item = encode_row(tokenizer, verdict, episode, schema,
                              episode["gold_source_episode_sha256"], cycle=cycle)
            result.update(target_type=item["target_type"],
                          target_tokens=item["target_tokens"],
                          student_prefix_sha256=item["student_prefix_sha256"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    print(stable(preflight(args.cycle, args.task_id)), flush=True)
