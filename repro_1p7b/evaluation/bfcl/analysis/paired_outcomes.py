#!/usr/bin/env python3
"""Paired BFCL V3 outcome reconstruction using official per-task score files."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = ANALYSIS_DIR.parent
REPORT_DIR = ANALYSIS_DIR / "reports"
CATEGORIES = ("base", "miss_func", "miss_param", "long_context")
CATEGORY_NAMES = {
    "base": "Base",
    "miss_func": "Miss Func",
    "miss_param": "Miss Param",
    "long_context": "Long Context",
}
DEFAULT_SEED = 20260913
DEFAULT_BOOTSTRAPS = 10_000


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def category_from_id(task_id: str) -> str:
    for category in CATEGORIES:
        if task_id.startswith(f"multi_turn_{category}_"):
            return category
    raise ValueError(f"unexpected BFCL task id: {task_id}")


def result_files(label: str) -> list[Path]:
    return sorted((EVAL_DIR / "artifacts" / "full" / label / "result").glob("*/*_result.json"))


def score_files(label: str) -> list[Path]:
    return sorted((EVAL_DIR / "artifacts" / "full" / label / "score").glob("*/*_score.json"))


def load_result_ids(label: str):
    ids: list[str] = []
    path_by_id: dict[str, str] = {}
    for path in result_files(label):
        for row in load_jsonl(path):
            task_id = row["id"]
            ids.append(task_id)
            path_by_id[task_id] = str(path.relative_to(EVAL_DIR))
    counts = Counter(ids)
    duplicates = sorted(task_id for task_id, count in counts.items() if count != 1)
    return set(ids), duplicates, path_by_id


def load_score_failures(label: str):
    failure_rows: dict[str, list[dict]] = defaultdict(list)
    aggregates: dict[str, dict] = {}
    for path in score_files(label):
        rows = load_jsonl(path)
        aggregate = next(rows)
        category = path.name.removeprefix("BFCL_v3_multi_turn_").removesuffix("_score.json")
        aggregates[category] = aggregate
        for row in rows:
            failure_rows[row["id"]].append(row)
    return dict(failure_rows), aggregates


def transition_counts(ids: list[str], base_fail: set[str], sft_fail: set[str]) -> dict[str, int]:
    counts = Counter(
        ("T" if task_id not in base_fail else "F")
        + ("T" if task_id not in sft_fail else "F")
        for task_id in ids
    )
    return {key: counts[key] for key in ("TT", "TF", "FT", "FF")}


def mcnemar_exact(ft: int, tf: int) -> float:
    discordant = ft + tf
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(ft, tf) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap_ci(
    ids: list[str],
    base_fail: set[str],
    sft_fail: set[str],
    resamples: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    deltas = [
        int(task_id not in sft_fail) - int(task_id not in base_fail)
        for task_id in ids
    ]
    rng = random.Random(seed)
    samples = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(resamples)
    ]
    return percentile(samples, 0.025), percentile(samples, 0.975)


def wilson_ci(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return center - half, center + half


def build_analysis(resamples: int = DEFAULT_BOOTSTRAPS, seed: int = DEFAULT_SEED) -> dict:
    base_ids, base_duplicates, base_paths = load_result_ids("base")
    sft_ids, sft_duplicates, sft_paths = load_result_ids("sft")
    base_failure_rows, base_aggregates = load_score_failures("base")
    sft_failure_rows, sft_aggregates = load_score_failures("sft")
    base_fail = set(base_failure_rows)
    sft_fail = set(sft_failure_rows)
    shared = sorted(base_ids & sft_ids)

    sections: dict[str, dict] = {}
    for category in ("overall",) + CATEGORIES:
        ids = shared if category == "overall" else [
            task_id for task_id in shared if category_from_id(task_id) == category
        ]
        transitions = transition_counts(ids, base_fail, sft_fail)
        base_correct = sum(task_id not in base_fail for task_id in ids)
        sft_correct = sum(task_id not in sft_fail for task_id in ids)
        delta = (sft_correct - base_correct) / len(ids)
        sections[category] = {
            "total": len(ids),
            "base_correct": base_correct,
            "base_accuracy": base_correct / len(ids),
            "base_wilson_95_ci": wilson_ci(base_correct, len(ids)),
            "sft_correct": sft_correct,
            "sft_accuracy": sft_correct / len(ids),
            "sft_wilson_95_ci": wilson_ci(sft_correct, len(ids)),
            "delta": delta,
            "net_cases": sft_correct - base_correct,
            "transitions": transitions,
            "mcnemar_exact_p": mcnemar_exact(transitions["FT"], transitions["TF"]),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(
                ids, base_fail, sft_fail, resamples=resamples, seed=seed
            ),
        }

    aggregate_checks = {}
    for label, ids, failures, aggregates in (
        ("base", base_ids, base_fail, base_aggregates),
        ("sft", sft_ids, sft_fail, sft_aggregates),
    ):
        aggregate_checks[label] = {}
        for category in CATEGORIES:
            category_ids = {task_id for task_id in ids if category_from_id(task_id) == category}
            aggregate = aggregates[category]
            inferred_correct = len(category_ids - failures)
            aggregate_checks[label][category] = {
                "result_count": len(category_ids),
                "aggregate_total_count": aggregate["total_count"],
                "aggregate_correct_count": aggregate["correct_count"],
                "inferred_correct_count": inferred_correct,
                "match": (
                    len(category_ids) == aggregate["total_count"]
                    and inferred_correct == aggregate["correct_count"]
                ),
            }

    fixed = sorted(task_id for task_id in shared if task_id in base_fail and task_id not in sft_fail)
    regressed = sorted(task_id for task_id in shared if task_id not in base_fail and task_id in sft_fail)
    return {
        "seed": seed,
        "bootstrap_resamples": resamples,
        "completeness": {
            "base_result_count": len(base_ids),
            "sft_result_count": len(sft_ids),
            "shared_id_count": len(shared),
            "missing_from_base": sorted(sft_ids - base_ids),
            "missing_from_sft": sorted(base_ids - sft_ids),
            "base_duplicate_ids": base_duplicates,
            "sft_duplicate_ids": sft_duplicates,
            "invalid_category_ids": sorted(
                task_id for task_id in base_ids | sft_ids
                if not any(task_id.startswith(f"multi_turn_{category}_") for category in CATEGORIES)
            ),
            "aggregate_checks": aggregate_checks,
        },
        "sections": sections,
        "fixed_ids": fixed,
        "regressed_ids": regressed,
        "result_paths": {"base": base_paths, "sft": sft_paths},
        "failure_rows": {"base": base_failure_rows, "sft": sft_failure_rows},
    }


def serializable_summary(analysis: dict) -> dict:
    return {
        key: value for key, value in analysis.items()
        if key not in {"failure_rows", "result_paths"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "paired_outcomes.json",
    )
    args = parser.parse_args()
    analysis = build_analysis(args.bootstrap_resamples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(serializable_summary(analysis), indent=2) + "\n",
        encoding="utf-8",
    )
    overall = analysis["sections"]["overall"]
    print(
        f"Base={overall['base_correct']}/{overall['total']} "
        f"SFT={overall['sft_correct']}/{overall['total']} "
        f"TT/TF/FT/FF={overall['transitions']} "
        f"p={overall['mcnemar_exact_p']:.12g}"
    )


if __name__ == "__main__":
    main()
