#!/usr/bin/env python3
"""Export paired transition artifacts and deterministic audit samples."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from paired_outcomes import (
    CATEGORY_NAMES,
    CATEGORIES,
    DEFAULT_SEED,
    REPORT_DIR,
    build_analysis,
    category_from_id,
)


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def transition_rows(ids: list[str], analysis: dict, fixed: bool) -> list[dict]:
    rows = []
    for task_id in ids:
        rows.append({
            "task_id": task_id,
            "category": category_from_id(task_id),
            "base_correct": not fixed,
            "sft_correct": fixed,
            "base_result_path": analysis["result_paths"]["base"][task_id],
            "sft_result_path": analysis["result_paths"]["sft"][task_id],
        })
    return rows


def paired_markdown(analysis: dict) -> str:
    lines = [
        "# BFCL V3 Multi-Turn Paired Analysis",
        "",
        "All comparisons pair the same task IDs. McNemar uses the exact two-sided",
        "binomial test over discordant pairs. Delta intervals use 10,000 paired",
        f"bootstrap resamples with seed {analysis['seed']}.",
        "",
    ]
    for category in ("overall",) + CATEGORIES:
        result = analysis["sections"][category]
        name = "Overall" if category == "overall" else f"{CATEGORY_NAMES[category]} category"
        transitions = result["transitions"]
        lo, hi = result["paired_bootstrap_95_ci"]
        base_lo, base_hi = result["base_wilson_95_ci"]
        sft_lo, sft_hi = result["sft_wilson_95_ci"]
        lines.extend([
            f"## {name}",
            "",
            f"- Base: {result['base_correct']} / {result['total']} = {percent(result['base_accuracy'])}",
            f"- Base Wilson 95% CI: [{percent(base_lo)}, {percent(base_hi)}]",
            f"- SFT: {result['sft_correct']} / {result['total']} = {percent(result['sft_accuracy'])}",
            f"- SFT Wilson 95% CI: [{percent(sft_lo)}, {percent(sft_hi)}]",
            f"- Delta: {result['net_cases']:+d} cases, {100 * result['delta']:+.2f} pp",
            f"- TT: {transitions['TT']}",
            f"- TF (regressed): {transitions['TF']}",
            f"- FT (fixed): {transitions['FT']}",
            f"- FF: {transitions['FF']}",
            f"- McNemar exact p: {result['mcnemar_exact_p']:.12g}",
            f"- Paired bootstrap 95% CI: [{100 * lo:+.3f}, {100 * hi:+.3f}] pp",
            "",
        ])
    lines.extend([
        "## Interpretation",
        "",
        "The overall point estimate is positive, but its paired 95% interval includes",
        "zero and the exact McNemar test is not significant at 0.05. The result is",
        "therefore evidence of an observed improvement under this run, not a",
        "statistically established population-level gain.",
        "The intervals resample task IDs only; they do not include between-run",
        "generation variance from the unlocked stochastic server seeds.",
        "",
    ])
    return "\n".join(lines)


def miss_func_markdown(analysis: dict) -> str:
    result = analysis["sections"]["miss_func"]
    fixed = [task_id for task_id in analysis["fixed_ids"] if category_from_id(task_id) == "miss_func"]
    regressed = [task_id for task_id in analysis["regressed_ids"] if category_from_id(task_id) == "miss_func"]
    return "\n".join([
        "# Miss Func Transitions",
        "",
        f"- Total: {result['total']}",
        f"- Base correct: {result['base_correct']}",
        f"- SFT correct: {result['sft_correct']}",
        f"- Base wrong to SFT correct (fixed): {len(fixed)}",
        f"- Base correct to SFT wrong (regressed): {len(regressed)}",
        f"- Net: {len(fixed) - len(regressed):+d}",
        f"- McNemar exact p: {result['mcnemar_exact_p']:.12g}",
        "",
        "## Fixed task IDs",
        "",
        *[f"- {task_id}" for task_id in fixed],
        "",
        "## Regressed task IDs",
        "",
        *[f"- {task_id}" for task_id in regressed],
        "",
    ])


def main() -> None:
    analysis = build_analysis()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fixed_rows = transition_rows(analysis["fixed_ids"], analysis, fixed=True)
    regressed_rows = transition_rows(analysis["regressed_ids"], analysis, fixed=False)
    write_jsonl(REPORT_DIR / "fixed_cases.jsonl", fixed_rows)
    write_jsonl(REPORT_DIR / "regressed_cases.jsonl", regressed_rows)
    (REPORT_DIR / "PAIRED_ANALYSIS.md").write_text(
        paired_markdown(analysis), encoding="utf-8"
    )
    (REPORT_DIR / "MISS_FUNC_TRANSITIONS.md").write_text(
        miss_func_markdown(analysis), encoding="utf-8"
    )

    fixed_distribution = Counter(row["category"] for row in fixed_rows)
    regressed_distribution = Counter(row["category"] for row in regressed_rows)
    ff = sorted(
        task_id for task_id in analysis["result_paths"]["base"]
        if task_id in analysis["failure_rows"]["base"]
        and task_id in analysis["failure_rows"]["sft"]
    )
    rng = random.Random(DEFAULT_SEED)
    samples = {
        "seed": DEFAULT_SEED,
        "FT": rng.sample(analysis["fixed_ids"], min(5, len(analysis["fixed_ids"]))),
        "TF": rng.sample(analysis["regressed_ids"], min(5, len(analysis["regressed_ids"]))),
        "FF": rng.sample(ff, min(5, len(ff))),
    }
    (REPORT_DIR / "qualitative_sample_ids.json").write_text(
        json.dumps(samples, indent=2) + "\n", encoding="utf-8"
    )
    print("fixed", dict(fixed_distribution))
    print("regressed", dict(regressed_distribution))
    print("samples", json.dumps(samples, sort_keys=True))


if __name__ == "__main__":
    main()
