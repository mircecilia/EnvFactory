#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "ea13468e4423454d0c213704fb87cf7cb3990433"
MODEL_ID = "Qwen/Qwen3-1.7B-FC"
CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["base", "sft"], required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    run_root = EVAL_DIR / "artifacts" / "smoke" / args.label
    expected_map = json.loads((EVAL_DIR / "configs" / "smoke_ids.json").read_text(encoding="utf-8"))
    expected_ids = {item for category in CATEGORIES for item in expected_map[category]}
    result_files = sorted((run_root / "result").rglob("BFCL_v3_multi_turn*_result.json"))
    score_files = sorted((run_root / "score").rglob("BFCL_v3_multi_turn*_score.json"))
    if len(result_files) != 4 or len(score_files) != 4:
        raise RuntimeError(f"expected four result and four score files; got {len(result_files)} and {len(score_files)}")

    observed_ids = set()
    inference_errors = []
    empty_outputs = []
    result_counts = {}
    for path in result_files:
        rows = load_jsonl(path)
        category = path.name.removeprefix("BFCL_v3_").removesuffix("_result.json")
        result_counts[category] = len(rows)
        for row in rows:
            observed_ids.add(row["id"])
            value = row.get("result")
            if isinstance(value, str) and value.startswith("Error during inference:"):
                inference_errors.append(row["id"])
            if value in ("", [], None):
                empty_outputs.append(row["id"])

    if observed_ids != expected_ids:
        raise RuntimeError(f"smoke ID mismatch: missing={sorted(expected_ids-observed_ids)} extra={sorted(observed_ids-expected_ids)}")

    category_scores = {}
    invalid_details = []
    for path in score_files:
        rows = load_jsonl(path)
        metadata = rows[0]
        category = path.name.removeprefix("BFCL_v3_").removesuffix("_score.json")
        if metadata["total_count"] != len(expected_map[category]):
            raise RuntimeError(f"{category}: total_count={metadata['total_count']} expected={len(expected_map[category])}")
        category_scores[category] = {
            "accuracy": metadata["accuracy"],
            "correct_count": metadata["correct_count"],
            "total_count": metadata["total_count"],
        }
        for row in rows[1:]:
            invalid_details.append({"id": row.get("id"), "error": row.get("error")})

    server = json.loads((run_root / "server_smoke.json").read_text(encoding="utf-8"))
    if not server.get("ordinary_generation_nonempty") or not server.get("tool_calls_parsed") or not server.get("tool_response_refill_present"):
        raise RuntimeError("server/tool-call/refill smoke evidence is incomplete")

    mean_accuracy = sum(item["accuracy"] for item in category_scores.values()) / len(CATEGORIES)
    summary = {
        "bfcl_commit": EXPECTED_COMMIT,
        "benchmark_version": "BFCL V3",
        "scope": "pipeline smoke only",
        "label": args.label,
        "model_id": MODEL_ID,
        "model_path": args.model_path,
        "backend": "sglang",
        "temperature": 0.7,
        "run_ids": expected_map,
        "result_counts": result_counts,
        "observed_count": len(observed_ids),
        "category_scores": category_scores,
        "multi_turn_unweighted_accuracy": mean_accuracy,
        "ordinary_generation": "PASS",
        "tool_call_format_and_parse": "PASS",
        "multi_turn_tool_response_refill": "PASS",
        "inference_error_ids": inference_errors,
        "empty_output_ids": empty_outputs,
        "official_score_failure_details": invalid_details,
        "raw_result_root": str(run_root / "result"),
        "raw_score_root": str(run_root / "score"),
    }
    summaries = EVAL_DIR / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    output = summaries / f"smoke_{args.label}.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    comparison_path = summaries / "smoke_comparison.json"
    other = summaries / f"smoke_{'sft' if args.label == 'base' else 'base'}.json"
    if other.exists():
        other_data = json.loads(other.read_text(encoding="utf-8"))
        by_label = {summary["label"]: summary, other_data["label"]: other_data}
        comparison = {
            "bfcl_commit": EXPECTED_COMMIT,
            "benchmark_version": "BFCL V3",
            "scope": "pipeline smoke only; do not infer model quality",
            "run_ids": expected_map,
            "settings": {"model_id": MODEL_ID, "backend": "sglang", "temperature": 0.7},
            "base": {"model_path": by_label["base"]["model_path"], "accuracy": by_label["base"]["multi_turn_unweighted_accuracy"]},
            "sft": {"model_path": by_label["sft"]["model_path"], "accuracy": by_label["sft"]["multi_turn_unweighted_accuracy"]},
        }
        comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    experiments = EVAL_DIR / "experiments.csv"
    rows = list(csv.DictReader(experiments.open(encoding="utf-8", newline="")))
    run_id = f"bfcl_v3_smoke_{args.label}_seed20260913"
    rows = [row for row in rows if row["run_id"] != run_id]
    rows.append({
        "run_id": run_id,
        "model": MODEL_ID,
        "checkpoint": args.model_path,
        "benchmark": "BFCL V3",
        "benchmark_commit": EXPECTED_COMMIT,
        "category": "multi_turn (8-case stratified smoke)",
        "backend": "sglang-0.5.9",
        "temperature": "0.7",
        "seed": "20260913",
        "result_path": str(run_root / "result"),
        "score_path": str(run_root / "score"),
        "metric": str(mean_accuracy),
        "status": "pipeline_smoke_pass",
    })
    with experiments.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
