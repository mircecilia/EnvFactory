#!/usr/bin/env python3
import argparse
from pathlib import Path

from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH, PROMPT_PATH
from bfcl_eval.eval_checker.eval_runner import get_handler, multi_turn_runner
from bfcl_eval.utils import extract_test_category, find_file_with_suffix, load_file

def main():
    parser = argparse.ArgumentParser(description="Evaluate a run-id subset with official BFCL V3 scoring.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-FC")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    escaped_model = args.model.replace("/", "_")
    result_model_dir = project_root / "result" / escaped_model
    score_root = project_root / "score"
    result_files = sorted(result_model_dir.glob("BFCL_v3_multi_turn*_result.json"))
    if len(result_files) != 4:
        raise RuntimeError(f"expected four multi-turn result files, found {len(result_files)}")

    handler = get_handler(args.model)
    summary = {}
    for result_file in result_files:
        category = extract_test_category(result_file)
        model_result = load_file(result_file, sort_by_id=True)
        result_ids = [row["id"] for row in model_result]
        if len(result_ids) != len(set(result_ids)):
            raise RuntimeError(f"{category}: duplicate result IDs")

        prompt = load_file(find_file_with_suffix(PROMPT_PATH, category), sort_by_id=True)
        possible = load_file(find_file_with_suffix(POSSIBLE_ANSWER_PATH, category), sort_by_id=True)
        prompt_by_id = {row["id"]: row for row in prompt}
        possible_by_id = {row["id"]: row for row in possible}
        missing = [item for item in result_ids if item not in prompt_by_id or item not in possible_by_id]
        if missing:
            raise RuntimeError(f"{category}: IDs absent from official data: {missing}")

        selected_prompt = [prompt_by_id[item] for item in result_ids]
        selected_possible = [possible_by_id[item] for item in result_ids]
        accuracy, total_count = multi_turn_runner(
            handler,
            model_result,
            selected_prompt,
            selected_possible,
            escaped_model,
            category,
            score_root,
        )
        summary[category] = {"accuracy": accuracy, "total_count": total_count}
        print(f"{category}: accuracy={accuracy} total_count={total_count}")

    if sum(item["total_count"] for item in summary.values()) == 0:
        raise RuntimeError("partial evaluation produced zero evaluated cases")

if __name__ == "__main__":
    main()
