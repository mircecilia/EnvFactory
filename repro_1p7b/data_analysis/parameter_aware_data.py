#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
LOW_INFORMATION_STRINGS = {
    "true", "false", "none", "null", "yes", "no", "ok", "usd", "eur", "gbp",
    "active", "inactive", "success", "failed", "annual", "quarterly", "daily",
}


def sample_text_size(sample: Dict[str, Any]) -> int:
    total = 0
    for key in ("instruction", "input", "output", "system"):
        value = sample.get(key, "")
        total += len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False))
    for pair in sample.get("history") or []:
        if isinstance(pair, list):
            for value in pair:
                total += len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False))
    return total


def normalized_scalar(value: Any) -> Optional[Tuple[str, str]]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if len(value) < 4 or value.lower() in LOW_INFORMATION_STRINGS:
            return None
        return ("s", value)
    if isinstance(value, int):
        if value in (-1, 0, 1):
            return None
        return ("n", str(value))
    if isinstance(value, float):
        if not math.isfinite(value) or value in (-1.0, 0.0, 1.0):
            return None
        return ("n", repr(value))
    return None


def leaf_scalars(value: Any, path: str = "") -> Iterable[Tuple[str, Tuple[str, str]]]:
    scalar = normalized_scalar(value)
    if scalar is not None:
        yield path, scalar
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from leaf_scalars(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_path = f"{path}[{idx}]"
            yield from leaf_scalars(child, child_path)


def parse_json_fragments(text: str, pattern: re.Pattern) -> List[Any]:
    values = []
    for fragment in pattern.findall(text or ""):
        try:
            values.append(json.loads(fragment))
        except (json.JSONDecodeError, TypeError):
            continue
    return values


def pair_sequence(sample: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for pair in sample.get("history") or []:
        if isinstance(pair, list) and len(pair) >= 2:
            pairs.append((str(pair[0] or ""), str(pair[1] or "")))
    pairs.append((str(sample.get("instruction") or ""), str(sample.get("output") or "")))
    return pairs


def extract_features(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    available: Dict[Tuple[str, str], int] = {}
    pending_call_depths: List[int] = []
    tool_calls = 0
    dependent_calls = 0
    matched_args = 0
    max_depth = 0
    response_objects = 0
    parsed_calls = 0

    for input_text, output_text in pair_sequence(sample):
        producer_depth = max(pending_call_depths, default=0)
        for response in parse_json_fragments(input_text, TOOL_RESPONSE_RE):
            response_objects += 1
            for _, scalar in leaf_scalars(response):
                previous = available.get(scalar)
                if previous is None or producer_depth > previous:
                    available[scalar] = producer_depth

        current_depths: List[int] = []
        for call in parse_json_fragments(output_text, TOOL_CALL_RE):
            if not isinstance(call, dict):
                continue
            parsed_calls += 1
            tool_calls += 1
            arguments = call.get("arguments", {})
            match_depths: List[int] = []
            call_matches = 0
            seen = set()
            for _, scalar in leaf_scalars(arguments):
                if scalar in seen:
                    continue
                seen.add(scalar)
                if scalar in available:
                    call_matches += 1
                    match_depths.append(available[scalar])
            if call_matches:
                dependent_calls += 1
                matched_args += call_matches
                depth = 1 + max(match_depths)
            else:
                depth = 0
            current_depths.append(depth)
            max_depth = max(max_depth, depth)
        pending_call_depths = current_depths

    return {
        "index": index,
        "chars": sample_text_size(sample),
        "history_turns": len(sample.get("history") or []),
        "tool_calls": tool_calls,
        "parsed_calls": parsed_calls,
        "response_objects": response_objects,
        "dependent_calls": dependent_calls,
        "matched_args": matched_args,
        "max_dependency_depth": max_depth,
        "has_internal_dependency": dependent_calls > 0,
    }


def percentile(sorted_values: Sequence[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def distribution(values: Sequence[int], include_counts: bool = True) -> Dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    counts = Counter(values)
    result = {
        "count": len(values),
        "min": ordered[0],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }
    if include_counts:
        result["counts"] = {str(k): counts[k] for k in sorted(counts)}
    return result


def summarize(features: Sequence[Dict[str, Any]], input_path: str) -> Dict[str, Any]:
    count = len(features)
    dep_count = sum(int(x["has_internal_dependency"]) for x in features)
    depth_bins = Counter(
        "0" if x["max_dependency_depth"] == 0 else
        "1" if x["max_dependency_depth"] == 1 else
        "2" if x["max_dependency_depth"] == 2 else "3+"
        for x in features
    )
    match_bins = Counter(
        "0" if x["matched_args"] == 0 else
        "1" if x["matched_args"] == 1 else
        "2" if x["matched_args"] == 2 else "3+"
        for x in features
    )
    return {
        "schema_version": 1,
        "input": input_path,
        "method": {
            "name": "exact_scalar_reuse_proxy_v1",
            "description": "A later tool-call argument exactly matches a non-trivial scalar parsed from an earlier tool response in the same flattened SFT prefix.",
            "exclusions": "null, booleans, -1/0/1, strings shorter than 4, and a small low-information string stoplist",
            "limitation": "The filtered SFT JSON does not retain ToolGraph dependency edges; this is a conservative proxy and can contain coincidental exact matches.",
        },
        "sample_count": count,
        "total_chars": sum(x["chars"] for x in features),
        "dependency_proxy_samples": dep_count,
        "dependency_proxy_rate": dep_count / count if count else 0.0,
        "dependency_depth_bins": dict(sorted(depth_bins.items())),
        "matched_argument_bins": dict(sorted(match_bins.items())),
        "chars": distribution([x["chars"] for x in features], include_counts=False),
        "history_turns": distribution([x["history_turns"] for x in features]),
        "tool_calls": distribution([x["tool_calls"] for x in features]),
        "dependent_calls": distribution([x["dependent_calls"] for x in features]),
        "matched_args": distribution([x["matched_args"] for x in features]),
        "max_dependency_depth": distribution([x["max_dependency_depth"] for x in features]),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_dataset(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data[:limit] if limit is not None else data


def run_analyze(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    samples = load_dataset(input_path, args.limit)
    features_path = Path(args.features)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    with features_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            feature = extract_features(sample, index)
            features.append(feature)
            handle.write(json.dumps(feature, ensure_ascii=False) + "\n")
            if (index + 1) % 2000 == 0:
                print(f"analyzed {index + 1}/{len(samples)}", flush=True)
    summary = summarize(features, str(input_path))
    write_json(Path(args.summary), summary)
    print(json.dumps({
        "status": "PASS",
        "sample_count": summary["sample_count"],
        "dependency_proxy_rate": summary["dependency_proxy_rate"],
        "total_chars": summary["total_chars"],
        "features": args.features,
        "summary": args.summary,
    }, indent=2))


def read_features(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def weight(feature: Dict[str, Any]) -> float:
    return (
        1.0
        + 1.25 * min(int(feature["dependent_calls"]), 3)
        + 0.75 * min(int(feature["max_dependency_depth"]), 3)
        + 0.20 * min(int(feature["matched_args"]), 5)
    )


def choose_indices(features: Sequence[Dict[str, Any]], sample_count: int, seed: int, bin_width: int) -> List[int]:
    if sample_count != len(features):
        raise ValueError("This controlled selector requires sample_count to equal the candidate count.")
    bins: Dict[int, List[int]] = defaultdict(list)
    for pos, feature in enumerate(features):
        bins[int(feature["chars"]) // bin_width].append(pos)

    rng = random.Random(seed)
    selected: List[int] = []
    for bin_id in sorted(bins):
        members = bins[bin_id]
        weights = [weight(features[pos]) for pos in members]
        selected.extend(rng.choices(members, weights=weights, k=len(members)))
    rng.shuffle(selected)
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_select(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    samples = load_dataset(input_path, args.limit)
    if args.limit is not None and args.sample_count is None:
        sample_count = len(samples)
    else:
        sample_count = args.sample_count or len(samples)

    if args.features:
        all_features = read_features(Path(args.features))
        features = all_features[:len(samples)]
        if len(features) != len(samples):
            raise ValueError("Feature row count does not match the selected candidate pool.")
    else:
        features = [extract_features(sample, index) for index, sample in enumerate(samples)]

    selected_positions = choose_indices(features, sample_count, args.seed, args.bin_width)
    selected_samples = [samples[pos] for pos in selected_positions]
    selected_features = [features[pos] for pos in selected_positions]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(selected_samples, handle, ensure_ascii=False)

    before = summarize(features, str(input_path))
    after = summarize(selected_features, str(output_path))
    unique_count = len(set(selected_positions))
    report = {
        "schema_version": 1,
        "status": "PASS",
        "selector": "length_stratified_weighted_resampling_v1",
        "seed": args.seed,
        "bin_width_chars": args.bin_width,
        "weight_formula": "1 + 1.25*min(dependent_calls,3) + 0.75*min(max_dependency_depth,3) + 0.20*min(matched_args,5)",
        "input": str(input_path),
        "output": str(output_path),
        "sample_count": len(selected_samples),
        "unique_source_samples": unique_count,
        "duplicate_draws": len(selected_samples) - unique_count,
        "unique_source_rate": unique_count / len(selected_samples) if selected_samples else 0.0,
        "before": {
            "total_chars": before["total_chars"],
            "dependency_proxy_samples": before["dependency_proxy_samples"],
            "dependency_proxy_rate": before["dependency_proxy_rate"],
            "dependency_depth_bins": before["dependency_depth_bins"],
            "tool_calls": before["tool_calls"],
            "max_dependency_depth": before["max_dependency_depth"],
        },
        "after": {
            "total_chars": after["total_chars"],
            "dependency_proxy_samples": after["dependency_proxy_samples"],
            "dependency_proxy_rate": after["dependency_proxy_rate"],
            "dependency_depth_bins": after["dependency_depth_bins"],
            "tool_calls": after["tool_calls"],
            "max_dependency_depth": after["max_dependency_depth"],
        },
        "char_budget_ratio": after["total_chars"] / before["total_chars"] if before["total_chars"] else 0.0,
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
    }
    write_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_validate(args: argparse.Namespace) -> None:
    path = Path(args.input)
    samples = load_dataset(path, None)
    required = {"instruction", "input", "output", "system", "history"}
    bad = []
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict) or not required.issubset(sample):
            bad.append(idx)
        elif not isinstance(sample.get("history"), list):
            bad.append(idx)
        if len(bad) >= 20:
            break
    result = {
        "status": "PASS" if not bad else "FAIL",
        "input": str(path),
        "sample_count": len(samples),
        "bad_indices": bad,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if args.report:
        write_json(Path(args.report), result)
    print(json.dumps(result, indent=2))
    if bad:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze and reweight EnvFactory SFT samples using a prior-tool-response reuse proxy.")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--features", required=True)
    analyze.add_argument("--summary", required=True)
    analyze.add_argument("--limit", type=int)
    analyze.set_defaults(func=run_analyze)

    select = sub.add_parser("select")
    select.add_argument("--input", required=True)
    select.add_argument("--features")
    select.add_argument("--output", required=True)
    select.add_argument("--report", required=True)
    select.add_argument("--sample-count", type=int)
    select.add_argument("--seed", type=int, default=20260913)
    select.add_argument("--bin-width", type=int, default=512)
    select.add_argument("--limit", type=int)
    select.set_defaults(func=run_select)

    validate = sub.add_parser("validate")
    validate.add_argument("--input", required=True)
    validate.add_argument("--report")
    validate.set_defaults(func=run_validate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
