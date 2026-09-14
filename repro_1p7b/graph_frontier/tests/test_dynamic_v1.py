import argparse
import json

import pytest

from repro_1p7b.graph_frontier import dynamic_v1 as dynamic


def capability(reach=(0.5, 0.6, 0.7), conditional=(0.8, 0.3, 0.1), semantic=(0.4, 0.3, 0.2)):
    buckets = {}
    for name, r, c, s in zip(dynamic.BUCKETS[1:], reach, conditional, semantic):
        buckets[name] = {
            "task_support": 40,
            "internal_reach": {"smoothed_rate": r},
            "conditional_propagation": {"smoothed_rate": c},
            "semantic_success": {"smoothed_rate": s},
        }
    return {
        "consumed_split": "diagnosis",
        "heldout_tasks_consumed": 0,
        "runtime": {"valid": 240},
        "overall_semantic_success": {"smoothed_rate": 0.3},
        "buckets": buckets,
    }


def sample(call='{"name":"a","arguments":{"x":"value-123"}}'):
    return {
        "instruction": "do it", "input": "", "system": "tools", "history": [],
        "output": f"<tool_call>{call}</tool_call>",
    }


def test_frontier_formula_prefers_reached_but_incorrect():
    rows = dynamic.priorities(capability(), 0.05)
    assert rows["depth3plus_internal"]["propagation_need"] > rows["depth1_internal"]["propagation_need"]
    assert sum(row["normalized_probability"] for row in rows.values()) == pytest.approx(1.0)
    assert all(row["normalized_probability"] >= 0.05 for row in rows.values())


def test_largest_remainder_is_exact_and_deterministic():
    probabilities = {bucket: 0.25 for bucket in dynamic.BUCKETS}
    first = dynamic.largest_remainder(probabilities, 13)
    second = dynamic.largest_remainder(probabilities, 13)
    assert first == second
    assert sum(first.values()) == 13


def test_overlap_projection_respects_cap_and_floor():
    requested = dict(zip(dynamic.BUCKETS, (10, 10, 60, 20)))
    total = {bucket: 100 for bucket in dynamic.BUCKETS}
    unseen = dict(zip(dynamic.BUCKETS, (100, 100, 20, 100)))
    priorities = {bucket: {"raw_priority": idx + 1} for idx, bucket in enumerate(dynamic.BUCKETS)}
    answer = dynamic.constrain_allocation(requested, total, unseen, 100, 10, 5, priorities)
    assert sum(answer.values()) == 100
    assert sum(max(0, answer[b] - unseen[b]) for b in dynamic.BUCKETS) <= 10
    assert min(answer.values()) >= 5


def test_quality_gate_rejects_malformed_and_repeated_calls():
    assert dynamic.quality_audit(sample())["eligible"]
    repeated = sample()
    repeated["output"] += repeated["output"]
    assert not dynamic.quality_audit(repeated)["eligible"]
    assert not dynamic.quality_audit(sample("not-json"))["eligible"]


def test_content_identity_is_key_order_stable():
    first = sample()
    second = json.loads(json.dumps(first))
    second = {key: second[key] for key in reversed(list(second))}
    assert dynamic.content_id(first) == dynamic.content_id(second)


def test_capability_split_gate_rejects_heldout(tmp_path):
    value = capability(); value["heldout_tasks_consumed"] = 1
    source, features = fixture_source_and_features()
    source_path = tmp_path / "source.json"; source_path.write_text(json.dumps(source))
    features_path = tmp_path / "features.jsonl"
    features_path.write_text("".join(json.dumps(row) + "\n" for row in features))
    stage1 = tmp_path / "stage1.json"; stage1.write_text(json.dumps(source[:8]))
    capability_path = tmp_path / "capability.json"; capability_path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="diagnosis-only"):
        dynamic.build_stage2(stage2_args(tmp_path, source_path, features_path, stage1, capability_path, "bad"))


def fixture_source_and_features():
    rows = []
    features = []
    for bucket_index, depth in enumerate((0, 1, 2, 3)):
        for item in range(12):
            index = len(rows)
            rows.append(sample(json.dumps({
                "name": f"tool_{bucket_index}_{item}",
                "arguments": {"x": f"value-{bucket_index}-{item}"},
            })))
            features.append({
                "index": index, "chars": 100 + item, "tool_calls": 1,
                "dependent_calls": int(depth > 0), "max_dependency_depth": depth,
                "has_internal_dependency": depth > 0,
            })
    return rows, features


def stage2_args(tmp_path, source, features, stage1, capability_path, suffix):
    return argparse.Namespace(
        source=source, features=features, stage1=stage1, capability=capability_path,
        output=tmp_path / f"stage2_{suffix}.json", plan=tmp_path / f"plan_{suffix}.json",
        stats=tmp_path / f"stats_{suffix}.json", count=12, seed=dynamic.SEED,
        overlap_cap=0.20, exploration_floor=0.05,
    )


def test_dataset_builders_are_deterministic_and_respect_constraints(tmp_path):
    source, features = fixture_source_and_features()
    source_path = tmp_path / "source.json"; source_path.write_text(json.dumps(source))
    features_path = tmp_path / "features.jsonl"
    features_path.write_text("".join(json.dumps(row) + "\n" for row in features))
    stage1 = tmp_path / "stage1.json"; stage1_stats = tmp_path / "stage1_stats.json"
    dynamic.build_stage1(argparse.Namespace(
        source=source_path, features=features_path, output=stage1, stats=stage1_stats,
        count=8, seed=dynamic.SEED,
    ))
    capability_path = tmp_path / "capability.json"; capability_path.write_text(json.dumps(capability()))
    first = stage2_args(tmp_path, source_path, features_path, stage1, capability_path, "first")
    second = stage2_args(tmp_path, source_path, features_path, stage1, capability_path, "second")
    dynamic.build_stage2(first); dynamic.build_stage2(second)
    assert first.output.read_bytes() == second.output.read_bytes()
    assert json.loads(first.plan.read_text())["target_allocation"] == json.loads(second.plan.read_text())["target_allocation"]
    selected = json.loads(first.output.read_text())
    assert len(selected) == len({dynamic.content_id(row) for row in selected}) == 12
    assert json.loads(first.stats.read_text())["cross_stage_overlap"] <= 2
