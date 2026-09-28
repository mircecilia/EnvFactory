"""CPU-only gates for the on-policy Complete-C + fresh-anchor runner."""
from __future__ import annotations

import pytest

from repro_1p7b.graph_frontier.recursive_opd_v3_experiment import rollout, schedule


def test_real_d0_schedule_uses_audited_pi0_and_no_anchor():
    rows, plan = schedule.build_plan(0)
    assert len(rows) == 128
    assert plan["total_exposures"] == 128
    assert plan["optimizer_steps"] == 64
    assert plan["fresh_anchors_available"] == plan["fresh_anchors_exposed"] == 0
    assert plan["parity_filler"] == 0
    assert sum(plan["sampled_natural_main_composition"].values()) == 128
    assert sum(plan["sampled_composition"].values()) == 128
    assert plan["sampled_natural_main_tool_targets"] + plan["sampled_natural_main_eos_targets"] == 128
    assert len({item["identity"] for item in plan["stream"]}) == 128
    assert all(item["source"] == "natural" and item["tag"].startswith("NATURAL_")
               for item in plan["stream"])


def test_no_pi5_train_or_pi0_fresh_rollout():
    with pytest.raises(ValueError):
        schedule.build_plan(4)
    with pytest.raises(ValueError):
        rollout.lock(0)


def test_rollout_digest_is_stable():
    assert rollout.sha256_text("same") == rollout.sha256_text("same")
    assert rollout.sha256_text("same") != rollout.sha256_text("different")


def test_odd_anchor_count_gets_distinct_natural_filler_and_all_anchors_once():
    natural = [{"complete_c_source_label": "CURRENT_FRONTIER_CORRECTION",
                "complete_c_sample_identity": f"natural-{i}"} for i in range(140)]
    anchors = [{"anchor_sample_identity": f"anchor-{i}"} for i in range(3)]
    stream, natural_order, filler = schedule.exposure_stream(natural, anchors, 17)
    assert filler == 1
    assert len(stream) == 132
    assert len({item["identity"] for item in stream}) == 132
    assert {item["identity"] for item in stream if item["tag"] == "FRESH_ANCHOR"} == {
        "anchor-0", "anchor-1", "anchor-2"}
    assert [item["identity"] for item in stream if item["tag"] == "DDP_PARITY_FILLER"] == [
        natural[natural_order[128]]["complete_c_sample_identity"]]
    assert stream == schedule.exposure_stream(natural, anchors, 17)[0]
