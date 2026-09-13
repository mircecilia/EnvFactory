"""Tiny CPU-only demonstration of the Graph-Frontier offline pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import normalized_rollout_bundle
from .capability import aggregate_profiles
from .profiler import profile_rollout
from .selector import select_curriculum


def graph_spec():
    return {
        "expected_tool_nodes": ["lookup_user", "lookup_order", "cancel_order"],
        "dependency_edges": [
            {
                "edge_id": "user-to-order",
                "producer_tool": "lookup_user",
                "consumer_tool": "lookup_order",
                "edge_type": "parameter_flow",
                "source_parameter": "user_id",
                "target_parameter": "user_id",
                "required": True,
                "internal_parameter": True,
            },
            {
                "edge_id": "order-to-cancel",
                "producer_tool": "lookup_order",
                "consumer_tool": "cancel_order",
                "edge_type": "parameter_flow",
                "source_parameter": "order_id",
                "target_parameter": "order_id",
                "required": True,
                "internal_parameter": True,
            },
        ],
        "tool_metadata": {
            "lookup_user": {"state_changing": False},
            "lookup_order": {"state_changing": False},
            "cancel_order": {"state_changing": True},
        },
    }


def events(wrong_value=False):
    return [
        {"step_index": 0, "tool_name": "lookup_user", "arguments": {"email": "demo@example.test"}, "result": {"user_id": "user-7"}, "execution_success": True},
        {"step_index": 2, "tool_name": "lookup_order", "arguments": {"user_id": "wrong" if wrong_value else "user-7"}, "result": {"order_id": "order-9"}, "execution_success": not wrong_value},
        {"step_index": 4, "tool_name": "cancel_order", "arguments": {"order_id": "order-9"}, "result": {"cancelled": not wrong_value}, "execution_success": not wrong_value},
    ]


def build_demo():
    graph = graph_spec()
    bundles = [
        normalized_rollout_bundle("happy", events(), graph, True, True),
        normalized_rollout_bundle("wrong-propagation", events(True), graph, False, False),
        normalized_rollout_bundle("final-state-only", events(), graph, False, False),
    ]
    profiles = [profile_rollout(bundle) for bundle in bundles]
    capability = aggregate_profiles(profiles)
    candidates = [
        {"task_id": "candidate-depth-1", "character_count": 900, "token_count": 220, "diversity_group": "lookup", "capability_keys": ["depth:1"]},
        {"task_id": "candidate-depth-2", "character_count": 1100, "token_count": 275, "diversity_group": "mutation", "capability_keys": ["depth:2", "internal_parameter_flow"]},
        {"task_id": "candidate-explore", "character_count": 1000, "token_count": 250, "diversity_group": "novel", "capability_keys": ["unseen:capability"]},
    ]
    selection = select_curriculum(
        capability,
        candidates,
        sample_count=2,
        seed=20260913,
        target_character_budget=2000,
        target_token_budget=500,
        repeat_cap=1,
        group_cap=1,
        exploration_floor=0.10,
    )
    return {
        "demo_only": True,
        "cpu_only": True,
        "profiles": profiles,
        "capability_map": capability,
        "selection": selection,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_demo()
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
