# Graph-Frontier CPU Prototype

This directory contains offline infrastructure only. It does not start training, model inference, rollout execution, Ray, vLLM, or GPU work.

## Components

- `SCHEMA_AUDIT.md`: source and serialization audit, including what the existing artifacts cannot recover.
- `profile.schema.json`: versioned rollout-profile JSON schema with explicit `unknown` values.
- `adapters.py`: live `ToolGraph`, QueryGen artifact, and typed-rollout adapters.
- `profiler.py`: programmatic dependency checks and first-root/downstream attribution.
- `capability.py`: Beta(1,1)-smoothed capability aggregation and scoring components.
- `selector.py`: deterministic, budget-aware selector with exploration, repeat, and group caps.
- `demo.py`: tiny synthetic end-to-end example; its checked-in output is `mock_demo_output.json`.

## CPU checks

From the repository root:

```bash
python -m py_compile repro_1p7b/graph_frontier/*.py
python -m unittest discover -s repro_1p7b/graph_frontier/tests -v
python -m repro_1p7b.graph_frontier.demo \
  --output repro_1p7b/graph_frontier/mock_demo_output.json
```

The mock output is illustrative, not an experiment result. It uses three synthetic rollouts and three synthetic curriculum candidates.

## Integration boundary

For real profiling, export a graph projection while the live official `ToolGraph` is still available and pair it with ordered executor events carrying typed `execution_success`, an explicit terminal verifier, and (when state correctness matters) an explicit state verifier. Legacy QueryGen JSON is accepted, but missing correctness and graph fields remain `unknown`; they are never guessed from text.
