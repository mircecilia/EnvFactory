# Graph-Frontier CPU Prototype

This directory is CPU-only by default. Importing the library does not start training, model inference, rollout execution, Ray, vLLM, or GPU work. The explicit `real_probe_smoke.py` entry point connects to a caller-started SGLang endpoint only when invoked.

## Components

- `SCHEMA_AUDIT.md`: source and serialization audit, including what the existing artifacts cannot recover.
- `profile.schema.json`: versioned rollout-profile JSON schema with explicit `unknown` values.
- `GOLD_SIDECAR_SCHEMA.json` / `gold_sidecar.py`: task-local generation-time graph export.
- `ROLLOUT_TRACE_SCHEMA.json` / `rollout_trace.py`: typed executor trace recorder and wrappers.
- `export_pipeline.py`: strict task-ID join from gold sidecar and typed trace to profiler.
- `EXPORT_PIPELINE.md`: real hook-point audit and integration instructions.
- `real_probe_smoke.py`: opt-in one-task QueryGen -> sidecar -> typed FastMCP rollout -> profiler acceptance driver.
- `REAL_PROBE_READINESS.md`: current semantic/runtime gates and real-smoke evidence and limitations.
- `PROBE_PLAN.md`: frozen 300-task diagnostic pool design; no pool is generated here.
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

The mock output is illustrative, not an experiment result. It uses three synthetic rollouts and three synthetic curriculum candidates. `tests/test_export_pipeline.py` separately verifies the real sidecar-to-typed-trace interface with structured values `123`, `456`, and the mismatched `999`.

## Integration boundary

For real profiling, export a graph projection while the live official `ToolGraph` is still available and pair it with ordered executor events carrying typed `execution_success`, an explicit terminal verifier, and (when state correctness matters) an explicit state verifier. Legacy QueryGen JSON is accepted, but missing correctness and graph fields remain `unknown`; they are never guessed from text.
