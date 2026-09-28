# Recursive OPD V3 / Complete-C

Complete-C preserves three naturally occurring, verifier-defined supervision
classes from one fresh current-policy rollout:

1. `CURRENT_FRONTIER`: the V2 FIRST_ERROR correction (START, ADVANCE, or
   TERMINATE);
2. `SUCCESS_TERMINAL`: the verified current-policy terminal prefix followed by
   EOS;
3. `CROSSED_FRONTIER_RETENTION`: one current student tool action that strictly
   crosses the latest relevant historical START/ADVANCE frontier.

No class quota or class weight is part of the protocol.  Each task contributes
at most one main behavior row and one retention row.  Only the latest relevant
pre-terminal historical correction is considered, so ancestor replay cannot
grow mechanically with cycle count.

Retention is stricter than task-level frontier movement.  A materialized row
requires the current on-policy prompt boundary, one structured student call,
successful typed execution, exact response/state progression, immutable
policy/rollout/gold hashes, and a train-split source.  Multi-call crossings,
alternative ambiguity, gold-only states, off-gold states, heldout tasks,
UNCERTAIN trajectories, and context overflow are excluded.

CPU preparation:

```bash
python -m repro_1p7b.graph_frontier.recursive_opd_v3.prepare build --workers 6
python -m repro_1p7b.graph_frontier.recursive_opd_v3.prepare finalize
```

Runtime datasets live under `recursive_opd_v3_run/` and are not source
artifacts.  The audited report is written to
`graph_frontier/reports/recursive_opd_v3_cpu_prep.{json,md}`.

## GPU entry is deliberately fail closed

`run_gpu_train.sh` has no defaults.  A later explicit decision must provide
dataset, initialization, steps, exposure budget, learning rate, seed, output,
and an exact approval manifest.  The default state is:

```text
V3_GPU_STEPS = UNDECIDED
V3_EXPOSURE_BUDGET = UNDECIDED
V3_LR = UNDECIDED
```

CPU readiness does not authorize GPU training.
