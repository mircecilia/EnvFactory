# Recursive OPD V3 Complete-C

This isolated protocol adds at most one current-policy, execution-verified
continuation positive to the unchanged V2 MAIN row for each task and cycle.
Qualification depends only on the current saved on-policy trajectory, exact
typed execution evidence, exact state progress, and exact student prefix/action.
Historical-frontier lineage is analysis metadata, never an eligibility gate.

The retrospective CPU audit is run with:

```bash
python -m repro_1p7b.graph_frontier.recursive_opd_v3_complete_c.prepare build --workers 8
python -m repro_1p7b.graph_frontier.recursive_opd_v3_complete_c.prepare finalize
```

The natural union is uniformly shuffled in the future training entry.  There
are no class quotas, oversampling rules, stop/continue weights, or inherited
training defaults.  `run_gpu_train.sh` fails closed unless cycle, dataset,
initial model, steps, exposure budget, learning rate, seed, output, and the
matching immutable approval manifest are all explicitly supplied.

CPU preparation does not run model inference, environment replay, or training.
