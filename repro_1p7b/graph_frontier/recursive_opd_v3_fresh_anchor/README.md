# Fresh Crossed-Frontier Anchor Replay

Fresh anchors are an event-triggered one-cycle retention mechanism:

```text
previous-cycle START/ADVANCE FIRST_ERROR
  -> current-cycle exact execution-verified crossing
  -> current student prefix/action anchor for the next recursive update
  -> expire
```

This is not continuation class balancing.  The number of anchors is determined
by newly acquired frontiers, not a fixed class ratio or quota.  Natural
Complete-C data and the fresh-anchor pool remain physically separate.  The CPU
audit reports overlap, unique anchor-only supervision, and descriptive exposure
scenarios; it does not approve a training budget or sampling protocol.

```bash
python -m repro_1p7b.graph_frontier.recursive_opd_v3_fresh_anchor.prepare build --workers 8
python -m repro_1p7b.graph_frontier.recursive_opd_v3_fresh_anchor.prepare finalize
```

No inference, environment replay, training, or GPU service is run by either
command.
