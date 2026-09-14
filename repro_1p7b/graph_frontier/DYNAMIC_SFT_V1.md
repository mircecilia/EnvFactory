# Dynamic Graph-Frontier SFT v1

## Budget and stages

Dynamic v1 uses a fixed budget of 26,463 source-example exposures: 13,232 in
Stage 1 and 13,231 in Stage 2.  With two GPUs, micro-batch 1 and gradient
accumulation 32, the intended boundary is optimizer step 207 and the final
horizon is step 414.

Stage 1 is a seed-20260914 deterministic shuffle without replacement from the
official filtered EnvFactory SFT source.  It does not use probe results.
Its generated statistics also report source-versus-selection bucket-rate deltas
so broad-distribution preservation is auditable.

## Frontier allocation

Only the 240 frozen `diagnosis` probes from the Stage-1 checkpoint are consumed.
The 60 `heldout` probes are neither executed nor read for allocation.

For each depth bucket, Beta(1,1)-smoothed rates define:

```
propagation_need = reach * (1 - conditional_propagation)
semantic_frontier = 4 * p_semantic * (1 - p_semantic)
priority = 0.70 * propagation_need + 0.30 * semantic_frontier
```

The structural training buckets are `shallow_general`, `depth1_internal`,
`depth2_internal`, and `depth3plus_internal`.  Shallow data has no directly
corresponding probe edge, so its propagation need is zero and it uses the global
diagnosis semantic frontier.  Every bucket receives a 5% exploration floor.

Stage 2 is sampled without replacement by content identity.  Exact duplicate
tool calls, mismatched tool-call tags, and malformed tool-call JSON are rejected.
The available flattened SFT metadata cannot recover gold graph edges; therefore
the training-bucket mapping uses the documented exact scalar-reuse dependency
proxy, not guessed graph structure. Cross-stage overlap is
at most 20%; combined content uniqueness must be at least 90%.

## Training continuity

The implementation uses Option B: native Transformers/DeepSpeed checkpoint
resume plus a minimal callback outside LLaMAFactory core.  Both stages declare
the same 414-step cosine horizon.  The callback saves and stops Stage 1 at step
207, before Stage 2 resumes from the complete ZeRO-3 checkpoint with
`ignore_data_skip=true` so the new dataset starts at its first batch.

Transformers constructs the Stage-2 scheduler for 414 steps before loading the
saved scheduler state.  Optimizer, scheduler, global step, scaler and RNG are
loaded through the standard DeepSpeed/Trainer resume path.  Boundary metadata
is copied to `stage1_boundary_metadata` so it survives checkpoint rotation.

The required 2+2-step smoke checks global step 2 to 3, optimizer and scheduler
restoration, fixed horizon, RNG files, and a changed tokenized dataset signature.

## Automation and recovery

`run_graph_frontier_dynamic_v1.sh start` refuses existing formal output.  The
detached orchestrator advances through:

```
STAGE1_TRAINING -> STAGE1_DONE -> DIAGNOSING -> BUILDING_STAGE2
-> STAGE2_READY -> STAGE2_TRAINING -> TRAINING_DONE
```

Each subprocess exit code is persisted.  `resume` uses a valid latest checkpoint
for an interrupted training stage, resumes partial diagnosis task files, and
never overwrites an incomplete Stage-2 dataset.  Any failed gate writes `FAILED`.

No final BFCL, held-out probe run, full 300-probe evaluation, RL, or subsequent
training is part of this orchestrator.
