# Recursive OPD V3: Complete-C + fresh crossed-frontier anchor

This package is an isolated, fail-closed runner for the authorized Cπ0→Cπ4
mechanism experiment. Its source of truth is the frozen V1 protocol and original
Dynamic-v1 π0. Runtime output is under the ignored
`recursive_opd_v3_experiment_run/` directory; historical V1/V2/retrospective
Complete-C and anchor artifacts are read-only.

## Chain

1. Recheck the audited Complete-C D0 source and freeze 128 natural exposures.
2. Run an optimizer-free two-A100 longest-selected-example DDP smoke.
3. Train full-parameter BF16 Cπ1. Its checkpoint appears atomically only after
   successful saving and hash recording.
4. For each Cπ1–Cπ4, collect all 2158 fresh tasks using the frozen T=0 V1
   protocol, independent gold source, two isolated SGLang shards, and typed
   per-task episodes. An interrupted launcher skips only completed, matching
   episodes. Explicit 32K context rejection uses the existing typed boundary
   adapter; unrelated errors stop the run.
5. Verify all 2158 episode/gold/schema identities, then classify with the V1
   execution verifier. Build current-frontier, success-terminal, and at most
   one current verified continuation positive per train task. All targets use
   the exact current student prefix and assistant-only mask. Exclude uncertain
   and >16K context samples, with explicit counts. Heldout task IDs and query
   hashes are barred from training.
6. Build fresh anchors exclusively from the immediately previous policy's
   START/ADVANCE first errors crossed by this policy. The target is the current
   verified student action. All context-compatible anchors are exposed once,
   even if the same prefix/target also appears as a natural continuation. A
   deterministic extra natural row resolves odd DDP parity.
7. Repeat smoke/train for Cπ2–Cπ4. Cπ4 gets a fresh full rollout, natural D4
   and anchor A4 audit, but there is no Cπ5 update.
8. After full internal auditing, evaluate Cπ3 and Cπ4 on the identical V2
   Frozen300 and Rich24 plans, then generate machine-readable same-task
   comparison and retention traces.

The orchestration entry is `bash run_pipeline.sh`, normally in one dedicated
`tmux` session. It uses a pipeline lock and per-stage locks. A remote reset
does not erase completed episode files on the persistent workspace. After
checking that no old process/port/GPU is active, re-entering the same script
rechecks hashes, skips complete stages and continues the fixed plan. A
partially completed optimizer run must restart from the same input checkpoint
and immutable exposure plan; old attempt logs/temporary checkpoints remain
separate. No stage silently changes its data budget.

## Hard gates

- Frozen branch/source protocol, π0 model and tokenizer hashes, train/heldout
  split, and zero query-hash leakage.
- Full 2158/2158 on-policy rollout and typed gold/schema provenance before
  verifier mining.
- MAIN ≤1 and current continuation ≤1 per task; fresh anchor ≤1 and age=1.
- Every available context-compatible fresh anchor exposed exactly once.
- Natural 128-row deterministic sample without class quotas; actual sample
  composition and interleaved exposure stream recorded per update.
- Optimizer-free DDP longest-selected-example smoke before each update;
  finite loss/gradient, complete checkpoint save and hash lock afterward.
- Exact same Frozen300/Rich24 plan and base comparison as V2.

Any failed identity, verifier, context, checkpoint, or external comparison gate
stops the pipeline. It does not launch Cπ5, GRPO, Static retraining, or BFCL.
Training exposure/optimizer steps increase with the number of fresh anchors;
this is a mechanism feasibility experiment, **not** a matched-compute causal
ablation against V2. Human-readable reporting belongs on the local machine,
not in runtime output. Only small source/tests/protocol files should enter Git.
