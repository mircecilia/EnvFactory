# Research heartbeat — 2026-10-07

## Start

- Connected GPU host 10.251.171.6:28143, branch repro/envfactory-1p7b; both A100 idle, no tmux.
- Priority: actual MAIN correction exposure, not candidate-pool membership.
- Extracting successful optimizer attempt stream + dataset identities, excluding the failed cycle0 attempt from trained-policy exposure.
- Existing collector does NOT truncate rollout at first divergence; the miner excludes off-path supervision. Recovery motivation must reflect this distinction.
- Current stage CPU-only; no inference or training launched. Next: exposure-conditioned and selection-adjusted paired audit; related-work novelty check.

## 2026-10-07 18:20 Asia/Singapore — audit complete / prototype gated

- MAIN exposure rebuilt from successful optimizer inputs: 341 task-transition events, each exposed once. Strict crossing 23/309 (7.44%) vs 282/3495 (8.07%); task-cluster difference CI [-3.53,+2.50] pp. Eligible matched and no-other-exposure sensitivity do not reverse the conclusion.
- Structural proxy heldout signal in cycle0 disappears under environment matching (5 treated/7 controls, difference0); later cycles do not reproduce it. Real dependency metadata remains unknown.
- All-task START grows496→734, frontier CDF/transition matrices show no monotonic self-curriculum. Logging/verifier coverage changes are retained as UNCERTAIN, not success.
- Independent100-task recovery CPU preflight:93 exact passes,7 failures. Repeated diagnostics: all7 saved state/post-state roundtrips match;6 observation array order differences and1 process-dependent GitHub comment_id. Original failed gate preserved; no GPU/model request/optimizer update launched.
- Four CPU rollback/mock tests PASS remotely. Local Windows asyncio mock invocations stalled and were interrupted; remote Linux tests are the authoritative test run.
- Next research step: versioned observation-portability protocol, then rerun whole fixed100-task gate before any recovery sampling. No new training authorized or started. Package small analysis/scripts/reports into a scoped commit, preserve existing BFCL change, no push.
