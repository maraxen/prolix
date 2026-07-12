---
session_id: b350d77e
topic: B1-FULL shrink GPU footprint profiles: first test on L40S + multi-GPU measure tiers without OOM from full Trajectory buffers
task_type: constrained-technical
winner: Option A: chunked EnsemblePlan (fixed chunk_size, last-frame only) + --footprint tiers (l40s/a100/h200/prereg) mapped to replicas/ps/chunk and slurm GRES
created_at: 2026-07-12T14:29:09.570330+00:00
---

# Brainstorm: B1-FULL shrink GPU footprint profiles: first test on L40S + multi-GPU measure tiers without OOM from full Trajectory buffers

## Problem Frame
Fixed: Claim-1 uses bathos b1_init_exec (not pytest); cold-start timing segments stay; no H100/nodelist pinning; must run under uv 3.13 workspace. Negotiable for first L40S pass: B, trajectory length, and whether we keep full Trajectory stacks (we should not — chunked steps, last-frame only). Prereg 100ps×B=64 remains a later footprint tier (h200/prereg), not blocked.

## Idea Pool
- [ai] PEGS: Process=b1_init_exec cold-start MD timing; Event=OOM on L40S from lax.scan stacking full (n_steps,N,3); Goal=multi-GPU footprint tiers with first L40S green measure; State=traj buffer dominates VRAM.
- [ai] Option A — Chunked scan + last-frame only (keep EnsemblePlan API): Python loop of chunk_size steps, discard traj each chunk; peak VRAM ~ chunk×N not n_steps×N.
- [ai] Option B — Shrink B/ps only (replicas=2, 1ps): still OOMs if any chunk is large and 1ake×16×long remains.
- [ai] Option C — Change EnsemblePlan to final-only carry: bigger API change, out of leaf scope.
- [ai] Winner lean: A+tier presets (l40s/a100/h200/prereg) controlling replicas+ps+chunk; slurm FOOTPRINT→gres.
- [user] PEGS mapped. Options: (A) chunked EnsemblePlan runs + last-frame only + --footprint tiers; (B) shrink B/ps only without chunking — insufficient; (C) EnsemblePlan final-only API — out of scope. Prefer A. Converge on A.

## Decision Log
- [ACCEPT] Option A: chunked EnsemblePlan + footprint tiers: Fixes OOM root cause (full traj stack) while enabling multi-GPU comparable measures; prereg remains a later tier.
- [DEFER] Option C: EnsemblePlan final-only API: Correct long-term but out of B1 first-test leaf scope.
- [REJECT] Option B: shrink B/ps only: Insufficient alone if any full-scan length remains large; superseded by A.

## Assumptions

## TBDs

## Pre-mortem Record
**User:** Failure mode: people cite l40s-tier t_total as the Claim-1 headline without noting footprint≠prereg, or chunk recompiles inflate AOT and falsely trip aot_ratio. Mitigation: tag footprint in JSON/tags; never call l40s 'prereg'; document tiers in next_steps. Implement now.
**AI:** _not recorded_

## Acceptance Criteria
**Given** Fixed: Claim-1 uses bathos b1_init_exec (not pytest); cold-start timing segments stay; no H100/nodelist pinning; must run under uv 3.13 workspace. Negotiable for first L40S pass: B, trajectory length, and whether we keep full Trajectory stacks (we should not — chunked steps, last-frame only). Prereg 100ps×B=64 remains a later footprint tier (h200/prereg), not blocked.
**When** implementing Option A: chunked EnsemblePlan (fixed chunk_size, last-frame only) + --footprint tiers (l40s/a100/h200/prereg) mapped to replicas/ps/chunk and slurm GRES
**Then**
  - [ ] _add specific measurable criteria_
