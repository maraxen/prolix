---
title: "XR-BUCKET — tiling #746 bucketing invariant"
backlog_id: XR-BUCKET
praxia_id: 3286
epic: 260709_xtrax_rewire
depends_on: [XR-BUDGET]
priority: P3
difficulty: extended
status: completed
challenge_verdict: pass
challenge_summary: "round_up_to_multiple + compute_dense_tiling_dims; tile_reduction asserts; dense LJ/Coulomb routed; 5 unit tests green."
---

# XR-BUCKET

## Goal
Enforce FlashMD tiling bucketing so `inner_tile_size` / `pad_dim` invariants always hold by construction (#746) — no silent atom drop in `tile_reduction`.

## Locked decisions

| Topic | Lock |
|-------|------|
| Helper | `round_up_to_multiple(n, m)` + `compute_dense_tiling_dims(n_atoms, n_excl_rows, tile_size)` in `prolix.physics.tiling` |
| Invariants | `inner % tile_size == 0`; `pad_dim = max(tile_size, inner)`; `pad_dim >= max(n_atoms, n_excl_rows)` (via need); after `pad_to_tile(r, pad_dim)`, `n % tile == 0` and `n % inner == 0` |
| Loud fail | `tile_reduction` / `tile_reduction_nl` raise `ValueError` if shapes violate the multiple invariant |
| Call sites | Dense LJ/Coulomb fwd+bwd in `optimization.py` use `compute_dense_tiling_dims` (replace ad-hoc `((need+tile-1)//tile)*tile`) |
| Non-goals | xtrax host `Bucket` strategy execute; Claim-1; changing NL default inner=1024 policy beyond assert |

## Acceptance Criteria
1. Given helpers, when unit-tested, then round-up and dims satisfy the invariants above.
2. Given `tile_reduction` with `n` not divisible by `tile_size` or `inner_tile_size`, when called, then raise `ValueError` (not silent truncate).
3. Given dense chunked LJ/Coulomb paths, when grepped, then they call `compute_dense_tiling_dims` (no duplicated ad-hoc formula).
4. Given Claim-1 / full host Bucket pipeline, when reviewed, then out of scope.

## References
- `.praxia/ideas.jsonl` `tiling_bucketing_invariant_2026_05_29`
- `src/prolix/physics/tiling.py`, `optimization.py`
- CLAUDE.md tiling bug note
