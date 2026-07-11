---
title: "XR-PARITY-OMM-WATER — OpenMM TIP3P confirmatory"
backlog_id: XR-PARITY-OMM-WATER
praxia_id: 3278
epic: 260709_xtrax_rewire
depends_on: [XR-FIT-FLIP, XR-DISPATCH, XR-SHADOW]
priority: P1
difficulty: extended
status: completed
challenge_verdict: pass
challenge_summary: "Campaign 5ffe2644 pass on Titanix: |ΔE|=0.040, force_rmse=0.011, mean_T=303.6 K. Sparse excl fix; liquid-density T. XR-KILL-FORK unblocked."
---

# XR-PARITY-OMM-WATER

## Goal
Bathos confirmatory campaign proving rewire did not break OpenMM energy/force/T parity on TIP3P water (rewire falsification gate — not Claim-1 throughput).

## Locked decisions (delivery B + T2)

| Topic | Lock |
|-------|------|
| Delivery | Full confirmatory through `bth campaign conclude` (Union Gate) |
| Host | Prefer Titanix: rsync → `~/projects/prolix`, `uv sync --extra cuda --extra openmm`, then `bth run` |
| Driver | `scripts/experiments/xr_parity_omm_tip3p.py` (+ `.bth.toml`) — **not** throughput scripts |
| Claim | `.bth/claims/xr-parity-omm-water.claim.toml` |
| Campaign slug | `xr-parity-omm-water` |
| Static ΔE | 4-water TIP3P grid; `ABS(delta_e_kcal) <= 0.1` |
| Static ΔF | same; `force_rmse_kcal_mol_A < 3.0` |
| T NVT | 64 waters, dt=0.5 fs, gamma=10 ps⁻¹; `ABS(mean_t_k - 300) <= 15` |
| Residual | fail → rewire falsified; do not start XR-KILL-FORK |

## Scope
- New experiment driver + `[experiment]` sidecar + claim.toml
- Confirmation campaign: create → register claim → run → conclude
- Titanix GPU preferred (Engaging SLURM fallback)

## Non-goals
Protein systems; B1-full ns/day; paper figure; XR-KILL-FORK implementation.

## Acceptance Criteria
1. Given campaign created via `bth campaign create --mode confirmation`, when confirmatory runs complete, then outcomes evaluate via sidecar DuckDB conditions with exactly one `is_residual=true`.
2. Given TIP3P fixture (T2 table), when prolix vs OpenMM compare, then ΔE/ΔF/T satisfy pass conditions in the sidecar.
3. Given claim registered before confirmatory runs, when `bth campaign conclude` runs, then Union Gate clauses for rewire-correctness are covered.
4. Given residual/fail outcome, when reviewed, then XR-KILL-FORK must not proceed.

## Rollback
N/A (read-only campaigns); pin claim sha.

## References
- using-bathos claim-tier; `tests/physics/test_explicit_langevin_tip3p_parity.py`
