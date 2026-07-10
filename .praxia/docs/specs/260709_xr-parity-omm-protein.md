---
title: "XR-PARITY-OMM-PROTEIN — OpenMM vacuum protein confirmatory"
backlog_id: XR-PARITY-OMM-PROTEIN
epic: 260709_xtrax_rewire
depends_on: [XR-A2A3, XR-KILL-FORK, XR-VACUUM-DT]
priority: P2
difficulty: extended
status: completed
challenge_verdict: pass
challenge_summary: "exception_* wired into energy_fn_from_bundle; 2GB1 shared-amber14 ΔE≈0.001, force_rmse≈2e-6; pytest gate green."
---

# XR-PARITY-OMM-PROTEIN

## Goal
Falsifiable OpenMM vacuum-protein energy/force parity on the EnsemblePlan bundle path after A2A3 + VACUUM-DT — not Claim-1 throughput.

## Locked decisions

| Topic | Lock |
|-------|------|
| Fixture | `data/pdb/2GB1.pdb`, vacuum, `NoCutoff`, no constraints |
| Param source | **Shared OpenMM `amber14-all.xml`** → extract bonded+NB+`ExclusionSpec` → `make_bundle_from_system` (not cross-FF ff19SB vs amber14) |
| Energy path | `energy_fn_from_bundle` (must include `exception_*` 1-4 pairs) |
| Static ΔE | `ABS(delta_e_kcal) <= 0.1` |
| Static ΔF | `force_rmse_kcal_mol_A < 3.0` (water-matched; measured ~2e-6) |
| Dynamics | Optional smoke: `EnsemblePlan.run(n_steps=1000, dt=0.5, gamma=50)` finite (VACUUM-DT policy) — not a T-parity gate |
| Delivery | Pytest gate (required). Bathos confirmatory campaign = optional stretch. |
| Non-goals | Claim-1 ns/day; solvated protein; ff19SB↔amber14 cross-FF; paper figures |

## Root cause fixed (2026-07-10)
`energy_fn_from_bundle` applied dense 1-2/1-3 exclusions but **omitted AMBER 1-4 `exception_*` pair energy**. Adding masked exception LJ+Coulomb closes 2GB1 static parity to ΔE≈0.001 kcal/mol, force_rmse≈2e-6.

## Acceptance Criteria
1. Given 2GB1 built from shared OpenMM amber14 params + exclusions, when `energy_fn_from_bundle` vs OpenMM Reference compare, then `|ΔE|≤0.1` and `force_rmse<3.0`.
2. Given scope review, when Claim-1 / solvated / cross-FF appear, then they are rejected as out of scope.
3. Given VACUUM-DT policy, when documented, then protein MD uses `gamma≥50` at `dt=0.5` fs (or `dt≤0.1` at `gamma=10`).

## References
- Water template: `.praxia/docs/specs/260709_xr-parity-omm-water.md`
- Gate: `tests/physics/test_xr_parity_omm_protein.py`
- VACUUM-DT: `.praxia/docs/specs/260709_xr-vacuum-dt.md`
