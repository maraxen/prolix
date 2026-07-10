---
title: "XR-VACUUM-DT — vacuum-protein timestep policy"
backlog_id: XR-VACUUM-DT
epic: 260709_xtrax_rewire
depends_on: [XR-A2A3]
priority: P1
difficulty: standard
status: completed
challenge_verdict: pass
challenge_summary: "dt was raw AKMA (~24 fs); gamma must be ps⁻¹→AKMA. Vacuum at dt=0.5 needs gamma≥50 (or dt≤0.1 at gamma=10)."
filed_reason: "XR-A2A3 C3 exception path — B1-pinned dt=0.5 fs not achievable for vacuum proteins after A1+A2"
---

# XR-VACUUM-DT

## Goal
Establish a falsifiable vacuum-protein MD timestep policy so B1-pinned `dt=0.5` fs is production-safe on `EnsemblePlan`, unblocking `XR-PARITY-OMM-PROTEIN`.

## Root cause (locked 2026-07-09)
1. **`dt` unit bug:** `EnsemblePlan._run_single` passed caller `dt` straight into `settle_langevin` **without** `dt_fs_to_akma`. Call sites mean `dt=0.5` as **0.5 fs**, but true AKMA needs `0.5 / 48.888 ≈ 0.01023`. Raw `0.5` is ~24 fs → step-1 KE explosion.
2. **`gamma` unit bug (compounding):** factory used `gamma=10.0` without `gamma_ps_to_akma`. Water NVT tests convert (`gamma_ps * AKMA_TIME_UNIT_FS * 1e-3`). The unconverted `10.0` is ~204 ps⁻¹ and accidentally overdamps vacuum runs — masking the need for a real vacuum policy once `dt` is fixed.
3. **Vacuum physics:** with *correct* units, unconstrained-H vacuum 2GB1 at `gamma=10` ps⁻¹ holds 1000 steps only for **dt ≤ 0.1 fs**; at **dt=0.5 fs** need **gamma ≥ 50** ps⁻¹ (sweep 2026-07-09).

## Locked decisions

| Topic | Lock |
|-------|------|
| `EnsemblePlan.run(dt=)` | **Femtoseconds.** Convert via `dt_fs_to_akma`. |
| `EnsemblePlan.run(gamma=)` | **ps⁻¹.** Convert via `gamma_ps_to_akma`. Default `10` (water NVT). |
| Escape hatch | `dt_unit="akma"` for pre-converted `dt`. |
| Vacuum policy (unconstrained H) | **dt ≤ 0.5 fs and gamma ≥ 50 ps⁻¹**, or **dt ≤ 0.1 fs at gamma=10**. 1.0 fs not claimed. |
| Water NVT | Unchanged: dt ≤ 1.0 fs at production scale with gamma ≈ 10. |

## Acceptance Criteria
1. Given 2GB1 from `_b1_paramize` (A1+A2), when `EnsemblePlan.run(n_steps=1000, dt=0.5, gamma=50, ...)`, then all positions finite.
2. Given docs/`EnsemblePlan.run` docstring, when read, then `dt` is femtoseconds and `gamma` is ps⁻¹ (AKMA conversion internal).
3. Given `dt_unit="akma"`, when `dt=dt_fs_to_akma(0.5)` is passed, then behavior matches default fs path (parity smoke).

## Non-goals
Claim-1 throughput; OpenMM protein parity numerics (PROT); H-bond SHAKE (stretch — may raise vacuum dt / lower gamma later).

## References
- `prolix.physics.kups_adapter.dt_fs_to_akma` / `gamma_ps_to_akma`
- `tests/physics/test_settle_temperature_control.py` (canonical conversion)
- Research: `.praxia/docs/research/260709_xr-a2a3-a3-vacuum-dt.md`
- Gate: `tests/api/test_xr_vacuum_dt.py`
