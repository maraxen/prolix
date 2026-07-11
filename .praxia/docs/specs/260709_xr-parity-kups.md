---
title: "XR-PARITY-KUPS — kUPS thermostat crossval bathos sidecar"
backlog_id: XR-PARITY-KUPS
epic: 260709_xtrax_rewire
depends_on: [XR-PARITY-OMM-WATER]
priority: P2
difficulty: standard
status: completed
challenge_verdict: pass
challenge_summary: "Sidecar+driver landed; adapter probe bathos pass (991b1851); always-on unit tests green; full T crossval optional when kups installed."
---

# XR-PARITY-KUPS

## Goal
Promote the existing kUPS thermostat cross-validation (Sprint 5 harmonic oscillator) to a bathos `[experiment]` sidecar so rewire regressions can be tracked like XR-PARITY-OMM-WATER — secondary, not on the kill path.

## Locked decisions

| Topic | Lock |
|-------|------|
| Physics | Harmonic oscillator, N=64, k=0.01 eV/Å², T=300 K, free space (same as `test_kups_thermostat_crossval.py`) |
| Integrators | BAOAB (γ=10 ps⁻¹) and CSVR (τ=0.1 ps) |
| Gates (when kups present) | BAOAB: `|T_prolix−T_kups|≤2 K` at dt∈{0.5,1.0} fs; CSVR dt=0.5: ≤2 K; CSVR dt=1.0: ≤10 K (bias-consistency) |
| Always-on gate | `kups_adapter` unit roundtrips (fs↔AKMA, γ, τ, spring eV↔kcal) — no kups package required |
| Driver | `scripts/experiments/xr_parity_kups.py` + `xr_parity_kups.bth.toml` |
| Campaign slug | `xr-parity-kups` |
| Non-goals | Claim-1 throughput; SETTLE water; OpenMM; installing kups into default CI |

## Acceptance Criteria
1. Given `kups_adapter` conversions, when pytest always-on gate runs, then roundtrips hold to 1e-9 relative.
2. Given bathos sidecar next to the driver, when `bth run` executes the adapter probe, then outcomes evaluate with exactly one `is_residual=true`.
3. Given kups installed, when crossval probe runs BAOAB dt=0.5, then `|ΔT|≤2 K` and `gate_pass=1`.
4. Given Claim-1 / kill-path scope, when reviewed, then rejected (secondary leaf only).

## References
- `tests/physics/test_kups_thermostat_crossval.py`
- `src/prolix/physics/kups_adapter.py`
- `.praxia/docs/daily/260426_sprint5-kups-crossval.md`
- Water template: `scripts/experiments/xr_parity_omm_tip3p.bth.toml`
