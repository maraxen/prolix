---
title: "XR-A2A3 A3 — vacuum-protein settle-dt root cause"
date: 2026-07-09
leaf: XR-A2A3
status: complete_via_exception
exception_leaf: XR-VACUUM-DT
---

# Vacuum-protein dt root cause (A3)

## Setup
- Fixture: `data/pdb/2GB1.pdb` via `scripts/benchmarks/_b1_paramize.py` (A2 exclusions + A1 masses)
- Force scale after A2: median |grad| ≈ 16 kcal/mol/Å (C1 pass)
- Integrators: `settle_langevin` (EnsemblePlan) and `jax_md.simulate.nvt_langevin`
- C3 target: finite ≥1000 steps at B1-pinned `dt=0.5` fs — closed via **XR-VACUUM-DT** (unit fix + vacuum gamma policy)

## Findings (2026-07-09)

### Early probes (before unit fix — treat as confounded)
Prior table rows that passed `dt=0.5` / `0.01` / `0.025` into EnsemblePlan without `dt_fs_to_akma` were using **~24 fs / ~0.5 fs / ~1.2 fs** true timesteps, not the labeled fs values. FIRE did not unlock the mis-scaled 0.5.

### Unit-corrected envelope (XR-VACUUM-DT)

| gamma (ps⁻¹, converted) | dt (fs, converted) | n_steps | Finite? |
|-------------------------|--------------------|---------|---------|
| 10 | 0.5 | 1000 | **No** (~step 540) |
| 10 | 0.25 | 1000 | **No** (~step 800) |
| 10 | 0.1 | 1000 | **Yes** |
| 10 | 0.05 | 1000 | **Yes** |
| 50 | 0.5 | 1000 | **Yes** |
| 100 | 0.5 | 1000 | **Yes** |
| 10 raw (≈204 ps⁻¹, pre-fix) | 0.5 | 1000 | Yes (overdamped artifact) |

### Interpretation
1. **Primary bug:** EnsemblePlan omitted `dt_fs_to_akma` → callers' `dt=0.5` was ~24 fs.
2. **Secondary bug:** EnsemblePlan passed `gamma=10` without `gamma_ps_to_akma` (~204 ps⁻¹). That overdamping made the false-positive “1000 finite after converting only dt” diagnostic.
3. **SETTLE-only hypothesis remains falsified** for the pre-fix blowups; post-fix vacuum limit is unconstrained H + friction/dt envelope.
4. **Policy:** vacuum unconstrained-H → `dt≤0.5` **and** `gamma≥50`, or `dt≤0.1` at `gamma=10`.

## Policy (delivered by XR-VACUUM-DT)
- `EnsemblePlan.run(dt=)` = femtoseconds; `gamma=` = ps⁻¹; both converted to AKMA internally.
- Water NVT production policy unchanged (gamma≈10, dt≤1.0 at scale).
- PROT unblocked for vacuum protein MD under the vacuum gamma/dt policy above.

## C3 checklist
- [x] Minimization probe
- [x] SETTLE vs jax_md comparison
- [x] Step-1 KE explosion documented (pre-unit-fix)
- [x] Exception leaf XR-VACUUM-DT filed
- [x] XR-VACUUM-DT delivers ratified dt/gamma policy
