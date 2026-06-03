# Phase 5 R-step liquid-density NVT: root-cause chain

Date: 2026-06-03 · Branch: `worktree-p5-rstep-liquid-density` · Campaign ef45b8b4

## Summary

The 895-water NVT gate (`test_nvt_216water_temperature_stability`) failed on the
cluster with `mean T = 7.09e53 K`. Investigation found **two independent bugs**
(plus four oracle-script bugs). Both are now fixed: Bug 1 (PBC min-image in the
R-step impulse) and Bug 2 (the genuine Phase 5 deliverable — `_settle_water_batch`
discarding rigid-body rotation, fixed via a mass-weighted Horn quaternion rigid
fit). The 895-water cluster gate is the remaining end-to-end confirmation.

## Bug 1 — PBC minimum-image in the R-step impulse (FIXED, commit 8192ea1)

`settle_langevin` apply_fn R-step:

    dp = mass * (x_con - x_unc) / half_dt

used a **raw** coordinate difference. `shift_fn` wraps an atom across the
periodic boundary while `settle_positions` reconstructs the rigid water in the
unwrapped frame of `positions_old`, so `x_con - x_unc` jumps by a full box
vector for any atom starting near/outside the primary cell.

Evidence: the liquid asset has **91/2685 atoms outside [0,30) Å**; substep trace
showed step-1 `dp ≈ mass·30 Å/half_dt` → T = 1.1e9 K **in one step**. NVE
(gamma=0) diverged identically → deterministic integrator, not the thermostat.
The dilute grid asset keeps all atoms in-cell → never wrapped → always passed.
It was never about density.

Fix: minimum-image `(x_con - x_unc)` before forming dp in both R-steps. No-op
for genuine sub-Å corrections → in-cell trajectories unchanged (settle suite +
dilute smoke: 10 passed).

## Bug 2 — `settle_positions` removes rigid-body rotation (FIXED, branch `worktree-p5-bug2-rotation`)

After Bug 1, the gate is **stable** but equilibrates at **158 K, not 300 K**.
KE decomposition (`scripts/explore/p5_transrot_decomp.py`):

    T_trans = 323 K   (3N-3 translational DOF — thermalized)
    T_rot   =   1 K   (3N rotational DOF — frozen)
    T_total = 162 K   (6N-3, what the gate measures)

Per-substep trace (`p5_rot_substep.py`) localizes the loss to the R-step dp
impulse:

    init 283 → after B 286 → after dp_1 58 → after O 57 → after dp_2 12 → end 12

Each dp step cuts T_rot ~5×; T_trans untouched. The OU noise generator is
correct (full 6D rigid subspace, E[KE]=3kT). The dp removes rotation because
`settle_positions`/`_settle_water_batch` reconstructs the rigid water using the
**old-position reference frame**, snapping orientation back toward
`positions_old` instead of only fixing bond lengths.

Decisive test: feed `settle_positions` an **already-rigid** water rotated 5°
about x (zero constraint violation). A correct constraint projection must leave
it untouched. Instead it moved atoms by **0.045 Å ≈ the full 0.051 Å rotation**,
reconstructing at the old orientation. So it is not a constraint projection — it
discards rigid rotation.

### Fix (implemented — mass-weighted Horn quaternion rigid fit)

`_settle_water_batch` was reimplemented as a proper rotation-preserving
constraint: for each water it fits the fixed ideal TIP3P body-frame template to
the COM-centred unconstrained positions and returns `COM_new + R · template`,
where `R` is the mass-weighted least-squares rotation. `positions_old` is now
used **only** for the PBC unwrap, never for orientation. With the rigid
configuration carrying the molecule's net rotation, `dp = m(x_con−x_unc)/half_dt`
removes only the radial bond-stretch velocity and preserves angular momentum —
matching OpenMM's LangevinMiddleIntegrator.

**Method choice — Horn (1987) quaternion, not SVD/Kabsch.** TIP3P water is
planar, so the 3×3 cross-covariance is rank-deficient and the SVD's smallest
singular vectors flip sign discontinuously between steps. An initial SVD-Kabsch
implementation passed the single-step geometry unit tests but **detonated
`test_integrator_energy_conservation`** (0.92 → 1.5e7 over 100 steps) because the
discontinuous rotation matrix injected spurious work; a `trace(R)<1.5 → R·(−1)`
"continuity" patch only masked it (and `−R` is a reflection, det = −1, not a
rotation). Horn's largest-eigenvalue quaternion is well-separated and continuous
through the planar case and always yields a proper rotation. Switching to Horn
fixed energy conservation, the PBC-crossing test, and rotation preservation
simultaneously.

**Verification (local):**
- `test_settle_rotation_preserving.py` (5 new tests): already-rigid invariance
  under 5° about-x and 17° generic 3D rotation (≤1e-6 Å), bond/angle restoration
  (≤1e-6), COM preservation (≤1e-9), batched consistency — all pass.
- `test_settle.py` (9 tests incl. energy conservation + PBC crossing) — all pass.
- `p5_rot_substep.py` T_rot trace, **after** the fix (compare the collapse above):

      init 282.9 → after B 286.0 → after dp_1 286.7 → after O 283.0
      → after dp_2 283.0 → after final B 286.3 → after SETTLE_vel 283.2

  T_rot is preserved (≈283–287 K, matched to T_trans ≈299 K) instead of
  collapsing to ~12 K. The 895-water cluster gate (300±5 K) is the remaining
  end-to-end confirmation.

## Oracle (`openmm_oracle_tip3p.py`) — 4 bugs fixed

1. residue `WAT` + no bonds → `HOH` + explicit O–H bonds (template match)
2. set periodic box on the topology before `createSystem` (PME requirement)
3. `getState(getKineticEnergy=)` → `getState(getEnergy=)` (OpenMM 8.x)
4. molar KE (kJ/mol) paired with per-particle `k_B` → use gas constant R

Verified: builds + integrates + measures finite ~300 K on 8 dilute waters.

## Repro scripts (scripts/explore/)

- `p5_rstep_diagnostic.py` — per-step T/PE/E_tot, thermostat + NVE modes
- `p5_rstep_substep_trace.py` — single-step substep magnitudes (Bug 1 localization)
- `p5_transrot_decomp.py` — trans vs rot temperature decomposition (Bug 2)
- `p5_rot_substep.py` — per-substep T_rot (Bug 2 localization)
