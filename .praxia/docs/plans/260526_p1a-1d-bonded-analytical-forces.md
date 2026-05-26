---
task_id: 260526_p1a-1d-bonded-analytical-forces
backlog_id: 558
sprint_id: 2
date: 260526
status: planned
parent_task: 260526_p1a-bundle-continuation (id 163, in_progress)
deliverables:
  - src/prolix/physics/analytical_forces.py (extend with 5 functions)
  - tests/physics/test_analytical_forces.py (create or extend)
gates:
  - per_function_parity: jnp.allclose(F_analytical, -jax.grad(E)(x), atol=1e-10) float64
  - padded_zero_force: padded entries (0 indices + 0 params) → 0 force contribution
  - newton_third_law: sum(F_i) ≈ 0 across atoms (sanity check)
estimated_effort_hours: 4
---

# P1a-1d: Expand `analytical_forces.py` to All Bonded Terms

## Context

Phase 1a (id 163, in_progress) is the MolecularBundle bucketed dynamic topology
JIT boundary. Sub-step 1d (custom_jvp analytical force shim) is currently
**partial**: `src/prolix/physics/analytical_forces.py` (292 lines) has nonbonded
analytical forces (LJ + variants) but no bonded analytical force functions.
Completing 1d eliminates autograd overhead in the inference path and lets
`make_energy_fn_from_bundle` route through ShimMode.ANALYTICAL for bonded
energies.

This work is **not blocked on P2a**: bonded analytical forces operate on
indices + params present in both `PhysicsSystem` and `MolecularBundle`. P2a's
field audit affects what fields the Bundle exposes, not what mathematical form
the analytical force functions take.

## Scope (firm)

Five new functions, one per bonded term type, added to
`src/prolix/physics/analytical_forces.py`:

1. `bond_forces_analytical(positions, bond_indices, bond_params, displacement_fn)` — harmonic bond, `U = k*(r - r0)^2`
2. `angle_forces_analytical(positions, angle_indices, angle_params, displacement_fn)` — harmonic angle, `U = 0.5*k*(θ - θ0)^2`
3. `dihedral_forces_analytical(positions, dihedral_indices, dihedral_params, displacement_fn)` — periodic torsion, `U = Σ_t k_t * (1 + cos(n_t*φ - phase_t))`
4. `improper_forces_analytical(positions, improper_indices, improper_params, displacement_fn)` — dispatches on `improper_params.shape[-1]`: 3 → periodic; 2 → harmonic
5. `urey_bradley_forces_analytical(positions, ub_indices, ub_params, displacement_fn)` — 1–3 pair harmonic (same form as bond)

Each function must:
- Be jit-compatible (no Python branches on traced values)
- Return forces in `kcal/mol/Å` matching the energy module's unit convention
- Agree with `-jax.grad(energy_fn)` at machine precision (float64, atol ≤ 1e-10)
- Support arrays of arbitrary length N with padded entries (`0` indices + `0` params → `0` force contribution; project convention)

## Phases

### Phase 1 — Bond + Angle (~1h)

**Math:**
- Bond: `dU/dr = 2k(r - r0)`, force `F = -2k(r-r0) · r̂_ij`, opposite signs on i,j
- Angle: `cos(θ) = (v_ji · v_jk) / (|v_ji||v_jk|)`. Chain-rule through arccos; safe denominator `sin(θ) + 1e-12` for θ → 0, π.

**Subtasks:**
1. Test `bond_forces_analytical` vs `-jax.grad(bond_energy_fn)` on 3-atom system, float64
2. Implement `bond_forces_analytical` (existing `bond_forces` nonbonded variant may serve as a style template, but verify line numbers — planner suggested lines 300-366 but file is 292 lines, so verify before reusing)
3. Test `angle_forces_analytical` vs grad on 4-atom system
4. Implement `angle_forces_analytical`
5. **Gate:** both Phase-1 tests pass at `atol=1e-10`
6. Commit: `feat(analytical_forces): add bond and angle analytical forces`

### Phase 2 — Dihedral + Improper (~1.5h)

**Math:**
- Dihedral angle φ from 4 atoms (i,j,k,l) using vectors b0=ri-rj, b1=rk-rj, b2=rl-rk. φ = atan2(cross/dot of projections of b0,b2 onto plane perpendicular to b1).
- Energy: `U = Σ_t k_t · (1 + cos(n_t·φ − phase_t))`
- Gradient: `dU/dφ = -Σ_t k_t · n_t · sin(n_t·φ − phase_t)`. **Sign of sin** is the #1 failure mode — see Risk table.
- `dφ/dr` via `jax.jacobian(dihedral_angle_fn, argnums=0)` (autodiff the angle, hand-derive `dU/dφ` — hybrid keeps the implementation tractable).

**Subtasks:**
1. Extract/refactor dihedral angle computation to a helper `dihedral_angle(positions, indices, displacement_fn) → (D,)`
2. Test `dihedral_forces_analytical` on 5-atom system with 1 periodic term
3. Implement `dihedral_forces_analytical` (use `jax.vmap` over dihedrals; `jax.jacobian` for `dφ/dr`)
4. Test `improper_forces_analytical` for both branches (periodic 3-param; harmonic 2-param)
5. Implement `improper_forces_analytical` with shape dispatch
6. **Gate:** 3 Phase-2 tests pass at `atol=1e-10`
7. Commit: `feat(analytical_forces): add dihedral and improper analytical forces`

### Phase 3 — Urey-Bradley (~30m)

Trivial: same form as bond, applied to i,k (skip j) of a 3-atom triplet.

**Subtasks:**
1. Test on UB pair
2. Implement (copy bond_forces logic; use `ub_indices[:, 0]` and `ub_indices[:, 1]`)
3. **Gate:** UB test passes
4. Commit: `feat(analytical_forces): add urey-bradley analytical forces`

### Phase 4 — Comprehensive Parity Tests (~1h)

Create or extend `tests/physics/test_analytical_forces.py`:

1. `test_bond_analytical_vs_grad` — 1 bond, 3 atoms, float64
2. `test_angle_analytical_vs_grad` — 1 angle, 4 atoms, float64
3. `test_dihedral_analytical_vs_grad` — 1 dihedral, multi-term, float64
4. `test_improper_periodic_vs_grad` — periodic branch
5. `test_improper_harmonic_vs_grad` — harmonic branch (angle wrapping)
6. `test_urey_bradley_analytical_vs_grad` — 1 UB pair
7. `test_padded_entries_zero_force` — zero-indices + zero-params → zero force
8. `test_mixed_real_padded` — real + padded entries; real match grad, padded zero
9. `test_force_sum_zero` — Newton's 3rd law sanity (sum F_i ≈ 0)

**Gate (final):** Full suite passes; no xfail; no skipped tests.

Commit: `test(analytical_forces): add comprehensive parity tests for all bonded terms`

## Critical files

| Path | Role |
|------|------|
| `src/prolix/physics/analytical_forces.py` | **MODIFY** — add 5 functions |
| `src/prolix/physics/bonded.py` | **READ** — energy forms; extract dihedral angle helper |
| `src/prolix/types/bundles.py` | **READ** — MolecularBundle field contract (context) |
| `tests/physics/test_analytical_forces.py` | **CREATE or EXTEND** — parity test suite |

## Risk table

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Dihedral gradient sign error** (`sin` vs `-sin`) | **High** | Test scalar `dU/dφ` against `jax.grad(dihedral_energy_fn)(positions)` BEFORE full force; inspect sign on a small example first |
| Angle singularity (sin(θ) → 0) | Medium | Safe denominator `sin(θ) + 1e-12`; explicit test case at θ ≈ 0 and θ ≈ π |
| Periodic difference wrapping (improper harmonic) | Medium | `(φ − φ0) mod 2π` then map to [-π, π]; dedicated wrapping test |
| Padded-entry handling (zero indices, zero params) | Low | `jnp.where(mask, ..., 0.0)`; mixed real/padded test case |
| Asymmetric force accumulation (Newton's 3rd law violation) | Low | Sanity check `sum(F_i) ≈ 0` on periodic system |

## Verification commands

```bash
# Per-function parity
uv run pytest tests/physics/test_analytical_forces.py -v --tb=short

# Final gate
uv run pytest tests/physics/test_analytical_forces.py -v 2>&1 | tee tmp/p1a_1d_gate.log
grep -E "^\d+ passed" tmp/p1a_1d_gate.log
```

## Next dispatch

`specification-specialist` to convert this plan into a fixer-executable spec.
Spec output: `.praxia/docs/specs/260526_p1a-1d-bonded-analytical-forces.md`.

**Top concern for spec-spec to resolve before fixer:** confirm whether the
planner's reuse-suggestion (existing `bond_forces` at lines 300-366) is valid
— the file is 292 lines, so the line range is wrong. Spec-spec should re-read
the file and either find a genuine reuse target or write fresh.
