---
task_id: 260526_p1a-1d-bonded-analytical-forces
plan_doc: .praxia/docs/plans/260526_p1a-1d-bonded-analytical-forces.md
sprint_id: 2
backlog_id: 558
date: 260526
status: specified
gates:
  per_function_parity: "jnp.allclose(F_analytical, -jax.grad(E)(x), atol=1e-10) float64"
  padded_zero_force: "padded entries (0 indices + 0 params) → 0 force contribution"
  newton_third_law: "sum(F_i) ≈ 0 across atoms (sanity check)"
fixer_tasks:
  - f1_bond_angle_impl
  - f2_dihedral_angle_helper
  - f3_dihedral_forces_impl
  - f4_improper_forces_impl
  - f5_urey_bradley_impl
  - f6_test_suite
---

# Spec: P1a-1d Bonded Analytical Forces

## Summary

Add five analytical force functions to `src/prolix/physics/analytical_forces.py`:
`bond_forces_analytical`, `angle_forces_analytical`, `dihedral_forces_analytical`,
`improper_forces_analytical`, `urey_bradley_forces_analytical`. Each computes
`F = -dU/dr` analytically (not via autograd), agrees with `-jax.grad(E)` at
`atol=1e-10` float64, and zeroes padded entries. Eliminates autograd overhead in
the ShimMode.ANALYTICAL inference path. New test file
`tests/physics/test_analytical_forces.py` is created.

---

## Cross-Checks Against Existing Code

### What is actually in `analytical_forces.py` (lines 1–292)

Three functions, all nonbonded:
- **`lj_forces_dense`** (23–106): pairwise LJ with soft-core
- **`coulomb_forces_dense`** (109–164): dense Coulomb
- **`gb_ace_forces_dense`** (167–292): GB+ACE via decomposed VJP

No bonded code anywhere. Planner's "lines 300-366" claim was wrong — file ends at 292.

### Bonded energy precursor code

Lives in two places (confirmed read):
1. **`src/prolix/physics/bonded.py`** (135 lines): `make_bond_energy_fn` (31-55), `make_angle_energy_fn` (58-92), `make_dihedral_energy_fn` (95-134). Improper energy delegates to `make_dihedral_energy_fn` (see `system.py:33`); urey-bradley delegates to `make_bond_energy_fn` (see `system.py:35`). No separate `make_improper_energy_fn` or `make_urey_bradley_energy_fn`.
2. **`src/prolix/batched_energy.py`** (`_bond_energy_masked` 30-43, `_angle_energy_masked` 45-67, `_dihedral_energy_masked` 69-103). Dihedral angle computation fully inlined here using `b0/b1/b2` + `phi = atan2(y, x) - pi`.

### `tests/physics/test_analytical_forces.py`

**Does not exist.** Created from scratch. Style template: `tests/physics/test_bonded.py`.

---

## Energy Form Table

| Term | Energy fn ref | File:lines | Params shape |
|---|---|---|---|
| bond | `_bond_energy_masked` / `make_bond_energy_fn` | `batched_energy.py:30-43` / `bonded.py:31-55` | `(N, 2)` — `[r0, k]` |
| angle | `_angle_energy_masked` / `make_angle_energy_fn` | `batched_energy.py:45-67` / `bonded.py:58-92` | `(N, 2)` — `[theta0, k]` |
| dihedral | `_dihedral_energy_masked` / `make_dihedral_energy_fn` | `batched_energy.py:69-103` / `bonded.py:95-134` | `(N, 3)` — `[periodicity, phase, k]` |
| improper (periodic) | delegates to dihedral (`system.py:33`) | `batched_energy.py:69-103` | `(N, 3)` — `[periodicity, phase, k]` |
| improper (harmonic) | **NOT in codebase** — derive fresh | n/a | `(N, 2)` — `[phi0, k]` |
| urey_bradley | delegates to bond (`system.py:35`) | `batched_energy.py:30-43` | `(N, 2)` — `[r0, k]` |

**Dihedral angle sign convention (canonical):** `phi = atan2(y, x) - pi` where
`y = dot(cross(b1_unit, v), w)`, `x = dot(v, w)`, with `b0 = r_j - r_i`,
`b1 = r_k - r_j`, `b2 = r_l - r_k`. Reproduce exactly from `batched_energy.py:77-96`.

---

## Fixer Task Decomposition

### f1_bond_angle_impl (45 min, deps: none)
**Goal:** `bond_forces_analytical` + `angle_forces_analytical` in `analytical_forces.py` (append after line 292).

**Math (bond):** `U = 0.5*k*(r-r0)^2`; `F_i = -k*(r-r0) * (r_i - r_j)/r`, opposite on j.
**Math (angle):** `U = 0.5*k*(θ-θ0)^2`; chain through `arccos`; safe `sin(θ) + 1e-12`.

**Signatures:**
```python
def bond_forces_analytical(positions, bond_indices, bond_params, displacement_fn, bond_mask=None) -> Array  # (N, 3)
def angle_forces_analytical(positions, angle_indices, angle_params, displacement_fn, angle_mask=None) -> Array
```

**Acceptance:**
```bash
uv run pytest tests/physics/test_analytical_forces.py::test_bond_analytical_vs_grad tests/physics/test_analytical_forces.py::test_angle_analytical_vs_grad -v --tb=short
```

### f2_dihedral_angle_helper (25 min, deps: none — parallel-safe with f1, f5)
**Goal:** Extract `_dihedral_angle_batched(positions, indices, displacement_fn, mask=None) -> (D,)` into `analytical_forces.py`. Match `phi = atan2(y,x) - pi` convention exactly. Include `overlap_mask` NaN-guard from `batched_energy.py:91-93`.

**Do NOT modify `bonded.py` or `batched_energy.py`** — copy convention, don't couple.

**Acceptance:**
```bash
uv run pytest tests/physics/test_analytical_forces.py::test_dihedral_angle_helper_vs_bonded -v --tb=short
```

### f3_dihedral_forces_impl (50 min, deps: f2)
**Goal:** `dihedral_forces_analytical` via hybrid: `jax.jacobian(_dihedral_angle_batched, argnums=0)` for `dφ/dr`, hand-derived `dU/dφ = -Σ_t k_t * n_t * sin(n_t*φ - phase_t)` for the scalar.

**Strategy:** vmap over dihedrals; per-dihedral, jacobian on 4-atom sub-positions; scatter via `segment_sum` (NOT `jnp.add.at`).

**Signature:**
```python
def dihedral_forces_analytical(positions, dihedral_indices, dihedral_params, displacement_fn, dihedral_mask=None) -> Array
```

**Acceptance:**
```bash
uv run pytest tests/physics/test_analytical_forces.py::test_dihedral_analytical_vs_grad -v --tb=short
```

### f4_improper_forces_impl (35 min, deps: f2, f3)
**Goal:** `improper_forces_analytical` with shape dispatch (`params.shape[-1] == 3` → periodic reuse; `== 2` → harmonic).

**Harmonic math:** `U = 0.5*k*(φ-φ0)^2` with angle wrapping `delta = delta - 2π * round(delta/(2π))` before squaring.

Two internal helpers: `_improper_forces_periodic` (delegates to dihedral) + `_improper_forces_harmonic`.

**Acceptance:**
```bash
uv run pytest tests/physics/test_analytical_forces.py::test_improper_periodic_vs_grad tests/physics/test_analytical_forces.py::test_improper_harmonic_vs_grad -v --tb=short
```

### f5_urey_bradley_impl (15 min, deps: f1)
**Goal:** `urey_bradley_forces_analytical` — identical math to bond on `ub_indices[:, 0]` and `ub_indices[:, 1]` (skipping j).

**Note:** `batched_energy.py:379` confirms UB uses `_bond_energy_masked` with `[r0, k]` layout — exact match to bond.

**Acceptance:**
```bash
uv run pytest tests/physics/test_analytical_forces.py::test_urey_bradley_analytical_vs_grad -v --tb=short
```

### f6_test_suite (45 min, deps: f1, f3, f4, f5)
**Goal:** Create `tests/physics/test_analytical_forces.py` with 9 tests:

1. `test_bond_analytical_vs_grad` (float64, atol=1e-10)
2. `test_angle_analytical_vs_grad`
3. `test_dihedral_angle_helper_vs_bonded` (convention check)
4. `test_dihedral_analytical_vs_grad` (multi-term)
5. `test_improper_periodic_vs_grad`
6. `test_improper_harmonic_vs_grad` (include near-π wrapping case)
7. `test_urey_bradley_analytical_vs_grad`
8. `test_padded_entries_zero_force`
9. `test_force_sum_zero` (Newton's 3rd law)

**Style:** `jax.config.update("jax_enable_x64", True)` at module level; `space.free()`; `jnp.allclose` with `atol=1e-10`; no `print` (logging only). **Tests at module level — NOT class-wrapped** (avoid the P2a class/path mismatch issue caught by reviewer).

**Acceptance (final gate):**
```bash
uv run pytest tests/physics/test_analytical_forces.py -v 2>&1 | tee tmp/p1a_1d_gate.log
grep -E "^[0-9]+ passed" tmp/p1a_1d_gate.log
```

---

## Gate Verification Commands

```bash
# Per-function (run after each f-task)
uv run pytest tests/physics/test_analytical_forces.py::test_bond_analytical_vs_grad -v --tb=short
uv run pytest tests/physics/test_analytical_forces.py::test_angle_analytical_vs_grad -v --tb=short
uv run pytest tests/physics/test_analytical_forces.py::test_dihedral_analytical_vs_grad -v --tb=short
uv run pytest tests/physics/test_analytical_forces.py::test_improper_periodic_vs_grad -v --tb=short
uv run pytest tests/physics/test_analytical_forces.py::test_improper_harmonic_vs_grad -v --tb=short
uv run pytest tests/physics/test_analytical_forces.py::test_urey_bradley_analytical_vs_grad -v --tb=short

# Padded-entry zero force
uv run pytest tests/physics/test_analytical_forces.py::test_padded_entries_zero_force -v --tb=short

# Newton's 3rd law
uv run pytest tests/physics/test_analytical_forces.py::test_force_sum_zero -v --tb=short

# Final gate
uv run pytest tests/physics/test_analytical_forces.py -v 2>&1 | tee tmp/p1a_1d_gate.log
grep -E "^[0-9]+ passed" tmp/p1a_1d_gate.log

# Regression
uv run pytest tests/physics/test_bonded.py -v --tb=short
```

---

## Design Decisions

**D1 — Hybrid `jax.jacobian` for dihedral:** RECOMMENDED. Pure analytical chain through `atan2(y,x)-π` is ~40 LOC vector calc with high bug risk; hybrid is correct-by-construction for `dφ/dr` (autograd handles it) and keeps the hand-derived `dU/dφ` performance gain. Use `jax.jacfwd` if compile-time becomes a concern.

**D2 — Extract dihedral helper, DON'T modify `bonded.py`/`batched_energy.py`:** Production energy path stays untouched; helper lives next to its consumer.

**D3 — Test file location:** `tests/physics/test_analytical_forces.py` (new file). Do NOT extend `test_bonded.py` (that tests energy factories, not analytical forces).

**D4 — Use `segment_sum`/explicit `.at[idx].add(...)` for force scatter, NOT `jnp.add.at`:** Both work under JIT, but explicit pattern matches codebase convention and gives correct `jax.grad` behavior.

**D5 — Tests at module level, NOT class-wrapped:** The P2a reviewer caught a spec/path mismatch from class-wrapped tests. Module-level functions match literal gate command paths.

---

## Risk Register

| Risk | Likelihood | Detection | Recovery |
|---|---|---|---|
| **Dihedral grad sign error** (`sin` vs `-sin`) | High | `test_dihedral_analytical_vs_grad` fails with negated forces; check `dU/dφ` at `phi=0, phase=0, n=1` → expect 0 | Negate `sin` term; re-run |
| **Wrong φ convention** (`φ` vs `φ-π`) | High | `test_dihedral_angle_helper_vs_bonded` fails | Copy `phi = atan2(y,x) - pi` exactly from `batched_energy.py:96` |
| Angle singularity (`sin(θ) → 0`) | Medium | Test at `θ ≈ π` (linear); should be finite | `sin(θ) + 1e-12` safe denom |
| Harmonic improper wrapping | Medium | `test_improper_harmonic_vs_grad` fails near `φ0 ≈ ±π` | `delta = delta - 2π*round(delta/(2π))` before square |
| `arctan2(0,0)` NaN in padded dihedrals | Medium | `test_padded_entries_zero_force` returns NaN | `safe_x = jnp.where(degenerate, 1.0, x)` per `batched_energy.py:92-93` |
| `jnp.add.at` under JIT — wrong gradient | Low | Force test passes but grad-of-grad diverges | Use `segment_sum` |
| Newton's 3rd law violation | Low | `test_force_sum_zero` fails | Verify symmetric scatter (i,j with opposite signs) |

---

## Out of Scope

- LJ analytical forces — exist
- Coulomb / PME — deferred
- 1-4 scaled nonbonded — separate
- `custom_jvp` registration — this task adds the forces; JVP wiring is separate
- CMAP analytical forces — deferred
- Modifying `bonded.py` / `batched_energy.py` — read-only references
- Performance benchmarking

---

## First Fixer Task to Dispatch

**`f1_bond_angle_impl`** — no deps, straightforward math, establishes style/mask pattern. After f1 tests pass, dispatch f2 + f5 in parallel.
