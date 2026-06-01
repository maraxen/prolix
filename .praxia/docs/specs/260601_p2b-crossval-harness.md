---
task_id: 260601_p2b_crossval_harness
plan_doc: .praxia/docs/plans/260527_p2b-nonbonded-comparator.md
sprint_id: 3
date: 260601
status: draft
gates:
  nb_lj_energy: "per-term |dE(LJ)| < 1.0 kcal/mol on vacuum ala-dip (float64)"
  nb_coul_energy: "per-term |dE(Coulomb)| < 1.0 kcal/mol on vacuum ala-dip (float64)"
  nb_force: "RMS(|dF_nb via FD|) < 0.5 kcal/mol/Å across all atoms"
  nb_14_self_consistency: "exception_14 prolix self-consistency |dE| < 0.2 kcal/mol"
fixer_tasks:
  - f1-fixture-nonbonded-extension
  - f2-exclusion-bridge
  - f3-prolix-nb-bridge
  - f4-nb-energy-parity-test
  - f5-nb-force-parity-test
  - f6-nb-field-audit-extension
  - f7-pme-stretch-optional
---

# Spec: P2b — OpenMM Parity Harness (Nonbonded: LJ + Coulomb)

## Scope

**This spec covers only the nonbonded energy/force parity leg of Phase 2b.**

Phase 2b per the strategic roadmap (`.praxia/docs/roadmaps/prolix/260515_prolix-strategic-roadmap.md`) requires six test categories:

1. ✅ **Nonbonded energy/force parity** (THIS SPEC) — LJ + Coulomb on vacuum ala-dip
2. ⏳ **NVE energy conservation** — long trajectory with settle_langevin (separate spec)
3. ⏳ **NVT temperature stability** — 216-water box, settle_langevin, ±5 K gate (separate spec)
4. ⏳ **NPT cross-validation** — 216-water box, settle_csvr_npt, ±20 bar / ±5 K (separate spec)
5. ⏳ **ANALYTICAL vs AUTOGRAD** — per-term force agreement for bonded terms (separate spec)
6. ⏳ **kUPS 3-size expansion** — 100/500/2000 atoms with kUPS reference (separate spec)

This spec is the **first slice** — nonbonded parity only. The remaining five categories will be specced in separate documents as Phase 2b progresses. All six are required for full Phase 2b completion; this harness provides one quarter of the cross-validation surface.

---

## Target API

**Primary target: `PhysicsSystem` API**

The fixture builds `PhysicsSystem` objects via `build_prolix_nonbonded_system(...)` and evaluates energy via `make_energy_fn(...)`. This is the current stable API as of v1.0.

**Migration note:** Phase 2b later targets `MolecularBundle` after the P1a MolecularBundle refactor is confirmed stable on main. The parity evidence produced by this spec is valid for both APIs (PhysicsSystem is the composing building block of Bundle). If the test fixture is migrated to MolecularBundle in a future pass, the gate thresholds and decomposition semantics remain unchanged; only the API layer changes. This is intentional and not a regression.

---

## Goal

Extend the P2a bonded parity harness to cover **Lennard-Jones** and
**Coulomb** nonbonded forces on vacuum alanine dipeptide, including correct 1-4
exception handling. This closes the nonbonded leg of the §7.1 external-comparator
figure and establishes a repeatable numerical gate for prolix kernels against OpenMM
Reference-platform energies.

---

## Test System: Vacuum Ala-Dip

Vacuum alanine dipeptide (ACE-ALA-NME), OpenMM `NoCutoff`, prolix `cutoff_distance=0`, `pme_alpha=0.0`.

With `cutoff_distance=0`, the mask guard `if cutoff > 0:` evaluates False, so all pairs are included (equivalent to `NoCutoff`). With `pme_alpha=0.0`, damping term evaluates to zero, giving bare Coulomb. Vacuum pairwise sum is exact—any discrepancy is a coding or parameter bug.

---

## Tolerance Rationale

- **LJ & Coulomb: `|dE| ≤ 1.0 kcal/mol`** — Float64 accumulates to <0.01 kcal/mol vs OpenMM. Gate is intentionally loose (1.0) to catch unit errors, sign errors, or wrong sigma rules while allowing float32 fallback (~0.1 kcal/mol).
- **1-4 exception self-consistency: `|dE| ≤ 0.2 kcal/mol`** — Direct per-pair sum with ~12–16 pairs. Catches factor-of-2 errors.
- **Force RMS: `RMS(|dF_nb|) ≤ 0.5 kcal/mol/Å`** — Finite differences at eps=1e-3 Å introduce ~0.01 kcal/mol/Å noise. Gate allows noise while catching factor-of-2 errors.

---

## Nonbonded Energy Decomposition

OpenMM `NonbondedForce` (ForceGroup 3) is split via two-pass charge-zeroing:

1. Standard run → total nonbonded energy `E_nb`
2. Zero per-atom charges, re-query → LJ-only energy `E_LJ` (includes 1-4 LJ + 1-4 Coulomb from exception params)
3. Compute `E_Coul = E_nb − E_LJ` (1-5+ Coulomb only, since exception Coulomb is in `E_LJ`)
4. Restore charges, reinitialize

This matches prolix's decomposition: `lj_energy_fn_bound` (1-5+) + `exception_energy_fn_bound` (1-4).

---

## Exclusion Handling

`ExclusionSpec` is built from OpenMM `getException(k)` entries:

- **`idx_12_13`**: pairs with `epsilon == 0.0 AND chargeProd == 0.0` (full exclusions)
- **`exception_pairs`**: all other pairs (1-4 modified interactions)
- Max exclusions assertion: `max_excl_per_atom < 32` to fit within `max_exclusions=32`

---

## Fixer Task Decomposition

### f1-fixture-nonbonded-extension

Add three functions to `tests/physics/fixtures_openmm_parity.py`:

1. `extract_nonbonded_params(omm_system)` — extract per-atom charges/sigmas/epsilons and exception pairs from OpenMM `NonbondedForce`
2. `get_openmm_nonbonded_energies(omm_system, positions_ang)` — two-pass charge-zeroing split
3. `get_openmm_nonbonded_forces(omm_system, positions_ang)` — query ForceGroup 3 forces, convert units

**Success**: `extract_nonbonded_params` returns dict with `charges.shape == (N,)`, `N ≥ 31`, and `exception_pairs.shape[0] > 0`. `get_openmm_nonbonded_energies` returns finite energies with `abs(lj + coulomb - total_nb) < 1e-4`.

**Effort:** 45 min. **Deps:** none.

---

### f2-exclusion-bridge

Add `build_exclusion_spec(omm_system, n_atoms) -> ExclusionSpec` to classify OpenMM exceptions into `ExclusionSpec` format with unit conversion (nm→Å, kJ/mol→kcal/mol).

**Verification includes import check**: `from prolix.physics.neighbor_list import ExclusionSpec` to confirm P1a refactor didn't move the symbol.

**Success**: Returns `ExclusionSpec` with `idx_12_13.shape[0] > 0` and `exception_pairs.shape[0]` matching f1.

**Effort:** 30 min. **Deps:** f1.

---

### f3-prolix-nb-bridge

Add two functions:

1. `build_prolix_nonbonded_system(nb_params, bonded_params, positions_ang) -> (PhysicsSystem, displacement_fn)` — reuse `build_prolix_bonded_system` and replace nonbonded fields
2. `get_prolix_nonbonded_energies(system, displacement_fn, positions_ang, exclusion_spec)` — call `make_energy_fn` with `cutoff_distance=0, pme_alpha=0.0, exclusion_spec=...`

**Success**: Returns dict with finite `lj`, `coulomb`, `exception_14` with `lj < 0`.

**Effort:** 45 min. **Deps:** f1, f2.

---

### f4-nb-energy-parity-test

Create `tests/physics/test_openmm_parity_nonbonded.py` with:
- `nb_parity_bundle` fixture (module scope) collecting all prior functions
- `test_lj_energy_parity()` — assert `|prolix_lj - omm_lj| < 1.0`
- `test_coulomb_energy_parity()` — assert `|prolix_coul - omm_coul| < 1.0`
- `test_exception_14_energy_parity()` — self-consistency via `make_exception_pair_energy_fn`

**Success**: 3/3 tests pass at stated gates.

**Effort:** 40 min. **Deps:** f1, f2, f3.

---

### f5-nb-force-parity-test

Add `test_nb_force_parity(nb_parity_bundle)` to existing test file with mandatory comment block:

```python
# NOTE: jax.grad(chunked_lj_energy) returns zeros due to custom VJP stub.
# (src/prolix/physics/optimization.py:68-71 — known limitation.)
# Forces are computed here via central finite differences (eps=1e-3 Å).
# Do NOT replace with jax.grad — it will trivially pass at zero force everywhere.
```

Compare FD forces against OpenMM ForceGroup 3 forces. Assert `RMS < 0.5`.

**Success**: Test passes with `RMS < 0.5 kcal/mol/Å`.

**Effort:** 35 min. **Deps:** f4.

---

### f6-nb-field-audit-extension

Extend `.praxia/docs/audits/260526_p2a-bonded-field-audit.md` with:
- New section documenting nonbonded fields: `charges`, `sigmas`, `epsilons`, `excl_indices`, `excl_scales_vdw`, `excl_scales_elec`
- Update "NOT exercised" table to remove now-covered fields
- Update `.praxia/docs/INDEX.md` audit entry

**Success**: Audit doc contains all required field names.

**Effort:** 20 min. **Deps:** f3.

---

### f7-pme-stretch-optional

**Ship only if f1–f6 all PASS.**

Add `build_ala_dip_periodic_openmm_system()` and `test_pme_coulomb_energy_parity()` for PME on 30 Å box with `pme_alpha ≈ 0.292 Å⁻¹` and 32 grid points.

**Tolerance**: `|dE| < 2.0 kcal/mol`.

**Effort:** 2–4 h. **Deps:** f1–f6 green.

---

## Verification Gates

**MVP:**
- Energy parity: 3/3 tests pass
- Force parity: 1/1 test passes
- P2a regression: 5/5 tests pass (no breakage)
- Audit gate: required field names present
- No new test failures in full suite

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `chunked_lj_energy` VJP stub returns zeros | Spec mandates FD; required comment block in test |
| Charge-zeroing corrupts OpenMM context | Fixture restores charges; verification checks `lj + coulomb == total_nb` |
| Unit conversion bugs (nm vs Å) | Dual conversion in f2 and f1; exception test catches ×10 error |
| Slot overflow (>32 exclusions/atom) | Assertion in f2 |
| Exception pairs double-counted | `ExclusionSpec` design ensures 1-4 pairs only in `exception_pairs`, not `idx_12_13` |
| PME alpha mismatch (f7) | Derive explicitly; log both prolix & OpenMM values |

---

## Files Touched

| File | Action | Task |
|---|---|---|
| `tests/physics/fixtures_openmm_parity.py` | extend (~200 lines) | f1, f2, f3 |
| `tests/physics/test_openmm_parity_nonbonded.py` | new (~180 lines) | f4, f5 |
| `.praxia/docs/audits/260526_p2a-bonded-field-audit.md` | extend | f6 |
| `.praxia/docs/INDEX.md` | update | f6 |

All `src/` files, existing test files, and other docs are untouched.

