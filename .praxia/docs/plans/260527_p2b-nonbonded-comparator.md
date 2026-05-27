---
task_id: 260527_p2b-nonbonded-comparator
spec_doc: .praxia/docs/specs/260527_p2b-nonbonded-comparator.md
sprint_id: 3
date: 260527
status: draft
estimated_effort_hours: 7
---

# Implementation Plan: P2b — OpenMM Parity Harness (Nonbonded: LJ + Coulomb)

## Overview

Extend the P2a bonded parity harness to cover Lennard-Jones and Coulomb nonbonded forces on vacuum alanine dipeptide, including 1-4 exception handling. This closes the nonbonded leg of the §7.1 external-comparator figure and establishes numerical gates for `chunked_lj_energy` + `chunked_coulomb_energy` against OpenMM Reference-platform energies.

## Success Criteria

- **f1–f3**: Fixture functions produce finite energies; OpenMM charge-zeroing split self-consistent (`lj + coulomb ≈ total_nb` to <1e-4).
- **f4**: 3/3 energy parity tests pass at gates `|dE(LJ)| < 1.0 kcal/mol`, `|dE(Coul)| < 1.0 kcal/mol`, `|dE(exc14)| < 0.2 kcal/mol`.
- **f5**: Force RMS via finite differences passes `< 0.5 kcal/mol/Å`.
- **f6**: Audit doc extension complete; P2a regression stays 5/5 PASS.
- **f7** (stretch): PME test passes if f1–f6 green on first auditor pass.

## Phases

### Phase 1: Fixture Setup (f1–f3) (~2h)

Build OpenMM nonbonded extraction and prolix bridge.

**1.1 f1-fixture-nonbonded-extension** — Add three functions to `tests/physics/fixtures_openmm_parity.py` (append after line 386):
- `extract_nonbonded_params(omm_system)` → charges (Å), sigmas (Å), epsilons (kcal/mol), exception data
- `get_openmm_nonbonded_energies(omm_system, positions_ang)` → two-pass charge-zeroing (E_nb, E_LJ via zeroed charges, E_Coul by subtraction). **Docstring MUST state** that exception entries are NOT affected by `setParticleParameters`, so `E_LJ` includes 1-4 LJ + 1-4 Coulomb, and `E_Coul` is 1-5+-only — matching prolix's `chunked_coulomb_energy` semantics where 1-4 is routed through `exception_chargeprods`. (plan-auditor required fix)
- `get_openmm_nonbonded_forces(omm_system, positions_ang)` → ForceGroup 3 forces in kcal/mol/Å

Files: `tests/physics/fixtures_openmm_parity.py` (+85 lines). Verification: inline smoke test from spec (f1 section) checking charges shape, exceptions present, split consistency <1e-4.

**Deps:** none. **Effort:** 45 min.

**1.2 f2-exclusion-bridge** — Add `build_exclusion_spec(omm_system, n_atoms)` to fixture:
- Classify `getException(k)` by epsilon & chargeProd: zero both → idx_12_13; nonzero → exception_pairs
- Convert units (nm→Å for sigma, kJ/mol→kcal/mol for epsilon)
- Assert max exclusions per atom <32
- Return `ExclusionSpec(idx_12_13=..., idx_14=zeros((0,2)), exception_pairs=..., exception_sigmas=..., exception_epsilons=..., exception_chargeprods=...)`

Files: `tests/physics/fixtures_openmm_parity.py` (+45 lines). Verification: assert idx_12_13 > 0; `map_exclusions_to_dense_padded(spec)` produces (N, 32) shape.

**Deps:** f1. **Effort:** 30 min.

**1.3 f3-prolix-nb-bridge** — Add two functions:
- `build_prolix_nonbonded_system(nb_params, bonded_params, positions_ang)` → PhysicsSystem with charges, sigmas, epsilons; reuse bonded sub-structure via `build_prolix_bonded_system` + `replace`; NO dihedral shape duplication
- `get_prolix_nonbonded_energies(system, displacement_fn, positions_ang, exclusion_spec)` → calls `make_energy_fn(cutoff_distance=0, pme_alpha=0.0, exclusion_spec=..., return_decomposed=True)`; returns dict with `lj`, `coulomb`, `exception_14`

Files: `tests/physics/fixtures_openmm_parity.py` (+65 lines). Verification: all three energy terms finite; lj < 0 (typical).

**Deps:** f1, f2. **Effort:** 45 min.

**Phase 1 Verification**: Inline smoke test from spec f3 section checks finite energies + expected sign.

---

### Phase 2: Energy Parity Gates (f4) (~45 min)

Create `tests/physics/test_openmm_parity_nonbonded.py` with energy tests.

**2.1 f4-nb-energy-parity-test** — Create new test file:
- Module-level: `jax.config.update("jax_enable_x64", True)` and `pytestmark = pytest.mark.openmm`
- Fixture `nb_parity_bundle(scope="module")` bundles both OpenMM and prolix evaluations + positions
- `test_lj_energy_parity`: assert `abs(prolix_lj - omm_lj) < 1.0`
- `test_coulomb_energy_parity`: assert `abs(prolix_coul - omm_coul) < 1.0`
- `test_exception_14_energy_parity`: self-consistency check (direct eval vs composed) `< 0.2 kcal/mol`. Print both values + delta.

Each test prints `prolix=X, omm=Y, delta=Z`.

Files: `tests/physics/test_openmm_parity_nonbonded.py` (new, ~130 lines). Verification: 3/3 tests pass.

**Deps:** f1, f2, f3. **Effort:** 40 min.

**Phase 2 Verification**: `uv run pytest tests/physics/test_openmm_parity_nonbonded.py::test_lj_energy_parity tests/physics/test_openmm_parity_nonbonded.py::test_coulomb_energy_parity tests/physics/test_openmm_parity_nonbonded.py::test_exception_14_energy_parity -m openmm -v` → 3 PASS.

---

### Phase 3: Force Parity Gate (f5) (~40 min)

Finite-difference force validation.

**3.1 f5-nb-force-parity-test** — Add `test_nb_force_parity` to test file from Phase 2:
- **Required comment block** immediately before FD loop (verbatim from spec) explaining custom VJP stub in `optimization.py:68-71` returns zeros — FD is mandatory, not jax.grad
- Central FD loop: `eps=1e-3 Å`, compute `(E(r+ε) - E(r-ε)) / 2ε` for each atom/axis
- Compute total nb energy scalar: `lj_fn(r) + coul_fn(r) + exc14_fn(r)`
- Compare RMS against OpenMM forces: assert `rms < 0.5`, print rms + max |delta|

Files: `tests/physics/test_openmm_parity_nonbonded.py` (extend). Verification: `test_nb_force_parity` passes at RMS < 0.5.

**Deps:** f4. **Effort:** 35 min.

**Phase 3 Verification**: `uv run pytest tests/physics/test_openmm_parity_nonbonded.py::test_nb_force_parity -m openmm -v --tb=long` → 1 PASS; grep log for RMS value.

---

### Phase 4: Audit & Documentation (f6) (~25 min)

Extend P2a field-audit doc to cover nonbonded fields.

**4.1 f6-nb-field-audit-extension** — Extend `.praxia/docs/audits/260526_p2a-bonded-field-audit.md`:
- Run `audit_bonded_fields` on `build_prolix_nonbonded_system` output (proxy intercepts `charges`, `sigmas`, `epsilons`, `excl_indices`)
- Append new section "## Nonbonded fields exercised by P2b path" with table: Field | Role (charges, sigmas, epsilons, excl_indices, excl_scales_vdw, excl_scales_elec)
- Update "## Fields NOT exercised" table: remove charges, sigmas, epsilons (now covered by P2b)
- Update `.praxia/docs/INDEX.md` audit entry: append `; extended for P2b nonbonded fields (260527)` to description

Files: `.praxia/docs/audits/260526_p2a-bonded-field-audit.md` (extend), `.praxia/docs/INDEX.md` (update). Verification: audit doc contains "Nonbonded fields exercised" + all 6 field names.

**Deps:** f3. **Effort:** 20 min.

**Phase 4 Verification**: Inline check from spec f6 section verifies audit extension present. **Then (plan-auditor required fix) re-run the P2a regression gate as the final f6 step:** `uv run pytest tests/physics/test_openmm_parity_bonded.py -m openmm -v` must still return 5/5 PASS. This catches any accidental edit to test infrastructure during the audit-doc extension before the auditor pass.

---

### Phase 5: Stretch — PME Nonbonded (f7 OPTIONAL) (~2–4h)

Only dispatch if f1–f6 all PASS on first auditor pass.

**5.1 f7-pme-stretch-optional** — Add periodic-box PME test:
- Add `build_ala_dip_periodic_openmm_system(box_side_ang=30.0)` to fixture: `nonbondedMethod=PME`, `cutoff=9.0 Å`, `ewaldErrorTolerance=5e-4`
- Derive `pme_alpha = sqrt(-log(2 * 5e-4)) / 9.0 ≈ 0.334 Å⁻¹`
- Pass `pme_alpha=alpha, pme_grid_points=32` to prolix `make_energy_fn`
- Add `test_pme_coulomb_energy_parity` with tolerance `|dE| < 2.0 kcal/mol`. Print alpha + grid count.

Files: `tests/physics/fixtures_openmm_parity.py` (extend), `tests/physics/test_openmm_parity_nonbonded.py` (extend). Verification: `test_pme_coulomb_energy_parity` passes.

**Deps:** f1–f6 green. **Effort:** 2–4h.

---

## Pre-Flight Gates (before any fixer dispatch)

1. **Git state clean**: `git status --short` must show no `??` or `M` lines in source/test dirs
2. **Current main HEAD ref**: note SHA before dispatch
3. **OpenMM installed**: `uv run python -c "import openmm; print(openmm.__version__)"`
4. **P2a regression baseline**: `uv run pytest tests/physics/test_openmm_parity_bonded.py -m openmm -v` must be 5/5 PASS
5. **Fixture helper present**: `uv run python -c "import sys; sys.path.insert(0, 'tests/physics'); from fixtures_openmm_parity import build_ala_dip_openmm_system; print('ok')"`

---

## Reviewer & Auditor Checkpoints

**After f5 (Phase 3 complete):**
- Reviewer checks energy + force tests for correctness and clarity (comment blocks, printouts)
- Code review: gate-assertion logic, FD loop correctness, no floating-point surprises
- Auditor verifies 3 energy tests + 1 force test all PASS at stated thresholds; no regressions on P2a

**After f6 (Phase 4 complete):**
- Auditor reads extended audit doc; confirms it accurately reflects nonbonded field reads via prolix energy path
- Auditor confirms INDEX.md updated correctly
- Final gate: P2a still 5/5 PASS; all "not slow" tests still pass

**Before f7 dispatch (if chosen):**
- Confirm f1–f6 have landed cleanly on main (no outstanding fixups)
- Get explicit user sign-off on f7 scope (PME is >2h; if schedule tight, defer)

---

## Risks & Rollback

| Risk | Level | Rollback Path | Notes |
|---|---|---|---|
| `chunked_lj_energy` VJP returns zeros | HIGH | Mandatory FD in f5; comment block required | Known limitation; verified in spec |
| Charge-zeroing leaves OpenMM context corrupted | medium | Fixture restores charges + reinitialize; f1 smoke test checks split consistency | Gate on lj+coulomb==total_nb |
| Exception sigma unit bug (nm vs Å) | medium | Both `extract_nonbonded_params` and `build_exclusion_spec` convert nm→Å; exception parity test gates on ×10 error | Gate on test_exception_14 |
| excl_indices overflow (>32 per atom) | low | Assert `max_excl_per_atom < 32` in f2; ala-dip guaranteed ≤8 neighbors | Ala-dip fixture guarantees this |
| cutoff_distance=0 mask guard behavior | medium | Verified in spec: Python `0.0 > 0` is False, disables mask (correct) | Smoke test in f3 confirms |
| Exception double-count (1-4 in both idx_12_13 and exception_pairs) | medium | ExclusionSpec design: 1-4 pairs go only into exception_pairs; verify set intersection is empty | Spec design prevents this |
| PME alpha mismatch (f7 only) | high | Derive explicitly; log both prolix + OpenMM alpha in test | f7 comment block includes derivation |
| **Rollback condition:** Any energy gate >5× tolerance → HALT; commit snapshot; investigate | N/A | `git reset --hard <last-good-sha>` + escalate | Energy gate is firm gate |
| **Continue condition:** Energy gate 1.5–5× tolerance → continue as known defect; file for v1.1 | N/A | Commit with audit note explaining overage + cause | Auditor decision only |

---

## Estimated Effort per Phase

| Phase | Tasks | Hours | Notes |
|---|---|---|---|
| 1. Fixture Setup | f1, f2, f3 | 2.0 | 45+30+45 min; three tight builds on shared fixture file |
| 2. Energy Tests | f4 | 0.75 | 40 min; creates new test file; 3 tests straightforward |
| 3. Force Tests | f5 | 0.6 | 35 min; FD loop + comment block (required); comparisons |
| 4. Audit & Docs | f6 | 0.35 | 20 min; mechanical extension of existing audit doc |
| 5. PME Stretch | f7 | 2.5 | 2–4h; only if MVP passes first audit; lower priority |
| **MVP Total (f1–f6)** | — | **3.7h** | Can land in one session |
| **Full Sprint (f1–f7)** | — | **6.2h** | If f7 approved post-audit |

---

## Handoff at Sprint Close

After all PASS/FAIL gates are resolved, orchestrator will write:
- `.praxia/docs/handoffs/260527_p2b-sprint-handoff.md` — summary of what landed, any known defects, next-session tasks (e.g., f7 if deferred)

---

## Files to Create/Modify/Delete

| Path | Kind | Scope | Notes |
|---|---|---|---|
| `tests/physics/fixtures_openmm_parity.py` | extend | +195 lines (f1+f2+f3) | append only; no existing changes |
| `tests/physics/test_openmm_parity_nonbonded.py` | create | ~180 lines | new file; f4+f5 |
| `.praxia/docs/audits/260526_p2a-bonded-field-audit.md` | extend | +~30 lines | add nonbonded section; no deletions |
| `.praxia/docs/INDEX.md` | modify | 1 line | update audit entry description |

All source files (`src/`), all existing test files, and all prior spec/plan docs are untouched.

---

## Phase Dependencies (DAG)

```
f1 → f2 → f3 → {f4, f5} → f6
         ↓            ↓
       [energy]    [force]
         ↓            ↓
         └──→ gate→ f7 (optional)
```

- **f1, f2, f3 must serialize** (f2 needs f1's funcs; f3 needs both)
- **f4 and f5 can run in parallel** after f3 complete (same fixture bundle)
- **f6 can run in parallel** after f3 (only reads f3's system output; no conflict)
- **f7 gates on f1–f6 all PASS** on first auditor pass; optional dispatch decision

---

## Key Architectural Decisions (Locked; do NOT revisit)

1. **Vacuum ala-dip, NoCutoff mode** — no PME approx error; pairwise sum exact → any discrepancy is a bug, not noise
2. **Two-pass charge-zeroing for LJ/Coulomb split** — OpenMM canonical technique; avoids re-registering forces
3. **Finite differences for f5 forces** — `jax.grad` returns zeros due to custom VJP stub; FD is mandatory
4. **1-4 pairs via explicit exception path** — not global scale factors; OpenMM exception list is the source of truth
5. **Tolerance gates are FIRM** — 1.0 kcal/mol for energy, 0.5 for force, 0.2 for exceptions; auditor decides if observed delta overages warrant halt or continue-as-defect
