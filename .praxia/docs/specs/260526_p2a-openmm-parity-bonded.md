---
task_id: 260526_p2a-openmm-parity-bonded
plan_doc: .praxia/docs/plans/260526_p2a-openmm-parity-bonded.md
sprint_id: 2
backlog_id: 557
date: 260526
status: specified
gates:
  phase_2_energy: "per-term |dE| < 0.05 kcal/mol at fixed conformation"
  phase_3_force: "RMS(|df|) < 0.01 kcal/mol/Å across all atoms"
  phase_4_audit: "field-audit doc lists all bonded-exercised PhysicsSystem fields"
fixer_tasks:
  - f1-openmm-ala-dip-fixture
  - f2-prolix-bonded-bridge
  - f3-energy-parity-test
  - f4-force-parity-test
  - f5-field-audit-module
  - f6-audit-doc
---

# Spec: P2a — OpenMM Parity Harness (Bonded-Only)

## Summary

Builds a per-term energy and force parity harness comparing prolix `PhysicsSystem`
bonded energies against OpenMM Reference-platform energies on solvated alanine
dipeptide. This does not exist anywhere in the repo today — existing OpenMM tests
(`test_openmm_parity.py`, `test_openmm_equivalence.py`) exercise full-FF or PME
paths using 1UAO; none isolate bonded-only terms or use alanine dipeptide. The
field-audit module (`field_audit.py`) is also new and has no precursor.

---

## Cross-checks Against Existing Code

**Existing OpenMM usage:** `tests/physics/` has 38 files importing `openmm`. The
most relevant are `test_openmm_explicit_anchor.py` (PME anchor, 2-charge system),
`test_openmm_equivalence.py` (full-FF 1UAO, `@pytest.mark.slow`), and
`test_solvated_openmm_explicit_parity.py` (two-water electrostatics). None target
bonded-only terms or alanine dipeptide.

**Parameter extraction pattern:** `test_openmm_equivalence.py` already contains a
complete `extract_omm_params()` function (lines 47–90) that parses
`HarmonicBondForce`, `HarmonicAngleForce`, `PeriodicTorsionForce`, and
`NonbondedForce` into NumPy arrays with unit conversions. The new fixture must
implement the same extraction pattern but scoped to bonded forces only, using
ForceGroup isolation to get per-term energies.

**PDB fixtures:** `data/pdb/` contains 1UAO, 1UAO_solvated_tip3p, 1CRN, 1VII,
2JOF, 1UBQ, water boxes. No alanine dipeptide PDB is checked in. The fixture
must build it from sequence via OpenMM `Modeller` (see Task f1).

**PhysicsSystem field names:** Confirmed from `src/prolix/physics/types.py`:
`bonds`/`bond_params`, `angles`/`angle_params`, `dihedrals`/`dihedral_params`,
`impropers`/`improper_params`. The `make_energy_fn` (non-pure variant) bakes
indices and params at construction time; the pure variant (`make_energy_fn_pure`)
takes `DifferentiableParams` at call time. The bonded-only harness uses the
non-pure path for simplicity.

**field_audit:** No existing `field_audit.py` or `FieldAudit` anywhere in
`src/`. No precedent to reuse.

**Test style:** `jax.config.update("jax_enable_x64", True)` at module level;
`try/except ImportError` guard for `openmm` with `HAS_OPENMM = True/False`
flag; `@pytest.mark.openmm` decorator; `scope="module"` fixtures for expensive
OpenMM system construction (pattern from `test_openmm_parity.py` line 19).

---

## Bonded Energy Function Mapping Table

| OpenMM Force class | ForceGroup | prolix function | prolix indices field | prolix params field | param layout |
|---|---|---|---|---|---|
| `HarmonicBondForce` | 0 | `bonded.make_bond_energy_fn` | `bonds` (N,2) int | `bond_params` (N,2) float | `[length_Å, k_kcal/mol/Å²]` |
| `HarmonicAngleForce` | 1 | `bonded.make_angle_energy_fn` | `angles` (N,3) int | `angle_params` (N,2) float | `[theta0_rad, k_kcal/mol/rad²]` |
| `PeriodicTorsionForce` (proper) | 2 | `bonded.make_dihedral_energy_fn` | `dihedrals` (N,4) int | `dihedral_params` (N,3) float | `[periodicity, phase_rad, k_kcal/mol]` |
| `PeriodicTorsionForce` (improper) | 3 | `bonded.make_dihedral_energy_fn` | `impropers` (N,4) int | `improper_params` (N,3) float | `[periodicity, phase_rad, k_kcal/mol]` |

**Critical unit conversions (OpenMM SI → prolix AKMA-like Å-based):**
- Bond length: `nm → Å` (×10)
- Bond spring constant: `kJ/mol/nm² → kcal/mol/Å²` (÷418.4)
- Angle equilibrium: radians (no conversion)
- Angle spring constant: `kJ/mol/rad² → kcal/mol/rad²` (÷4.184)
- Dihedral phase: radians (no conversion)
- Dihedral force constant: `kJ/mol → kcal/mol` (÷4.184)

**ForceGroup isolation strategy:** Before creating the OpenMM Context, assign each
`Force` object to a numbered ForceGroup (0–3 for bond/angle/proper/improper).
Disable all groups except the one being evaluated using
`context.getState(getEnergy=True, groups={group_id})`. This gives per-term
reference energies without reimplementing decomposition.

**Proper vs improper distinction in AMBER14SB:** OpenMM uses a single
`PeriodicTorsionForce` for both proper and improper dihedrals. AMBER14SB marks
impropers by atom ordering convention (the central atom is atom index 2, not 1,
in the 4-atom tuple). For v1, treat all `PeriodicTorsionForce` entries as a single
"torsion" group. The gate is against total torsion energy; per-proper vs
per-improper split is a v1.1 enhancement.

---

## Test Fixture Design

**Conformation source:** Build alanine dipeptide (ACE-ALA-NME capped dipeptide)
from sequence using OpenMM `Modeller` with AMBER residue templates. Create
`Topology` with ACE+ALA+NME residues, call `Modeller.addHydrogens(ff)` where
`ff = ForceField('amber14-all.xml')`, then minimize using `LocalEnergyMinimizer`.
~31 atoms total. No internet required — OpenMM bundles AMBER14SB XML.

**Single fixed conformation:** Use the minimized geometry as the evaluation
conformation. Freeze it as a `np.ndarray` (float64, Å). The same array is passed
to both OpenMM and prolix. No MD steps.

**Seed handling:** No randomness in the fixture (deterministic minimize from
starting geometry).

**Unit conversion layer:** Module-level helpers `_kj_to_kcal(x) = x / 4.184` and
`_nm_to_ang(x) = x * 10.0`. All OpenMM quantities converted to prolix units
(Å, kcal/mol) before comparison.

**Boundary conditions:** `space.free()` displacement (no PBC). OpenMM uses
`NoCutoff` nonbonded method (irrelevant — nonbonded forces in a disabled group).

**Fixture scope:** `@pytest.fixture(scope="module")` for the OpenMM system
construction (~2–3 seconds; avoid rebuilding per-test).

---

## Design Decisions

### Field-access logger placement: decorator/wrapper (non-intrusive)

The field-audit logger lives in `src/prolix/physics/field_audit.py` as a
standalone function `audit_bonded_fields(system, displacement_fn, positions)`
that uses a `FieldAuditProxy` wrapping `PhysicsSystem` without modifying
`make_energy_fn` itself.

**Rationale:** Intrusive instrumentation would require modifying `system.py`,
risking regressions and polluting production code. The decorator approach
intercepts `getattr` via a proxy class, collects accessed attribute names, then
returns the set after running the energy function. Zero production-code changes,
trivially removable post-audit, no JAX tracing cooperation needed (factory-time
reads at Python level).

**Note on jit-compiled access:** Reads inside `jax.jit` after the first trace
(cache hit) are NOT re-triggered. For audit purposes the first trace enumerates
all accessed fields, which is sufficient.

### Fast vs slow test classification

Mark with `@pytest.mark.openmm` only — NOT `@pytest.mark.slow`. OpenMM setup
for ala-dip (~3–5s) is well under `timeout=60`. The `openmm` marker already gates
on optional-dep availability. Adding `slow` would exclude it from the standard
developer loop, defeating the parity gate.

---

## Fixer Task Decomposition

### f1-openmm-ala-dip-fixture

**Goal:** OpenMM reference fixture — topology construction, ForceGroup assignment,
per-group energy extraction, position/energy export as NumPy arrays.

**Files:** `tests/physics/fixtures_openmm_parity.py` (new, ~120 lines)

**Contents:** `build_ala_dip_openmm_system()`, `extract_bonded_params(omm_system)`,
`get_openmm_per_term_energies(omm_system, positions_ang)`,
`get_openmm_forces(omm_system, positions_ang)`,
`@pytest.fixture(scope="module") def ala_dip_reference()`.

**Acceptance:**
```bash
uv run pytest tests/physics/fixtures_openmm_parity.py --collect-only -m openmm
```

**Effort:** 40 min. **Deps:** none.

### f2-prolix-bonded-bridge

**Goal:** Build a minimal `PhysicsSystem` with all nonbonded fields zeroed, plus
per-term prolix bonded energy callables (bypass `make_energy_fn` for isolation).

**Files:** `tests/physics/fixtures_openmm_parity.py` (extend)

**Resolve `make_bond_energy_fn` signature inconsistency:** `system.py` line 27
bakes params at factory time; `bonded.py` takes params at call time. Use the
`bonded.py` direct path (params at call time) for isolation. Verify both paths
produce same result on the test conformation.

**Acceptance:**
```bash
uv run python -c "from tests.physics.fixtures_openmm_parity import build_ala_dip_openmm_system, extract_bonded_params, build_prolix_bonded_system; omm,pos,topo = build_ala_dip_openmm_system(); p = extract_bonded_params(omm); s,d = build_prolix_bonded_system(p,pos); print('bonds:', s.bonds.shape)"
```

**Effort:** 30 min. **Deps:** f1.

### f3-energy-parity-test

**Goal:** Phase 2 gate — assert per-term `|ΔE| < 0.05 kcal/mol`.

**Files:** `tests/physics/test_openmm_parity_bonded.py` (new, ~100 lines)

**Contents:** `jax.config.update("jax_enable_x64", True)`, `pytestmark = pytest.mark.openmm`,
`@pytest.fixture(scope="module") def parity_bundle(ala_dip_reference)`,
three tests: `test_bond_energy_parity`, `test_angle_energy_parity`,
`test_dihedral_energy_parity`. Each prints diagnostic delta.

**Assertion:**
```python
assert abs(prolix_e - omm_e) < 0.05, f"Bond energy mismatch: prolix={prolix_e:.6f}, omm={omm_e:.6f}, delta={abs(prolix_e - omm_e):.6f}"
```

**Acceptance:**
```bash
uv run pytest tests/physics/test_openmm_parity_bonded.py::test_bond_energy_parity tests/physics/test_openmm_parity_bonded.py::test_angle_energy_parity tests/physics/test_openmm_parity_bonded.py::test_dihedral_energy_parity -m openmm -v
```

**Effort:** 30 min. **Deps:** f1, f2.

### f4-force-parity-test

**Goal:** Phase 3 gate — assert `RMS(|Δf|) < 0.01 kcal/mol/Å`.

**Files:** `tests/physics/test_openmm_parity_bonded.py` (extend)

**Contents:** `test_force_parity` using `jax.grad(total_bonded_energy)(positions)`
vs OpenMM forces. Include FD sanity check (`eps=1e-4`) to verify `jax.grad`
matches FD before blaming the comparison.

**Assertion:**
```python
rms = float(np.sqrt(np.mean((np.array(prolix_f) - omm_f)**2)))
assert rms < 0.01, f"Force RMS mismatch: {rms:.4e} kcal/mol/Å (gate: 0.01)"
```

**Acceptance:**
```bash
uv run pytest tests/physics/test_openmm_parity_bonded.py::test_force_parity -m openmm -v
```

**Effort:** 35 min. **Deps:** f3.

### f5-field-audit-module

**Goal:** Non-intrusive `FieldAuditProxy` + `audit_bonded_fields()` entry point.

**Files:** `src/prolix/physics/field_audit.py` (new, ~80 lines)

**Contents:** `class FieldAuditProxy` (wraps PhysicsSystem; records accessed attrs
via `__getattr__`); `def audit_bonded_fields(system, displacement_fn, positions)
-> frozenset[str]`. Module docstring documents the jit-cache-hit limitation.

**Acceptance:**
```bash
uv run python -c "from prolix.physics.field_audit import audit_bonded_fields; print('ok')"
uv run pytest tests/physics/test_openmm_parity_bonded.py::test_field_audit_smoke -m openmm -v
```

**Effort:** 35 min. **Deps:** none.

### f6-audit-doc

**Goal:** Phase 4 deliverable — write the field-audit markdown doc, update INDEX.

**Files:**
- `.praxia/docs/audits/260526_p2a-bonded-field-audit.md` (new)
- `.praxia/docs/INDEX.md` (add audit entry under `## Audits`)

**Contents:** Frontmatter (task_id, date, status), `## Exercised fields` table
(field | role), `## Fields NOT exercised` table (Bundle field | expected
exerciser), `## Notes` (urey_bradley/cmap absence rationale for AMBER14SB).

**Acceptance:**
```bash
uv run python3 -c "
import pathlib
doc = pathlib.Path('.praxia/docs/audits/260526_p2a-bonded-field-audit.md').read_text()
required = ['bonds', 'bond_params', 'angles', 'angle_params', 'dihedrals', 'dihedral_params', 'NOT exercised']
missing = [f for f in required if f not in doc]
assert not missing, f'Audit doc missing: {missing}'
print('Audit doc gate: PASS')
"
```

**Effort:** 20 min. **Deps:** f5, f3.

---

## Gate Verification Commands

### Phase 2 — Energy Gate
```bash
uv run pytest \
  tests/physics/test_openmm_parity_bonded.py::test_bond_energy_parity \
  tests/physics/test_openmm_parity_bonded.py::test_angle_energy_parity \
  tests/physics/test_openmm_parity_bonded.py::test_dihedral_energy_parity \
  -m openmm -v --tb=short 2>&1 | tee tmp/gate_phase2_energy.log
grep "3 passed" tmp/gate_phase2_energy.log
```

### Phase 3 — Force Gate
```bash
uv run pytest tests/physics/test_openmm_parity_bonded.py::test_force_parity \
  -m openmm -v --tb=long 2>&1 | tee tmp/gate_phase3_force.log
grep "1 passed" tmp/gate_phase3_force.log
```

### Phase 4 — Field-Audit Doc Gate
```bash
uv run python -c "from prolix.physics.field_audit import audit_bonded_fields; print('import: ok')"
uv run pytest tests/physics/test_openmm_parity_bonded.py::test_field_audit_smoke -m openmm -v
uv run python3 -c "
import pathlib
doc = pathlib.Path('.praxia/docs/audits/260526_p2a-bonded-field-audit.md').read_text()
required = ['bonds', 'bond_params', 'angles', 'angle_params', 'dihedrals', 'dihedral_params', 'NOT exercised']
missing = [f for f in required if f not in doc]
assert not missing, f'Audit doc missing: {missing}'
print('Audit doc gate: PASS')
"
```

---

## Risk Register

| Risk | Likelihood | Concrete mitigation |
|---|---|---|
| AMBER14SB ala-dip PDB not in repo | resolved | Use `Modeller` with residue templates; no PDB file needed. Fall-back: inline 31-atom PDB string. |
| Unit confusion (nm↔Å, kJ↔kcal, rad/deg) | medium | Add `test_unit_conversion_sanity` test; print raw OpenMM (kJ/mol) alongside converted (kcal/mol). |
| Force RMS gate fails despite energy match | low–medium | FD check: `(E(pos+eps*e) - E(pos-eps*e))/(2*eps)`, `eps=1e-4 Å`. If prolix matches FD but not OpenMM → bug in OpenMM force extraction unit conversion. |
| Proper/improper mixed in PeriodicTorsionForce | medium (AMBER-specific) | v1: treat all torsions as one group; gate against total torsion energy. Split deferred to v1.1. |
| `FieldAuditProxy.__getattr__` misses jit-compiled cache-hit reads | low | First trace catches all reads; subsequent cache hits don't re-trigger. Documented in module docstring. Run audit before any cached jit calls. |
| `make_bond_energy_fn` signature inconsistency (system.py vs bonded.py) | medium | Use `bonded.py` direct path (params at call time). Verify both paths produce same result on test conformation. |

---

## Out of Scope

- LJ, PME/Coulomb, nonbonded exceptions, Urey–Bradley, CMAP (P2a-nonbonded sprint)
- `make_bundle_from_system` (HP2, id 162 — internal Bundle-path counterpart)
- NVE/NVT/NPT cross-validation (Phase 2b)
- Performance benchmarking
- Proper/improper torsion split (v1.1)
- Multi-system coverage beyond ala-dipeptide (v1.1)

---

## First Fixer Task to Dispatch

**f1-openmm-ala-dip-fixture** — no dependencies, self-contained, produces the
fixture module that all subsequent tasks depend on, validates OpenMM reference
data is stable and reproducible before any prolix comparison code is written.
