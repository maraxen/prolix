---
task_id: 260526_p2a-openmm-parity-bonded
backlog_id: 557
title: P2a — OpenMM Parity Harness (Bonded-Only) on Current PhysicsSystem
date: 260526
status: planned
phase: planning
sprint_id: 2
blocks_freeze_of: [MolecularBundle field list (P1a, id 163)]
deliverables:
  - tests/physics/test_openmm_parity_bonded.py
  - tests/physics/fixtures_openmm_parity.py
  - src/prolix/physics/field_audit.py
  - .praxia/docs/audits/260526_p2a-bonded-field-audit.md
gates:
  - phase_2_energy: per-term |dE| < 0.05 kcal/mol at fixed conformation
  - phase_3_force: RMS(|df|) < 0.01 kcal/mol/Å across all atoms
estimated_effort_hours: 6
---

# P2a — OpenMM Parity Harness (Bonded-Only)

## Context

Lessons 32–33 (importance 0.9 / 0.85) showed the §7.1 external-comparator
campaign ran without scope-equivalence checks: espaloma is a
parameterization tool not an MD engine; DMFF GPU is slower than DMFF CPU;
bonded-only and full-FF comparisons were mixed. Oracle remediation:
**open P2a before any further external benchmark work**. P2a is also the
hard prerequisite for finalising P1a's MolecularBundle field list (id 163,
in_progress) — the field audit produced here closes the Phase 1 exit
criterion that "all PhysicsSystem fields are identified and present in
Bundle".

`PhysicsSystem` is defined at `src/prolix/physics/system.py:27`;
`make_energy_fn` at line 30. No OpenMM parity code exists in the repo today.

**Derisked at planning time:** `openmm>=8.3.1` is already a project dep
(pyproject.toml), and an `openmm` pytest marker is registered — no
`uv add openmm` step needed.

## Scope (firm)

- Subject system: solvated alanine dipeptide
- In scope (v1): harmonic bonds, harmonic angles, proper dihedrals, improper dihedrals
- Out of scope (deferred): LJ, PME/Coulomb, nonbonded exceptions, Urey–Bradley, CMAP
- Compare: per-term energy + per-atom force at a single fixed conformation
- Boundary conditions: free (no PBC for v1)

## Phases

### Phase 1 — OpenMM reference setup (~1.5h)
- Load solvated alanine dipeptide PDB via OpenMM (AMBER14SB + TIP3P)
- Construct OpenMM System, extract per-term parameters (bond/angle/dihedral indices + force constants)
- Compute and verify OpenMM single-point energies per force group
- Exit: OpenMM reference fixture produces stable per-term energies on the same conformation across re-runs.

### Phase 2 — Prolix bonded extraction (~2h)
- Convert the extracted OpenMM system into the prolix `PhysicsSystem` dict shape
- Ensure per-term bonded energy functions exist on the prolix side (bonds, angles, propers, impropers)
- Comparison harness computes per-term energies on both sides at the same conformation
- **Gate (energy):** per-term `|ΔE| < 0.05 kcal/mol`

### Phase 3 — Force validation (~1.5h)
- Compute prolix forces via `jax.grad` of the per-term energy on positions
- Compute OpenMM reference forces via `context.getState(getForces=True)`
- Compare with per-atom force RMS across all atoms
- **Gate (force):** `RMS(|Δf|) < 0.01 kcal/mol/Å`

### Phase 4 — Field-audit documentation (~1h)
- Instrument `make_energy_fn` (or a wrapper) with field-access logging
- Run on the bonded-only fixture; record which `PhysicsSystem` fields are read
- Write `.praxia/docs/audits/260526_p2a-bonded-field-audit.md` listing the exercised fields with brief role-of-each notes
- **Gate (audit):** doc lists all bonded-exercised fields; explicitly flags anything currently in `MolecularBundle` (id 163) that bonded-path does NOT touch (candidates for nonbonded-path audit later)

## Critical files

| File | Purpose | New? |
|---|---|---|
| `tests/physics/fixtures_openmm_parity.py` | OpenMM fixture + PhysicsSystem bridge | new |
| `tests/physics/test_openmm_parity_bonded.py` | Energy + force parity tests | new |
| `src/prolix/physics/field_audit.py` | Field-access logger | new |
| `src/prolix/physics/system.py` (line 27, 30) | `PhysicsSystem`, `make_energy_fn` — read only here | existing |
| `.praxia/docs/audits/260526_p2a-bonded-field-audit.md` | Final audit deliverable | new |

## Risk table

| Risk | Likelihood | Mitigation |
|---|---|---|
| AMBER14SB ala-dip PDB unavailable in repo | low | OpenMM ships standard test PDBs; fall back to building from `Modeller` + sequence |
| Per-term energy mismatch from unit confusion (nm↔Å, kJ↔kcal) | medium | First-line check: unit conversions and degree-vs-radian on torsion angles |
| Force RMS mismatch despite energy match | low–medium | Add finite-difference baseline (`(E(x+ε)−E(x−ε))/2ε`) as third reference |
| Incomplete field-audit (instrument misses indirect access) | medium | Run audit on two systems (dipeptide + small protein) to widen coverage |

## Dependencies

- `openmm>=8.3.1` — **already present** in `pyproject.toml`
- AMBER14SB + TIP3P force-field XML (bundled with OpenMM)
- `jax.grad` for autodiff forces. No `custom_jvp` needed for v1.

## Out of scope (deferred — separate tasks)

- P2a-nonbonded: LJ + PME/Coulomb parity (separate sprint)
- HP2 / id 162: `make_bundle_from_system` internal-parity audit (Bundle-path counterpart to this PhysicsSystem-path harness)
- Phase 2b: NVE / NVT / NPT cross-validation
- Performance / throughput benchmarking

## Verification commands

```bash
uv run pytest tests/physics/test_openmm_parity_bonded.py -v -m openmm
uv run python -c "from prolix.physics.field_audit import audit_make_energy_fn; print('ok')"
```

## Next dispatch

`specification-specialist` to convert this plan into a fixer-executable
spec with explicit task decomposition, gates, and risk-table refinement.
Spec output: `.praxia/docs/specs/260526_p2a-openmm-parity-bonded.md`.
