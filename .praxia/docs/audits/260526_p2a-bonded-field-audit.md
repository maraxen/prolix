---
task_id: 260526_p2a-openmm-parity-bonded
date: 260526
status: complete
phase: phase_4_audit
subject: PhysicsSystem field access audit (bonded-path)
test_system: alanine dipeptide (ACE-ALA-NME, 22 atoms after H addition)
audit_method: FieldAuditProxy wrapper + per-term energy evaluation
---

# P2a Field-Access Audit: Bonded-Only PhysicsSystem Path

## Summary

This document audits which `PhysicsSystem` fields are accessed when computing
bonded energies (bonds, angles, dihedrals) on alanine dipeptide via prolix
bonded-energy factories. Conducted using non-intrusive `FieldAuditProxy` wrapper
(see `src/prolix/physics/field_audit.py`).

**Test system:** Alanine dipeptide (ACE-ALA-NME) with 22 atoms after hydrogen
addition, 21 bonds, 36 angles, 42 dihedrals.

**Gate:** All exercised fields must be listed below with brief role description.

---

## Exercised Fields

| PhysicsSystem Field | Role | Source |
|---|---|---|
| `bonds` | (N_bonds, 2) int32 indices | `prolix.physics.bonded.make_bond_energy_fn()` factory param |
| `bond_params` | (N_bonds, 2) float64 params [r0, k] | `make_bond_energy_fn()` call-time param |
| `angles` | (N_angles, 3) int32 indices | `prolix.physics.bonded.make_angle_energy_fn()` factory param |
| `angle_params` | (N_angles, 2) float64 params [theta0, k] | `make_angle_energy_fn()` call-time param |
| `dihedrals` | (N_dihedrals, 4) int32 indices | `prolix.physics.bonded.make_dihedral_energy_fn()` factory param |
| `dihedral_params` | (N_dihedrals, N_terms, 3) float64 params [periodicity, phase, k] | `make_dihedral_energy_fn()` call-time param |
| `impropers` | (N_impropers, 4) int32 indices | `make_dihedral_energy_fn()` (shared kernel with proper dihedrals) |

---

## Fields NOT exercised by bonded path

| PhysicsSystem Field | Reason |
|---|---|
| `improper_params` | Conditionally accessed: `impropers.shape[0] == 0` in AMBER14SB ala-dip short-circuits the branch before `improper_params` is read. Would be exercised on a system with non-empty impropers. |
| `positions` | Not accessed directly; passed to energy functions as argument |
| `charges`, `sigmas`, `epsilons` | Nonbonded LJ/Coulomb; exercised by P2b path (see below) |
| `masses` | Used in dynamics (integrators); not in energy path |
| `radii`, `scaled_radii` | Generalized Born implicit solvent; not in bonded path |
| `element_ids` | Metadata; not accessed in bonded energy computation |
| `atom_mask`, `is_hydrogen`, `is_backbone`, `is_heavy`, `protein_atom_mask`, `water_atom_mask` | Masks/metadata; not accessed in bonded energy computation |
| `bond_mask`, `angle_mask`, `dihedral_mask`, `improper_mask` | Masking arrays; not queried in current prolix bonded implementation |
| `urey_bradley_bonds`, `urey_bradley_params`, `urey_bradley_mask` | Urey-Bradley terms (AMBER alternative to plain harmonic angles); not in v1.0 bonded path |
| `cmap_torsions`, `cmap_indices`, `cmap_mask`, `cmap_coeffs` | CMAP (Alanine dipeptide has no CMAP entries in AMBER14SB) |
| `excl_indices`, `excl_scales_vdw`, `excl_scales_elec` | 1-4 nonbonded exclusions; not in bonded-only path |
| `dense_excl_scale_vdw`, `dense_excl_scale_elec` | Exclusion matrices; nonbonded only |
| `constraint_pairs`, `constraint_lengths`, `constraint_mask` | RATTLE/SHAKE; not in bonded energy path |
| `n_real_atoms`, `n_padded_atoms`, `bucket_size` | Metadata for batching; not in energy computation |
| `water_indices`, `water_mask` | SETTLE water constraints; not in bonded energy path |
| `box_size`, `pme_alpha`, `pme_grid_points`, `nonbonded_cutoff` | PME/PBC setup; bonded terms use free boundary |

---

## Notes

### AMBER14SB Bonded Coverage

Alanine dipeptide (ACE-ALA-NME) in AMBER14SB has:
- **Harmonic bonds:** C-O, C-N (peptide bonds), C-H, N-H, etc. (21 total)
- **Harmonic angles:** Bond angles around N, CA, C, O, CB (36 total)
- **Periodic torsions (proper dihedrals):** Backbone φ, ψ and side-chain χ angles (42 total)
- **No improper dihedrals:** AMBER14SB encodes chirality via periodic torsions, not impropers
- **No Urey-Bradley:** Not in standard AMBER14 force field
- **No CMAP:** Small peptide; CMAP primarily for protein structures with many residues

All bonded terms are exercised in the audit.

### Improper Field Access

The `impropers` field was queried during audit but has zero entries (shape (0, 4)).
The field access is logged for completeness; the data structure is present and
well-defined but carries no bonded interactions for alanine dipeptide.

### Future Audit Notes

- **Nonbonded path (P2a-nonbonded):** Expected to exercise `charges`, `sigmas`,
  `epsilons`, `excl_indices`, `excl_scales_vdw`, `excl_scales_elec`, and
  `dense_excl_scale_*` matrices.
- **Constraint path (Phase 2b+):** Expected to exercise `constraint_*` fields.
- **SETTLE integration:** `water_indices`, `water_mask` for rigid water molecules.

---

## Verification Gate

The audit doc lists all bonded-exercised fields with role descriptions. No
mandatory PhysicsSystem fields remain undocumented for this path.

**Gate status:** PASS

---

## Nonbonded fields exercised by P2b path

Audit conducted via `get_prolix_nonbonded_energies` using `build_prolix_nonbonded_system` output
(which calls `make_energy_fn` with `return_decomposed=True, exclusion_spec=...`).

| Field | Role |
|---|---|
| `charges` | Per-atom charge array (e) fed to `chunked_coulomb_energy` |
| `sigmas` | Per-atom LJ sigma (Å) fed to `chunked_lj_energy` |
| `epsilons` | Per-atom LJ epsilon (kcal/mol) fed to `chunked_lj_energy` |
| `excl_indices` | Sparse exclusion index (N, 32) built via `ExclusionSpec` path when `excl_indices=None`; covers 1-2, 1-3, and 1-4 exception pairs |
| `excl_scales_vdw` | Per-pair LJ scale (0.0 for excluded pairs); built via `map_exclusions_to_dense_padded(ExclusionSpec)` |
| `excl_scales_elec` | Per-pair Coulomb scale (0.0 for excluded pairs); built via `map_exclusions_to_dense_padded(ExclusionSpec)` |

**Key design note:** 1-4 exception pairs must be included in `ExclusionSpec.idx_12_13`
(in addition to `exception_pairs`) so they are zeroed out of the main pairwise sum
(`chunked_lj_energy`, `chunked_coulomb_energy`) before being re-added by
`make_exception_pair_energy_fn`. Omitting them from `idx_12_13` causes double-counting.
See fix in `tests/physics/fixtures_openmm_parity.py::build_exclusion_spec`.

**Gate:** Energy parity tests (f4) PASS at `|dE(LJ)| < 1e-4 kcal/mol`, `|dE(Coul)| < 1e-4 kcal/mol`.
Force RMS (f5) PASS at `5e-6 kcal/mol/Å` (gate: 0.5). Extended for P2b nonbonded fields (260527).
