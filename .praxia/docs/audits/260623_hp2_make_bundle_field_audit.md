# HP2 Field Audit: `make_bundle_from_system` vs `make_energy_fn`

**Date:** 2026-06-23 | **Backlog:** #162 | **Task:** 260615_autonomous-loop Sprint 42 Track C

## Verdict

**Advisory PASS with 4 gaps** — bundle factory covers the bonded/nonbonded core used by
`make_energy_fn`, but is not yet a drop-in replacement for full PhysicsSystem energy
evaluation. Gaps block strict V1 parity and HP2 OpenMM term-by-term close.

## Field mapping

| PhysicsSystem / make_energy_fn | MolecularBundle | Status |
|-------------------------------|-----------------|--------|
| `positions` | `positions` + `atom_mask` + `n_atoms` | OK (padded) |
| `charges`, `sigmas`, `epsilons` | same | OK |
| `radii`, `scaled_radii` | same | OK (optional in PS; defaulted in make_energy_fn) |
| `bonds`, `bond_params` | `bond_idx`, `bond_params`, `bond_mask` | OK |
| `angles`, `angle_params` | `angle_idx`, `angle_params`, `angle_mask` | OK |
| `dihedrals` / `proper_dihedrals` | `dihedral_idx`, `dihedral_params` | **GAP:** PS uses `(N, N_terms, 3)`; bundle expects flattened `(D, 4)` — factory TODO at system.py:423 |
| `impropers` | `improper_idx`, `improper_params`, `improper_is_periodic` | **GAP:** same multi-term flattening; `improper_is_periodic` hardcoded `False` |
| `urey_bradley_bonds` | `urey_bradley_idx` `(U, 3)` | **SHAPE:** PS uses 2-atom bonds; factory pads to 3-col (angle bucket) |
| `cmap_torsions`, `cmap_indices`, `cmap_energy_grids` | `cmap_*` | **GAP:** factory leaves CMAP empty (system.py:553 comment T6) |
| `water_indices` | `water_indices`, `water_mask`, `n_waters` | OK |
| `excl_indices` (+ scales) | `excl_indices`, `excl_scales_*`, `excl_mask` | **GAP:** PS `(N, max_excl)` per-atom layout vs bundle `(E, 2)` pair list — conversion not implemented for non-empty |
| `exception_pairs` (1-4) | `exception_*` | **GAP:** factory zeroes; make_energy_fn reads from PS |
| `box_size` / PBC | `box`, `shape_spec.has_pbc`, `boundary_condition` | OK (diagonal box mat) |
| `pme_alpha`, `nonbonded_cutoff` | `pme_alpha`, `cutoff_distance` | OK |
| `atom_names` (implicit solvent radii) | not stored | **GAP:** bundle has no atom_names; mbondi2 fallback unavailable on bundle path |

## Implications

1. **V1 (#263):** `EnsemblePlan.run()` must call bundle-backed `make_energy_fn(MolecularBundle)`
   (or equivalent) — not stub zero energy — before solvated-AKE parity is meaningful.
2. **HP2 OpenMM parity:** PhysicsSystem-path harness (P2a #557) is the right near-term gate;
   bundle-path parity requires closing dihedral multi-term flattening + excl pair conversion.
3. **#162 full close** deferred until: (a) multi-term dihedral/improper flatten in factory,
   (b) excl_indices pair-list conversion, (c) CMAP mapping, (d) OpenMM term RMS < 0.05 kcal/mol
   on bundle-constructed system.

## Recommended next sprint items

| Priority | Item |
|----------|------|
| P1 | Flatten `(N, N_terms, 3)` dihedral/improper params in `make_bundle_from_system` |
| P1 | Wire real bundle energy into `EnsemblePlan.run()` |
| P2 | excl_indices layout converter (per-atom → pair list) |
| P2 | CMAP field mapping from PhysicsSystem |
