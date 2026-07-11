# HP2 Field Audit: `make_bundle_from_system` vs `make_energy_fn`

**Date:** 2026-06-23 (updated Sprint 57) | **Backlog:** #162

## Verdict

**PASS (bonded core + factory mapping)** — Sprint 57 closes the P1 bonded parity gate
and exclusion/exception factory mapping. Remaining gaps are nonbonded bundle-path
energy evaluation and implicit-solvent metadata (documented below).

## Field mapping

| PhysicsSystem / make_energy_fn | MolecularBundle | Status |
|-------------------------------|-----------------|--------|
| `positions` | `positions` + `atom_mask` + `n_atoms` | OK (padded) |
| `charges`, `sigmas`, `epsilons` | same | OK |
| `radii`, `scaled_radii` | same | OK |
| `bonds`, `bond_params` | `bond_idx`, `bond_params`, `bond_mask` | OK |
| `angles`, `angle_params` | `angle_idx`, `angle_params`, `angle_mask` | OK |
| `dihedrals` / multi-term params | `dihedral_*` | OK — `_flatten_multi_term_torsions` |
| `impropers` | `improper_*` | OK — same flatten helper |
| `urey_bradley_bonds` | `urey_bradley_idx` `(U, 3)` | OK (2-col padded to angle bucket) |
| `cmap_torsions`, grids | `cmap_torsion_idx`, `cmap_energy_grids` | OK when present on system |
| `water_indices` | `water_indices`, `water_mask`, `n_waters` | OK |
| dense `excl_indices` `(N, M)` | `excl_indices` `(E, 2)` | OK — `_dense_excl_to_pair_list` + `exclusion_spec` |
| `exception_pairs` (1-4) | `exception_*` | OK via `exclusion_spec` or system fields |
| `box_size` / PBC | `box`, `shape_spec.has_pbc` | OK |
| `pme_alpha`, `nonbonded_cutoff` | same | OK |
| `atom_names` (GB radii) | not stored | **GAP** — bundle path lacks mbondi2 fallback |

## CI gates (Sprint 57)

- `tests/physics/test_hp2_bundle_factory_parity.py::test_hp2_bonded_total_energy_matches_physics_system_path`
- `tests/physics/test_bundle_factory.py` — flatten + dense excl conversion unit tests

## Remaining gaps (not blocking #162 bonded close)

1. **Bundle-path nonbonded energy** — `bonded_energy_fn_from_bundle` is bonded-only; full
   `make_energy_fn(MolecularBundle)` for LJ/Coulomb/PME not yet wired into `EnsemblePlan.run`.
2. **Implicit solvent / atom_names** — GB path in `make_energy_fn` needs atom names on bundle.
3. **OpenMM term-by-term on bundle-constructed NB** — follow-on after bundle NB energy lands.

## Implications

- **V1 strict solvated-AKE parity** still requires full FF on bundle path (not just bonded).
- **P2a PhysicsSystem harness** remains authoritative for OpenMM NB parity until (1) closes.
