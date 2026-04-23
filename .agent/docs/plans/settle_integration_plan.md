# Implementation Plan: SETTLE Integration & Explicit Solvent Optimization (Cycle 3/3 FINAL)

## Goal Description
This plan addresses two critical undocumented gaps in the explicit solvent architecture:
1. **SETTLE Integration**: Integrating the SETTLE constraint algorithm (rigid water) into the main `run_simulation` scalar path (in `simulate.py`), controlled via `SimulationSpec.rigid_water`.
2. **Implicit GB NL Scaling Bottleneck**: Eliminating the O(N^2) dense matrix gather for Generalized Born (GB) neighbor list masks by leveraging sparse `ExclusionSpec` directly.

## User Review Required
> [!IMPORTANT]
> - `SimulationSpec` will gain a new `rigid_water: bool = False` flag. Setting it to `True` for explicit solvent systems will enable SETTLE.
> - `physics/simulate.py` will introduce `settle_rattle_langevin`, an integrator combo that applies SETTLE to waters and SHAKE/RATTLE to solute constraints simultaneously.
> - **Mathematical Purity**: Unlike the batched path, the standard runner will *not* contain heavy gradient clapping or limiters (`VLIMIT`). If capping is required for a specific run, numerical stability must be managed by the generic `SimulationSpec` parameters, keeping the core path purely mathematically sound for NVT properties.

## Proposed Changes

---
### simulate_runner (src/prolix/simulate.py)
Summary: Add `rigid_water` configuration and dispatch to the new SETTLE integrator.
#### [MODIFY] simulate.py
- Extend `SimulationSpec` with `rigid_water: bool = False`.
- In `run_simulation`, extract `water_indices` if available.
- Update integrator selection: If `spec.rigid_water` is True and `water_indices` is present, dispatch to `physics_simulate.settle_rattle_langevin`.

---
### physics_simulate (src/prolix/physics/simulate.py)
Summary: Add joint pure SETTLE+RATTLE integrator.
#### [MODIFY] simulate.py
- Add `settle_rattle_langevin(energy_or_force_fn, shift_fn, dt, kT, gamma, mass, constraints, water_indices, box)`.
- Use `settle_positions` and `settle_velocities` alongside `project_positions` and `project_momenta`.
- In the `shift_fn` unrolling step, ensure the water topology remains intact upon PBC wrap by calling `solvation.fix_water_geometry_padded` if a `box` is present.
- Implement the pure BAOAB integrator logic WITHOUT mandatory heavy TPU gradient limits.

---
### physics_system (src/prolix/physics/system.py)
Summary: Remove O(N^2) dense GB mask gathering.
#### [MODIFY] system.py
- In `make_energy_fn`, detect when sparse exclusions can bypass dense masking.
- Use `mask_vdw, mask_elec, mask_hard = nl.compute_exclusion_mask_neighbor_list(exclusion_spec, neighbor_idx, N)`.
- **Target Mapping**: For `pair_mask_born` and `pair_mask_energy`, use `mask_elec` (which appropriately maps `0.0` for 1-2/1-3 and `scale_14_elec` for 1-4) thereby perfectly recovering the standard O(N^2) `exclusion_mask` parity without explicit dense gathering. For AMBER vs OpenMM GB behaviors, continue respecting standard parity rules.

## Open Questions
> [!NOTE]
> None remaining. Cycle 3 critique passed all architecture hurdles.

## Verification Plan
### Automated Tests
- Run `uv run pytest tests/physics/test_electrostatic_methods_openmm.py`
- Run `uv run pytest tests/physics/test_openmm_explicit_anchor.py`
- Verify that `batched_simulate_test` passes.
### Manual Verification
- Dry run `run_simulation` with an explicit solvent box and `rigid_water=True`, checking that it does not crash and performance scaling is healthy.
