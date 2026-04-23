# Explicit solvent — production runbook (Phase 9)

End-to-end **happy path** for periodic explicit-solvent MD with PME (default). Optional paths (OpenMM validation, RF/DSF) are noted separately.

## Prerequisites

- **Parameterized structure:** `proxide` `parse_structure` with `OutputSpec(parameterize_md=True)` and a supported AMBER-style XML (e.g. ff19SB).
- **Solvation (optional):** `physics.solvation.solvate_protein` → `MergedTopology`; pad with `padding.pad_solvated_system` for batched / padded workflows.
- **JAX:** GPU or CPU; for strict parity with reference tests, use `jax_enable_x64` where tests do.
- **OpenMM (optional):** Needed only for OpenMM-marked validation tests. Install the project extra your environment documents (e.g. `openmm`) so `pytest -m openmm` can run.

## Happy path: `SimulationSpec` + `run_simulation`

Use the **canonical** production entry point:

```python
from prolix.simulate import SimulationSpec, run_simulation
```

(There is a separate legacy `physics.simulate.SimulationSpec` for older paths; new work should use `prolix.simulate`.)

1. **Load protein** from PDB/mmCIF using proxide (see tests under `tests/physics/` for patterns).
2. **Set periodic box** in `SimulationSpec.box` (orthorhombic `(Lx, Ly, Lz)` in Å) when using explicit solvent.
3. **Enable PBC:** `use_pbc=True` (required for standard explicit PME path in the production runner).
4. **Neighbor list (recommended):** `use_neighbor_list=True`, set `neighbor_cutoff` (Å) and update intervals consistent with your timestep (see field docstrings on `SimulationSpec`).
5. **PME:** Defaults `pme_alpha`, `pme_grid_size` / optional `pme_grid_spacing` match `physics.system.make_energy_fn`. **Regression / OpenMM alignment:** the SSOT for explicit-solvent PME knobs used in parity tests is `REGRESSION_EXPLICIT_PME` in `src/prolix/physics/regression_explicit_pme.py` (fixture `regression_pme_params`). After changing it, run `python scripts/export_regression_pme.py` and commit the updated [openmm_comparison_protocol](openmm_comparison_protocol.md); CI runs `--check` on that script.
6. **Minimization:** `run_simulation` runs FIRE with your spec before production; ensure initial clashes are not catastrophic (short EM in external tool if needed).

## Explicit electrostatics options (advanced)

| Method | Where | When |
|--------|--------|------|
| `ElectrostaticMethod.PME` | `make_energy_fn`, `SimulationSpec.electrostatic_method` | Default; reciprocal + erfc direct space |
| `REACTION_FIELD` | Same | OpenMM CutoffPeriodic–style RF; **no** reciprocal sum; pinned in `tests/physics/test_electrostatic_methods_openmm.py` |
| `DAMPED_SHIFTED_FORCE` | Same | Shifted erfc Coulomb (energy vanishes at cutoff); not identical to OpenMM PME |

**RF/DSF require `implicit_solvent=False`.** Do not mix with implicit GB in the same `make_energy_fn` call.

## SETTLE and constraints

Rigid water (**SETTLE**) and related constraints for explicit batched MD are implemented on the **batched** explicit path (`batched_simulate.make_langevin_step_explicit` and related helpers). See `batched_simulate` docstrings for SETTLE and constraint wiring.

OpenMM comparisons that use `HBonds` / `rigidWater` are **not** step-by-step trajectory matches to Prolix SETTLE; treat dynamics checks as **smoke or distribution-level** (temperature bands, finiteness), and reserve tight thermostat targets for long runs or manual benchmarks (see [openmm_comparison_protocol](openmm_comparison_protocol.md) § Constraints).

## Failure modes (what to check)

| Symptom | Check |
|---------|--------|
| `sigma <= 0` / parameterization errors | `strict_parameterization` on `make_energy_fn`; fix FF coverage or disable for debugging only |
| Non-finite energy | Box size vs cutoff; PME grid; initial clashes |
| NL overflow | Neighbor list capacity / cutoff; increase skin or buffer if using jax-md neighbor API |
| Implicit GB + NL parity | Not claimed for production — see `TODO(implicit_GB_NL)` in `physics/system.py` |

## Validation against OpenMM

- **Anchor:** `tests/physics/test_openmm_explicit_anchor.py` (two-charge PME).
- **Protein + solvent energy/force parity (integration):** `tests/physics/test_explicit_solvation_parity.py` (`TestOpenMMSolvationParity`; `pytest -m integration`).
- **TIP3P box Langevin-style smoke:** `tests/physics/test_explicit_langevin_tip3p_parity.py` (OpenMM Reference + Prolix SETTLE legs; slow / OpenMM; `PROLIX_TIP3P_PARITY_N_WATERS` optional).
- **RF opt-in:** `tests/physics/test_electrostatic_methods_openmm.py` (requires OpenMM, `pytest -m openmm`).
- **OpenMM thermostat statistics (manual / long-run):** `scripts/benchmarks/openmm_langevin_temperature_stats.py`.
- **Throughput (not validation):** `scripts/benchmarks/prolix_vs_openmm_speed.py` — see [explicit_solvent_benchmarks](explicit_solvent_benchmarks.md).

## Related docs

- [current_implementation](current_implementation.md)
- [explicit_solvent_benchmarks](explicit_solvent_benchmarks.md)
- [spatial_sorting_profile_gate](spatial_sorting_profile_gate.md)
