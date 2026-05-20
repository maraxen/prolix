# Changelog

All notable changes to Prolix are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.0.0 — 2026-04-26

### Added
- **Explicit Solvent Dynamics**: SETTLE rigid water constraints with TIP3P water model
- **NVT Thermostat**: `settle_langevin` integrator for constant-volume, constant-temperature simulations
- **NPT Barostat**: `settle_csvr_npt` implementing stochastic cell rescaling (Bernetti & Bussi 2020) for isothermal-isobaric ensemble
- **CSVR Thermostat**: `settle_csvr` for canonical (NVT) sampling with velocity rescaling
- **Batched Dynamics**: `safe_map` chunked batching for memory-efficient multi-trajectory runs
- **Quality Gates**: Per-trajectory warning accumulation in `LangevinState` for force capping, velocity limits, constraint violations
- **kUPs Cross-Validation**: Sprint 5 cross-validation of thermostat performance against kUPs reference (1CRN, 5k-step trajectories)

### Fixed
- **safe_map Reshape Bug**: Fixed crash on heterogeneous pytrees (different leaf leading dimensions)
  - Added explicit validation requiring all pytree leaves have consistent batch dimension B
  - Updated `LangevinState.tree_flatten` to properly batch `warn_counts` field
  - Prevents silent data corruption from incompatible tree structures

### Changed
- **Temperature Control**: Standard Langevin thermostat now accounts for SETTLE constraint-induced kinetic energy removal
  - Friction coefficient coupling ensures re-equilibration between constraint steps
  - Separate `project_ou_momentum_rigid=True` for 6D rigid-body noise sampling (water-specific)

### Known Constraints

#### Timestep Limit
- **dt ≤ 0.5 fs (AKMA units)** required for SETTLE + Langevin coupling
- Larger timesteps cause positive feedback: SETTLE removes KE → thermostat adds it → SETTLE removes again
- At dt ≤ 0.5 fs, per-step impulse magnitude is small enough for Langevin friction/noise to re-equilibrate
- **Roadmap**: Constraint-aware thermostat (v2.0, 2-4 weeks) will eliminate this limit

#### Batched Production
- **`batched_equilibrate` NaN issue**: Known NaN emergence in batched initialization (Sprint 8 fix)
- **Workaround**: Initialize states directly with cold-start positions, skip equilibration step
- Impact: Batched production runs OK; multi-system equilibration requires sequential initialization

#### NPT Validation
- **Validated**: 20ps runs at dt=0.5fs, 4 TIP3P waters, T=300±5K, P=1±50bar (Sprint 7)
- **Status**: Experimental for production use; consider NVT + external pressure rescaling until v1.1

### Removed
- None

### Deprecated

## v1.1.x — Deprecations

### Deprecated

- **`batched_produce`** (`src/prolix/batched_simulate.py`): Issues `DeprecationWarning` on call.
  Replacement: `EnsemblePlan.from_bundles(...).run(...)` (available v1.2).
  Removal target: v2.0.

- **`LangevinState` re-export from `prolix.batched_simulate`**: Issues `DeprecationWarning` on import.
  Replacement: `from prolix.types.integrators import LangevinState`.
  Removal target: v2.0.

- **`pad_protein`** (`src/prolix/padding.py`): Issues `DeprecationWarning` on call.
  Replacement: `MolecularBundle.from_protein()` (available v1.2).
  Removal target: v2.0.

- **`PaddedSystem`** (`src/prolix/__init__.py`, `src/prolix/typing.py`): Docstring-level deprecation note + single `DeprecationWarning` in `__init__.py` re-export.
  Replacement: `MolecularBundle` (`prolix.types.bundles`).
  Removal target: v2.0.

- **`collate_batch`** (`src/prolix/padding.py`): Issues `DeprecationWarning` on call.
  Replacement: `EnsemblePlan.from_bundles(bundles)` (available v1.2).
  **Removal target: v1.2** (hard-deprecate, one cycle only).

### Security
- None

### Roadmap (v2.0, estimated 4-8 weeks)

1. **Constraint-Aware Thermostat** (2-4 weeks)
   - Couple thermostat only to unconstrained degrees of freedom
   - Eliminates dt ≤ 0.5 fs limitation
   - Enables dt ≥ 1.0 fs with stable temperature control

2. **Fix batched_equilibrate NaN** (1 week)
   - Root cause analysis of multi-trajectory equilibration
   - Sentinel checks and debugging

3. **LINCS/CCMA Alternative to SETTLE** (3-4 weeks, high risk)
   - More flexible constraint solver
   - Enables longer timesteps independent of coupling method
   - Requires extensive validation against reference ensembles

### Contributors

- Implementation and validation: Marielle Russo
- Oracle guidance and decision gates: Sprint 7 Oracle Review

### Acknowledgments

- SETTLE algorithm: Miyamoto & Kollman (1992)
- CSVR barostat: Bernetti & Bussi (2020)
- JAX molecular dynamics framework: DeepMind JAX-MD project
