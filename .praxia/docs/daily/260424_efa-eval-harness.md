# 2026-04-24 Sprint 2: EFA Evaluation Harness

## Summary
Implemented complete evaluation harness for EFA vs PME electrostatic comparison. All phases
delivered, auditor findings remediated, atomic commits made.

## Delivered
- **Phase 0**: Threaded `n_rff_features`, `rff_seed` through `flash_explicit_forces/energy/total_energy`
- **Phase 1**: `src/prolix/physics/eval_harness.py` — 9 utilities including `make_tip3p_water_system`,
  `run_nve` (freeze_geometry mode), `make_comparison_energies`
- **Phase 2**: `tests/physics/test_efa_vs_pme_forces.py` — 5 tests: relative RMSE (<15%), per-component
  t-stat bias (<4.0), exclusion correction with/without comparison, NVT smoke, D-scaling monotonicity
- **Phase 3**: `tests/physics/test_efa_energy_consistency.py` — 3 tests: PME determinism, fixed-ω
  reproducibility, resampled-ω variance ratio (>5×)
- **Phase 4**: `scripts/benchmark_efa_vs_pme.py` — wall-clock profiling, N=32–256 waters, JSON output
- **Phase 5**: `scripts/ablation_efa.py` — D-sweep RMSE, ω-resampling variance ratio
- **Phase 6**: Pytest marker `electrostatic_comparison`, calibration comment in conftest

## Auditor Findings Remediated
- **C2**: Bond/exclusion arrays now populated in `make_tip3p_water_system` (was zeros)
- **C4**: `run_nve` now recomputes forces at each position (was stale closure)
- **H10**: Mask slicing fixed in tests — uses full `atom_mask` not `[:n_real_atoms]`
- **H8**: Exclusion correction test compares with/without, not just isfinite
- **H1**: `run_nve` honors `forces_fn` parameter
- **H7**: Unused REGRESSION_EFA/efa_params replaced with calibration comment
- **M1**: Benchmark uses `jax.block_until_ready` not `jax.effects_barrier`

## OODA verdict: APPROVE (after 1 NEEDS_WORK cycle)
Oracle conditions addressed: NVE bond treatment → frozen-geometry energy consistency; absolute
RMSE threshold → relative 15%; bias z-score → per-component t-stat with axis and threshold specified.

## Next
- First calibration run of `test_efa_pme_force_rmse_relative` to pin `REGRESSION_EFA['relative_rmse_D512']`
- MTT Phase 2 (backlogged): Lanczos log-det on Coulomb-weighted Laplacian via matfree
- MIST Sprint 3+: Requires trajectory ensembles from working MD
