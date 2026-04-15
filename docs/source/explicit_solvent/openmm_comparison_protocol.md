# OpenMM comparison protocol (explicit solvent)

Shared conventions for **Prolix vs OpenMM Reference** parity tests and benchmarks. Use the same numerical choices everywhere unless a test documents an intentional deviation.

## Units and box

| Quantity | Prolix | OpenMM |
|----------|--------|--------|
| Positions | Å | nm (`setPositions`; multiply by 0.1 from Å) |
| Box edge \(L\) | Å orthorhombic `(Lx,Ly,Lz)` | nm (`setDefaultPeriodicBoxVectors`; divide Å by 10) |
| Ewald \(\alpha\) | Å\(^{-1}\) | nm\(^{-1}\): **multiply by 10** when calling `setPMEParameters` |

## PME (`NonbondedForce.PME`)

- **Grid:** `setPMEParameters(alpha_nm, nx, ny, nz)` with `alpha_nm = alpha_A * 10`.
- **Cutoff:** `setCutoffDistance(cutoff_A / 10)` (nm).
- **Dispersion:** `setUseDispersionCorrection(False)` when comparing to Prolix paths that apply **explicit** LJ long-range tail handling in `explicit_corrections` (or when LJ is off).
- **Switching:** `setUseSwitchingFunction(False)` unless the Prolix path explicitly enables a switch.

## Regression config (reference defaults)

These match the minimal anchor in `tests/physics/test_openmm_explicit_anchor.py` and `tests/physics/test_pbc_end_to_end.py`:

```python
REGRESSION_EXPLICIT_PME = {
    "pme_alpha_per_angstrom": 0.34,
    "pme_grid_points": 32,  # cubic grid nx=ny=nz
    "cutoff_angstrom": 9.0,  # two-particle tests; protein tests may use 10–12
    "use_dispersion_correction": False,
    "openmm_platform": "Reference",  # deterministic CPU parity
}
```

Tighten tolerances only after locking grid-spacing policy in `make_energy_fn` / `single_padded_energy` (mesh-dependent SPME error).

## Force field files

- For **Amber ff19SB protein** + **TIP3P** water in OpenMM, prefer **one canonical pair** of files on disk:
  - Protein: resolve the same path Prolix uses (e.g. `proxide/.../assets/protein.ff19SB.xml` via `scripts/run_batched_pipeline._resolve_ff_xml()`).
  - Water: OpenMM built-in `amber14/tip3p.xml` (or equivalent) **if** atom definitions are consistent with the Prolix solvation model.
- **Do not** mix `amber14-all.xml` protein with Prolix `ff19SB` from proxide without documenting systematic energy offsets.

## Constraints (SETTLE / HBonds)

- OpenMM `createSystem(..., constraints=HBonds)` vs Prolix **SETTLE** / RATTLE paths are **not** bitwise comparable step-to-step.
- **Static E/F** parity: run OpenMM **without** constraints (`constraints=None`) when possible, or document constraint mismatch.
- **Dynamics**: compare **distribution-level** thermostat statistics, not bitwise trajectories.

## Benchmark JSON

Machine-readable runs should conform to [benchmark_run.schema.json](schemas/benchmark_run.schema.json) (see [explicit_solvent_parity_and_benchmark_requirements](explicit_solvent_parity_and_benchmark_requirements.md)).

## Related tests and scripts

- `tests/physics/test_openmm_explicit_anchor.py`
- `tests/physics/test_pbc_end_to_end.py`
- `tests/physics/test_solvated_explicit_integration.py`
- `tests/physics/test_solvated_openmm_explicit_parity.py`
- `scripts/benchmarks/openmm_langevin_temperature_stats.py` (OpenMM-only thermostat **statistics**)
