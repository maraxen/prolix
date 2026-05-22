# Prolix

API documentation for Prolix.

## Validation (explicit solvent)

Start with **OpenMM anchor parity**: `tests/physics/test_openmm_explicit_anchor.py` (two-charge PME energy + forces vs OpenMM Reference). Related integration tests live under `tests/physics/test_pbc_end_to_end.py`. **As-built modules and phased follow-ups:** [current_implementation](explicit_solvent/current_implementation.md). **Parity vs OpenMM, heat tracking, and comparable benchmarks (requirements matrix):** [explicit_solvent_parity_and_benchmark_requirements](explicit_solvent/explicit_solvent_parity_and_benchmark_requirements.md). **Benchmarks / runbook / profiling policy:** [explicit_solvent_benchmarks](explicit_solvent/explicit_solvent_benchmarks.md), [explicit_solvent_runbook](explicit_solvent/explicit_solvent_runbook.md), [spatial_sorting_profile_gate](explicit_solvent/spatial_sorting_profile_gate.md). Status and roadmap: [explicit_solvent_progress](explicit_solvent/explicit_solvent_progress.md).

```{toctree}
:maxdepth: 2

api
examples
explicit_solvent/explicit_solvent_architecture
explicit_solvent/explicit_solvent_implementation_plan
explicit_solvent/current_implementation
explicit_solvent/explicit_solvent_progress
explicit_solvent/explicit_solvent_benchmarks
explicit_solvent/explicit_solvent_parity_and_benchmark_requirements
explicit_solvent/openmm_comparison_protocol
explicit_solvent/explicit_solvent_runbook
explicit_solvent/spatial_sorting_profile_gate
explicit_solvent/gpu_optimization_strategies
explicit_solvent/notes
explicit_solvent/phase4_solvation_implementation_plan
explicit_solvent/energy_accumulation_precision
explicit_solvent/ewald_boundary_conditions
```
