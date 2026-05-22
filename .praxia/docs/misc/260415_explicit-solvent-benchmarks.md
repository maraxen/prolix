# Explicit solvent — benchmarking matrix (Phase 8)

This document ties **ad hoc** repository scripts to a single **smoke vs cluster** mental model. It is not a substitute for site-specific cluster tuning (partitions, GPU types, filesystems).

## Local smoke (developer machine)

**Goal:** Sanity-check throughput and scaling after physics or kernel changes, without claiming production cluster numbers.

| Artifact | Role |
|----------|------|
| `benchmarks/verify_pbc_end_to_end.py` | PBC / electrostatics checks |
| `benchmarks/benchmark_scaling.py`, `benchmarks/benchmark_batched.py` | Scaling experiments |
| `benchmarks/compare_jax_openmm_validity.py`, `benchmarks/verify_end_to_end_physics.py` | Cross-engine or integration checks |
| `scripts/benchmark_nlvsdense.py`, `scripts/benchmark_gb_forces.py`, `scripts/profile_step.py` | Optional timing / profiling helpers |
| [`scripts/benchmarks/prolix_vs_openmm_speed.py`](../../../scripts/benchmarks/prolix_vs_openmm_speed.py) | **Prolix (JAX) vs OpenMM** wall-clock throughput on each available OpenMM platform + current JAX backend; explicit minimal PME box (same construction as `test_openmm_explicit_anchor`); `--json` emits schema 1.0 fields |
| [`scripts/benchmarks/prolix_vs_openmm_t1_solvated.py`](../../../scripts/benchmarks/prolix_vs_openmm_t1_solvated.py) | Same script with **tier T1** defaults (larger **N** via `--charge-copies`, wider box)—throughput tier, not a full solvated peptide build |
| [`scripts/benchmarks/regression/t0_minimal.example.json`](../../../scripts/benchmarks/regression/t0_minimal.example.json) | Example archived **T0** JSON shape for release notes |

### Prolix vs OpenMM speed (`scripts/benchmarks/prolix_vs_openmm_speed.py`)

**Not physics validation.** Throughput numbers are orthogonal to parity tests; for PME energy/force tolerances see `tests/physics/test_openmm_explicit_anchor.py`.

**What it measures**

- **Prolix:** `jax.jit(jax.value_and_grad(make_energy_fn(...)))` on the minimal periodic explicit-PME two-charge system (optionally `--charge-copies` to scale **N**).
- **OpenMM:** `getState(getEnergy=True, getForces=True)` per iteration (default). Optional `--measure step` adds `integrator.step(1)` timings (OpenMM only; not a matched full MD step in Prolix).

**Backends**

- JAX: whatever `jax.devices()` reports (set `JAX_PLATFORMS=cpu,cuda` etc. if needed). On hosts without a working CUDA stack, prefer `JAX_PLATFORMS=cpu` to avoid plugin init noise.
- OpenMM: discovers installed platforms (typically `CUDA`, `OpenCL`, `CPU`, `Reference`, sometimes `HIP`). Failures (e.g. no OpenCL device) are recorded as **SKIP** rows, not hard errors. **Reference** is labeled **diagnostic** — orders of magnitude slower than production GPU paths; use `--skip-reference` for faster sweeps.

**Precision**

- `--x64` / `--no-x64` toggles JAX float64.
- `--openmm-precision mixed|single|double` maps to `CudaPrecision` / `OpenCLPrecision` where applicable.

**Example**

```bash
cd /path/to/prolix
JAX_PLATFORMS=cpu uv run python scripts/benchmarks/prolix_vs_openmm_speed.py --warmup 5 --repeats 30 --skip-reference
uv run python scripts/benchmarks/prolix_vs_openmm_speed.py --json bench.json
```

**Suggested command pattern:** run with a **small** system first, then scale **N**; record JAX platform (`jax.devices()`), X64 flag, and whether neighbor lists are used.

## Cluster / throughput matrix (reference)

Use this as a **checklist** when running on a shared cluster; fill in partition/GPU types for your site.

| Axis | What to vary | Notes |
|------|----------------|-------|
| Hardware | CPU vs GPU type | GPU: record driver + JAX CUDA wheel |
| | GPU count | Single-GPU strong scaling vs multi-replica |
| System size | Atom count / bucket | Align with `padding.ATOM_BUCKETS` |
| Kernel | NL vs dense | Dense is reference-only for large **N** |
| Electrostatics | PME (default) vs RF/DSF | RF/DSF are opt-in; see `electrostatic_methods` |

## SLURM example (generic)

```bash
#!/bin/bash
#SBATCH --job-name=prolix-bench
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8

# Replace with your environment (uv, conda, module load).
cd /path/to/prolix
uv run python benchmarks/benchmark_scaling.py
```

**Checkpointing / preemptable** jobs are environment-specific; prefer resumable trajectories (e.g. `SimulationSpec.save_path` / `accumulate_steps`) if your queue preempts.

## Engaging cluster (MIT)

For **rsync**, SLURM templates, `Justfile` recipes (`push-engaging`, `submit-bench-chignolin`, log tail/pull), manifest JSON, and a **chignolin → DHFR** checklist, see **`.agent/docs/plans/engaging_protein_benchmarks.md`** in the repository (not part of the Sphinx tree).

## Related

- [explicit_solvent_parity_and_benchmark_requirements](explicit_solvent_parity_and_benchmark_requirements.md) — OpenMM parity layers, heat/thermostat observables, benchmark comparability
- [current_implementation](current_implementation.md)
- [explicit_solvent_implementation_plan](explicit_solvent_implementation_plan.md) — original phase definitions
