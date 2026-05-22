# Explicit solvent: parity, heat tracking, and OpenMM-comparable benchmarks

This document defines **what we need to claim** rigorous explicit-solvent validation and **what already exists** in the repository. It complements [explicit_solvent_benchmarks](explicit_solvent_benchmarks.md) (smoke vs cluster) and [current_implementation](current_implementation.md) (as-built snapshot).

---

## 1. Three separate questions

| Question | Meaning |
|----------|---------|
| **Parity** | Do **energy and forces** (and optionally **short dynamics**) match a reference—typically **OpenMM** with the same periodic box, cutoff, PME parameters, exclusions, and constraints? |
| **Heat / thermostat** | Under **NVT**, is the **stochastic thermostat** behaving sensibly (target temperature, kinetic energy fluctuations)? Under **NVE**, is **total energy drift** acceptable for the timestep and precision? |
| **Benchmarking vs OpenMM** | **Wall-clock throughput** on **comparable work** (e.g. energy+forces per call, or integrator step), **not** physics proof—must use **matched system size and API cost class**. |

Implicit GB (`benchmark_nlvsdense.py`, chignolin SLURM stage) is **out of scope** for this page.

**Canonical protocol:** [openmm_comparison_protocol](openmm_comparison_protocol.md) (units, PME mapping, regression defaults). **Benchmark JSON schema:** [schemas/benchmark_run.schema.json](schemas/benchmark_run.schema.json).

---

## 2. Parity layers (OpenMM as reference)

### 2.1 Layer A — Static energy & forces (highest ROI)

**Goal:** Same positions → same **scalar energy** and **forces** within documented tolerances.

| Item | Status | Location / notes |
|------|--------|------------------|
| Minimal periodic **two-charge PME** (dense, no NL) | Done | `tests/physics/test_openmm_explicit_anchor.py` |
| **PME + PBC** multiple configs vs OpenMM | Done | `tests/physics/test_pbc_end_to_end.py` |
| **Reaction-field** vs OpenMM Reference | Done (optional dep) | `tests/physics/test_electrostatic_methods_openmm.py` |
| **Internal** NL vs dense, Flash vs `make_energy_fn`, padded vs system | Done | `tests/physics/test_explicit_validation_expansion.py` |
| **Protein** topology, **ExclusionSpec**, NL vs dense explicit PME | Done (mostly `@pytest.mark.slow`) | `tests/physics/test_protein_nl_explicit_parity.py` |
| **Solvated** merged topology → pad → finite NL energy | Done (slow) | `tests/physics/test_solvated_explicit_integration.py` |
| Four-particle PME extended OpenMM Reference | Done (slow + `openmm`) | `tests/physics/test_explicit_slow_validation.py` |
| Bond/angle/dihedral terms vs OpenMM (vacuum / modeller path) | Partial | `tests/physics/test_explicit_parity.py` (`TestOpenMMParity`) |

**Gaps to close for “comprehensive” static parity**

1. **Single solvated protein** (water + protein) **energy + forces vs OpenMM** on an **identical** `openmm.System` (same XML, box, PME grid policy, cutoff, constraints)—one **small** system (e.g. tiny water box + mini-peptide) with tight tolerances.
2. **Documented PME grid / Ewald splitting policy** vs OpenMM (`setPMEParameters`, dispersion correction flags)—already partially in anchor docstrings; enforce one **regression** config for production comparisons.
3. **Neighbor-list vs dense** for **solvated** systems at production cutoffs—integration test exists for finiteness; **numeric parity** vs dense is the stronger bar (subset of protein tests).

---

### 2.2 Layer B — Dynamics parity (harder; optional milestones)

**Goal:** Same initial state + integrator class → **trajectories** stay close for a **short** horizon, or **ensemble statistics** (e.g. mean temperature) agree.

| Item | Status | Notes |
|------|--------|--------|
| Short **NVE** sanity (finite energy, drift bound) | Done | `tests/physics/test_explicit_slow_validation.py::test_explicit_pbc_nve_short_run_finite` |
| **NVT** mean **T** vs target (statistical window) | Done | `tests/physics/test_explicit_slow_validation.py::test_explicit_pbc_nvt_mean_temperature_targets_spec` |
| **Step-by-step** Prolix vs OpenMM Langevin for explicit PME | **Not done** | Requires matched **Langevin/BAOAB** splitting, constraint schedule, and identical noise **or** distribution-level comparison. |
| **OpenMM** `integrator.step` vs Prolix `simulate` on **same** explicit system | **Ad hoc only** | `benchmarks/compare_jax_openmm_validity.py` exists but is not a CI gate; depends on full stack. |

**Recommendation:** Treat **Layer B** as **release milestones**: **NVE** drift is bounded on the toy explicit-PBC system; **NVT** mean **T** is checked statistically; optional OpenMM reference stats live in `scripts/benchmarks/openmm_langevin_temperature_stats.py`. Prefer **mean/variance of T** or **histograms** over bitwise stochastic trajectories.

---

## 3. Heat tracking and thermostat observables

### 3.1 What “heat” means here

- **NVE:** Track **total energy** \(E = K + U\) drift vs time (should stay within numerical + timestep error).
- **NVT:** Track **instantaneous temperature** \(T = 2 K / (k_B N_{\mathrm{df}})\) (or equivalent), **rolling average** \( \langle T \rangle \), and optionally **thermostat coupling** metadata (friction \(\gamma\), `jax.random` key splitting for reproducibility).

### 3.2 What Prolix already records

- `SimulationState` / trajectory plumbing in `simulate.py` includes **`kinetic_energy`** and temperature-related setup for **NVT Langevin** (see `NVTLangevinState`, `SimulationSpec.temperature_k`, logging of `kT`).
- Explicit path uses **`make_langevin_step_explicit`** / SETTLE-related machinery in `batched_simulate.py` for **solvated** workflows (see [explicit_solvent_runbook](explicit_solvent_runbook.md)).

### 3.3 Gaps for “comprehensive” heat tracking

| Gap | Recommendation |
|-----|------------------|
| No **single** documented metric bundle for benchmarks | Standardize **CSV/JSON** fields: `step`, `time_ps`, `T_inst`, `K`, `U`, `E_tot`, `gamma`, `box`, `seed`. |
| No **automated** check that **NVT** \(\langle T \rangle \approx T_{\mathrm{target}}\) over N steps | Addressed by **slow** test `test_explicit_pbc_nvt_mean_temperature_targets_spec` (statistical tolerance). |
| No **OpenMM-side** duplicate run for thermostat comparison | Optional: `scripts/benchmarks/openmm_langevin_temperature_stats.py` reports **mean T** / spread on a tiny PME system (distribution-level, not trajectory match). |
| **Heat bath work** / stochastic heat flow not decomposed | Research-grade; defer unless needed for method papers—document that only **total energy** and **T** are tracked. |

---

## 4. Benchmarking comparable to OpenMM

### 4.1 What we have

| Artifact | Compares | Physics? |
|----------|----------|----------|
| `scripts/benchmarks/prolix_vs_openmm_speed.py` | Prolix **JIT energy+grad** vs OpenMM **getState(E, F)** on **minimal explicit PME** two-charge box; optional OpenMM **step**; `--json` includes **schema 1.0** fields | **Throughput only**; not parity |
| `scripts/benchmarks/prolix_vs_openmm_t1_solvated.py` | Same as T0 with **tier T1** JSON + larger default **N** (`--charge-copies`, `--box`) | Throughput tier; not a full solvated peptide build |
| `scripts/benchmarks/openmm_langevin_temperature_stats.py` | OpenMM **LangevinMiddleIntegrator** mean **T** on four-charge PME box | Reference thermostat **statistics** only |
| `benchmarks/benchmark_scaling.py`, `benchmark_batched.py` | Prolix internal scaling | No OpenMM |
| Engaging `bench_chignolin` pipeline | **Implicit** NL vs dense + pytest PBC + minimal PME speed | Mixed; **not** full explicit protein |

### 4.2 What’s needed for explicit-solvent **comparability**

1. **Matched workload**
   - Same **N**, **cutoff**, **PME grid**, **precision** (mixed vs double).
   - Prolix: `jax.jit` **energy + forces** (same as parity tests) or documented **single MD step** cost if/when matched to OpenMM’s integrator.
   - OpenMM: `getState(getEnergy=True, getForces=True)` and/or **`integrator.step(1)`** with the **same** constraint model.

2. **Standard explicit benchmark systems (suggested)**

   | Tier | System | Purpose |
   |------|--------|---------|
   | **T0** | Minimal PME box (current script) | Regression throughput; no FF ambiguity |
   | **T1** | Small **solvated** peptide in periodic box | NL + PME + water; still fits CI with `slow` |
   | **T2** | Production-sized bucket (e.g. DHFR-sized) | Cluster-only; strong scaling |

3. **Reporting**
   - Always log: **JAX devices**, **`jax_enable_x64`**, **OpenMM platform**, **OpenMM precision**, **GPU model**, **atom count**, **cutoff**, **PME settings**.
   - Store JSON alongside human-readable table (already supported by `prolix_vs_openmm_speed.py --json`).

---

## 5. Consolidated checklist (use for releases)

**Static parity**

- [ ] `pytest tests/physics/test_openmm_explicit_anchor.py` (`openmm` extra)
- [ ] `pytest tests/physics/test_pbc_end_to_end.py -m "not slow"` (fast gate)
- [ ] `pytest tests/physics/test_explicit_validation_expansion.py`
- [ ] Periodic full sweep: `pytest -m slow` including `test_protein_nl_explicit_parity`, `test_solvated_explicit_integration`, `test_explicit_slow_validation`

**Heat / dynamics**

- [ ] NVE short test passes; tighten tolerances when possible
- [ ] NVT: log **T** and **K** in benchmark JSON for explicit runs
- [ ] (Stretch) Mean **T** vs target over fixed windows

**OpenMM-comparable benchmarks**

- [ ] `prolix_vs_openmm_speed.py` on **T0** with `--json` archived
- [ ] (When ready) **T1** solvated script mirroring OpenMM **Modeller + ForceField** setup

**Documentation**

- [ ] This page + [explicit_solvent_benchmarks](explicit_solvent_benchmarks.md) + runbook cross-links stay in sync when adding scripts

---

## Related

- [explicit_solvent_benchmarks](explicit_solvent_benchmarks.md) — local vs cluster matrix
- [explicit_solvent_runbook](explicit_solvent_runbook.md) — production `SimulationSpec` path
- [current_implementation](current_implementation.md) — test index and limitations
- `.agent/docs/plans/explicit_solvent_gaps_and_espaloma_charges.md` — charges / Espaloma boundary
