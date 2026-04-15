# Explicit solvent: current implementation (as-built)

This page is a **snapshot of what exists in the repository today**—modules, tests, and known gaps. For the forward-looking roadmap and phase definitions, see [explicit_solvent_implementation_plan](explicit_solvent_implementation_plan.md). For a chronological log, see [explicit_solvent_progress](explicit_solvent_progress.md).

## Scope

**Explicit solvent** here means periodic boundary conditions (PBC), particle-mesh Ewald (PME) reciprocal electrostatics, erfc-damped direct space, Lennard-Jones with exclusions, and shared long-range corrections—not implicit Generalized Born (GB).

**Implicit solvent** (GBSA, etc.) uses different code paths in `physics.system.make_energy_fn`. The neighbor-list GB implementation applies the same dense Born/energy **masks** as the \(N^2\) path via per-neighbor gathers (`compute_gb_energy_neighbor_list`); long-range GB polarization is still **approximate** when the neighbor **cutoff** omits pairs that the dense sum would include (see Limitations).

## Implemented modules (as-built)

| Area | Role | Key paths |
|------|------|-----------|
| SPME reciprocal + forces | Grid-based PME with `custom_vjp` | `src/prolix/physics/pme.py` |
| Total energy / forces | Dense and JAX-MD neighbor-list nonbonded; PME wiring; optional **reaction-field** or **damped shifted** direct Coulomb (opt-in); `explicit_corrections` (PME exclusion + LJ tail) | `src/prolix/physics/system.py`, `src/prolix/physics/electrostatic_methods.py` |
| Exclusions for NL | `ExclusionSpec`, sparse maps, `max_exclusion_slots_needed` | `src/prolix/physics/neighbor_list.py` |
| Flash explicit | Tiled nonbonded + same PME/corrections as system | `src/prolix/physics/flash_explicit.py` |
| Padded batch energy | `single_padded_energy` aligned with explicit PME + corrections when `implicit_solvent=False` | `src/prolix/batched_energy.py` |
| Periodic space | Displacement / wrapping helpers | `src/prolix/physics/pbc.py` |
| Simulation spec | `use_pbc`, PME params, optional neighbor list, **`electrostatic_method`** | `src/prolix/simulate.py` (`SimulationSpec`) |
| Solvation + merge | `solvate_protein`, `merge_solvated_topology`, `MergedTopology` | `src/prolix/physics/solvation.py`, `src/prolix/physics/topology_merger.py` |
| Padding | `PaddedSystem`, `pad_protein`, `pad_solvated_system` | `src/prolix/padding.py` |
| Cell-list (optional) | Spatial decomposition for tiled SR kernels; **not** the default production MD path | `src/prolix/physics/cell_list.py`, `src/prolix/physics/cell_nonbonded.py` |

The cell-list stack is documented in-module as secondary to **JAX-MD neighbor lists** + `make_energy_fn` for production MD.

## Validation and tests

| Test file | What it covers |
|-----------|----------------|
| `tests/physics/test_openmm_explicit_anchor.py` | Minimal two-charge PME vs OpenMM Reference |
| `tests/physics/test_pbc_end_to_end.py` | PBC/PME energy and forces vs OpenMM (multiple configs) |
| `tests/physics/test_explicit_validation_expansion.py` | Internal parity: NL vs dense, Flash vs system, `single_padded_energy` vs system |
| `tests/physics/test_protein_nl_explicit_parity.py` | Protein `ExclusionSpec` + PBC + explicit PME: NL vs dense (slow tests); fast guard on exclusion slot counts |
| `tests/physics/test_explicit_slow_validation.py` | Short explicit NVE sanity; optional OpenMM four-particle check (`slow` / `openmm`) |
| `tests/physics/test_solvated_explicit_integration.py` | **Solvated** merged topology → `pad_solvated_system` → finite explicit NL energy (`slow`) |
| `tests/physics/test_electrostatic_methods_openmm.py` | Optional reaction-field energy vs OpenMM Reference (`openmm`) |

**CI:** Default fast runs use `pytest -m "not slow"` (see `pyproject.toml` `[tool.pytest.ini_options]`). Slow tests include full-protein NL/dense parity and solvated integration. OpenMM-marked tests require `openmm` (optional dev extra).

## Partial charges (Espaloma)

- **Assignment lives in proxide:** Optional **Espaloma Charge** (AM1-BCC surrogate) is exposed from **proxide** (`proxide.chem.partial_charges`, optional dependency group `espaloma` in proxide’s `pyproject.toml`). Inference runs outside JAX; output is a NumPy charge vector in **atom order** matching the RDKit / `Molecule` pipeline. Golden regression vectors live under **proxide** `tests/data/espaloma_golden/` (see that README for bump policy).
- **Prolix consumes arrays only:** `make_energy_fn` and `simulate` take `Protein.charges` (or hierarchical `AtomicConstants`) with no ML dependency. A lightweight consistency test is `tests/test_injected_charges_energy.py` (scaled charges change explicit energy).

## Known limitations

- **Implicit GB + neighbor list:** The neighbor-list GB path applies the same `(N,N)` Born and energy masks as the dense path via gathers onto `(N,K)` neighbors (`compute_gb_energy_neighbor_list`). Coverage is still limited by the neighbor **cutoff** unless the list is built to be effectively complete.
- **Electrostatic alternatives:** Reaction-field (OpenMM CutoffPeriodic-style) and damped shifted Coulomb are **opt-in** via `ElectrostaticMethod` in `physics/electrostatic_methods.py` and `make_energy_fn` / `SimulationSpec`. Default remains **PME**.
- **Spatial sorting:** No Morton / Z-order reordering pass in `src/prolix` for PME scatter or cell lists by default — **profiling-first** policy is documented in [spatial_sorting_profile_gate](spatial_sorting_profile_gate.md).
- **Water box assets:** `data/water_boxes/tip3p.npz` ships for TIP3P solvation. OPC3 parameters live in `physics/water_models.py`; a pre-equilibrated `opc3.npz` is **not** required for tests that use TIP3P only.

## Interim benchmarks (repository today)

The design plan’s Phase 8 (SLURM GPU matrix, preemptable checkpointing) is documented as **smoke vs cluster** in [explicit_solvent_benchmarks](explicit_solvent_benchmarks.md). For a **requirements matrix** (parity layers, heat tracking, OpenMM-comparable throughput), see [explicit_solvent_parity_and_benchmark_requirements](explicit_solvent_parity_and_benchmark_requirements.md). Ad hoc scripts remain under `benchmarks/` and `scripts/benchmark_*.py`. **Prolix vs OpenMM throughput** (explicit minimal PME, per-platform sweep): [`scripts/benchmarks/prolix_vs_openmm_speed.py`](../../../scripts/benchmarks/prolix_vs_openmm_speed.py) — throughput only, not physics validation.

---

## Roadmap follow-up: Phases 3, 4, 6, 8, 9

Aligned with [explicit_solvent_implementation_plan](explicit_solvent_implementation_plan.md). Formal plan: `.agent/docs/plans/explicit_solvent_phases_3_4_6_8_9.md`.

### Phase 3 — RF + DSF alternatives

**Status:** **Implemented (opt-in).** `ElectrostaticMethod` in `src/prolix/physics/electrostatic_methods.py`; `make_energy_fn` and `SimulationSpec` accept `electrostatic_method`, `reaction_field_dielectric`, `dsf_alpha`. RF is validated vs OpenMM Reference in `tests/physics/test_electrostatic_methods_openmm.py`. Default remains PME.

### Phase 4 — Solvation pipeline enhancement

**Status:** **Partial → integration test added.** `solvate_protein`, `merge_solvated_topology`, `pad_solvated_system` exist. **Integration test:** `tests/physics/test_solvated_explicit_integration.py` (small solvated box, `slow`). **SETTLE:** implemented on the batched explicit integrator path (`make_langevin_step_explicit` in `batched_simulate.py`); not a Phase 4 engineering blocker.

**Water models:** TIP3P box asset present; OPC3/TIP3P parameters registered in `water_models.py` (fast registry test in same file).

### Phase 6 — Spatial sorting + tuning

**Status:** **Profile gate documented** — [spatial_sorting_profile_gate](spatial_sorting_profile_gate.md). No default Morton pass until profiling warrants it.

### Phase 8 — Benchmarking

**Status:** **Documented** — [explicit_solvent_benchmarks](explicit_solvent_benchmarks.md) (local smoke vs cluster checklist, SLURM template, links to scripts).

### Phase 9 — Production integration

**Status:** **Runbook** — [explicit_solvent_runbook](explicit_solvent_runbook.md) (`SimulationSpec`, explicit defaults, RF/DSF caveats, SETTLE pointer, failure modes).

---

## Design review summary (oracle cycles)

Multi-axis review of this roadmap (correctness, completeness, feasibility, risk, alignment) concluded:

1. Phase 3 wiring targets **`system.py` / `make_energy_fn` first**; default PME unchanged for anchor tests.
2. Phase 4 closure (integration test) precedes claiming full production hardening for arbitrary solvated systems.
3. Phase 6 scope stays **profiling-driven** unless a clear bottleneck is identified.
4. Phase 8 distinguishes **local smoke** benchmarks from **cluster full matrix**.

**Verdict:** Approved for documentation and phased implementation; residual risk is environment-specific GPU benchmarking.
