# Explicit solvent: current implementation (as-built)

This page is a **snapshot of what exists in the repository today**—modules, tests, and known gaps. For the forward-looking roadmap and phase definitions, see [explicit_solvent_implementation_plan](explicit_solvent_implementation_plan.md). For a chronological log, see [explicit_solvent_progress](explicit_solvent_progress.md).

## Scope

**Explicit solvent** here means periodic boundary conditions (PBC), particle-mesh Ewald (PME) reciprocal electrostatics, erfc-damped direct space, Lennard-Jones with exclusions, and shared long-range corrections—not implicit Generalized Born (GB).

**Implicit solvent** (GBSA, etc.) uses different code paths in `physics.system.make_energy_fn`; neighbor-list GB parity with dense GB is still incomplete (see Limitations).

## Implemented modules (as-built)

| Area | Role | Key paths |
|------|------|-----------|
| SPME reciprocal + forces | Grid-based PME with `custom_vjp` | `src/prolix/physics/pme.py` |
| Total energy / forces | Dense and JAX-MD neighbor-list nonbonded; PME wiring; `explicit_corrections` (PME exclusion + LJ tail) | `src/prolix/physics/system.py` |
| Exclusions for NL | `ExclusionSpec`, sparse maps, `max_exclusion_slots_needed` | `src/prolix/physics/neighbor_list.py` |
| Flash explicit | Tiled nonbonded + same PME/corrections as system | `src/prolix/physics/flash_explicit.py` |
| Padded batch energy | `single_padded_energy` aligned with explicit PME + corrections when `implicit_solvent=False` | `src/prolix/batched_energy.py` |
| Periodic space | Displacement / wrapping helpers | `src/prolix/physics/pbc.py` |
| Simulation spec | `use_pbc`, PME params, optional neighbor list | `src/prolix/simulate.py` (`SimulationSpec`) |
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

**CI:** Default fast runs use `pytest -m "not slow"` (see `pyproject.toml` `[tool.pytest.ini_options]`). Slow tests include full-protein NL/dense parity and solvated-PDB checks.

## Known limitations

- **Implicit GB + neighbor list:** Dense GB uses exclusion masks; the neighbor-list GB path does not yet mirror that policy. Tracked in-code as `TODO(implicit_GB_NL)` in `src/prolix/physics/system.py`. Do not assert implicit GB NL vs dense parity until fixed.
- **Electrostatic alternatives:** No reaction-field (RF) or damped shifted force (DSF) module (`electrostatic_methods.py` not present)—Phase 3 of the design plan.
- **Spatial sorting:** No Morton / Z-order reordering pass in `src/prolix` for PME scatter or cell lists—Phase 6 of the design plan.
- **Design doc vs repo:** Phase 8 cluster benchmark matrix and `scripts/benchmarks/explicit_solvent_bench.py` are planning references; see “Interim benchmarks” below.

## Interim benchmarks (repository today)

The design plan’s Phase 8 (SLURM GPU matrix, preemptable checkpointing) is **not** implemented as a single script. The repo does contain **ad hoc** verification and scaling scripts under `benchmarks/` (project root), including for example:

- `benchmarks/verify_pbc_end_to_end.py` — PBC / electrostatics checks
- `benchmarks/benchmark_scaling.py`, `benchmarks/benchmark_batched.py` — scaling experiments
- `benchmarks/compare_jax_openmm_validity.py`, `benchmarks/verify_end_to_end_physics.py` — cross-engine or integration checks

Treat these as **development utilities**, not the Phase 8 “production benchmark matrix” until documented and unified.

---

## Roadmap follow-up: Phases 3, 4, 6, 8, 9

This section tracks **remaining work** aligned with [explicit_solvent_implementation_plan](explicit_solvent_implementation_plan.md). It does not change phase status in `explicit_solvent_progress.md` unless that file is updated separately.

### Phase 3 — RF + DSF alternatives

**Status:** Not implemented.

**Suggested approach:** Add `src/prolix/physics/electrostatic_methods.py` (or equivalent) with an enum `PME | RF | DSF` and pairwise direct-space models. **Wire first** into `make_energy_fn` / `system.py` (production path used by `simulate.py`); optionally extend `cell_nonbonded.py` if cell-list SR becomes primary.

**Tests:** Small-system reference parity (OpenMM `CustomNonbondedForce` or analytic limits), consistent with the anchor test philosophy.

### Phase 4 — Solvation pipeline enhancement

**Status:** Partial (`solvate_protein`, `merge_solvated_topology`, `pad_solvated_system` exist).

**Follow-ups:**

- [ ] OPC3 (or chosen model) water box assets and parameter parity vs design checklist
- [ ] SETTLE / constraint integration completeness in `batched_simulate.py` (audit vs design)
- [ ] **Integration test:** merged solvated topology → `PaddedSystem` → explicit `make_energy_fn` energy (suggested: dedicated pytest, optionally `slow`)

### Phase 6 — Spatial sorting + tuning

**Status:** Not implemented.

**Decision point:** Optimize **PME charge spreading** (scatter order), **cell-list** assignment, or **both**—prefer profiling before large investment.

**Follow-ups:**

- [ ] Morton or cell-index sort for hot paths (if profiling warrants)
- [ ] Cell overflow / resize policy for `cell_list` if used at scale
- [ ] Benchmark sorted vs unsorted (design targets N ~ 40k–100k; cluster-oriented)

### Phase 8 — Benchmarking

**Status:** Not completed per progress table; interim scripts exist (see above).

**Follow-ups:**

- [ ] Document cluster job template (partition, GPU type, preemptable checkpointing) as in the implementation plan
- [ ] Optional: small **local** repeatable benchmark script for smoke throughput (subset of design matrix)
- [ ] Optional: `scripts/benchmarks/explicit_solvent_bench.py` entry point when ready (name referenced in design doc)

### Phase 9 — Production integration

**Status:** Not completed; `simulate.py` already exposes explicit PBC + PME + neighbor list.

**Follow-ups:**

- [ ] Documented end-to-end “happy path”: structure → solvate → minimize → MD (config / `SimulationSpec`)
- [ ] Stable defaults and error messages for incompatible option combinations
- [ ] Coordination with Phase 5/5b (RESPA, barostat) when those land—out of scope for this snapshot unless expanded

---

## Design review summary (oracle cycles)

Multi-axis review of this roadmap (correctness, completeness, feasibility, risk, alignment) concluded:

1. Phase 3 wiring must target **`system.py` / `make_energy_fn` first**, not only `cell_nonbonded.py`.
2. Phase 4 closure (integration test) should precede claiming Phase 9 “production ready.”
3. Phase 6 scope should stay **profiling-driven** unless a clear bottleneck is identified.
4. Phase 8 should distinguish **local smoke** benchmarks from **cluster full matrix**.

**Verdict:** Approved for documentation and phased implementation; residual risk is environment-specific GPU benchmarking.
