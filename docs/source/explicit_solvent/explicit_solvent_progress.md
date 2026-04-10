# Explicit Solvent: Progress Tracker

**As-built snapshot (modules, tests, roadmap follow-ups for phases 3–4, 6, 8–9):** [current_implementation](current_implementation.md).

## Status Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Cell-List SR Engine | ✅ Complete | 19/19 tests passing |
| Phase 2: PME Reciprocal Space | ✅ Kernel + integration | `pme.py` SPME; wired in `physics.system.make_energy_fn`, `flash_explicit.py`, shared corrections in `explicit_corrections.py` |
| Phase 3: RF + DSF Alternatives | ⬜ Not Started | |
| Phase 4: Solvation Pipeline | 🔶 Partial | `solvate_protein` patched, `pad_solvated_system` working, `flash_explicit_forces` implemented |
| Phase 5: HMR + RESPA | 🔨 HMR Complete | 15/15 tests passing, RESPA not started |
| Phase 5b: MC Barostat | ⬜ Not Started | |
| Phase 6: Spatial Sorting | ⬜ Not Started | |
| Phase 7: Validation L1 | ✅ In progress | Anchor test: `tests/physics/test_openmm_explicit_anchor.py`; broader cases in `tests/physics/test_pbc_end_to_end.py` |
| Phase 7: Validation L2 | 🔶 In Progress | BAOAB+SETTLE integrator implemented, validation script pending |
| Phase 8: Benchmarking | ⬜ Not Started | engaging cluster SLURM jobs |
| Phase 9: Production | ⬜ Not Started | |

---

## Explicit PME wiring (current)

Production explicit electrostatics use:

- **`physics.system.make_energy_fn`** (`simulate.py` path): SPME via `make_spme_energy_fn`, background term, **PME exclusion correction** and **LJ dispersion tail** via `explicit_corrections.py`.
- **`flash_explicit.py` / `flash_explicit_forces`**: same physics alignment for the FlashMD padded path (`_total_energy_fn`).
- **`pme.py`**: custom SPME with `jax.custom_vjp` (`spme_energy_with_forces`), `make_spme_energy_fn`, `make_pme_energy_fn` (bound-charges helper for tests).

**Follow-on validation** (not yet covered by the anchor test alone):

- Neighbor-list vs dense parity for nonbonded (protein topologies: `tests/physics/test_protein_nl_explicit_parity.py`; implicit GB + NL still deferred — see `TODO(implicit_GB_NL)` in `physics/system.py`).
- `SimulationSpec` / FIRE neighbor refresh (integration-level follow-on).
- FlashMD vs `system.make_energy_fn` on identical `PaddedSystem` / positions.
- `batched_energy.single_padded_energy` completeness vs `system` when using explicit PME + corrections (audit if used for production energy).
- L2: short trajectory vs OpenMM; script `scripts/validation/validate_explicit_l2.py` (see session log below) when GPU resources allow.

### R2C FFT Memory Optimization (Confirmed ✅)

The PME implementation already uses `jnp.fft.rfftn` / `jnp.fft.irfftn` (R2C FFTs)
which halve the Z-dimension, yielding ~50% FFT memory + compute savings vs full
complex FFTs. This is documented in:
- `pme.py:1-11` (module docstring)
- `pme.py:387` (`rfftn`), `pme.py:396` (`irfftn`)
- `pme.py:274` (influence function uses `rfftfreq` for half-Z)
- `gpu_optimization_strategies.md:170, 215`
- `explicit_solvent_implementation_plan.md:148, 156`

---

## Session Log

### Session 1: 2026-03-30 evening (autonomous)

**Phase**: Plan finalization + Phase 1 kickoff

**Completed**:
- [x] Oracle critique produced (REVISE → 2 critical resolved)
- [x] Implementation plan updated with all critique resolutions
- [x] HMR/RESPA confirmed as optional config flags
- [x] Benchmarks specified for mit_preemptable with preemptable-safe checkpointing
- [x] PME strategy confirmed: custom_vjp from the start
- [x] Autonomous mode rules updated (no notify_user, background commands)
- [x] `cell_list.py` written (~240 LOC)
  - CellList NamedTuple data structure
  - compute_grid_shape(), compute_cell_size()
  - build_cell_list() with ghost atom sanitization contract
  - 13 half-shell shift vectors
- [x] `cell_nonbonded.py` written (~320 LOC)
  - _cell_pair_energy() core pairwise kernel
  - cell_energy_scan() — Option A (lax.scan over 27 shifts)
  - cell_energy_grid_shift() — Option B (13 parallel jnp.roll)
  - ewald_exclusion_correction() — Layer 2 exclusion architecture
- [x] `test_cell_list.py` written (~300 LOC)
  - Grid shape, ghost sanitization, overflow detection tests
  - Scan vs grid-shift parity test
  - Analytical LJ comparison
  - Ewald exclusion tests

**In Progress**:
- [x] All 19 tests passing (8.58s)
- [x] Fixed NaN bug: ghost-atom self-interaction at dist~0 caused 0*inf=NaN.
      Fix: masked pairs get `safe_dist=1.0` before LJ/Coulomb computation.

**Blockers**: None

**Next Steps**:
1. ~~Commit Phase 1 core to feat/explicit-solvent~~ Deferred — `../prolix/` is
   a submodule and `feat/explicit-solvent` is locked by cluster worktree.
   Will commit when terminal stabilizes. Files are written and tested.
2. Wire cell-list into batched_energy.py dispatch
3. Begin Phase 2 (PME rewrite) — critical path continues

---

### Session 1 (continued): Phase 2 kickoff

**Phase**: PME reciprocal space rewrite

**Completed**:
- [x] `pme.py` full rewrite (~590 LOC):
  - B-spline order-4 charge spreading + derivatives
  - R2C FFT via `jnp.fft.rfftn/irfftn` (halves Z-dim)
  - Influence function with proper B-spline modulation
  - Grid dim selection with {2,3,5,7} factorizability constraint
  - `spme_reciprocal_energy()` — full reciprocal-space energy
  - `spme_self_energy()` — analytical self-correction
  - `spme_energy_with_forces()` — `jax.custom_vjp` wrapper
  - Analytical gradient via B-spline derivative interpolation on θ grid
  - `make_spme_energy_fn()` — factory API
- [x] `test_pme.py` (~280 LOC):
  - Grid dim factorizability and resolution tests
  - B-spline partition of unity, symmetry, non-negativity, derivative sum=0
  - Charge conservation on grid
  - Self-energy analytical formula verification
  - Reciprocal energy stability (finite, reasonable magnitude)
  - custom_vjp gradient vs numerical finite-difference comparison
  - Force direction correctness (pulls opposite charges together)
- [x] 20/20 PME tests passing (8.54s)

**Next Steps**:
1. Phase 3 (RF/DSF alternatives) — low priority, defer
2. Phase 4 (solvation pipeline) — prepare OPC3 water + topology
3. Phase 5 (HMR/RESPA) — optional performance optimization

---

### Session 2: 2026-04-01 (L2 trajectory validation)

**Phase**: P4 solvation glue + P7 Level 2 trajectory validation

**Completed**:
- [x] `PaddedSystem` updated: added `box_size: Array | None = None` field (backward-compatible)
- [x] `solvate_protein()` patched: fixed `Protein` dataclass schema mismatch (removed `.physics` attribute access)
- [x] `single_padded_force()` updated: added `explicit_solvent=True` dispatch path → `flash_explicit_forces`
- [x] `make_langevin_step_explicit()` implemented in `batched_simulate.py` (~160 LOC):
  - BAOAB Langevin integration with O-step thermostat
  - Protein X-H RATTLE constraints (via existing `project_positions`/`project_momenta`)
  - Water SETTLE rigid geometry (via existing `settle_positions`/`settle_velocities`)
  - PBC box wrapping for whole water molecules
  - Velocity limiting, displacement capping, force capping safety guards
  - Quality gate accumulation (vlimit, force_cap, constraint_violation, dx_cap)
- [x] Temperature/restraint schedule helpers added to `batched_simulate.py`
- [x] TDD test written: `tests/physics/test_explicit_trajectory.py`
- [x] Docs: `docs/openmm_parity/` created with L1 single-point parity methodology

**Design Decisions**:
- **Drift control**: Measure-first approach with pure float32. Kahan summation / periodic
  recentering will be implemented only if energy drift exceeds 0.5 kcal/mol/ns.
- **Electrostatics for L2**: Dense Coulomb (no cutoff, `pme_alpha=0`). Physically exact
  for small systems. PME wiring deferred to pre-L3.
- **Minimization**: Reuse existing `run_minimization(energy_fn, positions)` from
  `physics/simulate.py` with `single_padded_energy(sys, disp_fn, implicit_solvent=False)`
  as the energy function. No new minimization code needed.

**In Progress**:
- [ ] `scripts/validation/validate_explicit_l2.py` — orchestration script wiring all components
- [ ] GPU validation on engaging cluster (node4007, 48h SLURM sleep job allocated)
- [ ] Workspace synced to `/home/maarxaru/projects/noised_cb_explicit_solv/`

**Blocking**:
- Test runs on CPU are too slow for solvated systems; need GPU (engaging cluster)
- ~~`energy_fn` NameError in production scan~~ **Fixed**: replaced with direct `flash_explicit_energy` call (2026-04-02)

**Next Steps**:
1. Commit all pending changes (314 lines across 21 files)
2. rsync to engaging, submit `run_l2.sh` via SLURM
3. Measure: energy drift, SETTLE fidelity, temperature stability
4. If L2 passes → wire PME into force pipeline for L3

---

## File Inventory

| File | LOC | Phase | Status |
|------|-----|-------|--------|
| `../prolix/src/../prolix/physics/cell_list.py` | ~240 | 1 | Written, testing |
| `../prolix/src/../prolix/physics/cell_nonbonded.py` | ~320 | 1 | Written, testing |
| `tests/test_cell_list.py` | ~300 | 1 | Written, running |
| `../prolix/src/../prolix/physics/pme.py` | ~590 | 2 | Complete, tested |
| `tests/test_pme.py` | ~280 | 2 | 20/20 passing |
| `../prolix/src/../prolix/physics/hmr.py` | ~210 | 5 | Complete, tested |
| `tests/test_hmr.py` | ~250 | 5 | 15/15 passing |
| `../prolix/src/../prolix/physics/flash_explicit.py` | ~178 | 4 | Complete, tested via L1 parity |
| `../prolix/src/../prolix/physics/electrostatic_methods.py` | ~100 | 3 | Not started |
