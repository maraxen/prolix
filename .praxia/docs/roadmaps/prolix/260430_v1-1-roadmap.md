# Prolix v1.1 Roadmap

**Oracle Decision Date**: 2026-04-30  
**Status**: APPROVED — Sprint B closed, all items unblocked  
**Strategic Goals**:
1. Seamless JAX pipeline integration for protein design/analysis
2. StableHLO → WASM/WebGPU export for browser deployment

---

## Context: What Changed at v1.0 → v1.1 Boundary

### Sprint B Closed (2026-04-30)

The NVT temperature offset (+42 to +115 K at N=8–64 waters) was a **slow-equilibration artifact** from gas-phase initial conditions (10 Å grid spacing), not a thermostat bug. Evidence:

- γ=50/ps brings prolix BAOAB to 302.3 K from the same bad initial conditions
- OpenMM LFMiddle gave 305 K at γ=1/ps (faster algorithm, not a prolix fix)
- `_equil_water_positions` added; temperature tests confirmed passing

**Implication**: BAOAB thermostat, OU formula, SETTLE constraints, and KE estimator are all correct. No thermostat fix needed.

### ADR-005 Items Dropped / Deferred

Oracle explicitly ruled out at the v1.1 planning review:

- **LFMiddle (ADR-005 Phase 3)**: DROPPED. H1 (integration error) falsified by Sprint B. LFMiddle offers no advantage over BAOAB for the dt ≤ 0.5 fs constraint; the constraint is from SETTLE+thermostat coupling, not discretization order.
- **Constraint-aware thermostat (Phase 5)**: DEFERRED to v2.0+. Requires integrator-thermostat redesign; not on critical path for export goals.
- **NHC chain integration (Phase 7)**: DEFERRED to v2.0+. Lower priority than export work.

---

## v1.1 Work Items (Priority Order)

### Item 1: `closure→explicit-params` Refactor

**Goal**: Add `energy_fn_pure(params, positions, box)` parallel API alongside existing closure-based `energy_fn(positions, box)`.

**Why**: `jax.export` requires pure functions with explicit parameters — closures over Python objects (charges, sigmas, box) cannot be traced to portable StableHLO artifacts. This is the direct unblock for Item 4.

**Scope**:
- Add `make_energy_fn_pure(sys_dict) → (params, fn)` factory in `system.py`
- `fn(params, positions, box)` is fully explicit, JIT-compilable, exportable
- Keep existing `make_energy_fn(...)` closure API unchanged (backward compat)
- Unit test: `jax.export(fn).call(params, positions, box)` succeeds without error

**Estimated effort**: 1–2 days  
**Blocks**: Item 4 (prolix.export)

---

### Item 2: Fix / Retire `batched_equilibrate` NaN Bug

**Goal**: Either fix the NaN in `batched_equilibrate` or document a clean replacement and remove the broken path.

**Why**: v1.0 ships with a documented workaround (cold-start init instead of `batched_equilibrate`). This is tech debt that blocks reliable batched production runs. The CLAUDE.md safe pattern is a workaround, not a fix.

**Scope**:
- Trace NaN source in `batched_simulate.py::batched_equilibrate`
- If fixable in < 1 day: fix and add regression test
- If structural: deprecate `batched_equilibrate`, remove from public API, promote cold-start pattern to first-class with proper docs
- Update CLAUDE.md Safe Pattern section accordingly

**Estimated effort**: 1 day  
**Blocks**: nothing, but cleans up v1.0 known limitation #3

---

### Item 3: kUPS Convention Adapter

**Goal**: Align prolix propagator signatures with kUPS so kUPS can be used as a drop-in reference backend for cross-validation.

**Why**: kUPS (eV/Å/amu units) is now a validated external reference for thermostat behavior. Keeping a thin adapter allows ongoing cross-validation without ad-hoc unit conversion in each test.

**Scope**:
- `src/prolix/physics/kups_adapter.py`: unit conversion constants + thin wrapper
- Align: force units (eV/Å → kcal/mol/Å), time units (AKMA → fs), mass (amu)
- Update `tests/physics/test_kups_thermostat_crossval.py` to use adapter

**Estimated effort**: 1 day  
**Blocks**: nothing independently, but enables cleaner future validation

---

### Item 4: `prolix.export` Module (StableHLO)

**Goal**: Exportable force/energy and Langevin step artifacts for WASM/WebGPU deployment.

**Why**: Primary strategic goal. jax.export spike already PASSED (JAX 0.8.1, 2708 bytes clean StableHLO for LJ energy). Full module gates the WASM/WebGPU path.

**Scope**:
- `src/prolix/export.py` module with:
  - `export_energy_fn(params, example_positions, example_box) → StableHLO artifact`
  - `export_langevin_step(params, example_state) → StableHLO artifact`
  - `save_artifact(artifact, path)` / `load_artifact(path)` helpers
- Requires Item 1 (explicit-params API) as input
- Integration test: round-trip export → load → call produces same output as jit call
- Documentation: `docs/EXPORT_GUIDE.md` with WASM/WebGPU deployment notes

**Estimated effort**: 2–3 days  
**Depends on**: Item 1

---

## Deferred to v2.0+

| Item | Reason |
|------|--------|
| LFMiddle integrator | H1 falsified; no dt improvement expected; drops from roadmap |
| Constraint-aware thermostat | Requires integrator redesign; not on critical path |
| NHC chain (settle_with_nhc) | Lower priority than export goals |
| NPT long-trajectory stability | Active investigation but not blocking v1.1 goals |
| Large-scale SETTLE batching | v1.0 smoke-tested; full validation deferred |

---

## Sprint B Closure Deliverables (Completed 2026-04-30)

These close Sprint B and are already merged:

- [x] `_equil_water_positions(n_waters, seed)` added to `test_explicit_langevin_tip3p_parity.py`
  - Loads `data/water_boxes/tip3p.npz` (30 Å cube, 895 waters, liquid density)
  - TODO in code: high-γ thermalization option for custom N/box
- [x] `_mean_rigid_t_after_burn` and `_mean_rigid_t_csvr_after_burn` updated to use `_equil_water_positions`
- [x] Temperature control tests passing with equilibrated box
- [x] Sprint B verdict: initialization artifact, thermostat correct

---

## Sequencing

```
Item 2 (batched_equilibrate fix)   ──── independent, can start anytime
Item 3 (kUPS adapter)              ──── independent, can start anytime
Item 1 (closure→params refactor)   ──→  Item 4 (prolix.export)
```

Recommended order: **Item 1 → Item 4** as the critical path (export is the strategic goal).  
Items 2 and 3 can run in parallel or as warm-up tasks.

---

## Success Criteria for v1.1 Release

| Criterion | Target |
|-----------|--------|
| `make_energy_fn_pure` ships | `jax.export(fn)` succeeds on 8-water system |
| `batched_equilibrate` resolved | No open NaN bug in batch init path |
| `prolix.export` module ships | Round-trip export→load→call passes |
| kUPS adapter present | `test_kups_thermostat_crossval.py` uses adapter, no raw unit math |
| Temperature tests | All pass with `_equil_water_positions` |
| No regressions | Full test suite green |
