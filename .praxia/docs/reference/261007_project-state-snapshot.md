---
title: "Project state snapshot"
description: "Verbatim relocation of CLAUDE.md blocks that record sprint status, release history, roadmap plans, and dated campaign results."
status: reference
date: '261007'
---

## CURRENT STATE

last-edit: 2026-06-01

ongoing-work: P1a MolecularBundle (bucketed JIT boundary). NPT KE init bug resolved
(settle.py:1944 `/mu` → `*mu`, 2026-06-01, commit b6e5bb9). LFMiddle campaign 89c9a900
concluded FALSIFIED (2026-06-01) — 46 runs, 0 passes, all dt values; see
v1.1_next_steps.md. Phase 5 (C3 AM conservation, 678c9cb) lifted the dt cap to
≤ 1.0 fs at production scale (n ≳ 16, gamma ≈ 10 ps⁻¹) — gate job 15870804 +
size sweep ba334c1f (2026-06-13). Residual small-N warm bias is translational
finite-size; see `.praxia/docs/research/260612_p5-dt1fs-size-crossover.md`.

  Current fix rounds up to the next `tile_size` multiple. PROPER FIX (bucketing, so tiling
  invariants always hold by construction) is on the backlog — see `.praxia/ideas.jsonl`.

## Research Ideas & Roadmap

 Current focus: stabilize core MD engine (Phase 2 constraints, NPT stability).

## Phase 2: Explicit Solvent Integration

**Status**: v1.0 Release; dt cap lifted to ≤ 1.0 fs at production scale (2026-06-13)  

### Temperature Control (NVT Mode)

With NVT configuration:
- Target temperature: 300 K
- Achieved stability: ±5 K over 50+ ps simulations
- No divergence or runaway heating observed
- *Note: This is NVT (constant volume). NPT long-trajectory use is not recommended in v1.0.*

### Future Improvements (v2.0+)

A constraint-aware thermostat that only couples to unconstrained DOF could eliminate the NVT dt limitation, allowing dt ≥ 1.0fs. A decoupled CSVR implementation may fix the NPT long-trajectory divergence by avoiding rigid-body KE feedback loops. See `.agent/docs/RELEASE_DECISION_v1.0.md` for detailed analysis and roadmap.

### Sprint 7: Batching + NPT Validation (v1.0)

**Safe_map fix**: Fixed reshape bug in `safe_map` that failed on heterogeneous pytrees (different leaf shapes). Added validation to require all pytree leaves have consistent batch dimension. (Step 2)

**LangevinState batching**: Updated `LangevinState.tree_flatten` to properly batch `warn_counts` field, ensuring consistent batched tree structure. (Step 2)

**Regression test**: Added `test_safe_map_heterogeneous_pytree()` to validate error handling for incompatible batched structures. (Step 3)

### Production Status

**v1.0**: settle_langevin validated and production-ready (NVT only). dt ≤ 1.0 fs at
production scale (n ≳ 16, gamma ≈ 10 ps⁻¹; gate 15870804 + sweep ba334c1f); dt ≤ 0.5 fs
for n ≲ 16 / weak friction.
**v1.1+**: ~~LFMiddle hypothesis test~~ (falsified), constraint-aware thermostat, NPT fix planned

## Phase 2–4: Integrator Modular Architecture (v1.0 Release)

**Status**: v1.0 Release with modular integrator factory (make_integrator)

### v1.0 Scope — What Ships

✅ **Phase 1 (Complete)**: Constraint system with ConstraintDOFMask
- Explicit DOF decomposition (rigid water vs free solute)
- Projection operators for constraint-aware dynamics
- Comprehensive unit tests (22 tests, all passing)

✅ **Phase 2 (Complete)**: Modular integrator factory with make_integrator
- Step primitives library (O, V, A, SETTLE_vel, CSVR, NHC)
- Step-sequence registry (composition pattern)
- make_integrator factory for BAOAB_LANGEVIN and BAOAB_CSVR_NPT
- Bitwise equivalence to settle_langevin validated (RMSD < 1e-12 Å)
- kUPS cross-validation passed (RMSD < 0.1 Å, KE drift < ±1%)

✅ **Phase 4 (Partial)**: Batching support via vmap
- Unconstrained batching (e.g., 16 parallel solute-only trajectories)
- Validated with machine-epsilon equivalence (RMSD < 1e-15 Å)
- Performance: 2–3x speedup for batch_size ∈ {4, 16}
- **SETTLE-path batching**: smoke test added (4 waters, 100 steps) as final v1.0 validation

### Known Limitations (v1.0)

- dt ≤ 1.0 fs at production scale (n ≳ 16, gamma ≈ 10 ps⁻¹); dt ≤ 0.5 fs for n ≲ 16 / weak friction (documented in Phase 2 section)
- NPT long-trajectory divergence (> 10 ps); use NVT for production (documented in Phase 2 section)
- Batched SETTLE: smoke-tested but not exhaustively validated at scale (see v1.1 roadmap)

---

## v1.1 Roadmap (Deferred Features)

### Phase 3: LFMiddle Optimization & dt-Sweep Hypothesis — ~~FALSIFIED~~ (2026-06-01)

**Result**: Hypothesis closed. Campaign 89c9a900 (46 runs, all dt ∈ {0.25, 0.5, 1.0} fs,
both lfmiddle and baoab control, system sizes 2–895 waters) produced 0 passes. Mean T
ranged from 13,818 K (dt=0.25) to 3.35×10⁵⁷ K (dt=1.0). LFMiddle O-step splitting does
not resolve SETTLE+thermostat coupling. The dt ≤ 0.5 fs constraint stands. **Phase 5
(constraint-aware thermostat) is the only known viable path to lifting it.**

*Note: small-system runs (n=2,64) were also confounded by the open tiling/exclusion-buffer
bug (backlog #746), making those results doubly unreliable. The 895-water production runs
show genuine thermal runaway independent of the tiling bug.*

### Phase 4 (Extended): Large-Scale Batched SETTLE Validation

**Objective**: Comprehensive validation of batched integrators on large water systems

**Deliverables**:
- 64-water batching equivalence test (full 10 ps trajectory)
- Constrained batching performance benchmarking
- Optional: batched kUPS cross-validation

**Rationale for deferral**: v1.0 includes smoke test (4 waters, 100 steps); large-scale testing deferred as optimization/validation phase

**Estimated effort**: 2–3 days

### Phase 5 (New): Constraint-Aware Thermostat — **dt cap lifted at production scale (2026-06-13)**

**Status**: The C3 AM-conservation correction (678c9cb) achieved stable dt=1.0 fs NVT at
production scale — gate job 15870804 (895 waters, T_rot 299.6 K) + size sweep ba334c1f.
The remaining sub-goal is the small-N regime: a translational finite-size warm bias
(n ≲ 16, only 3·N−3 translational DOF) and the weak-friction (gamma ≈ 1 ps⁻¹) case still
need dt ≤ 0.5 fs. See `.praxia/docs/research/260612_p5-dt1fs-size-crossover.md`.

**Objective**: Fix dt ≤ 0.5 fs limitation via constraint-aware thermostat that only couples to unconstrained DOF

**Rationale for promotion**: LFMiddle (Phase 3) falsified on 2026-06-01. Phase 5 is now
the *only known path* to lifting the dt constraint. Paper critical path depends on it
starting no later than after P1a MolecularBundle.

**Deliverables**: 
- Redesigned Langevin coupling (per-DOF vs global)
- Validation: dt ≥ 1.0 fs without divergence

**Estimated effort**: 4–6 days

### Phase 6 (New): NPT Long-Trajectory Stability

**Objective**: Fix NPT temperature runaway (> 10 ps) via decoupled CSVR implementation

**Deliverables**: 
- Modified CSVR that avoids rigid-water KE feedback loops
- Validation: 50+ ps NPT trajectory without divergence

**Rationale**: Requires detailed coupling analysis; beyond Phase 2 scope

**Estimated effort**: 3–5 days

### Phase 7 (New): Nosé-Hoover Chain Integration

**Objective**: Implement NHC_Step for settle_with_nhc chain thermostat

**Deliverables**: Full NHC state propagation, integration with make_integrator

**Rationale**: Lower priority than LFMiddle hypothesis and SETTLE batching

**Estimated effort**: 2–3 days

---

## v1.0 Release Notes

**Version**: Prolix v1.0 — Modular Integrator Architecture

### Highlights

- Pluggable constraint algorithms (ConstraintDOFMask layer)
- Reusable step primitives library (O, V, A, SETTLE, CSVR)
- Composition factory (make_integrator) enabling custom integrator sequences
- Batching support (unconstrained validated, SETTLE smoke-tested)
- Full backward compatibility with settle_langevin, settle_csvr_npt APIs
- kUPS cross-validation passed

### Breaking Changes

None. New APIs are additive.

### Known Limitations

1. dt ≤ 1.0 fs for NVT at production scale (n ≳ 16, gamma ≈ 10 ps⁻¹; gate 15870804 + sweep ba334c1f, 2026-06-13); dt ≤ 0.5 fs for n ≲ 16 or weak friction. LFMiddle hypothesis falsified (campaign 89c9a900, 2026-06-01); the cap was instead lifted by C3 AM conservation (678c9cb). Residual small-N warm bias is translational finite-size — see `.praxia/docs/research/260612_p5-dt1fs-size-crossover.md`.
2. NPT long-trajectory divergence beyond ~10 ps (use NVT for production). *KE init spike fixed (commit b6e5bb9, 2026-06-01).*
3. Batched SETTLE validated on small systems (4 waters, 100 steps); large-scale testing in v1.1

### v1.1 Priorities

- **Phase 5: Constraint-aware thermostat** (only remaining path to dt ≥ 1.0 fs; P1 after P1a)
- Large-scale SETTLE batching validation
- ~~LFMiddle hypothesis test~~ (falsified 2026-06-01)
