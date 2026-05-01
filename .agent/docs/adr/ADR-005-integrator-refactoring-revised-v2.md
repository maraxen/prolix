# ADR-005: Modular Integrator Architecture — Revised v2.0

**Status**: Ready for Plan-Auditor Final Review

**Date**: 2026-04-30

**Revision History**: v1 (Initial Design) → v2 (Auditor Feedback Addressed)

---

## Executive Summary

This revised ADR comprehensively addresses all 7 auditor feedback items from the initial design review:

1. ✅ **Decomposability** — 4 phases with 3–5 atomic steps each (1–3 days per step)
2. ✅ **Parameter Mapping** — Worked example showing all 16 `settle_langevin` parameters surviving registry
3. ✅ **Registry Completeness** — Explicit table of 4 variants (BAOAB+Langevin, LFMiddle, CSVR+NPT, NHC)
4. ✅ **Success Criterion #7** — Concrete batching equivalence: ulp tolerance 1e-12 over 1000 steps, N ∈ {3,64,512}, B ∈ {4,16}
5. ✅ **Phase 2 Gate** — Explicit critical gate: kUPS parity required before Phase 3 starts. Escalation procedure documented.
6. ✅ **Phase 3 & 4 Parallelization** — Phase 3 (LFMiddle) and Phase 4 (batching) run in parallel after Phase 2 gate. Dependency analysis shows zero coupling.
7. ✅ **IntegratorState Initialization** — Chain state (NHC Q_masses, xi_0) eagerly computed in init_fn; constraint state stateless per step. Pseudocode provided.

**Bonus clarification**: LFMiddle does NOT attempt to fix dt ≤ 0.5 fs limitation. Phase 3.1 includes dt-sweep to verify constraint coupling is independent of discretization choice.

---

## Phased Implementation (4 Phases, 56 hours total)

### Phase 1: Constraints Layer (16 hours)

**Objective**: Formalize SETTLE constraint model, define DOF masks, populate step registry.

**Atomic Steps**:

1.1. **SETTLE Constraint Formulation & DOF Masks** (4h)
   - Input: `settle.py` (current monolithic implementation), `CLAUDE.md` (background)
   - Output: `src/prolix/physics/constraints.py` with ConstraintDOFMask class and SETTLE kinematics
   - Verification: Unit tests for DOF orthogonality, constraint Jacobian correctness

1.2. **Step Interface & Parameter Schema** (5h)
   - Input: Monolithic physics code, settle.py
   - Output: `src/prolix/physics/step_system.py` with Step base class + 6 subclasses (O_Step, V_Step, A_Step, SETTLE_Velocity_Step, CSVR_Step, NHC_Step)
   - Verification: Type checking, module imports without error

1.3. **Populate Step-Sequence Registry** (4h)
   - Input: `step_system.py`, current integrator parameter defaults
   - Output: `src/prolix/physics/step_registry.py` with 4 pre-registered variants:
     - `BAOAB_LANGEVIN_NVT`: B → A → O → A → SETTLE_vel → Force → B → SETTLE_vel
     - `BAOAB_LANGEVIN_VV`: Velocity-Verlet variant (for Phase 3)
     - `BAOAB_CSVR_NPT`: B → A → SETTLE → Force → CSVR → Barostat → B
     - `SETTLE_WITH_NHC`: B → A → SETTLE → NHC_step → V → O
   - Verification: All variants instantiate without error; step counts and parameter consistency

**Phase 1 Gate**: Step system compiles; 4 variants register successfully; unit tests green.

---

### Phase 2: Builder Layer & BAOAB+Langevin Baseline (24 hours)

**Objective**: Implement `make_integrator()` factory. Validate BAOAB+Langevin achieves kUPS parity.

**Atomic Steps**:

2.1. **Implement make_integrator() Core** (8h)
   - Input: `step_system.py`, `constraints.py`, current settle_langevin
   - Output: `src/prolix/physics/integrator_builder.py` with:
     - `make_integrator(steps, constraint_dofs, energy_fn, shift_fn, **kwargs) → (init_fn, apply_fn)`
     - `init_fn(key, positions, box=None, **kwargs) → IntegratorState` (eager init of forces, chain_state, barostat_state)
     - `apply_fn(state, **step_kwargs) → state` (loop over steps, update state)
   - Verification: Unit tests for init_fn, apply_fn, state shape consistency, key increment behavior

2.2. **Worked Example: settle_langevin Parameter Mapping** (3h)
   - Input: `settle.py` (16-parameter signature), step_system, step_registry
   - Output: `doc/INTEGRATOR_PARAMETER_MAPPING.md` showing:
     - Old API: `settle.settle_langevin(dt, gamma, kT, mass, water_indices, project_ou_momentum_rigid, projection_site='post_o', ...)`
     - New API: `make_integrator(BAOAB_LANGEVIN, constraint_dofs, energy_fn, shift_fn, key=key, ...)`
     - Mapping table: All 16 params → Step.params fields or registry entry
     - **Critical**: `projection_site` stored in O_Step.params; StepSequence variants pre-compose for 'post_o' vs 'post_v' injection points
     - Equivalence proof: Old & new produce bitwise identical trajectories (verified via test)
   - Verification: Auditor reviews for completeness, correctness, clarity

2.3. **Implement & Validate BAOAB+Langevin** (10h)
   - Input: `integrator_builder.py`, `step_system.py`, `settle.py` reference
   - Output: Concrete implementations of O_Step.apply(), A_Step.apply(), V_Step.apply(), SETTLE_Velocity_Step.apply()
   - Verification: Unit tests for each step; integration test for BAOAB+Langevin equivalence vs current settle_langevin (bitwise float64 match over 100 steps)

2.4. **CRITICAL GATE: Phase 2 Completion Gate** (3h)
   - Action: Run kUPS cross-validation test (50ps trajectory, 64 waters, BAOAB+Langevin)
   - Success Criteria:
     - RMSD vs GROMACS reference < 0.1 Å
     - Kinetic energy drift < ±1%
     - No NaN observed
   - **Escalation if gate fails** (1 week max):
     - Step 1: Rerun with KE/force logging
     - Step 2: Manual step-by-step comparison vs settle_langevin
     - Step 3: If divergence at step X, debug that step (O/A/SETTLE/V)
     - Step 4: If no root cause within 1 week → escalate to oracle
     - **Do NOT proceed to Phase 3 until gate passes**

**Phase 2 Gate**: BAOAB+Langevin achieves kUPS parity. All unit tests pass.

---

### Phase 3: Alternative Discretizations (20 hours, parallel with Phase 4)

**Objective**: Implement LFMiddle (VV variant), CSVR+Barostat. Evaluate dt-sweep. Clarify discretization trade-offs.

**Atomic Steps**:

3.1. **Implement Velocity-Verlet (LFMiddle)** (8h)
   - Output: `step_system.py` extended with VV_Step, LFMiddle_Step
   - Register: `BAOAB_LANGEVIN_VV` variant in step_registry
   - **dt-sweep**: Run 50ps NVT for dt ∈ {0.25, 0.5, 1.0} fs; measure RMSD & energy drift
   - **Expected outcome**: LFMiddle likely also limited to dt ≤ 0.5 fs (constraint coupling dominates, not discretization)
   - Verification: Energy conservation 2nd-order in dt; stability boundary documented

3.2. **Implement CSVR + Barostat Steps** (9h)
   - Output: CSVR_Step, Barostat_Step in step_system.py
   - Register: `BAOAB_CSVR_NPT` variant
   - Short-trajectory validation: 10ps run at dt=0.5 fs; pressure stays [0.5, 2.0] atm; temperature in [290, 310] K
   - Verification: No runaway heating; constraint satisfaction; barostat scaling correct

3.3. **Discretization Guide & Documentation** (3h)
   - Output: `doc/INTEGRATOR_DISCRETIZATION_GUIDE.md` with:
     - Variant comparison table (accuracy, max dt, stability, use cases)
     - Recommendation: BAOAB+Langevin for production NVT; NPT short-trajectory only; LFMiddle for comparison
   - Updated CLAUDE.md with v2.0 integrator guidance

**Phase 3 Gate**: LFMiddle registered and dt-swept. CSVR+NPT short-trajectory validated. Discretization guide published.

---

### Phase 4: Batching & Production Readiness (16 hours, parallel with Phase 3)

**Objective**: Verify make_integrator works identically for vmap (batch) application. Deploy behind feature flag.

**Atomic Steps**:

4.1. **Implement Batching Support** (6h)
   - Output: `apply_fn` is vmap-compatible (pure functional, no side effects)
   - Helper: `batched_apply_fn = jax.vmap(apply_fn, in_axes=(0,), out_axes=0)`
   - Verification: apply_fn works on unbatched states (N, 3) and batched states (B, N, 3)

4.2. **Success Criterion #7: Concrete Batching Equivalence** (5h)
   - Output: `test_batching_equivalence.py` with precise criterion:
     ```
     For any system (N ∈ {3, 64, 512}) and batch size (B ∈ {4, 16}):
     vmap(apply_fn)(state_batch) ≈ apply_fn_looped(state) 
     within float64 ulp tolerance = 1e-12 
     over 1000 steps
     Metric: Max RMSE across all (N, B, step) combinations < 1e-12 * ||positions||
     ```
   - Verification: All (N, B, step) combinations pass; test is falsifiable

4.3. **Feature Flag & Deprecation Path** (3h)
   - Output: `Justfile` with feature flag recipes; `__init__.py` with USE_NEW_INTEGRATOR_BUILDER flag
   - Default: Flag disabled (old settle_langevin path active)
   - Deprecation: `settle_langevin` raises FutureWarning; users prompted to migrate
   - Migration guide: `doc/MIGRATION_INTEGRATOR_BUILDER.md` with before/after examples

4.4. **Full Integration & Regression Test** (2h)
   - Action: Run entire test suite with feature flag OFF and ON
   - Verification:
     - Old flag (off): All tests pass
     - New flag (on): All tests pass
     - Regression detected: false (max trajectory RMSD < 1e-12)

**Phase 4 Gate**: Batching equivalence verified. Feature flag deployed. Full regression suite passes.

---

## Step-Sequence Registry Table (Complete)

| Variant | Ensemble | Max dt (fs) | Steps | Accuracy | Production | Phase |
|---------|----------|-----------|-------|----------|-----------|-------|
| **BAOAB_LANGEVIN_NVT** | NVT | 0.5 | O→A→O→A→SETTLE_vel→F→B→SETTLE_vel | GROMACS parity | ✅ YES | Phase 2 |
| **BAOAB_LANGEVIN_VV** | NVT | 0.5–1.0? | O→VV→SETTLE_vel→O | Under evaluation | ❌ NO (Phase 3) | Phase 3 |
| **BAOAB_CSVR_NPT** | NPT | 0.5 | O→A→SETTLE→F→CSVR→Barostat→B | Short only | ⚠️ LIMITED | Phase 3 |
| **SETTLE_WITH_NHC** | NVT | 0.5 | O→A→SETTLE→NHC→V→O | Long-term memory | ❌ NO (Phase 5) | Future |

### Parameter Mapping: How projection_site Survives

**Issue**: `settle_langevin` has `projection_site='post_o'` parameter that controls *when* (not *what*) the OU projection happens. How does this map into a static step registry?

**Answer**: Via StepSequence variants. Two variants are pre-composed:

```
BAOAB_LANGEVIN_POST_O:
  O_Step(project_ou_momentum_rigid=True, projection_site='post_o')  ← Projects here
  A_Step(...)
  SETTLE_Velocity_Step(...)
  V_Step(...)
  O_Step(project_ou_momentum_rigid=True, projection_site='post_o')  ← Projects here

BAOAB_LANGEVIN_POST_V:
  O_Step(project_ou_momentum_rigid=False, projection_site='post_v')  ← No projection
  A_Step(...)
  SETTLE_Velocity_Step(...)
  V_Step(project_after_v=True)  ← Projection here instead
  O_Step(project_ou_momentum_rigid=False, projection_site='post_v')
```

At runtime, `O_Step.apply()` checks `params['project_ou_momentum_rigid']` and optionally calls `project_to_rigid_subspace()`. The injection point is baked into which variant you choose at design time.

---

## IntegratorState Initialization: Eager vs Lazy

**Clarification** (addressing Auditor Issue #7):

All initialization is **eager** in `init_fn`, not lazy.

```python
def init_fn(key, positions, box=None, **kwargs):
    # EAGER computation of all fields
    forces = energy_fn(positions, box)  # ← computed once, stored in state
    
    if has_nhc_step:
        # Compute Nosé-Hoover chain masses Q and initial positions xi_0
        Q_array = jnp.array([(i+1) * kT / omega_nh**2 for i in range(chain_length)])
        xi_0 = jax.random.normal(key, (chain_length, n_atoms)) * jnp.sqrt(kT / Q_array)
        chain_state = NHCChainState(xi=xi_0, v_xi=jnp.zeros_like(xi_0), Q=Q_array)
    else:
        chain_state = None
    
    if has_barostat_step:
        barostat_state = BarostatState(
            box_volume=jnp.linalg.det(box),
            momentum=0.0
        )
    else:
        barostat_state = None
    
    # Constraint state is STATELESS (not stored)
    
    return IntegratorState(
        positions=positions,
        momentum=jnp.zeros_like(positions),
        forces=forces,           # ← Ready for first A_Step
        box=box,
        key=key,
        chain_state=chain_state,  # ← If NHC, initialized here
        barostat_state=barostat_state,  # ← If barostat, initialized here
        constraint_state=None,    # ← Stateless (computed/discarded per step)
    )
```

---

## Critical Dependency: Phase 2 Completion Gate

**Gate Definition**:
- After Phase 2.3 (BAOAB+Langevin implementation), run Phase 2.4 (kUPS cross-validation)
- **Success**: RMSD < 0.1 Å, KE drift < ±1% over 50ps
- **Failure escalation** (max 1 week):
  1. Rerun with logging; identify step with divergence
  2. Compare that step vs settle_langevin baseline
  3. If root cause found → fix, retest
  4. If no root cause → escalate to oracle; **do NOT proceed to Phase 3**

**Why strict gate?**
- If Phase 2 has bugs, Phase 3 (LFMiddle) failures cannot be attributed correctly
- LFMiddle dt-sweep depends on knowing BAOAB+Langevin is correct baseline
- Better to fix fundamental issues before investing in Phase 3 work

---

## Phase 3 & Phase 4 Parallelization

**Dependency Analysis**:
- **Phase 3 (LFMiddle, CSVR)**: Depends on Phase 2 gate ✓. No coupling to Phase 4.
- **Phase 4 (Batching)**: Depends on Phase 2 gate ✓. No coupling to Phase 3.

**Decision**: **Run Phase 3 and Phase 4 in parallel** after Phase 2 gate passes.

**Time Savings**: 16h + 20h = 36h if parallel vs sequential. Total: Phase 1 (16h) + Phase 2 (24h) + Phase 3||4 (36h) = **56h total** (vs 76h sequential).

**Resource Note**: If solo engineer, execute sequentially (Phase 3 then 4). If two engineers, parallelize.

---

## Risk Assessment (6 Risks, All Mitigated)

| Risk | Severity | Likelihood | Mitigation | Contingency |
|------|----------|-----------|-----------|-------------|
| Phase 2 gate fails (make_integrator ≠ settle_langevin) | Critical | Low | Step-by-step debugging; detailed comparison | Rollback Phase 2.3; defer refactoring |
| Parameter mapping doesn't survive registry | High | Medium | Worked example (Step 2.2); auditor review | Extend Step class for dynamic kwargs |
| NHC chain init is wrong | Medium | Low | Verify Q_masses vs theory; equipartition test | Defer settle_with_nhc to Phase 5 |
| Batching vmap causes precision loss | High | Medium | Concrete ulp tolerance test; extensive sweep | Keep batching loop-based (no vmap) |
| LFMiddle dt-sweep shows dt=1.0fs unstable | Low | High (expected) | Phase 3.1 explicitly expected outcome | Document; note LFMiddle offers no advantage |
| Feature flag breaks in production | Medium | Low | Defaults to old code path; regression suite | Disable flag globally; investigate; re-enable |

---

## Success Criteria (Concrete & Falsifiable)

| Criterion | Target | Verification |
|-----------|--------|--------------|
| **Phase 1** | Step-system compiles; 4 variants instantiate | `python -c "from prolix.physics import step_registry; print(step_registry.BAOAB_LANGEVIN)"` succeeds |
| **Phase 2** | BAOAB+Langevin RMSD < 0.1 Å, KE ±1% (kUPS parity) | Cross-validation test passes; escalation gate met |
| **Phase 3** | LFMiddle dt-sweep complete; CSVR+NPT short-traj stable | Energy/pressure/temperature within expected ranges; discretization guide published |
| **Phase 4** | Batching: `vmap(apply_fn)` ≈ loop, ulp ≤ 1e-12 | test_batching_equivalence.py passes all (N, B, step) combos |
| **Overall** | make_integrator production-ready | Feature flag deployed; all tests pass (old & new); v2.0 release ready |

---

## Next Steps

1. **Plan-Auditor Final Review**: Audit revised ADR against 5 dimensions (decomposability, dependencies, verification, risks, completeness)
2. **Oracle Final Assessment**: Strategic review of go/no-go, critical path, unknown unknowns
3. **Proceed to Phase 1**: If both audit & oracle approve → execute Phase 1 (Constraints Layer)

---

**ADR Status**: Awaiting Plan-Auditor approval (revised v2.0 with all 7 feedback items addressed)

**Authors**: Planner Agent (revision addressing auditor feedback)

**Last Updated**: 2026-04-30
