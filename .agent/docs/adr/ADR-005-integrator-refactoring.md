# ADR-005: Modular Integrator Architecture & Constraint Pluggability

**Status**: In Design Review (awaiting plan-auditor approval)

**Date**: 2026-04-30

**Decision Drivers**:
- Current integrator implementations (settle_langevin, settle_csvr, settle_with_nhc) duplicate ~150 lines each
- Adding new integrator types (LFMiddle) currently requires copying 150+ lines of boilerplate
- SETTLE constraints are inlined; not pluggable (no LINCS, ShakeRattle variants)
- State management is fragmented (NVTLangevinState, NPTState, LangevinState)
- Batched integrators exist only for BAOAB+Langevin (no CSVR batched variant)

## Proposal: Three-Layer Integrator Architecture

### Layer 1: Constraint Plugins (ConstraintAlgorithm Protocol)
- Define protocol: `apply_positions(state, R_unconstrained, R_old, box) → R_constrained`
- Implementations: NullConstraint, SETTLEConstraint, LINCSConstraint (future)
- Decouple constraint algorithm from integrator logic
- Enable swapping SETTLE ↔ LINCS without touching integrator code

### Layer 2: Step Primitives Library
- Reusable elementary steps: step_b, step_a, step_o, step_csvr, step_constraint_pos, step_constraint_vel, step_remove_com
- Extract from existing settle.py helpers (_langevin_step_b, etc.)
- Each step is JIT-safe, pure, takes explicit parameters

### Layer 3: Integrator Builder & Step Order Registry
- Central mapping: `(integrator_type, thermostat_type) → step_sequence`
- `make_integrator(integrator_type, thermostat_type, ..., constraint=None) → (init_fn, apply_fn)`
- Registry examples:
  - `baoab/langevin`: B→A→O→A→Constraint_pos→Force→B→Constraint_vel→COM_remove
  - `lfmiddle/langevin`: B→A→O/2→A→Constraint_pos→Force→A→O/2→B→Constraint_vel
  - `vv/csvr`: B→A→Constraint_pos→Force→CSVR→B→Constraint_vel

### Unified State
- Single `IntegratorState` dataclass replacing NVTLangevinState, NPTState, LangevinState
- Required fields: position, momentum, force, mass, rng
- Optional fields: box (NPT), potential_energy (cache), chain_state (NHC), constraint_state

## Implementation Roadmap

| Phase | Duration | Deliverables | Key Validation |
|-------|----------|--------------|-----------------|
| Phase 1: Spike | 1–2 weeks | constraints.py, step_primitives.py, IntegratorState, unit tests | Unit tests pass; no regressions in existing code |
| Phase 2: Refactor | 2–3 weeks | integrators.py (make_integrator + INTEGRATOR_REGISTRY), settle_langevin migration | All existing tests pass; integration tests for BAOAB/CSVR variants |
| Phase 3: Validation | 2–3 weeks | LFMiddle registry entry, spike tests (dt=1.0 fs hypothesis), cross-validation vs kUPS | LFMiddle produces stable dynamics (or documented xfail); kUPS parity validated |
| Phase 4: Extension | 2–4 weeks | Batching support, CSVR+batched variant, extensible registry | Batched integrators work; registry ready for future types |
| **Total** | **9–12 weeks** | **Production-ready modular integrator framework** | **All success criteria met** |

## Success Criteria

1. **Modularity**: Adding LFMiddle requires ≤50 lines of code (vs 150+ currently)
2. **Backward Compatibility**: All existing tests pass; settle_langevin API unchanged
3. **Constraint Pluggability**: Can swap SETTLE ↔ null ↔ future constraints without touching integrator logic
4. **Cross-Validation**: Prolix BAOAB+Langevin matches kUPS within tolerance (RMSD < 0.1 Å, KE within ±1%)
5. **LFMiddle Hypothesis Test**: Spike confirms whether LFMiddle enables dt ≥ 1.0 fs (or documents finding if not)
6. **Performance**: No regression in critical paths (step_b, step_a, SETTLE) relative to current implementation
7. **Batching**: make_integrator works for unbatched and batched (B, N, 3) systems identically

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| R1: Step order complexity → unmaintainable registry | Medium | Start small (BAOAB/CSVR/LFMiddle only); validate each entry with cross-validation |
| R2: Performance regression from abstraction layers | Medium | Benchmark critical paths; JAX JIT should fuse step sequences into single kernel |
| R3: Breaking API changes (settle_langevin signature) | High | Keep settle_langevin as thin wrapper around make_integrator; API stable |
| R4: LFMiddle hypothesis wrong (doesn't fix dt constraint) | Low | Spike test is xfail-ready; hypothesis disproven is acceptable outcome (documented) |
| R5: Cross-validation fails (Prolix ≠ kUPS) | Medium | Start with simple systems (single water); use git bisect to find regression |
| R6: Constraint plugin overhead is bottleneck | Low | Benchmark SETTLEConstraint vs inlined code; optimize or special-case if needed |

## Key Files Affected

**New Files:**
- `src/prolix/physics/constraints.py` — ConstraintAlgorithm protocol + implementations
- `src/prolix/physics/step_primitives.py` — Reusable step functions
- `src/prolix/physics/integrators.py` — make_integrator factory + INTEGRATOR_REGISTRY
- `tests/physics/test_constraint_plugins.py` — Constraint unit tests
- `tests/physics/test_step_primitives.py` — Step primitive unit tests
- `tests/physics/test_integrator_builder.py` — Integration tests
- `tests/physics/test_cross_validation_kups.py` — Cross-validation suite

**Modified Files:**
- `src/prolix/physics/settle.py` — Migrate settle_langevin, settle_csvr_npt to use make_integrator internally; keep public API
- `src/prolix/physics/simulate.py` — Define IntegratorState; deprecate old state classes
- `src/prolix/batched_simulate.py` — Integrate with make_integrator; support batched CSVR variant
- `CLAUDE.md` — Document LFMiddle findings

## Design Rationale

**Why Three Layers?**
- **Layer 1 (Constraints)**: Orthogonal to integrator type; enables SETTLE ↔ LINCS swapping
- **Layer 2 (Steps)**: Atomic, reusable operations; composition without duplication
- **Layer 3 (Builder)**: Central registry as single source of truth for integrator variants

**Why step sequence registry instead of class hierarchy?**
- Symplectic integrators are characterized by their step order, not behavior
- Registry table is self-documenting; easy to verify against published sequences
- No inheritance or OOP complexity; registry is data, not logic
- Adding new integrator = adding new entry (low code duplication)

**Why LFMiddle in Phase 3?**
- Hypothesis: splitting O-step around Force may reduce SETTLE+thermostat coupling
- Only testable after make_integrator is working (Phase 2 complete)
- If hypothesis fails, LFMiddle still valuable as alternate discretization
- Early spike (2–3 weeks) saves 3–6 months if hypothesis is wrong

**Why unified IntegratorState?**
- Current fragmentation (NVTLangevinState, NPTState, LangevinState) is maintenance burden
- Single state interface simplifies make_integrator logic (no polymorphism on state type)
- Optional fields (box, chain_state, constraint_state) handle all ensemble/thermostat variants without subclassing

## Backward Compatibility

- **settle_langevin, settle_csvr_npt, settle_with_nhc**: Continue to work as before (wrappers around make_integrator)
- **NVTLangevinState, NPTState**: Deprecated in v1.1; marked with DeprecationWarning; removed in v2.0
- **New API**: make_integrator is opt-in; no breaking changes for existing users

## Next Steps

1. **Plan-Auditor Review**: Critique this design for completeness, dependency ordering, risk coverage, success metrics
2. **Refinement Cycle**: Address plan-auditor feedback; iterate until PASS verdict
3. **Oracle Final Review**: Strategic assessment of go/no-go, dependencies, unknown unknowns
4. **Implementation**: Execute Phase 1 spike (1–2 weeks) → validation → proceed to Phase 2

---

**Design Authors**: Planner Agent (2026-04-30)
**Status**: Draft (awaiting audit)
