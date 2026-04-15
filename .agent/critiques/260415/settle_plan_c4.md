# Oracle Critique: Cycle 4

**Verdict:** ✅ APPROVE (Confidence: High)

**Strategic Assessment:**
The Cycle 4 plan correctly addresses all critical findings from Cycle 3. The `mask_hard` → `exclusion_mask` parity is now provably correct (both are binary: 0.0 for 1-2/1-3, 1.0 for 1-4 and normal — verified against code at `neighbor_list.py:147` and `system.py:329`). The duck-typing via `getattr` is the minimal-impact path for `water_indices` access. Extending `settle_langevin` avoids code duplication. The verification plan now includes SETTLE-specific tests.

## Concerns (Non-Blocking)

### 1. `MergedTopology` Missing `constrained_bonds` (Suggestion)
**Issue:** `MergedTopology` does not carry `constrained_bonds`/`constrained_bond_lengths` fields (verified by grep). When `run_simulation` receives a `MergedTopology`, `protein_system.constrained_bonds` at line 814 will raise `AttributeError` before reaching the new SETTLE dispatch.

**Recommendation:** Add `getattr` guarding: `constrained_bonds = getattr(protein_system, 'constrained_bonds', None)`. This is a 1-line fix consistent with the duck-typing pattern proposed for `water_indices`.

### 2. Constraint Ordering (Suggestion)
**Issue:** The batched path (`batched_simulate.py:2036-2048`) applies SHAKE before SETTLE for positions. The plan says to apply SETTLE then optional RATTLE. While both orderings converge for disjoint constraint sets, matching the batched path's ordering maximizes consistency.

**Recommendation:** Apply `project_positions` BEFORE `settle_positions` in the extended `settle_langevin`, matching the batched path's SHAKE → SETTLE ordering.

## Verdict Rationale
All critical concerns from previous cycles are resolved:
- ✅ C1: GB mask mapping specified precisely
- ✅ C2: VLIMIT removed from standard runner
- ✅ C3-Fix1: `mask_hard` (binary) replaces `mask_elec` (fractional)
- ✅ C3-Fix2: `getattr` duck-typing for `water_indices`
- ✅ C3-Fix3: Integrator consolidated into existing `settle_langevin`
- ✅ C3-Fix4: Verification plan expanded

The two remaining suggestions are non-blocking defensive improvements. The plan is **approved for execution**.
