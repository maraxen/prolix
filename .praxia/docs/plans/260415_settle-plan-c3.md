# Oracle Critique: Cycle 3

**Verdict:** REVISE (Confidence: High)

**Strategic Assessment:**
The plan is architecturally sound on SETTLE integration and correctly resolved the VLIMIT purity concern from Cycle 2. However, it contains two factual errors that would cause implementation bugs: (1) the `mask_elec` → GB substitute claim is numerically incorrect because `exclusion_mask` is binary but `mask_elec` carries fractional 1-4 scales, and (2) the plan does not specify how `run_simulation` will obtain `water_indices` since `Protein` objects lack this field — only `MergedTopology` carries it.

## Concerns

### 1. GB Mask Mapping: `mask_elec` vs `exclusion_mask` (Critical)
**Issue:** The plan claims `mask_elec` "perfectly recovers the standard O(N^2) `exclusion_mask` parity". This is **wrong**. `exclusion_mask` (system.py:329) is binary: `(mat_vdw > 0.0).astype(float32)`, yielding 0.0 for 1-2/1-3 and **1.0** for 1-4 and normal pairs. But `mask_elec` returns **fractional** scales: 0.0 for 1-2/1-3, `scale_14_elec` (~0.833) for 1-4, and 1.0 for normal. Using `mask_elec` as `pair_mask_born` would incorrectly attenuate GB Born radii from 1-4 pairs by ~17%.

**Recommendation:** Use `mask_hard` (not `mask_elec`) as the sparse replacement. `mask_hard` is binary (0.0 for fully-excluded 1-2/1-3, 1.0 for everything else including 1-4), which exactly matches `exclusion_mask` semantics. The O(N^2) optimization only matters in the fallback `else` branch (system.py:451-453).

### 2. `water_indices` Access Path (Critical)
**Issue:** The plan says "extract `water_indices` if available" but `Protein` has no `water_indices` field. Only `MergedTopology` carries it. `run_simulation` normalizes all inputs to `Protein` at line 459. The plan doesn't specify how `water_indices` flows to the integrator dispatch.

**Recommendation:** Use `getattr(protein_system, 'water_indices', None)` duck-typing, since `MergedTopology` already has all `Protein` fields plus `water_indices`.

### 3. Integrator Duplication (Warning)
**Issue:** `settle.py` already has a complete `settle_langevin` BAOAB integrator. Creating `settle_rattle_langevin` in `physics/simulate.py` makes a third BAOAB copy.

**Recommendation:** Extend existing `settle_langevin` with optional `constraints` parameter for solute RATTLE.

### 4. Verification Plan Gaps (Suggestion)
**Issue:** No test covers the combined SETTLE+RATTLE path or production `run_simulation(..., rigid_water=True)`.

**Recommendation:** Add `test_settle.py` to the plan and create `test_settle_rattle_combined.py`.
