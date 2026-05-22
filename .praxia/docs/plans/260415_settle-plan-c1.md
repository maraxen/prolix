# Oracle Critique: Cycle 1

**Verdict:** REVISE (Confidence: High)

**Strategic Assessment:**
The plan correctly identifies the need for SETTLE integration and the GB mask scaling issue. However, the exact mapping of `mask_hard` and `mask_elec` to GB components needs precise alignment with existing `system.py` logic to preserve validation parity.

## Concerns

### 1. GB Pair Mask Mapping (Critical)
**Issue:** The plan states "Map pair_mask_born and pair_mask_energy using mask_hard (and/or mask_elec based on existing logic)". This is vague. `system.py` currently maps `gb_mask = exclusion_mask` and `gb_energy_mask = None` for standard `Protein` structures, meaning both Radii and Energy inherit the full exclusion mask (0 for 1-2/1-3, scaled for 1-4).
**Recommendation:** Specify exactly that `pair_mask_born = mask_elec` and `pair_mask_energy = mask_elec` should be used when replacing `gb_mask`. `mask_elec` (from `compute_exclusion_mask_neighbor_list`) natively embodies both the hard exclusions (0.0) and 1-4 electrical scaling, matching the original `exclusion_mask`.

### 2. SETTLE API usage in simulate.py (Warning)
**Issue:** The plan suggests passing `water_indices` and `box` to `settle_rattle_langevin`. But `make_langevin_step_explicit` limits velocities and forces for numerical stability in the batched path. Will the non-batched path also need those caps?
**Recommendation:** Explicitly state that `settle_rattle_langevin` should mirror the safe bounding/capping (e.g., `VLIMIT`) from `batched_simulate.py`'s `make_langevin_step_explicit` to guarantee equal stability on the unbatched explicitly solvated path.
