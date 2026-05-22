# Oracle Critique: Cycle 2

**Verdict:** REVISE (Confidence: High)

**Strategic Assessment:**
The mapping of `mask_elec` correctly resolves the O(N^2) scaling limit on the non-batched GB neighbor path. The SETTLE design brings the correct functions to bear. However, implementing stability limits on the standard unbatched scalar path (`simulate.py` / `physics_simulate.py`) fundamentally goes against the pure numerical design of the primary runner. Capping should not be implicitly smuggled into a standard integration step.

## Concerns

### 1. Velocity/Force Limiters in Standard Runner (Critical)
**Issue:** The plan states `MAX_DX_PROD = 0.5`, `VLIMIT = 20.0`, and force clamping will be added to `physics/simulate.py`. This is incorrect. The unbatched API should remain mathematically strict and un-clamped, as capping breaks NVT detailed balance. Limiting was a hack for TPUs in the batched path due to severe collision limits. In standard runs, SETTLE is unconditionally stable if the timestep and forces are sane.
**Recommendation:** Remove the mandatory limiters from the proposed `settle_rattle_langevin`. Implement the pure BAOAB integrator with pure SETTLE/RATTLE mapping. If force capping is needed, it should remain an external force clip wrapper, not embedded in the integrator.

### 2. Water exclusions verify (Warning)
**Issue:** The `mask_elec` solves GB, but double check water geometries. SETTLE solves water intra-molecular distances, and typically nonbonded exclusions need to ignore water intramolecular forces. Since `exclusion_spec` is built from system bonds, if H-O-H angles aren't explicit bonds, 1-3 pairs might compute LJ/Coulomb inside water, fighting SETTLE.
**Recommendation:** Since this plan is just for SETTLE/GB wiring, this is more of a topology-parser issue, but the integration plan itself is safe. Provide note to just trust the `exclusion_spec`.
