# Sprint B: Temperature Drift Hypotheses

## Problem Statement

NVT temperature control fails at long timescales (>100 ps) when using SETTLE + Langevin with n_waters=64:
- Baseline (n_waters=8, dt=0.5fs, 200 steps → 100 ps): PASSES (T ≈ 300 K)
- Failing case (n_waters=64, dt=0.5fs, 200k steps → 100 ps): FAILS (T ≈ 334 K, +34 K drift)

**Key observation**: Temperature drift is system-size dependent AND timescale dependent.

## Hypothesis H1: Integration/Projection Error

**Mechanism**: O(dt^2) bias accumulation in Ornstein-Uhlenbeck (OU) projection or SETTLE/OU coupling discretization.

**Why it matters**: SETTLE removes kinetic energy from constrained DOF. The OU thermostat tries to restore it. At each step:
1. SETTLE constraint removes KE_constraint
2. OU noise adds KE_thermal
3. If the per-step O(dt^2) bias is positive, it accumulates over many steps

**Evidence for H1**:
- dt=0.5 fs constraint exists → suggests O(dt^2) accumulation is real
- Smaller timesteps should reduce O(dt^2) bias
- Ablation test shows n_waters=8 works (fewer DOF → lower absolute bias accumulation)

**Test to confirm**: Reduce dt from 0.5 fs to 0.25 fs (half)
- If T_obs → 300 K at dt=0.25 fs: H1 CONFIRMED
- If T_obs stays ≈ 334 K at dt=0.25 fs: H1 REJECTED

**Fix pathway if confirmed**: 
- Implement higher-order discretization of OU projection
- Or use constraint-aware OU that respects rigid-body DOF manifold
- Or implement variable-order integration based on constraint coupling strength

---

## Hypothesis H2: DOF/KE Mismatch

**Mechanism**: Temperature formula uses wrong degree-of-freedom count or KE computation is misaligned with the DOF definition.

**Why it matters**: For rigid water with SETTLE constraints:
- True DOF = 6*N_w - 3 (translational + rotational for N_w rigid bodies, minus COM constraint)
- But KE measurement might implicitly use different DOF counting:
  - Using all 3N atomic coordinates: T = 2*KE / (3N * k_B) instead of T = 2*KE / ((6N-3) * k_B)
  - Factor: 3N / (6N-3) ≈ 1.11 for large N

**Evidence against H2**:
- DOF mismatch would produce T_obs / T_target = 1.11x constantly across all dt values
- But we observe 334/300 = 1.113x, suggesting pure DOF error OR accumulation that scales as 1.113
- Ablation test (test_ke_measurement_ablation) was designed to catch this

**Test to confirm**: Run KE measurement ablation
- Compute T using both methods: rigid-body KE vs simple atomic KE
- If both methods agree on T ≈ 334 K: DOF mismatch is NOT the issue
- If only one method shows 334 K: That method has the bug

**Current status**: PENDING (Ablation test available but not recently executed)

---

## Hypothesis H3: Force-Side Mechanism (PME Grid Coupling)

**Mechanism**: PME Coulomb forces have systematic bias when paired with rigid-body constraints and/or water system size.

**Why it matters**: PME grid discretization introduces forces that:
- Depend on system size (N water molecules → more grid coupling)
- Couple with SETTLE constraint forces
- Produce systematic energy/momentum bias that manifests as temperature drift

**Specific sub-hypothesis**: PME grid (32x32x32) couples with box size (≈100 Å for 64 waters at 10 Å spacing).
- Grid resolution: 100 Å / 32 ≈ 3.1 Å per cell
- At this resolution, water-water PME forces may have systematic bias
- Reducing dt doesn't help because the bias is in the force field, not the integration

**Evidence for H3**:
- Drift appears only at larger N and/or longer timescales → force coupling effects grow with scale
- n_waters=8 (smaller box, ~45 Å) works fine
- n_waters=64 (larger box, ~100 Å) fails at 334 K
- Suggests grid cell-to-box-size ratio matters

**Test to confirm (Step 2)**: Vary PME grid density
- Run at pme_grid_points=16 (coarser, bigger cells)
- Run at pme_grid_points=64 (finer, smaller cells)
- If T_obs improves with finer grid: H3 CONFIRMED (grid-induced force bias)
- If T_obs unchanged: H3 REJECTED (not a grid effect)

**Fix pathway if confirmed**:
- Increase default PME grid density
- OR use adaptive grid tuning based on system size
- OR implement constraint-aware PME that avoids rigid-body DOF coupling

---

## Hypothesis H4: JAX Autodiff/Compilation Artifact

**Mechanism**: JIT compilation or JAX autodiff introduces systematic numerical error in force/gradient computation or OU sampling.

**Why it matters**: JAX's functional programming model can introduce subtle biases:
- Force autodiff may lose precision in constraint force computation
- OU random sampling may have subtle correlations in JIT mode
- State updates may have off-by-one or truncation artifacts

**Evidence for H4**:
- Only appears in JIT mode (not in Python loop)? (untested)
- Correlates with compile cache behavior? (untested)
- Affects only certain problem sizes/shapes? (observed: 64 water system)

**Test to confirm (Step 3)**: Sign/magnitude analysis of force/momentum changes
- Log forces F, constraint impulses J, and momentum changes Δp each step
- Check for sign bias: are forces consistently in one direction?
- Check for magnitude bias: are constraint impulses systematically too large/small?
- Compare to analytical expectations for 64-water system

**Fix pathway if confirmed**:
- Disable JIT for constraint force computation
- OR implement manual precision-preserving autodiff for constraints
- OR tune JAX precision settings for specific hardware
- OR file issue with JAX team (if it's a real JAX bug)

---

## Test Roadmap (Sprint B)

**Step 1** (Current): Discriminator test at dt=0.25 fs
- Distinguishes H1 (integration error) from H3 (force-side)
- If dt=0.25fs → T ≈ 300K: H1 likely, proceed to fix
- If dt=0.25fs → T ≈ 334K: H3/H4 likely, proceed to Step 2

**Step 2** (Conditional): PME grid scaling
- Runs at pme_grid_points = [16, 32, 64]
- Distinguishes H3 (grid coupling) from H4 (other force/autodiff)
- If grid helps: implement grid scaling fix
- If grid doesn't help: escalate to Step 3

**Step 3** (Conditional): Sign/magnitude analysis
- Deep dive into force vectors and constraint impulses
- Identifies whether bias is in forces or in OU projection
- Guides targeted fix implementation

---

## Confidence Assessment

**H1 (Integration error)**: 40% prior confidence
- Rationale: O(dt^2) accumulation is a known pathology for coupled systems
- dt constraint exists → suggests real effect
- But 11% temperature rise is large for O(dt^2) over 100 ps

**H3 (Force-side)**: 50% prior confidence
- Rationale: System-size dependence strongly suggests force/grid effects
- PME + rigid-body coupling is a known difficult problem
- Grid resolution @ 3.1 Å is marginal for TIP3P water

**H2 (DOF mismatch)**: 5% prior confidence
- Rationale: Multiple tests have already checked this
- If true, would show constant T ≈ 334K across all dt values

**H4 (JAX bug)**: 5% prior confidence
- Rationale: Unlikely but possible in JIT autodiff edge cases
- Would only confirm via Step 3 sign/magnitude analysis

---

## Key Unknowns

1. **Does dt=0.25 fs improve temperature?** (This test answers it)
2. **What's the correct temperature formula?** (H2 vs H1/H3)
3. **Is the issue in forces or in integration?** (H3 vs H1)
4. **Is PME grid the specific culprit?** (H3 sub-hypothesis)
5. **How does this scale with N?** (Need n_waters ∈ [4, 8, 16, 32, 64, 128])

---

## References

- CLAUDE.md: dt ≤ 0.5 fs constraint documentation
- test_settle_temperature_control.py::test_temperature_langevin_dt0_5fs_green(): Failing test docstring
- Miyamoto & Kollman 1992: SETTLE algorithm (rigid constraints)
- Bussi & Parinello 2007: CSVR thermostat (alternative to Langevin)
