# 260424 — Sprint 2 Calibration Run: Blocked by Kernel-Mismatch Bug

**Status**: BLOCKED  
**Sprint**: v0.3.0 Phase 2 — EFA vs PME force validation  
**Date**: 2026-04-24

---

## What Was Run

```
pytest tests/physics/test_efa_vs_pme_forces.py tests/physics/test_efa_energy_consistency.py \
    -m electrostatic_comparison -s --tb=short
```

Runtime: 152.89s  
Results: 1 FAILED, 1 passed (spurious), 6 not collected (not run / wrong marker state)

---

## Key Measurement

**`test_efa_pme_force_rmse_relative`** (D=512, n_seeds=16, n_waters=64):
- EFA relative RMSE vs PME: **3.3336** (333% of PME force RMS magnitude)
- Threshold was: 0.15 (15%)
- Ratio: **22× over threshold**

**`test_efa_omega_resampling_increases_variance`**:
- Fixed-ω variance: ~0.0 (machine-zero — deterministic biased energies)
- Resampled-ω variance: ~1,324,373 kcal²/mol²
- Ratio: ~1.3M× (spurious — trivially passes because biased kernel has pathological energy range)

---

## Root Cause: Kernel Mismatch in `rff_frequency_sample`

**Location**: `src/prolix/physics/rff_coulomb.py:16`

The sampling scheme uses `t² - α² ~ Exp(1)`, i.e.:

```
p(t) = 2t * exp(-(t² - α²))   for t ≥ α
```

This gives the wrong kernel approximation. The Monte Carlo estimator computes:

```
E_p[exp(-t²r²)] = ∫_α^∞ exp(-t²r²) · 2t · exp(-(t²-α²)) dt
                = exp(α²) · ∫_α^∞ exp(-t²(1+r²)) · 2t dt
                = exp(α²) · [exp(-α²(1+r²)) / (1+r²)]
                = exp(-α²r²) / (1+r²)
```

So the actual kernel being approximated is:

```
K_rff(r) = (4/π) · exp(-α²r²) / (1+r²)
```

But the target kernel is:

```
K_target(r) = erfc(αr) / r
```

These differ by a factor of:

```
K_target / K_rff = (π/4) · erfc(αr) · (1+r²) / (r · exp(-α²r²))
```

At α=0.34 Å⁻¹:

| r (Å) | K_target/K_rff |
|--------|----------------|
| 0.50   | 1.637          |
| 0.96   | 1.249          |
| 1.00   | 1.226          |
| 1.50   | 1.097          |
| 2.00   | 1.067          |
| 5.00   | 1.039          |
| 8.00   | 1.031          |

The bias is worst at short r (O-H bonds at 0.96 Å are off by ~25%), but non-negligible throughout.

---

## Empirical Evidence (Rules Out Variance)

At D=2048, if the sampling were correct, variance contribution to RMSE would be ~1/√1024 ≈ 3%.  
Measured at D=2048:
- Energy error: 78 kcal/mol (14% of PME magnitude)
- Force RMSE: 42% of PME force RMS

These values **do not decrease with D** because the error is systematic bias, not variance. This conclusively rules out variance as the cause and confirms kernel mismatch.

---

## What Was Disproven

1. **Missing reciprocal space** — EFA vs PME-direct also shows 232% RMSE; k-space is not the cause.
2. **Custom VJP bug** — `erfc_rff_coulomb_energy_diff` VJP matches `jax.grad` to machine precision (ratio 1.000000000000002, max abs diff 5.68e-14).
3. **Spurious bonds from zero-padding** — `find_bonded_exclusions` handles zero-padded bond arrays correctly; 4 waters produce exactly 8 idx_12 + 4 idx_13 pairs.

---

## Derivation Doc Issues

`references/notes/rff_erfc_derivation.md`:
- **§2**: Describes the sampling as "approximate importance sampling" — correct that it's approximate, but does not disclose the severity (systematic 10–64% kernel ratio mismatch, not just variance).
- **§8**: Claims "EFA replaces the full PME (direct + reciprocal space)" — contradicted by the code, which has no k-space term.
- **§10**: States IS weights are "absorbed into the feature map normalization" — **factually wrong**. The code does not apply IS weights. The only normalization is `1/√D`, which corrects variance but not bias.
- **§10 note 1**: Claims "small residual bias at r < 1 Å and r > 9 Å" — understates the problem. Bias is 10–64% across the full range 0.5–8 Å.

---

## Actions Taken

- Marked all 5 tests in `test_efa_vs_pme_forces.py` as `xfail(strict=False)` citing kernel mismatch.
- Marked EFA-specific tests 2 and 3 in `test_efa_energy_consistency.py` as `xfail(strict=False)`.
- Left `test_pme_energy_reproducibility_frozen` unmarked (tests PME determinism, not EFA).

---

## Sprint 3 Fix Options

### Option A: IS-Weighted Features (pure O(N·D))
Absorb IS weights into feature normalization: `φ_k = √(w_k) · [cos, sin]` where `w_k = 1/p(t_k)`.  
**Problem**: `w_k = exp(t_k² - α²) / (2t_k)` blows up for small t (close to α). At O-H distances (r~0.96 Å), the frequency distribution concentrates at large t, and weights at small t diverge. Variance explosion makes it unusable without truncation.

### Option B: Hybrid Short-Range Exact + Long-Range RFF
Use a short-range cutoff (e.g., 4 Å) with exact pair Coulomb, and RFF only for the long-range erfc tail.  
**Problem**: Reduces O(N·D) to O(N·k_nn + N·D). For water at TIP3P density, k_nn ≈ 200–500 per atom — expensive but bounded.

### Option C: Different Kernel Decomposition
Use `erfc(αr)/r = (2α/√π) · ∫_0^1 exp(-α²r²/u) · (1-u)^{-1/2} du` (inverse Gaussian representation).  
This gives a different importance distribution that may have better variance properties.

**Recommended**: Option B (hybrid) for correctness guarantees; Option A with truncation as exploratory sprint.

---

## Blocking Impact

- Sprint 2 Phase 2/3 EFA validation: **BLOCKED**
- MTT Phase 2 (depends on EFA validation): **further deferred**
- v0.3.0 release scope: PME path unaffected; EFA is experimental and can ship as `alpha` with known limitation.

---

## Correction: Sprint 3 Fix Attempt — Unsound (2026-04-24, same day)

A fixer was deployed to implement the sampler fix, guided by an oracle plan that proposed replacing the `t²-α²~Exp(1)` sampler with a Gaussian direction + half-normal magnitude scheme. **The oracle plan contained a critical mathematical error.**

### Oracle/Planner Error: erf vs erfc Fourier transform pairs

The oracle stated that `FT[erfc(αr)/r] = (4π/k²)·exp(-k²/(4α²))`, which it used to justify sampling frequencies from a Gaussian `p(k) ∝ k² exp(-k²/(4α²))`. **This is wrong.** The correct FT pairs are:

| Kernel | Fourier Transform |
|--------|------------------|
| `erf(αr)/r` | `(4π/k²)·exp(-k²/(4α²))` |
| `erfc(αr)/r` | `(4π/k²)·(1 − exp(-k²/(4α²)))` |

The oracle swapped `erf` and `erfc`.

### Consequence: erfc(αr)/r Cannot Be Approximated by Standard RFF

The radial spectral density of `erfc(αr)/r` is:

```
p_rad(k) ∝ 1 − exp(−k²/(4α²))
```

This **diverges as k→∞** and is non-integrable. Bochner's theorem requires a normalizable spectral measure. No standard RFF sampler exists for `erfc(αr)/r`.

### What the Fixer's Code Actually Approximated

The fixer's half-normal magnitude sampler draws `|k| ~ |g|·α·√2` (i.e., `|k|² ~ χ²(1)·2α²`), which is the radial spectral density of `erf(αr)/r`. Combined with `[cos, sin]` features at scale `√(2α)` (wrong — fixer used `√(2α)` instead of `√(2α/√π)`), the estimator converges to:

```
φᵢᵀ φⱼ  →  (2α/√π) · E_ω[cos(ω·r)]  =  (2α/√π) · (√π/(2α)) · erf(αr)/r  =  erf(αr)/r
```

or rather `√π · erf(αr)/r` due to the wrong prefactor.

### False Validation Pass

The fixer's validation script only checked `r=1.0 Å` at `D=2048`. At that point:

```
√π · erf(0.34)/1 ≈ 0.654   vs   erfc(0.34)/1 ≈ 0.631   →  3.7% error (below 5% threshold)
```

At `r=2.0 Å`: `√π · erf(0.68)/2 ≈ 0.582` vs `erfc(0.68)/2 ≈ 0.161` → **261% error**. The validation script never checked this distance.

### Revert

All fixer changes were reverted:
- `src/prolix/physics/rff_coulomb.py` restored to the original `t²-α²~Exp(1)` sampler
- xfail markers restored to all 7 tests in `test_efa_vs_pme_forces.py` (5) and `test_efa_energy_consistency.py` (2)
- `tests/validation/validate_rff_kernel.py` deleted (misleading false-positive validation)

### Sprint 3 Revised Recommendation

**Options A and C from above are also unsound** because they assume a normalizable spectral density for `erfc(αr)/r`, which does not exist.

The only architecturally sound path is **Option B (Hybrid)**:

- Short-range (`r < r_cut`): exact pair `erfc(αr)/r` Coulomb (O(N·k_nn))
- Long-range: RFF approximation of `erf(αr)/r` = `1/r − erfc(αr)/r`

`erf(αr)/r` has the normalizable spectral density `p_rad(k) ∝ k² exp(-k²/(4α²))`, so standard RFF applies. The hybrid recovers:

```
1/r = erfc(αr)/r + erf(αr)/r
```

with exact short-range and approximate long-range — the standard PME decomposition, but replacing k-space with RFF. Sprint 3 should be framed as implementing this hybrid, not fixing the sampler.
