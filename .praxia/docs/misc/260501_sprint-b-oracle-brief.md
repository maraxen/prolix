# Sprint B Step 1: Oracle Brief

**Prepared for**: Oracle review and decision-making  
**Date**: 2026-04-28  
**Status**: Test in execution, result pending

---

## Executive Summary

Executing a single diagnostic test to determine whether the NVT temperature drift (334 K vs 300 K target) in 64-water systems is caused by:

1. **H1: Integration Error** — Fix pathway available, 1-2 sprints
2. **H3: Force-Side Bias** — Fix pathway available, 1-2 sprints  
3. **Mixed (H1+H3)** — Dual fix required, 2-3 sprints
4. **Unknown (H4)** — Requires deep investigation, 2-3 sprints

Test design: Run 64-water NVT at **dt=0.25 fs** (half of failing baseline) for 100 ps. If temperature improves, it's H1 (integration). If temperature doesn't improve, it's H3 (force-side).

---

## What Oracle Needs to Know

### The Problem

- **Failing case**: n_waters=64, dt=0.5 fs → T=334 K (34 K above target)
- **Working case**: n_waters=8, dt=0.5 fs → T=300 K (OK)
- **Pattern**: Failure is system-size AND timescale dependent
- **Impact**: Blocks v1.0 release decision unless root cause is understood

### The Test

| Parameter | Value | Reason |
|-----------|-------|--------|
| dt | 0.25 fs | Half the failing value; distinguishes H1 (integration) from H3 (forces) |
| n_waters | 64 | Matches failing scenario (n_waters=8 is passing) |
| duration | 100 ps | Same wall-clock time as baseline test |
| steps | 400,000 | Maintains temporal resolution |
| seed | 7 | Same seed as baseline for reproducibility |

### Decision Point

Single measurement: **T_observed at dt=0.25 fs**

**If T_obs drops significantly (290-310 K)**: H1 is the culprit
- Root cause: O(dt^2) bias in OU/SETTLE discretization coupling
- Fix complexity: Moderate (improve integrator order)
- Timeline: 1-2 sprints, can defer to v1.1

**If T_obs stays high (330-340 K)**: H3 is the culprit  
- Root cause: PME force field has systematic bias with 64-water system
- Fix complexity: Moderate (scale PME grid or decouple constraints)
- Timeline: 1-2 sprints, can defer to v1.1

**If results are mixed (320-330 K)**: Both mechanisms
- Both H1 and H3 contribute; need dual fix
- Timeline: 2-3 sprints
- More complex, still deferrable to v1.1

**If results are unexpected (>340 K or noisy)**: Unknown mechanism
- Escalate to detailed analysis (Step 3)
- Timeline: 2-3 sprints
- May require v1.0 release deferral depending on priority

---

## What Oracle Should Decide

When result arrives, oracle should:

1. **Verify result is clean** (not NaN, not noisy, falls into one scenario)
2. **Assign confidence level** to diagnosis
3. **Recommend release decision for v1.0**:
   - **Option A**: Release v1.0 with documented constraint (dt ≤ 0.5 fs) and plan fix in v1.1
   - **Option B**: Defer release pending fix (if root cause is complex)
4. **Assign subagent for next step** if H3 is confirmed (need Step 2 PME grid test)

---

## Key Unknowns Before Test

1. **Is it integration or forces?** (This test answers it)
2. **Can we fix it in 1-2 sprints?** (Likely yes either way)
3. **Should v1.0 release include this fix or defer?** (Oracle decision)

---

## Supporting Documents

For detailed understanding, oracle should review:

1. **`sprint_b_hypotheses.md`** — Complete hypothesis framework
   - H1, H2, H3, H4 definitions and evidence
   - How each mechanism could produce 334 K result

2. **`sprint_b_step1_rationale.md`** — Why this test design
   - Why dt=0.25 fs specifically
   - Why 100 ps duration
   - Why 64 waters
   - What each scenario means

3. **`sprint_b_step1_execution.md`** — Test execution details
   - Exact parameters and code paths
   - Baseline comparison data
   - Expected runtime

4. **Test code**: `tests/physics/test_settle_temperature_control.py`
   - Line 21: `_mean_rigid_t_after_burn()` — measurement function
   - Line 319: `test_temperature_langevin_dt0_5fs_green()` — failing baseline test

---

## Success Criteria for Test

Result is actionable if:

1. ✓ Test completes without numerical error (no NaN/Inf)
2. ✓ Temperature has <5% statistical variance (burn-in adequate)
3. ✓ Result falls into one of four clear scenarios (A/B/C/D)
4. ✓ Result differs from baseline by >5 K or differs by <5 K (clear signal)

Failure mode (test doesn't help):
- ✗ Result is within ±5 K of 334 K with high variance (inconclusive)
- ✗ Numerical instability or divergence (suggests unrelated issue)
- ✗ NaN in temperature measurement (suggests code bug)

---

## Timeline

- **Test execution**: 45-90 minutes (started ~10:10 UTC, expected completion ~12:00-13:00 UTC)
- **Oracle review**: <5 minutes (result is self-interpreting)
- **Next step** (if needed): Step 2 PME grid test (~60 minutes) or implementation planning

---

## Contingency

If test encounters error before completion:
1. Check for obvious issues: module imports, file paths, venv activation
2. Try reducing n_steps or n_waters to test function independently
3. If fundamental issue, report error message and code path to oracle
4. Oracle decides whether to retry with modified parameters or escalate

---

## Relationship to Overall Strategy

This test is **Step 1 of 3** in Sprint B:

**Step 1** (current): dt=0.25 fs test → Determines H1 vs H3
**Step 2** (conditional): PME grid scaling → Confirms if grid is H3 culprit
**Step 3** (conditional): Force analysis → Detailed debugging if needed

If Step 1 result is clear, oracle can approve proceeding directly to implementation (fix H1 or H3 based on diagnosis).

If Step 1 result is unclear, oracle should recommend Step 2.

---

## Cost-Benefit

| Scenario | Benefit | Cost |
|----------|---------|------|
| Clear H1 or H3 result | Can plan 1-2 sprint fix | 90 min CPU time ✓ |
| Inconclusive result | Need more testing | 90 min + Step 2 (60 min) |
| Unmodeled mechanism | Saves time avoiding wrong fix | 90 min + Step 3 (120 min) |

**ROI**: High. Single test eliminates 5+ sprints of wrong-direction investigation.

---

## Ready for Execution

Test is properly configured and running. Oracle can expect result notification within 2 hours (10:10-12:10 UTC, 2026-04-28).
