# Recon: Explicit Solvent Validation Plan — Cycle 1 Gap Analysis

**Date:** 2026-04-16  
**Purpose:** Ground oracle critique findings in codebase facts

## Key Findings

### 1. SETTLE Scalar Path Status (Oracle Concern #2)

**Finding:** SETTLE scalar path **IS IMPLEMENTED AND MERGED** (commit 6a47281).

**Evidence:**
- `src/prolix/simulate.py:SimulationSpec` contains `rigid_water: bool = False` flag
- `src/prolix/physics/settle.py` provides `settle_langevin()` function (NOT `settle_rattle_langevin` as draft plan stated)
- `simulate.py` line ~2100: `if spec.rigid_water and water_indices is not None: from prolix.physics.settle import settle_langevin`
- Implementation dispatches to `settle_langevin(integrator_energy_fn, shift_fn, dt, ...)`

**Impact on Plan:** 
- **Effort estimate for P1b (PME policy doc) is reasonable.** SETTLE wiring is done; P1b focuses on documentation, not implementation.
- **Clarification needed:** The comprehensive plan said "P1b: PME Grid / Ewald Policy" but should note that SETTLE is already integrated. The "P1b scalar path SETTLE wiring" work mentioned in settle_integration_plan.md is **complete, not pending**.

---

### 2. Solvated Protein Test Status (Oracle Concern #1)

**Finding:** Multiple solvated test files exist with varying scope.

**Tests & System Sizes:**

| Test File | System | Protein? | Water Count | Purpose | Status |
|-----------|--------|----------|-------------|---------|--------|
| `test_solvated_openmm_explicit_parity.py` | 2 TIP3P waters | No | 6 atoms | Minimal anchor | ✅ Done |
| `test_explicit_solvation_parity.py::test_energy_parity` | 1UAO (real protein) | **YES (52 res)** | ~5000 | Full parity test | Depends on 1UAO.pdb presence |
| `test_solvated_explicit_integration.py` | Small merged system | Unclear | Parameterized | Finiteness test | ✅ Done (slow) |
| `test_settle.py` | Various (10–50 waters) | No | Variable | SETTLE validation | ✅ Done |

**Impact on Plan:**
- **P1a target already partially exists** in `test_explicit_solvation_parity.py` but:
  - Depends on external PDB file (`1UAO.pdb`) not in repo
  - Is marked with `@pytest.mark.openmm` (optional dep; not always run)
  - May not enforce strict PME grid/alpha regression (unclear from fixture code)
- **Recommendation:** Either (a) use existing test if 1UAO.pdb is available and it includes OpenMM parity check, OR (b) create new minimal test with bundled system (Ala10 solvated, embedded geometry)

---

### 3. PME Grid Policy Documentation (Oracle Concern #1)

**Finding:** No centralized regression config exists yet.

**Evidence:**
- `test_openmm_explicit_anchor.py` has PME params hardcoded in test docstring
- `docs/source/explicit_solvent/openmm_comparison_protocol.md` referenced in requirements doc but not checked for content
- `settle_integration_plan.md` mentions "regression defaults" but doesn't specify which params

**Missing:**
- Config file (`.yaml` or dataclass) defining canonical PME params
- CI gate enforcing param consistency
- Clear inheritance path for new tests (should all use same grid/alpha policy)

**Impact on Plan:** 
- **P1b is feasible but requires artifact creation** (config file + CI gate)
- Not a blocker; can be done in parallel with P1a

---

### 4. Langevin Parity Strategy (Oracle Concern #3)

**Current Infrastructure:**

| Item | Status | Notes |
|------|--------|-------|
| NVE short-run test | ✅ `test_explicit_slow_validation.py::test_explicit_pbc_nve_short_run_finite` | Energy drift bounded on ~100 atom box |
| NVT mean-T test | ✅ `test_explicit_slow_validation.py::test_explicit_pbc_nvt_mean_temperature_targets_spec` | Mean T validated statistically |
| Langevin step-by-step parity | ❌ Not done | Would require RNG matching or distribution-level comparison |
| Distribution-level metrics doc | ❌ Not done | No published tolerance policy for mean T / variance |

**Gap:** P2a metrics are defined informally. Need explicit tolerance table.

---

## Revisions for Cycle 2

### 1. P1a Clarification
- Distinguish between **P1a-minimal** (new test: Ala10 + 500 waters, ~3k atoms) and **P1a-realistic** (1UAO if available)
- Confirm whether `test_explicit_solvation_parity.py` includes OpenMM parity comparison or just Prolix-internal checks
- If 1UAO test exists and passes, that **satisfies P1a**; else, create new minimal test

### 2. P1b Simplification
- **SETTLE integration is complete**; P1b is **purely documentation**
  - Extract PME params from `test_openmm_explicit_anchor.py`
  - Create `docs/source/explicit_solvent/regression_pme_config.yaml`
  - Add CI gate: weekly rerun of anchor test with frozen config
  - Effort: 2–3 days (as estimated)

### 3. P2a Metrics Detail
- **Specify tolerance table:**
  - System: 4-charge PME box (~100 atoms, 10Å cutoff)
  - Duration: 10 ps (500 steps × 20 fs)
  - Window: mean T over last 5 ps (250 steps) after equilibration
  - Tolerance: 295K ≤ mean T ≤ 305K (target 300K, ±1.67%)
  - Precision: float64 (JAX x64)
  - Platform: CPU + GPU (test on both if feasible)

### 4. Timeline Adjustment
- **P1 effort:** 2 weeks (reduced from 3 if 1UAO test exists and passes)
  - P1a: 1 week (confirm existing test or create new)
  - P1b: 2–3 days (config file + CI gate)
- **P2a effort:** 2 weeks (unchanged; RNG/distribution approach is solid)
- **Critical path remains:** 6–7 weeks total (P1 → P2 → P5)

---

## Remaining Uncertainties

1. **1UAO.pdb location**: Is this file available in the repo or test data? If not, P1a requires new test creation (adds 1 week).
2. **test_explicit_solvation_parity.py OpenMM parity**: Does this test compare Prolix energy/forces against OpenMM's `getState(E, F)`? Or is it Prolix-internal validation only?
3. **Phase 6 profiling trigger thresholds**: `spatial_sorting_profile_gate.md` defines decision logic; needs to be inlined or summarized in comprehensive plan for clarity.

---

## Recommended Changes to Plan (Cycle 2)

- [ ] Clarify P1a target: use existing 1UAO test if available, else create Ala10+waters minimal test
- [ ] Move "SETTLE scalar path wiring" out of P1b; it's complete. P1b is **PME policy documentation only**
- [ ] Add tolerance table to P2a (mean T, variance, window, precision, platform)
- [ ] Revise timeline: P1 is 2 weeks (not 3), if 1UAO test counts; else add 1 week for new test
- [ ] Inline key decision thresholds from Phase 6 gate into P3 section (scatter % trigger for Morton, cell-list switchover metric)
