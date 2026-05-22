# Oracle Critique — Cycle 87

**Target:** `explicit_solvent_validation_comprehensive.md` (v2.1, 2026-04-17)
**Verdict:** **REVISE**
**Confidence:** High
**Approved for execution:** No

## Strategic Assessment

v2.1 is a substantive, honest response to Cycle 86. Six warnings and four suggestions are materially addressed. However, v2.1 retains one execution-blocking grounding error (plan assumes `@pytest.mark.openmm` on a test that doesn't have it) plus three real inconsistencies. All fixable editorially.

## Concerns

### Critical
1. **P1a marker mismatch (execution blocker).** `TestOpenMMSolvationParity` has only `@pytest.mark.integration` + `skipif(not openmm_available())`; plan's CI selectors (`-m 'integration and openmm'` / `-m openmm -xvs`) match zero tests.

### Warnings
2. **P2a integrator/system row contradicts itself.** n=4 baseline has no water, but integrator row says `settle_langevin` (requires water_indices). Use `jmd_simulate.nvt_langevin` for baseline.
3. **P1a Option B force RMSE bound unrealistic.** 0.1 kcal/mol/Å is the 2-charge anchor bound; 1UAO+5000-water uses `< 3.0`. Tier the bound.
4. **Critical-path duration inconsistency.** 5-weeks vs 3–4-weeks cited in three places. Pick one.

### Suggestions
5. P1b field enumeration drift (`dispersion_correction` → `use_dispersion_correction`; missing `openmm_platform`).
6. §7 Checklist missing CSV logging for P2a.
7. §P2a 1.67% tolerance statistical justification doesn't reconstruct for N=4 Langevin.
8. §P3 Done Criteria's 10% Morton threshold missing "proposed" attribution.
9. Risk register missing CI-marker-selector risk.

**Verdict: REVISE | Confidence: high | Concerns: 9 (1 critical, 3 warning, 5 suggestion)**
