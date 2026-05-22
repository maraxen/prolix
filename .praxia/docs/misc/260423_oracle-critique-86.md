# Oracle Critique — Cycle 86

**Target:** `explicit_solvent_validation_comprehensive.md` (v2.0, 2026-04-17)
**Verdict:** **REVISE**
**Confidence:** High
**Approved for execution:** No

## Strategic assessment

The v2.0 reset is a genuine improvement over the hallucinated v1.x cycles: it stays inside the physics, keeps scope to a 5–9 week validation roadmap, and correctly identifies the real gaps — 1UAO CI integration, PME regression policy, Langevin distribution-level parity, the Phase 6 profile gate, SLURM wiring, and release docs.

However, a handful of grounding errors — exactly the failure mode v2.0 is trying to prevent — still leak in. A SLURM filename is invented, numeric thresholds are attributed to a doc that doesn't contain them, and one tolerance target contradicts the reference test it's supposed to mirror. These need to be corrected before dispatch; they are editorial fixes, not rewrites.

## Concerns

### Warning (fix before executing)

1. **P4a names a SLURM script that doesn't exist** (`bench_chignolin_explicit.slurm`). Use actual names: `bench_explicit_openmm_pi_so3.slurm`, `bench_chignolin_pi_so3.slurm`.
2. **P1a Option B tolerance contradicts existing 1UAO test.** Existing test uses `atol=40.0` kcal/mol due to PME background shift; plan's `0.5 kcal/mol` is unachievable at that scale. Use force RMSE as primary metric.
3. **P3 quotes decision-tree thresholds not in the gate doc.** `spatial_sorting_profile_gate.md` contains no such numbers. Label them as this plan's proposal or drop them.
4. **P1a's new CI workflow duplicates `openmm-nightly.yml`.** Extend the existing workflow instead.
5. **Two `SimulationSpec` classes — only one has `rigid_water`.** Runbook must specify canonical import path. Also `settle_langevin` location reference is wrong (`src/prolix/physics/settle.py`, not `simulate.py`).
6. **P2a overstates existing-test scale.** Existing NVE/NVT tests use n=4 mock charges, not "~100 atom PBC box". Either build a real 100-atom system (increasing effort) or relabel.

### Suggestions (non-blocking)

7. **P1b/P4b path mismatches.** PME regression config already exists as `REGRESSION_EXPLICIT_PME` in `tests/physics/conftest.py:140`. Schema lives at `docs/source/explicit_solvent/schemas/benchmark_run.schema.json`, not `docs/schemas/`.
8. **P2a tolerance self-inconsistency.** Table says ±1.67%, §4 says 5%. Pick one.
9. **§6b checkbox state mixed.** Separate pre-verified from to-do.
10. **Risk register missing two risks.** Add `SimulationSpec` duplication and PME background shift.

**Verdict: REVISE | Confidence: high | Concerns: 10 (0 critical, 6 warning, 4 suggestion)**
