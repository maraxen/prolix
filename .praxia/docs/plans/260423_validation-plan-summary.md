# Explicit Solvent Validation Plan: Executive Summary

**Status:** ✅ **APPROVED & EXECUTION-READY** (Oracle Cycle 3, 92% confidence)  
**Plan Document:** `explicit_solvent_validation_comprehensive.md`  
**Kickoff Guide:** `WEEK_1_KICKOFF.md`  
**Timeline:** 5 weeks to release; 3–4 week critical path

---

## What This Plan Does

Validates the explicit solvent engine (water + protein, PME, NL, SETTLE constraints) from static parity through production release. Covers:

1. **P1a:** Solvated protein parity test vs OpenMM (1UAO + 5000 waters)
2. **P1b:** PME grid/Ewald regression configuration
3. **P2a:** Langevin thermostat validation (ensemble statistics)
4. **P3:** Performance profiling (Morton/cell-list decision gate)
5. **P4a/P4b:** Cluster benchmarking automation + schema
6. **P5a/P5b:** Production runbook + release notes

---

## Critical Decisions (Locked)

| Decision | Your Call | Rationale |
|----------|-----------|-----------|
| **P2b (RDF Observables)** | Post-Release Skeleton Only | RDF is research-grade, not blocking for functional validation. Doc + empty test scaffold now; implementation post-release. |
| **P3 (Profiling Gate)** | Manual Gate (Not Auto-Scheduled) | Profiling informs optimization planning for Phase 6+. Does NOT block benchmarking (P4a) or release. |
| **P1a (1UAO Test)** | Use Existing Test | File verified in repo (data/pdb/1UAO.pdb). Existing test written. Faster path (1 week vs 2 weeks). |

---

## Timeline (Release Path)

**Critical Path (Blocks Release):**
```
Week 1: P1a (1 wk) + P1b (3 days)
  ↓
Week 4: P5a (1 wk) [Runbook]
  ↓
Week 5: P5b (1 wk) [Release]
─────────────────────────────
Total: 3–4 weeks elapsed
```

**Parallel (Non-Blocking):**
- **P2a (2–3 weeks):** Langevin dynamics validation — quality assurance; can follow release
- **P3 (1 week):** Performance profiling — informational for future optimization
- **P4a/P4b (1–2 weeks):** Cluster infrastructure — benchmarking; can be post-release
- **P2b (1–2 days):** RDF skeleton — post-release implementation framework

**Total Project:** 5–6 weeks (critical path + key parallel work)

---

## What's Ready to Go

✅ **1UAO.pdb file** — In repo; verified  
✅ **test_explicit_solvation_parity.py** — Exists; 1UAO fixture complete  
✅ **SETTLE scalar path** — Merged (commit 6a47281); `settle_langevin` + `rigid_water` flag  
✅ **PME regression config** — Centralized in conftest.py (REGRESSION_EXPLICIT_PME)  
✅ **Tolerance tables** — Documented (P2a metrics, force/energy ATOL)  
✅ **Profiling gate doc** — `spatial_sorting_profile_gate.md` exists  
✅ **SLURM templates** — Exist in `scripts/slurm/`; parameterization needed  

---

## What Starts Week 1

| Phase | Week 1 Activity | Owner | Done By |
|-------|-----------------|-------|---------|
| **P1a** | Confirm 1UAO test runs locally; add to CI gate | TBD | End Week 1 |
| **P1b** | Create PME config YAML; update anchor test | TBD | End Week 1 |
| **P2a** | Write Langevin parity test; implement distribution-level metrics | TBD | End Week 3 |
| **P3** | Create profiling script (start Week 2 after P1a) | TBD | End Week 2 |
| **P4a/b** | Parameterize SLURM; add smoke CI test (start Week 3) | TBD | End Week 4 |
| **P5a** | Runbook update (start Week 4 after P1 done) | TBD | End Week 4 |
| **P5b** | Release notes + tag (start Week 5 after P5a) | TBD | End Week 5 |

---

## Pre-Work Checklist (Do Before Monday Week 1)

- [ ] **C1:** Run `pytest tests/physics/test_explicit_solvation_parity.py::test_energy_parity -m openmm -xvs` locally
  - If passes → **PROCEED with P1a Option A** (1 week)
  - If fails → **Fallback to Option B** (create new minimal test, +1 week)
- [ ] **C2:** Team confirms P3/P4a decoupling (profiling doesn't gate benchmarking)
- [ ] **C3:** GPU required for CI; CPU optional; <2% variance accepted
- [ ] **C4:** Runbook can draft Week 2 (parallel with P1a execution)

**Target Go/No-Go:** Monday, 2026-04-22

---

## Release Criteria (Week 5 Gate)

✅ **Must Have:**
- P1a test passes (1UAO solvated parity vs OpenMM)
- P1b PME config centralized + CI gate merged
- P5a runbook updated + example runs
- P5b release notes + changelog + tag created

⚠️ **Nice-to-Have (can be post-release):**
- P2a Langevin test passes (high-quality assurance)
- P3 Profiling complete + decision documented
- P4a/b Cluster automation + schema finalized
- P2b RDF skeleton doc

---

## Execution Documents

**Use These:**
1. **`explicit_solvent_validation_comprehensive.md`** — Full plan (read before starting)
2. **`WEEK_1_KICKOFF.md`** — Week-by-week tasks, owners, done criteria (copy to project board)
3. **`.agent/docs/plans/revisions_c1_to_c2.md`** — What changed between oracle cycles (reference)

---

## Risk Summary

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| **1UAO test fails** | Very Low | Fallback to Option B (+1 week new test) |
| **OpenMM optional dep unavailable** | Low | Pre-work check catches this; escalate early |
| **P2a variance > tolerance** | Low | Adjust tolerance (e.g., ±2–3% instead of 1.67%) |
| **Profiling inconclusive** | Medium | Document & defer Phase 6 decision (doesn't block release) |
| **Timeline slip 1–2 weeks** | Medium | De-prioritize P2a/P3; keep P1/P5 on critical path |

**Overall Risk Level:** 🟢 **LOW** — Plan has explicit fallbacks; all blockers identified upfront

---

## Key Contacts & Escalation

| Issue | Escalate To |
|-------|-------------|
| Tests fail; need debugging | Explicit solvent code owner |
| OpenMM CI infrastructure | DevOps / Python environment maintainer |
| Profiling results interpretation | GPU optimization expert |
| Timeline concerns | Project lead |

---

## Next Steps

1. **Confirm 3 decisions** (already locked via user input) ✅
2. **Run pre-work checklist** (C1–C4) this week
3. **Schedule Week 1 kickoff** (2026-04-22 target)
4. **Assign owners** to P1a/P1b/P2a/P3/P4a/P4b/P5a/P5b
5. **Copy WEEK_1_KICKOFF.md to project board** (linear, Jira, etc.)

---

## Questions?

- **Plan unclear?** → See `explicit_solvent_validation_comprehensive.md` sections 1–9
- **Week 1 confused?** → See `WEEK_1_KICKOFF.md` assignments
- **Why this decision?** → See `.agent/docs/plans/revisions_c1_to_c2.md` (oracle feedback)
- **What changed?** → 3-cycle oracle critique; all major concerns resolved

---

**Plan Status:** ✅ **READY TO EXECUTE**  
**Approval:** Oracle Cycle 3 (92% confidence)  
**Target Release:** Week 5 (2026-05-14)
