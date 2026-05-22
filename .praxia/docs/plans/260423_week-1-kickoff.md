# Week 1 Kickoff: Explicit Solvent Validation Plan
**Status:** APPROVED (Oracle Cycle 3, 92% confidence)  
**Go/No-Go Date:** 2026-04-16  
**Target Release Date:** Week 5 (2026-05-14)

---

## Pre-Work Checklist (Do Before Monday Week 1)

Complete these 4 items to confirm execution readiness:

### ✅ C1: P1a CI Integration Path
- [ ] Run locally: `pytest tests/physics/test_explicit_solvation_parity.py::test_energy_parity -m openmm -xvs`
  - **Expected:** Test passes (1UAO solvated protein vs OpenMM parity)
  - **File verified:** `data/pdb/1UAO.pdb` exists ✅
  - **If fails:** Investigate or fallback to P1a Option B (create new minimal test, adds 1 week)
  - **Go/No-Go:** If passes → **PROCEED**; if fails → escalate before Week 1

### ✅ C2: P3/P4a Decoupling Confirmed
- [ ] **Team understanding:** Profiling (P3) informs optimization decisions; does NOT gate benchmarking (P4a)
  - P4a starts Week 3 **regardless of P3 outcome**
  - P3 output is advisory (e.g., "scatter bottleneck found; consider Morton for Phase 6")
  - **Go/No-Go:** Confirm with team → **PROCEED**

### ✅ C3: P2a GPU/CPU Scope Locked
- [ ] **GPU required for CI; CPU optional for local dev**
- [ ] **Variance tolerance:** Accept <2% float64 platform differences (document if found)
- [ ] **Go/No-Go:** Acknowledge → **PROCEED**

### ✅ C4: P5a Runbook Drafting Timeline
- [ ] **Understand:** Runbook can draft Week 2 (parallel with P1a execution)
  - Will reference 1UAO (or Ala10 if Option B)
  - Example script: topology → `SimulationSpec(rigid_water=True)` → `run_simulation`
- [ ] **Go/No-Go:** Confirm → **PROCEED**

---

## Week 1 Work Assignments

### **P1a (1 week): Solvated Protein End-to-End Test**
**Owner:** TBD  
**Goal:** Confirm 1UAO test runs in CI with OpenMM validation  

**Tasks:**
1. [ ] Verify local test runs: `pytest tests/physics/test_explicit_solvation_parity.py -m openmm -xvs`
2. [ ] Check test markers: Confirm `@pytest.mark.integration @pytest.mark.openmm` present
3. [ ] Document energy/force tolerances in test fixture docstring (reference anchor policy: ATOL ≤ 0.5 kcal/mol, force RMSE ≤ 0.1 kcal/mol/Å)
4. [ ] Create CI job: `.github/workflows/explicit-solvent-validation.yml`
   - Weekly run of OpenMM integration tests
   - Triggers on: code changes to `src/prolix/physics/explicit*.py` + test files
5. [ ] Document in `TESTING.md`: How to run solvated protein parity test locally

**Done Criteria:** Test passes locally + CI job merged + documented

---

### **P1b (2–3 days): PME Grid / Ewald Policy Config**
**Owner:** TBD  
**Goal:** Centralize regression PME parameters; enforce consistency in CI  

**Tasks:**
1. [ ] Extract PME params from `tests/physics/conftest.py` (REGRESSION_EXPLICIT_PME already exists ✅)
   - Values: `pme_alpha_per_angstrom=0.34`, `pme_grid_points=32`, `cutoff_angstrom=9.0`, `use_dispersion_correction=False`
2. [ ] Create config file: `docs/source/explicit_solvent/regression_pme_config.yaml`
   ```yaml
   pme_alpha_per_angstrom: 0.34
   pme_grid_points: 32
   cutoff_angstrom: 9.0
   use_dispersion_correction: false
   platform: "Reference"  # deterministic CPU
   ```
3. [ ] Update `test_openmm_explicit_anchor.py`: Load config from YAML (don't hardcode)
4. [ ] Add CI gate: Check that anchor test uses frozen config (weekly validation)
5. [ ] Document in `openmm_comparison_protocol.md`: Link to regression config; explain why it's pinned

**Done Criteria:** Config file exists + anchor test loads it + CI gate merged + documented

---

### **P2a (2–3 weeks, parallel with P1): Langevin Distribution-Level Parity**
**Owner:** TBD  
**Goal:** Validate thermostat behavior via ensemble statistics (mean T, variance)  

**Tasks:**
1. [ ] Create test file: `tests/physics/test_explicit_langevin_parity.py`
   - Markers: `@pytest.mark.slow @pytest.mark.openmm`
2. [ ] Implement test:
   - System: 4-charge PME solvated box (~100 atoms)
   - Duration: 10 ps (500 steps × 20 fs)
   - Log every step: `step, time_ps, T_inst, K, U, E_tot`
   - Compare Prolix `settle_langevin` vs OpenMM `LangevinMiddleIntegrator`
3. [ ] Tolerance table (document in docstring):
   - Mean T: 295–305K (±1.67%)
   - Sampling window: last 5 ps (steps 250–500)
   - Precision: JAX x64 (float64)
   - Platform: GPU (CI); CPU optional (dev)
4. [ ] Create protocol doc: `docs/source/explicit_solvent/l2_dynamics_protocol.md`
   - Define metrics, windows, tolerances
   - Include variance bounds
5. [ ] Run test on CI (slow marker; ~3–5 min)

**Done Criteria:** Test passes on GPU + protocol doc written + tolerances documented + CI job merged

---

### **P3 (Week 2, after P1a): Profiling Gate**
**Owner:** TBD  
**Goal:** Measure PME/NL bottlenecks; decide if Morton/cell-list needed  

**Tasks:**
1. [ ] Create profiling script: `scripts/benchmarks/profile_explicit_scatter.py`
   - Profile on systems: N=1000, 5000, 10000 atoms (explicit periodic)
   - Measure: PME scatter %, NL cost %, direct space %
   - Tool: JAX profiler or `nsys` GPU timeline
2. [ ] Run locally on GPU (not in CI)
3. [ ] Generate JSON report with metrics
4. [ ] Execute decision tree:
   - IF scatter_time_ms / total_time_ms > 15% for N ≥ 5000 → **Morton sort candidate**
   - IF cell_list > 20% faster than JAX-MD → **Consider cell-list switch**
   - ELSE → **Current NL sufficient**
5. [ ] Update `spatial_sorting_profile_gate.md` with findings + recommendation
6. [ ] **Manual gate:** If bottleneck found, document & present to team; **do NOT auto-schedule Phase 6 optimization**

**Done Criteria:** Profiling artifact generated + decision tree executed + gate doc updated + findings documented

---

### **P4a (Week 3–4, parallel with P2a/P3): SLURM Wiring + Smoke CI**
**Owner:** TBD  
**Goal:** Wire cluster benchmarking; add local smoke test to CI  

**Tasks:**
1. [ ] Set up CI gate: Local smoke test on every PR
   - Run: `python scripts/benchmarks/prolix_vs_openmm_speed.py --json`
   - System: T0 minimal PME (two-charge box)
   - Output: JSON with schema 1.0
   - Time budget: ~10s on GPU
2. [ ] Parameterize SLURM templates for site (Engaging cluster or generic):
   - Review existing: `scripts/slurm/bench_chignolin_*.slurm`
   - Update for current partition/GPU availability
3. [ ] Document cluster setup: `docs/source/explicit_solvent/cluster_benchmarking_setup.md`
   - How to run manual cluster job
   - Results collection (where to find JSON outputs)
4. [ ] Optional: Add cron job for weekly cluster runs (if infrastructure supports)

**Done Criteria:** Smoke test in CI + SLURM templates parameterized + cluster doc written + tested

---

### **P4b (Week 3, parallel with P4a): Benchmark Schema + Collection**
**Owner:** TBD  
**Goal:** Formalize benchmark result format; create collection tool  

**Tasks:**
1. [ ] Verify/update benchmark JSON schema: `docs/schemas/benchmark_run.schema.json`
   - Fields: `timestamp`, `commit_hash`, `platform`, `system_size`, `kernel`, `wall_time_ms`, `throughput_steps_per_sec`, `metadata`
2. [ ] Create collection tool: `scripts/benchmarks/collect_results.py`
   - Aggregate local + cluster JSON files
   - Summarize into table/CSV
3. [ ] Document in `explicit_solvent_benchmarks.md`: How results are collected + stored

**Done Criteria:** Schema finalized + collection tool works + documentation updated

---

### **P5a (Week 4, after P1 complete): Runbook Updates**
**Owner:** TBD  
**Goal:** Update production-ready runbook with finalized API  

**Tasks:**
1. [ ] Update `explicit_solvent_runbook.md`:
   - Happy path: topology → `SimulationSpec` (with `rigid_water=True`) → minimize → `run_simulation`
   - Example system: 1UAO (or Ala10 if P1a Option B)
   - Full code example from structure to results
2. [ ] Clarify SETTLE usage:
   - "SETTLE applies only to water molecules (O-H, H-H constraints)"
   - "Solute uses RATTLE for non-water constraints"
   - "Enable with `SimulationSpec(rigid_water=True)` if water indices available"
3. [ ] PME policy: Link to regression config created in P1b
4. [ ] RF/DSF usage: Reference `electrostatic_methods.py` + caveat that PME is default
5. [ ] Error cases: Clear messages for common failures (missing water indices, unsupported constraints)
6. [ ] Run example locally to verify no crashes

**Done Criteria:** Runbook updated + example runs locally + documented + reviewed

---

### **P5b (Week 5, after P5a): Release Notes + Changelog**
**Owner:** TBD  
**Goal:** Finalize release documentation  

**Tasks:**
1. [ ] Draft release notes:
   - "Explicit solvent module: stable, production-ready"
   - "Validated against OpenMM for PME, NL, SETTLE, RF/DSF"
   - "Tested on: [list target systems]"
2. [ ] Known limitations:
   - "RDF observable framework post-release (skeleton doc available)"
   - "RESPA (multi-rate integrator) deferred"
   - "Performance optimizations (Morton, cell-list) gated by profiling (Phase 6)"
3. [ ] Update CHANGELOG.md with P1–P5 work items
4. [ ] Tag release: `git tag v<X.Y.Z>-explicit-solvent`

**Done Criteria:** Release notes written + changelog updated + tag created

---

## Parallel Work (Non-Blocking)

### **P2b (Post-Release, 1–2 days): RDF Observables Skeleton**
**Timeline:** After release (Week 5+)  
**Owner:** TBD  
**Tasks:**
1. [ ] Create skeleton protocol: `docs/source/explicit_solvent/l3_observables_protocol.md`
   - Define RDF metrics (water-water, water-protein, dihedral)
   - Tolerance policy (peak within 1 bin, height within 10%)
   - Reference trajectory format
2. [ ] Create empty test scaffold: `tests/physics/test_explicit_solvation_rdf.py`
   - Docstring explaining what to implement
   - Placeholder function

**Not in release; enables post-release implementation.**

---

## Dependencies & Blockers

| Item | Depends On | Blocker? |
|------|-----------|----------|
| P1a | 1UAO.pdb (verified ✅) | No (fallback: create new test) |
| P1b | Nothing | No |
| P2a | Nothing | No (quality assurance, not release-blocking) |
| P3 | P1a (just for confidence; can run independently) | No (manual gate only) |
| P4a | Nothing | No (P3 output is advisory, not gating) |
| P4b | Nothing | No |
| P5a | P1a + P1b (complete) | **YES** (blocks release) |
| P5b | P5a (complete) | **YES** (blocks release) |

**Critical Path:** P1a (1 wk) → P1b (3 days) → P5a (1 wk) → P5b (1 wk) = **3–4 weeks elapsed**

---

## Success Criteria (Week 5 Release Gate)

Before tagging release, verify:

- [ ] **P1a:** 1UAO test passes locally + CI gate merged + documented
- [ ] **P1b:** PME config file exists + anchor test uses it + CI weekly gate merged
- [ ] **P5a:** Runbook updated + example runs without errors + documented
- [ ] **P5b:** Release notes written + changelog updated + tag created

**Optional (nice-to-have, can follow release):**
- [ ] **P2a:** Langevin parity test passes (high-quality assurance)
- [ ] **P3:** Profiling artifact generated + decision documented
- [ ] **P4a/P4b:** SLURM + schema finalized
- [ ] **P2b:** RDF skeleton (post-release)

---

## Escalation Contacts

| Issue | Contact |
|-------|---------|
| P1a test fails (1UAO system) | [Code owner for explicit solvent tests] |
| OpenMM optional dep not available | [DevOps / Python environment maintainer] |
| CI jobs not triggering | [DevOps / GitHub Actions maintainer] |
| Profiling results unclear | [GPU optimization expert] |
| Timeline slipping | [Project lead] |

---

## Week-by-Week Summary

| Week | Critical Path | Parallel Work | Status |
|------|---------------|---------------|--------|
| **1** | P1a, P1b | P2a start | Tests written + configs created |
| **2** | (P1a/P1b wrap) | P2a, P3 start | Profiling begins; dynamics tests in progress |
| **3** | (Waiting for P1 done) | P2a, P3, P4a/b | Benchmarking infrastructure wired |
| **4** | P5a (runbook) | P2a complete | Runbook updated; finalized API locked |
| **5** | P5b (release) | P2b (skeleton) | Release tagged; post-release work ready |

---

## How to Unblock if Stuck

| Scenario | Resolution |
|----------|-----------|
| 1UAO test fails locally | Debug with OpenMM team; if unresolvable in 1 day, activate P1a Option B (new minimal test, +1 week) |
| P2a tolerances too tight | Adjust mean T tolerance from ±1.67% to ±2–3%; re-baseline on reference system |
| P3 profiling inconclusive | Run on 3+ systems; if scatter is borderline (12–18%), document & defer decision to Phase 6 planning |
| SLURM setup delayed | Smoke CI test can run on local GPU; cluster jobs can be manual for now; automate later |
| Timeline slipping | De-prioritize P2a/P3/P4b; keep P1/P5 on critical path; push non-blocking work to post-release |

---

**GO/NO-GO DECISION:** If all pre-work checks pass → **GO for Week 1** (2026-04-22)  
**Contact:** [Project lead] if any blockers found during pre-work
