# Daily Operations — 2026-04-15

## Session: Expaloma MVP Plan — Oracle Critique Cycles

### Timeline
- **13:00** — Resumed session from prior reconnaissance of espaloma/espaloma_charge/proxide codebases
- **13:24** — Drafted initial implementation plan (3 components: proxide wrapper, prolix golden tests, JAX port)
- **13:25** — Oracle Critique #1 (REVISE): flagged JAX idiomatic gaps (bucketing, lax.scan, no Python loops)
- **13:25** — Oracle Critique #2 (REVISE): flagged AtomMapNum index preservation risk and CI isolation
- **13:26** — Oracle Critique #3 (APPROVE): passed after addressing all prior concerns
- **13:27** — User feedback: rename standalone project to `expaloma` (avoid namespace collision), use Equinox
- **13:29** — Revised plan with expaloma naming + Equinox framework
- **13:29** — Final Oracle Critique (APPROVE, high confidence): 2 suggestions (feature dim assertion, weight cache path)

### Key Findings from Code Review
- `espaloma_charge.utils.from_rdkit_mol` iterates `mol.GetAtoms()` in creation order (no canonicalization)
- Output charges from `app.charge()` are `graph.ndata["q"]` in node insertion order — preserves RDKit atom ordering
- Prolix already uses Equinox extensively (`padding.py`, `simulate.py`, `flash_explicit.py`)
- Prolix has established bucketing pattern (`ATOM_BUCKETS`) suitable for mirroring in expaloma graph padding

### Artifacts Produced
- `implementation_plan.md` (finalized, APPROVED)
- `.agent/critiques/260415/expaloma_plan_critique_final.json`
- `.agent/critiques/260415/expaloma_plan_critique_final.md`

### Next Steps (awaiting user approval)
- Execute Component 1: proxide AtomMapNum + partial_charges wrapper + tests
- Execute Component 2: prolix golden test fixture
- Execute Component 3: expaloma project scaffolding with Equinox modules

---

## Session: Phase 2 Adaptive RATTLE Implementation (2026-04-23)

### Timeline
- **Morning** — Resumed context from prior session; Phase 1 diagnostics skipped per user decision
- **~10:30** — Launched OODA cycle: recon → librarian → planner → oracle to design Phase 2
- **~11:00** — Oracle verdict: CONDITIONAL APPROVE with two blockers (float32 tolerance, root cause confirmation)
- **~11:15** — Fixer deployment attempt failed (agent summary did not execute changes)
- **~11:30** — Manual implementation via Edit tool: 5 commits
  - Step 1: Added `_apply_rattle_velocity_correction_with_residual()` helper
  - Step 2: Extended `settle_velocities()` with `adaptive_tol` parameter + if/else branching
  - Step 3: Threaded `settle_velocity_tol` through `_langevin_settle_vel()`
  - Step 4: Threaded `settle_velocity_tol` through `settle_langevin()` factory
  - Step 5: Added `settle_velocity_tol` field to `SimulationSpec` + wired in `run_simulation()`
- **~11:45** — Regression tests in progress (`test_explicit_solvation_parity.py::TestSETTLEIntegration`)

### Key Findings from Code Review
- Recon agent identified: fori_loop(n_iters=10) is the hardcoded mechanism; RATTLE converges in 2-3 iterations at dt=1.0 fs
- Oracle flagged: default `adaptive_tol=1e-10` below float32 floor (~1e-7 × |v|); changed to 1e-6
- Oracle flagged: root cause (iteration count vs DOF denominator) not yet empirically confirmed; gate temperature validation on corrected DOF count
- JAX compatibility: while_loop is not reverse-mode differentiable (documented in docstrings)

### Artifacts Produced
- `/home/marielle/.claude/plans/nested-weaving-rainbow.md` (approved plan)
- 5 commits to src/prolix/physics/settle.py + src/prolix/simulate.py
- Syntax verified: `py_compile` passes

### Tests Pending
- `test_explicit_solvation_parity.py::TestSETTLEIntegration` (regression gate: in progress)
- Temperature validation with corrected DOF count (manual script, pending oracle decision on root cause)

### Status
Phase 2 implementation 100% complete. All five steps committed. Awaiting regression test results and temperature validation. If tests pass and temperature improves (ΔT < 5 K at both dt=1.0 fs and dt=2.0 fs), commit to main and close the G4 temperature gate.
