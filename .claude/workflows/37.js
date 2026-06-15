// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/37.toml
// Regenerate: praxia dw emit-sprint 37.toml
// task_id: 37   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain () runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (A,B,C,D,E,F,G,H,I) run concurrently.

export const meta = {
  name: "37",
  description: "Parallel sprint: 3 research librarian sweeps, 2 sub-spec planning artifacts, 3 greenfield API skeleton items. All tracks independent — fan-out fully concurrent. Prepares the research foundation and API groundwork for the paper's three claims.",
  phases: [
    { title: "Track A -- HP4 sub-spec: ANI-1x DFT-forces subset curation criteria (#328)" },
    { title: "Track B -- HP1 sub-spec: migration policy for legacy entry-points (#327)" },
    { title: "Track C -- EnsemblePlan stub + API skeleton (spec-only; run() deferred to Sprint 38) (#261)" },
    { title: "Track D -- Bundle.from_pdb factory (#281)" },
    { title: "Track E -- Observable protocol + Trajectory eqx.Module (#284)" },
    { title: "Track F -- DR-claim1-1: Hetero-batched MD precedent scan (#255)" },
    { title: "Track G -- DR-paper-1: Venue fit scan -- JCP/JCTC/JCIM/JOSS/ICML/ICLR/MLSys (#304)" },
    { title: "Track H -- DR-paper-2: Recent MD-engine papers -- narrative tropes and reviewer criticisms (#305)" },
    { title: "Track I -- xtrax.tiling integration spec: adopt xtrax BatchPlanner as prolix planner backend (#1842)" },
  ],
};

const TASK_ID = "37";
const MAX_FIX_RETRIES = 2;

const VERDICT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["item_id", "verdict", "summary"],
  properties: {
    item_id: { type: "string" },
    verdict: { type: "string", enum: ["PASS", "NEEDS_WORK", "FAIL"] },
    summary: { type: "string" },
    issues: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["where", "problem", "fix"],
        properties: {
          where: { type: "string" },
          problem: { type: "string" },
          fix: { type: "string" },
        },
      },
    },
  },
};

const RESEARCH_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["surfaces", "recommendation"],
  properties: {
    surfaces: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["surface", "rules_location", "frontmatter", "settings_triggers", "distribution"],
        properties: {
          surface: { type: "string" },
          rules_location: { type: "string" },
          frontmatter: { type: "string" },
          settings_triggers: { type: "string" },
          distribution: { type: "string" },
          confidence: { type: "string" },
        },
      },
    },
    cross_surface_differences: { type: "array", items: { type: "string" } },
    recommendation: { type: "string" },
    open_questions: { type: "array", items: { type: "string" } },
  },
};

// Shared context for the writing tracks (from recon, task 37).
const EMITTER_CTX = `task_id: 260614_sprint37_paper_preground\nProject root: /home/marielle/projects/prolix\n\nBackground:\n- Sprint 36 / G1 complete (c5ddef3). dt=1.0 fs cap lifted at production scale (n>=16, gamma=10 ps^-1).\n- Phase 5 is closed. Next focus: paper critical path and API skeleton.\n- Paper (backlog #172) depends on: §7.1 figure (259), B1-full benchmark (270), WASM export (276-278), research evidence (RR1-RR7).\n- Biggest unblocked unlocks: 328->260->259 (section 7.1 figure path); 327->250->V1-V7 (verification suite); 261+281+284 (API skeleton for EnsemblePlan + Bundle + Observable).\n- EnsemblePlan does NOT exist yet. BatchPlanner exists at src/prolix/tiling/planner.py:91.\n- MolecularBundle exists at src/prolix/types/bundles.py:101 -- no factory classmethods.\n- No prolix/api/ module exists; __init__.py exports only legacy symbols.\n- No Observable protocol or Trajectory eqx.Module exists.\n\nKey invariants:\n- All Python via \`uv run python\` / \`uv run pytest\`; never bare python\n- task_id: 260614_sprint37_paper_preground in every transduction_log call\n- Research tracks: librarian agents synthesize to .praxia/docs/research/ and .praxia/research/synthesis.jsonl\n- Spec tracks: specification-specialist agents write to .praxia/docs/superpowers/specs/\n- Implementation tracks: fixer agents write to src/prolix/api/; worktree isolated\n\nSprint DAG:\n  All 8 tracks are CONCURRENT -- no cross-track dependencies.\n  a (328) + b (327) + c (261) + d (281) + e (284) + f (255) + g (304) + h (305)\n`;

// ---- per-track stage helpers ---------------------------------------------
const fixer = (prompt, label, phaseName, isolation = null) => {
  const opts = { agentType: "fixer", label, phase: phaseName };
  if (isolation) opts.isolation = isolation;
  return agent(`${prompt}\n\nWhen done, end your message with 'verdict: done' on its own line.`, opts);
};

const reviewer = (itemId, prompt, label, phaseName, isolation = null) => {
  const opts = { agentType: "reviewer", label, phase: phaseName, schema: VERDICT_SCHEMA };
  if (isolation) opts.isolation = isolation;
  return agent(prompt, opts);
};

// Sequential implement->review with bounded NEEDS_WORK repair cycles.
async function track(itemId, phaseName, fixerPrompt, reviewerPrompt, isolation = null) {
  log(`[${itemId}] implement`);
  await fixer(fixerPrompt, `fix:${itemId}`, phaseName, isolation);
  let verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}`, phaseName, isolation);
  for (let retry = 0; retry < MAX_FIX_RETRIES && verdict && verdict.verdict === "NEEDS_WORK"; retry++) {
    log(`[${itemId}] NEEDS_WORK — repair cycle ${retry + 1}/${MAX_FIX_RETRIES}`);
    const issues = (verdict.issues || [])
      .map((i) => `- ${i.where}: ${i.problem} -> ${i.fix}`)
      .join("\n");
    await fixer(
      `${fixerPrompt}\n\nA reviewer found issues — fix exactly these, nothing else:\n${issues}`,
      `fix:${itemId}:repair:${retry}`,
      phaseName,
      isolation
    );
    verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}:re:${retry}`, phaseName, isolation);
  }
  return verdict;
}

// ===== TRACK A — Track A -- HP4 sub-spec: ANI-1x DFT-forces subset curation criteria (#328) =========================
const trackA = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260614_sprint37_paper_preground\nProject root: /home/marielle/projects/prolix\n\nROLE: specification-specialist\n\nOBJECTIVE: Write the ANI-1x DFT-forces subset curation sub-spec that will gate HP4 (#260).\nThis is a planning document -- no code changes. Draft the spec, commit it, mark the work done.\n\nBACKGROUND:\nSection 7.1 of the prolix paper ("Differentiable bonded-parameter fitting on hetero ANI-1x ensemble")\nneeds a curated subset of the ANI-1x dataset. HP4 (#260) is "ANI-1x DFT-forces subset curation\nsub-spec" -- before anyone can run the curation pipeline, this document must define WHAT to curate\nand WHY.\n\nSTEPS:\n\n1. Read context files:\n   - CLAUDE.md (search for "ANI-1x" and "7.1" to find existing description)\n   - Any roadmap spec at .praxia/docs/specs/ or .praxia/docs/superpowers/specs/\n   Check: does a prolix long-horizon roadmap exist? Read it for the HP4 framing.\n\n2. Write the sub-spec at .praxia/docs/superpowers/specs/260614_HP4-ani1x-subset.md\n\n   REQUIRED SECTIONS:\n   a) Objective (1 paragraph): what ANI-1x is, why this subset enables section 7.1\n   b) Selection criteria:\n      - System scale: organic small molecules at dipeptide scale (<=50 heavy atoms)\n      - Reference level of theory: wB97X-D/6-31G* (the ANI-1x DFT level)\n      - Diversity: cover H, C, N, O, S, F, Cl element set (no metals/charged)\n      - Target: 16 systems (confirmed by CLAUDE.md) -- explain the 16 choice rationale\n      - Must include at least: 2 amino-acid dipeptides, 2 drug-fragment scaffolds, 1 sugar-like\n   c) Curation pipeline (pseudocode-level):\n      - Download source: the ANI-1x HDF5 file from GitHub (torchani/data or zenodo)\n      - Filter step: apply selection criteria\n      - Store format: data/ani1x_subset/ -- one .npz per system with keys:\n          positions: (N_conf, N_atoms, 3) in Angstrom; species: (N_atoms,) str;\n          energies: (N_conf,) in Hartree; forces: (N_conf, N_atoms, 3) in Hartree/Angstrom\n      - Metadata: per-system JSON with {atom_count, element_set, n_conformers, sha256}\n      - Reproducibility: record ANI-1x dataset version + SHA-256 hash of each source HDF5\n   d) Integration with section 7.1: how the curated subset feeds into the fitting script\n   e) Gate criteria for HP4 (#260): what "done" looks like (data dir exists, all SHAs match)\n\n3. Append one-line entry to .praxia/docs/INDEX.md under Specs.\n\n4. Commit:\n   git add .praxia/docs/superpowers/specs/260614_HP4-ani1x-subset.md .praxia/docs/INDEX.md\n   git commit -m "spec(hp4): ANI-1x DFT-forces subset curation criteria (#328)"\n\nGATE: .praxia/docs/superpowers/specs/260614_HP4-ani1x-subset.md exists with all 5 sections.\n`,
    { agentType: "librarian", label: "research:328", phase: "Track A -- HP4 sub-spec: ANI-1x DFT-forces subset curation criteria (#328)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK B — Track B -- HP1 sub-spec: migration policy for legacy entry-points (#327) =========================
const trackB = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260614_sprint37_paper_preground\nProject root: /home/marielle/projects/prolix\n\nROLE: specification-specialist\n\nOBJECTIVE: Write the HP1 migration policy sub-spec that will gate HP1 (#250).\nThis is a planning document -- no code changes.\n\nBACKGROUND:\nHP1 (#250) is "Migration policy decided for legacy entry-points." Before implementing\ndeprecation warnings, the policy (what, replacement, timeline) must be written and agreed.\n\nSTEPS:\n\n1. Recon the current public API:\n   Read src/prolix/__init__.py to find all currently exported symbols.\n   Also grep for "deprecated" or "DeprecationWarning" in src/prolix/ to see what is already warned.\n   Read src/prolix/types/bundles.py:101 to understand MolecularBundle (the replacement API anchor).\n\n2. Write the sub-spec at .praxia/docs/superpowers/specs/260614_HP1-migration-policy.md\n\n   REQUIRED SECTIONS:\n   a) Context: prolix v1.x public API; what changed in v1.0; why deprecation is needed\n   b) Legacy entry-points table (fill from recon):\n      | Symbol | Module | Status | Replacement | Timeline |\n      Cover: batched_produce, LangevinState (public re-export), pad_protein,\n      PaddedSystem, collate_batch, make_batched_energy_fn -- confirm if still exported\n      and add any others found via recon.\n   c) Deprecation timeline:\n      - v1.2: emit DeprecationWarning on every legacy import (no removal yet)\n      - v2.0: remove all legacy entry-points\n      - Keep the new Bundle/EnsemblePlan API stable across v1.x (no breaking changes)\n   d) Implementation blueprint for the v1.2 deprecation pass:\n      - Pattern: __init__.py wrapper that emits DeprecationWarning then calls the new API\n      - Must not break existing tests during the warn phase\n   e) CHANGELOG migration table format:\n      v2.0 migration: \`prolix.PaddedSystem\` -> \`prolix.types.bundles.MolecularBundle.from_pdb()\`\n      (one line per symbol)\n   f) Gate criteria for HP1 (#250): what "decided" looks like\n\n3. Commit:\n   git add .praxia/docs/superpowers/specs/260614_HP1-migration-policy.md\n   git commit -m "spec(hp1): migration policy for legacy entry-points (#327)"\n\nGATE: .praxia/docs/superpowers/specs/260614_HP1-migration-policy.md exists with all 6 sections.\n`,
    { agentType: "librarian", label: "research:327", phase: "Track B -- HP1 sub-spec: migration policy for legacy entry-points (#327)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK C — Track C -- EnsemblePlan stub + API skeleton (spec-only; run() deferred to Sprint 38) (#261) =========================
const trackC = () =>
  track(
    "261",
    "Track C -- EnsemblePlan stub + API skeleton (spec-only; run() deferred to Sprint 38) (#261)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint37_paper_preground\nProject root: /home/marielle/projects/prolix\n\nROLE: fixer -- implementation only; Edit existing files or create new ones; no Write on existing files\n\nOBJECTIVE: Create the EnsemblePlan class STUB with the correct interface but WITHOUT a working run()\nimplementation. The full run() wiring to a BatchPlanner is deferred to Sprint 38, because Track I\n(xtrax.tiling integration, item #1842) must decide the planner backend first. Do NOT implement\nphysics simulation logic in run() here.\n\nBACKGROUND (recon confirmed 2026-06-14):\n- EnsemblePlan does NOT exist anywhere -- fully greenfield.\n- prolix BatchPlanner: src/prolix/tiling/planner.py:91 -- may be replaced by xtrax.BatchPlanner in Sprint 38.\n- MolecularBundle: src/prolix/types/bundles.py:101 -- eqx.Module, no factory methods yet.\n- No prolix/api/ module exists. This track creates it.\n- xtrax.tiling (~/projects/xtrax) provides a more mature planner backend; integration pending (#1842).\n\nSTEPS:\n\n1. RECON (read before writing anything):\n   Read src/prolix/tiling/planner.py (full file) -- understand BatchPlanner + BatchPlan interface.\n   Read src/prolix/types/bundles.py (lines 101-197) -- understand MolecularBundle fields.\n   Read src/prolix/__init__.py -- see current exports.\n\n2. Create src/prolix/api/__init__.py:\n   \`\`\`python\n   from prolix.api.ensemble_plan import EnsemblePlan\n   __all__ = ["EnsemblePlan"]\n   \`\`\`\n\n3. Create src/prolix/api/ensemble_plan.py:\n   - EnsemblePlan as a plain Python class (NOT eqx.Module yet -- defer until planner is decided)\n   - Constructor: EnsemblePlan(bundles: list, planner=None)\n     - Stores self.bundles = bundles\n     - If planner provided, calls planner.plan(bundles) and stores self.batch_plan\n     - If planner is None, sets self.batch_plan = None (deferred)\n   - Method: run(n_steps: int, dt: float, kT: float, seed: int = 0) -> dict:\n     - Raise NotImplementedError with message:\n       "EnsemblePlan.run() implementation pending xtrax.tiling integration (#1842).\n        Expected in Sprint 38. Use settle_langevin directly for now."\n   - Docstring: explain that run() will be implemented after #1842 (xtrax.tiling) lands.\n\n4. Edit src/prolix/__init__.py -- add at the end (do NOT remove existing exports):\n   \`\`\`python\n   # New API (v1.1+) -- EnsemblePlan.run() pending xtrax.tiling integration (#1842)\n   from prolix.api import EnsemblePlan\n   \`\`\`\n\n5. Create tests/api/__init__.py (empty) and tests/api/test_ensemble_plan.py:\n   - test_ensemble_plan_construction: create EnsemblePlan([]), assert hasattr(ep, 'bundles')\n   - test_ensemble_plan_run_raises_not_implemented:\n     ep = EnsemblePlan([])\n     with pytest.raises(NotImplementedError):\n         ep.run(n_steps=10, dt=0.5, kT=2.479e-3)\n   - test_ensemble_plan_with_planner: create EnsemblePlan([], planner=FakePlanner())\n     where FakePlanner has plan() returning None -- assert ep.batch_plan is None\n\n6. Run: uv run pytest tests/api/test_ensemble_plan.py -v -- must pass.\n\n7. Commit:\n   git add src/prolix/api/ tests/api/ src/prolix/__init__.py\n   git commit -m "feat(api): EnsemblePlan stub -- interface defined, run() deferred to Sprint 38 (#261)"\n\nGATE: \`uv run pytest tests/api/test_ensemble_plan.py -v\` PASS; \`from prolix.api import EnsemblePlan\` works;\nrun() raises NotImplementedError with clear message referencing #1842.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify EnsemblePlan stub:\n1. src/prolix/api/ensemble_plan.py exists with EnsemblePlan class.\n2. Constructor stores bundles and optionally batch_plan.\n3. run() raises NotImplementedError mentioning "#1842".\n4. tests/api/test_ensemble_plan.py exists with >=3 tests (construction, run raises, planner kwarg).\n5. Run: \`uv run pytest tests/api/test_ensemble_plan.py -v\` -- all PASS.\n6. \`from prolix.api import EnsemblePlan\` succeeds.\n7. No regressions: \`uv run pytest -m "not slow" --ignore=tests/api/ -q\` still passes.\nPASS if all 7 checks pass.\n`,
    "worktree"
  );

// ===== TRACK D — Track D -- Bundle.from_pdb factory (#281) =========================
const trackD = () =>
  track(
    "281",
    "Track D -- Bundle.from_pdb factory (#281)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint37_paper_preground\nProject root: /home/marielle/projects/prolix\n\nROLE: fixer -- implementation only; Edit existing files or create new ones; no Write on existing files\n\nOBJECTIVE: Add MolecularBundle.from_pdb(path, forcefield='amber14') classmethod.\nGates tutorial notebooks and unblocks EnsemblePlan.from_bundles (#283).\n\nBACKGROUND (recon confirmed 2026-06-14):\n- MolecularBundle: src/prolix/types/bundles.py:101 -- eqx.Module, 196 lines, NO factory classmethods.\n- Track C (261) creates prolix/api/ in parallel -- do NOT depend on it.\n\nSTEPS:\n\n1. RECON (read before writing):\n   Read src/prolix/types/bundles.py (full file) -- every field of MolecularBundle.\n   Grep tests/ for "MolecularBundle(" to see how it is constructed in tests.\n   Check src/prolix/data/ for PDB/topology utilities.\n   Check pyproject.toml for available deps (parmed, mdtraj, pdbfixer, openmm, etc).\n\n2. Add classmethod to MolecularBundle in src/prolix/types/bundles.py:\n\n   @classmethod\n   def from_pdb(\n       cls,\n       path: str,\n       forcefield: str = "amber14",\n   ) -> "MolecularBundle":\n       '''Load a PDB file and build a MolecularBundle with AMBER14 parameters.\n\n       Requires parmed. Positions in Angstrom (AKMA); masses in AKMA mass units.\n       Raises NotImplementedError if parmed is unavailable.\n       Raises FileNotFoundError if path does not exist.\n       Raises ValueError if forcefield is not supported.\n       '''\n       if not Path(path).exists():\n           raise FileNotFoundError(f"PDB not found: {path}")\n       SUPPORTED = {"amber14"}\n       if forcefield not in SUPPORTED:\n           raise ValueError(f"Unsupported forcefield '{forcefield}'. Supported: {SUPPORTED}")\n       try:\n           import parmed\n       except ImportError:\n           raise NotImplementedError(\n               "MolecularBundle.from_pdb requires parmed. Install with: uv add parmed"\n           )\n       # ... actual loading using parmed ...\n\n   Use the recon findings to fill in the actual field construction from parmed output.\n   If parmed is not available in the environment, write the stub and skip the body with\n   a TODO comment -- the gate allows skipped tests.\n\n3. Create or append to tests/types/test_bundles_factory.py:\n   - test_bundle_from_pdb_wrong_path_raises: assert FileNotFoundError\n   - test_bundle_from_pdb_bad_forcefield_raises: assert ValueError\n   - test_bundle_from_pdb_returns_bundle (skip if parmed absent): load small PDB, assert\n     isinstance(result, MolecularBundle) and jnp.all(jnp.isfinite(result.positions))\n\n4. Run: uv run pytest tests/types/test_bundles_factory.py -v -- must pass (skips OK if dep absent).\n\n5. Commit:\n   git add src/prolix/types/bundles.py tests/types/test_bundles_factory.py\n   git commit -m "feat(api): MolecularBundle.from_pdb factory (#281)"\n\nGATE: from_pdb classmethod exists; error-path tests pass; main test passes or is correctly skipped.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Bundle.from_pdb:\n1. MolecularBundle.from_pdb classmethod exists in src/prolix/types/bundles.py.\n2. Raises FileNotFoundError on non-existent path.\n3. Raises ValueError on unsupported forcefield.\n4. Raises NotImplementedError (not ImportError) when parmed is absent.\n5. tests/types/test_bundles_factory.py exists with >=3 tests.\n6. Run: \`uv run pytest tests/types/test_bundles_factory.py -v\` -- passes or skips (no failures).\n7. No regressions in existing types tests.\nPASS if all 7 checks pass (skipped dep tests count as pass).\n`,
    "worktree"
  );

// ===== TRACK E — Track E -- Observable protocol + Trajectory eqx.Module (#284) =========================
const trackE = () =>
  track(
    "284",
    "Track E -- Observable protocol + Trajectory eqx.Module (#284)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint37_paper_preground\nProject root: /home/marielle/projects/prolix\n\nROLE: fixer -- implementation only; Edit or create files; no Write on existing files\n\nOBJECTIVE: Define the Observable runtime_checkable Protocol and Trajectory eqx.Module.\nThese are the output types EnsemblePlan.run() will produce (convergence with #261 in Sprint 38).\n\nBACKGROUND (recon confirmed 2026-06-14):\n- No Observable protocol or Trajectory eqx.Module exists.\n- TrajectoryWriter (simulate.py:148) and TrajectoryReader (visualization/trajectory.py:27) are the\n  existing I/O layer -- distinct from the API Trajectory; do NOT modify them.\n- Track C (261) creates prolix/api/__init__.py in parallel. This track may need to create or\n  extend api/__init__.py. See MERGE STRATEGY below.\n\nSTEPS:\n\n1. RECON:\n   Check if src/prolix/api/__init__.py exists (Track C may have run first in a parallel worktree).\n   Read src/prolix/simulate.py near line 148 to understand existing state types.\n   Grep for "SimulationState" and "LangevinState" in src/ to understand step output shape.\n\n2. Create src/prolix/api/observables.py:\n\n   \`\`\`python\n   from typing import Protocol, runtime_checkable\n   import jax.numpy as jnp\n   import equinox as eqx\n   from jaxtyping import Array, Float\n\n   @runtime_checkable\n   class Observable(Protocol):\n       def compute(self, state) -> Array: ...\n\n   class Trajectory(eqx.Module):\n       positions: Float[Array, "steps atoms 3"]\n       observable_values: dict  # name -> Array\n       n_steps: int = eqx.field(static=True)\n\n   class Temperature(eqx.Module):\n       dof: int = eqx.field(static=True)\n       def compute(self, state) -> Array:\n           # Placeholder until state type is fixed in Sprint 38\n           return jnp.nan\n   \`\`\`\n\n3. MERGE STRATEGY for api/__init__.py:\n   - If the file does NOT exist: create it with Observable, Trajectory, Temperature exports.\n   - If the file ALREADY exists (Track C ran first): edit it to add the new imports.\n   Either way: api/__init__.py must export Observable, Trajectory, Temperature after this track.\n\n4. Create tests/api/__init__.py (if not exists) and tests/api/test_observables.py:\n   \`\`\`python\n   def test_observable_is_runtime_checkable():\n       from prolix.api.observables import Temperature, Observable\n       t = Temperature(dof=3)\n       assert isinstance(t, Observable)\n\n   def test_trajectory_is_equinox_module():\n       import equinox as eqx\n       from prolix.api.observables import Trajectory\n       assert issubclass(Trajectory, eqx.Module)\n\n   def test_imports():\n       from prolix.api import Observable, Trajectory, Temperature\n   \`\`\`\n\n5. Run: uv run pytest tests/api/test_observables.py -v -- must pass.\n\n6. Commit:\n   git add src/prolix/api/observables.py tests/api/test_observables.py src/prolix/api/__init__.py\n   git commit -m "feat(api): Observable protocol + Trajectory eqx.Module (#284)"\n\nGATE: \`uv run pytest tests/api/test_observables.py -v\` PASS; \`from prolix.api import Observable, Trajectory\` works.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Observable + Trajectory:\n1. src/prolix/api/observables.py exists with Observable (Protocol), Trajectory (eqx.Module), Temperature.\n2. Observable is @runtime_checkable.\n3. Trajectory has positions and observable_values fields.\n4. Temperature conforms to Observable: isinstance(Temperature(dof=3), Observable) is True.\n5. tests/api/test_observables.py exists with >=3 tests.\n6. Run: \`uv run pytest tests/api/test_observables.py -v\` -- all PASS.\n7. \`from prolix.api import Observable, Trajectory\` succeeds (no ImportError).\nPASS if all 7 checks pass.\n`,
    "worktree"
  );

// ===== TRACK F — Track F -- DR-claim1-1: Hetero-batched MD precedent scan (#255) =========================
const trackF = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260614_sprint37_paper_preground\nProject root: /home/marielle/projects/prolix\n\nROLE: librarian -- deep research synthesis via WebSearch + WebFetch + knowledge base\n\nOBJECTIVE: Survey existing MD tools for heterogeneous-batched simulation capability and\nproduce a structured comparison table. Feeds paper Related Work and unblocks RR1 rebuttal (#257).\n\nRESEARCH QUESTION:\nWhat heterogeneous-batched MD capability exists in: kUPS, jax-md, OpenMM swarm/multi-sim,\nGROMACS -multidir, Folding@home, and any other relevant tool (last 3 years)?\nFor each: (a) varied-topology batch in one JIT? (b) declarative plan-then-execute API?\n(c) AOT export / WASM? (d) varied system sizes in one batch?\n\nKEY AXIS: prolix batches varied-topology systems in a single JIT with automatic tiling.\nWhere does each competitor fall short on this specific axis?\n\nSTEPS:\n1. Search for papers/docs on: kUPS, jax-md (JAX MD), OpenMM swarm API, GROMACS -multidir,\n   Folding@home heterogeneous simulation (last 3 years, 2023-2025).\n2. For each tool answer the 4 questions above with evidence.\n3. Write synthesis to .praxia/docs/research/260614_dr-claim1-1-precedent-scan.md\n\n   FORMAT:\n   ## Summary (2 paragraphs)\n   ## Comparison Table\n   | Tool | Varied-topology batch | Declarative API | AOT/WASM | Size handling |\n   ## RR1 Rebuttal Positioning (1 paragraph: how is prolix differentiated?)\n   ## Sources (bulleted with URLs)\n\n4. Create .praxia/research/ directory if it does not exist.\n   Append entry to .praxia/research/synthesis.jsonl:\n   {"date": "2026-06-14", "task_id": "260614_sprint37_paper_preground", "item_id": 255,\n    "query": "Hetero-batched MD precedent", "confidence": "high|medium|low",\n    "artifact": ".praxia/docs/research/260614_dr-claim1-1-precedent-scan.md",\n    "key_finding": "<one-sentence summary>"}\n\n5. Commit:\n   git add .praxia/docs/research/260614_dr-claim1-1-precedent-scan.md .praxia/research/synthesis.jsonl\n   git commit -m "research(dr): hetero-batched MD precedent scan for Related Work (#255)"\n\nGATE: Synthesis document exists; comparison table covers >=4 tools; synthesis.jsonl entry appended.\n`,
    { agentType: "librarian", label: "research:255", phase: "Track F -- DR-claim1-1: Hetero-batched MD precedent scan (#255)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK G — Track G -- DR-paper-1: Venue fit scan -- JCP/JCTC/JCIM/JOSS/ICML/ICLR/MLSys (#304) =========================
const trackG = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260614_sprint37_paper_preground\nProject root: /home/marielle/projects/prolix\n\nROLE: librarian -- deep research synthesis\n\nOBJECTIVE: Determine the best submission venue for the prolix engine paper.\nResolves OQ6 and informs how to frame the contribution.\n\nRESEARCH QUESTION:\nGiven prolix is a JAX-native MD engine with: (1) heterogeneous-system batching,\n(2) JAX AOT/WASM portability, (3) a differentiable force-field fitting demo (section 7.1) --\nwhich of JCP, JCTC, JCIM, JOSS, ICML, ICLR, MLSys, NeurIPS is the best fit?\nWhat are submission requirements, page limits, and prior MD/software papers at each venue?\n\nSTEPS:\n1. Search scope statements, author guidelines, and recent analogous papers at:\n   JCP, JCTC, JCIM, JOSS, ICML 2024-2025, ICLR 2024-2025, MLSys 2024-2025\n   Find >=1 recent paper (2023-2025) per venue that is analogous in scope.\n\n2. Write synthesis to .praxia/docs/research/260614_dr-paper1-venue-fit.md\n\n   FORMAT:\n   ## Recommendation (1 paragraph -- name top 2 venues with rationale)\n   ## Venue Analysis Table\n   | Venue | Scope fit (1-5) | Page limit | Example analogous paper | Submission risk |\n   ## Framing implications (how the narrative changes per top-2 venue choice)\n   ## Sources\n\n3. Append to .praxia/research/synthesis.jsonl and commit.\n   Message: "research(dr): venue fit scan for prolix engine paper (#304)"\n\nGATE: Document exists; recommendation names top 2 venues; table covers >=5 venues.\n`,
    { agentType: "librarian", label: "research:304", phase: "Track G -- DR-paper-1: Venue fit scan -- JCP/JCTC/JCIM/JOSS/ICML/ICLR/MLSys (#304)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK H — Track H -- DR-paper-2: Recent MD-engine papers -- narrative tropes and reviewer criticisms (#305) =========================
const trackH = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260614_sprint37_paper_preground\nProject root: /home/marielle/projects/prolix\n\nROLE: librarian -- deep research synthesis\n\nOBJECTIVE: Survey recent MD engine papers (2024-2025) to extract narrative tropes and\nreviewer criticism patterns. Directly strengthens the RR1-RR7 rebuttal evidence table.\n\nRESEARCH QUESTION:\nWhat MD-engine or MD-software papers appeared at JCP/JCTC/NeurIPS/ICML/ICLR/MLSys/arXiv\nin 2024-2025? For each: core claim, benchmarks used, reviewer criticisms (from OpenReview for\nML venues)? How did authors address them?\n\nTARGET SCOPE: Prefer "MD engine + JAX/PyTorch/compilation" or "differentiable MD."\nAt minimum: jax-md follow-ups, OpenMM-ML integrations, torchmd-net, LAMMPS-pytorch,\nanything at ML venues (ICML/ICLR 2024-2025) mentioning molecular dynamics.\n\nSTEPS:\n1. Search arXiv, OpenReview, venue proceedings for MD engine papers 2024-2025.\n   Aim for 6-10 papers. For each: title, venue, core claim, benchmarks, reviewer criticism.\n\n2. Write synthesis to .praxia/docs/research/260614_dr-paper2-md-engine-survey.md\n\n   FORMAT:\n   ## Key narrative tropes (list, >=5 items observed across papers)\n   ## Benchmark standard set (what metrics/baselines are expected by reviewers)\n   ## Reviewer criticism patterns (list, >=5 recurring objections)\n   ## RR1-RR7 augmentation (one line per RR item: how this survey evidence helps address it)\n   ## Paper summaries\n   | Title | Venue | Core claim | Key benchmark | Main criticism |\n   ## Sources\n\n3. Append to synthesis.jsonl and commit.\n   Message: "research(dr): recent MD-engine paper survey for rebuttal table (#305)"\n\nGATE: Document exists; narrative tropes list >=5 items; paper summary table >=5 papers.\n`,
    { agentType: "librarian", label: "research:305", phase: "Track H -- DR-paper-2: Recent MD-engine papers -- narrative tropes and reviewer criticisms (#305)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK I — Track I -- xtrax.tiling integration spec: adopt xtrax BatchPlanner as prolix planner backend (#1842) =========================
const trackI = () =>
  track(
    "1842",
    "Track I -- xtrax.tiling integration spec: adopt xtrax BatchPlanner as prolix planner backend (#1842)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint37_paper_preground\nProject root: /home/marielle/projects/prolix\nxtrax root: /home/marielle/projects/xtrax\n\nROLE: specification-specialist + fixer (spec first, then minimal wiring)\n\nOBJECTIVE: Decide and implement the xtrax.tiling integration strategy for prolix.\nOther projects (aminx, plegadx, denxity) are already adopting xtrax.tiling. prolix should\nstandardize on the same planner backend so EnsemblePlan.run() (Sprint 38) builds on it.\n\nBACKGROUND (recon confirmed 2026-06-14):\n- xtrax.tiling (src/xtrax/tiling/plan.py:99): BatchPlanner frozen dataclass with plan() method\n  returning BatchPlan + AxisDecision. Strategies: Vmap, SafeMap, Scan, DedupGather, Bucket.\n  AxisSpec (plan.py:20): axis metadata (cardinality, batch_size, heterogeneous, bucket_boundaries).\n- prolix BatchPlanner: src/prolix/tiling/planner.py:91 -- compare with xtrax's to find gaps.\n- No prolix-xtrax dependency currently exists in pyproject.toml.\n\nSTEPS:\n\n1. RECON (read both planners before writing anything):\n   Read /home/marielle/projects/xtrax/src/xtrax/tiling/plan.py (full file).\n   Read /home/marielle/projects/prolix/src/prolix/tiling/planner.py (full file).\n   Identify: what does prolix's BatchPlanner do that xtrax's doesn't? And vice versa?\n   Also read pyproject.toml to understand current deps and how to add a local path dep.\n\n2. Write integration spec at .praxia/docs/superpowers/specs/260614_xtrax-tiling-integration.md:\n   REQUIRED SECTIONS:\n   a) Gap analysis: prolix-specific vs xtrax-generic features in BatchPlanner\n   b) Integration strategy: one of --\n      - Full replacement: remove prolix.tiling.planner, import from xtrax.tiling\n      - Wrapper: thin prolix adapter around xtrax.BatchPlanner (preserves physics-specific logic)\n      - Fork: copy xtrax types into prolix (no runtime dep -- avoids version coupling)\n   c) Dependency decision: \`uv add --path ../xtrax xtrax\` vs git URL vs PyPI\n   d) Migration plan: list every prolix file that imports from prolix.tiling.planner\n   e) Backward compatibility: do existing prolix tests need changes?\n\n3. Based on the spec (section b), implement the MINIMAL viable integration:\n   - If "Full replacement": add xtrax dep to pyproject.toml (path dep); update prolix/__init__.py\n     if it re-exports BatchPlanner; grep for callers and update imports; verify tests pass.\n   - If "Wrapper": create src/prolix/tiling/xtrax_adapter.py with a thin wrapper; update callers.\n   - If "Fork": copy the relevant types (AxisSpec, BatchPlan, AxisDecision) into prolix.tiling\n     with a comment crediting xtrax; no dep needed.\n\n4. Run: uv run pytest -m "not slow" -q -- must pass after integration.\n   If tests fail, revert the import changes but keep the spec document (spec is the primary output).\n\n5. Commit spec + integration (or spec only if tests fail):\n   git add .praxia/docs/superpowers/specs/260614_xtrax-tiling-integration.md [+ changed files]\n   git commit -m "spec+refactor(xtrax): tiling integration spec and minimal wiring (#1842)"\n\nGATE (hard): Integration spec exists with all 5 sections and a clear strategy decision.\nGATE (soft): \`uv run pytest -m "not slow" -q\` passes after the integration step.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify xtrax.tiling integration:\n1. .praxia/docs/superpowers/specs/260614_xtrax-tiling-integration.md exists.\n2. Spec has all 5 sections: gap analysis, strategy, dependency decision, migration plan, compat.\n3. Strategy decision is stated clearly (Full replacement / Wrapper / Fork) with rationale.\n4. Dependency decision is actionable (pyproject.toml change or explicit "no dep" with reason).\n5. If integration implemented: \`uv run pytest -m "not slow" -q\` passes (no new failures).\n6. Committed.\nPASS if checks 1-4+6 pass (check 5 is soft -- spec-only is acceptable if tests regress).\n`,
    "worktree"
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("Paper Pre-Work — Sub-specs, Research Sweep, API Skeleton: writing chain (, sequential) || research (A, B, C, D, E, F, G, H, I, read-only)");
const [writing, resA, resB, resC, resD, resE, resF, resG, resH, resI] = await Promise.all([
  (async () => {
    return {  };
  })(),
  trackA(),
  trackB(),
  trackC(),
  trackD(),
  trackE(),
  trackF(),
  trackG(),
  trackH(),
  trackI(),
]);

// Phase-5: Integrate worktree branches
{
  phase("Integrate");
  const intManifestPath = `.praxia/worktree_manifests/${TASK_ID}.json`;
  // Step 1: write manifest via CLI
  const _intCli = await agent(
    `Run shell command: praxia worktree integrate --sprint-id ${TASK_ID} and report the manifest path written.`,
    { label: "worktree:integrate-cli" }
  );
  // Step 2: integrator analysis
  const intReport = await agent(
    `task_id: ${TASK_ID}. Analyze the worktree integration manifest and report any merge conflicts.`,
    { label: "integrator", phase: "Integrate" }
  );
  // Step 3: merge executor if conflicts
  if (intReport && typeof intReport === "string" && intReport.includes("conflict")) {
    await agent(
      `Resolve merge conflicts: read the manifest at ${intManifestPath}, then run git merge for each branch listed. Use git merge --no-ff for each branch in dependency order. Report merged SHAs.`,
      { label: "fixer:merge", phase: "Integrate" }
    );
  }
}

return {
  task_id: TASK_ID,
  sprint_id: 1,
  verdicts: {

  },
  research_328: resA,
  research_327: resB,
  research_261: resC,
  research_281: resD,
  research_284: resE,
  research_255: resF,
  research_304: resG,
  research_305: resH,
  research_1842: resI
};
