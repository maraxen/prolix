// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/38.toml
// Regenerate: praxia dw emit-sprint 38.toml
// task_id: 38   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain () runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (A,B,C,D,E,F,G,H) run concurrently.

export const meta = {
  name: "38",
  description: "API Implementation: EnsemblePlan, Bundle factories, Observables, xtrax wiring",
  phases: [
    { title: "Track A -- xtrax.tiling wiring: import cleanup in prolix planner (#1842)" },
    { title: "Track B -- Bundle.from_pdb factory implementation (#281)" },
    { title: "Track C -- EnsemblePlan.run() implementation (#261)" },
    { title: "Track D -- Observable protocol + Trajectory + Temperature implementation (#284)" },
    { title: "Track E -- Bundle.from_system_dict migration factory (#282)" },
    { title: "Track F -- S1 D1: per-term ANALYTICAL vs AUTOGRAD force agreement (#292)" },
    { title: "Track G -- S1 D2: jax.grad through trajectory finite-diff parity (#293)" },
    { title: "Track H -- Energy observable: prolix.api.observables.Energy (#254)" },
  ],
};

const TASK_ID = "38";
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

// Shared context for the writing tracks (from recon, task 38).
const EMITTER_CTX = `task_id: 260614_sprint38_api_impl\nProject root: /home/marielle/projects/prolix\n\nKEY FACTS (verified against codebase before emit):\n- xtrax is ALREADY a dependency: pyproject.toml line ~23: 'xtrax @ file:///home/marielle/projects/xtrax'\n  xtrax.tiling.plan has: BatchPlanner (frozen dataclass, strategies=[Vmap|SafeMap|Scan|DedupGather|Bucket])\n  AxisSpec at xtrax/src/xtrax/tiling/plan.py:20; BatchPlan, BatchPlanner at :99\n- prolix BatchPlanner: src/prolix/tiling/planner.py:91 -- greedy budget-driven demotion, plan()->BatchPlan\n  AxisSpec:44, AxisDecision:56, BatchPlan:64 -- prolix owns these types currently\n- MolecularBundle: src/prolix/types/bundles.py:66 (frozen dataclass, eqx.Module)\n  from_pdb stub: :198-249 -- error-path validation done; raises NotImplementedError at :247\n  from_system_dict: DOES NOT EXIST -- must be added as new classmethod\n  make_bundle_from_system factory: src/prolix/physics/system.py:419 (converts PhysicsSystem -> MolecularBundle)\n- EnsemblePlan: src/prolix/api/ensemble_plan.py:12 (77 lines)\n  __init__:30 (takes bundles: list, planner=None)\n  run():45 -- raises NotImplementedError at :74; must be implemented this sprint\n- Observable: src/prolix/api/observables.py:14 -- @runtime_checkable Protocol, compute(state)->Array\n  Trajectory: :44 -- eqx.Module, positions/observable_values/n_steps\n  Temperature: :61 -- compute() returns jnp.nan placeholder (needs real impl)\n  Energy: DOES NOT EXIST -- must be added\n- prolix.api.__init__.py exports: EnsemblePlan, Observable, Trajectory, Temperature\n- Test suite: uv run pytest -m 'not slow' for fast CI; full: uv run pytest\n  Results written to tmp/pytest.json\n- Sprint 37 xtrax spec decision: v1.0 = type imports only, greedy loop stays; v1.1 = behavioral delegation\n  Spec at: .praxia/docs/superpowers/specs/260614_xtrax-tiling-integration.md\n`;

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

// ===== TRACK A — Track A -- xtrax.tiling wiring: import cleanup in prolix planner (#1842) =========================
const trackA = () =>
  track(
    "1842",
    "Track A -- xtrax.tiling wiring: import cleanup in prolix planner (#1842)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nBacklog: #1842 -- Integrate xtrax.tiling into prolix (v1.0: type imports only)\n\nDECISION (from .praxia/docs/superpowers/specs/260614_xtrax-tiling-integration.md):\n  v1.0 = keep prolix's greedy loop; add xtrax as type-import only\n  v1.1 = behavioral delegation to xtrax.BatchPlanner (deferred)\n\nCURRENT STATE:\n  - xtrax already in pyproject.toml: 'xtrax @ file:///home/marielle/projects/xtrax'\n  - src/prolix/tiling/planner.py:91 -- prolix BatchPlanner with greedy plan()\n  - planner.py:112 -- comment mentions xtrax but no actual import\n\nWHAT TO DO:\n1. Read src/prolix/tiling/planner.py in full.\n2. Read src/xtrax/tiling/plan.py (at ~/projects/xtrax) to understand xtrax interface.\n   Key types: XtraxBatchPlanner (line ~99), XtraxAxisSpec (:20), XtraxBatchPlan, Strategy enum.\n3. Add TYPE_CHECKING import block to planner.py:\n   \`\`\`python\n   from __future__ import annotations\n   from typing import TYPE_CHECKING\n   if TYPE_CHECKING:\n       from xtrax.tiling.plan import BatchPlanner as XtraxBatchPlanner\n       from xtrax.tiling.plan import AxisSpec as XtraxAxisSpec\n   \`\`\`\n4. Add a module-level docstring note that xtrax.tiling is available as a future\n   backend (cite #1842 and the spec).\n5. Verify \`from xtrax.tiling.plan import BatchPlanner\` is importable at runtime\n   (not just TYPE_CHECKING) by adding a try/import in a __init__.py or running\n   \`uv run python -c 'from xtrax.tiling.plan import BatchPlanner; print(BatchPlanner)'\`.\n6. Run: uv run pytest src/prolix/tiling/ -x --tb=short\n   Also run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -20\n\nSUCCESS GATE:\n  - \`from xtrax.tiling.plan import BatchPlanner\` succeeds in prolix env\n  - All existing planner tests pass\n  - No behavioral change to prolix.tiling.BatchPlanner.plan()\n  - planner.py has type-import block referencing xtrax types\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nReviewing Track A: xtrax.tiling wiring (#1842)\n\nVERIFY:\n1. Run: uv run python -c 'from xtrax.tiling.plan import BatchPlanner; print("ok")'\n   PASS if prints "ok". FAIL if ImportError.\n2. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n   PASS if no failures. FAIL if any test fails.\n3. Read src/prolix/tiling/planner.py -- check that plan() body is UNCHANGED (greedy loop preserved).\n   FAIL if greedy loop was replaced with xtrax.BatchPlanner.plan() delegation.\n4. Check planner.py has TYPE_CHECKING block with xtrax imports.\n   FAIL if missing.\n\nPASS if all 4 checks pass. FAIL otherwise.\n`,
    "worktree"
  );

// ===== TRACK B — Track B -- Bundle.from_pdb factory implementation (#281) =========================
const trackB = () =>
  track(
    "281",
    "Track B -- Bundle.from_pdb factory implementation (#281)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nBacklog: #281 -- MolecularBundle.from_pdb(path, forcefield='amber14')\n\nCURRENT STATE:\n  - src/prolix/types/bundles.py:198-249: stub exists, error-path validation done\n    (FileNotFoundError, ValueError for unsupported FF, NotImplementedError for missing parmed)\n    Body raises NotImplementedError at line 247.\n  - src/prolix/physics/system.py:419: make_bundle_from_system(system, boundary_condition='periodic')\n    Takes an object with .positions, .bonds, .bond_params, .angles, .angle_params, etc.\n    Returns a padded MolecularBundle.\n\nWHAT TO DO:\n1. Read bundles.py:66-196 to understand MolecularBundle field names and types.\n2. Read physics/system.py:419-500 to understand PhysicsSystem fields and how\n   make_bundle_from_system consumes them.\n3. Read parmed docs via: uv run python -c 'import parmed; help(parmed.load_file)' or\n   inspect parmed.Structure attributes.\n4. Implement the from_pdb body (replacing the NotImplementedError at :247):\n   \`\`\`python\n   import parmed\n   struct = parmed.load_file(path)\n   # Apply AMBER14 parameters\n   if forcefield == 'amber14':\n       from parmed.amber import AmberParameterSet\n       # or use parmed's built-in AMBER param application\n       pass\n   # Build a PhysicsSystem-like namespace or direct arrays from struct\n   # ... extract positions (Angstrom), bonds, angles, dihedrals, masses ...\n   # Delegate to make_bundle_from_system\n   from prolix.physics.system import make_bundle_from_system\n   system = _parmed_struct_to_system(struct)\n   return make_bundle_from_system(system)\n   \`\`\`\n5. Add a helper _parmed_struct_to_system(struct) -> SimpleNamespace that maps\n   parmed.Structure fields to the PhysicsSystem protocol make_bundle_from_system expects.\n   Key fields: positions (Angstrom -> keep as-is, AKMA uses Angstrom),\n   masses (in atomic mass units), bonds (index pairs), bond_params (k, r0 in kcal/mol/A^2),\n   angles, angle_params, dihedrals, dihedral_params.\n6. Write tests/types/test_bundle_from_pdb.py:\n   - test_from_pdb_file_not_found: assert FileNotFoundError on missing path\n   - test_from_pdb_bad_forcefield: assert ValueError on unsupported FF\n   - test_from_pdb_no_parmed: mock parmed import failure -> NotImplementedError\n   - test_from_pdb_roundtrip: if parmed available, load a small test PDB (use existing\n     test fixtures or create a minimal 3-atom PDB) and check bundle.positions.shape[0] > 0\n7. Run: uv run pytest tests/types/test_bundle_from_pdb.py -x --tb=short\n\nNOTE: If parmed is not in the dev dependencies, add it:\n  uv add --optional parmed  (or uv add parmed if it's a hard dep)\n  The stub already has try/except ImportError -> NotImplementedError, so\n  tests that don't need parmed can be run without it.\n\nSUCCESS GATE:\n  - Error-path tests all pass without parmed installed\n  - If parmed is available: roundtrip test produces a valid MolecularBundle\n  - uv run pytest -m 'not slow' passes\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nReviewing Track B: Bundle.from_pdb (#281)\n\nVERIFY:\n1. Read src/prolix/types/bundles.py:198-260 -- confirm from_pdb body is implemented\n   (no longer raises NotImplementedError unconditionally at line 247).\n   FAIL if still raises NotImplementedError without parmed check.\n2. Run: uv run pytest tests/types/test_bundle_from_pdb.py -x --tb=short\n   PASS if all tests pass. FAIL if missing or failing.\n3. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n   PASS if no regressions. FAIL if new failures.\n4. Confirm error-path tests exist: test_from_pdb_file_not_found,\n   test_from_pdb_bad_forcefield, test_from_pdb_no_parmed.\n   FAIL if any of these are missing.\n\nPASS if all 4 checks pass. FAIL otherwise.\n`,
    "worktree"
  );

// ===== TRACK C — Track C -- EnsemblePlan.run() implementation (#261) =========================
const trackC = () =>
  track(
    "261",
    "Track C -- EnsemblePlan.run() implementation (#261)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nBacklog: #261 -- EnsemblePlan.run integration with BatchPlanner\n\nCURRENT STATE:\n  - src/prolix/api/ensemble_plan.py:12-77 (77 lines):\n    __init__:30 accepts bundles: list, planner=None (stores batch_plan if planner given)\n    run():45 raises NotImplementedError at :74\n  - prolix.tiling.BatchPlanner at src/prolix/tiling/planner.py:91\n    BatchPlanner.plan() -> BatchPlan with decisions per AxisSpec\n  - xtrax is available (in pyproject.toml); xtrax.tiling.plan.BatchPlanner has .plan()\n    Per Sprint 37 decision: v1.0 uses prolix BatchPlanner (greedy), xtrax in v1.1\n  - settle_langevin is the production integrator: src/prolix/physics/settle.py\n    settle_langevin(energy_fn, shift_fn, dt, kT, gamma, mass, water_indices, ...) -> (init_fn, apply_fn)\n  - LangevinState: src/prolix/batched_simulate.py\n\nWHAT TO DO:\n1. Read ensemble_plan.py in full. Read prolix/tiling/planner.py:64-160 (BatchPlan + BatchPlanner).\n2. Read prolix/physics/settle.py:1-50 for settle_langevin signature.\n3. Implement EnsemblePlan.run(n_steps, dt, kT, seed=0) to:\n   a. If self.batch_plan is None, call BatchPlanner.plan() using a default set of AxisSpecs\n      based on self.bundles (e.g., one AxisSpec per bundle axis with heterogeneous=True if shapes differ).\n   b. Based on batch_plan.decisions: if decision.batch_size == 0, use jax.vmap;\n      if > 0, use prolix safe_map or jax.lax.map with tile size.\n   c. Run a simplified Langevin NVT integrator on each bundle.\n      For v1.0, it's acceptable to loop over bundles (sequential safe_map),\n      with a clear TODO comment noting that the batched vmap path comes in v1.1\n      once xtrax Bucket strategy is adopted.\n   d. Return a Trajectory object (from prolix.api.observables):\n      positions=(n_steps, n_atoms, 3), observable_values={}, n_steps=n_steps.\n4. Write tests/api/test_ensemble_plan.py:\n   - test_ensemble_plan_init_no_planner: EnsemblePlan([bundle]).batch_plan is None\n   - test_ensemble_plan_init_with_planner: planner.plan() called; batch_plan set\n   - test_ensemble_plan_run_returns_trajectory: run(n_steps=10, dt=0.5, kT=0.596)\n     returns Trajectory with positions.shape == (10, n_atoms, 3)\n   - test_ensemble_plan_run_single_bundle_parity: EnsemblePlan([bundle]).run(n_steps=5)\n     produces positions close to direct settle_langevin run (V1 parity)\n5. Run: uv run pytest tests/api/test_ensemble_plan.py -x --tb=short\n\nNOTE: For a minimal MolecularBundle to test with, use an existing test fixture or\ncreate a water molecule (3 atoms) with appropriate bond topology and masses.\n\nSUCCESS GATE:\n  - run() no longer raises NotImplementedError\n  - Returns Trajectory object\n  - test_ensemble_plan_run_returns_trajectory passes\n  - uv run pytest -m 'not slow' passes\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nReviewing Track C: EnsemblePlan.run() (#261)\n\nVERIFY:\n1. Read src/prolix/api/ensemble_plan.py -- run() must NOT raise NotImplementedError unconditionally.\n   FAIL if line 74 still raises NotImplementedError.\n2. Run: uv run pytest tests/api/test_ensemble_plan.py -x --tb=short\n   PASS if all tests pass. FAIL if missing or failing.\n3. Confirm test_ensemble_plan_run_returns_trajectory exists and passes:\n   checks return type is Trajectory and positions.shape[0] == n_steps.\n4. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n   PASS if no regressions.\n\nPASS if all 4 checks pass. FAIL otherwise.\n`,
    "worktree"
  );

// ===== TRACK D — Track D -- Observable protocol + Trajectory + Temperature implementation (#284) =========================
const trackD = () =>
  track(
    "284",
    "Track D -- Observable protocol + Trajectory + Temperature implementation (#284)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nBacklog: #284 -- Observable protocol + Trajectory eqx.Module\n\nCURRENT STATE (mostly done by Sprint 37):\n  - src/prolix/api/observables.py:14 -- Observable Protocol (@runtime_checkable), compute(state)->Array\n  - :44 -- Trajectory eqx.Module with positions, observable_values, n_steps\n  - :61 -- Temperature eqx.Module -- compute() returns jnp.nan (PLACEHOLDER, needs real impl)\n  - prolix.api.__init__.py already exports: Observable, Trajectory, Temperature\n\nWHAT TO DO:\n1. Read observables.py in full (87 lines).\n2. Implement Temperature.compute(self, state) properly:\n   - state is likely LangevinState: has .momentum (N, 3) and .mass (N,) fields\n   - T = (2 * KE) / (dof * k_B) where KE = sum(p^2 / (2*m))\n   - k_B in AKMA units = 0.00198720425 kcal/(mol*K)\n   - dof: use self.dof (passed at construction)\n   - Check what fields LangevinState has: read src/prolix/batched_simulate.py top\n   - Replace the \`return jnp.nan\` placeholder at line 87\n3. Write tests/api/test_observables.py:\n   - test_observable_protocol_compliance: Temperature() satisfies isinstance(t, Observable)\n   - test_temperature_compute_equipartition: create synthetic state with known KE,\n     verify Temperature.compute() returns expected T\n   - test_trajectory_fields: Trajectory has .positions, .observable_values, .n_steps\n   - test_temperature_returns_scalar: Temperature.compute() returns scalar Array, not nan\n4. Verify prolix.api imports: \`from prolix.api import Observable, Trajectory, Temperature\`\n5. Run: uv run pytest tests/api/test_observables.py -x --tb=short\n6. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n\nSUCCESS GATE:\n  - Temperature.compute() no longer returns jnp.nan\n  - isinstance(Temperature(dof=9), Observable) is True\n  - All 4 tests in test_observables.py pass\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nReviewing Track D: Observable + Trajectory + Temperature (#284)\n\nVERIFY:\n1. Run: uv run python -c 'from prolix.api import Observable, Trajectory, Temperature; print("ok")'\n   PASS if prints "ok". FAIL if ImportError.\n2. Run: uv run python -c '\nfrom prolix.api import Observable, Temperature\nt = Temperature(dof=9)\nprint(isinstance(t, Observable))\n'\n   PASS if prints True. FAIL otherwise.\n3. Read observables.py:86-87 -- confirm Temperature.compute() no longer returns jnp.nan.\n   FAIL if still returns jnp.nan.\n4. Run: uv run pytest tests/api/test_observables.py -x --tb=short\n   PASS if all tests pass. FAIL if missing or failing.\n5. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n   PASS if no regressions.\n\nPASS if all 5 checks pass. FAIL otherwise.\n`,
    "worktree"
  );

// ===== TRACK E — Track E -- Bundle.from_system_dict migration factory (#282) =========================
const trackE = () =>
  track(
    "282",
    "Track E -- Bundle.from_system_dict migration factory (#282)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nBacklog: #282 -- MolecularBundle.from_system_dict(legacy_dict) migration factory\n\nCURRENT STATE:\n  - src/prolix/types/bundles.py:198 -- from_pdb classmethod exists (stub/impl from Track B)\n  - from_system_dict does NOT exist yet on MolecularBundle\n  - make_bundle_from_system at src/prolix/physics/system.py:419 converts a PhysicsSystem-like\n    object to MolecularBundle -- this is what legacy entry-points produce\n  - Legacy entry-points produce dicts like: {'positions': ..., 'masses': ..., 'bonds': ...,\n    'bond_params': ..., 'angles': ..., 'angle_params': ..., 'boundary_condition': ...}\n\nWHAT TO DO:\n1. Read bundles.py:66-200 to understand MolecularBundle fields.\n2. Read physics/system.py:419-490 to understand what make_bundle_from_system expects\n   (it uses getattr, so a SimpleNamespace works).\n3. Add classmethod from_system_dict to MolecularBundle after from_pdb:\n   \`\`\`python\n   @classmethod\n   def from_system_dict(\n       cls,\n       d: dict,\n       boundary_condition: str = "periodic",\n   ) -> "MolecularBundle":\n       '''Build MolecularBundle from a legacy system dict.\n\n       Emits DeprecationWarning. Use Bundle.from_pdb for new code.\n\n       Args:\n           d: Dict with keys: positions, masses, bonds, bond_params,\n              angles, angle_params (optional: dihedrals, dihedral_params,\n              water_indices, box).\n           boundary_condition: "periodic" or "free".\n       '''\n       import warnings\n       from types import SimpleNamespace\n       from prolix.physics.system import make_bundle_from_system\n       warnings.warn(\n           'MolecularBundle.from_system_dict is deprecated. '\n           'Use MolecularBundle.from_pdb for new code.',\n           DeprecationWarning, stacklevel=2,\n       )\n       ns = SimpleNamespace(**d)\n       return make_bundle_from_system(ns, boundary_condition=boundary_condition)\n   \`\`\`\n4. Write tests/types/test_bundle_from_system_dict.py:\n   - test_from_system_dict_warns: assert DeprecationWarning emitted\n   - test_from_system_dict_roundtrip: pass a minimal dict (3-atom water toy),\n     verify returned MolecularBundle has correct positions shape\n   - test_from_system_dict_invalid_key: graceful error for missing required key\n5. Run: uv run pytest tests/types/test_bundle_from_system_dict.py -x --tb=short\n6. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n\nSUCCESS GATE:\n  - MolecularBundle.from_system_dict() exists and emits DeprecationWarning\n  - Roundtrip test passes (dict -> MolecularBundle)\n  - uv run pytest -m 'not slow' passes\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nReviewing Track E: Bundle.from_system_dict (#282)\n\nVERIFY:\n1. Run: uv run python -c '\nimport warnings\nfrom prolix.types.bundles import MolecularBundle\nprint(hasattr(MolecularBundle, "from_system_dict"))\n'\n   PASS if prints True. FAIL otherwise.\n2. Run: uv run python -c '\nimport warnings\nfrom prolix.types.bundles import MolecularBundle\nwith warnings.catch_warnings(record=True) as w:\n    warnings.simplefilter("always")\n    try:\n        MolecularBundle.from_system_dict({})\n    except Exception:\n        pass\n    print(any(issubclass(x.category, DeprecationWarning) for x in w))\n'\n   PASS if prints True. FAIL otherwise.\n3. Run: uv run pytest tests/types/test_bundle_from_system_dict.py -x --tb=short\n   PASS if all tests pass. FAIL if missing or failing.\n4. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n   PASS if no regressions.\n\nPASS if all 4 checks pass. FAIL otherwise.\n`,
    "worktree"
  );

// ===== TRACK F — Track F -- S1 D1: per-term ANALYTICAL vs AUTOGRAD force agreement (#292) =========================
const trackF = () =>
  track(
    "292",
    "Track F -- S1 D1: per-term ANALYTICAL vs AUTOGRAD force agreement (#292)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nBacklog: #292 -- S1 D1: per-term ANALYTICAL vs AUTOGRAD force agreement < 1e-6 kcal/mol/A\n\nCONTEXT:\n  S1 is the differentiability claim. D1 validates that per-term bonded force analytical\n  implementations agree with jax.grad-computed autograd forces. This is paper-critical:\n  it's the evidence that jax.grad through prolix produces physically correct gradients.\n\nRELEVANT FILES:\n  - src/prolix/physics/analytical_forces.py -- per-term analytical force implementations\n  - src/prolix/physics/bonded.py -- bonded energy terms (bonds, angles, dihedrals, impropers)\n  - src/prolix/physics/system.py:419 -- make_bundle_from_system\n  - Existing tests: tests/physics/ -- look for any existing force agreement tests\n\nWHAT TO DO:\n1. Read analytical_forces.py and bonded.py to understand the function signatures.\n   Key functions: harmonic_bond_energy, harmonic_angle_energy, periodic_dihedral_energy,\n   or however they are named. Check both files.\n2. For each bonded term, write a parametric pytest test that:\n   a. Creates synthetic parameters (random k, r0, positions near equilibrium)\n   b. Computes analytical forces via prolix's analytical implementation\n   c. Computes autograd forces via jax.grad of the energy function\n   d. Asserts max |analytical - autograd| < 1e-6 (kcal/mol/A)\n3. Write tests/physics/test_s1_force_parity.py covering at minimum:\n   - test_bond_force_parity: harmonic bond term\n   - test_angle_force_parity: harmonic angle term\n   - test_dihedral_force_parity: periodic dihedral term\n   Each test should use jax.random for reproducible random geometries (seed=42).\n4. Mark tests with @pytest.mark.slow if they are expensive (>1s); otherwise leave unmarked.\n5. Run: uv run pytest tests/physics/test_s1_force_parity.py -x --tb=short -v\n6. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n\nSUCCESS GATE:\n  - test_bond_force_parity, test_angle_force_parity, test_dihedral_force_parity all pass\n  - Each asserts max |f_analytical - f_autograd| < 1e-6\n  - uv run pytest -m 'not slow' passes\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nReviewing Track F: S1 D1 force parity (#292)\n\nVERIFY:\n1. Run: uv run pytest tests/physics/test_s1_force_parity.py -x --tb=short -v\n   PASS if all tests pass. FAIL if file missing or any test fails.\n2. Grep for assertion pattern in the test file:\n   grep -n "1e-6" tests/physics/test_s1_force_parity.py\n   grep -En "atol|allclose|max.*force" tests/physics/test_s1_force_parity.py\n   FAIL if no assertion against 1e-6 tolerance is found.\n3. Confirm tests exist for at least 3 terms (bond, angle, dihedral):\n   grep -n "def test_" tests/physics/test_s1_force_parity.py | wc -l\n   FAIL if fewer than 3 test functions.\n4. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n   PASS if no regressions.\n\nPASS if all 4 checks pass. FAIL otherwise.\n`,
    "worktree"
  );

// ===== TRACK G — Track G -- S1 D2: jax.grad through trajectory finite-diff parity (#293) =========================
const trackG = () =>
  track(
    "293",
    "Track G -- S1 D2: jax.grad through trajectory finite-diff parity (#293)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nBacklog: #293 -- S1 D2: jax.grad / jax.jacrev through EnsemblePlan.run finite-diff parity, RMS < 1e-4\n\nCONTEXT:\n  This validates the differentiability claim end-to-end: jax.grad(loss)(params) through\n  a short MD trajectory must agree with finite-difference gradients to RMS < 1e-4.\n  This is used directly in the §7.1 paper figure (bonded-parameter fitting).\n\nNOTE: EnsemblePlan.run() is being implemented in Track C. This track can proceed\nin parallel assuming run() will exist; write the test structure even if run() is\nstill a stub -- the test should be marked xfail(strict=False) if run() not yet complete,\nthen converted to a real test once Track C lands.\n\nRELEVANT FILES:\n  - src/prolix/api/ensemble_plan.py -- EnsemblePlan.run() (Track C implementing this)\n  - src/prolix/physics/bonded.py -- bonded energy with differentiable params\n  - tests/physics/test_settle_temperature_control.py -- reference for how integration tests work\n\nWHAT TO DO:\n1. Read ensemble_plan.py and bonded.py to understand the parameter structure.\n2. Write tests/api/test_s1_jaxgrad_parity.py:\n   \`\`\`python\n   import pytest\n   import jax\n   import jax.numpy as jnp\n\n   def test_jaxgrad_bond_params_parity():\n       '''S1 D2: jax.grad through 5-step trajectory agrees with finite-diff to RMS < 1e-4.'''\n       # Setup: single water molecule or minimal 2-atom system with one bond\n       # Parameterize bond_k as a differentiable parameter\n       # Define loss = mean squared displacement over n_steps=5 trajectory\n       # Compute jax.grad(loss)(bond_k)\n       # Compute finite-diff: (loss(bond_k + eps) - loss(bond_k - eps)) / (2*eps)\n       # Assert jnp.sqrt(jnp.mean((grad_jax - grad_fd)**2)) < 1e-4\n       ...\n   \`\`\`\n   Key: use a SHORT trajectory (n_steps=5, dt=0.5) and small system (2-4 atoms) to keep\n   this fast and not slow-marked.\n3. If EnsemblePlan.run() is not yet implemented, mark the test:\n   @pytest.mark.xfail(strict=False, reason='EnsemblePlan.run() pending Track C')\n4. If jax.grad doesn't compose through run() as written, try jax.linear_util or\n   use the lower-level settle_langevin directly (differentiate through the apply_fn loop).\n5. Run: uv run pytest tests/api/test_s1_jaxgrad_parity.py -x --tb=short -v\n6. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n\nSUCCESS GATE:\n  - Test file exists with the RMS < 1e-4 assertion\n  - Either passes (if Track C complete) or correctly xfail\n  - uv run pytest -m 'not slow' passes\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nReviewing Track G: S1 D2 jax.grad parity (#293)\n\nVERIFY:\n1. Run: uv run pytest tests/api/test_s1_jaxgrad_parity.py -x --tb=short -v\n   PASS if all tests pass or xfail. FAIL if file missing or ERROR.\n2. Grep for RMS assertion:\n   grep -n "1e-4" tests/api/test_s1_jaxgrad_parity.py\n   grep -En "sqrt.*mean|RMS|rms" tests/api/test_s1_jaxgrad_parity.py\n   FAIL if no assertion against 1e-4 tolerance.\n3. Grep for jax.grad or jax.jacrev usage:\n   grep -En "jax.grad|jax.jacrev|jax.jacobian" tests/api/test_s1_jaxgrad_parity.py\n   FAIL if neither appears.\n4. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n   PASS if no regressions.\n\nPASS if all 4 checks pass. FAIL otherwise.\n`,
    "worktree"
  );

// ===== TRACK H — Track H -- Energy observable: prolix.api.observables.Energy (#254) =========================
const trackH = () =>
  track(
    "254",
    "Track H -- Energy observable: prolix.api.observables.Energy (#254)",
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nBacklog: #254 -- prolix.api.observables.Energy implementation\n\nCURRENT STATE:\n  - src/prolix/api/observables.py: Observable Protocol, Trajectory, Temperature all exist\n  - Temperature.compute() is being fixed in Track D (returns jnp.nan currently)\n  - Energy class does NOT exist in observables.py\n  - make_energy_fn pattern: src/prolix/physics/md_potential_bundle.py or\n    src/prolix/batched_energy.py -- find the energy evaluation path\n\nRELEVANT FILES:\n  - src/prolix/api/observables.py:14 -- Observable Protocol\n  - src/prolix/batched_energy.py or src/prolix/physics/md_potential_bundle.py -- energy fn\n  - src/prolix/types/bundles.py -- MolecularBundle fields (positions, masses, etc.)\n  - src/prolix/batched_simulate.py -- LangevinState fields (positions, momentum, force, mass)\n\nWHAT TO DO:\n1. Read observables.py in full.\n2. Read batched_simulate.py:1-50 to understand LangevinState fields.\n3. Read batched_energy.py or md_potential_bundle.py to understand energy fn signature.\n   The energy fn typically takes (positions, bundle) or (positions, params) -> scalar.\n4. Add Energy class to observables.py:\n   \`\`\`python\n   class Energy(eqx.Module):\n       '''Observable: total potential energy from integrator state.\n\n       Args:\n           energy_fn: Callable[[positions, bundle], Float[Array, '']]\n               -- the potential energy function (from make_energy_fn or similar).\n               Capture bundle at construction via functools.partial or closure.\n           bundle: MolecularBundle used to compute energy.\n       '''\n       energy_fn: Any  # callable; not jit-traced at construction\n       bundle: Any     # MolecularBundle; static at construction\n\n       def compute(self, state) -> Float[Array, '']:\n           '''Return total potential energy at current state.positions.'''\n           return self.energy_fn(state.positions, self.bundle)\n   \`\`\`\n   Adjust signature to match how make_energy_fn / single_padded_energy actually works.\n5. Export Energy from prolix/api/__init__.py.\n6. Write test in tests/api/test_observables.py (or new file tests/api/test_energy_observable.py):\n   - test_energy_matches_make_energy_fn: create a minimal bundle, create Energy(energy_fn, bundle),\n     call compute(state) and compare to energy_fn(state.positions, bundle) directly.\n     Assert relative error < 1e-6.\n   - test_energy_isinstance_observable: isinstance(Energy(fn, bundle), Observable) is True.\n7. Run: uv run pytest tests/api/ -x --tb=short -v -k energy\n8. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n\nSUCCESS GATE:\n  - prolix.api.Energy exists and is importable\n  - isinstance(Energy(fn, bundle), Observable) is True\n  - test_energy_matches_make_energy_fn passes with < 1e-6 relative error\n  - uv run pytest -m 'not slow' passes\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260614_sprint38_api_impl\nReviewing Track H: Energy observable (#254)\n\nVERIFY:\n1. Run: uv run python -c 'from prolix.api import Energy; print("ok")'\n   PASS if prints "ok". FAIL if ImportError.\n2. Run: uv run python -c '\nfrom prolix.api import Observable, Energy\nimport jax.numpy as jnp\nfn = lambda pos, bundle: jnp.sum(pos)\nbundle = None\ne = Energy(energy_fn=fn, bundle=bundle)\nprint(isinstance(e, Observable))\n'\n   PASS if prints True. FAIL otherwise.\n3. Run: uv run pytest tests/api/ -x --tb=short -v -k energy 2>&1 | tail -10\n   PASS if test(s) pass. FAIL if missing or failing.\n4. Run: uv run pytest -m 'not slow' -x --tb=short -q 2>&1 | tail -5\n   PASS if no regressions.\n\nPASS if all 4 checks pass. FAIL otherwise.\n`,
    "worktree"
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("38: writing chain (, sequential) || research (A, B, C, D, E, F, G, H, read-only)");
const [writing, resA, resB, resC, resD, resE, resF, resG, resH] = await Promise.all([
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
  research_1842: resA,
  research_281: resB,
  research_261: resC,
  research_284: resD,
  research_282: resE,
  research_292: resF,
  research_293: resG,
  research_254: resH
};
