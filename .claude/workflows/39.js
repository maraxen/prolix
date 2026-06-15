// Sprint 39 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/39.toml
// Regenerate: praxia dw emit-sprint 39.toml
// task_id: 39   sprint_id: 39
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain () runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (A,B,C,D,E,F,G,H) run concurrently.

export const meta = {
  name: "39",
  description: "Critical-path unblocking sprint toward the §7.1 paper figure.\nResearch tracks (#327, #328, #311, #255) produce spec docs and evidence.\nImplementation tracks (#285, #286, #288, #253) harden the prolix.api surface.\n\nDependency unlocks after this sprint:\n  #327 → #250 → #283 → #259 (§7.1 figure)\n  #328 → #260 → #259 (§7.1 figure)\n  #253 now executable (unblocked by #261 done in Sprint 38)\n  #311 now executable (unblocked by #292, #293 done in Sprint 38)\n",
  phases: [
    { title: "Track A — HP1 sub-spec: migration policy for legacy entry-points (#327)" },
    { title: "Track B — HP4 sub-spec: ANI-1x DFT-forces subset curation criteria (#328)" },
    { title: "Track C — Core observables: KineticEnergy, RMSD, Pressure (#285)" },
    { title: "Track D — A1: API contract acceptance test (#286)" },
    { title: "Track E — A3: mypy --strict on prolix.api (#288)" },
    { title: "Track F — B1-smoke: hetero-batch B=4 init+exec benchmark (#253)" },
    { title: "Track G — RR7 evidence: differentiability synthesis (#311)" },
    { title: "Track H — DR-claim1-1: hetero-batched MD precedent scan (#255)" },
  ],
};

const TASK_ID = "39";
const MAX_FIX_RETRIES = 1;

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

// Shared context for the writing tracks (from recon, task 39).
const EMITTER_CTX = `task_id: 260615_sprint39\n\n=== API ANCHORS (src/prolix/api/) ===\n\nobservables.py:\n  Observable Protocol (@runtime_checkable): line 14-41\n    def compute(self, state) -> Array\n  Trajectory eqx.Module: line 44-58\n    positions: Float[Array, 'steps atoms 3']\n    observable_values: dict  # name -> Array\n    n_steps: int (static)\n  Temperature eqx.Module: line 61-111\n    dof: int (static)\n    compute(state): uses state.momentum, state.mass, BOLTZMANN_KCAL from prolix.simulate\n    formula: T = (2 * KE) / (dof * k_B)\n  Energy eqx.Module: line 114-143\n    energy_fn: callable (static field)\n    bundle: any\n    compute(state): returns self.energy_fn(state.positions, self.bundle)\n  NOTE: currently has 'any' type annotations (lines 130-131) — mypy strict violation\n\nensemble_plan.py:\n  EnsemblePlan: line 12-150\n    __init__(bundles: list, planner: Any = None): line 30\n    run(n_steps, dt, kT, seed=0) -> Trajectory: line 45\n    v1.0: single-bundle only; raises NotImplementedError for len(bundles) > 1\n    v1.1: will use xtrax.tiling.BatchPlanner for vmap/safe_map decisions\n\napi/__init__.py — current exports:\n  EnsemblePlan, Observable, Trajectory, Temperature, Energy\n\n=== TILING ANCHORS ===\n\nsrc/prolix/tiling/planner.py:\n  BatchPlanner (line 106): axes, budget_bytes, estimate_memory\n  .plan() -> BatchPlan (line 120)\n  BatchPlan.decisions: list[AxisDecision]\n  AxisDecision.batch_size: int (0=vmap, >0=safe_map tile size)\n\n=== TYPES ===\n\nsrc/prolix/types/bundles.py:\n  MolecularBundle (line 66): from_pdb(path), from_system_dict(d)\n  fields: positions, n_atoms, n_waters, water_indices, masses, etc.\n\n=== SIMULATE ===\n\nsrc/prolix/simulate.py: exports BOLTZMANN_KCAL constant\n\n=== SPRINT 38 COMPLETIONS (all on main, commit range 47e4522..96b9d1c) ===\n#261 EnsemblePlan.run() — DONE\n#281 MolecularBundle.from_pdb — DONE\n#282 from_system_dict — DONE\n#284 Observable/Trajectory eqx.Module — DONE\n#292 S1 D1 force parity tests — DONE (tests/physics/test_s1_force_parity.py)\n#293 S1 D2 jax.grad parity tests — DONE (tests/api/test_s1_jaxgrad_parity.py)\n#254 Energy observable — DONE\n\n=== EXISTING TESTS ===\ntests/api/test_ensemble_plan.py — EnsemblePlan.run() tests\ntests/api/test_observables.py — Observable/Trajectory/Temperature tests\ntests/physics/test_s1_force_parity.py — bond/angle/dihedral force parity\ntests/api/test_s1_jaxgrad_parity.py — jax.grad finite-diff parity\n`;

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

// ===== TRACK A — Track A — HP1 sub-spec: migration policy for legacy entry-points (#327) =========================
const trackA = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a specification-specialist. Write the HP1 migration policy spec document.\n\n## Goal\nCreate \`.praxia/docs/superpowers/specs/260615_hp1-migration-policy.md\` covering the\ndeprecation and replacement plan for prolix's legacy entry-points. This unblocks\nbacklog item #250 (HP1: Migration policy decided) → #283 (EnsemblePlan.from_bundles)\n→ #259 (§7.1 figure).\n\n## Step 1 — Audit legacy entry-points\nRead these files to find all public symbols that will be deprecated:\n- \`src/prolix/batched_simulate.py\` (batched_produce, batched_equilibrate, LangevinState\n  re-export, collate_batch, pad_protein, PaddedSystem)\n- \`src/prolix/simulate.py\` (legacy top-level functions)\n- \`src/prolix/api/__init__.py\` (new canonical API surface)\n- \`src/prolix/types/bundles.py\` (MolecularBundle — the new type)\n\n## Step 2 — Write the spec document\n\nThe document must cover:\n\n### 1. Legacy entry-points inventory\nTable with columns: symbol | current location | category (deprecated/removed/renamed)\n\n### 2. Replacement mapping\nOne-line replacement per legacy symbol:\n- \`batched_produce(batch, state, n_saves, steps_per_save)\` → \`EnsemblePlan(bundles).run(n_steps, dt, kT)\`\n- \`batched_equilibrate\` → removed (use cold-start with real forces — see CLAUDE.md)\n- \`LangevinState\` (public re-export) → \`prolix.typing.LangevinState\` (internal only)\n- \`pad_protein\`, \`PaddedSystem\`, \`collate_batch\` → \`MolecularBundle.from_pdb\` / \`MolecularBundle.from_system_dict\`\n- Add any others found in the audit\n\n### 3. Timeline\n- v1.0 (current): legacy symbols emit \`DeprecationWarning\` on import\n- v1.1: warning text updated to 'use EnsemblePlan instead'\n- v2.0: symbols removed entirely\n\n### 4. CHANGELOG migration table\nMarkdown table format:\n| Legacy | Replacement | Since | Removed |\n| batched_produce | EnsemblePlan.run() | v1.0 | v2.0 |\n...\n\n### 5. Rationale\nOne paragraph: why this migration (paper-facing API alignment, batch-planner\nintegration path, xtrax.tiling backend abstraction).\n\n## Output\nWrite the spec to \`.praxia/docs/superpowers/specs/260615_hp1-migration-policy.md\`\nthen update \`.praxia/docs/INDEX.md\` to add it under the Specs section.\nFinally commit with message: \`spec(api): HP1 migration policy for legacy entry-points (#327)\`\n`,
    { agentType: "librarian", label: "research:327", phase: "Track A — HP1 sub-spec: migration policy for legacy entry-points (#327)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK B — Track B — HP4 sub-spec: ANI-1x DFT-forces subset curation criteria (#328) =========================
const trackB = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a specification-specialist. Write the HP4 ANI-1x curation spec document.\n\n## Goal\nCreate \`.praxia/docs/superpowers/specs/260615_hp4-ani1x-subset.md\` specifying exactly\nhow the 16-system ANI-1x subset for §7.1 is curated. This unblocks #260 (HP4:\nANI-1x DFT-forces sub-spec) → #259 (§7.1 figure).\n\n## Background\n§7.1 figure shows differentiable bonded-parameter fitting on a heterogeneous ANI-1x\nensemble. ANI-1x is a DFT-reference dataset of small organic molecules computed at\nthe ωB97X-D/6-31G* level. We need 16 diverse systems that span dipeptide-scale\ncomplexity, test hetero-batching (varied atom counts), and have reliable DFT forces.\n\n## Reference material to consult\n1. Check if \`.praxia/docs/superpowers/specs/\` has any existing ANI-1x or HP4 docs\n2. Check \`.praxia/research/\` for any prior HP4 findings\n3. The HP4 NLM notebook (accession 301840a8-1c9a-4e9a-b41c-1ea7b3ea8b76) if available\n4. The MD theory NLM notebook (accession 9230d5f7-cff8-49a1-9ccd-8b65e8e207a7)\n\n## Spec document must cover\n\n### 1. Dataset provenance\n- Source: ANI-1x (Smith et al. 2020, DOI: 10.1038/s41597-020-0473-z)\n- Reference level: ωB97X-D/6-31G* DFT forces + energies\n- Download: torchani/ANI1x_releases or HuggingFace chembl-mirna\n\n### 2. Selection criteria (16 systems)\n- Atom count range: 5–35 atoms (dipeptide scale, test hetero-batching)\n- Element diversity: C, H, N, O required; optionally S, F\n- Geometry filter: no strained bonds (bond > 1.6x equilibrium)\n- Force filter: max force component < 200 kcal/mol/Å (remove pathological geometries)\n- Diversity: max pairwise Tanimoto similarity < 0.85 (ECFP4 fingerprints)\n- At least 3 distinct atom counts to validate hetero-batch vmap/safe_map decisions\n\n### 3. Curation pipeline\nStep-by-step script outline:\n\`\`\`\nscripts/curate/hp4_ani1x_subset.py\n  --input  data/raw/ani1x/\n  --output data/ani1x_subset/\n  --n 16\n  --force-ceiling 200.0  # kcal/mol/Å\n  --max-atoms 35\n  --min-atoms 5\n\`\`\`\n\n### 4. Per-system metadata schema\nJSON fields per system:\n- mol_id: str (ANI-1x molecule hash)\n- n_atoms: int\n- elements: list[str]\n- reference_energy_kcal: float\n- reference_forces_shape: [n_atoms, 3]\n- selection_criteria_passed: list[str]\n- reproducibility_sha256: str (hash of coordinates + forces)\n\n### 5. Storage layout\n\`\`\`\ndata/ani1x_subset/\n  manifest.json        # list of 16 mol_ids\n  <mol_id>/\n    coords.npy         # (n_atoms, 3) float32\n    forces.npy         # (n_atoms, 3) float32\n    energy.npy         # scalar float32\n    meta.json          # per-system metadata\n\`\`\`\n\n### 6. Reproducibility\n- Git-tracked manifest.json with sha256 hashes\n- Script version pinned in pyproject.toml or requirements-curation.txt\n\n## Output\nWrite the spec to \`.praxia/docs/superpowers/specs/260615_hp4-ani1x-subset.md\`\nthen update \`.praxia/docs/INDEX.md\` to reference it.\nCommit: \`spec(data): HP4 ANI-1x subset curation criteria for §7.1 (#328)\`\n`,
    { agentType: "librarian", label: "research:328", phase: "Track B — HP4 sub-spec: ANI-1x DFT-forces subset curation criteria (#328)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK C — Track C — Core observables: KineticEnergy, RMSD, Pressure (#285) =========================
const trackC = () =>
  track(
    "285",
    "Track C — Core observables: KineticEnergy, RMSD, Pressure (#285)",
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a fixer. Implement the three remaining core observables for prolix.api.\n\n## Context\nRead \`src/prolix/api/observables.py\` (Observable Protocol line 14, Temperature line 61,\nEnergy line 114 — these are the reference patterns to follow).\nRead \`src/prolix/api/__init__.py\` (current exports).\nRead \`tests/api/test_observables.py\` (existing test patterns).\n\n## What to implement\n\n### 1. KineticEnergy observable (src/prolix/api/observables.py)\nAdd after the Temperature class:\n\n\`\`\`python\nclass KineticEnergy(eqx.Module):\n    '''Observable computing total kinetic energy from integrator state.\n\n    KE = sum_i p_i^2 / (2 * m_i)\n    '''\n\n    def compute(self, state) -> Float[Array, '']:\n        momentum = state.momentum  # (N, 3)\n        mass = state.mass           # (N,) or (N, 1)\n        if mass.ndim == 1:\n            mass_expanded = mass[:, None]\n        else:\n            mass_expanded = mass\n        ke_per_atom = jnp.sum(momentum**2 / (2.0 * mass_expanded), axis=-1)\n        return jnp.sum(ke_per_atom)\n\`\`\`\n\n### 2. RMSD observable (src/prolix/api/observables.py)\nAdd after KineticEnergy:\n\n\`\`\`python\nclass RMSD(eqx.Module):\n    '''Observable computing RMSD vs a stored reference structure.\n\n    RMSD = sqrt(mean over atoms of ||r_i - ref_i||^2)\n    '''\n    reference: Float[Array, 'atoms 3']\n\n    def compute(self, state) -> Float[Array, '']:\n        positions = state.positions  # (N, 3)\n        diff = positions - self.reference\n        return jnp.sqrt(jnp.mean(jnp.sum(diff**2, axis=-1)))\n\`\`\`\n\n### 3. Pressure observable (src/prolix/api/observables.py)\nAdd after RMSD — use ideal-gas approximation (virial requires force decomposition,\nwhich is a v1.1 task):\n\n\`\`\`python\nclass Pressure(eqx.Module):\n    '''Observable computing instantaneous pressure (ideal-gas approximation).\n\n    P = N * k_B * T / V  (ideal gas)\n    Note: virial contribution deferred to v1.1 (requires per-pair force decomposition).\n    '''\n    n_atoms: int = eqx.field(static=True)\n    volume_angstrom3: float = eqx.field(static=True)\n\n    def compute(self, state) -> Float[Array, '']:\n        from prolix.simulate import BOLTZMANN_KCAL\n        momentum = state.momentum\n        mass = state.mass\n        if mass.ndim == 1:\n            mass_expanded = mass[:, None]\n        else:\n            mass_expanded = mass\n        ke_per_atom = jnp.sum(momentum**2 / (2.0 * mass_expanded), axis=-1)\n        total_ke = jnp.sum(ke_per_atom)\n        # T = 2*KE / (3*N*k_B); P = N*k_B*T/V = 2*KE / (3*V)\n        # Volume in Angstrom^3; BOLTZMANN_KCAL in kcal/mol/K\n        # Pressure in kcal/mol/Angstrom^3; convert to bar: 1 kcal/mol/A^3 ~ 68568 bar\n        KCAL_MOL_PER_A3_TO_BAR = 68568.0\n        pressure_kcal = (2.0 * total_ke) / (3.0 * self.volume_angstrom3)\n        return pressure_kcal * KCAL_MOL_PER_A3_TO_BAR\n\`\`\`\n\n## Step 2 — Update exports\nAdd KineticEnergy, RMSD, Pressure to \`src/prolix/api/__init__.py\` imports and \`__all__\`.\n\n## Step 3 — Write tests\nCreate \`tests/api/test_core_observables.py\`:\n- test_kinetic_energy_compute: construct a state with known momentum/mass, verify KE\n- test_rmsd_at_reference: compute RMSD at reference (should be 0.0)\n- test_rmsd_displaced: displace by known amount, verify RMSD\n- test_pressure_positive: check pressure is positive and finite for warm state\n- test_all_implement_protocol: assert isinstance(KineticEnergy(...), Observable) etc.\n\n## Step 4 — Verify\n\`uv run pytest tests/api/test_core_observables.py tests/api/test_observables.py -v\`\nAll tests must pass.\n\n## Commit\n\`feat(api): add KineticEnergy, RMSD, Pressure core observables (#285)\`\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a reviewer. Verify the core observables implementation.\n\n## Verification steps\n\n1. Run: \`uv run pytest tests/api/test_core_observables.py tests/api/test_observables.py -v\`\n   VERIFY: all tests pass\n\n2. Check src/prolix/api/observables.py contains KineticEnergy, RMSD, Pressure classes\n   VERIFY: each has a \`compute(self, state)\` method returning a JAX array\n\n3. Check src/prolix/api/__init__.py exports KineticEnergy, RMSD, Pressure\n   VERIFY: all three appear in __all__\n\n4. Check tests/api/test_core_observables.py exists with at least 4 tests\n   VERIFY: includes RMSD-at-reference==0 and protocol conformance test\n\n5. Run: \`uv run pytest tests/ -m 'not slow' --tb=no -q\`\n   VERIFY: no regressions (same pass count as before this track or better)\n\nPASS if all VERIFY items satisfied.\nFAIL if any observable class is missing, any test fails, or exports are incomplete.\n`,
    "worktree"
  );

// ===== TRACK D — Track D — A1: API contract acceptance test (#286) =========================
const trackD = () =>
  track(
    "286",
    "Track D — A1: API contract acceptance test (#286)",
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a fixer. Write the A1 API contract acceptance test for prolix.api.\n\n## Context\nRead \`src/prolix/api/__init__.py\` (canonical export surface).\nRead \`src/prolix/api/ensemble_plan.py\` (EnsemblePlan, run() signature).\nRead \`src/prolix/api/observables.py\` (Observable Protocol, Trajectory, Temperature, Energy).\nRead \`tests/api/test_ensemble_plan.py\` (existing patterns).\n\n## What to create: tests/api/test_api_contract.py\n\nWrite a comprehensive acceptance test that verifies the entire prolix.api contract.\nStructure as a single test class \`TestApiContract\` with these test methods:\n\n### 1. test_canonical_imports\nVerify that all expected symbols can be imported from prolix.api:\n  from prolix.api import EnsemblePlan, Observable, Trajectory, Temperature, Energy\n  (After Track C ships: also KineticEnergy, RMSD, Pressure — use try/import guards)\n\n### 2. test_observable_protocol_conformance\nFor each concrete observable class (Temperature, Energy, KineticEnergy if available),\nverify isinstance(obs, Observable) is True (runtime_checkable Protocol check).\n\n### 3. test_trajectory_fields\nConstruct a minimal Trajectory and verify:\n- .positions has shape (n_steps, n_atoms, 3)\n- .observable_values is a dict\n- .n_steps equals the declared n_steps\n\n### 4. test_ensemble_plan_construction\nConstruct EnsemblePlan with a minimal mock bundle (or real MolecularBundle.from_system_dict\nif available). Verify it constructs without error and .bundles is set.\n\n### 5. test_ensemble_plan_run_returns_trajectory\nCall EnsemblePlan([bundle]).run(n_steps=5, dt=0.5, kT=2.479e-3) and verify:\n- Returns a Trajectory instance\n- trajectory.n_steps == 5\n- trajectory.positions.shape[0] == 5\n- trajectory.positions is finite (no NaN)\n\n### 6. test_observables_implement_compute\nFor Temperature(dof=10): call obs.compute(mock_state) where mock_state has\n.momentum = jnp.ones((10, 3)) and .mass = jnp.ones(10). Verify result is a scalar.\n\n## Implementation hints\n- Use a minimal MolecularBundle: construct via from_system_dict or mock with a namedtuple\n  that has .positions, .n_atoms, .n_waters=0, .water_indices, .masses\n- Keep tests fast (no actual MD — use n_steps=2 or 5)\n- Use \`@pytest.mark.usefixtures\` if shared setup is needed\n\n## Verify\n\`uv run pytest tests/api/test_api_contract.py -v\`\nAll 6 tests (or more) must pass.\n\n## Commit\n\`test(api): A1 API contract acceptance test for prolix.api surface (#286)\`\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a reviewer. Verify the A1 API contract test.\n\nVERIFY: \`uv run pytest tests/api/test_api_contract.py -v\` passes (all tests green)\nVERIFY: File has at least 5 test methods in TestApiContract class\nVERIFY: test_ensemble_plan_run_returns_trajectory actually calls .run() (not mocked)\nVERIFY: test_trajectory_fields checks .positions shape and finiteness\nVERIFY: \`uv run pytest tests/ -m 'not slow' --tb=no -q\` shows no regressions\n\nPASS if all VERIFY items satisfied.\nFAIL if fewer than 5 tests, any test is empty/pass-only, or run() is not actually called.\n`,
    "worktree"
  );

// ===== TRACK E — Track E — A3: mypy --strict on prolix.api (#288) =========================
const trackE = () =>
  track(
    "288",
    "Track E — A3: mypy --strict on prolix.api (#288)",
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a fixer. Make prolix.api pass mypy --strict type checking.\n\n## Step 1 — Check if mypy is available\nRun: \`uv run python -m mypy --version 2>&1\`\nIf mypy is not installed, add it: \`uv add --dev mypy\`\nAlso add: \`uv add --dev types-all\` or specific stubs as needed.\n\n## Step 2 — Run mypy --strict and capture errors\nRun: \`uv run python -m mypy --strict src/prolix/api/ 2>&1\`\nCapture the full error list.\n\n## Step 3 — Fix errors in src/prolix/api/\n\nKnown issues in observables.py:\n- Line 130: \`energy_fn: any\` — change to \`energy_fn: Any\` (already imported from typing)\n  or a more specific Callable type: \`energy_fn: Callable[..., Float[Array, '']]\`\n- Line 131: \`bundle: any\` — change to \`bundle: Any\`\n- Line 57: \`observable_values: dict\` — should be \`dict[str, Array]\`\n- The Protocol's \`compute\` method uses untyped \`state\` — add \`Any\` annotation:\n  \`def compute(self, state: Any) -> Array:\`\n\nKnown issues in ensemble_plan.py:\n- Return type of run() is missing: add \`-> Trajectory\`\n- \`def energy_fn(positions, **kwargs)\` — add type: \`def energy_fn(positions: Any, **kwargs: Any) -> Any:\`\n- Various local variable types may need annotation\n\n## Step 4 — Fix iteratively\nRun mypy after each batch of fixes. Goal: zero errors under \`--strict\`.\n\nIf some errors are genuinely hard (e.g., jaxtyping annotations conflict with mypy),\nuse \`# type: ignore[specific-code]\` ONLY as a last resort and add a comment explaining why.\n\n## Step 5 — Add mypy config to pyproject.toml\nIf not already present, add under [tool.mypy]:\n\`\`\`\n[tool.mypy]\nstrict = true\nignore_missing_imports = true\n\`\`\`\n(This ensures \`uv run python -m mypy src/prolix/api/\` picks up strict by default.)\n\n## Verify\n\`uv run python -m mypy --strict src/prolix/api/ 2>&1 | grep -c '^src/prolix/api/.*error'\`\nResult should be 0 (no errors).\n\nAlso run: \`uv run pytest tests/api/ -q\` — no regressions.\n\n## Commit\n\`fix(types): mypy --strict compliance for prolix.api module (#288)\`\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a reviewer. Verify mypy --strict compliance.\n\nVERIFY: \`uv run python -m mypy --strict src/prolix/api/ 2>&1 | grep error\` returns empty\n  (or only \`ignore\` comments with explanations)\nVERIFY: observable_values field in Trajectory has explicit type (not bare \`dict\`)\nVERIFY: energy_fn and bundle fields in Energy have explicit types (not bare \`any\`)\nVERIFY: EnsemblePlan.run() has explicit return type annotation \`-> Trajectory\`\nVERIFY: \`uv run pytest tests/api/ -q\` still passes\n\nPASS if mypy --strict exits 0 and all tests pass.\nFAIL if any unignored type error remains or if tests regress.\n`,
    "worktree"
  );

// ===== TRACK F — Track F — B1-smoke: hetero-batch B=4 init+exec benchmark (#253) =========================
const trackF = () =>
  track(
    "253",
    "Track F — B1-smoke: hetero-batch B=4 init+exec benchmark (#253)",
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a fixer. Write the B1-smoke benchmark as a pytest test file.\n\n## Context\nRead \`src/prolix/api/ensemble_plan.py\` (EnsemblePlan, run()).\nRead \`src/prolix/types/bundles.py\` (MolecularBundle, from_system_dict).\nRead \`tests/api/test_ensemble_plan.py\` (existing patterns for constructing bundles).\n\n## What to create: tests/bench/test_b1_smoke.py\n\nThis test is a nightly CI regression detector for the §7.1 hetero-batch benchmark.\nIt tests B=4 different-sized systems through EnsemblePlan.\n\n### Test structure\n\n\`\`\`python\nimport time\nimport pytest\nimport jax.numpy as jnp\nfrom prolix.api import EnsemblePlan\nfrom prolix.types.bundles import MolecularBundle\n\ndef _make_bundle(n_atoms: int, seed: int = 0) -> MolecularBundle:\n    '''Create a synthetic bundle with n_atoms atoms.'''\n    # Use from_system_dict with random positions and unit masses\n    ...\n\n@pytest.mark.slow\n@pytest.mark.benchmark\nclass TestB1Smoke:\n    def test_b1_smoke_b4_wall_clock(self):\n        '''B1-smoke: 4 varied-size bundles, 10 steps, wall-clock logged.'''\n        # B=4 bundles with varied atom counts (5, 10, 20, 35 atoms)\n        # Simulates hetero-batch: varied N tests BatchPlanner vmap/safe_map decisions\n        bundles = [_make_bundle(n) for n in (5, 10, 20, 35)]\n\n        t0 = time.perf_counter()\n        # v1.0: single-bundle only — test each bundle independently as smoke\n        # v1.1 will accept bundles list directly\n        trajectories = []\n        for bundle in bundles:\n            plan = EnsemblePlan([bundle])\n            traj = plan.run(n_steps=10, dt=0.5, kT=2.479e-3)\n            trajectories.append(traj)\n        t_total = time.perf_counter() - t0\n\n        # Verify outputs\n        for traj in trajectories:\n            assert traj.n_steps == 10\n            assert traj.positions.shape[-1] == 3\n            assert jnp.all(jnp.isfinite(traj.positions))\n\n        # Log wall-clock (informational, not gated — baseline for R4 AOT monitor)\n        print(f'B1-smoke wall_clock={t_total:.3f}s for B=4 x 10 steps')\n\n        # Loose timing gate: must complete in < 300s on CPU\n        assert t_total < 300.0, f'B1-smoke timed out: {t_total:.1f}s'\n\`\`\`\n\n## Steps\n\n1. Create \`tests/bench/__init__.py\` (empty) if it doesn't exist\n2. Write \`tests/bench/test_b1_smoke.py\` with the above structure\n3. Add a helper \`_make_bundle(n_atoms, seed)\` that uses MolecularBundle.from_system_dict\n   or constructs a minimal valid bundle with the right fields\n4. Register \`benchmark\` mark in \`pyproject.toml\` under \`[tool.pytest.ini_options]\`\n   markers = [..., "benchmark: marks tests as benchmark (deselect with -m 'not benchmark')"]\n\n## Verify (collection only — don't run the full benchmark locally)\n\`uv run pytest tests/bench/test_b1_smoke.py --collect-only\`\nShould show: 1 test collected (TestB1Smoke::test_b1_smoke_b4_wall_clock)\n\n## Commit\n\`test(bench): B1-smoke hetero-batch B=4 init+exec benchmark (#253)\`\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a reviewer. Verify the B1-smoke benchmark file.\n\nVERIFY: tests/bench/test_b1_smoke.py exists\nVERIFY: tests/bench/__init__.py exists\nVERIFY: \`uv run pytest tests/bench/test_b1_smoke.py --collect-only\` shows 1+ test collected\nVERIFY: test is marked @pytest.mark.slow AND @pytest.mark.benchmark\nVERIFY: test constructs bundles with at least 3 different atom counts\nVERIFY: test calls EnsemblePlan.run() with actual bundles (not mocked)\nVERIFY: test checks traj.positions is finite\nVERIFY: \`uv run pytest tests/ -m 'not slow' --tb=no -q\` shows no regressions\n\nPASS if all VERIFY items satisfied.\nFAIL if test file is missing, has wrong marks, or doesn't call EnsemblePlan.run().\n`,
    "worktree"
  );

// ===== TRACK G — Track G — RR7 evidence: differentiability synthesis (#311) =========================
const trackG = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a specification-specialist. Write the RR7 reviewer rebuttal evidence document.\n\n## Context\nRR7 is the anticipated reviewer objection: 'Differentiability via jax.grad is not\nnovel — this is already done in DMFF, TorchMD, Espaloma, and other tools.'\n\nThe key evidence: prolix's value is the COMBINATION of (1) heterogeneous batching\n(varied-system, bucketed, compile-once) AND (2) differentiability. The §7.1 figure\ndemonstrates this multiplicativity — it could not be produced by a tool with only\none capability.\n\nSprint 38 completed the S1 D1 and D2 tests:\n- D1: tests/physics/test_s1_force_parity.py — per-term analytical vs autograd force agreement\n- D2: tests/api/test_s1_jaxgrad_parity.py — jax.grad through trajectory, finite-diff parity\n\n## Step 1 — Read the Sprint 38 test files to understand what was proved\nRead: \`tests/physics/test_s1_force_parity.py\` (D1 — what bonds/angles/dihedrals are tested)\nRead: \`tests/api/test_s1_jaxgrad_parity.py\` (D2 — what gradient is being verified)\n\n## Step 2 — Write the evidence document\nCreate \`.praxia/docs/research/260615_rr7-differentiability-evidence.md\`\n\nStructure:\n### 1. The objection (verbatim anticipated reviewer text)\n'Differentiability via automatic differentiation is well-established in JAX-MD,\nDMFF, TorchMD-Net, and Espaloma. This does not constitute a novel contribution.'\n\n### 2. Evidence from prolix S1 tests\nCite D1 and D2 results:\n- D1 (test_s1_force_parity.py): bond/angle/dihedral analytical == autograd forces\n  to rtol=1e-4. Confirmed on [N_atoms] atoms, [coverage].\n- D2 (test_s1_jaxgrad_parity.py): jax.grad through full 10-step trajectory vs\n  finite difference, agreement to rtol=1e-4.\n\n### 3. The multiplicativity argument\n'Differentiability × hetero-batching = §7.1 figure. No prior tool provides both.\nDMFF and TorchMD are differentiable but process one system at a time.\njax-md supports batching but with homogeneous shapes (pad-to-max).\nprolix's bucketed hetero-batch + jax.grad enables the exact §7.1 experiment:\ntraining bonded parameters on a MIXED ensemble in a single jit-compiled backward pass.'\n\n### 4. Positioning statement (1 paragraph, paper-draft ready)\n'We do not claim jax.grad itself is novel. We claim that differentiability through\na heterogeneous-batch trajectory is novel, and that the §7.1 result is enabled\nby the combination rather than either capability alone.'\n\n### 5. Evidence status table\n| Evidence item | Status | File |\n| D1 force parity | PASS | tests/physics/test_s1_force_parity.py |\n| D2 grad parity | PASS | tests/api/test_s1_jaxgrad_parity.py |\n| §7.1 figure | PENDING | blocked on HP4 sub-spec (#260) |\n\n## Output\nWrite to \`.praxia/docs/research/260615_rr7-differentiability-evidence.md\`\nUpdate \`.praxia/docs/INDEX.md\` under Research section.\nCommit: \`docs(evidence): RR7 differentiability rebuttal evidence synthesis (#311)\`\n`,
    { agentType: "librarian", label: "research:311", phase: "Track G — RR7 evidence: differentiability synthesis (#311)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK H — Track H — DR-claim1-1: hetero-batched MD precedent scan (#255) =========================
const trackH = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260615_sprint39\n\nYou are a librarian-researcher. Complete the hetero-batched MD precedent scan.\n\n## Goal\nProduce a structured comparison document answering: 'What do existing MD tools do\nregarding heterogeneous batch simulation, and where does prolix's contribution fit?'\n\nThis feeds:\n- Paper §Related Work section\n- RR1 rebuttal: 'How does this differ from existing tools?'\n- §7.1 positioning: why no prior tool can produce this figure\n\n## Tools to survey\n\n### kUPS (Kernel Universal Potential Simulation)\n- Key question: does kUPS support varied-size system batches or only fixed-size?\n- Look for: batch axis, padding approach, system-type mixing\n\n### jax-md\n- Key question: batching approach (pad-to-max? vmap over homogeneous systems?)\n- Look for: NVT batch, safe_map, SpaceFunction design\n\n### OpenMM swarm / multi-simulation\n- Key question: does multi-simulation run varied-size systems concurrently?\n- Look for: swarm API, MPI ranks, memory layout\n\n### GROMACS -multidir\n- Key question: -multidir flag — same topology, varied coordinates? or varied topology?\n- Look for: multi-simulation design documentation\n\n### Folding@home (FAH)\n- Key question: client batching — does FAH send varied work units concurrently?\n- Look for: WU design, protein-type diversity per GPU\n\n### TorchSim (2024/2025 preprint if available)\n- Key question: how does TorchSim handle hetero batches?\n\n## Comparison axes\nFor each tool, assess (Y/N/Partial):\n1. Declarative API (user specifies intent, system optimizes execution)\n2. Varied-topology batch (different atom counts in one compiled forward pass)\n3. Bucketed compilation (separate compiled functions per bucket, not pad-to-max)\n4. Differentiable trajectory (jax.grad or torch.grad through n_steps)\n5. Compile-once for varied shapes (xla shape polymorphism or equivalent)\n\n## Sources to use\n1. WebSearch for each tool's recent papers and documentation\n2. NLM notebook MD Theory (9230d5f7-cff8-49a1-9ccd-8b65e8e207a7) if available\n3. Any \`.praxia/research/\` files from prior Sprint 37 librarian runs\n\n## Output document: \`.praxia/docs/research/260615_claim1-hetero-batch-precedent.md\`\n\nStructure:\n### Executive summary (2-3 sentences)\n### Per-tool analysis (one subsection each)\n### Comparison table\n| Tool | Decl. API | Varied-topo batch | Bucketed compile | Diff. traj | Compile-once |\n| prolix | Y | Y | Y | Y | Partial (v1.1) |\n| kUPS | ? | ? | ? | ? | ? |\n| jax-md | ? | N | N | Y | N |\n... (fill in from research)\n### Prolix contribution framing (1 paragraph, paper-draft ready)\n\n## Commit\n\`docs(research): DR-claim1-1 hetero-batched MD precedent scan (#255)\`\n`,
    { agentType: "librarian", label: "research:255", phase: "Track H — DR-claim1-1: hetero-batched MD precedent scan (#255)", schema: RESEARCH_SCHEMA }
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("Sprint 39 — Spec Completion, Core Observables, API Hardening: writing chain (, sequential) || research (A, B, C, D, E, F, G, H, read-only)");
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
  sprint_id: 39,
  verdicts: {

  },
  research_327: resA,
  research_328: resB,
  research_285: resC,
  research_286: resD,
  research_288: resE,
  research_253: resF,
  research_311: resG,
  research_255: resH
};
