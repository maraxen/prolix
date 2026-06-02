export const meta = {
  name: 'phase5-settle-rstep',
  description: 'Phase 5: implement and validate OpenMM R-step correction for liquid-density NVT stability',
  phases: [
    { title: 'Parallel Setup', detail: 'Write oracle script (A0) and implement R-step correction (B0) in parallel worktrees' },
    { title: 'Smoke & Stage', detail: 'L1 smoke both artifacts, update t2 gate sidecar (B1), commit and push to cluster' },
    { title: 'Cluster Submit', detail: 'sbatch oracle job (A1) and prolix gate job (B2), record job IDs' },
    { title: 'Pull Results', detail: 'Sync cluster catalog, evaluate bathos outcomes for A1 and B2' },
    { title: 'Parity Check', detail: 'Write and submit trajectory comparison script (A2)' },
    { title: 'Phase 5 Close', detail: 'Merge R-step fix, update CLAUDE.md, pre-register dt-sweep campaign (C0)' },
  ],
};

// ─── invocation modes ────────────────────────────────────────────────────────
// args = undefined | { phase: 'impl' }
//   → runs Phases 1–3 (code work + cluster submit), returns { oracle_job, gate_job }
//
// args = { phase: 'verify', oracle_job: '<id>', gate_job: '<id>' }
//   → runs Phases 4–6 (pull results, parity, close)
//
// args = { phase: 'full', oracle_job: '<id>', gate_job: '<id>' }
//   → runs Phases 4–6 directly if job IDs already known
// ─────────────────────────────────────────────────────────────────────────────

const CAMPAIGN_ID = 'ef45b8b4';
const PROLIX_ROOT = '/home/marielle/projects/prolix';
const CLUSTER_REMOTE = 'engaging';

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    success: { type: 'boolean' },
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    commit_sha: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['success', 'summary'],
};

const JOB_SCHEMA = {
  type: 'object',
  properties: {
    job_id: { type: 'string' },
    submitted: { type: 'boolean' },
    partition: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['job_id', 'submitted'],
};

const OUTCOME_SCHEMA = {
  type: 'object',
  properties: {
    job_id: { type: 'string' },
    state: { type: 'string' },
    outcome: { type: 'string' },
    gate_pass: { type: 'boolean' },
    mean_t: { type: 'number' },
    n_passed: { type: 'number' },
    n_xfailed: { type: 'number' },
    n_failed: { type: 'number' },
    notes: { type: 'string' },
  },
  required: ['job_id', 'state', 'gate_pass'],
};

// ═══════════════════════════════════════════════════════════════════════════════
// IMPL PHASE (1–3): code work + cluster submit
// ═══════════════════════════════════════════════════════════════════════════════
async function runImpl() {
  // ── Phase 1: Parallel Setup ─────────────────────────────────────────────────
  phase('Parallel Setup');
  log('Starting Track A (oracle script) and Track B (R-step implementation) in parallel worktrees.');

  const [a0Result, b0Result] = await parallel([
    // Track A: write OpenMM oracle script
    () => agent(
      `You are a fixer agent working in the prolix repo at ${PROLIX_ROOT}.

TASK: Write the OpenMM oracle script for the Phase 5 NVT validation.
Backlog item: #856 (P5-A0)
Campaign: ${CAMPAIGN_ID} (phase5-settle-rstep)

Create TWO files:

1. scripts/experiments/openmm_oracle_tip3p.py
   - argparse CLI: --out PATH (required), --smoke (dry-run), --n-waters INT (default 895), --steps INT (default 3000), --burn INT (default 1000), --dt-fs FLOAT (default 0.5), --gamma-ps FLOAT (default 10.0), --seed INT (default 42)
   - Loads equilibrated 895-water positions from prolix test fixtures:
       from tests.physics.test_explicit_langevin_tip3p_parity import _equil_water_positions
       positions_a, box_edge = _equil_water_positions(args.n_waters, seed=args.seed)
   - Builds OpenMM System: TIP3P rigid water via ForceField('tip3p.xml'), PME electrostatics, LJ cutoff 9 Å
   - Uses LangevinMiddleIntegrator(300*kelvin, gamma/picosecond, dt*femtosecond) with SETTLE constraints
   - Runs args.steps steps, measures KE every step via context.getState(getKineticEnergy=True)
   - DOF = 6*n_waters - 3 (rigid TIP3P)
   - mean_t = 2 * mean(KE[burn:]) / (DOF * BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA) in Kelvin
   - std_t = standard deviation of T(t) over production window
   - Outputs JSON: {"mean_t": float, "std_t": float, "gate_pass": int (1 if mean_t in [295,305] and std_t < 10), "n_steps": int}
   - Out path must be resolved to absolute before writing (Path(args.out).resolve())
   - Use uv run python for imports; import openmm and openmm.app

2. scripts/experiments/openmm_oracle_tip3p.bth.toml
   [experiment]
   hypothesis = "OpenMM LangevinMiddleIntegrator with SETTLE rigid water maintains T=300±5 K for 895-water TIP3P at liquid density (dt=0.5 fs, gamma=10 ps^-1). This validates OpenMM as a trustworthy oracle for prolix Phase 5 R-step validation."

   [outcomes.pass]
   condition = "gate_pass = 1 AND mean_t >= 295 AND mean_t <= 305 AND std_t < 10"
   decision = "Oracle validated. Proceed to prolix R-step gate (P5-B2) and trajectory comparison (P5-A2)."
   reasoning = "OpenMM stays within ±5 K — confirms the integrator is working correctly and can serve as reference."
   is_residual = false

   [outcomes.marginal]
   condition = "gate_pass = 0 AND mean_t >= 290 AND mean_t <= 310"
   decision = "Oracle borderline. Check DOF counting and KE measurement. May need longer burn-in."
   reasoning = "T slightly outside target but not wildly wrong — likely measurement or equilibration issue."
   is_residual = false

   [outcomes.fail]
   condition = "gate_pass = 0"
   decision = "Oracle failed. Do not proceed. Check SETTLE setup, PME parameters, or OpenMM installation."
   reasoning = "Catch-all for OpenMM divergence or large T offset — oracle is untrustworthy."
   is_residual = true

   [result_schema]
   mean_t = "float"
   std_t = "float"
   gate_pass = "int"
   n_steps = "int"

Run L1 smoke: uv run python scripts/experiments/openmm_oracle_tip3p.py --smoke --out /tmp/oracle_smoke.json
Commit with message: "feat(p5): add OpenMM oracle script for liquid-density NVT validation (A0)"
Return summary of files created, commit SHA, and smoke result.`,
      { label: 'A0: oracle-script', phase: 'Parallel Setup', isolation: 'worktree', agentType: 'fixer', schema: RESULT_SCHEMA }
    ),

    // Track B: implement R-step correction
    () => agent(
      `You are a fixer agent working in the prolix repo at ${PROLIX_ROOT}.

TASK: Implement the OpenMM-style R-step correction in settle_langevin.
Backlog item: #858 (P5-B0)
Campaign: ${CAMPAIGN_ID} (phase5-settle-rstep)

ROOT CAUSE (from OpenMM source research):
OpenMM's LangevinMiddleIntegrator applies v += (x - x1) / (dt/n_R) immediately
after each addConstrainPositions() call (_add_R_step method). This is the conservative
momentum correction that makes the position constraint non-energy-injecting. Our
settle_langevin apply_fn was missing this correction entirely.

CHANGE REQUIRED in src/prolix/physics/settle.py, settle_langevin apply_fn:

Replace the current integration order (B-A-O-A-SETTLE_pos-F-B-SETTLE_vel) with
OpenMM-equivalent R-steps (B-R-O-R-F-B), where each R-step is:
  1. x_unconstrained = x + (dt/2) * p/m     [A: unconstrained half-step]
  2. x_constrained = settle_positions(x_unconstrained, x_ref, ...)  [SETTLE_pos]
  3. dp = mass * (x_constrained - x_unconstrained) / (dt/2)        [Δp correction]
  4. p = p + dp                                                      [apply correction]
  5. p = SETTLE_vel(p, x_ref, x_constrained, dt/2)                 [velocity constraint]

The full new apply_fn sequence:
  positions_old = state.positions       # reference for first R-step

  # B: half force kick
  momentum = _langevin_step_b(state.momentum, state.force, _dt)

  # R1: first half-step (A + SETTLE + dp-correction + SETTLE_vel)
  x_unc_1 = _langevin_step_a(state.positions, momentum, state.mass, _dt, shift_fn)
  if constraints: x_unc_1 = project_positions(x_unc_1, ...)
  x_con_1 = settle_positions(x_unc_1, positions_old, ...)
  dp_1 = state.mass * (x_con_1 - x_unc_1) / (_dt / 2.0)
  momentum = momentum + dp_1
  if constraints: momentum = project_momenta(momentum, x_con_1, ...)
  momentum = _langevin_settle_vel(momentum, positions_old, x_con_1, state.mass,
                                  water_indices, _dt/2.0, ...)
  position = x_con_1
  positions_mid = x_con_1              # reference for second R-step

  # O: stochastic step (unchanged)
  if project_ou_momentum_rigid:
    momentum, key = _langevin_step_o_constrained(momentum, position, ...)
  else:
    momentum, key = _langevin_step_o(momentum, ...)

  # R2: second half-step (A + SETTLE + dp-correction + SETTLE_vel)
  x_unc_2 = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)
  if constraints: x_unc_2 = project_positions(x_unc_2, ...)
  x_con_2 = settle_positions(x_unc_2, positions_mid, ...)
  dp_2 = state.mass * (x_con_2 - x_unc_2) / (_dt / 2.0)
  momentum = momentum + dp_2
  if constraints: momentum = project_momenta(momentum, x_con_2, ...)
  momentum = _langevin_settle_vel(momentum, positions_mid, x_con_2, state.mass,
                                  water_indices, _dt/2.0, ...)
  position = x_con_2

  # Force at new constrained positions
  force = force_fn(position, **kwargs)

  # B: final half force kick
  momentum = _langevin_step_b(momentum, force, _dt)

  # Final velocity constraint (catches residual from force kick)
  if constraints: momentum = project_momenta(momentum, position, ...)
  momentum = _langevin_settle_vel(momentum, positions_mid, position, state.mass,
                                  water_indices, _dt, ...)

  # Optional: rigid projection and COM removal (unchanged)
  ...
  return NVTLangevinState(position, momentum, force, state.mass, key)

IMPORTANT implementation notes:
- _langevin_settle_vel signature: (momentum, positions_old, positions_new, mass, water_indices, dt, mass_oxygen, mass_hydrogen, n_iters, settle_velocity_tol)
- The dt argument to _langevin_settle_vel is _dt/2.0 for the mid-step calls and _dt for the final call
- state.mass shape is (N, 1) — the dp correction uses this directly for broadcasting
- Read the actual current apply_fn code first to preserve all existing kwargs, projection_site logic, and COM removal
- Do NOT change settle_lfmiddle_langevin or settle_csvr_npt
- Run uv run pytest tests/physics/test_p2b_nvt_216water.py -m slow -k dilute -v --timeout=600 to verify dilute smoke still passes
Commit with message: "fix(settle): implement OpenMM R-step momentum correction in settle_langevin (P5-B0)"
Return summary of changes, commit SHA, and dilute smoke result.`,
      { label: 'B0: r-step-impl', phase: 'Parallel Setup', isolation: 'worktree', agentType: 'fixer', schema: RESULT_SCHEMA }
    ),
  ]);

  log('Track A (oracle): ' + (a0Result ? a0Result.summary : 'FAILED'));
  log('Track B (R-step): ' + (b0Result ? b0Result.summary : 'FAILED'));

  if (!a0Result || !a0Result.success) {
    log('ERROR: Oracle script (A0) failed. Check fixer output.');
    return { success: false, phase: 'Parallel Setup', failed: 'A0' };
  }
  if (!b0Result || !b0Result.success) {
    log('ERROR: R-step implementation (B0) failed. Check fixer output.');
    return { success: false, phase: 'Parallel Setup', failed: 'B0' };
  }

  // ── Phase 2: Smoke & Stage ──────────────────────────────────────────────────
  phase('Smoke & Stage');
  log('Both artifacts ready. Merging to main, updating gate sidecar, pushing to cluster.');

  const stageResult = await agent(
    `You are a fixer agent working in the prolix repo at ${PROLIX_ROOT}.

TASK: Merge Phase 5 worktree branches, update gate sidecar (B1), and push to cluster.
Backlog items: #859 (P5-B1)

Steps:
1. Check git log to find the Phase 5 R-step commit (from B0 worktree branch). Cherry-pick or merge it to the current worktree-glowing-orbiting-bear branch.
2. Check git log to find the oracle script commit (from A0 worktree branch). Cherry-pick or merge it similarly.
3. Update scripts/experiments/p2b_nvt_gate.bth.toml:
   - hypothesis: "Phase 5 R-step correction (OpenMM-style momentum update dp=m*dx/(dt/2) after each SETTLE_pos) resolves liquid-density NVT energy injection. Both tests PASS: n_passed=2, n_xfailed=0."
   - [outcomes.pass]: condition = "gate_pass = 1 AND n_failed = 0 AND n_xfailed = 0", campaign ${CAMPAIGN_ID}
   - [outcomes.marginal]: condition = "gate_pass = 1 AND n_xfailed = 1 AND n_failed = 0" (code push issue)
   - [outcomes.fail]: condition = "gate_pass = 0 OR n_failed > 0" (is_residual = true)
4. Remove @pytest.mark.xfail from test_nvt_216water_temperature_stability in tests/physics/test_p2b_nvt_216water.py
5. Update scripts/slurm/p2b_slow_gate.slurm: set CAMPAIGN_ID default to ${CAMPAIGN_ID}
6. Run L1 smoke: uv run python scripts/experiments/p2b_nvt_gate.py --smoke --out /tmp/gate_smoke.json
7. Commit: "feat(p5): update gate sidecar and remove xfail for R-step fix (P5-B1)"
8. Push src/, scripts/, tests/ to cluster via rsync:
   rsync -az src/ engaging:~/projects/prolix/src/
   rsync -az scripts/ engaging:~/projects/prolix/scripts/
   rsync -az tests/ engaging:~/projects/prolix/tests/
9. Clear stale bytecode on cluster:
   ssh engaging 'find ~/projects/prolix/src ~/projects/prolix/tests -name "*.pyc" -delete'
Return: success, commit SHA, smoke result, confirmation of cluster push.`,
    { label: 'B1: stage-and-push', phase: 'Smoke & Stage', agentType: 'fixer', schema: RESULT_SCHEMA }
  );

  log('Stage result: ' + (stageResult ? stageResult.summary : 'FAILED'));
  if (!stageResult || !stageResult.success) {
    return { success: false, phase: 'Smoke & Stage', failed: 'B1' };
  }

  // ── Phase 3: Cluster Submit ─────────────────────────────────────────────────
  phase('Cluster Submit');
  log('Submitting oracle (A1) and gate (B2) jobs to cluster in parallel.');

  const ORACLE_SLURM = `#!/bin/bash
#SBATCH --job-name=plx-p5-oracle
#SBATCH --partition=mit_preemptable
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/engaging/%j.out
#SBATCH --error=outputs/logs/engaging/%j.err
set -euo pipefail
_PROJ="\${SLURM_SUBMIT_DIR:-\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/../../" && pwd)}"
cd "\${_PROJ}"
source scripts/slurm/_common_env.sh
WORKSPACE_ROOT="\${ENGAGING_WORKSPACE_ROOT:-\${HOME}/projects}"
export UV_PROJECT="\${WORKSPACE_ROOT}"
source scripts/slurm/_workspace_uv_sync_inner.sh && _workspace_uv_sync_run
cd "\${PROLIX_ROOT}"
_NODE="\${SLURM_JOB_NODELIST:-\$(hostname -s)}"
if [[ "\${_NODE}" == *node4007* ]] || [[ "\${_NODE}" == *node4008* ]]; then
  export XLA_FLAGS="\${XLA_FLAGS:+\${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi
export JAX_ENABLE_X64=1
source scripts/slurm/_bth_env.sh
mkdir -p tmp outputs/p5_oracle
OUT="outputs/p5_oracle/result_\${SLURM_JOB_ID:-local}.json"
CAMPAIGN_ID="\${CAMPAIGN_ID:-${CAMPAIGN_ID}}"
uv run bth run python scripts/experiments/openmm_oracle_tip3p.py \\
  --tag phase5-oracle --tag "slurm_job=\${SLURM_JOB_ID:-local}" \\
  --campaign "\${CAMPAIGN_ID}" --out "\${OUT}" \\
  -- --out "\${OUT}" 2>&1 | tee "\${LOG_ROOT}/app/p5_oracle_\${SLURM_JOB_ID:-local}.log"`;

  const [oracleJob, gateJob] = await parallel([
    () => agent(
      `Write the file scripts/slurm/p5_oracle.slurm in ${PROLIX_ROOT} with this exact content:
\`\`\`
${ORACLE_SLURM}
\`\`\`
Make it executable. Then submit from cluster:
  ssh engaging 'cd ~/projects/prolix && sbatch scripts/slurm/p5_oracle.slurm'
Return the SLURM job ID.`,
      { label: 'A1: oracle-submit', phase: 'Cluster Submit', schema: JOB_SCHEMA }
    ),
    () => agent(
      `Submit the prolix gate job to cluster (campaign ${CAMPAIGN_ID}):
  ssh engaging 'cd ~/projects/prolix && CAMPAIGN_ID=${CAMPAIGN_ID} sbatch scripts/slurm/p2b_slow_gate.slurm'
Return the SLURM job ID.`,
      { label: 'B2: gate-submit', phase: 'Cluster Submit', schema: JOB_SCHEMA }
    ),
  ]);

  const oracleJobId = oracleJob ? oracleJob.job_id : 'UNKNOWN';
  const gateJobId = gateJob ? gateJob.job_id : 'UNKNOWN';

  log('Oracle job submitted: ' + oracleJobId);
  log('Gate job submitted: ' + gateJobId);
  log('To monitor: ssh engaging sacct -j ' + oracleJobId + ',' + gateJobId + ' --format=JobID,State,Elapsed,End');
  log('When complete, run verification phase: Workflow({scriptPath}, {phase:"verify", oracle_job:"' + oracleJobId + '", gate_job:"' + gateJobId + '"})');

  return {
    impl_complete: true,
    oracle_job: oracleJobId,
    gate_job: gateJobId,
    campaign: CAMPAIGN_ID,
    next: 'Run verification phase when both jobs complete (sacct State=COMPLETED)',
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// VERIFY PHASE (4–6): pull results, parity, close
// ═══════════════════════════════════════════════════════════════════════════════
async function runVerify(oracleJobId, gateJobId) {
  // ── Phase 4: Pull Results ───────────────────────────────────────────────────
  phase('Pull Results');
  log('Syncing cluster catalog and evaluating outcomes for oracle=' + oracleJobId + ' and gate=' + gateJobId);

  const [oracleOutcome, gateOutcome] = await parallel([
    () => agent(
      `Check the result of OpenMM oracle job ${oracleJobId} on the engaging cluster.
1. Run: ssh engaging 'cat ~/projects/prolix/outputs/p5_oracle/result_${oracleJobId}.json'
2. Run: ssh engaging 'tail -20 ~/projects/prolix/outputs/logs/engaging/${oracleJobId}.out'
3. Sync bathos: bth sync engaging --pull && bth campaign review ${CAMPAIGN_ID}
Report: mean_t, std_t, gate_pass, any errors.`,
      { label: 'A1: oracle-result', phase: 'Pull Results', schema: OUTCOME_SCHEMA }
    ),
    () => agent(
      `Check the result of prolix gate job ${gateJobId} on the engaging cluster.
1. Run: ssh engaging 'cat ~/projects/prolix/outputs/p2b_nvt_gate/result_${gateJobId}.json'
2. Run: ssh engaging 'tail -30 ~/projects/prolix/outputs/logs/engaging/${gateJobId}.out'
Report: n_passed, n_xfailed, n_failed, gate_pass, actual T value from AssertionError if failed.`,
      { label: 'B2: gate-result', phase: 'Pull Results', schema: OUTCOME_SCHEMA }
    ),
  ]);

  log('Oracle outcome: gate_pass=' + (oracleOutcome ? oracleOutcome.gate_pass : '?') + ' mean_t=' + (oracleOutcome ? oracleOutcome.mean_t : '?'));
  log('Gate outcome: gate_pass=' + (gateOutcome ? gateOutcome.gate_pass : '?'));

  const oraclePassed = oracleOutcome && oracleOutcome.gate_pass;
  const gatePassed = gateOutcome && gateOutcome.gate_pass;

  if (!oraclePassed) {
    log('BLOCKED: Oracle failed (T=' + (oracleOutcome ? oracleOutcome.mean_t : '?') + ' K). OpenMM cannot serve as reference. Investigate before proceeding.');
    return { success: false, blocker: 'oracle', outcome: oracleOutcome };
  }
  if (!gatePassed) {
    log('BLOCKED: Prolix R-step gate failed. T still diverging. R-step correction insufficient or incorrectly implemented.');
    return { success: false, blocker: 'gate', outcome: gateOutcome };
  }

  // ── Phase 5: Parity Check ───────────────────────────────────────────────────
  phase('Parity Check');
  log('Both gates pass. Writing and submitting trajectory comparison (A2).');

  const a2Result = await agent(
    `You are a fixer agent. Write scripts/experiments/p5_trajectory_comparison.py + bth.toml in ${PROLIX_ROOT}.

The script loads the same 895-water equilibrated positions used in the t2 gate, runs BOTH:
  1. Prolix settle_langevin (with R-step correction)
  2. OpenMM LangevinMiddleIntegrator
from identical initial state (same seed). Records T every 10 steps for 3000 steps.
Computes mean absolute deviation |T_prolix(t) - T_openmm(t)| over steps 1000-3000.

Output JSON: {
  "mad_t": float,  // mean absolute deviation in K
  "mean_t_prolix": float,
  "mean_t_openmm": float,
  "std_t_prolix": float,
  "std_t_openmm": float,
  "gate_pass": int  // 1 if mad_t < 50 K and both mean_t in [290,310]
}

Sidecar [outcomes.pass]: mad_t < 50 AND gate_pass = 1
Campaign: ${CAMPAIGN_ID}

Then submit to cluster: ssh engaging 'cd ~/projects/prolix && CAMPAIGN_ID=${CAMPAIGN_ID} sbatch scripts/slurm/p5_comparison.slurm'
(Write the slurm script too — follow the pattern of p2b_slow_gate.slurm.)
Return comparison result or job ID if submitted to cluster.`,
    { label: 'A2: parity-script', phase: 'Parity Check', isolation: 'worktree', agentType: 'fixer', schema: RESULT_SCHEMA }
  );

  log('Parity script: ' + (a2Result ? a2Result.summary : 'FAILED'));
  if (!a2Result || !a2Result.success) {
    return { success: false, phase: 'Parity Check', failed: 'A2' };
  }

  // ── Phase 6: Close ──────────────────────────────────────────────────────────
  phase('Phase 5 Close');
  log('Closing Phase 5: updating CLAUDE.md, v1.1_next_steps.md, pre-registering dt-sweep.');

  const closeResult = await agent(
    `You are a fixer agent. Close Phase 5 in the prolix repo at ${PROLIX_ROOT}.
Backlog item: #862 (P5-C0)

1. In CLAUDE.md:
   - Update "ongoing-work" field to reflect Phase 5 closed (R-step fix, date 2026-06-02)
   - Remove "NVT timestep cap: dt ≤ 0.5 fs" from Known Limitations
   - Update Phase 5 section: status CLOSED, describe the R-step fix

2. In .praxia/docs/v1.1_next_steps.md:
   - Mark Phase 5 complete with outcome and commit SHA from B0

3. Create bathos campaign for dt-sweep via CLI:
   uv run bth campaign create "phase5-dt-sweep" --mode confirmation --question "Does the R-step corrected settle_langevin support dt=1.0 fs and dt=2.0 fs at liquid density (895-water TIP3P)?"

4. Commit: "docs: close Phase 5 SETTLE R-step fix, pre-register dt-sweep campaign (P5-C0)"

Return: success, commit SHA, dt-sweep campaign ID.`,
    { label: 'C0: phase5-close', phase: 'Phase 5 Close', agentType: 'fixer', schema: RESULT_SCHEMA }
  );

  log('Phase 5 closed: ' + (closeResult ? closeResult.summary : 'FAILED'));

  return {
    success: closeResult && closeResult.success,
    phase5_closed: true,
    oracle_mean_t: oracleOutcome ? oracleOutcome.mean_t : null,
    prolix_gate_pass: gatePassed,
    close_commit: closeResult ? closeResult.commit_sha : null,
    next: 'P2b NVT cross-validation is now complete. Proceed to P3 benchmarks.',
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════════
const mode = args && args.phase;

if (mode === 'verify' || mode === 'full') {
  if (!args.oracle_job || !args.gate_job) {
    log('ERROR: verify phase requires args.oracle_job and args.gate_job');
    throw new Error('Missing oracle_job or gate_job in args');
  }
  return await runVerify(args.oracle_job, args.gate_job);
} else {
  return await runImpl();
}
