// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 36   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (C,D,E,H,I) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (A,B,F,G) run concurrently.

export const meta = {
  name: "36",
  description: "PBC unit test for AM conservation, tiling hotfix audit, dt=1.0 fs cluster gate, and CI hardening with tiling regression + xfail parametrize sweep.",
  phases: [
    { title: "Track C — L2: Local smoke, dt=1.0 fs, 16-water, 2 seeds (#1517)" },
    { title: "Track D — S1: Bathos sidecar p5_nvt_gate_dt1fs.bth.toml (#1518)" },
    { title: "Track E — S2: Submit 895-water dt=1.0 fs cluster gate (#1519)" },
    { title: "Track H — G1: Gate-pass actions, remove xfail, lift docstring cap (#1525)" },
    { title: "Track I — G2: Gate-fail retrospective (#1527)" },
    { title: "Track A — D1: PBC unit test for _r_step_conserve_angular_momentum (#1515)" },
    { title: "Track B — D2: Tiling hotfix audit at optimization.py (_need=125 case) (#1516)" },
    { title: "Track F — C1: Tiling regression test at chunked_lj_energy formula level (#1520)" },
    { title: "Track G — C2: Parametrize dt xfail sweep [0.5, 1.0, 2.0] fs with n=16 (#1521)" },
  ],
};

const TASK_ID = "260603_p5_settle_correctness";
const MAX_FIX_RETRIES = 2;

function extractVerdict(text) {
  const m = String(text ?? "").match(/verdict:\s*([a-z_]+)/i);
  return m ? m[1].toLowerCase() : "advance";
}

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

// Shared context for the writing tracks (from recon, task 36).
const EMITTER_CTX = `task_id: 260603_p5_settle_correctness\nProject root: /home/marielle/projects/prolix\n\nBackground:\n- Phase 5 / C3 hypothesis confirmed: AM conservation added to SETTLE R-step (commit 678c9cb)\n  eliminates T_rot deficit (was 283.6K, now 296.6–301.9K at dt=0.5 fs, job 15800837 gate_pass=1).\n- PBC split-molecule bug fixed (commit 5d65edb): H atoms unwrapped relative to O in x_unc before\n  computing L_target in _r_step_conserve_angular_momentum. Without this, |d_unc|~L_box~29A caused\n  catastrophic impulse → NaN (root cause of jobs 15770523 and 15775369 returning NaN).\n- Tiling hotfix at src/prolix/physics/optimization.py (function chunked_lj_energy, ~line 28):\n    inner_tile_size = ((_need + tile_size - 1) // tile_size) * tile_size\n  Ceiling-division ensures tile alignment. Bug: non-multiple inner_tile_size silently dropped\n  atoms near the tile boundary → 10^62 K blowup.\n- Spec: .praxia/docs/specs/260611_p5-extended-dt-validation-tiling-regression.md\n\nKey invariants:\n- All Python via \`uv run python\` / \`uv run pytest\`; never bare python\n- task_id must appear in every transduction_log call\n- Gate L1→L2→L3 discipline: never submit to cluster without passing local smoke first\n- Never modify a running bathos sidecar; create a new one\n\nSprint DAG:\n  Phase 1 (local, ~2.5 hrs): a + b (parallel) → c → d → e (gate submit, async)\n  Phase 3 (during gate queue, ~3 hrs): f + g (parallel, concurrent with gate wait)\n  Post-gate: h (if gate_pass=1) OR i (if gate_pass=0)\n`;

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

// ===== TRACK C — Track C — L2: Local smoke, dt=1.0 fs, 16-water, 2 seeds (#1517) =========================
const trackC = () =>
  track(
    "1517",
    "Track C — L2: Local smoke, dt=1.0 fs, 16-water, 2 seeds (#1517)",
    `task_id: ${TASK_ID}. task_id: 260603_p5_settle_correctness\nProject root: /home/marielle/projects/prolix\n\nAdd L2 smoke test to tests/physics/test_settle_temperature_control.py.\n\nNew test: test_dt1fs_16water_smoke\n\nThis is an L2 liveness gate (not paper evidence). It verifies dt=1.0 fs does not\nimmediately NaN on a 16-water system before burning cluster queue time.\n\n@pytest.mark.slow\ndef test_dt1fs_16water_smoke() -> None:\n    '''L2 liveness gate only — not sufficient to validate dt=1.0 fs stability.\n\n    Runs 16-water TIP3P with settle_langevin at dt=1.0 fs for 10 ps.\n    Asserts: no NaN (finite positions and momenta), rough temperature proximity.\n    See cluster gate P5E-S2 (job p5-nvt-gate-dt1fs) for 895-water stability evidence.\n    '''\n    n_waters = 16\n    dt_fs = 1.0\n    sim_ps = 10.0\n    steps = int(sim_ps * 1000.0 / dt_fs)\n    burn = max(100, steps // 3)\n\n    for seed in [42, 43]:\n        mean_t, _ = _mean_rigid_t_after_burn(\n            dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn\n        )\n        assert jnp.isfinite(mean_t), f"seed={seed}: mean_t is NaN/Inf"\n        assert abs(mean_t - 300.0) < 50.0, f"seed={seed}: T={mean_t:.1f}K far from 300K (liveness check)"\n\nIf _mean_rigid_t_after_burn uses gamma internally and does not expose it, use the\ndefault (gamma=1.0 ps^-1) and note: "gamma=1.0 ps^-1 (default). For gate-comparable\nconditions use gamma=10.0 ps^-1 directly via settle_langevin."\nOnly modify _mean_rigid_t_after_burn's signature if it can be done without breaking\nany existing call sites.\n\nGate: \`uv run pytest tests/physics/test_settle_temperature_control.py::test_dt1fs_16water_smoke -v\`\nPASS in < 60 seconds (CPU).\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify the L2 smoke test:\n1. test_dt1fs_16water_smoke exists in test_settle_temperature_control.py.\n2. Uses n_waters=16 (NOT n=2).\n3. Uses dt_fs=1.0, sim_ps=10.0, seeds=[42, 43].\n4. Asserts jnp.isfinite(mean_t) — no NaN.\n5. Asserts |mean_t - 300| < 50 K (loose liveness tolerance).\n6. Docstring says "L2 liveness gate only" and "not sufficient to validate stability."\n7. Marked @pytest.mark.slow.\n8. Run: \`uv run pytest tests/physics/test_settle_temperature_control.py::test_dt1fs_16water_smoke -v\` — PASS in < 60s.\n9. No regressions in existing test_settle_temperature_control.py tests.\n`,
  );

// ===== TRACK D — Track D — S1: Bathos sidecar p5_nvt_gate_dt1fs.bth.toml (#1518) =========================
const trackD = () =>
  track(
    "1518",
    "Track D — S1: Bathos sidecar p5_nvt_gate_dt1fs.bth.toml (#1518)",
    `task_id: ${TASK_ID}. task_id: 260603_p5_settle_correctness\nProject root: /home/marielle/projects/prolix\n\nCreate scripts/experiments/p5_nvt_gate_dt1fs.bth.toml.\n\nRULES:\n- Do NOT modify scripts/experiments/p5_nvt_gate_diag.bth.toml (never-patch-a-running-sidecar)\n- Use [experiment] schema (NOT [benchmark])\n- Exactly one outcome must have is_residual = true (on the fail branch)\n- mode = "confirmation" — threshold pre-registered before run\n\nWRITE THIS FILE (scripts/experiments/p5_nvt_gate_dt1fs.bth.toml) with these fields:\n\n[experiment] section:\n  mode = "confirmation"\n  hypothesis = multi-line string: "dt=1.0 fs SETTLE+Langevin NVT at 895 TIP3P waters,\n    gamma=10 ps^-1, 5 seeds (42-46), 50 ps trajectory is thermally stable:\n    |mean T_rot - 300 K| <= 5 K. Pre-conditions: C3 AM conservation (678c9cb) and\n    PBC fix (5d65edb) both applied. Phase 5 paper claim (C3 AM architecture) is\n    independent of this gate result."\n\n[outcomes.pass] section:\n  condition = "ABS(mean_t_rot - 300.0) <= 5.0"\n  decision = "Remove dt=1.0 fs xfail from test_dt_sweep_16water_nvt and\n    test_temperature_dt1fs_near_target. Update settle_langevin docstring to lift cap to dt <= 1.0 fs."\n  reasoning = multi-line: "Threshold |mean T_rot - 300| <= 5 K anchored at ~4x the T_rot std\n    from job 15800837: at dt=0.5 fs, 5 seeds, T_rot ranged 296.6-301.9 K (std ~1.2 K).\n    4 * 1.2 K = 4.8 K ~ 5 K. This accommodates mild noise at dt=1.0 fs while detecting\n    genuine instability."\n\n[outcomes.fail] section (is_residual = true):\n  is_residual = true\n  condition = "ABS(mean_t_rot - 300.0) > 5.0 OR gate_pass == 0"\n  decision = "Retain all dt=1.0 fs xfails. Write sprint retrospective classifying failure\n    as A (thermostat instability), B (infrastructure regression), or C (near-miss / too tight)."\n  reasoning = multi-line: "T_rot deviation > 5 K at dt=1.0 fs indicates instability or\n    infrastructure regression. Retrospective must distinguish cases for next sprint."\n\n[result_schema] section:\n  mean_t_rot = "float"\n  gate_pass = "int"\n  n_waters = "int"\n  dt_fs = "float"\n  gamma_ps = "float"\n  n_seeds = "int"\n  job_id = "str"\n\n[metadata] section:\n  task_id = "260603_p5_settle_correctness"\n  phase = "Phase 5 Extended"\n  depends_on_commits = ["678c9cb", "5d65edb"]\n  prior_gate_job = "15800837"\n  campaign_slug = "p5-nvt-gate-dt1fs"\n\nAfter writing, verify it passes the bathos gate:\n  uv run bth check scripts/experiments/p5_nvt_gate_dt1fs.bth.toml\n\nRead the structured error JSON if it fails and fix the sidecar. Do NOT use --no-sidecar.\n\nGate: \`uv run bth check scripts/experiments/p5_nvt_gate_dt1fs.bth.toml\` exits 0.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify the bathos sidecar:\n1. File exists at scripts/experiments/p5_nvt_gate_dt1fs.bth.toml.\n2. Uses [experiment] schema (NOT [benchmark]).\n3. mode = "confirmation".\n4. Exactly one outcome has is_residual = true (on the fail branch).\n5. Pass threshold: ABS(mean_t_rot - 300.0) <= 5.0.\n6. reasoning field on pass outcome references job 15800837, 1.2 K std, 4x anchoring.\n7. [result_schema] includes mean_t_rot and gate_pass.\n8. [metadata] includes task_id and depends_on_commits with both 678c9cb and 5d65edb.\n9. Run: \`uv run bth check scripts/experiments/p5_nvt_gate_dt1fs.bth.toml\` — exits 0.\n10. EXISTING p5_nvt_gate_diag.bth.toml is UNCHANGED.\n`,
  );

// ===== TRACK E — Track E — S2: Submit 895-water dt=1.0 fs cluster gate (#1519) =========================
const trackE = () =>
  track(
    "1519",
    "Track E — S2: Submit 895-water dt=1.0 fs cluster gate (#1519)",
    `task_id: ${TASK_ID}. task_id: 260603_p5_settle_correctness\nProject root: /home/marielle/projects/prolix\n\nSubmit the 895-water dt=1.0 fs cluster gate.\n\nPREREQUISITES (verify before proceeding):\n1. Tracks A, B, C, D committed on main branch.\n2. D1, D2, L2 pass locally.\n3. p5_nvt_gate_dt1fs.bth.toml passes \`bth check\`.\n\nSTEPS:\n\n1. Create bathos campaign:\n   uv run bth campaign create "p5-nvt-gate-dt1fs" --mode confirmation --question "Does settle_langevin at dt=1.0 fs achieve T_rot within 5K of 300K at 895 waters?"\n   Record campaign_id.\n\n2. Check if p5_nvt_gate_diag.py accepts --dt-fs argument:\n   uv run python scripts/experiments/p5_nvt_gate_diag.py --help 2>&1 | grep dt\n   If yes: use --dt-fs 1.0 in the bth run invocation.\n   If no: copy p5_nvt_gate_diag.py to p5_nvt_gate_dt1fs.py and hard-code dt_fs=1.0.\n   Do NOT modify the original p5_nvt_gate_diag.py.\n\n3. L1 dry-run (local):\n   uv run python scripts/experiments/p5_nvt_gate_dt1fs.py --dry-run --out /dev/null\n   Must exit 0.\n\n4. Push to cluster (from main checkout, NOT worktree):\n   cd /home/marielle/projects/prolix && just -g cluster-push-workspace prolix engaging\n   OR use mcp__myxcel__push_project.\n   Verify on cluster: ssh engaging "cd ~/projects/prolix && git log --oneline -5"\n   Commits 678c9cb and 5d65edb must appear.\n\n5. Submit:\n   sbatch scripts/slurm/p5_nvt_gate_dt1fs.sbatch\n   OR create the sbatch if it doesn't exist, modeled on existing p5_nvt_gate_diag.sbatch,\n   with dt=1.0 fs, 895 waters, 5 seeds (42-46), gamma=10 ps^-1, 50 ps NVT.\n   The sbatch must source scripts/slurm/_bth_env.sh and use bth run with --campaign $CAMPAIGN_ID.\n\n6. Record job_id and log:\n   transduction_log(action="append_daily", task_id="260603_p5_settle_correctness",\n     payload={entry_date: "YYYY-MM-DD",\n       summary: "Submitted dt=1.0 fs gate, job <id>, campaign p5-nvt-gate-dt1fs (<campaign_id>)",\n       outcomes: ["Sidecar validated", "Campaign created", "Gate submitted async"],\n       next_actions: ["Check gate result when complete", "Proceed with C1/C2"]})\n\nGate: gate job submitted, job ID and campaign ID recorded.\n(Gate pass/fail handled in Tracks H and I after results arrive.)\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify the gate submission:\n1. Campaign p5-nvt-gate-dt1fs exists (uv run bth campaign list).\n2. A dt=1.0 fs gate script exists (p5_nvt_gate_diag.py with --dt-fs arg, OR new p5_nvt_gate_dt1fs.py).\n3. The sbatch script includes: 895 waters, 5 seeds 42-46, dt=1.0 fs, gamma=10 ps^-1, 50 ps.\n4. The bth run invocation in the sbatch includes --campaign $CAMPAIGN_ID.\n5. Cluster has commits 678c9cb and 5d65edb: ssh engaging "cd ~/projects/prolix && git log --oneline -5".\n6. Job ID recorded in transduction_log or session notes.\n`,
  );

// ===== TRACK H — Track H — G1: Gate-pass actions, remove xfail, lift docstring cap (#1525) =========================
const trackH = () =>
  track(
    "1525",
    "Track H — G1: Gate-pass actions, remove xfail, lift docstring cap (#1525)",
    `task_id: ${TASK_ID}. task_id: 260603_p5_settle_correctness\nProject root: /home/marielle/projects/prolix\n\nPRECONDITION: Run only if gate_pass=1 from P5E-S2 (job p5-nvt-gate-dt1fs).\nCheck first: uv run bth campaign review p5-nvt-gate-dt1fs\nIf gate_pass=0, STOP and run Track I (G2) instead.\n\nIF gate_pass=1:\n\n1. Remove dt=1.0 fs xfail from test_dt_sweep_16water_nvt (test_settle_temperature_control.py):\n   Change pytest.param(1.0, marks=pytest.mark.xfail(...), id="dt1.0fs")\n   to:     pytest.param(1.0, id="dt1.0fs")\n   Leave dt=2.0 fs xfail UNCHANGED.\n\n2. Remove xfail from test_temperature_dt1fs_near_target (line ~58):\n   Remove the @pytest.mark.xfail decorator block (lines ~58-66).\n   The test itself stays; only the decorator is removed.\n\n3. Update settle_langevin docstring in src/prolix/physics/settle.py:\n   Search for "dt=0.5" or "0.5 fs" or "dt <= 0.5 fs" near the \`settle_langevin\` function\n   definition or its docstring (around line 930, based on CLAUDE.md reference).\n   Change the dt cap from "≤ 0.5 fs" to "≤ 1.0 fs" everywhere it appears in the docstring.\n   Also update the CLAUDE.md Phase 2 Section "Phase 2: Explicit Solvent Integration":\n   Change "dt=0.5,  # AKMA units (0.5 fs) — do NOT exceed this" to\n           "dt=1.0,  # AKMA units (1.0 fs) — do NOT exceed this (validated by gate P5E-S2)"\n   AND update "dt ≤ 0.5 fs (rigid body + thermostat feedback coupling)" in known limitations.\n\n4. Log:\n   transduction_log(action="append_daily", task_id="260603_p5_settle_correctness",\n     payload={entry_date: "YYYY-MM-DD",\n       summary: "Gate PASSED (T_rot <value>K). Lifted dt cap to 1.0 fs in docstrings. Removed dt=1.0 xfails.",\n       outcomes: ["gate_pass=1", "xfails removed for dt=1.0 fs", "settle.py and CLAUDE.md updated"],\n       next_actions: ["Merge to main", "Close sprint 36"]})\n\nGate: \`uv run pytest tests/physics/test_settle_temperature_control.py -v -m "not slow"\` all pass;\ntest_temperature_dt1fs_near_target has no xfail; test_dt_sweep_16water_nvt dt1.0fs has no xfail.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify gate-pass actions (only if gate_pass=1):\n1. test_dt_sweep_16water_nvt dt1.0fs: NO xfail marker.\n2. test_temperature_dt1fs_near_target (n=2, ~line 67): NO @pytest.mark.xfail decorator.\n3. test_dt_sweep_16water_nvt dt2.0fs: xfail UNCHANGED (still permanent).\n4. settle.py settle_langevin docstring says dt ≤ 1.0 fs.\n5. CLAUDE.md dt constraint updated to 1.0 fs.\n6. Run: \`uv run pytest tests/physics/test_settle_temperature_control.py -m "not slow" -v\` — all pass.\n`,
  );

// ===== TRACK I — Track I — G2: Gate-fail retrospective (#1527) =========================
const trackI = () =>
  track(
    "1527",
    "Track I — G2: Gate-fail retrospective (#1527)",
    `task_id: ${TASK_ID}. task_id: 260603_p5_settle_correctness\nProject root: /home/marielle/projects/prolix\n\nPRECONDITION: Run only if gate_pass=0 from P5E-S2 (job p5-nvt-gate-dt1fs).\nIf gate_pass=1, run Track H (G1) instead. Do NOT run both.\n\nIF gate_pass=0:\n\n1. Pull the gate log:\n   ssh engaging "tail -200 ~/projects/prolix/outputs/logs/slurm/<job_id>.out"\n   Note: observed T_rot value, any NaN/Inf, which seeds failed.\n\n2. Classify the failure:\n   A) Thermostat-scale instability: T_rot drifts steadily away from 300K over 50 ps.\n   B) Infrastructure regression: T_rot is NaN or extreme from step 1 (suggests PBC/tiling bug).\n   C) Near-miss: T_rot is in [295, 310] K but fails the ≤5K threshold.\n\n3. Write retrospective at .praxia/docs/research/260611_p5e-gate-fail.md:\n   Sections: Gate result, Failure classification (A/B/C), Evidence, Root cause hypothesis,\n   Recommended next action:\n     A → Phase 5 dt constraint stands at ≤0.5 fs; no xfail removal; investigate AM fix at dt=1.0 fs\n     B → Debug infrastructure (tiling? PBC? branch sync?); resubmit with fix\n     C → Consider relaxing threshold to 8 K with updated justification; resubmit\n   Also state: "Phase 5 paper claim (C3 AM architecture, commit 678c9cb) is NOT affected. dt gate is bonus validation."\n\n4. Do NOT modify any xfails. Do NOT update settle.py docstring. Do NOT update CLAUDE.md.\n\n5. Log:\n   transduction_log(action="append_daily", task_id="260603_p5_settle_correctness",\n     payload={entry_date: "YYYY-MM-DD",\n       summary: "Gate FAILED. T_rot=<value>K. Classification: <A/B/C>. Retrospective written.",\n       outcomes: ["gate_pass=0", "xfails retained", "retrospective at .praxia/docs/research/260611_p5e-gate-fail.md"],\n       next_actions: ["Review retrospective", "Decide on threshold relaxation or re-investigation"]})\n\nGate: Retrospective document at .praxia/docs/research/260611_p5e-gate-fail.md with failure\nclassification and recommended next action.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify gate-fail retrospective (only if gate_pass=0):\n1. Retrospective exists at .praxia/docs/research/260611_p5e-gate-fail.md.\n2. Contains: T_rot observed, failure classification (A/B/C), evidence, next action.\n3. States paper claim (C3 AM architecture) is NOT affected by gate failure.\n4. NO xfails removed from test_dt_sweep_16water_nvt.\n5. test_temperature_dt1fs_near_target xfail UNCHANGED.\n6. settle.py docstring UNCHANGED (still dt ≤ 0.5 fs).\n`,
  );

// ===== TRACK A — Track A — D1: PBC unit test for _r_step_conserve_angular_momentum (#1515) =========================
const trackA = () =>
  track(
    "1515",
    "Track A — D1: PBC unit test for _r_step_conserve_angular_momentum (#1515)",
    `task_id: ${TASK_ID}. task_id: 260603_p5_settle_correctness\nProject root: /home/marielle/projects/prolix\n\nAdd a PBC split-molecule unit test to tests/physics/test_settle_temperature_control.py,\nimmediately after the existing test_r_step_conserves_angular_momentum (line ~858).\n\nNew test name: test_r_step_conserves_angular_momentum_pbc\n\nContext: _r_step_conserve_angular_momentum takes an optional box=Array|None argument\n(added in commit 5d65edb). When box is provided, H atoms are unwrapped relative to O\nvia minimum-image before computing L_target. Without this, a water molecule straddling a\nPBC boundary gives |d_unc| ~ L_box → catastrophic impulse → NaN.\n\nFIXTURE: n_waters=1, split-molecule water\n  x_unc[0] = [0.1, 0.1, 0.1]   # O at x=0.1\n  x_unc[1] = [3.0, 0.1, 0.1]   # H1 straddling PBC (x=3.0, box=3.1)\n  x_unc[2] = [0.1, 0.3, 0.1]   # H2 within box\n  box = jnp.array([3.1, 3.1, 3.1])\n  Minimum-image d_OH1_x = 3.0 - 0.1 - 3.1 = -0.2 → H1_unwrapped_x = 0.1 - 0.2 = -0.1\n\nSTEPS:\n1. Enable float64: jax.config.update("jax_enable_x64", True)\n2. Import: from prolix.physics import settle\n           from prolix.physics.settle import _r_step_conserve_angular_momentum, WaterIndices\n3. Build water_indices: water_indices = settle.get_water_indices(0, 1)\n4. Add small SETTLE displacement (simulate R-step magnitude):\n   settle_displacement = jnp.array([[0.001, 0.001, 0.001], [-0.001, 0.001, 0.001], [0.001, -0.001, 0.001]])\n   x_con = x_unc + settle_displacement\n5. Build momenta p_pre_a via jax.random.normal(PRNGKey(99), (3,3), dtype=float64) * 5.0\n   half_dt = 0.25\n   mass_arr = jnp.array([15.999, 1.008, 1.008])\n   dp_r_w = mass_arr[:, None] * settle_displacement / half_dt\n   momentum_after_r = p_pre_a + dp_r_w\n6. Call with box:\n   momentum_corrected = _r_step_conserve_angular_momentum(\n     momentum_after_r, p_pre_a, x_unc, x_con, water_indices,\n     mass_oxygen=15.999, mass_hydrogen=1.008, box=jnp.array([3.1, 3.1, 3.1])\n   )\n7. Compute L_target from (x_unc_unwrapped, p_pre_a):\n   Unwrap H1: dH1 = x_unc[1] - x_unc[0]; box_arr = jnp.array([3.1,3.1,3.1])\n   dH1_mi = dH1 - box_arr * jnp.round(dH1 / box_arr)\n   r_H1_uw = x_unc[0] + dH1_mi\n   m = jnp.array([15.999, 1.008, 1.008])\n   com = (m[0]*x_unc[0] + m[1]*r_H1_uw + m[2]*x_unc[2]) / m.sum()\n   L_target = (m[0]*jnp.cross(x_unc[0]-com, p_pre_a[0])\n             + m[1]*jnp.cross(r_H1_uw-com, p_pre_a[1])  # use unwrapped H1\n             + m[2]*jnp.cross(x_unc[2]-com, p_pre_a[2]))\n8. Compute L_actual from (x_unc_unwrapped, momentum_corrected) with same unwrapped positions\n9. Assertions:\n   assert jnp.all(jnp.isfinite(momentum_corrected)), "NaN in corrected momentum (PBC fixture)"\n   assert jnp.allclose(L_actual, L_target, atol=1e-8), f"AM not restored: {L_actual} vs {L_target}"\n\nGate: \`uv run pytest tests/physics/test_settle_temperature_control.py::test_r_step_conserves_angular_momentum_pbc -v\` PASS.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify the PBC unit test:\n1. test_r_step_conserves_angular_momentum_pbc exists in test_settle_temperature_control.py (~after line 858).\n2. Fixture: O at [0.1,0.1,0.1], H1 at [3.0,0.1,0.1] (straddling PBC), box=[3.1,3.1,3.1].\n3. Calls _r_step_conserve_angular_momentum with box=jnp.array([3.1,3.1,3.1]).\n4. Asserts jnp.isfinite(momentum_corrected) — no NaN.\n5. Asserts jnp.allclose(L_actual, L_target, atol=1e-8) using unwrapped H1 positions.\n6. Run: \`uv run pytest tests/physics/test_settle_temperature_control.py::test_r_step_conserves_angular_momentum_pbc -v\` — PASS.\n7. Existing test_r_step_conserves_angular_momentum (without box, line ~858) still PASS.\n`,
    "worktree"
  );

// ===== TRACK B — Track B — D2: Tiling hotfix audit at optimization.py (_need=125 case) (#1516) =========================
const trackB = () =>
  track(
    "1516",
    "Track B — D2: Tiling hotfix audit at optimization.py (_need=125 case) (#1516)",
    `task_id: ${TASK_ID}. task_id: 260603_p5_settle_correctness\nProject root: /home/marielle/projects/prolix\n\nAdd a computed-check test for the tiling hotfix to tests/physics/test_optimization.py.\n\nCONTEXT: The tiling hotfix formula (in src/prolix/physics/optimization.py, chunked_lj_energy):\n  inner_tile_size = ((_need + tile_size - 1) // tile_size) * tile_size\nwhere _need = max(1024, int(excl_indices.shape[0]) + 128).\n\nAdd: def test_inner_tile_size_alignment()\n\nPure Python (no JAX tracing), verifies the formula for key boundary cases:\n\ndef test_inner_tile_size_alignment():\n    '''Regression: inner_tile_size must be divisible by tile_size.\n\n    The original bug silently dropped atoms when inner_tile_size was not a tile_size multiple.\n    At 895-water simulations, this caused 10^62 K blowup.\n    '''\n    tile_size = 128\n    test_cases = [\n        # (_need, expected_inner_tile_size)\n        (125, 128),     # NOT a multiple of 128 without ceiling → was the atom-drop case\n        (128, 128),     # already aligned\n        (129, 256),     # one over → rounds up to next multiple\n        (256, 256),     # aligned\n        (1024, 1024),   # default lower bound of _need\n        (1025, 1152),   # 1025 → ceil(1025/128)*128 = 9*128 = 1152\n        (897, 1024),    # _need = max(1024, 897+128) = 1024; already aligned\n    ]\n    for _need, expected in test_cases:\n        result = ((_need + tile_size - 1) // tile_size) * tile_size\n        assert result == expected, f"_need={_need}: expected {expected}, got {result}"\n        assert result % tile_size == 0, f"_need={_need}: {result} not divisible by {tile_size}"\n\nALSO verify the formula is present in the actual optimization.py:\nRead src/prolix/physics/optimization.py and confirm the line\n    inner_tile_size = ((_need + tile_size - 1) // tile_size) * tile_size\nis present (or equivalent ceiling-division). Include a brief comment in the test docstring:\n  # Formula confirmed present in src/prolix/physics/optimization.py\n\nGate: \`uv run pytest tests/physics/test_optimization.py::test_inner_tile_size_alignment -v\` PASS.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify the tiling hotfix audit test:\n1. test_inner_tile_size_alignment exists in tests/physics/test_optimization.py.\n2. Tests include _need=125 (expected=128) — the original atom-drop boundary case.\n3. All test cases have result % tile_size == 0 assertion.\n4. Docstring mentions "10^62 K blowup" and "895-water".\n5. Run: \`uv run pytest tests/physics/test_optimization.py::test_inner_tile_size_alignment -v\` — PASS.\n6. Pre-existing optimization tests still PASS.\n`,
    "worktree"
  );

// ===== TRACK F — Track F — C1: Tiling regression test at chunked_lj_energy formula level (#1520) =========================
const trackF = () =>
  track(
    "1520",
    "Track F — C1: Tiling regression test at chunked_lj_energy formula level (#1520)",
    `task_id: ${TASK_ID}. task_id: 260603_p5_settle_correctness\nProject root: /home/marielle/projects/prolix\n\nAdd a tiling regression test to tests/physics/test_optimization.py that exercises\nchunked_lj_energy with an exclusion count that produces a non-multiple _need,\nverifying the tiling hotfix prevents atom-drop.\n\nCONTEXT:\nThe tiling formula in optimization.py (chunked_lj_energy):\n  _need = max(1024, int(excl_indices.shape[0]) + 128)\n  inner_tile_size = ((_need + tile_size - 1) // tile_size) * tile_size\n\nTo get _need that requires rounding: need excl_count such that excl_count + 128 > 1024\nAND (excl_count + 128) % 128 != 0.\n  excl_count = 897 → _need = max(1024, 1025) = 1025 → rounded to 1152 ✓\n\nAdd: def test_chunked_lj_tiling_alignment_regression()\n\nThe test passes synthetic positions + large excl_indices to chunked_lj_energy and\nasserts the result is finite (no atom-drop NaN/Inf).\n\nFirst check the actual function signature of chunked_lj_energy in optimization.py.\nThe signature observed is:\n  chunked_lj_energy(r, sigmas, epsilons, excl_indices, excl_scales, displacement_fn, cutoff=9.0, tile_size=128)\n\ndef test_chunked_lj_tiling_alignment_regression():\n    '''Regression for tiling atom-drop bug: inner_tile_size must be tile_size multiple.\n\n    At excl_count=897 (excl_count+128=1025, not divisible by 128), the pre-fix code\n    used inner_tile_size=1025, silently dropped atoms in tile_reduction, and produced\n    10^62 K blowup at 895-water scale.\n    '''\n    from jax_md import space\n    displacement_fn, _ = space.free()\n\n    n_atoms = 8\n    key = jax.random.PRNGKey(0)\n    positions = jax.random.normal(key, (n_atoms, 3)) * 2.0\n    sigmas = jnp.ones(n_atoms) * 3.0\n    epsilons = jnp.ones(n_atoms) * 0.1\n\n    # excl_count=897 → _need=max(1024,1025)=1025 → NOT tile-aligned without the fix\n    excl_count = 897\n    excl_indices = jnp.zeros((excl_count, 2), dtype=jnp.int32)\n    excl_scales = jnp.ones((excl_count, 2))\n\n    result = chunked_lj_energy(positions, sigmas, epsilons, excl_indices, excl_scales, displacement_fn)\n    assert jnp.isfinite(result), f"NaN/Inf at excl_count={excl_count} (tiling regression)"\n\n    # Also verify aligned baseline (excl_count=896 → _need=1024, aligned)\n    excl_count_2 = 896\n    excl_indices_2 = jnp.zeros((excl_count_2, 2), dtype=jnp.int32)\n    excl_scales_2 = jnp.ones((excl_count_2, 2))\n    result_2 = chunked_lj_energy(positions, sigmas, epsilons, excl_indices_2, excl_scales_2, displacement_fn)\n    assert jnp.isfinite(result_2), f"NaN/Inf at excl_count={excl_count_2} (aligned baseline)"\n\nAdjust the call signature if the actual function signature differs.\nIf chunked_lj_energy_nl exists, add a third assertion using it with the same excl_count=897 case.\n\nGate: \`uv run pytest tests/physics/test_optimization.py::test_chunked_lj_tiling_alignment_regression -v\` PASS.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify the tiling regression test:\n1. test_chunked_lj_tiling_alignment_regression exists in test_optimization.py.\n2. Calls chunked_lj_energy with excl_count=897 (non-multiple _need case).\n3. Asserts jnp.isfinite(result) for both excl_count=897 and aligned baseline.\n4. Docstring mentions "atom-drop bug" and "10^62 K blowup".\n5. Run: \`uv run pytest tests/physics/test_optimization.py::test_chunked_lj_tiling_alignment_regression -v\` — PASS.\n6. All pre-existing optimization tests still PASS.\n`,
    "worktree"
  );

// ===== TRACK G — Track G — C2: Parametrize dt xfail sweep [0.5, 1.0, 2.0] fs with n=16 (#1521) =========================
const trackG = () =>
  track(
    "1521",
    "Track G — C2: Parametrize dt xfail sweep [0.5, 1.0, 2.0] fs with n=16 (#1521)",
    `task_id: ${TASK_ID}. task_id: 260603_p5_settle_correctness\nProject root: /home/marielle/projects/prolix\n\nAdd a parametrized dt sweep test to tests/physics/test_settle_temperature_control.py.\n\nIMPORTANT: Do NOT modify or remove existing tests test_temperature_dt1fs_near_target (line ~67,\nn=2) or test_temperature_dt2fs_near_target (line ~86, n=2). These use n=2 and are separate.\n\nAdd a NEW test test_dt_sweep_16water_nvt using n_waters=16.\n\nUse this exact parametrize pattern (individual pytest.param with marks, not conditional pytest.xfail):\n\nDT_CASES = [\n    pytest.param(0.5, id="dt0.5fs"),\n    pytest.param(\n        1.0,\n        marks=pytest.mark.xfail(\n            strict=False,\n            reason="dt=1.0 fs pending cluster gate P5E-S2 (job p5-nvt-gate-dt1fs); remove xfail on gate_pass=1"\n        ),\n        id="dt1.0fs"\n    ),\n    pytest.param(\n        2.0,\n        marks=pytest.mark.xfail(\n            strict=False,\n            reason="dt=2.0 fs untested — no gate run; do NOT interpret xfail as expected-stable"\n        ),\n        id="dt2.0fs"\n    ),\n]\n\n@pytest.mark.slow\n@pytest.mark.parametrize("dt_fs", DT_CASES)\ndef test_dt_sweep_16water_nvt(dt_fs: float) -> None:\n    '''Parametrized dt sweep: 16-water NVT, SETTLE+Langevin.\n\n    dt=0.5 fs: passes (no xfail). dt=1.0 fs: xfail until gate_pass=1.\n    dt=2.0 fs: PERMANENT xfail — no gate run planned; not evidence of stability.\n\n    NOTE: Existing n=2 tests (test_temperature_dt1fs_near_target, test_temperature_dt2fs_near_target)\n    are separate and not the gate evidence. Do not confuse them with this sweep.\n    '''\n    n_waters = 16\n    sim_ps = 10.0\n    steps = int(sim_ps * 1000.0 / dt_fs)\n    burn = max(100, steps // 3)\n    seed = 777\n    mean_t, _ = _mean_rigid_t_after_burn(dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn)\n    assert abs(mean_t - 300.0) < 15.0, f"dt={dt_fs}: T={mean_t:.1f}K, expected 300±15K"\n\nCRITICAL constraints:\n- dt=2.0 fs xfail reason must contain the exact string "do NOT interpret xfail as expected-stable"\n- The dt=2.0 fs xfail is PERMANENT — it must NOT be removed when the dt=1.0 fs gate passes\n- n_waters=16 (not 2)\n\nGate: \`uv run pytest tests/physics/test_settle_temperature_control.py::test_dt_sweep_16water_nvt -v\`\nShows: dt0.5fs PASSED, dt1.0fs XFAIL, dt2.0fs XFAIL.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify the xfail parametrize sweep:\n1. test_dt_sweep_16water_nvt exists with n_waters=16 (NOT n=2).\n2. Three dt cases: 0.5 (no xfail), 1.0 (xfail removable), 2.0 (permanent xfail).\n3. dt=2.0 reason contains "do NOT interpret xfail as expected-stable".\n4. dt=1.0 reason mentions "P5E-S2" and "gate_pass=1".\n5. Marked @pytest.mark.slow.\n6. Existing n=2 tests (test_temperature_dt1fs_near_target, test_temperature_dt2fs_near_target) UNCHANGED.\n7. Run: \`uv run pytest tests/physics/test_settle_temperature_control.py::test_dt_sweep_16water_nvt -v\`\n   Must show: dt0.5fs PASSED, dt1.0fs XFAIL, dt2.0fs XFAIL (both xfails expected to xfail locally).\n`,
    "worktree"
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("Phase 5 Extended: dt=1.0 fs Validation + Tiling Regression: writing chain (C -> D -> E -> H -> I, sequential) || research (A, B, F, G, read-only)");
const [writing, resA, resB, resF, resG] = await Promise.all([
  (async () => {
    const c = await trackC();
    const d = await trackD();
    const e = await trackE();
    const h = await trackH();
    const i = await trackI();
    return { c, d, e, h, i };
  })(),
  trackA(),
  trackB(),
  trackF(),
  trackG(),
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
    "1517": writing.c,
    "1518": writing.d,
    "1519": writing.e,
    "1525": writing.h,
    "1527": writing.i
  },
  research_1515: resA,
  research_1516: resB,
  research_1520: resF,
  research_1521: resG
};
