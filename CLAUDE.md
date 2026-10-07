# Prolix Project CLAUDE.md

Prolix is a JAX-based molecular dynamics engine for protein folding and dynamics.

## Project-specific rules

notes:
- Tiling bug found in `src/prolix/physics/optimization.py`: `inner_tile_size` for the
  exclusion buffer must (a) exceed total exclusion-pair count AND (b) be a multiple of
  `tile_size`, because `tile_reduction` loops `range(n // tile_size)` and silently drops
  the remainder. A non-bucketed value dropped 125 atoms at 895 waters → 10^62 K blowup.
- Always invoke scripts via `uv run python` on the cluster; HOME and ORCD prolix checkouts
  are the same inode (worktree), so a single rsync updates both.

- **Testing & Validation** - pytest commands and CI JSON report queries -> `.praxia/docs/reference/261007_testing-and-validation.md`

## Project Documentation

Extended docs, sprint notes, and architectural decisions live in `.praxia/docs/`:
- `v1.1_next_steps.md` — prioritized candidate work items for v1.1 (LFMiddle, NPT init bug, batching)

## Research Ideas & Roadmap

Exploratory ideas for future sprints (electrostatics, allostery, spectral analysis) are logged in `.praxia/ideas.jsonl`.

## Phase 2: Explicit Solvent Integration

**Decision**: Use SETTLE + Langevin thermostat  
**Constraint**: dt ≤ 1.0 fs for production-scale NVT (n_waters ≳ 16, gamma ≈ 10 ps⁻¹);
dt ≤ 0.5 fs for very small systems (n ≲ 16) or weak friction (gamma ≈ 1 ps⁻¹).
Validated by gate job 15870804 + size sweep ba334c1f; the residual small-N warm bias
is translational finite-size, not dt instability. See
`.praxia/docs/research/260612_p5-dt1fs-size-crossover.md`.

- **Background** - explicit-solvent temperature-control approaches that were investigated -> `.praxia/docs/reference/261007_phase2-background.md`
- **Why the dt cap was 0.5 fs (and how it was lifted to 1.0 fs)** - SETTLE-thermostat kinetic-energy feedback and the production-scale dt lift -> `.praxia/docs/reference/261007_phase2-dt-cap.md`
- **Production Usage** - NVT and NPT call patterns for settle_langevin and settle_csvr_npt -> `.praxia/docs/reference/261007_phase2-production-usage.md`

### Known Limitation: NPT Long-Trajectory Instability

The CSVR thermostat coupling with rigid-body water kinetic energy produces temperature divergence (→ 10^115 K) at timescales beyond ~10 ps. Root cause: CSVR + SETTLE + rigid-water KE interaction feedback. **Short NPT tests pass** (pressure sanity, dt sweep, 4-step validation). **Long trajectories (≥20 ps)** fail with thermal runaway.

**Impact**: Use NPT for short equilibrations only (< ~10 ps). For longer production runs, use NVT ensemble or defer to Sprint 11 fix.

- **Batched Production Simulations** - cold-start LangevinState initialization for batched_produce -> `.praxia/docs/reference/261007_batched-production-simulations.md`

### Known Limitations (v1.0 / v1.1)

1. **NVT timestep cap**: dt ≤ 1.0 fs at production scale (n_waters ≳ 16, gamma ≈ 10 ps⁻¹; gate job 15870804, sweep ba334c1f). dt ≤ 0.5 fs for n ≲ 16 or weak friction (gamma ≈ 1 ps⁻¹). Residual small-N warm bias is translational finite-size (3·N−3 DOF), not dt instability — see `.praxia/docs/research/260612_p5-dt1fs-size-crossover.md`.
2. **NPT long-trajectory divergence**: Temperature runaway (→ 10^115 K) beyond ~10 ps due to CSVR + rigid-water KE coupling. Use NVT for longer production runs. *KE init spike (T≈5000 K at step 0) fixed 2026-06-01: `settle.py:1944` `/ mu` → `* mu` (Bernetti-Bussi sign correction); `test_npt_20ps_liquid_water` xfail removed (commit b6e5bb9).* Long-trajectory divergence root cause (CSVR+SETTLE decoupling) addressed in Phase 6.
3. **Batched SETTLE constraints**: `make_integrator(..., water_indices=...)` is not supported in v1.0. For batched simulations with SETTLE-constrained water, use `settle.settle_langevin` directly and wrap in `jax.vmap` (see v1.1 roadmap for full modular support).

- **Files Affected** - settle.py, simulate.py, and the temperature-control tests -> `.praxia/docs/reference/261007_phase2-files-affected.md`
- **References** - SETTLE and stochastic cell rescaling citations -> `.praxia/docs/reference/261007_phase2-references.md`

## Phase 2–4: Integrator Modular Architecture (v1.0 Release)

### Backward Compatibility

- settle_langevin, settle_csvr_npt APIs unchanged (wrappers around make_integrator)
- Existing code continues to work without modification
- New make_integrator API is opt-in

## Experiment Tracking (bathos)

This project uses **bathos** (`bth`) for experiment provenance, campaign tracking, and result aggregation. The rules below are HARD requirements derived from concrete failures — see the anti-patterns section for what NOT to do.

### Hard rules

1. **Every script in `scripts/experiments/` and `scripts/benchmarks/` MUST run through `bth run`, never raw `python` or `uv run python`.**
   - Local: `uv run bth run python scripts/benchmarks/foo.py --tag X --out out.json -- <args>`
   - Slurm: `uv run bth run python ...` *inside* the slurm script, AFTER sourcing `scripts/slurm/_bth_env.sh`
   - `bth run` captures `SLURM_JOB_ID`, git state, exit code, output paths. Bypassing it loses all provenance.

2. **Every script in those dirs MUST have a per-script sidecar `<stem>.bth.toml` next to it.**
   - Use `[experiment]` schema (verified working). Avoid `[benchmark]` schema — it has gate edge cases.
   - Required sections: `[experiment].hypothesis`, `[outcomes.pass]`, `[outcomes.fail]`, `[result_schema]`, `[metadata]`
   - Each outcome MUST have `condition` (DuckDB SQL), `decision` (next action), `reasoning` (mechanistic justification — not bare narrative).
   - Exactly one outcome MUST have `is_residual = true` (catch-all branch).
   - Reference: `scripts/experiments/fit_bonded_hp4.bth.toml` is the canonical working pattern.

3. **Create the campaign BEFORE running:**
   ```bash
   bth campaign create "campaign-slug" --mode exploration --question "..." 
   # Returns campaign_id like "edbd0b84"
   ```
   Pass `--campaign <id>` to every `bth run` that belongs to the campaign. Without this, runs are orphaned and `bth campaign review` returns empty.

4. **Slurm wrappers must source `_bth_env.sh` and use `bth run`:**
   ```bash
   source scripts/slurm/_bth_env.sh
   uv run bth run python scripts/benchmarks/foo.py \
       --tag cluster --tag <campaign-slug> \
       ${CAMPAIGN_ID:+--campaign $CAMPAIGN_ID} \
       --out "${OUT}" \
       -- <script args>
   ```
   See `scripts/slurm/fit_bonded_hp4.slurm` and `scripts/slurm/bench_external_baseline.slurm` as references.

5. **Sync from cluster via `bth sync`, not manual rsync:**
   ```bash
   bth remote add engaging engaging:~/projects/prolix  # one-time
   bth sync engaging --pull                              # before each analysis
   ```
   If `bth sync` errors on path resolution, fall back to direct catalog rsync:
   ```bash
   rsync -az engaging:~/.bth/catalog/runs/prolix/ ~/.bth/catalog/runs/prolix/
   ```
   But always prefer `bth sync` — it handles incremental fragments correctly.

6. **When the bath gate fails, fix the sidecar — never `--no-sidecar`.**
   - Gate failure writes structured JSON to stderr with `errors[]` and `remediation`. Read it.
   - `--no-sidecar` is reserved for ad-hoc exploration in `scripts/explore/` only.
   - If you find yourself wanting to bypass the gate, that's a signal the experiment isn't ready for `scripts/experiments/` — move it to `scripts/explore/` instead.

- **Query patterns** - bth sql, campaign review, and compact commands -> `.praxia/docs/reference/261007_bathos-query-patterns.md`
- **Anti-patterns observed (don't repeat)** - bathos provenance failures and the fix for each -> `.praxia/docs/reference/261007_bathos-anti-patterns.md`

### Reference files

- Canonical sidecar: `scripts/experiments/fit_bonded_hp4.bth.toml`
- Canonical slurm wrapper: `scripts/slurm/fit_bonded_hp4.slurm`
- Project bath config: `.bth.toml` (project slug + `[remotes.engaging]` block)
- Skill (full reference): load `using-bathos` skill in Claude Code

---

## Cluster Infrastructure

This project uses the Engaging SLURM cluster (SSH, rsync, sbatch) for large-scale MD simulations.

**Configuration & Quick Start:**
- Project-specific defaults: `.agent/docs/CLUSTER_CONFIG.md`
- Global cluster reference: `~/.claude/CLUSTER_INFRASTRUCTURE.md`
- Global recipes: `just -g cluster-*` (login, push-workspace, submit, logs, etc.)
- Cluster rules: `~/.claude/rules/CLUSTER.md`

- **Common Commands** - Engaging login, workspace sync, submit, and queue -> `.praxia/docs/reference/261007_cluster-common-commands.md`

See `.agent/docs/CLUSTER_CONFIG.md` for project-specific settings (partition, GPU, array specs).
