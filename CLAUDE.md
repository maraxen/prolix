# Prolix Project CLAUDE.md

Prolix is a JAX-based molecular dynamics engine for protein folding and dynamics.

## CURRENT STATE

last-edit: 2026-05-29

ongoing-work: LFMiddle dt-sweep falsification (campaign 89c9a900). Running 895-water
TIP3P liquid-box temperature-control sweep on engaging (SLURM job 14718802, fast 2 ps
check for dt=0.5/1.0 fs). Five-bug cascade fixed today — see
`.claude/projects/.../memory/project_lfmiddle_debug.md` for full list. Awaiting
confirmation that 895-water mean T lands near 300 K before declaring the LFMiddle
hypothesis testable.

notes:
- Tiling bug found in `src/prolix/physics/optimization.py`: `inner_tile_size` for the
  exclusion buffer must (a) exceed total exclusion-pair count AND (b) be a multiple of
  `tile_size`, because `tile_reduction` loops `range(n // tile_size)` and silently drops
  the remainder. A non-bucketed value dropped 125 atoms at 895 waters → 10^62 K blowup.
  Current fix rounds up to the next `tile_size` multiple. PROPER FIX (bucketing, so tiling
  invariants always hold by construction) is on the backlog — see `.praxia/ideas.jsonl`.
- Always invoke scripts via `uv run python` on the cluster; HOME and ORCD prolix checkouts
  are the same inode (worktree), so a single rsync updates both.

## Testing & Validation

For detailed testing instructions and CI-safe JSON report querying, see `.copilot/instructions.md`.

Quick reference (use `uv run` to ensure correct environment):
```bash
uv run pytest -m "not slow"              # Fast CI mode (smoke tests only)
uv run pytest                            # Full suite
uv run pytest tests/physics/test_settle.py  # Specific test file
jq '.summary' tmp/pytest.json            # Get pass/fail counts
jq '.tests[] | select(.outcome=="failed") | .nodeid' tmp/pytest.json  # List failures
```

Results are written to `tmp/pytest.json` (added to .gitignore) for grepping and CI integration.

## Project Documentation

Extended docs, sprint notes, and architectural decisions live in `.praxia/docs/`:
- `v1.1_next_steps.md` — prioritized candidate work items for v1.1 (LFMiddle, NPT init bug, batching)

## Research Ideas & Roadmap

Exploratory ideas for future sprints (electrostatics, allostery, spectral analysis) are logged in `.praxia/ideas.jsonl`. Current focus: stabilize core MD engine (Phase 2 constraints, NPT stability).

## Phase 2: Explicit Solvent Integration

**Status**: v1.0 Release with known constraint  
**Decision**: Use SETTLE + Langevin thermostat  
**Constraint**: dt ≤ 0.5 fs (timestep limitation)

### Background

Explicit solvent (TIP3P water with SETTLE rigid constraints) requires careful integration with temperature control. Three approaches were investigated:

1. **Phase 2B (Reorder SETTLE_vel)**: Breaks BAOAB symplectic structure
2. **Phase 2C (Constrained OU)**: Temperature unstable at all timesteps
3. **settle_with_nhc (NHC wrapper)**: Chain state desynchronization at longer timescales

### Why dt ≤ 0.5fs?

SETTLE velocity constraints remove kinetic energy from constrained degrees of freedom. Standard thermostats expect to regulate total kinetic energy, creating a feedback loop:

1. SETTLE constraints remove KE from rigid-body DOF
2. Thermostat tries to add KE back
3. SETTLE removes it again next step
4. Oscillation/divergence emerges

At smaller timesteps (dt ≤ 0.5fs), the per-step constraint impulse magnitude is small, giving the Langevin thermostat (friction + noise) time to re-equilibrate before the next SETTLE constraint applies.

### Production Usage

#### NVT (Constant Volume, Temperature Control)

```python
from prolix.physics import settle

init_fn, apply_fn = settle.settle_langevin(
    energy_fn, shift_fn,
    dt=0.5,  # AKMA units (0.5 fs) — do NOT exceed this
    kT=kT,
    gamma=1.0,  # ps^-1
    mass=masses,
    water_indices=water_indices,
    project_ou_momentum_rigid=True,  # Required for correct equipartition
    projection_site="post_o",
)
```

**Key parameters**:
- `dt=0.5`: Maximum recommended timestep for SETTLE + Langevin coupling
- `project_ou_momentum_rigid=True`: Samples noise in 6D rigid-body subspace per water
- `projection_site="post_o"`: Apply projection after O-step (Ornstein-Uhlenbeck stochastic update)

#### NPT (Pressure Control + Isobaric Barostat)

```python
from prolix.physics import settle

init_fn, apply_fn = settle.settle_csvr_npt(
    energy_fn, shift_fn,
    dt=0.5,  # AKMA units — do NOT exceed 0.5 fs
    kT=kT,
    target_pressure_bar=1.0,  # 1 atm
    tau_barostat_akma=2000.0,  # 0.1 ps time constant
    tau_thermostat_akma=2000.0,  # 0.1 ps time constant
    mass=masses,
    water_indices=water_indices,
    box_init=box_vec,
)

state = init_fn(key, positions, mass=masses, box=box_vec)
for step in range(n_steps):
    state = apply_fn(state, box=state.box)
```

**Status**: NPT short-trajectory mode validated (NVT-like tests pass; long-trajectory stability under investigation)

### Known Limitation: NPT Long-Trajectory Instability

The CSVR thermostat coupling with rigid-body water kinetic energy produces temperature divergence (→ 10^115 K) at timescales beyond ~10 ps. Root cause: CSVR + SETTLE + rigid-water KE interaction feedback. **Short NPT tests pass** (pressure sanity, dt sweep, 4-step validation). **Long trajectories (≥20 ps)** fail with thermal runaway.

**Impact**: Use NPT for short equilibrations only (< ~10 ps). For longer production runs, use NVT ensemble or defer to Sprint 11 fix.

### Temperature Control (NVT Mode)

With NVT configuration:
- Target temperature: 300 K
- Achieved stability: ±5 K over 50+ ps simulations
- No divergence or runaway heating observed
- *Note: This is NVT (constant volume). NPT long-trajectory use is not recommended in v1.0.*

### Batched Production Simulations

For production batched runs using `batched_produce`, initialize `LangevinState`
with real forces computed from the energy function (cold-start):

```python
import dataclasses
import jax
import jax.numpy as jnp
from jax_md import space
from prolix.batched_simulate import LangevinState
from prolix.batched_energy import single_padded_energy
from prolix.physics.md_potential_bundle import value_energy_and_grad_energy

displacement_fn, _ = space.free()

def _init_force(sys):
    def _e(r):
        return single_padded_energy(dataclasses.replace(sys, positions=r), displacement_fn)
    _, f = value_energy_and_grad_energy(_e, sys.positions)
    return f

B = batch.positions.shape[0]
initial_forces = jax.vmap(_init_force)(batch)
keys = jax.random.split(jax.random.PRNGKey(0), B)

state = LangevinState(
    positions=batch.positions,
    momentum=jnp.zeros_like(batch.positions),
    force=initial_forces,
    mass=batch.masses,
    key=keys,
    cap_count=jnp.zeros(B, dtype=jnp.int32),
)
final_state, traj = batched_produce(batch, state, n_saves=n_saves, steps_per_save=steps_per_save)
```

`batched_equilibrate` is **deprecated** (v1.1): it returned `force=zeros` which caused
NaN on the first production step. Use cold-start as shown above. For neighbor-list
equilibration, use `batched_equilibrate_nl`.

### Known Limitations (v1.0 / v1.1)

1. **NVT timestep cap**: dt ≤ 0.5 fs (rigid body + thermostat feedback coupling)
2. **NPT long-trajectory divergence**: Temperature runaway (→ 10^115 K) beyond ~10 ps due to CSVR + rigid-water KE coupling. Use NVT for longer production runs or wait for Sprint 11 fix. See `tests/physics/test_npt_barostat.py::test_npt_20ps_liquid_water` (marked xfail).
3. **Batched SETTLE constraints**: `make_integrator(..., water_indices=...)` is not supported in v1.0. For batched simulations with SETTLE-constrained water, use `settle.settle_langevin` directly and wrap in `jax.vmap` (see v1.1 roadmap for full modular support).

### Future Improvements (v2.0+)

A constraint-aware thermostat that only couples to unconstrained DOF could eliminate the NVT dt limitation, allowing dt ≥ 1.0fs. A decoupled CSVR implementation may fix the NPT long-trajectory divergence by avoiding rigid-body KE feedback loops. See `.agent/docs/RELEASE_DECISION_v1.0.md` for detailed analysis and roadmap.

### Files Affected

- `src/prolix/physics/settle.py`: Main implementation (line ~531)
- `src/prolix/physics/simulate.py`: Langevin integrator components
- `tests/physics/test_settle_temperature_control.py`: Validation tests

### Sprint 7: Batching + NPT Validation (v1.0)

**Safe_map fix**: Fixed reshape bug in `safe_map` that failed on heterogeneous pytrees (different leaf shapes). Added validation to require all pytree leaves have consistent batch dimension. (Step 2)

**LangevinState batching**: Updated `LangevinState.tree_flatten` to properly batch `warn_counts` field, ensuring consistent batched tree structure. (Step 2)

**Regression test**: Added `test_safe_map_heterogeneous_pytree()` to validate error handling for incompatible batched structures. (Step 3)

### References

- Miyamoto, S., & Kollman, P. A. (1992). SETTLE: An analytical version of the SHAKE and RATTLE algorithm for rigid water models. *Journal of Computational Chemistry*, 13(8), 952-962.
- Bernetti, M., & Bussi, G. (2020). Pressure control using stochastic cell rescaling. *Journal of Chemical Physics*, 153(11), 114107.
- Phase 2 investigation summary: `.agent/docs/RELEASE_DECISION_v1.0.md`
- Phase 2 failure analysis: `.agent/docs/daily/P2_FINAL_REPORT.txt`

### Production Status

**v1.0**: settle_langevin with dt ≤ 0.5 fs validated and production-ready (NVT only)
**v1.1+**: LFMiddle hypothesis test, constraint-aware thermostat, NPT fix planned

## Phase 2–4: Integrator Modular Architecture (v1.0 Release)

**Status**: v1.0 Release with modular integrator factory (make_integrator)

### v1.0 Scope — What Ships

✅ **Phase 1 (Complete)**: Constraint system with ConstraintDOFMask
- Explicit DOF decomposition (rigid water vs free solute)
- Projection operators for constraint-aware dynamics
- Comprehensive unit tests (22 tests, all passing)

✅ **Phase 2 (Complete)**: Modular integrator factory with make_integrator
- Step primitives library (O, V, A, SETTLE_vel, CSVR, NHC)
- Step-sequence registry (composition pattern)
- make_integrator factory for BAOAB_LANGEVIN and BAOAB_CSVR_NPT
- Bitwise equivalence to settle_langevin validated (RMSD < 1e-12 Å)
- kUPS cross-validation passed (RMSD < 0.1 Å, KE drift < ±1%)

✅ **Phase 4 (Partial)**: Batching support via vmap
- Unconstrained batching (e.g., 16 parallel solute-only trajectories)
- Validated with machine-epsilon equivalence (RMSD < 1e-15 Å)
- Performance: 2–3x speedup for batch_size ∈ {4, 16}
- **SETTLE-path batching**: smoke test added (4 waters, 100 steps) as final v1.0 validation

### Backward Compatibility

- settle_langevin, settle_csvr_npt APIs unchanged (wrappers around make_integrator)
- Existing code continues to work without modification
- New make_integrator API is opt-in

### Known Limitations (v1.0)

- dt ≤ 0.5 fs (SETTLE+thermostat coupling; documented in Phase 2 section)
- NPT long-trajectory divergence (> 10 ps); use NVT for production (documented in Phase 2 section)
- Batched SETTLE: smoke-tested but not exhaustively validated at scale (see v1.1 roadmap)

---

## v1.1 Roadmap (Deferred Features)

### Phase 3: LFMiddle Optimization & dt-Sweep Hypothesis

**Objective**: Test whether O-step splitting (Leimkuhler-Matthews discretization) reduces SETTLE+thermostat coupling

**Deliverables**: 
- Implement force-step marker in step_registry (enable mid-step force recompute)
- Add LFMiddle_Step and register lfmiddle_langevin sequence
- Hypothesis test: dt-sweep at 0.25, 0.5, 1.0 fs

**Rationale for deferral**: LFMiddle requires architectural changes (force-step refactor) not critical for v1.0 modular framework; hypothesis is exploratory, not contractual

**Estimated effort**: 2–3 days

### Phase 4 (Extended): Large-Scale Batched SETTLE Validation

**Objective**: Comprehensive validation of batched integrators on large water systems

**Deliverables**:
- 64-water batching equivalence test (full 10 ps trajectory)
- Constrained batching performance benchmarking
- Optional: batched kUPS cross-validation

**Rationale for deferral**: v1.0 includes smoke test (4 waters, 100 steps); large-scale testing deferred as optimization/validation phase

**Estimated effort**: 2–3 days

### Phase 5 (New): Constraint-Aware Thermostat

**Objective**: Fix dt ≤ 0.5 fs limitation via constraint-aware thermostat that only couples to unconstrained DOF

**Deliverables**: 
- Redesigned Langevin coupling (per-DOF vs global)
- Validation: dt ≥ 1.0 fs without divergence

**Rationale**: Requires integrator-thermostat redesign; beyond scope of Phase 2 modularization

**Estimated effort**: 4–6 days

### Phase 6 (New): NPT Long-Trajectory Stability

**Objective**: Fix NPT temperature runaway (> 10 ps) via decoupled CSVR implementation

**Deliverables**: 
- Modified CSVR that avoids rigid-water KE feedback loops
- Validation: 50+ ps NPT trajectory without divergence

**Rationale**: Requires detailed coupling analysis; beyond Phase 2 scope

**Estimated effort**: 3–5 days

### Phase 7 (New): Nosé-Hoover Chain Integration

**Objective**: Implement NHC_Step for settle_with_nhc chain thermostat

**Deliverables**: Full NHC state propagation, integration with make_integrator

**Rationale**: Lower priority than LFMiddle hypothesis and SETTLE batching

**Estimated effort**: 2–3 days

---

## v1.0 Release Notes

**Version**: Prolix v1.0 — Modular Integrator Architecture

### Highlights

- Pluggable constraint algorithms (ConstraintDOFMask layer)
- Reusable step primitives library (O, V, A, SETTLE, CSVR)
- Composition factory (make_integrator) enabling custom integrator sequences
- Batching support (unconstrained validated, SETTLE smoke-tested)
- Full backward compatibility with settle_langevin, settle_csvr_npt APIs
- kUPS cross-validation passed

### Breaking Changes

None. New APIs are additive.

### Known Limitations

1. dt ≤ 0.5 fs for NVT (SETTLE+Langevin coupling; workaround: use smaller dt)
2. NPT unstable beyond ~10 ps (use NVT for longer production runs)
3. Batched SETTLE validated on small systems (4 waters, 100 steps); large-scale testing in v1.1

### v1.1 Priorities

- LFMiddle hypothesis test (may enable dt ≥ 1.0 fs)
- Large-scale SETTLE batching validation
- Constraint-aware thermostat (long-term fix for dt limit)

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

### Query patterns

```bash
# Recent runs by campaign tag (bath find can be finicky with multi-tag — use sql)
bth sql "SELECT id, status, tags FROM read_parquet('~/.bth/catalog/runs/prolix/run_*.parquet') WHERE list_contains(tags, 's71-external-baseline') ORDER BY timestamp DESC LIMIT 10"

# Campaign-level review
bth campaign review <campaign_id>

# After many fragments accumulate
bth compact
bth sql "SELECT tool, AVG(per_mol_step_seconds) FROM runs WHERE list_contains(tags, 'external-baseline') GROUP BY tool"
```

### Anti-patterns observed (don't repeat)

| Anti-pattern | What happens | Fix |
|---|---|---|
| Slurm wrapper calls `uv run python foo.py` directly | No provenance, no SLURM_JOB_ID capture, no campaign association — run is invisible to bath | Wrap with `uv run bth run python foo.py --tag ... --campaign ...` |
| Putting `[benchmark]` instead of `[experiment]` in the sidecar | Gate may reject "no [outcomes] section found" depending on bath version | Use `[experiment]` schema (verified) |
| Single campaign-level TOML treated as a sidecar | Gate rejects (wrong file location/structure) | Per-script sidecars + a separate `campaign_design.toml` (non-`.bth.toml` extension) for shared design notes |
| Manual `rsync` of `/outputs/` directly to local instead of bath catalog | Results visible but not tracked; can't query later | Use `bth sync` for catalog; direct rsync only for non-bath artifacts |
| Running smoke without registering a campaign first | Run lands in catalog but `bth campaign review` shows nothing | `bth campaign create` BEFORE the first `bth run` |
| Hand-editing sidecars to remove `is_residual = true` from `[outcomes.fail]` | Gate failure: "exactly one outcome must have is_residual=true" | Keep the residual on the catch-all fail branch |

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

**Common Commands:**
```bash
just -g cluster-login engaging                          # SSH control master
just -g cluster-push-workspace prolix engaging          # Sync workspace
just -g cluster-submit prolix script.sh                 # Submit job
just -g cluster-queue engaging                          # View queue
```

See `.agent/docs/CLUSTER_CONFIG.md` for project-specific settings (partition, GPU, array specs).
