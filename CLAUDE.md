# Prolix Project CLAUDE.md

Prolix is a JAX-based molecular dynamics engine for protein folding and dynamics.

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
