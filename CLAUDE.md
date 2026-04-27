# Prolix Project CLAUDE.md

Prolix is a JAX-based molecular dynamics engine for protein folding and dynamics.

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

For production batched runs using `batched_produce` or `batched_equilibrate`:

**Safe Pattern (v1.0)**:
Use cold-start state initialization directly rather than `batched_equilibrate`. A known NaN issue in `batched_equilibrate` during batched initialization is scheduled for Sprint 8 fix.

```python
# Instead of: equilibrated_batch = batched_equilibrate(...)
# Use: initialize states directly with explicit warn_counts
from prolix.batched_simulate import LangevinState
import jax.numpy as jnp

state = LangevinState(
    positions=batch.positions,
    momentum=jnp.zeros_like(batch.positions),
    force=initial_forces,   # compute with energy_fn(batch.positions, box) first
    mass=batch.masses,
    key=jax.random.PRNGKey(0),
    cap_count=jnp.int32(0),
    warn_counts=None,        # auto-initialized by __post_init__
)
result = batched_produce(batch, state, steps=n_steps, chunk_size=1)
```

### Known Limitations (v1.0)

1. **NVT timestep cap**: dt ≤ 0.5 fs (rigid body + thermostat feedback coupling)
2. **NPT long-trajectory divergence**: Temperature runaway (→ 10^115 K) beyond ~10 ps due to CSVR + rigid-water KE coupling. Use NVT for longer production runs or wait for Sprint 11 fix. See `tests/physics/test_npt_barostat.py::test_npt_20ps_liquid_water` (marked xfail).
3. **Batched initialization**: `batched_equilibrate` has a known NaN issue; use cold-start initialization instead (see Safe Pattern above).

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
