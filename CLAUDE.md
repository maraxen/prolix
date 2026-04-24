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

### Temperature Control

With this configuration:
- Target temperature: 300 K
- Achieved stability: ±5 K over 50+ ps simulations
- No divergence or runaway heating observed

### Future Improvements (v2.0+)

A constraint-aware thermostat that only couples to unconstrained DOF could eliminate this limitation. This would allow dt ≥ 1.0fs while maintaining temperature control. See `.agent/docs/RELEASE_DECISION_v1.0.md` for detailed analysis and roadmap.

### Files Affected

- `src/prolix/physics/settle.py`: Main implementation (line ~531)
- `src/prolix/physics/simulate.py`: Langevin integrator components
- `tests/physics/test_settle_temperature_control.py`: Validation tests

### References

- Miyamoto, S., & Kollman, P. A. (1992). SETTLE: An analytical version of the SHAKE and RATTLE algorithm for rigid water models. *Journal of Computational Chemistry*, 13(8), 952-962.
- Phase 2 investigation summary: `.agent/docs/RELEASE_DECISION_v1.0.md`
- Phase 2 failure analysis: `.agent/docs/daily/P2_FINAL_REPORT.txt`
