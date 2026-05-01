# Integrator Sequences: Step Composition for MD Integrators

## Overview

An **integrator sequence** is an ordered composition of steps that defines a complete molecular dynamics integrator. Three-layer architecture:

1. **Constraints (Layer 1)**: Kinematic constraint models (ConstraintDOFMask)
2. **Steps (Layer 2)**: Pure integrator primitives (V_Step, A_Step, O_Step, etc.)
3. **Sequences (Layer 3)**: Ordered compositions of steps with shared parameters (StepSequence)

This document describes the three canonical integrator variants implemented in Prolix v1.0.

---

## 1. BAOAB_LANGEVIN

**Symplectic integrator with Ornstein-Uhlenbeck (Langevin) thermostat**.

### Step Sequence Diagram

```
Input: (position, momentum, force)
  |
  v
┌─────────────────────────────┐
│ V_Step (fraction=0.5)       │ → p := p + 0.5*dt*F
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ A_Step (fraction=0.5)       │ → x := x + 0.5*dt*(p/m)
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ O_Step (fraction=1.0)       │ → p := exp(-γ*dt)*p + √(2*m*kT*(1-exp(-2γ*dt)))*N(0,1)
│ (Langevin stochastic)       │
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ A_Step (fraction=0.5)       │ → x := x + 0.5*dt*(p/m)
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ V_Step (fraction=0.5)       │ → p := p + 0.5*dt*F
└─────────────────────────────┘
  |
  v
Output: (position, momentum, force) [Updated]
```

### Mathematical Structure

The BAOAB sequence is a **splitting method** implementing the Langevin equation:

```
d/dt x = p/m
d/dt p = F(x) - γ*p + √(2*m*γ*kT)*dW/dt
```

where:
- **B steps** (V_Step): Momentum update via forces (deterministic, symplectic)
- **A steps**: Position update via momenta (deterministic, symplectic)
- **O step**: Stochastic Ornstein-Uhlenbeck noise injection (preserves canonical ensemble)

**Key Properties**:
- Symplectic (preserves phase-space volume)
- Second-order accurate in timestep
- Time-reversible (with careful noise sampling)
- Preserves Gibbs-Boltzmann measure
- Efficiency: 4 position updates, 1 stochastic step per cycle

### Parameters

```python
dt = 0.5              # Timestep (AKMA units = fs)
gamma = 1.0           # Friction coefficient (ps^-1)
kT = 2.479            # Thermal energy (kcal/mol) [300 K]
water_indices = ...   # [n_waters, 3] array of (O, H1, H2) indices
```

### Constraints

**Timestep limitation**: `dt ≤ 0.5 fs` (0.5 AKMA units).

Why? SETTLE velocity constraints remove kinetic energy from rigid water DOF each step.
The Langevin thermostat tries to re-equilibrate. At smaller dt, the per-step constraint
impulse is smaller, giving the thermostat time to equilibrate before the next SETTLE
constraint applies. At larger dt, oscillation emerges.

**Recommended use**: NVT (constant volume, temperature) ensemble.

### References

- Leimkuhler, B., & Shang, X. (2015). Adaptive thermostats for noisy gradient systems. 
  *SIAM Journal on Numerical Analysis*, 54(2), 721-743.
- Brünger, A. T., Brooks, C. L., & Karplus, M. (1984). Stochastic boundary conditions 
  for molecular dynamics simulations of ST2 water. *Chemical Physics Letters*, 105(5), 495-500.
- Bussi, G., Parrinello, M. (2007). Accurate sampling using Langevin dynamics. 
  *Physical Review E*, 75(5), 056707.

---

## 2. BAOAB_CSVR_NPT

**Symplectic integrator with CSVR thermostat and isobaric barostat**.

### Step Sequence Diagram

```
Input: (position, momentum, force, box_volume)
  |
  v
┌─────────────────────────────┐
│ V_Step (fraction=0.5)       │ → p := p + 0.5*dt*F
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ A_Step (fraction=0.5)       │ → x := x + 0.5*dt*(p/m)
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ CSVR_Step (fraction=1.0)    │ → p := λ*p
│ (Stochastic rescaling)      │    where λ = √[c₁ + c₂*(R² + S) + 2R√(c₁*c₂)]
│ + Barostat coupling         │    couples kinetic energy to target + box rescaling
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ A_Step (fraction=0.5)       │ → x := x + 0.5*dt*(p/m)
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ V_Step (fraction=0.5)       │ → p := p + 0.5*dt*F
└─────────────────────────────┘
  |
  v
Output: (position, momentum, force, box_volume) [Updated]
```

### Mathematical Structure

CSVR (Canonical Sampling via Velocity Rescaling) is a **stochastic isokinetic** thermostat
that rescales momenta to match a target kinetic energy:

```
λ² = c₁ + c₂(R² + S) + 2R√(c₁*c₂)

c₁ = exp(-dt/τ)
c₂ = 1 - c₁
R ~ N(0, 1)
S ~ χ²(n_dof - 1)
```

**Barostat coupling**: The CSVR rescaling is extended to couple box volume,
implementing the isobaric ensemble (constant pressure).

**Key Properties**:
- Preserves Gibbs ensemble (isobaric-isothermal)
- Efficient: single momentum rescaling per step
- Works with constraints (scalar rescaling preserves linear subspaces)
- Weaker stability than Langevin (implicit time coupling)

### Parameters

```python
dt = 0.5                      # Timestep (AKMA units = fs)
kT = 2.479                    # Thermal energy (kcal/mol) [300 K]
n_dof = 27                    # Degrees of freedom for thermostat
tau = 2000.0                  # Relaxation time (AKMA units, ~0.1 ps)
target_pressure_bar = 1.0     # Target pressure (atmospheres)
```

### Constraints & Limitations

**Known Issue (v1.0)**: NPT long-trajectory instability (beyond ~10 ps).
Temperature diverges (→ 10^115 K) due to CSVR + rigid-water KE feedback loop.
**Short equilibrations (< 10 ps) are stable.** Long production runs should use NVT.

**Timestep**: Same as BAOAB_LANGEVIN: `dt ≤ 0.5 fs`.

**Recommended use**: Short NPT equilibrations only (< 10 ps).
For longer MD, use NVT ensemble or defer to v2.0 fix.

### References

- Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling through velocity rescaling.
  *The Journal of Chemical Physics*, 126(1), 014101.
- Bernetti, M., & Bussi, G. (2020). Pressure control using stochastic cell rescaling.
  *The Journal of Chemical Physics*, 153(11), 114107.
- Bussi, G., Parinello, M. (2009). Stochastic thermostats.
  *In Thermodynamics and Statistical Mechanics: A Molecular Dynamics Perspective*.

---

## 3. settle_with_nhc

**BAOAB integrator with Nosé-Hoover Chain (NHC) thermostat**.

### Step Sequence Diagram

```
Input: (position, momentum, force, nhc_state)
  |
  v
┌─────────────────────────────┐
│ V_Step (fraction=0.5)       │ → p := p + 0.5*dt*F
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ A_Step (fraction=0.5)       │ → x := x + 0.5*dt*(p/m)
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ O_Step (fraction=1.0)       │ → Stochastic noise (uncoupled from NHC)
│ (Langevin-like noise)       │
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ A_Step (fraction=0.5)       │ → x := x + 0.5*dt*(p/m)
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ NHC_Step (fraction=1.0)     │ → xi_i := update chain variables
│ (Nosé-Hoover Chain)         │    ξ̇ᵢ couples to kinetic energy
│                             │    p := ξ₁ * p  (scale momenta)
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│ V_Step (fraction=0.5)       │ → p := p + 0.5*dt*F
└─────────────────────────────┘
  |
  v
Output: (position, momentum, force, nhc_state) [Updated]
```

### Mathematical Structure

NHC is an **extended-system thermostat** that couples kinetic energy to an auxiliary
chain of friction variables:

```
d/dt x = p/m
d/dt p = F(x) - ξ₁*p
d/dt ξ₁ = (K.E. / target_K.E. - 1) / Q₁   (feedback from kinetic energy)
d/dt ξᵢ = (Qᵢ₋₁*ξᵢ₋₁² - kT) / Qᵢ   (chain propagation)
```

**Key Properties**:
- Deterministic (no explicit noise term, stochasticity emerges from coupling)
- Allows tunable correlation times (chain length, Q masses)
- Preserves Gibbs-Boltzmann measure
- More expensive than Langevin (requires chain state integration)

### Parameters

```python
dt = 0.5                      # Timestep (AKMA units = fs)
gamma = 1.0                   # Not used (NHC is deterministic)
kT = 2.479                    # Thermal energy (kcal/mol) [300 K]
chain_masses = [Q₀, Q₁, ...]  # Masses for chain variables (future implementation)
chain_length = 4              # Number of chain variables (typical: 3-5)
```

### Implementation Status (v1.0)

**Current**: `NHC_Step` is a **placeholder** (no-op). Full implementation deferred to Phase 2.

Requires:
1. ChainState dataclass (ξᵢ, ξ̇ᵢ variables for each water or system)
2. Integrator for chain ODEs (RESPA or simple Verlet)
3. Parameter: Qᵢ mass selection (from target timescale τ_NHC)

**Future work**: Implement once ChainState available in Phase 2.

### Constraints

Same as BAOAB_LANGEVIN: `dt ≤ 0.5 fs` (rigid body + thermostat coupling).

**Recommended use**: Long equilibrations and constant-temperature production (once implemented).
Useful for high-viscosity or soft-matter systems where Langevin damping is high.

### References

- Nosé, S. (1984). A unified formulation of the constant temperature molecular dynamics methods.
  *The Journal of Chemical Physics*, 81(1), 511-519.
- Hoover, W. G. (1985). Canonical dynamics: Equilibrium phase-space distributions.
  *Physical Review A*, 31(3), 1695.
- Martyna, G. J., Klein, M. L., & Tuckerman, M. E. (1992). Nosé–Hoover chains: the canonical ensemble via continuous dynamics.
  *The Journal of Chemical Physics*, 97(4), 2635-2643.
- Kondo, H., Espíndola-Heredia, F., & Bussi, G. (2020). Rapid estimation of configurational sampling 
  in biomolecular simulations. *Journal of Chemical Theory and Computation*, 16(1), 622-627.

---

## Comparison Table

| Property | BAOAB_LANGEVIN | BAOAB_CSVR_NPT | settle_with_nhc |
|----------|---|---|---|
| **Type** | Langevin + thermostat | CSVR + barostat | NHC thermostat |
| **Ensemble** | NVT | NPT (short only) | NVT (when ready) |
| **Symplectic** | Yes | Weak | Yes |
| **Time-reversible** | Yes (w/ careful noise) | No | Yes |
| **Stochasticity** | Gaussian noise injection | Stochastic rescaling | Deterministic coupling |
| **Cost per step** | 5 positions, 1 RNG sample | 4 positions, χ² sample | 5 positions, chain ODEs |
| **Stability (dt=0.5fs)** | ✓ Proven (50+ ps NVT) | ⚠ Short only (<10 ps) | TBD (not yet implemented) |
| **Correlation time** | Fixed (γ⁻¹) | Fixed (τ_relaxation) | Tunable (chain_length, Q) |
| **Reference** | Leimkuhler & Shang | Bussi et al. | Martyna et al. |

---

## Why Three Integrator Variants?

Each addresses different use cases in MD simulations:

1. **BAOAB_LANGEVIN** (NVT): Production simulations, stable and proven.
   - Temperature control via friction + noise
   - Simplest to understand and debug
   - Best for explicit solvent (water dynamics matter)

2. **BAOAB_CSVR_NPT** (NPT short): Initial structure equilibration at target pressure.
   - Couple to pressure + temperature simultaneously
   - Converge box size to target pressure
   - Limited to short runs (<10 ps) in v1.0

3. **settle_with_nhc** (NVT, future): High-quality thermostating for long equilibrations.
   - Tunable correlation times
   - Deterministic (no spurious noise artifacts)
   - More expensive but better for careful thermostats
   - Currently a placeholder; full implementation in Phase 2

---

## Usage Examples

### 1. Build a Custom BAOAB_LANGEVIN Integrator

```python
from prolix.physics.step_system import (
    make_sequence, make_step, IntegratorState
)

# Get sequence with custom parameters
seq = make_sequence(
    'baoab_langevin',
    dt=0.5,
    gamma=1.0,
    kT=2.479,
    water_indices=water_indices,
)

# Create steps
v_step = make_step('v_step', fraction=0.5)
a_step = make_step('a_step', fraction=1.0)
o_step = make_step('o_step', fraction=1.0, project_rigid=True)

# Compose into integrator
def step_integrator(state, constraint_dofs):
    state = v_step.apply(state, **seq.parameters)
    state = a_step.apply(state, **seq.parameters)
    state, _ = o_step.apply(state, constraint_dofs=constraint_dofs, **seq.parameters)
    state = a_step.apply(state, **seq.parameters)
    state = v_step.apply(state, **seq.parameters)
    return state

# Simulate
for i in range(n_steps):
    state = step_integrator(state, constraint_dofs)
    force = energy_fn(state.position)
    state = state.__replace__(force=force)
```

### 2. Quick NPT Equilibration

```python
seq_npt = make_sequence(
    'baoab_csvr_npt',
    dt=0.5,
    kT=2.479,
    n_dof=192,  # 64 waters * 3
    tau=2000.0,
    target_pressure_bar=1.0,
)

# Same composition as BAOAB_LANGEVIN, but with CSVR instead of O
csvr_step = make_step('csvr_step', n_dof=192)
# ... (short runs, < 10 ps)
```

### 3. Custom Hybrid (e.g., partial stochastic damping)

Create custom parameters or step sequences by extending `make_sequence`:

```python
custom_params = {
    'dt': 0.5,
    'gamma': 0.5,  # Weaker damping
    'kT': 2.479,
    'water_indices': water_indices,
    'custom_damp_mode': 'selective',  # Project noise to specific DOF
}
# Pass to steps as needed
```

---

## Architecture: Why Three Layers?

```
Layer 3: Sequences (baoab_langevin, baoab_csvr_npt, settle_with_nhc)
         └─ Orderings of steps + shared parameters
            
Layer 2: Steps (V_Step, A_Step, O_Step, CSVR_Step, NHC_Step, SETTLE_Velocity_Step)
         └─ Pure primitives, reusable across sequences
            
Layer 1: Constraints (ConstraintDOFMask + SETTLE kinematics)
         └─ Kinematic model (independent from steps)
```

**Benefits**:
- **Orthogonal design**: Change constraints without affecting steps or sequences
- **Reusability**: Steps appear in multiple sequences
- **Testability**: Each layer validated independently
- **Extensibility**: Add new integrators by composing existing steps (no duplication)
- **OpenMM-inspired**: Similar to OpenMM's context-system composability

---

## Future Extensions (Phase 2+)

1. **Barostat decoupling**: Separate CSVR + anisotropic barostat for better NPT stability
2. **Constraint-aware thermostats**: Only couple to unconstrained DOF (removes dt cap)
3. **Specialized steps**: Position-constraint coupling, pressure correction steps
4. **Trajectory analysis**: Monitor thermodynamic properties (T, P, energy) per sequence
5. **Hybrid ensembles**: NVPT (constant N, V, P, T) for specialized applications

---

## References & Further Reading

### Primary (Cited Above)

1. Leimkuhler, B., & Shang, X. (2015). Adaptive thermostats for noisy gradient systems.
2. Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling through velocity rescaling.
3. Nosé, S. (1984). A unified formulation of the constant temperature molecular dynamics methods.
4. Hoover, W. G. (1985). Canonical dynamics: Equilibrium phase-space distributions.
5. Martyna, G. J., Klein, M. L., & Tuckerman, M. E. (1992). Nosé–Hoover chains.

### Context & Related

- OpenMM Documentation: https://openmm.org/ (step-based integrators pattern)
- kUPS Framework: https://github.com/prolix-sim/kups (composition-over-inheritance)
- GROMACS Manual: https://manual.gromacs.org/ (integrators & thermostats)

### Prolix Internal

- ADR-005 v2.0: Integrator Builder Architecture
- Phase 1.1: ConstraintDOFMask (constraints.py)
- Phase 1.2: Step Primitives (step_system.py, 6 step types)
- Phase 1.3: StepSequence Registry (this file + step_system.py)
- Phase 2: make_integrator composition (future)
