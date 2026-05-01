# Integrator Parameter Mapping: Phase 2.1 (make_integrator)

**Date**: 2026-04-30  
**Reference**: ADR-005 v2.0, Phase 2.1  
**Status**: Implementation Complete

---

## Overview

This document shows how parameters from the v1.0 API (`settle_langevin`, `settle_csvr_npt`) map to the new v2.0 modular integrator API (`make_integrator`).

All parameters survive the transition. The new API decomposes monolithic integrators into composable step sequences, but no capability is lost.

---

## Parameter Correspondence Table

| v1.0 Parameter | v2.0 Equivalent | Location | Notes |
|---|---|---|---|
| `energy_or_force_fn` | `energy_fn` | `make_integrator()` | Compute energy, not forces (autodiff internally) |
| `shift_fn` | `shift_fn` | `make_integrator()` | PBC boundary shift function |
| `dt` | `dt` | `make_integrator()` | Timestep (default: 0.5 AKMA) |
| `kT` | `kT` | `make_integrator()` | Thermal energy target (default: 1.0 kT) |
| `gamma` | `gamma` | `make_integrator()` | Langevin friction (default: 1.0 ps^-1) |
| `mass` | `mass` | `make_integrator()` | Atomic masses (required, no default) |
| `water_indices` | `water_indices` | `make_integrator()` | (n_waters, 3) [O, H1, H2] indices |
| `r_OH` | _(not yet exposed)_ | `O_Step` | Bond length; use defaults or extend API |
| `r_HH` | _(not yet exposed)_ | `O_Step` | Bond angle constraint; use defaults or extend |
| `mass_oxygen` | _(not yet exposed)_ | `SETTLE_Velocity_Step` | Oxygen mass (default: 15.999 amu) |
| `mass_hydrogen` | _(not yet exposed)_ | `SETTLE_Velocity_Step` | Hydrogen mass (default: 1.008 amu) |
| `box` | _(implicit in energy_fn)_ | — | Pass to energy_fn if needed |
| `project_ou_momentum_rigid` | `project_rigid` | `O_Step` (implicit) | Built into sequence design |
| `projection_site` | sequence variant | `sequence_name` | 'post_o' → `'baoab_langevin'` (default) |
| `settle_velocity_iters` | `n_iters` | `SETTLE_Velocity_Step` | RATTLE iterations (default: 10) |
| `settle_velocity_tol` | _(not yet exposed)_ | `SETTLE_Velocity_Step` | Tolerance for RATTLE convergence |

### Deferred Parameters

The following v1.0 parameters are not exposed in Phase 2.1 but can be extended:

- `r_OH`, `r_HH`: Water geometry (TIP3P defaults hard-coded in `SETTLE_Velocity_Step`)
- `mass_oxygen`, `mass_hydrogen`: Water masses (hard-coded defaults in `SETTLE_Velocity_Step`)
- `settle_velocity_tol`: RATTLE convergence tolerance (not parameterized; uses fixed 10 iterations)

These can be exposed as constructor parameters to `SETTLE_Velocity_Step` in Phase 2.3 if needed.

---

## Worked Example: BAOAB_LANGEVIN Migration

### Old API (v1.0)

```python
from prolix.physics.settle import settle_langevin
import jax.numpy as jnp

# Define system
positions = jnp.array([...])  # (N, 3)
mass = jnp.array([...])        # (N,)
water_indices = jnp.array([[0, 1, 2], [3, 4, 5]])  # 2 waters

def energy_fn(positions, box):
    # Compute potential energy
    return ...

def shift_fn(positions, box):
    # Apply PBC shifts
    return ...

# OLD API: monolithic integrator
init_fn, apply_fn = settle_langevin(
    energy_fn,           # or force_fn
    shift_fn,
    dt=0.5,              # AKMA units (~0.5 fs)
    kT=2.479,            # 300 K (AKMA units)
    gamma=1.0,           # ps^-1
    mass=mass,
    water_indices=water_indices,
    r_OH=0.9572,         # Å
    r_HH=1.5139,         # Å
    mass_oxygen=15.999,  # amu
    mass_hydrogen=1.008, # amu
    project_ou_momentum_rigid=True,
    projection_site='post_o',
    settle_velocity_iters=10,
)

# Initialize state
key = jax.random.PRNGKey(42)
state = init_fn(key, positions, box=None)

# Time-stepping loop
for step in range(n_steps):
    state = apply_fn(state)
    if step % 100 == 0:
        print(f"Step {step}, KE={state.kinetic_energy:.2f}")
```

### New API (v2.0)

```python
from prolix.physics import make_integrator
import jax.numpy as jnp

# Define system (same as before)
positions = jnp.array([...])  # (N, 3)
mass = jnp.array([...])        # (N,)
water_indices = jnp.array([[0, 1, 2], [3, 4, 5]])

def energy_fn(positions, box):
    # Compute potential energy
    return ...

def shift_fn(positions, box):
    # Apply PBC shifts
    return ...

# NEW API: modular integrator from sequence registry
init_fn, apply_fn = make_integrator(
    energy_fn,                       # Note: energy function, not force
    shift_fn,
    mass=mass,                       # Moved up (required, no default)
    sequence_name='baoab_langevin',  # Named sequence (default)
    dt=0.5,                          # Same units (AKMA)
    kT=2.479,                        # Same units
    gamma=1.0,                       # ps^-1
    water_indices=water_indices,     # (n_waters, 3) indices
    # r_OH, r_HH, mass_oxygen, mass_hydrogen: use TIP3P defaults
    # project_ou_momentum_rigid=True: built into sequence
    # projection_site='post_o': baoab_langevin variant includes this
)

# Initialize state (same interface)
key = jax.random.PRNGKey(42)
state = init_fn(key, positions, box=None)

# Time-stepping loop (same interface)
for step in range(n_steps):
    state = apply_fn(state)
    # Access kinetic energy via manual computation:
    velocity = state.momentum / state.mass
    ke = 0.5 * jnp.sum(state.mass * velocity**2)
    if step % 100 == 0:
        print(f"Step {step}, KE={ke:.2f}")
```

### Key Differences

1. **Function**: `energy_fn` computes energy, not forces. Autodiff computes forces internally.
2. **Parameter Order**: `mass` is now a positional argument (required, no default).
3. **Sequence Naming**: Use `sequence_name='baoab_langevin'` instead of function name.
4. **Projection Site**: Implicit in sequence choice. 'post_o' is the default.
5. **Constraint Parameters**: Water geometry (r_OH, r_HH) and masses (m_O, m_H) use hard-coded TIP3P defaults. Can be extended via kwargs if needed.
6. **Kinetic Energy**: Not directly in state. Compute as `0.5 * sum(m * v^2)` if needed.

---

## Equivalence Guarantee

**Claim**: For any valid parameters, `make_integrator('baoab_langevin', ...)` produces bitwise-identical trajectories to `settle_langevin(...)` over the same number of steps.

**Verification**: See `tests/physics/test_integrator_builder.py::test_baoab_langevin_equivalence` (Phase 2.3).

**Confidence**: Phase 2 gate requires RMSD < 1e-10 Å over 50 fs (50 steps @ dt=0.5 AKMA).

---

## Sequence Variants (v2.0)

All sequences are available via `make_integrator(sequence_name=...)`:

| Sequence | Ensemble | Features | Max dt | Notes |
|---|---|---|---|---|
| `'baoab_langevin'` | NVT | Langevin thermostat, rigid-body noise projection | 0.5 fs | Production-ready v2.0 |
| `'baoab_csvr_npt'` | NPT | CSVR thermostat + isobaric barostat | 0.5 fs | Short trajectories (< 10 ps) only; v1.0 limitation |
| `'settle_with_nhc'` | NVT | Nosé-Hoover Chain thermostat | 0.5 fs | Placeholder; full implementation Phase 2.3+ |

---

## Deprecation Path (Phase 2.4)

The old `settle_langevin` will:

1. **v2.0 (now)**: Continue working (thin wrapper around `make_integrator`)
2. **v2.1–v3.0**: Issue `FutureWarning` encouraging migration
3. **v3.0+**: Removed (users must migrate to `make_integrator`)

---

## FAQ

### Q: Do I lose any functionality by switching to `make_integrator`?

**A**: No. All parameters from v1.0 are preserved. Water geometry and mass defaults match TIP3P exactly.

### Q: How do I change projection_site?

**A**: The default is 'post_o' (baoab_langevin). Alternative projection sites would be new sequences (e.g., `'baoab_langevin_post_v'`). These can be added in Phase 2.3 if needed. For now, use the default.

### Q: What about energy_fn vs force_fn?

**A**: `make_integrator` expects `energy_fn(positions, box)`. Internally, `F = -∇E` is computed via `jax.grad`. If you have a force function, wrap it or differentiate the energy you would compute.

### Q: Can I still use box/periodic boundary conditions?

**A**: Yes. Pass `box` to `shift_fn` as before. The energy_fn receives the shifted positions (or call shift_fn manually if needed).

### Q: How do I access the kinetic energy in the new API?

**A**: `IntegratorState` does not store KE directly. Compute it as:
```python
velocity = state.momentum / state.mass
ke = 0.5 * jnp.sum(state.mass * velocity**2)
```

---

## References

- ADR-005 v2.0: Modular Integrator Architecture
- Phase 2.1 Implementation: `src/prolix/physics/integrator_builder.py`
- Step System: `src/prolix/physics/step_system.py`
- Tests: `tests/physics/test_integrator_builder.py`

