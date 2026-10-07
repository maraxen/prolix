---
title: "Production Usage"
description: "NVT and NPT call patterns for settle_langevin and settle_csvr_npt, moved verbatim from CLAUDE.md."
status: reference
date: '261007'
---

### Production Usage

#### NVT (Constant Volume, Temperature Control)

```python
from prolix.physics import settle

init_fn, apply_fn = settle.settle_langevin(
    energy_fn, shift_fn,
    dt=1.0,  # AKMA units (1.0 fs) — validated at production scale (n≳16, gamma≈10); use 0.5 for n≲16 / weak friction
    kT=kT,
    gamma=10.0,  # ps^-1 — adequate friction is required for the dt=1.0 fs lift
    mass=masses,
    water_indices=water_indices,
    project_ou_momentum_rigid=True,  # Required for correct equipartition
    projection_site="post_o",
)
```

**Key parameters**:
- `dt=1.0`: Recommended timestep at production scale (n≳16, gamma≈10 ps⁻¹); use `dt=0.5` for very small systems (n≲16) or weak friction
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
