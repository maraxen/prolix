---
title: "Batched Production Simulations"
description: "Cold-start LangevinState initialization for batched_produce, moved verbatim from CLAUDE.md."
status: reference
date: '261007'
---

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
