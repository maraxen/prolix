from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prolix.physics.step_system import Step, IntegratorState
from prolix.physics.types import IntegratorParams

class MC_Barostat_Step(Step):
    """Monte Carlo Barostat step for NPT ensemble.

    Performs volume scaling moves periodically to maintain constant pressure.

    Attributes:
        barostat_interval: Number of steps between barostat attempts.
        pressure: Target pressure.
        volume_scale_factor: Magnitude of volume move (e.g. 0.05).
        energy_fn: Function to compute system energy (U(r, L)).
    """
    barostat_interval: int = eqx.field(static=True)
    pressure: float = eqx.field(static=True)
    volume_scale_factor: float = eqx.field(static=True)
    energy_fn: Callable = eqx.field(static=True)

    n_molecules: int = eqx.field(static=True, default=0)

    def __init__(
        self,
        barostat_interval: int,
        pressure: float,
        energy_fn: Callable,
        n_molecules: int,
        volume_scale_factor: float = 0.05,
    ):
        self.barostat_interval = barostat_interval
        self.pressure = pressure
        self.energy_fn = energy_fn
        self.n_molecules = n_molecules
        self.volume_scale_factor = volume_scale_factor

    def apply(
        self,
        state: IntegratorState,
        params: IntegratorParams,
    ) -> IntegratorState:
        new_step_count = state.step_count + 1
        
        should_attempt = (new_step_count % self.barostat_interval == 0)
        
        current_pos = state.position
        current_box = state.box
        if current_box is None:
            raise ValueError("MC Barostat requires a box dimension in state.")
        
        if params.molecule_indices is None:
            raise ValueError("MC Barostat requires molecule_indices in IntegratorParams.")
            
        molecule_indices = params.molecule_indices

        def attempt_barostat(state_tuple):
            current_pos, current_box, key = state_tuple
            
            current_energy = self.energy_fn(current_pos, current_box)
            
            key, subkey = jax.random.split(key)
            scale = 1.0 + self.volume_scale_factor * jax.random.uniform(subkey, minval=-1.0, maxval=1.0)
            
            # Box scale
            cbrt_scale = scale**(1.0/3.0)
            new_box = current_box * cbrt_scale
            
            # Scaled COM positions
            # Calculate COMs
            molecule_indices_flat = molecule_indices
            
            # Sum positions for each molecule
            molecule_sums = jax.ops.segment_sum(current_pos, molecule_indices_flat, num_segments=self.n_molecules)
            # Count atoms in each molecule
            molecule_counts = jax.ops.segment_sum(jnp.ones((current_pos.shape[0], 1)), molecule_indices_flat, num_segments=self.n_molecules)
            molecule_coms = molecule_sums / molecule_counts
            
            # New COMs
            new_coms = molecule_coms * cbrt_scale
            
            # New positions: atom_pos' = atom_pos - mol_com + new_mol_com
            new_pos = current_pos - molecule_coms[molecule_indices_flat] + new_coms[molecule_indices_flat]
            
            new_energy = self.energy_fn(new_pos, new_box)
            
            # Metropolis
            delta_U = new_energy - current_energy
            delta_V = jnp.prod(new_box) - jnp.prod(current_box)
            
            kT = params.kT
            beta = 1.0 / kT
            
            # NPT: ideal gas term uses N_molecules
            exponent = -beta * (delta_U + self.pressure * delta_V - self.n_molecules * kT * jnp.log(scale))
            
            key, subkey = jax.random.split(key)
            accept = jax.random.uniform(subkey) < jnp.exp(exponent)
            
            final_pos = jnp.where(accept, new_pos, current_pos)
            final_box = jnp.where(accept, new_box, current_box)
            return final_pos, final_box, key
            
        def skip_barostat(state_tuple):
            return state_tuple[0], state_tuple[1], state_tuple[2]
            
        final_pos, final_box, final_rng = jax.lax.cond(
            should_attempt,
            attempt_barostat,
            skip_barostat,
            (current_pos, current_box, state.rng)
        )
        
        return state.__replace__(
            position=final_pos,
            box=final_box,
            rng=final_rng,
            step_count=new_step_count
        )
