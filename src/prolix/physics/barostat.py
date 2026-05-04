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

class SCR_Barostat_Step(Step):
    """Stochastic Cell Rescaling barostat step (Bernetti & Bussi 2020).

    Modular implementation of the SCR barostat for NPT ensemble.
    Scales box and positions to maintain constant pressure via a 
    Langevin-like update on the log-volume.

    Attributes:
        mu_min: Safety clamp for the scaling factor mu (default 0.99).
        atom_mask: Optional mask for real atoms (for tail corrections).
        cutoff: Cutoff distance for tail corrections.
    """
    mu_min: float = eqx.field(static=True, default=0.99)
    atom_mask: Optional[jnp.ndarray] = eqx.field(static=True, default=None)
    cutoff: float = eqx.field(static=True, default=9.0)

    def apply(
        self,
        state: IntegratorState,
        params: IntegratorParams,
    ) -> IntegratorState:
        """Apply stochastic cell rescaling.

        Args:
            state: IntegratorState with position, momentum, force, box, and RNG.
            params: IntegratorParams with dt, kT, target_pressure_bar, compressibility, 
                    tau_barostat, and energy_params.

        Returns:
            Updated IntegratorState with scaled box and positions.
        """
        # 1. Physical Parameters
        dt = params.dt
        kT = params.kT
        tau = params.tau_barostat
        compressibility = params.compressibility
        target_pressure_bar = params.target_pressure_bar
        
        # AKMA pressure unit: kcal/mol/Å³
        from prolix.physics.units import AKMA_PRESSURE_PER_BAR
        target_pressure_akma = target_pressure_bar * AKMA_PRESSURE_PER_BAR
        
        # 2. Kinetic Energy (kcal/mol)
        velocity = state.momentum / state.mass
        ke_total = 0.5 * jnp.sum(state.mass * velocity**2)
        
        # 3. Virial Trace (kcal/mol)
        # Use positions_old if available (consistent with force eval), else current positions
        r_for_virial = jnp.where(params.positions_old is not None, params.positions_old, state.position)
        virial = jnp.sum(r_for_virial * state.force)
        
        # 4. Instantaneous Pressure (AKMA)
        volume = state.box[0] * state.box[1] * state.box[2]
        
        # Optional tail corrections
        from prolix.physics import explicit_corrections
        sigmas = jnp.maximum(params.energy_params.sigmas, 1e-6)
        epsilons = params.energy_params.epsilons
        
        if self.atom_mask is not None:
            p_tail = explicit_corrections.lj_dispersion_tail_pressure(
                state.box, sigmas, epsilons, self.cutoff, self.atom_mask
            )
            p_imp = explicit_corrections.lj_dispersion_tail_impulsive_pressure(
                state.box, sigmas, epsilons, self.cutoff, self.atom_mask
            )
        else:
            p_tail = 0.0
            p_imp = 0.0
            
        total_virial = virial + (p_tail + p_imp) * volume * 3.0
        pressure = (2.0 * ke_total + total_virial) / (3.0 * volume)
        
        # SCR Stochastic Dynamics
        # dε = (dt/τ) * β * (P - P_0) + sqrt(2*kT*β*dt / (τ*V)) * noise
        pressure_deviation = pressure - target_pressure_akma
        
        # Convert compressibility from bar^-1 to AKMA^-1
        from prolix.physics.units import BAR_PER_AKMA_PRESSURE
        compressibility_akma = compressibility * BAR_PER_AKMA_PRESSURE
        
        key, split = jax.random.split(state.rng)
        random_noise = jax.random.normal(split, dtype=state.box.dtype)
        
        depsilon_det = (dt / tau) * compressibility_akma * pressure_deviation
        depsilon_stoch = jnp.sqrt(2.0 * kT * compressibility_akma * dt / (tau * volume)) * random_noise
        depsilon = depsilon_det + depsilon_stoch
        
        # mu = exp(dε/3)
        mu = jnp.exp(depsilon / 3.0)
        
        # Safety clamp: μ ∈ [μ_min, 1/μ_min]
        mu_max = 1.0 / self.mu_min
        mu = jnp.clip(mu, self.mu_min, mu_max)
        
        # 6. Apply Scaling
        new_box = state.box * mu
        new_position = state.position * mu
        
        return state.__replace__(
            position=new_position,
            box=new_box,
            rng=key
        )
