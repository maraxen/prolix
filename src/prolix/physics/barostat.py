from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from prolix.physics import explicit_corrections, pbc
from prolix.physics.step_base import Step
from prolix.physics.units import AKMA_PRESSURE_PER_BAR, BAR_PER_AKMA_PRESSURE
from prolix.typing import IntegratorParams, IntegratorState


def _mask_converter(x: object) -> tuple[bool, ...] | None:
    if x is None:
        return None
    if isinstance(x, tuple):
        return x
    return tuple(bool(v) for v in jnp.asarray(x).reshape(-1).tolist())


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
    atom_mask: tuple[bool, ...] | None = eqx.field(default=None, static=True, converter=_mask_converter)
    anisotropic: bool = eqx.field(static=True, default=False)
    barostat_mode: str = eqx.field(static=True, default="isotropic")

    def __init__(
        self,
        barostat_interval: int,
        pressure: float,
        energy_fn: Callable,
        n_molecules: int,
        volume_scale_factor: float = 0.05,
        atom_mask: jnp.ndarray | None = None,
        anisotropic: bool = False,
        barostat_mode: str = "isotropic",
    ):
        if anisotropic or str(barostat_mode).lower() != "isotropic":
            raise NotImplementedError(
                "Anisotropic / semi-isotropic barostat scaling is not supported; use barostat_mode='isotropic'."
            )
        self.barostat_interval = barostat_interval
        self.pressure = pressure
        self.energy_fn = energy_fn
        self.n_molecules = n_molecules
        self.volume_scale_factor = volume_scale_factor
        self.atom_mask = atom_mask
        self.anisotropic = False
        self.barostat_mode = "isotropic" 

    def apply(
        self,
        state: IntegratorState,
        params: IntegratorParams,
    ) -> IntegratorState:
        new_step_count = state.step_count + 1
        
        should_attempt = (new_step_count % self.barostat_interval == 0)
        
        current_pos = state.positions
        current_box = state.box
        if current_box is None:
            raise ValueError("MC Barostat requires a box dimension in state.")
        
        if params.molecule_indices is None:
            raise ValueError("MC Barostat requires molecule_indices in IntegratorParams.")
            
        molecule_indices = params.molecule_indices
        n_atoms = int(current_pos.shape[0])
        num_segments = max(int(self.n_molecules), n_atoms)

        def attempt_barostat(state_tuple):
            current_pos, current_box, key = state_tuple
            
            current_energy = self.energy_fn(current_pos, current_box)
            
            key, subkey = jax.random.split(key)
            scale = 1.0 + self.volume_scale_factor * jax.random.uniform(subkey, minval=-1.0, maxval=1.0)
            
            # Box scale
            cbrt_scale = scale**(1.0 / 3.0)
            new_box = pbc.isotropic_box_scale(current_box, cbrt_scale)
            
            # Pad-safe & PBC-unwrapped molecular COM scaling (CA-7 / CB-5)
            molecule_indices_flat = jnp.clip(molecule_indices, 0, num_segments - 1)
            if self.atom_mask is not None:
                mask_bool = jnp.asarray(self.atom_mask, dtype=jnp.bool_)
            elif getattr(params, "atom_mask", None) is not None:
                mask_bool = jnp.asarray(params.atom_mask, dtype=jnp.bool_)
            else:
                mask_bool = jnp.ones((n_atoms,), dtype=jnp.bool_)
            mask_col = mask_bool[:, None].astype(current_pos.dtype)

            # Anchor atom index per molecule (smallest real atom index in each molecule)
            atom_ids = jnp.where(mask_bool, jnp.arange(n_atoms, dtype=jnp.int32), n_atoms - 1)
            anchor_per_mol = jax.ops.segment_min(
                atom_ids, molecule_indices_flat, num_segments=num_segments
            )
            anchor_per_mol = jnp.clip(anchor_per_mol, 0, max(n_atoms - 1, 0))
            anchor_pos = current_pos[anchor_per_mol[molecule_indices_flat]]

            # Unwrap each atom relative to its molecule's anchor atom via minimum image
            rel_disp = pbc.minimum_image_displacement(current_pos, anchor_pos, current_box)
            unwrapped_pos = anchor_pos + rel_disp

            # Sum unwrapped positions and count real atoms for each molecule
            molecule_sums = jax.ops.segment_sum(
                unwrapped_pos * mask_col, molecule_indices_flat, num_segments=num_segments
            )
            molecule_counts = jax.ops.segment_sum(
                mask_col, molecule_indices_flat, num_segments=num_segments
            )
            safe_counts = jnp.maximum(molecule_counts, 1.0)
            molecule_coms = molecule_sums / safe_counts
            
            # Rigid-body translation of each molecule by (cbrt_scale - 1) * R_COM
            com_shift = (cbrt_scale - 1.0) * molecule_coms[molecule_indices_flat]
            new_pos = jnp.where(mask_bool[:, None], unwrapped_pos + com_shift, current_pos)
            
            new_energy = self.energy_fn(new_pos, new_box)
            
            # Metropolis
            delta_U = new_energy - current_energy
            delta_V = pbc.box_volume(new_box) - pbc.box_volume(current_box)
            
            kT = params.kT
            beta = 1.0 / kT
            
            # NPT: ideal gas term uses dynamic real molecule count when padded
            n_real_molecules = jnp.where(
                jnp.sum(molecule_counts > 0.5) > 0,
                jnp.sum(molecule_counts > 0.5).astype(current_pos.dtype),
                jnp.asarray(float(self.n_molecules), dtype=current_pos.dtype),
            )
            exponent = -beta * (
                delta_U + self.pressure * delta_V - n_real_molecules * kT * jnp.log(scale)
            )
            
            key, subkey = jax.random.split(key)
            accept = jax.random.uniform(subkey) < jnp.exp(exponent)
            
            final_pos = jnp.where(accept, new_pos, current_pos).astype(current_pos.dtype)
            final_box = jnp.where(accept, new_box, current_box).astype(current_box.dtype)
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
            positions=final_pos,
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
    atom_mask: tuple[bool, ...] | None = eqx.field(default=None, static=True, converter=_mask_converter)
    cutoff: float = eqx.field(static=True, default=9.0)
    anisotropic: bool = eqx.field(static=True, default=False)
    barostat_mode: str = eqx.field(static=True, default="isotropic")

    def __post_init__(self):
        if self.anisotropic or str(self.barostat_mode).lower() != "isotropic":
            raise NotImplementedError(
                "Anisotropic / semi-isotropic barostat scaling is not supported; use barostat_mode='isotropic'."
            )

    def apply(
        self,
        state: IntegratorState,
        params: IntegratorParams,
    ) -> IntegratorState:
        """Apply stochastic cell rescaling."""
        # 1. Physical Parameters
        dt = params.dt
        kT = params.kT
        tau = params.tau_barostat
        compressibility = params.compressibility
        target_pressure_bar = params.target_pressure_bar
        
        # AKMA pressure unit: kcal/mol/Å³
        target_pressure_akma = target_pressure_bar * AKMA_PRESSURE_PER_BAR
        
        n_atoms = state.positions.shape[0]
        if self.atom_mask is not None:
            mask_bool = jnp.asarray(self.atom_mask, dtype=jnp.bool_)
        elif getattr(params, "atom_mask", None) is not None:
            mask_bool = jnp.asarray(params.atom_mask, dtype=jnp.bool_)
        else:
            mask_bool = jnp.ones((n_atoms,), dtype=jnp.bool_)
        mask_col = mask_bool[:, None].astype(state.positions.dtype)

        # 2. Kinetic Energy (kcal/mol) — masked for real atoms
        safe_mass = jnp.where(mask_col > 0, jnp.maximum(state.mass, 1e-12), 1.0)
        velocity = (state.momentum * mask_col) / safe_mass
        ke_total = 0.5 * jnp.sum(safe_mass * (velocity**2) * mask_col)
        
        # 3. Virial Trace (kcal/mol)
        if params.positions_old is not None and params.positions_old.shape == state.positions.shape:
            r_for_virial = params.positions_old
        else:
            r_for_virial = state.positions
        virial = jnp.sum(r_for_virial * state.force * mask_col)
        
        # 4. Instantaneous Pressure (AKMA)
        volume = pbc.box_volume(state.box)
        
        # Optional tail corrections
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
        
        # 6. Apply scaling (box, positions, momenta — match monolithic settle_csvr_npt)
        new_box = pbc.isotropic_box_scale(state.box, mu)
        new_position = jnp.where(mask_bool[:, None], state.positions * mu, state.positions)
        new_momentum = jnp.where(mask_bool[:, None], state.momentum / mu, jnp.zeros_like(state.momentum))

        return state.__replace__(
            positions=new_position,
            momentum=new_momentum,
            box=new_box,
            rng=key,
        )
