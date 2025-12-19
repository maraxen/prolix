"""Replica Exchange Molecular Dynamics (Parallel Tempering)."""
from __future__ import annotations

import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax_md import quantity, simulate
from proxide.physics.constants import BOLTZMANN_KCAL

from prolix.physics import simulate as physics_simulate
from prolix.physics import system
from prolix.pt import temperature

if TYPE_CHECKING:
    from collections.abc import Callable

    from proxide.md import SystemParams

logger = logging.getLogger(__name__)

Array = Any

@dataclasses.dataclass
class ReplicaExchangeSpec:
    """Configuration for Replica Exchange MD."""

    n_replicas: int
    min_temp: float = 300.0
    max_temp: float = 400.0
    total_time_ns: float = 10.0 # Per replica? Or total? usually per replica.
    step_size_fs: float = 2.0
    exchange_interval_ps: float = 10.0 # Attempt swaps every X ps
    save_interval_ns: float = 0.01

    # Physics settings
    gamma: float = 1.0
    use_pbc: bool = False
    box: Array | None = None

    # Output
    save_path: str = "remd_trajectory.array_record"


class ReplicaExchangeState(eqx.Module):
    """State of all replicas."""

    # Stacked arrays of shape (n_replicas, ...)
    positions: Array
    velocities: Array
    forces: Array
    mass: Array

    # Scalar/1D arrays
    # Mapping of which replica index currently holds which temperature?
    # Usually: Replicas are distinct simulations. Temperatures are fixed to indices 0..N.
    # We swap coordinates.
    # So `positions[i]` is the configuration at temperature `temperatures[i]`.

    # We might want to track which original walker is where for analysis (optional)
    walker_indices: Array # (n_replicas,) int

    step: int | Array
    time_ns: float | Array

    # Energies (n_replicas,)
    potential_energy: Array
    kinetic_energy: Array

    # Exchange stats (should be Arrays for JIT loop)
    exchange_attempts: int | Array = 0
    exchange_successes: int | Array = 0

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def numpy(self) -> dict[str, Any]:
        """Convert to numpy dict."""
        return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_put(x, jax.devices("cpu")[0])), dataclasses.asdict(self))

def attempt_exchange(
    state: ReplicaExchangeState,
    temperatures: Array,
    key: Array,
    energy_fn: Callable[[Array], Array]
) -> tuple[ReplicaExchangeState, dict]:
    """Attempt swaps between adjacent replicas.

    We alternate between swapping (0,1), (2,3)... and (1,2), (3,4)...
    based on the step count or a random choice?
    Usually deterministic even/odd phases to ensure detailed balance over 2 steps
    or just random pairs.
    Standard: odd/even phases.

    Metropolis Criterion:
    Delta = (beta_i - beta_j) * (E_j - E_i)  <-- Wait, checking derivation.
    P(acc) = min(1, exp(Delta))

    Derivation:
    P(old) ~ exp(-beta_i E_i) * exp(-beta_j E_j)
    P(new) ~ exp(-beta_i E_j) * exp(-beta_j E_i)   <-- config j at temp i, config i at temp j
    Ratio = P(new)/P(old) = exp( -beta_i E_j - beta_j E_i + beta_i E_i + beta_j E_j )
                          = exp( -beta_i(E_j - E_i) - beta_j(E_i - E_j) )
                          = exp( beta_i(E_i - E_j) + beta_j(E_j - E_i) )
                          = exp( (beta_i - beta_j) * (E_i - E_j) )

    If Delta > 0, exp > 1, accept.
    """
    n = temperatures.shape[0]
    betas = 1.0 / (BOLTZMANN_KCAL * temperatures)

    # Calculate current Potential Energies if not fresh
    # E = state.potential_energy
    # But let's re-calculate to be safe or trust state?
    # Trust state if updated.
    E = state.potential_energy

    # Randomly choose odd or even phase
    key, subkey = random.split(key)
    is_odd = random.bernoulli(subkey, 0.5)

    # Define pairs
    # If even: (0,1), (2,3), ...
    # If odd: (1,2), (3,4), ...

    # Mask for `i` indices
    # Even: 0, 2, 4... up to n-2
    # Odd: 1, 3, 5... up to n-2

    # We can construct swap indices.

    def propose_swaps(phase_offset):
        # Indices i
        i_idxs = jnp.arange(phase_offset, n - 1, 2)
        j_idxs = i_idxs + 1

        # Calculate Delta
        beta_i = betas[i_idxs]
        beta_j = betas[j_idxs]
        E_i = E[i_idxs]
        E_j = E[j_idxs]

        delta = (beta_i - beta_j) * (E_i - E_j)

        # Acceptance
        log_prob = jnp.minimum(0.0, delta)
        prob = jnp.exp(log_prob)

        # Random numbers
        rnd_key, _ = random.split(key) # This might be reused if inside if/else?
        # Actually we need split inside.

        # Vectorized randoms
        rnds = random.uniform(rnd_key, (i_idxs.shape[0],))
        accept = rnds < prob

        return i_idxs, j_idxs, accept

    # We need to branch on phase_offset
    # Logic:
    # 1. Get pairs
    # 2. Compute accept mask
    # 3. Permute state

    # For JAX compatibility, it's easier to always compute both sets of indices but mask one?
    # Or use lax.cond

    i_even, j_even, acc_even = propose_swaps(0)
    i_odd, j_odd, acc_odd = propose_swaps(1)

    # Select which one to apply
    # We can't easily dynamically select size of arrays in lax.select without padding.
    # n is static, so shapes are static.
    # But shapes of even/odd pairs differ if n is odd.
    # e.g. n=3. Even: (0,1). Odd: (1,2). Size 1. Consistent.
    # n=4. Even: (0,1), (2,3). Size 2. Odd: (1,2). Size 1. Inconsistent.

    # So we should run only one branch.

    def perform_swap(i_idxs, j_idxs, accept):
        # Construct permutation
        # start with identity
        perm = jnp.arange(n)

        # Where accept is True, swap perm[i] and perm[j]
        # But i and j are disjoint pairs.
        # perm[i] = selected_j
        # perm[j] = selected_i

        # new_idx_for_slot_i = j if accept else i
        # new_idx_for_slot_j = i if accept else j

        # Use scatter update
        target_i = jnp.where(accept, j_idxs, i_idxs)
        target_j = jnp.where(accept, i_idxs, j_idxs)

        perm = perm.at[i_idxs].set(target_i)
        perm = perm.at[j_idxs].set(target_j)

        # Now shuffle state arrays
        new_pos = state.positions[perm]
        new_vel = state.velocities[perm]
        new_force = state.forces[perm]
        new_mass = state.mass[perm]
        new_walker = state.walker_indices[perm]
        new_PE = state.potential_energy[perm]
        new_KE = state.kinetic_energy[perm]

        return state.replace(
            positions=new_pos,
            velocities=new_vel,
            forces=new_force,
            mass=new_mass,
            walker_indices=new_walker,
            potential_energy=new_PE,
            kinetic_energy=new_KE,
            exchange_attempts=state.exchange_attempts + i_idxs.shape[0],
            exchange_successes=state.exchange_successes + jnp.sum(accept)
        )

    # Branch
    return jax.lax.cond(
        is_odd,
        lambda: perform_swap(i_odd, j_odd, acc_odd),
        lambda: perform_swap(i_even, j_even, acc_even)
    )


def run_replica_exchange(
    system_params: SystemParams,
    r_init: Array, # (N_atoms, 3) or (N_replicas, N_atoms, 3)
    spec: ReplicaExchangeSpec,
    key: Array | None = None
) -> ReplicaExchangeState:
    """Run Replica Exchange Simulation."""
    if key is None:
        key = random.PRNGKey(int(time.time()))

    n_replicas = spec.n_replicas
    temps = temperature.generate_temperature_ladder(n_replicas, spec.min_temp, spec.max_temp)

    # 1. Setup Energy
    if spec.use_pbc:
        if spec.box is None:
             msg = "Box required for PBC"
             raise ValueError(msg)
        from prolix.physics import pbc
        displacement_fn, shift_fn = pbc.create_periodic_space(spec.box)
        energy_fn = system.make_energy_fn(displacement_fn, system_params, use_pbc=True, box=spec.box)
    else:
        displacement_fn, shift_fn = quantity.space.free()
        energy_fn = system.make_energy_fn(displacement_fn, system_params, implicit_solvent=True)

    # 2. Initialize State
    # Broadcast r_init if single
    r_replicas = jnp.tile(r_init[None, ...], (n_replicas, 1, 1)) if r_init.ndim == 2 else r_init

    masses = system_params.get("masses", 1.0)
    if not isinstance(masses, float):
        masses = masses / 418.4

    # Broadcast mass to (N_replicas, N_atoms, 1) if needed, or just let vmap handle it?
    # State expects matching leading dims usually.
    # physics_simulate.NVTLangevinState wrapper usually handles broadcasting inside.
    # But here we want explicit stacked state.

    key, init_key = random.split(key)
    init_keys = random.split(init_key, n_replicas)

    # We need a vmapped initializer
    # Create valid Langevin integrator
    # We use rattle if needed
    dt = spec.step_size_fs * 1e-3
    kT_ref = BOLTZMANN_KCAL * 300.0 # dummy for init

    constraints = None
    constrained_bonds = system_params.get("constrained_bonds")
    lengths = system_params.get("constrained_bond_lengths")
    if constrained_bonds is not None and len(constrained_bonds) > 0:
        constraints = (constrained_bonds, lengths)
        init_fn, apply_fn = physics_simulate.rattle_langevin(
            energy_fn, shift_fn, dt=dt, kT=kT_ref, gamma=spec.gamma, mass=masses, constraints=constraints
        )
    else:
        init_fn, apply_fn = simulate.nvt_langevin(
            energy_fn, shift_fn, dt=dt, kT=kT_ref, gamma=spec.gamma, mass=masses
        )

    # Vmap the init
    # init_fn(key, R, kT=...)
    # Note: simulate.nvt_langevin init_fn takes 'kT' kwarg optional?
    # physics_simulate.rattle_langevin wrapper init_fn DOES take kwarg kT.
    # jax_md.simulate.nvt_langevin init_fn DOES take kwarg kT.

    jax.vmap(init_fn, in_axes=(0, 0, 0)) # keys, Rs, kTs
    kTs = temps * BOLTZMANN_KCAL

    # We need to broadcast kwargs? vmap doesn't map kwargs automatically.
    # We should wrap it.
    def init_wrapper(k, r, t, m):
        return init_fn(k, r, kT=t, mass=m)

    v_init_wrapper = jax.vmap(init_wrapper, in_axes=(0, 0, 0, None))

    sub_states = v_init_wrapper(init_keys, r_replicas, kTs, masses)

    # Convert to ReplicaExchangeState
    # sub_states is NVTLangevinState with stacked arrays
    E_init = jax.vmap(energy_fn)(sub_states.position)

    def compute_ke(p, m):
        return quantity.kinetic_energy(momentum=p, mass=m)

    K_init = jax.vmap(compute_ke)(sub_states.momentum, sub_states.mass)

    # helper for velocity calculation: ensure mass broadcasts
    m_init = sub_states.mass
    while m_init.ndim < sub_states.momentum.ndim:
        m_init = m_init[..., None]

    state = ReplicaExchangeState(
        positions=sub_states.position,
        velocities=sub_states.momentum / m_init,
        forces=sub_states.force,
        mass=m_init,
        walker_indices=jnp.arange(n_replicas),
        step=jnp.array(0, dtype=jnp.int32),
        time_ns=jnp.array(0.0),
        potential_energy=E_init,
        kinetic_energy=K_init,
        exchange_attempts=jnp.array(0),
        exchange_successes=jnp.array(0)
    )

    # 3. Main Loop
    steps_per_exchange = round(spec.exchange_interval_ps / (spec.step_size_fs * 1e-3))
    total_intervals = int(spec.total_time_ns * 1000 / spec.exchange_interval_ps)

    # Vmap apply
    # apply_fn(state, kT=...)
    def step_wrapper(s, t):
        return apply_fn(s, kT=t)

    v_step_fn = jax.vmap(step_wrapper, in_axes=(0, 0))

    # Scan loop
    def epoch_fn(carrier, _):
        curr_state, rng = carrier

        # 1. Run Dynamics
        # We need to reconstruct NVTLangevinState to pass to v_step_fn
        # The stored state is ReplicaExchangeState (positions, velocities...)
        # The integrator wants (position, momentum, force, mass, rng)

        # We need to split RNG per replica for dynamics
        rng, step_rng = random.split(rng)
        step_rngs = random.split(step_rng, n_replicas)

        langevin_state = physics_simulate.NVTLangevinState(
            position=curr_state.positions,
            momentum=curr_state.velocities * curr_state.mass,
            force=curr_state.forces,
            mass=curr_state.mass,
            rng=step_rngs
        )

        def inner_step(i, s):
            return v_step_fn(s, kTs)

        final_langevin = jax.lax.fori_loop(0, steps_per_exchange, inner_step, langevin_state)

        # Update ReplicaExchangeState
        PE = jax.vmap(energy_fn)(final_langevin.position)

        # Re-use compute_ke from outer scope or redefine
        def compute_ke_step(p, m):
            return quantity.kinetic_energy(momentum=p, mass=m)

        KE = jax.vmap(compute_ke_step)(final_langevin.momentum, final_langevin.mass)

        m_step = final_langevin.mass
        while m_step.ndim < final_langevin.momentum.ndim:
            m_step = m_step[..., None]

        updated_state = curr_state.replace(
            positions=final_langevin.position,
            velocities=final_langevin.momentum / m_step, # velocity
            forces=final_langevin.force,
            step=curr_state.step + steps_per_exchange,
            time_ns=curr_state.time_ns + (steps_per_exchange * dt * 1e-3), # ns
            potential_energy=PE,
            kinetic_energy=KE
        )

        # 2. Attempt Exchange
        rng, exchange_rng = random.split(rng)
        final_state = attempt_exchange(updated_state, temps, exchange_rng, energy_fn)

        # We might want to save trajectory here?
        # For scan, we return stacked

        return (final_state, rng), final_state # Return full state for trajectory


    # Run
    # Warning: JAX scan might be memory intensive if we save all frames.
    # Maybe use an outer python loop for saving like in simulate.py

    # For now, let's just run.
    # Note: `run_replica_exchange` is expected to manage IO too based on spec.

    if spec.save_path:
        # TODO: Writer setup
        pass

    curr_carrier = (state, key)

    # Just Python loop for now to be safe with memory and IO
    logger.info("Starting REMD: %d replicas, %d intervals", n_replicas, total_intervals)

    @jax.jit
    def run_interval(carrier):
        return epoch_fn(carrier, None)

    for i in range(total_intervals):
        (state, key), _snapshot = run_interval(curr_carrier)

        # IO
        # We can write `snapshot` (ReplicaExchangeState) to disk
        # We need a Writer that handles ReplicaExchangeState or just flatten it?
        # Maybe just positions [n_replicas, n_atoms, 3] + walker_indices

        curr_carrier = (state, key)

        if i % 10 == 0:
            jax.block_until_ready(state.positions)
            frac = state.exchange_successes / (state.exchange_attempts + 1e-8)
            logger.info("Interval %d/%d: Acc Rate %.2f", i, total_intervals, frac)

    return state
