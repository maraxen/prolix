"""Simulation runner for minimization and thermalization."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random
from jax_md import quantity, simulate, space, util
from proxide.physics.constants import BOLTZMANN_KCAL

from prolix.physics import system

if TYPE_CHECKING:
  from collections.abc import Callable

  from proxide.core.containers import Protein
  from prolix.typing import SystemParams

Array = util.Array


@dataclass
class SimulationSpec:
  """Configuration for running a simulation."""

  total_time_ns: float
  step_size_fs: float = 2.0
  ensemble: str = "nvt_langevin"  # "nve", "nvt_nose_hoover", "nvt_langevin", "brownian"
  temperature: float = 300.0
  pressure: float | None = None  # Future support for NPT
  gamma: float = 1.0  # Friction coefficient (1/ps) for Langevin
  tau: float = 0.5  # Time constant (ps) for Nose-Hoover
  chain_length: int = 5  # Nose-Hoover chain length
  chain_steps: int = 1  # Nose-Hoover chain steps
  save_interval_ns: float = 0.001
  accumulate_steps: int = 500
  save_path: str = "trajectory.array_record"
  use_shake: bool = False

  @property
  def dt(self) -> float:
    """Time step in picoseconds."""
    return self.step_size_fs / 1000.0

  @property
  def steps(self) -> int:
    """Total number of steps."""
    return int(self.total_time_ns * 1000.0 / self.dt)


from prolix.typing import NPTState, NVTLangevinState


def rattle_langevin(

  energy_or_force_fn: Callable[..., Array],
  shift_fn: Callable[..., Array],
  dt: float,
  kT: float,
  gamma: float = 0.1,
  mass: float | Array = 1.0,
  constraints: tuple[Array, Array] | None = None,
):
  """Langevin dynamics with RATTLE constraints."""
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  def init_fn(key, R, mass=mass, **kwargs):
    _kT = kwargs.pop("kT", kT)
    key, split = random.split(key)
    force = force_fn(R, **kwargs)

    # Handle mass - keep as 1D for state, expand for broadcasting in calculations
    mass_arr = jnp.array(mass, dtype=R.dtype)
    if mass_arr.ndim == 0:
      mass_arr = jnp.ones((R.shape[0],), dtype=R.dtype) * mass_arr

    # Initialize momenta manually to avoid jax_md shape issues
    # p = sqrt(m * kT) * N(0,1)
    # Use mass_arr[:, None] for broadcasting with shape (N, 3)
    momenta = jnp.sqrt(mass_arr[:, jnp.newaxis] * _kT) * random.normal(
      split, R.shape, dtype=R.dtype
    )

    # Store mass as (N, 1) for broadcasting in apply_fn
    mass_for_state = mass_arr[:, jnp.newaxis]

    return NVTLangevinState(R, momenta, force, mass_for_state, key)

  def apply_fn(state, **kwargs):
    _dt = kwargs.pop("dt", dt)
    _kT = kwargs.pop("kT", kT)

    # Velocity Update 1
    momentum = state.momentum + 0.5 * _dt * state.force

    # Position Update 1
    velocity = momentum / state.mass
    position = shift_fn(state.position, 0.5 * _dt * velocity)

    # Stochastic Update
    c1 = jnp.exp(-gamma * _dt)
    c2 = jnp.sqrt(1 - c1**2)

    key, split = random.split(state.key)
    noise = random.normal(split, state.momentum.shape)
    momentum = c1 * momentum + c2 * jnp.sqrt(state.mass * _kT) * noise

    # Position Update 2
    velocity = momentum / state.mass
    position = shift_fn(position, 0.5 * _dt * velocity)

    # --- SHAKE (Position Constraint) ---
    if constraints is not None:
      pairs, lengths = constraints
      from prolix.physics.constraints import project_positions, project_momenta

      position = project_positions(position, pairs, lengths, state.mass, shift_fn)

    # Force Update
    force = force_fn(position, **kwargs)

    # Velocity Update 2
    momentum = momentum + 0.5 * _dt * force

    # --- RATTLE (Velocity Constraint) ---
    if constraints is not None:
      pairs, lengths = constraints
      from prolix.physics.constraints import project_momenta
      # Project momentum to be orthogonal to bonds
      momentum = project_momenta(momentum, position, pairs, state.mass, shift_fn)

    return NVTLangevinState(position, momentum, force, state.mass, key)

  return init_fn, apply_fn


def run_minimization(
  energy_fn: Callable[[Array], Array],
  initial_positions: Array,
  steps: int = 500,
  dt_start: float = 2e-3,  # 2 fs - typical for MD
  dt_max: float = 4e-3,  # 4 fs max
  max_displacement_per_step: float = 0.1,  # Maximum Å per step (prevents explosion)
  sd_steps: int = 200,  # Steepest descent pre-conditioning steps
) -> Array:
  """Run energy minimization using robust multi-stage approach.

  For high-energy structures with large forces, a simple FIRE minimization can
  cause positions to explode. This function uses:
  1. Steepest descent pre-conditioning with adaptive step size
  2. FIRE descent for final optimization

  FULLY JIT-COMPATIBLE - no eager Python conversions.

  Args:
      energy_fn: Energy function E(R).
      initial_positions: Initial positions (N, 3).
      steps: Number of FIRE minimization steps.
      dt_start: Initial time step for FIRE.
      dt_max: Maximum time step for FIRE.
      max_displacement_per_step: Maximum atomic displacement per step (Å).
      sd_steps: Steepest descent pre-conditioning steps.

  Returns:
      Minimized positions.

  """
  # Steepest descent with backtracking line search — robust against sharp LJ walls
  # (FIRE is avoided: its velocity accumulation diverges on 1-4 exception pairs at close range)

  @jax.jit
  def sd_backtrack_step(r, _):
    E0, g = jax.value_and_grad(energy_fn)(r)
    forces = -g
    max_force = jnp.max(jnp.linalg.norm(forces, axis=-1)) + 1e-8
    dt = jnp.minimum(max_displacement_per_step / max_force, dt_max)

    r1 = r + 1.00 * dt * forces;  E1 = energy_fn(r1)
    r2 = r + 0.50 * dt * forces;  E2 = energy_fn(r2)
    r3 = r + 0.25 * dt * forces;  E3 = energy_fn(r3)
    r4 = r + 0.10 * dt * forces;  E4 = energy_fn(r4)

    best_r, best_E = r4, E4
    best_r = jnp.where(E3 < best_E, r3, best_r); best_E = jnp.where(E3 < best_E, E3, best_E)
    best_r = jnp.where(E2 < best_E, r2, best_r); best_E = jnp.where(E2 < best_E, E2, best_E)
    best_r = jnp.where(E1 < best_E, r1, best_r); best_E = jnp.where(E1 < best_E, E1, best_E)

    return jnp.where(best_E < E0, best_r, r), max_force

  r_final, _ = jax.lax.scan(sd_backtrack_step, initial_positions, jnp.arange(sd_steps))
  jax.block_until_ready(r_final)
  return r_final


def run_thermalization(
  energy_fn: Callable[[Array], Array],
  initial_positions: Array,
  temperature: float = 300.0,
  steps: int = 1000,
  dt: float = 2e-3,  # 2 fs - standard with SHAKE
  gamma: float = 1.0,  # 1.0/ps - standard coupling
  mass: Array | float = 1.0,
  use_shake: bool = False,
  constraints: tuple[Array, Array] | None = None,
  key: Array | None = None,
) -> Array:
  """Run NVT thermalization using Langevin dynamics.

  Args:
      energy_fn: Energy function E(R).
      initial_positions: Initial positions (N, 3).
      temperature: Temperature in Kelvin.
      steps: Number of simulation steps.
      dt: Time step (ps).
      gamma: Friction coefficient (1/ps).
      mass: Particle mass(es).
      use_shake: Whether to use RATTLE constraints.
      constraints: (pairs, lengths) for RATTLE.
      key: PRNG key.

  Returns:
      Final positions.

  """
  if key is None:
    key = jax.random.PRNGKey(0)

  kT = BOLTZMANN_KCAL * temperature

  if use_shake and constraints is not None:
    init_fn, apply_fn = rattle_langevin(
      energy_fn,
      shift_fn=space.free()[1],
      dt=dt,
      kT=kT,
      gamma=gamma,
      mass=mass,
      constraints=constraints,
    )
  else:
    init_fn, apply_fn = simulate.nvt_langevin(
      energy_fn, shift_fn=space.free()[1], dt=dt, kT=kT, gamma=gamma, mass=mass
    )

  state = init_fn(key, initial_positions)

  @jax.jit
  def step_fn(i, state):
    return apply_fn(state)

  state = jax.lax.fori_loop(0, steps, step_fn, state)
  return state.position


def run_nve(
  energy_fn: Callable[[Array], Array],
  initial_positions: Array,
  steps: int,
  dt: float = 2e-3,
  mass: Array | float = 1.0,
  use_shake: bool = False,
  constraints: tuple[Array, Array] | None = None,
) -> Array:
  """Run NVE simulation using Velocity Verlet (or RATTLE if constrained)."""
  if use_shake and constraints is not None:
    # Reuse rattle_langevin with gamma=0 (effectively NVE with constraints)
    init_fn, apply_fn = rattle_langevin(
      energy_fn,
      shift_fn=space.free()[1],
      dt=dt,
      kT=0.0,
      gamma=0.0,
      mass=mass,
      constraints=constraints,
    )
    key = jax.random.PRNGKey(0)
    state = init_fn(key, initial_positions, kT=0.0)
  else:
    init_fn, apply_fn = simulate.nve(energy_fn, shift_fn=space.free()[1], dt=dt)
    kT = BOLTZMANN_KCAL * 300.0  # Default assumption if not passed
    key = jax.random.PRNGKey(0)
    state = init_fn(key, initial_positions, mass=mass, kT=kT)

  @jax.jit
  def step_fn(i, state):
    return apply_fn(state)

  state = jax.lax.fori_loop(0, steps, step_fn, state)
  return state.position


def run_nvt_nose_hoover(
  energy_fn: Callable[[Array], Array],
  initial_positions: Array,
  steps: int,
  dt: float = 2e-3,
  temperature: float = 300.0,
  tau: float = 0.5,
  chain_length: int = 5,
  chain_steps: int = 1,
  mass: Array | float = 1.0,
) -> Array:
  """Run NVT simulation using Nose-Hoover chain."""
  kT = BOLTZMANN_KCAL * temperature

  init_fn, apply_fn = simulate.nvt_nose_hoover(
    energy_fn,
    shift_fn=space.free()[1],
    dt=dt,
    kT=kT,
    tau=tau,
    chain_length=chain_length,
    chain_steps=chain_steps,
  )

  key = jax.random.PRNGKey(0)
  state = init_fn(key, initial_positions, mass=mass)

  @jax.jit
  def step_fn(i, state):
    return apply_fn(state)

  state = jax.lax.fori_loop(0, steps, step_fn, state)
  return state.positions


def run_brownian(
  energy_fn: Callable[[Array], Array],
  initial_positions: Array,
  steps: int,
  dt: float = 2e-3,
  temperature: float = 300.0,
  gamma: float = 0.1,
  mass: Array | float = 1.0,
) -> Array:
  """Run Brownian dynamics simulation."""
  kT = BOLTZMANN_KCAL * temperature

  # jax_md.simulate.brownian(energy_or_force, shift, dt, kT, gamma=0.1)
  init_fn, apply_fn = simulate.brownian(energy_fn, shift=space.free()[1], dt=dt, kT=kT, gamma=gamma)

  key = jax.random.PRNGKey(0)
  state = init_fn(key, initial_positions)

  @jax.jit
  def step_fn(i, state):
    return apply_fn(state)

  state = jax.lax.fori_loop(0, steps, step_fn, state)
  return state.positions


def run_simulation(
  system_params: SystemParams | Protein,
  initial_positions: Array,
  temperature: float = 300.0,
  min_steps: int = 500,
  therm_steps: int = 1000,
  dielectric_constant: float = 1.0,
  implicit_solvent: bool = True,
  solvent_dielectric: float = 78.5,
  solute_dielectric: float = 1.0,
  key: Array | None = None,
  sim_spec: SimulationSpec | None = None,
) -> Array:
  """Run full simulation: Minimization -> Thermalization -> Production.

  Args:
      system_params: System parameters.
      initial_positions: Initial positions.
      temperature: Temperature in Kelvin.
      min_steps: Minimization steps.
      therm_steps: Thermalization steps.
      dielectric_constant: Dielectric constant for screened Coulomb (if implicit_solvent=False).
      implicit_solvent: Whether to use Generalized Born implicit solvent.
      solvent_dielectric: Solvent dielectric for GB.
      solute_dielectric: Solute dielectric for GB.
      key: PRNG key.
      sim_spec: Simulation configuration specification (optional).

  Returns:
      Final positions.

  """
  dt_production = 2e-3  # Default
  steps_production = 0

  # Resolve Configuration
  if sim_spec is not None:
    # Use spec values for production defaults
    temperature = sim_spec.temperature
    dt_production = sim_spec.dt
    steps_production = sim_spec.steps
    # Note: min/therm steps might not be in spec efficiently yet,
    # but we use args for those stages or defaults.
 
  # Handle Migration Shim (P1.3)
  if not isinstance(system_params, Protein):
    from prolix.compat import system_params_to_protein
    protein = system_params_to_protein(system_params)
  else:
    protein = system_params

  displacement_fn, _ = space.free()
  energy_fn = system.make_energy_fn(
    displacement_fn,
    protein,
    dielectric_constant=dielectric_constant,
    implicit_solvent=implicit_solvent,
    solvent_dielectric=solvent_dielectric,
    solute_dielectric=solute_dielectric,
  )

  # 1. Minimize
  positions_minimized = run_minimization(energy_fn, initial_positions, steps=min_steps)

  # Extract masses if available, else default to 1.0
  # Convert amu to internal units (kcal/mol/A^2 * ps^2)^-1 ?
  # We want a = F/m in A/ps^2.
  # F in kcal/mol/A. m in amu.
  # Conversion factor is 418.4.
  # m_internal = m_amu / 418.4
  masses_raw = protein.masses
  if masses_raw is not None:
    masses = jnp.asarray(masses_raw) / 418.4
  else:
    masses = 1.0

  # Extract constraints
  constrained_bonds = protein.constrained_bonds
  constrained_lengths = protein.constrained_bond_lengths

  use_shake = False
  constraints = None
  if constrained_bonds is not None and constrained_lengths is not None:
    if len(constrained_bonds) > 0:
      use_shake = True
      constraints = (jnp.asarray(constrained_bonds), jnp.asarray(constrained_lengths))

  # 2. Thermalize
  # Always use NVT Langevin for thermalization for stability
  r_therm = run_thermalization(
    energy_fn,
    positions_minimized,
    temperature=temperature,
    steps=therm_steps,
    mass=masses,
    use_shake=use_shake,
    constraints=constraints,
    key=key,
  )

  # 3. Production (if spec provided)
  if sim_spec is not None and steps_production > 0:
    if sim_spec.ensemble == "nve":
      return run_nve(
        energy_fn,
        r_therm,
        steps_production,
        dt=dt_production,
        mass=masses,
        use_shake=sim_spec.use_shake or use_shake,  # Use spec shake or system implied shake
        constraints=constraints,
      )
    if sim_spec.ensemble == "nvt_nose_hoover":
      return run_nvt_nose_hoover(
        energy_fn,
        r_therm,
        steps_production,
        dt=dt_production,
        temperature=temperature,
        tau=sim_spec.tau,
        chain_length=sim_spec.chain_length,
        chain_steps=sim_spec.chain_steps,
        mass=masses,
      )
    if sim_spec.ensemble == "nvt_langevin":
      # Use existing thermalization/langevin runner but purely for production steps
      return run_thermalization(
        energy_fn,
        r_therm,
        temperature=temperature,
        steps=steps_production,
        dt=dt_production,
        gamma=sim_spec.gamma,
        mass=masses,
        use_shake=sim_spec.use_shake or use_shake,
        constraints=constraints,
        key=key,  # Re-use key? Ideally split.
      )
    if sim_spec.ensemble == "brownian":
      return run_brownian(
        energy_fn,
        r_therm,
        steps_production,
        dt=dt_production,
        temperature=temperature,
        gamma=sim_spec.gamma,
        mass=masses,
      )
    msg = f"Unknown ensemble: {sim_spec.ensemble}"
    raise ValueError(msg)

  return r_therm
