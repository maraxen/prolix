"""Integrator builder: factory for composing step sequences into working integrators.

This module implements Phase 2.1 of ADR-005: the composition layer that converts
step sequences (from step_system.py) into actual integrator functions. Phase 4
extends this with batched integrators via vmap for ensemble simulations.

**Architecture**:

The make_integrator factory is the critical composition layer:

1. Takes sequence name + parameters
2. Eagerly initializes state (forces, chain_state, barostat_state)
3. Returns (init_fn, apply_fn) tuple for time-stepping

**Batching** (Phase 4):

The make_integrator_batched factory creates batched integrators:

1. Wraps unbatched integrator with jax.vmap over batch dimension
2. Enables parallel simulation of B independent trajectories
3. Shares mass, box, water_indices across batch; trajectories are independent
4. Numerically equivalent to looping unbatched integrator (machine-epsilon equivalence)

**Design Principles**:

- **Eager initialization**: All state (forces, chain state, barostat state) computed
  in init_fn, not lazily during integration
- **Step composition**: apply_fn loops over step sequence; each step receives full
  state and returns modified state
- **RNG management**: Split key between stochastic steps (O_Step, CSVR_Step)
- **Batching transparency**: vmap(apply_fn) produces identical results to looping
- **No breaking changes**: settle_langevin remains available as thin wrapper

**Reference**: ADR-005 v2.0, Phase 2.1; Phase 4 Batching Support

**Author**: Fixer Agent (Phase 2.1 implementation, Phase 4 batching extension)
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array as ArrayType, PRNGKeyArray

from prolix.physics.constraints import ConstraintDOFMask
from prolix.physics.step_system import (
    Force_Step,
    IntegratorState,
    Step,
    StepSequence,
    make_step,
    step_sequences,
)
from prolix.typing import IntegratorParams, EnergyParams
from prolix.physics.virtual_sites_step import redistribute_forces
from prolix.typing import Array, WaterIndicesArray


def _compute_forces(
    positions: ArrayType,
    box: Optional[ArrayType],
    energy_fn: Callable,
    energy_params: Any = None,
    vs_def: Optional[ArrayType] = None,
    vs_params: Optional[ArrayType] = None,
) -> ArrayType:
  r"""Compute forces via autodiff of energy function.

  Args:
      positions: (N, 3) atomic positions.
      box: Optional (3,) box dimensions for periodic boundary.
      energy_fn: Energy function(positions, box, params) → scalar energy.
      energy_params: Optional parameters for energy_fn.
      vs_def: Optional virtual site definitions.
      vs_params: Optional virtual site parameters.

  Returns:
      (N, 3) force array F = -∇E, with virtual site forces redistributed.
  """
  def energy_wrapper(R):
    if energy_params is not None:
      return energy_fn(R, box, energy_params) if box is not None else energy_fn(R, energy_params)
    return energy_fn(R, box) if box is not None else energy_fn(R)

  forces = -jax.grad(energy_wrapper)(positions)

  if vs_def is not None and vs_params is not None:
    forces = redistribute_forces(forces, vs_def, vs_params)

  return forces


def _initialize_constraint_dofs(
    water_indices: Optional[WaterIndicesArray],
    n_atoms: int,
) -> Optional[ConstraintDOFMask]:
  """Initialize constraint DOF mask if water indices provided.

  Args:
      water_indices: Optional (n_waters, 3) array of [O, H1, H2] indices.
      n_atoms: Total number of atoms in system.

  Returns:
      ConstraintDOFMask if water_indices provided, else None.
  """
  if water_indices is None:
    return None
  return ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)


def make_integrator_batched(
    energy_fn: Callable,
    shift_fn: Callable,
    mass: Array,
    batch_size: int = 1,
    sequence_name: str = "baoab_langevin",
    dt: float = 0.5,
    kT: float = 1.0,
    gamma: float = 1.0,
    water_indices: Optional[Array] = None,
    target_pressure_bar: Optional[float] = None,
    tau_barostat_akma: Optional[float] = 2000.0,
    tau_thermostat_akma: Optional[float] = 2000.0,
    **kwargs,
) -> Tuple[Callable, Callable]:
  r"""Create a batched integrator via vmap composition over independent trajectories.

  This factory creates a batched integrator where multiple independent trajectories
  (shape [B, N, 3] for positions, B independent RNG keys, etc.) are integrated
  in parallel via JAX vmap.

  **Design**:
  - First creates unbatched integrator via make_integrator
  - Wraps apply_fn with jax.vmap over batch dimension
  - init_fn_batched handles batch initialization with per-trajectory RNG keys
  - All batch elements share mass, water_indices, energy_fn parameters

  **Batching Strategy**:
  - Batched fields: position [B, N, 3], momentum [B, N, 3], force [B, N, 3], rng [B, 2]
  - Shared fields: mass [N], water_indices [n_waters, 3], box [3]
  - Each trajectory evolves independently; no inter-batch communication

  Args:
      energy_fn: Callable(positions, box) → energy (scalar).
      shift_fn: Callable(positions, box) → positions_shifted.
      mass: Mass array [n_atoms] (shared across batch).
      batch_size: Number of independent trajectories (B).
      sequence_name: Name in step_sequences registry (default: 'baoab_langevin').
      dt: Timestep in AKMA units (default: 0.5).
      kT: Temperature in kT (default: 1.0).
      gamma: Langevin friction coefficient (default: 1.0).
      water_indices: Optional (n_waters, 3) array of [O, H1, H2] indices.
      target_pressure_bar: Target pressure in bar (required for NPT).
      tau_barostat_akma: Barostat time constant (default: 2000.0).
      tau_thermostat_akma: Thermostat time constant (default: 2000.0).
      **kwargs: Additional step-specific parameters.

  Returns:
      (init_fn_batched, apply_fn_batched): Initialization and integration functions.
          - init_fn_batched(key, positions_batch, box=None) → IntegratorState
          - apply_fn_batched(state) → IntegratorState

  Example:
      >>> init_fn_batched, apply_fn_batched = make_integrator_batched(
      ...     energy_fn, shift_fn,
      ...     batch_size=16,
      ...     sequence_name='baoab_langevin',
      ...     dt=0.5, kT=2.479, gamma=1.0,
      ...     mass=masses,
      ...     water_indices=water_indices
      ... )
      >>> key = jax.random.PRNGKey(0)
      >>> keys = jax.random.split(key, 16)
      >>> positions_batch = jnp.stack([perturb_trajectory(positions, k) for k in keys])
      >>> state_batch = init_fn_batched(keys[0], positions_batch, box=box_vec)
      >>> for step in range(100):
      ...     state_batch = apply_fn_batched(state_batch)
  """
  # Create unbatched integrator first
  init_fn_unbatched, apply_fn_unbatched = make_integrator(
      energy_fn=energy_fn,
      shift_fn=shift_fn,
      mass=mass,
      sequence_name=sequence_name,
      dt=dt,
      kT=kT,
      gamma=gamma,
      water_indices=water_indices,
      target_pressure_bar=target_pressure_bar,
      tau_barostat_akma=tau_barostat_akma,
      tau_thermostat_akma=tau_thermostat_akma,
      **kwargs,
  )

  def init_fn_batched(
      key: PRNGKeyArray,
      positions_batch: ArrayType,
      box: Optional[ArrayType] = None,
  ) -> IntegratorState:
    r"""Initialize batched integrator state for B independent trajectories.

    Args:
        key: JAX PRNGKey.
        positions_batch: (B, N, 3) batch of initial positions.
        box: Optional (3,) box dimensions (shared across batch).

    Returns:
        IntegratorState with batched fields: position [B, N, 3], momentum [B, N, 3],
        force [B, N, 3], rng [B, 2], and shared fields mass [N], box [3].
    """
    B = positions_batch.shape[0]

    # Split RNG key into B independent keys for each trajectory
    keys_batch = jax.random.split(key, B)

    # Initialize each trajectory independently via vmap
    def init_single(key_i, pos_i):
      return init_fn_unbatched(key_i, pos_i, box=box)

  # vmap over batch dimension with explicit out_axes structure
    # We want to batch position, momentum, force, rng, and step_count (all at axis 0)
    # but keep mass and box shared (not batched at all)
    out_axes_spec = IntegratorState(
        positions=0,      # batch at axis 0
        momentum=0,      # batch at axis 0
        force=0,         # batch at axis 0
        mass=None,       # do NOT batch (broadcast)
        rng=0,           # batch at axis 0
        box=None,        # do NOT batch (broadcast)
        step_count=0,    # batch at axis 0
    )

    state_list = jax.vmap(init_single, in_axes=(0, 0), out_axes=out_axes_spec)(
        keys_batch, positions_batch
    )

    return state_list

  # Create batched apply_fn via vmap
  def apply_fn_batched(state_batch: IntegratorState) -> IntegratorState:
    r"""Apply one timestep of batched integrator on all B trajectories.

    Args:
        state_batch: IntegratorState with batched dimensions [B, ...].

    Returns:
        Updated IntegratorState (still batched, axis 0 preserved).
    """
    # vmap over the batch dimension (axis 0) for batched fields
    # IntegratorState has fields: position, momentum, force, mass, rng, box, step_count
    # Flattened order: (position, momentum, force, mass, rng, box, step_count)
    in_axes_spec = (0, 0, 0, None, 0, None, 0)
    out_axes_spec = (0, 0, 0, None, 0, None, 0)

    # Create a wrapper that takes flattened arguments
    def apply_fn_flat(position, momentum, force, mass, rng, box, step_count):
      state = IntegratorState(positions=position, momentum=momentum, force=force,
                             mass=mass, rng=rng, box=box, step_count=step_count)
      result = apply_fn_unbatched(state)
      return (result.positions, result.momentum, result.force, result.mass,
              result.rng, result.box, result.step_count)

    # Apply vmap over the flattened function
    apply_fn_flat_vmapped = jax.vmap(apply_fn_flat, in_axes=in_axes_spec, out_axes=out_axes_spec)

    # Call with flattened state
    pos_out, mom_out, force_out, mass_out, rng_out, box_out, step_out = apply_fn_flat_vmapped(
        state_batch.positions, state_batch.momentum, state_batch.force,
        state_batch.mass, state_batch.rng, state_batch.box, state_batch.step_count
    )

    # Reconstruct state
    return IntegratorState(
        positions=pos_out,
        momentum=mom_out,
        force=force_out,
        mass=mass_out,
        rng=rng_out,
        box=box_out,
        step_count=step_out,
    )

  return init_fn_batched, apply_fn_batched


def make_integrator(
    energy_fn: Callable,
    shift_fn: Callable,
    mass: Array,
    sequence_name: str = "baoab_langevin",
    dt: float = 0.5,
    kT: float = 1.0,
    gamma: float = 1.0,
    water_indices: Optional[Array] = None,
    target_pressure_bar: Optional[float] = None,
    tau_barostat_akma: Optional[float] = 2000.0,
    tau_thermostat_akma: Optional[float] = 2000.0,
    energy_params: Any = None,
    vs_def: Optional[Array] = None,
    vs_params: Optional[Array] = None,
    n_molecules: Optional[int] = None,
    **kwargs,
) -> Tuple[Callable, Callable]:
  r"""Create a modular integrator from a sequence registry entry.

  This is the primary factory function for Phase 2.1. It takes a sequence name
  (e.g., 'baoab_langevin') and returns a tuple (init_fn, apply_fn) that fully
  implements time-stepping with eager state initialization.

  **Eager Initialization**: All state (forces, chain_state, barostat_state) is
  computed once in init_fn and stored in the returned IntegratorState. This
  allows apply_fn to be pure and JIT-compatible.

  **Step Composition**: apply_fn loops through the step sequence. Each step
  receives the current state and returns an updated state. Some steps (O_Step,
  CSVR_Step) may return (state, new_key) tuples; apply_fn handles both cases.

  **RNG Management**: The RNG key is split and propagated through stochastic
  steps. A fresh key is used for each step that requires randomness.

  **Force Recomputation**: Forces are recomputed after position-changing steps
  (detected by checking step name). This is a simplification; more sophisticated
  versions could use a Force-marker step to indicate when recomputation is needed.

  Args:
      energy_fn: Callable(positions, box) → energy (scalar). If shift_fn is
          provided, energy_fn should handle PBC shifts internally or be called
          after shifts. For simplicity, we compute forces at the position
          directly and assume energy_fn handles box correctly.
      shift_fn: Callable(positions, box) → positions_shifted. Used by A_Step
          for position updates with PBC.
      mass: Mass array [n_atoms] or [n_atoms, 1] in AKMA units.
      sequence_name: Name in step_sequences registry (default: 'baoab_langevin').
          Valid options: 'baoab_langevin', 'baoab_csvr_npt' (v1.0 scope).
          Note: 'settle_with_nhc' deferred to v1.1; 'lfmiddle_langevin' for solute-only modular path.
      dt: Timestep in AKMA units (default: 0.5, which is ~0.5 fs).
      kT: Temperature in kT (default: 1.0).
      gamma: Langevin friction coefficient in ps^-1 (default: 1.0). Used by O_Step.
      water_indices: Optional (n_waters, 3) array of [O, H1, H2] atom indices.
          If provided, enables constraint-aware projections in O_Step and
          SETTLE_Velocity_Step.
      target_pressure_bar: Target pressure in bar (required for
          'baoab_csvr_npt', ignored for others).
      tau_barostat_akma: Barostat time constant in AKMA units (default: 2000.0).
      tau_thermostat_akma: Thermostat time constant in AKMA units (default: 2000.0).
      **kwargs: Additional step-specific parameters (e.g., n_iters for
          SETTLE_Velocity_Step, n_dof for CSVR_Step). These override sequence
          defaults.

  Returns:
      (init_fn, apply_fn): Tuple of initialization and integration functions.
          - init_fn(key, positions, box=None) → IntegratorState
          - apply_fn(state) → IntegratorState

  Raises:
      ValueError: If sequence_name not in step_sequences registry.
      ValueError: If sequence incompatible with parameters (e.g.,
          'baoab_csvr_npt' without target_pressure_bar).
      ValueError: If mass shape or water_indices shape invalid.

  Example:
      >>> init_fn, apply_fn = make_integrator(
      ...     energy_fn, shift_fn,
      ...     sequence_name='baoab_langevin',
      ...     dt=0.5, kT=2.479, gamma=1.0,
      ...     mass=masses,
      ...     water_indices=water_indices
      ... )
      >>> state = init_fn(key, positions, box=box_vec)
      >>> for step in range(100):
      ...     state = apply_fn(state)
      ...     print(f"Step {step}, KE={state.kinetic_energy:.2f}")
  """
  # ========== VALIDATION ==========

  # Check sequence name
  if sequence_name not in step_sequences:
    raise ValueError(
        f"Unknown sequence '{sequence_name}'. "
        f"Available: {list(step_sequences.keys())}"
    )

  # Check parameter compatibility
  if sequence_name == "baoab_csvr_npt" and target_pressure_bar is None:
    raise ValueError(
        f"Sequence '{sequence_name}' requires target_pressure_bar, got None"
    )

  # Validate mass shape
  mass = jnp.asarray(mass)
  if mass.ndim == 1:
    n_atoms_mass = mass.shape[0]
  elif mass.ndim == 2 and mass.shape[1] == 1:
    n_atoms_mass = mass.shape[0]
  else:
    raise ValueError(f"mass must be 1D or (N, 1), got shape {mass.shape}")

  # Validate water_indices shape
  if water_indices is not None:
    water_indices = jnp.asarray(water_indices)
    if water_indices.ndim != 2 or water_indices.shape[1] != 3:
      raise ValueError(
          f"water_indices must be (n_waters, 3), got {water_indices.shape}"
      )

  # ========== SEQUENCE LOOKUP & PARAMETER MERGING ==========

  sequence = step_sequences[sequence_name]
  merged_params = {**sequence.parameters}

  # Override with explicit arguments
  merged_params.update({
      "dt": dt,
      "kT": kT,
      "gamma": gamma,
      "mass": mass,
      "shift_fn": shift_fn,
  })

  # Add conditional parameters
  if water_indices is not None:
    merged_params["water_indices"] = water_indices

  if target_pressure_bar is not None:
    merged_params["target_pressure_bar"] = target_pressure_bar

  # Add thermostat/barostat time constants
  merged_params["tau_barostat_akma"] = tau_barostat_akma
  merged_params["tau_thermostat_akma"] = tau_thermostat_akma

  # Merge any additional kwargs
  merged_params.update(kwargs)

  # Instantiate steps outside apply_fn for efficiency
  steps = []
  for step_name in sequence.steps:
    step_constructor_params = {}
    if step_name == "settle_velocity_step":
      if "n_iters" in merged_params:
        step_constructor_params["n_iters"] = merged_params["n_iters"]
    elif step_name == "csvr_step":
      # n_dof is now passed via IntegratorParams
      pass
    elif step_name == "mc_barostat_step":
      step_constructor_params["n_molecules"] = n_molecules
      step_constructor_params["pressure"] = target_pressure_bar
      step_constructor_params["energy_fn"] = energy_fn
    elif step_name == "scr_barostat_step":
      step_constructor_params["atom_mask"] = kwargs.get("atom_mask")
      step_constructor_params["cutoff"] = kwargs.get("nonbonded_cutoff", 9.0)
    elif step_name == "settle_position_step":
      if "mass_oxygen" in merged_params:
        step_constructor_params["mass_oxygen"] = merged_params["mass_oxygen"]
      if "mass_hydrogen" in merged_params:
        step_constructor_params["mass_hydrogen"] = merged_params["mass_hydrogen"]
    elif step_name == "virtual_site_reconstruction_step":
      step_constructor_params["vs_def"] = vs_def
      step_constructor_params["vs_params"] = vs_params
    elif step_name == "a_step":
      if "shift_fn" in merged_params:
        step_constructor_params["shift_fn"] = merged_params["shift_fn"]
    elif step_name == "o_step":
      if "fraction" in merged_params:
        step_constructor_params["fraction"] = merged_params["fraction"]
    elif step_name == "v_step":
      if "fraction" in merged_params:
        step_constructor_params["fraction"] = merged_params["fraction"]

    steps.append(make_step(step_name, **step_constructor_params))

  # Prepare base parameters PyTree
  base_integrator_params = IntegratorParams(
      dt=merged_params.get("dt", 0.5),
      kT=merged_params.get("kT", 1.0),
      gamma=merged_params.get("gamma", 1.0),
      energy_params=EnergyParams(params=energy_params),
      water_indices=merged_params.get("water_indices"),
      constraint_dofs=_initialize_constraint_dofs(
          merged_params.get("water_indices"), n_atoms_mass
      ),
      positions_old=None,  # Will be updated in apply_fn
      box=None,           # Will be updated in apply_fn
      target_pressure_bar=merged_params.get("target_pressure_bar", 0.0),
      barostat_interval=merged_params.get("barostat_interval", 1),
      compressibility=merged_params.get("compressibility", 4.5e-5),
      tau_barostat=merged_params.get("tau_barostat_akma", 2000.0),
  )

  # ========== INITIALIZATION FUNCTION ==========

  def init_fn(
      key: PRNGKeyArray,
      positions: ArrayType,
      box: Optional[ArrayType] = None,
  ) -> IntegratorState:
    r"""Initialize integrator state with eager computation of all fields.

    This function eagerly computes:
    - Forces via autodiff of energy_fn
    - Initial momentum (zero for cold start)
    - Constraint DOF mask (if water_indices provided)
    - RNG key for stochastic steps
    - Stores box for use in apply_fn force recomputation

    Args:
        key: JAX PRNGKey.
        positions: (N, 3) initial atomic positions.
        box: Optional (3,) box dimensions for periodic boundary.

    Returns:
        IntegratorState with all fields initialized, including box.
    """
    # Compute forces at initial positions
    forces = _compute_forces(
        positions,
        box,
        energy_fn,
        energy_params,
        vs_def=vs_def,
        vs_params=vs_params
    )

    # Initialize momentum (cold start: all zeros)
    momentum = jnp.zeros_like(positions)

    # Create initial state with all fields, including box for PBC
    state = IntegratorState(
        positions=positions,
        momentum=momentum,
        force=forces,
        mass=mass,
        rng=key,
        box=box,
    )

    return state

  # ========== APPLICATION FUNCTION ==========

  def apply_fn(state: IntegratorState) -> IntegratorState:
    r"""Apply one timestep of integrator sequence.

    Args:
        state: Current IntegratorState (including box for PBC).

    Returns:
        Updated IntegratorState after one timestep.
    """
    # Update params with current box and positions_old
    params = eqx.tree_at(
        lambda p: (p.box, p.positions_old),
        base_integrator_params,
        (state.box, state.positions),
        is_leaf=lambda x: x is None
    )

    current_state = state

    # Python for-loop is intentional — DO NOT convert to lax.scan.
    # Step pytrees are heterogeneous (O_Step, V_Step, SETTLE_Velocity_Step all have
    # different field sets), so scan's requirement for homogeneous carry pytrees
    # would fail. JAX unrolls static Python loops at trace time without emitting
    # custom_call ops, so this is fully StableHLO-compatible.
    for step in steps:
      current_state = step.apply(current_state, params)
      if isinstance(step, Force_Step):
        new_forces = _compute_forces(
            current_state.positions,
            current_state.box,
            energy_fn,
            energy_params,
            vs_def=vs_def,
            vs_params=vs_params,
        )
        current_state = current_state.__replace__(force=new_forces)

    # Recompute forces at the end of the timestep (with virtual site redistribution)
    new_forces = _compute_forces(
        current_state.positions,
        current_state.box,
        energy_fn,
        energy_params,
        vs_def=vs_def,
        vs_params=vs_params
    )
    current_state = current_state.__replace__(
        force=new_forces,
        step_count=current_state.step_count + 1
    )

    return current_state

  return init_fn, apply_fn
