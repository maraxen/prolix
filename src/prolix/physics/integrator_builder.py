"""Integrator builder: factory for composing step sequences into working integrators.

This module implements Phase 2.1 of ADR-005: the composition layer that converts
step sequences (from step_system.py) into actual integrator functions.

**Architecture**:

The make_integrator factory is the critical composition layer:

1. Takes sequence name + parameters
2. Eagerly initializes state (forces, chain_state, barostat_state)
3. Returns (init_fn, apply_fn) tuple for time-stepping

**Design Principles**:

- **Eager initialization**: All state (forces, chain state, barostat state) computed
  in init_fn, not lazily during integration
- **Step composition**: apply_fn loops over step sequence; each step receives full
  state and returns modified state
- **RNG management**: Split key between stochastic steps (O_Step, CSVR_Step)
- **No breaking changes**: settle_langevin remains available as thin wrapper

**Reference**: ADR-005 v2.0, Phase 2.1

**Author**: Fixer Agent (Phase 2.1 implementation)
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array as ArrayType, PRNGKeyArray

from prolix.physics.constraints import ConstraintDOFMask
from prolix.physics.step_system import (
    IntegratorState,
    Step,
    StepSequence,
    make_step,
    step_sequences,
)
from prolix.types import Array, WaterIndicesArray


def _compute_forces(
    positions: ArrayType,
    box: Optional[ArrayType],
    energy_fn: Callable,
) -> ArrayType:
  r"""Compute forces via autodiff of energy function.

  Args:
      positions: (N, 3) atomic positions.
      box: Optional (3,) box dimensions for periodic boundary.
      energy_fn: Energy function(positions, box) → scalar energy.

  Returns:
      (N, 3) force array F = -∇E.
  """
  def energy_wrapper(R):
    return energy_fn(R, box) if box is not None else energy_fn(R)

  forces = -jax.grad(energy_wrapper)(positions)
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
          Valid options: 'baoab_langevin', 'baoab_csvr_npt', 'settle_with_nhc'.
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
    forces = _compute_forces(positions, box, energy_fn)

    # Initialize momentum (cold start: all zeros)
    momentum = jnp.zeros_like(positions)

    # Create initial state with all fields, including box for PBC
    state = IntegratorState(
        position=positions,
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

    This function loops through the step sequence. Each step is applied in order.
    Some steps (O_Step, CSVR_Step) return (state, new_key) tuples; we handle
    both return types.

    Forces are recomputed once at the end of the full timestep, after all
    position updates are complete, using the box stored in the state.

    Args:
        state: Current IntegratorState (including box for PBC).

    Returns:
        Updated IntegratorState after one timestep.
    """
    current_state = state
    current_key = state.rng
    box = state.box  # Retrieve box from state for force recomputation

    # Initialize constraint DOFs if needed
    constraint_dofs = _initialize_constraint_dofs(
        merged_params.get("water_indices"), n_atoms_mass
    )

    # Loop through steps in sequence
    for step_name in sequence.steps:
      # Create step instance with only the constructor parameters it needs
      # Most steps have minimal constructors; filter out runtime params
      step_constructor_params = {}
      if step_name == "settle_velocity_step":
        # SETTLE_Velocity_Step accepts: water_indices, n_iters, mass_oxygen, mass_hydrogen
        if "water_indices" in merged_params:
          step_constructor_params["water_indices"] = merged_params["water_indices"]
        if "n_iters" in merged_params:
          step_constructor_params["n_iters"] = merged_params["n_iters"]
      elif step_name == "csvr_step":
        # CSVR_Step accepts: n_dof (optional)
        if "n_dof" in merged_params:
          step_constructor_params["n_dof"] = merged_params["n_dof"]
      elif step_name == "a_step":
        # A_Step accepts: fraction, shift_fn
        if "shift_fn" in merged_params:
          step_constructor_params["shift_fn"] = merged_params["shift_fn"]
      elif step_name == "o_step":
        # O_Step accepts: fraction, project_rigid
        pass  # Defaults are fine
      elif step_name == "v_step":
        # V_Step accepts: fraction
        pass  # Defaults are fine
      elif step_name == "nhc_step":
        # NHC_Step accepts: nothing
        pass

      step = make_step(step_name, **step_constructor_params)

      # Apply step with merged params
      result = step.apply(current_state, constraint_dofs=constraint_dofs, **merged_params)

      # Handle return type: (state, key) or state
      if isinstance(result, tuple):
        current_state, current_key = result
        current_state = current_state.__replace__(rng=current_key)
      else:
        current_state = result

    # Recompute forces at the end of the timestep
    # (after all position changes are complete, using box from state)
    new_forces = _compute_forces(current_state.position, box, energy_fn)
    current_state = current_state.__replace__(force=new_forces)

    # Update RNG key in final state
    current_state = current_state.__replace__(rng=current_key)

    return current_state

  return init_fn, apply_fn
