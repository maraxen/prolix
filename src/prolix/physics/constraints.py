"""Constraint algorithm plugin system for flexible integrator design.

Mirrors OpenMM's ReferenceConstraintAlgorithm pattern: constraint algorithms
implement apply_positions() and apply_velocities() methods, allowing them to be
injected into integrators at the correct BAOAB sub-steps.

Classes:
- ConstraintAlgorithm: Base class (Equinox module)
- NullConstraint: Identity (implicit solvent / unconstrained)
- SETTLEConstraint: SETTLE for rigid water (analytical closed-form)
- ShakeRattleConstraint: SHAKE/RATTLE for solute X-H bonds (iterative)
- CompositeConstraint: Combines multiple constraints in sequence
- make_constraint(): Factory that builds the appropriate constraint from topology
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable

from prolix.types import Array


class ConstraintAlgorithm(eqx.Module):
  """Base class for constraint algorithms. Both methods return corrected arrays."""

  def apply_positions(
      self, pos_start: Array, pos_unconstrained: Array, mass: Array, box: Array | None = None
  ) -> Array:
    """Apply position constraints. Return corrected positions.

    Args:
        pos_start: Reference positions (start-of-step, before any A steps).
        pos_unconstrained: Unconstrained positions (after A-O-A steps).
        mass: Atomic masses, shape (N,1).
        box: Periodic box vectors or None.

    Returns:
        Constrained positions.
    """
    raise NotImplementedError

  def apply_velocities(
      self,
      pos_start: Array,
      pos_constrained: Array,
      momenta: Array,
      mass: Array,
      dt: float,
      shift_fn: Callable | None = None,
  ) -> Array:
    """Apply velocity constraints. Return corrected momenta.

    Args:
        pos_start: Reference positions (start-of-step).
        pos_constrained: Positions after apply_positions.
        momenta: Atomic momenta (mass-weighted velocities), shape (N,3).
        mass: Atomic masses, shape (N,1).
        dt: Timestep.
        shift_fn: Displacement function for periodic boundary conditions.

    Returns:
        Constrained momenta.
    """
    raise NotImplementedError


class NullConstraint(ConstraintAlgorithm):
  """Identity constraint — no constraints. Used for implicit solvent / unconstrained systems.

  JAX JIT will eliminate this at compile time if used as a static branch.
  """

  def apply_positions(
      self, pos_start: Array, pos_unconstrained: Array, mass: Array, box: Array | None = None
  ) -> Array:
    return pos_unconstrained

  def apply_velocities(
      self,
      pos_start: Array,
      pos_constrained: Array,
      momenta: Array,
      mass: Array,
      dt: float,
      shift_fn: Callable | None = None,
  ) -> Array:
    return momenta


class SETTLEConstraint(ConstraintAlgorithm):
  """SETTLE constraint for rigid water molecules.

  Wraps settle_positions() and settle_velocities() from settle.py.
  """

  water_indices: Array  # (N_waters, 3) int32
  r_OH: float
  r_HH: float
  mass_oxygen: float
  mass_hydrogen: float
  n_iters: int
  settle_velocity_tol: float | None

  def apply_positions(
      self, pos_start: Array, pos_unconstrained: Array, mass: Array, box: Array | None = None
  ) -> Array:
    from prolix.physics.settle import settle_positions

    return settle_positions(
        pos_unconstrained,
        pos_start,
        self.water_indices,
        self.r_OH,
        self.r_HH,
        self.mass_oxygen,
        self.mass_hydrogen,
        box,
    )

  def apply_velocities(
      self,
      pos_start: Array,
      pos_constrained: Array,
      momenta: Array,
      mass: Array,
      dt: float,
      shift_fn: Callable | None = None,
  ) -> Array:
    from prolix.physics.settle import settle_velocities

    # settle_velocities expects velocities, not momenta. Convert and convert back.
    velocities = momenta / mass  # (N, 3) / (N, 1) = (N, 3)
    velocities_constrained = settle_velocities(
        velocities,
        pos_start,
        pos_constrained,
        self.water_indices,
        dt,
        self.mass_oxygen,
        self.mass_hydrogen,
        n_iters=self.n_iters,
        adaptive_tol=self.settle_velocity_tol,
    )
    return velocities_constrained * mass


class ShakeRattleConstraint(ConstraintAlgorithm):
  """SHAKE/RATTLE constraint for solute X-H bonds.

  Wraps project_positions() and project_momenta() from physics/simulate.py.
  Note: requires shift_fn for periodic boundary conditions.
  """

  pairs: Array  # (N_bonds, 2) int32
  lengths: Array  # (N_bonds,) float

  def apply_positions(
      self, pos_start: Array, pos_unconstrained: Array, mass: Array, box: Array | None = None
  ) -> Array:
    # SHAKE position constraint doesn't need shift_fn for unconstrained path
    # (shift_fn would be used inside project_positions if periodic, but we pass None here
    # and rely on caller to ensure consistency)
    from prolix.physics.simulate import project_positions

    return project_positions(
        pos_unconstrained, self.pairs, self.lengths, mass, shift_fn=None
    )

  def apply_velocities(
      self,
      pos_start: Array,
      pos_constrained: Array,
      momenta: Array,
      mass: Array,
      dt: float,
      shift_fn: Callable | None = None,
  ) -> Array:
    from prolix.physics.simulate import project_momenta

    return project_momenta(momenta, pos_constrained, self.pairs, mass, shift_fn)


class CompositeConstraint(ConstraintAlgorithm):
  """Composite of multiple constraints applied in sequence.

  Applies SHAKE before SETTLE (matching OpenMM and batched_simulate convention).
  """

  constraints: tuple  # tuple[ConstraintAlgorithm, ...]

  def apply_positions(
      self, pos_start: Array, pos_unconstrained: Array, mass: Array, box: Array | None = None
  ) -> Array:
    pos = pos_unconstrained
    for constraint in self.constraints:
      pos = constraint.apply_positions(pos_start, pos, mass, box)
    return pos

  def apply_velocities(
      self,
      pos_start: Array,
      pos_constrained: Array,
      momenta: Array,
      mass: Array,
      dt: float,
      shift_fn: Callable | None = None,
  ) -> Array:
    mom = momenta
    for constraint in self.constraints:
      mom = constraint.apply_velocities(pos_start, pos_constrained, mom, mass, dt, shift_fn)
    return mom


def make_constraint(
    water_indices: Array | None = None,
    constraint_pairs: Array | None = None,
    constraint_lengths: Array | None = None,
    r_OH: float | None = None,
    r_HH: float | None = None,
    mass_oxygen: float = 15.999,
    mass_hydrogen: float = 1.008,
    n_iters: int = 10,
    settle_velocity_tol: float | None = None,
) -> ConstraintAlgorithm:
  """Factory function: build appropriate constraint algorithm from topology.

  Returns NullConstraint if no constraints are specified.
  Returns SETTLEConstraint if only water_indices.
  Returns ShakeRattleConstraint if only solute constraints.
  Returns CompositeConstraint if both water and solute constraints (SHAKE before SETTLE).

  Args:
      water_indices: (N_waters, 3) int32 array of water atom indices, or None.
      constraint_pairs: (N_bonds, 2) int32 array of bond pairs, or None.
      constraint_lengths: (N_bonds,) float array of target bond lengths, or None.
      r_OH, r_HH: Water bond lengths. If None, uses TIP3P defaults.
      mass_oxygen, mass_hydrogen: Water masses.
      n_iters: Max iterations for SETTLE velocity constraint.
      settle_velocity_tol: Adaptive SETTLE tolerance or None (uses fixed n_iters).

  Returns:
      A ConstraintAlgorithm instance.
  """
  from prolix.physics.settle import TIP3P_ROH, TIP3P_RHH

  r_OH = r_OH or TIP3P_ROH
  r_HH = r_HH or TIP3P_RHH

  parts = []

  # Add SHAKE/RATTLE for solute constraints
  if constraint_pairs is not None and len(constraint_pairs) > 0:
    parts.append(ShakeRattleConstraint(pairs=constraint_pairs, lengths=constraint_lengths))

  # Add SETTLE for rigid water
  if water_indices is not None and water_indices.shape[0] > 0:
    parts.append(
        SETTLEConstraint(
            water_indices=water_indices,
            r_OH=r_OH,
            r_HH=r_HH,
            mass_oxygen=mass_oxygen,
            mass_hydrogen=mass_hydrogen,
            n_iters=n_iters,
            settle_velocity_tol=settle_velocity_tol,
        )
    )

  if len(parts) == 0:
    return NullConstraint()
  elif len(parts) == 1:
    return parts[0]
  else:
    return CompositeConstraint(constraints=tuple(parts))
