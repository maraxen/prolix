"""Phase 0 Spike: Minimal SETTLE Constraint Plugin for ADR-005.

This module prototypes a constraint-plugin interface that wraps existing
settle_positions() and settle_velocities() functions from settle.py.

Goal: Validate that SETTLE velocity projection can cleanly factor into a
plugin interface while preserving BAOAB symplectic structure and kUPS parity.

This is a minimal proof-of-concept; full ConstraintAlgorithm protocol
implementation is Phase 1 work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from prolix.physics import settle
from prolix.typing import WaterIndicesArray


class ConstraintState(NamedTuple):
  """Minimal state container for constraint plugin operations."""
  positions_old: Any  # Previous positions (for velocity constraint)
  positions_new: Any  # Current positions
  box: Any | None = None  # Optional periodic box


class SettleConstraintPlugin:
  """Minimal SETTLE constraint plugin wrapping settle.py functions.

  This plugin encapsulates SETTLE position and velocity constraints
  into a reusable interface that can be composed with integrators.

  Interface:
    - settle_positions(pos_unconstrained, pos_old, box) -> pos_constrained
    - settle_velocities(vel, pos_old, pos_new, box) -> vel_constrained

  The plugin preserves the exact same logic as the original settle_langevin
  integrator to enable direct equivalence testing.
  """

  def __init__(
    self,
    water_indices: WaterIndicesArray,
    r_OH: float = settle.TIP3P_ROH,
    r_HH: float = settle.TIP3P_RHH,
    mass_oxygen: float = 15.999,
    mass_hydrogen: float = 1.008,
    settle_velocity_iters: int = 10,
    settle_velocity_tol: float | None = None,
  ):
    """Initialize SETTLE constraint plugin.

    Args:
        water_indices: (N_waters, 3) array of [O, H1, H2] indices.
        r_OH: Target O-H bond length (Å).
        r_HH: Target H-H distance (Å).
        mass_oxygen: Mass of oxygen (amu).
        mass_hydrogen: Mass of hydrogen (amu).
        settle_velocity_iters: Number of RATTLE iterations.
        settle_velocity_tol: Adaptive tolerance for RATTLE convergence.
    """
    self.water_indices = water_indices
    self.r_OH = r_OH
    self.r_HH = r_HH
    self.mass_oxygen = mass_oxygen
    self.mass_hydrogen = mass_hydrogen
    self.settle_velocity_iters = settle_velocity_iters
    self.settle_velocity_tol = settle_velocity_tol

  def project_positions(
    self,
    positions_unconstrained: Any,
    positions_old: Any,
    box: Any | None = None,
  ) -> Any:
    """Apply SETTLE position constraints.

    Args:
        positions_unconstrained: Unconstrained positions after integrator step.
        positions_old: Positions before the integrator step.
        box: Optional periodic box dimensions.

    Returns:
        Constrained positions array.
    """
    return settle.settle_positions(
      positions_unconstrained,
      positions_old,
      self.water_indices,
      r_OH=self.r_OH,
      r_HH=self.r_HH,
      mass_oxygen=self.mass_oxygen,
      mass_hydrogen=self.mass_hydrogen,
      box=box,
    )

  def project_velocities(
    self,
    velocities: Any,
    positions_old: Any,
    positions_constrained: Any,
    dt: float,
    box: Any | None = None,
  ) -> Any:
    """Apply SETTLE velocity constraints.

    Args:
        velocities: Unconstrained velocities (N, 3).
        positions_old: Positions before constraint step.
        positions_constrained: Positions after constraint step.
        dt: Timestep (used by RATTLE).
        box: Optional periodic box (currently unused).

    Returns:
        Constrained velocities array.
    """
    return settle.settle_velocities(
      velocities,
      positions_old,
      positions_constrained,
      self.water_indices,
      dt,
      mass_oxygen=self.mass_oxygen,
      mass_hydrogen=self.mass_hydrogen,
      n_iters=self.settle_velocity_iters,
      adaptive_tol=self.settle_velocity_tol,
    )

  def project_momenta(
    self,
    momentum: Any,
    positions_old: Any,
    positions_constrained: Any,
    mass: Any,
    dt: float,
    box: Any | None = None,
  ) -> Any:
    """Apply SETTLE constraints to momenta (convenience wrapper).

    Args:
        momentum: Atomic momenta (N, 3).
        positions_old: Positions before constraint.
        positions_constrained: Positions after constraint.
        mass: Atomic masses (N,) or (N, 1).
        dt: Timestep.
        box: Optional periodic box.

    Returns:
        Constrained momenta array.
    """
    velocity = momentum / mass
    velocity = self.project_velocities(
      velocity, positions_old, positions_constrained, dt, box
    )
    return velocity * mass
