"""Constraint algorithm plugin system for flexible integrator design.

This module comprises two layers:

1. **Kinematics Layer** (ConstraintDOFMask):
   Formal constraint model and degree-of-freedom masks for SETTLE.
   - SETTLE kinematics: 3 constraint equations (2 OH bonds, 1 HH distance)
   - Constraint Jacobian: dC/dx relationships (n_constraints × DOF)
   - DOF decomposition: rigid-body DOF vs free DOF for projection operators

2. **Algorithm Layer** (ConstraintAlgorithm subclasses):
   Mirrors OpenMM's ReferenceConstraintAlgorithm pattern: constraint algorithms
   implement apply_positions() and apply_velocities() methods, allowing them to be
   injected into integrators at the correct BAOAB sub-steps.

Classes:
- ConstraintDOFMask: Formal SETTLE kinematics + DOF mask construction
- ConstraintAlgorithm: Base class (Equinox module)
- NullConstraint: Identity (implicit solvent / unconstrained)
- SETTLEConstraint: SETTLE for rigid water (analytical closed-form)
- ShakeRattleConstraint: SHAKE/RATTLE for solute X-H bonds (iterative)
- CompositeConstraint: Combines multiple constraints in sequence
- make_constraint(): Factory that builds the appropriate constraint from topology

Reference:
    Miyamoto, S., & Kollman, P. A. (1992). Settle: An analytical version of
    the SHAKE and RATTLE algorithm for rigid water models.
    Journal of Computational Chemistry, 13(8), 952-962.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array as ArrayType, Bool, Int

from prolix.typing import Array, WaterIndicesArray


@dataclass
class ConstraintDOFMask:
  r"""Formal SETTLE constraint model and degree-of-freedom mask decomposition.

  SETTLE is an analytical constraint solver for rigid 3-site water molecules
  (TIP3P or similar). This class formalizes the kinematic structure of SETTLE
  constraints and provides DOF masks for decomposing the system into rigid and
  free subspaces.

  **Constraint Equations (SETTLE Kinematics)**:

  For each water molecule with atoms (O, H1, H2):

    - C1: |r_O - r_H1|² = d_OH²     (O-H1 bond)
    - C2: |r_O - r_H2|² = d_OH²     (O-H2 bond)
    - C3: |r_H1 - r_H2|² = d_HH²    (H-H distance)

  These 3 holonomic constraints reduce the 9 DOF of one water to 6 DOF
  (3 translational + 3 rotational for rigid body motion).

  **Constraint Jacobian (Velocity Constraint Form)**:

  The time-derivative of constraint equations gives velocity constraints:

    dC_i/dt = (∂C_i/∂x) · ẋ = 0

  For water k with atom indices i_O, i_H1, i_H2:

    ∂C1/∂r_O   = 2(r_O - r_H1)
    ∂C1/∂r_H1  = -2(r_O - r_H1)
    ∂C1/∂r_H2  = 0
    ... (similarly for C2, C3)

  Constraint Jacobian M has shape (3*n_waters, 3*n_atoms) where:
    - Rows: constraint equations
    - Columns: spatial DOF (x, y, z per atom)
    - Non-zero blocks only for atoms in water molecules

  **DOF Decomposition**:

  System decomposed into rigid (constrained) and free (unconstrained) subspaces:

    - n_rigid_dof = 3 * n_waters   (3 constraints per water)
    - n_free_dof = 3*n_atoms - 3*n_waters
    - n_constraint_equations = 3 * n_waters

  **Projection Operator**:

  P projects velocities onto the free subspace (i.e., removes rigid DOF):

    P @ v_free = v_free    (identity on free subspace)
    P @ v_rigid = 0        (eliminates rigid-body motion)

  Example (64-water TIP3P system):
    - n_atoms = 64*3 = 192
    - n_rigid_dof = 64*3 = 192
    - n_free_dof = 192 - 192 = 0   (all atoms are in water!)
    - Projection operator P is the zero matrix (all water, no solute)

  Attributes:
    water_indices: (n_waters, 3) int32 array [O, H1, H2] atom indices per water.
    n_atoms: Total number of atoms in the system.

  Properties:
    rigid_dof_mask: (n_atoms, 3) bool array. True if atom is in water (rigid).
    free_dof_mask: (n_atoms, 3) bool array. True if atom is free (not in water).
    n_rigid_atoms: Count of atoms in rigid bodies.
    n_free_atoms: Count of free atoms.
    n_constraint_equations: Number of constraints (always 3 * n_waters).

  Methods:
    is_orthogonal_complement(): Verify rigid_dof_mask ⊥ free_dof_mask.
    projection_operator(): Construct projection matrix P onto free subspace.

  Reference:
    Miyamoto, S., & Kollman, P. A. (1992). Settle: An analytical version of
    the SHAKE and RATTLE algorithm for rigid water models.
    Journal of Computational Chemistry, 13(8), 952-962.
  """

  water_indices: WaterIndicesArray  # (n_waters, 3)
  n_atoms: int

  @property
  def n_waters(self) -> int:
    """Number of water molecules."""
    return self.water_indices.shape[0]

  @property
  def rigid_dof_mask(self) -> Bool[ArrayType, "n_atoms 3"]:
    """Bool mask (n_atoms, 3) where True = rigid atom in water, False = free."""
    mask = jnp.zeros((self.n_atoms, 3), dtype=jnp.bool_)
    # Flatten water indices to 1D array of all water atoms
    water_atom_indices = self.water_indices.reshape(-1)
    # Set mask to True for all water atoms across all 3 dimensions
    mask = mask.at[water_atom_indices, :].set(True)
    return mask

  @property
  def free_dof_mask(self) -> Bool[ArrayType, "n_atoms 3"]:
    """Bool mask (n_atoms, 3) where True = free atom, False = rigid."""
    return ~self.rigid_dof_mask

  @property
  def n_rigid_atoms(self) -> int:
    """Count of atoms in rigid bodies (number of atoms that are water)."""
    # Count unique atom indices in water_indices
    unique_rigid_indices = jnp.unique(self.water_indices.reshape(-1))
    return int(unique_rigid_indices.size)

  @property
  def n_free_atoms(self) -> int:
    """Count of free atoms (total atoms minus rigid atoms)."""
    return self.n_atoms - self.n_rigid_atoms

  @property
  def n_constraint_equations(self) -> int:
    """Number of independent constraint equations (3 per water)."""
    return 3 * self.n_waters

  def is_orthogonal_complement(self) -> bool:
    """Verify that rigid_dof_mask and free_dof_mask are orthogonal complements.

    Returns:
      True if (rigid_dof_mask & free_dof_mask) is all False and
             (rigid_dof_mask | free_dof_mask) is all True.
    """
    rigid = self.rigid_dof_mask
    free = self.free_dof_mask

    # Check no overlap: (rigid & free) should be all False
    has_overlap = jnp.any(rigid & free)
    if has_overlap:
      return False

    # Check completeness: (rigid | free) should be all True
    is_complete = jnp.all(rigid | free)
    return bool(is_complete)

  def projection_operator(self) -> Array:
    r"""Construct projection matrix P onto the free-DOF subspace.

    P is an (3*n_atoms, 3*n_atoms) matrix satisfying:
      - P @ v_free = v_free   (identity on free DOF)
      - P @ v_rigid = 0       (zero on rigid DOF)
      - P² = P                 (idempotent, projection property)

    Mathematically, P = I - M^T (M M^T)^{-1} M, where M is the constraint
    Jacobian. However, for computational simplicity, we construct P directly
    from the DOF masks.

    The projection operator explicitly enforces the constraint that velocities
    of atoms in rigid water bodies must satisfy the SETTLE constraint equations.

    Returns:
      (3*n_atoms, 3*n_atoms) array representing the projection operator.

    Note:
      For systems where all atoms are in water molecules (n_free_atoms == 0),
      this returns the zero matrix (all DOF are constrained).
    """
    # Create flat boolean mask: True if atom is free
    free_flat = self.free_dof_mask.reshape(-1)  # (3*n_atoms,)

    # Diagonal matrix: 1 on free DOF, 0 on rigid DOF
    diag_values = jnp.where(free_flat, 1.0, 0.0)
    P = jnp.diag(diag_values)

    return P


def project_positions(R, pairs, lengths, mass, shift_fn, constraint_mask=None, tol=1e-5, max_iter=20):
  """Iterative SHAKE projection for positions."""
  if constraint_mask is None:
    constraint_mask = jnp.ones(len(pairs), dtype=jnp.float32)
  else:
    constraint_mask = constraint_mask.astype(jnp.float32)

  def body_fn(i, R_curr):
    r1 = R_curr[pairs[:, 0]]
    r2 = R_curr[pairs[:, 1]]
    d_vec = r1 - r2  # Free space displacement
    d2 = jnp.sum(d_vec**2, axis=-1)
    diff = d2 - lengths**2

    inv_m1 = 1.0 / mass[pairs[:, 0], 0]
    inv_m2 = 1.0 / mass[pairs[:, 1], 0]
    w_sum = inv_m1 + inv_m2

    g = -diff / (2.0 * w_sum * d2 + 1e-8)
    g = g * constraint_mask

    delta = d_vec * g[:, None]
    d1 = delta * inv_m1[:, None]
    d2_corr = -delta * inv_m2[:, None]

    R_curr = R_curr.at[pairs[:, 0]].add(d1)
    R_curr = R_curr.at[pairs[:, 1]].add(d2_corr)
    return R_curr

  return jax.lax.fori_loop(0, max_iter, body_fn, R)


def project_momenta(P, R, pairs, mass, shift_fn, constraint_mask=None, tol=1e-6, max_iter=20):
  """Iterative RATTLE projection for momenta."""
  if constraint_mask is None:
    constraint_mask = jnp.ones(len(pairs), dtype=jnp.float32)
  else:
    constraint_mask = constraint_mask.astype(jnp.float32)

  inv_m1 = 1.0 / mass[pairs[:, 0], 0]
  inv_m2 = 1.0 / mass[pairs[:, 1], 0]
  w_sum = inv_m1 + inv_m2

  r1 = R[pairs[:, 0]]
  r2 = R[pairs[:, 1]]
  r12 = r1 - r2

  def body_fn(i, P_curr):
    v1 = P_curr[pairs[:, 0]] * inv_m1[:, None]
    v2 = P_curr[pairs[:, 1]] * inv_m2[:, None]
    v12 = v1 - v2

    dot = jnp.sum(v12 * r12, axis=-1)
    d2 = jnp.sum(r12**2, axis=-1)
    k = -dot / (w_sum * d2 + 1e-8)
    k = k * constraint_mask

    impulse = r12 * k[:, None]
    P_curr = P_curr.at[pairs[:, 0]].add(impulse)
    P_curr = P_curr.at[pairs[:, 1]].add(-impulse)
    return P_curr

  return jax.lax.fori_loop(0, max_iter, body_fn, P)


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
    from prolix.physics import settle

    return settle.settle_positions(
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
    from prolix.physics import settle

    # settle_velocities expects velocities, not momenta. Convert and convert back.
    velocities = momenta / mass  # (N, 3) / (N, 1) = (N, 3)
    velocities_constrained = settle.settle_velocities(
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
