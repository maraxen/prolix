"""Type definitions for prolix."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

ArrayLike = Array | np.ndarray

# Scalar Types
Scalar = Int[ArrayLike, ""]
ScalarFloat = Float[ArrayLike, ""]

# Common Shapes
Coordinates = Float[ArrayLike, "num_atoms 3"]
AtomsMask = Bool[ArrayLike, "num_atoms"]
Radii = Float[ArrayLike, "num_atoms"]
Energy = Float[ArrayLike, ""]
CmapGrid = Float[ArrayLike, "grid_size grid_size"]
CmapCoeffs = Float[ArrayLike, "num_maps grid_size grid_size 16"]
CmapEnergyGrids = Float[ArrayLike, "num_maps grid_size grid_size"]
CmapPoints = Float[ArrayLike, "num_points"]
TorsionAngles = Float[ArrayLike, "num_torsions"]
TorsionIndices = Int[ArrayLike, "num_torsions"]

# Hybrid & Packed Types
VirtualSiteDef = Int[ArrayLike, "num_sites 4"]
VirtualSiteParamsPacked = Float[ArrayLike, "num_sites 12"]
WaterIndicesArray = Int[ArrayLike, "num_waters 3"]
BondIndices = Int[ArrayLike, "num_bonds 2"]
BondParamsPacked = Float[ArrayLike, "num_bonds 2"]
AngleIndices = Int[ArrayLike, "num_angles 3"]
AngleParamsPacked = Float[ArrayLike, "num_angles 2"]

# aliases
PRNGKey = PRNGKeyArray


# -----------------------------------------------------------------------------
# Data Layout Definitions
# -----------------------------------------------------------------------------
# These constants define the layout of packed arrays in SystemParams.
# They provide a single source of truth for "magic indices".

# Virtual Site Parameter Row Layout
# Source: SystemParams["virtual_site_params"] (N_sites, 12)
# [vx, vy, vz, w_o1, w_o2, w_o3, w_x1, w_x2, w_x3, w_y1, w_y2, w_y3]
VS_IDX_POS = slice(0, 3)
VS_IDX_W_ORIGIN = slice(3, 6)
VS_IDX_W_X = slice(6, 9)
VS_IDX_W_Y = slice(9, 12)

# CMAP Torsion Index Row Layout
# Source: SystemParams["cmap_torsions"] (N_torsions, 5)
# [type_idx, a1, a2, a3, a4]
# Phi torsion atoms: a1-a2-a3-a4 -> indices [0, 1, 2, 3] relative to slice?
# OpenMM/Gromacs CMAP torsion format: [map_index, atom1, atom2, atom3, atom4, atom5?] -> usually 5 indices
# Let's verify existing usage in system.py:
# phi_indices = cmap_torsions[:, 0:4] -> indices 0,1,2,3
# psi_indices = cmap_torsions[:, 1:5] -> indices 1,2,3,4
# So the row is [i, j, k, l, m] where Phi=i,j,k,l and Psi=j,k,l,m
CMAP_IDX_PHI = slice(0, 4)
CMAP_IDX_PSI = slice(1, 5)

# Bonded Parameter Row Layouts
BOND_IDX_LENGTH = 0
BOND_IDX_K = 1

ANGLE_IDX_THETA0 = 0
ANGLE_IDX_K = 1

DIHEDRAL_IDX_PERIODICITY = 0
DIHEDRAL_IDX_PHASE = 1
DIHEDRAL_IDX_K = 2

# Water Indices Layout (Oxygen, Hydrogen 1, Hydrogen 2)
WATER_IDX_O = 0
WATER_IDX_H1 = 1
WATER_IDX_H2 = 2


class VirtualSiteParams(NamedTuple):
  """Parameters for virtual site reconstruction.

  Represents a single row from `SystemParams["virtual_site_params"]`.

  Fields:
      p_local: Local coordinates (x, y, z) in the local frame.
      origin_weights: Weights (w1, w2, w3) for parent atoms to define the origin.
      x_weights: Weights (w1, w2, w3) for parent atoms to define the X-axis.
      y_weights: Weights (w1, w2, w3) for parent atoms to define the Y-axis.

  """

  p_local: Array
  origin_weights: Array
  x_weights: Array
  y_weights: Array

  @classmethod
  def from_row(cls, row: Array) -> VirtualSiteParams:
    """Construct params from a packed parameter row.

    Args:
        row: A 1D array of shape (12,) from `SystemParams["virtual_site_params"]`.
             See Data Layout Definitions above for details.

    """
    return cls(
      p_local=row[VS_IDX_POS],
      origin_weights=row[VS_IDX_W_ORIGIN],
      x_weights=row[VS_IDX_W_X],
      y_weights=row[VS_IDX_W_Y],
    )


class CmapTorsionIndices(NamedTuple):
  """Indices for atoms in a CMAP torsion pair.

  Fields:
      phi_indices: Indices for phi torsion (i, j, k, l)
      psi_indices: Indices for psi torsion (j, k, l, m)
  """

  phi_indices: Int[ArrayLike, 4]
  psi_indices: Int[ArrayLike, 4]

  @classmethod
  def from_row(cls, row: Array) -> CmapTorsionIndices:
    """Construct indices from a packed CMAP torsion row.

    Args:
        row: A 1D array of shape (5,) from `SystemParams["cmap_torsions"]`.
             Format: [atom_i, atom_j, atom_k, atom_l, atom_m]
             Defines two sharing torsions:
             - Phi: i-j-k-l
             - Psi: j-k-l-m

    """
    return cls(
      phi_indices=row[CMAP_IDX_PHI],
      psi_indices=row[CMAP_IDX_PSI],
    )


class BondParams(NamedTuple):
  """Parameters for a harmonic bond.

  Fields:
      length: Equilibrium bond length.
      k: Bond spring constant (force constant).
  """

  length: ScalarFloat
  k: ScalarFloat

  @classmethod
  def from_row(cls, row: Array) -> BondParams:
    """Construct params from a packed parameter row."""
    return cls(
      length=row[BOND_IDX_LENGTH],
      k=row[BOND_IDX_K],
    )


class AngleParams(NamedTuple):
  """Parameters for a harmonic angle.

  Fields:
      theta0: Equilibrium angle in radians.
      k: Angle spring constant.
  """

  theta0: ScalarFloat
  k: ScalarFloat

  @classmethod
  def from_row(cls, row: Array) -> AngleParams:
    """Construct params from a packed parameter row."""
    return cls(
      theta0=row[ANGLE_IDX_THETA0],
      k=row[ANGLE_IDX_K],
    )


class DihedralParams(NamedTuple):
  """Parameters for a periodic dihedral.

  Fields:
      periodicity: Periodicity of the torsion.
      phase: Phase shift in radians.
      k: Force constant.
  """

  periodicity: ScalarFloat
  phase: ScalarFloat
  k: ScalarFloat

  @classmethod
  def from_row(cls, row: Array) -> DihedralParams:
    """Construct params from a packed parameter row."""
    return cls(
      periodicity=row[DIHEDRAL_IDX_PERIODICITY],
      phase=row[DIHEDRAL_IDX_PHASE],
      k=row[DIHEDRAL_IDX_K],
    )


class WaterIndices(NamedTuple):
  """Indices for atoms in a water molecule.

  Fields:
      oxygen: Index of the oxygen atom.
      hydrogen1: Index of the first hydrogen atom.
      hydrogen2: Index of the second hydrogen atom.
  """

  oxygen: Int[ArrayLike, ""]
  hydrogen1: Int[ArrayLike, ""]
  hydrogen2: Int[ArrayLike, ""]

  @classmethod
  def from_row(cls, row: Array) -> WaterIndices:
    """Construct indices from a row/array of length 3."""
    return cls(
      oxygen=row[WATER_IDX_O],
      hydrogen1=row[WATER_IDX_H1],
      hydrogen2=row[WATER_IDX_H2],
    )
