"""Type definitions for prolix."""

from __future__ import annotations

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

# aliases
PRNGKey = PRNGKeyArray
