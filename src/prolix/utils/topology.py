from __future__ import annotations

import collections
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from prolix import types


class TopologyExclusions(NamedTuple):
  """Indices for topologically excluded or scaled pairs.

  Fields:
      idx_12: Indices of 1-2 pairs (bonded).
      idx_13: Indices of 1-3 pairs (separated by 2 bonds).
      idx_14: Indices of 1-4 pairs (separated by 3 bonds).
  """

  idx_12: types.Int[types.Array, "N12 2"]
  idx_13: types.Int[types.Array, "N13 2"]
  idx_14: types.Int[types.Array, "N14 2"]


def find_bonded_exclusions(
  bonds: types.ArrayLike,
  n_atoms: int,
) -> TopologyExclusions:
  r"""Find all 1-2, 1-3, and 1-4 atom pairs from a bond list.

  Process:
  1.  **Adjacency List**: Build an undirected graph from the `bonds` array.
  2.  **Breadth-First Search**: For each atom, find neighbors at distance 1, 2, and 3.
  3.  **Unique-ify**: Ensure pairs are unique and represented as $(i, j)$ where $i < j$.
  4.  **Filter**: Ensure 1-3 and 1-4 sets do not contain closer pairs (e.g., $1-3 \cap 1-2 = \emptyset$).

  Args:
      bonds: (N_bonds, 2) array of atom indices.
      n_atoms: Total number of atoms in the system.

  Returns:
      TopologyExclusions containing indices for 1-2, 1-3, and 1-4 pairs.
  """
  if bonds is None or len(bonds) == 0:
    empty = jnp.zeros((0, 2), dtype=jnp.int32)
    return TopologyExclusions(idx_12=empty, idx_13=empty, idx_14=empty)

  bonds_np = np.array(bonds)
  adj = collections.defaultdict(list)
  for b in bonds_np:
    adj[b[0]].append(b[1])
    adj[b[1]].append(b[0])

  excl_12 = []
  excl_13 = []
  excl_14 = []

  for i in range(n_atoms):
    # 1-2 pairs
    for j in adj[i]:
      if j > i:
        excl_12.append((i, j))

      # 1-3 pairs
      for k in adj[j]:
        if k == i:
          continue
        if k > i:
          excl_13.append((i, k))

        # 1-4 pairs
        for l in adj[k]:
          if l in (j, i):
            continue
          if l > i:
            excl_14.append((i, l))

  # Use sets for efficient unique-ifying and filtering
  set_12 = set(excl_12)
  set_13 = set(excl_13) - set_12
  set_14 = (set(excl_14) - set_12) - set_13

  def to_jax(s):
    if not s:
      return jnp.zeros((0, 2), dtype=jnp.int32)
    return jnp.array(list(s), dtype=jnp.int32)

  return TopologyExclusions(
    idx_12=to_jax(set_12),
    idx_13=to_jax(set_13),
    idx_14=to_jax(set_14),
  )
