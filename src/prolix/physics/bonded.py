"""Bonded potential factories for JAX MD integration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_md import energy, space, util

from prolix.types import (
  AngleIndices,
  AngleParamsPacked,
  BondIndices,
  BondParamsPacked,
  DihedralParams,
)

Array = util.Array


def make_bond_energy_fn(
  displacement_fn: space.DisplacementFn,
  bond_indices: BondIndices,
  bond_params: BondParamsPacked,
) -> Callable[[Array], Array]:
  r"""Create a function to compute harmonic bond energy.

  Process:
  1.  **Unpack**: Transform `bond_params` array into `BondParams`.
  2.  **Spring Energy**: Invoke `jax_md.energy.simple_spring_bond`.

  Notes:
  $$ E = \frac{1}{2} k (r - r_0)^2 $$
  Where $r_0$ is the equilibrium length and $k$ is the spring constant.

  Args:
      displacement_fn: JAX MD displacement function.
      bond_indices: (N_bonds, 2) array of atom indices involved in bonds.
      bond_params: (N_bonds, 2) array of (length, k).

  Returns:
      A function energy(R) -> energy_scalar.
  """
  params = BondParams.from_row(bond_params.T)

  return energy.simple_spring_bond(
    displacement_fn,
    bond_indices,
    length=params.length,
    epsilon=params.k,
  )


def make_angle_energy_fn(
  displacement_fn: space.DisplacementFn,
  angle_indices: AngleIndices,
  angle_params: AngleParamsPacked,
) -> Callable[[Array], Array]:
  r"""Create a function to compute harmonic angle energy.

  Process:
  1.  **Unpack**: Transform `angle_params` array into `AngleParams`.
  2.  **Vectors**: Compute displacement vectors $\vec{v}_{ji}$ and $\vec{v}_{jk}$.
  3.  **Angle**: Compute $\theta = \arccos(\frac{\vec{v}_{ji} \cdot \vec{v}_{jk}}{|\vec{v}_{ji}| |\vec{v}_{jk}|})$.
  4.  **Energy**: Sum harmonic contributions $\frac{1}{2} k (\theta - \theta_0)^2$.

  Notes:
  $$ E = \frac{1}{2} \sum k (\theta - \theta_0)^2 $$

  Args:
      displacement_fn: JAX MD displacement function.
      angle_indices: (N_angles, 3) array of atom indices (i, j, k) where j is central.
      angle_params: (N_angles, 2) array of (theta0_rad, k).

  Returns:
      A function energy(R) -> energy_scalar.
  """
  params = AngleParams.from_row(angle_params.T)

  def angle_energy(r: Array, **kwargs) -> Array:
    r_i = r[angle_indices[:, 0]]
    r_j = r[angle_indices[:, 1]]
    r_k = r[angle_indices[:, 2]]

    v_ji = jax.vmap(displacement_fn)(r_i, r_j)
    v_jk = jax.vmap(displacement_fn)(r_k, r_j)

    d_ji = space.distance(v_ji)
    d_jk = space.distance(v_jk)

    denom = d_ji * d_jk + 1e-8
    cos_theta = jnp.sum(v_ji * v_jk, axis=-1) / denom
    cos_theta = jnp.clip(cos_theta, -0.999999, 0.999999)
    theta = jnp.arccos(cos_theta)

    return 0.5 * jnp.sum(params.k * (theta - params.theta0) ** 2)

  return angle_energy


def make_dihedral_energy_fn(
  displacement_fn: space.DisplacementFn,
  dihedral_indices: Array,
  dihedral_params: Array,
) -> Callable[[Array], Array]:
  r"""Create a function to compute periodic dihedral energy.

  Process:
  1.  **Unpack**: Transform `dihedral_params` into `DihedralParams`.
  2.  **Vectors**: Compute displacement vectors $\vec{b}_0, \vec{b}_1, \vec{b}_2$ along the chain.
  3.  **Orthonormal Basis**: Define a local frame using $\vec{b}_1$.
  4.  **Angle**: Compute dihedral angle $\phi$ using `atan2` on projected vectors.
  5.  **Energy**: Sum contributions $k(1 + \cos(n\phi - \gamma))$.

  Notes:
  $$ E = \sum k [1 + \cos(n\phi - \gamma)] $$

  Args:
      displacement_fn: JAX MD displacement function.
      dihedral_indices: (N_dihedrals, 4) array of atom indices (i, j, k, l).
      dihedral_params: (N_dihedrals, 3) array of (n, gamma, k).

  Returns:
      A function energy(R) -> energy_scalar.
  """
  params = DihedralParams.from_row(dihedral_params.T)

  def dihedral_energy(r: Array, **kwargs) -> Array:
    r_i = r[dihedral_indices[:, 0]]
    r_j = r[dihedral_indices[:, 1]]
    r_k = r[dihedral_indices[:, 2]]
    r_l = r[dihedral_indices[:, 3]]

    b0 = jax.vmap(displacement_fn)(r_j, r_i)
    b1 = jax.vmap(displacement_fn)(r_k, r_j)
    b2 = jax.vmap(displacement_fn)(r_l, r_k)

    b1_norm = jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8
    b1_unit = b1 / b1_norm

    v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)
    phi = jnp.arctan2(y, x)
    phi = phi - jnp.pi

    return jnp.sum(params.k * (1.0 + jnp.cos(params.periodicity * phi - params.phase)))

  return dihedral_energy
