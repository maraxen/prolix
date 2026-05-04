"""Bonded potential factories for JAX MD integration."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax_md import energy, space, util

from prolix.types import (
    AngleIndices,
    AngleParams,
    BondIndices,
    BondParams,
    DihedralParams,
)

Array = util.Array


class DifferentiableParams:
    """Container for parameters that are differentiable with respect to force fields."""

    def __init__(self, bond_params: Array, angle_params: Array, dihedral_params: Array):
        self.bond_params = bond_params  # (N_bonds, 2)
        self.angle_params = angle_params  # (N_angles, 2)
        self.dihedral_params = dihedral_params  # (N_dihedrals, 3)


def make_bond_energy_fn(
    displacement_fn: space.DisplacementFn,
    bond_indices: BondIndices,
) -> Callable[[Array, Array], Array]:
    r"""Create a function to compute harmonic bond energy.

    Args:
        displacement_fn: JAX MD displacement function.
        bond_indices: (N_bonds, 2) array of atom indices.

    Returns:
        A function energy(R, bond_params) -> energy_scalar.
    """

    def energy_fn(r: Array, bond_params: Array, **kwargs) -> Array:
        # bond_params: (N_bonds, 2)
        p = BondParams.from_row(bond_params.T)
        return energy.simple_spring_bond(
            displacement_fn,
            bond_indices,
            length=p.length,
            epsilon=p.k,
        )(r)

    return energy_fn


def make_angle_energy_fn(
    displacement_fn: space.DisplacementFn,
    angle_indices: AngleIndices,
) -> Callable[[Array, Array], Array]:
    r"""Create a function to compute harmonic angle energy.

    Args:
        displacement_fn: JAX MD displacement function.
        angle_indices: (N_angles, 3) array of atom indices.

    Returns:
        A function energy(R, angle_params) -> energy_scalar.
    """

    def energy_fn(r: Array, angle_params: Array, **kwargs) -> Array:
        # angle_params: (N_angles, 2)
        p = AngleParams.from_row(angle_params.T)
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

        return 0.5 * jnp.sum(p.k * (theta - p.theta0) ** 2)

    return energy_fn


def make_dihedral_energy_fn(
    displacement_fn: space.DisplacementFn,
    dihedral_indices: Array,
) -> Callable[[Array, Array], Array]:
    r"""Create a function to compute periodic dihedral energy.

    Args:
        displacement_fn: JAX MD displacement function.
        dihedral_indices: (N_dihedrals, 4) array of atom indices.

    Returns:
        A function energy(R, dihedral_params) -> energy_scalar.
    """

    def energy_fn(r: Array, dihedral_params: Array, **kwargs) -> Array:
        # dihedral_params: (N_dihedrals, 3)
        p = DihedralParams.from_row(dihedral_params.T)
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

        return jnp.sum(p.k * (1.0 + jnp.cos(p.periodicity * phi - p.phase)))

    return energy_fn
