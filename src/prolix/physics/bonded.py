"""Bonded potential factories for JAX MD integration."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax_md import energy, space
from jaxtyping import Array

from prolix.typing import (
    AngleIndices,
    AngleParams,
    BondIndices,
    BondParams,
    DihedralParams,
)


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


def compute_dihedral_angles(
    r: Array,
    indices: Array,
    displacement_fn: space.DisplacementFn,
) -> Array:
    """Compute dihedral angles."""
    r_i = r[indices[:, 0]]
    r_j = r[indices[:, 1]]
    r_k = r[indices[:, 2]]
    r_l = r[indices[:, 3]]

    b0 = jax.vmap(displacement_fn)(r_j, r_i)
    b1 = jax.vmap(displacement_fn)(r_k, r_j)
    b2 = jax.vmap(displacement_fn)(r_l, r_k)

    b1_norm = jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8
    b1_unit = b1 / b1_norm

    v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)
    return jnp.arctan2(y, x)


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
        if dihedral_indices.shape[0] == 0:
            return 0.0

        # components: periodicity (0), phase (1), k (2)
        n = dihedral_params[:, :, 0]
        phase = dihedral_params[:, :, 1]
        k = dihedral_params[:, :, 2]

        phi = compute_dihedral_angles(r, dihedral_indices, displacement_fn)
        # phi is (N_dih,) -> broadcast to (N_dih, 1)
        phi = phi[:, jnp.newaxis]

        # E = sum_i sum_j 0.5 * k_ij * (1 + cos(n_ij * phi_i - phase_ij))
        e_total = jnp.sum(0.5 * k * (1.0 + jnp.cos(n * phi - phase)))
        # jax.debug.print("DEBUG dihedral_fn: e_total={e}", e=e_total)
        return e_total

    return energy_fn


def make_harmonic_improper_energy_fn(
    displacement_fn: space.DisplacementFn,
    improper_indices: Array,
) -> Callable[[Array, Array], Array]:
    r"""Create a function to compute harmonic improper energy.

    Args:
        displacement_fn: JAX MD displacement function.
        improper_indices: (N_impropers, 4) array of atom indices.

    Returns:
        A function energy(R, improper_params) -> energy_scalar.
    """

    def energy_fn(r: Array, improper_params: Array, **kwargs) -> Array:
        if improper_indices.shape[0] == 0:
            return 0.0

        # improper_params: (N_imp, N_terms, 3) or (N_imp, N_terms, 2)
        # Note: Usually impropers are single-term harmonic, but proxide might pad them.
        # If they are harmonic, we expect [k, phi0] in the last dimension.
        # If they are periodic (from a force field like Amber), they might have [n, phase, k].

        # Let's check the last dimension size.
        if improper_params.shape[-1] == 3:
            # Periodic-style impropers (Amber)
            n = improper_params[:, :, 0]
            phase = improper_params[:, :, 1]
            k = improper_params[:, :, 2]
            phi = compute_dihedral_angles(r, improper_indices, displacement_fn)
            phi = phi[:, jnp.newaxis]
            return jnp.sum(0.5 * k * (1.0 + jnp.cos(n * phi - phase)))
        else:
            # Harmonic-style impropers: [k, phi0]
            k = improper_params[:, :, 0]
            phi0 = improper_params[:, :, 1]
            phi = compute_dihedral_angles(r, improper_indices, displacement_fn)
            phi = phi[:, jnp.newaxis]
            diff = phi - phi0
            diff = (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
            return jnp.sum(k * diff**2)

    return energy_fn


def make_exception_pair_energy_fn(
    displacement_fn: space.DisplacementFn,
    exception_pairs: Array,
    exception_sigmas: Array,
    exception_epsilons: Array,
    exception_chargeprods: Array,
    coulomb_constant: float = 332.0637,
) -> Callable[[Array], Array]:
    """Create energy function for explicit per-pair 1-4 exception interactions.

    Parameters are pre-scaled (already contain lj14scale / coulomb14scale).

    Args:
        displacement_fn: JAX MD displacement function.
        exception_pairs: (E, 2) int32 array of atom index pairs.
        exception_sigmas: (E,) float32 array of pre-scaled LJ sigmas.
        exception_epsilons: (E,) float32 array of pre-scaled LJ epsilons.
        exception_chargeprods: (E,) float32 array of pre-scaled charge products.
        coulomb_constant: Coulomb constant (default: 332.0637 for AKMA units).

    Returns:
        A function energy(r) -> energy_scalar computing 1-4 exception pair energies.
    """
    def energy_fn(r: Array, **kwargs) -> Array:
        if exception_pairs.shape[0] == 0:
            return jnp.array(0.0)

        ri = r[exception_pairs[:, 0]]   # (E, 3)
        rj = r[exception_pairs[:, 1]]   # (E, 3)
        dr = jax.vmap(displacement_fn)(ri, rj)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)  # (E,)

        # LJ 12-6
        inv_r = exception_sigmas / dist
        inv_r6 = inv_r**6
        e_lj = 4.0 * exception_epsilons * (inv_r6**2 - inv_r6)

        # Coulomb
        e_coul = coulomb_constant * exception_chargeprods / dist

        return jnp.sum(e_lj + e_coul)

    return energy_fn
