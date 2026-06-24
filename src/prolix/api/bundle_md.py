"""MolecularBundle-backed energy and displacement helpers for EnsemblePlan.run()."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax_md import space

from prolix.batched_energy import (
    _angle_energy_masked,
    _bond_energy_masked,
    _dihedral_energy_masked,
)
from prolix.types.bundles import MolecularBundle


def displacement_fn_for_bundle(
    bundle: MolecularBundle,
) -> tuple[space.DisplacementFn, space.ShiftFn]:
    """Reconstruct JAX-MD displacement and shift from bundle shape_spec."""
    spec = bundle.shape_spec
    if spec.boundary_condition == "periodic" and spec.has_pbc:
        box_vec = jnp.diag(bundle.box)
        return space.periodic(box_vec)
    return space.free()


def tip3p_masses(n_waters: int, dtype=jnp.float64) -> jnp.ndarray:
    """TIP3P O-H-H masses for ``n_waters`` molecules."""
    per = jnp.array([15.999, 1.008, 1.008], dtype=dtype)
    return jnp.tile(per, n_waters)


def masses_for_bundle(bundle: MolecularBundle) -> jnp.ndarray:
    """Infer atomic masses from bundle water count (TIP3P) or unit mass."""
    n_atoms = int(bundle.n_atoms)
    n_waters = int(bundle.n_waters)
    if n_waters > 0 and n_atoms == 3 * n_waters:
        return tip3p_masses(n_waters, dtype=bundle.positions.dtype)
    return jnp.ones(n_atoms, dtype=bundle.positions.dtype)


def active_positions(bundle: MolecularBundle) -> jnp.ndarray:
    """Real atom positions (prefix-packed by make_bundle_from_system)."""
    return bundle.positions[: int(bundle.n_atoms)]


def bonded_energy_fn_from_bundle(
    bundle: MolecularBundle,
) -> Callable[..., jnp.ndarray]:
    """Bonded-only energy callable compatible with settle_langevin."""
    disp_fn, _ = displacement_fn_for_bundle(bundle)

    def energy_fn(positions: jnp.ndarray, **kwargs: object) -> jnp.ndarray:
        del kwargs
        r = positions
        mask_dtype = r.dtype
        e = jnp.array(0.0, dtype=r.dtype)
        e = e + _bond_energy_masked(
            r,
            bundle.bond_idx,
            bundle.bond_params,
            bundle.bond_mask.astype(mask_dtype),
            disp_fn,
        )
        e = e + _angle_energy_masked(
            r,
            bundle.angle_idx,
            bundle.angle_params,
            bundle.angle_mask.astype(mask_dtype),
            disp_fn,
        )
        dih_params = bundle.dihedral_params[..., :3]
        e = e + _dihedral_energy_masked(
            r,
            bundle.dihedral_idx,
            dih_params,
            bundle.dihedral_mask.astype(mask_dtype),
            disp_fn,
        )
        e = e + _dihedral_energy_masked(
            r,
            bundle.improper_idx,
            bundle.improper_params,
            bundle.improper_mask.astype(mask_dtype),
            disp_fn,
        )
        return e

    return energy_fn
