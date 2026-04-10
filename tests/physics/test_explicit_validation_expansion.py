"""Internal explicit-solvent parity tests (no OpenMM): NL vs dense, Flash vs system, padded vs system.

See ``explicit_solvent_progress.md`` and the validation expansion plan for context.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from prolix.batched_energy import single_padded_energy
from prolix.padding import PaddedSystem
from prolix.physics import neighbor_list as nl
from prolix.physics import pbc, system
from prolix.physics.flash_explicit import flash_explicit_energy, flash_explicit_total_energy


def _small_periodic_system_dict(
    n: int,
    charges: list[float],
    sigmas: list[float],
    epsilons: list[float],
) -> dict:
    """Minimal dict accepted by ``make_energy_fn`` (bonded terms empty)."""
    return {
        "charges": jnp.array(charges, dtype=jnp.float64),
        "sigmas": jnp.array(sigmas, dtype=jnp.float64),
        "epsilons": jnp.array(epsilons, dtype=jnp.float64),
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "exclusion_mask": jnp.ones((n, n), dtype=jnp.float64) - jnp.eye(n, dtype=jnp.float64),
    }


def _padded_from_periodic_dict(
    positions: jnp.ndarray,
    sys_dict: dict,
    *,
    box_size: jnp.ndarray,
    pme_alpha: float,
    pme_grid_points: int,
    nonbonded_cutoff: float,
) -> PaddedSystem:
    """Build a ``PaddedSystem`` aligned with a plain dict used for ``make_energy_fn``."""
    n = int(positions.shape[0])
    charges = sys_dict["charges"]
    sigmas = sys_dict["sigmas"]
    epsilons = sys_dict["epsilons"]
    spec = nl.ExclusionSpec(
        idx_12_13=jnp.zeros((0, 2), dtype=jnp.int32),
        idx_14=jnp.zeros((0, 2), dtype=jnp.int32),
        scale_14_elec=0.83333333,
        scale_14_vdw=0.5,
        n_atoms=n,
    )
    ei, sv, se = nl.map_exclusions_to_dense_padded(spec, max_exclusions=32)

    box_np: npt.NDArray[np.float64] = np.asarray(box_size, dtype=np.float64)

    radii = jnp.ones(n, dtype=jnp.float64) * 1.5
    masses = jnp.ones(n, dtype=jnp.float64) * 12.0
    elem = jnp.ones(n, dtype=jnp.int32) * 6
    mask = jnp.ones(n, dtype=jnp.bool_)

    return PaddedSystem(
        positions=positions,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=radii,
        scaled_radii=radii * 0.85,
        masses=masses,
        element_ids=elem,
        atom_mask=mask,
        is_hydrogen=jnp.zeros(n, dtype=jnp.bool_),
        is_backbone=jnp.zeros(n, dtype=jnp.bool_),
        is_heavy=mask,
        protein_atom_mask=mask,
        water_atom_mask=jnp.zeros(n, dtype=jnp.bool_),
        bonds=jnp.zeros((0, 2), dtype=jnp.int32),
        bond_params=jnp.zeros((0, 2), dtype=jnp.float64),
        bond_mask=jnp.zeros(0, dtype=jnp.bool_),
        angles=jnp.zeros((0, 3), dtype=jnp.int32),
        angle_params=jnp.zeros((0, 2), dtype=jnp.float64),
        angle_mask=jnp.zeros(0, dtype=jnp.bool_),
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((0, 3), dtype=jnp.float64),
        dihedral_mask=jnp.zeros(0, dtype=jnp.bool_),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((0, 3), dtype=jnp.float64),
        improper_mask=jnp.zeros(0, dtype=jnp.bool_),
        urey_bradley_bonds=jnp.zeros((0, 2), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((0, 2), dtype=jnp.float64),
        urey_bradley_mask=jnp.zeros(0, dtype=jnp.bool_),
        cmap_torsions=None,
        cmap_indices=None,
        cmap_mask=None,
        cmap_coeffs=None,
        excl_indices=ei,
        excl_scales_vdw=sv,
        excl_scales_elec=se,
        constraint_pairs=None,
        constraint_lengths=None,
        constraint_mask=None,
        n_real_atoms=jnp.array(n, dtype=jnp.int32),
        n_padded_atoms=n,
        bucket_size=n,
        water_indices=None,
        water_mask=None,
        box_size=box_np,
        pme_alpha=pme_alpha,
        pme_grid_points=pme_grid_points,
        nonbonded_cutoff=nonbonded_cutoff,
        dense_excl_scale_vdw=None,
        dense_excl_scale_elec=None,
    )


class TestNeighborListVsDense:
    """``make_energy_fn`` with a neighbor list should match the dense path on a small system."""

    def test_energy_and_force_rmse(self):
        jax.config.update("jax_enable_x64", True)
        n = 4
        box_size = 40.0
        box = jnp.array([box_size, box_size, box_size], dtype=jnp.float64)
        positions = jnp.array(
            [
                [10.0, 10.0, 10.0],
                [30.0, 10.0, 10.0],
                [10.0, 30.0, 10.0],
                [30.0, 30.0, 10.0],
            ],
            dtype=jnp.float64,
        )
        charges = [1.0, -1.0, 0.5, -0.5]
        sigmas = [3.0, 3.0, 3.0, 3.0]
        epsilons = [0.1, 0.1, 0.1, 0.1]
        sys_dict = _small_periodic_system_dict(n, charges, sigmas, epsilons)
        displacement_fn, _ = pbc.create_periodic_space(box)
        cutoff = 12.0
        pme_grid = 32
        alpha = 0.34

        energy_fn = system.make_energy_fn(
            displacement_fn,
            sys_dict,
            box=box,
            use_pbc=True,
            implicit_solvent=False,
            pme_grid_points=pme_grid,
            pme_alpha=alpha,
            cutoff_distance=cutoff,
            strict_parameterization=False,
        )

        neighbor_fn = nl.make_neighbor_list_fn(displacement_fn, box, cutoff)
        nbr0 = neighbor_fn.allocate(positions)
        nbr = neighbor_fn.update(positions, nbr0)

        e_dense = float(energy_fn(positions))
        e_nl = float(energy_fn(positions, neighbor=nbr))

        def e_nl_fresh(r):
            n_init = neighbor_fn.allocate(r)
            n_up = neighbor_fn.update(r, n_init)
            return energy_fn(r, neighbor=n_up)
        assert np.isclose(e_dense, e_nl, rtol=1e-4, atol=1e-3), (
            f"NL vs dense energy mismatch: dense={e_dense}, nl={e_nl}, Δ={e_dense - e_nl}"
        )

        g_dense = jax.grad(energy_fn)(positions)
        g_nl = jax.grad(e_nl_fresh)(positions)
        rmse = float(jnp.sqrt(jnp.mean((g_dense - g_nl) ** 2)))
        assert rmse < 1e-2, f"Gradient RMSE (kcal/mol/Å) too large: {rmse}"


class TestFlashVsSystem:
    """Flash explicit energy should match ``make_energy_fn`` on the same unpadded geometry."""

    def test_nonbonded_and_total_energy(self):
        jax.config.update("jax_enable_x64", True)
        n = 8
        box_size = 35.0
        box = jnp.array([box_size, box_size, box_size], dtype=jnp.float64)
        rng = np.random.RandomState(7)
        positions = jnp.asarray(rng.uniform(5.0, 30.0, (n, 3)), dtype=jnp.float64)
        charges = [0.3, -0.2, 0.1, -0.15, 0.05, -0.05, 0.12, -0.07]
        sigmas = [3.2] * n
        epsilons = [0.12] * n
        sys_dict = _small_periodic_system_dict(n, charges, sigmas, epsilons)
        displacement_fn, _ = pbc.create_periodic_space(box)
        cutoff = 10.0
        pme_grid = 32
        alpha = 0.34

        energy_fn = system.make_energy_fn(
            displacement_fn,
            sys_dict,
            box=box,
            use_pbc=True,
            implicit_solvent=False,
            pme_grid_points=pme_grid,
            pme_alpha=alpha,
            cutoff_distance=cutoff,
            strict_parameterization=False,
        )

        padded = _padded_from_periodic_dict(
            positions,
            sys_dict,
            box_size=box,
            pme_alpha=alpha,
            pme_grid_points=pme_grid,
            nonbonded_cutoff=cutoff,
        )

        e_sys = float(energy_fn(positions))
        e_flash_nb = float(flash_explicit_energy(padded, T=4))
        e_flash_tot = float(flash_explicit_total_energy(padded, T=4))

        assert np.isclose(e_sys, e_flash_nb, rtol=2e-3, atol=0.15), (
            f"Flash nonbonded vs system: sys={e_sys}, flash_nb={e_flash_nb}"
        )
        assert np.isclose(e_sys, e_flash_tot, rtol=2e-3, atol=0.15), (
            f"Flash total vs system (no bonded): sys={e_sys}, flash_tot={e_flash_tot}"
        )


class TestSinglePaddedVsSystem:
    """``single_padded_energy`` explicit PME path matches ``make_energy_fn`` on real atoms."""

    def test_total_energy(self):
        jax.config.update("jax_enable_x64", True)
        n = 4
        box_size = 40.0
        box = jnp.array([box_size, box_size, box_size], dtype=jnp.float64)
        positions = jnp.array(
            [
                [12.0, 11.0, 10.0],
                [28.0, 12.0, 11.0],
                [11.0, 29.0, 12.0],
                [27.0, 28.0, 11.0],
            ],
            dtype=jnp.float64,
        )
        charges = [1.0, -1.0, 0.4, -0.4]
        sigmas = [3.0, 3.0, 3.0, 3.0]
        epsilons = [0.1, 0.1, 0.1, 0.1]
        sys_dict = _small_periodic_system_dict(n, charges, sigmas, epsilons)
        displacement_fn, _ = pbc.create_periodic_space(box)
        cutoff = 11.0
        pme_grid = 32
        alpha = 0.34

        energy_fn = system.make_energy_fn(
            displacement_fn,
            sys_dict,
            box=box,
            use_pbc=True,
            implicit_solvent=False,
            pme_grid_points=pme_grid,
            pme_alpha=alpha,
            cutoff_distance=cutoff,
            strict_parameterization=False,
        )

        padded = _padded_from_periodic_dict(
            positions,
            sys_dict,
            box_size=box,
            pme_alpha=alpha,
            pme_grid_points=pme_grid,
            nonbonded_cutoff=cutoff,
        )

        e_sys = float(energy_fn(positions))
        e_pad = float(single_padded_energy(padded, displacement_fn, implicit_solvent=False))
        assert np.isclose(e_sys, e_pad, rtol=1e-3, atol=0.05), (
            f"single_padded_energy vs system: sys={e_sys}, padded={e_pad}, Δ={e_sys - e_pad}"
        )
