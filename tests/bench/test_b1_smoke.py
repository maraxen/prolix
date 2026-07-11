"""B1-smoke: Nightly CI regression detector for hetero-batch benchmark.

Tests B=4 different-sized systems through EnsemblePlan.run() to verify:
1. Smoke coverage (< 10 steps) across varied system sizes
2. Wall-clock regression threshold
3. AOT-ratio alert (prereg: t_aot / t_cold < 0.5)
4. Output structure and numerical correctness

Cadence: ``@pytest.mark.slow`` + ``benchmark`` (deselected from default GitHub CI).
Prereg: ``.praxia/docs/specs/260528_b1-preregistration.md``.
"""

import time

import jax
import jax.numpy as jnp
import pytest

from prolix.api import EnsemblePlan
from prolix.types.bundles import ATOM_BUCKETS, MolecularBundle, MolecularShapeSpec, _bucket_idx


def _make_bundle(n_atoms: int, seed: int = 0) -> MolecularBundle:
    """Create a synthetic MolecularBundle with n_atoms.

    Uses random positions and unit masses. All topology arrays are empty.
    For smoke testing hetero-batch decisions in EnsemblePlan.

    Args:
        n_atoms: Number of atoms (will be padded to nearest ATOM_BUCKET)
        seed: PRNG seed for position initialization

    Returns:
        MolecularBundle with n_atoms atoms, all other topology empty
    """
    key = jnp.ones((2,), dtype=jnp.uint32) * seed

    # Determine bucket index and padded size
    atom_bucket_idx = _bucket_idx(n_atoms, ATOM_BUCKETS)
    padded_n_atoms = ATOM_BUCKETS[atom_bucket_idx]

    # Random positions (normalized to unit cube)
    import jax.random

    key, subkey = jax.random.split(key)
    positions = jax.random.normal(subkey, (padded_n_atoms, 3), dtype=jnp.float32)
    positions = positions.at[n_atoms:].set(0.0)  # Pad unused rows

    # Atom mask: mark first n_atoms as real
    atom_mask = jnp.arange(padded_n_atoms) < n_atoms

    # Unit charges, sigmas, epsilons, radii
    charges = jnp.zeros(padded_n_atoms, dtype=jnp.float32)
    sigmas = jnp.ones(padded_n_atoms, dtype=jnp.float32) * 3.15
    epsilons = jnp.zeros(padded_n_atoms, dtype=jnp.float32)
    radii = jnp.ones(padded_n_atoms, dtype=jnp.float32)
    scaled_radii = jnp.ones(padded_n_atoms, dtype=jnp.float32)

    # Empty topology arrays (all padded to smallest bucket)
    empty_bond = jnp.zeros((ATOM_BUCKETS[0], 2), dtype=jnp.int32)
    empty_bond_params = jnp.zeros((ATOM_BUCKETS[0], 2), dtype=jnp.float32)
    empty_bond_mask = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.bool_)

    empty_angle = jnp.zeros((ATOM_BUCKETS[0], 3), dtype=jnp.int32)
    empty_angle_params = jnp.zeros((ATOM_BUCKETS[0], 2), dtype=jnp.float32)
    empty_angle_mask = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.bool_)

    empty_dihedral = jnp.zeros((ATOM_BUCKETS[0], 4), dtype=jnp.int32)
    empty_dihedral_params = jnp.zeros((ATOM_BUCKETS[0], 4), dtype=jnp.float32)
    empty_dihedral_mask = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.bool_)

    empty_improper = jnp.zeros((ATOM_BUCKETS[0], 4), dtype=jnp.int32)
    empty_improper_params = jnp.zeros((ATOM_BUCKETS[0], 3), dtype=jnp.float32)
    empty_improper_mask = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.bool_)

    empty_ub = jnp.zeros((ATOM_BUCKETS[0], 3), dtype=jnp.int32)
    empty_ub_params = jnp.zeros((ATOM_BUCKETS[0], 2), dtype=jnp.float32)
    empty_ub_mask = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.bool_)

    empty_cmap = jnp.zeros((8, 24, 24), dtype=jnp.float32)
    empty_cmap_mask = jnp.zeros(8, dtype=jnp.bool_)

    empty_water = jnp.zeros((8, 3), dtype=jnp.int32)
    empty_water_mask = jnp.zeros(8, dtype=jnp.bool_)

    empty_excl = jnp.zeros((ATOM_BUCKETS[0], 2), dtype=jnp.int32)
    empty_excl_vdw = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.float32)
    empty_excl_elec = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.float32)
    empty_excl_mask = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.bool_)

    empty_exc = jnp.zeros((ATOM_BUCKETS[0], 2), dtype=jnp.int32)
    empty_exc_sigma = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.float32)
    empty_exc_epsilon = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.float32)
    empty_exc_charge = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.float32)
    empty_exc_mask = jnp.zeros(ATOM_BUCKETS[0], dtype=jnp.bool_)

    shape_spec = MolecularShapeSpec(
        atom_bucket_idx=atom_bucket_idx,
        bond_bucket_idx=0,
        angle_bucket_idx=0,
        dihedral_bucket_idx=0,
        water_bucket_idx=0,
        excl_bucket_idx=0,
        cmap_bucket_idx=0,
        exception_bucket_idx=0,
        has_pbc=False,
        has_implicit_solvent=False,
        boundary_condition="free",
    )

    return MolecularBundle(
        positions=positions,
        masses=jnp.ones_like(charges),
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=radii,
        scaled_radii=scaled_radii,
        atom_mask=atom_mask,
        n_atoms=jnp.array(n_atoms, dtype=jnp.int32),
        box=jnp.zeros((3, 3), dtype=jnp.float32),
        bond_idx=empty_bond,
        bond_params=empty_bond_params,
        bond_mask=empty_bond_mask,
        n_bonds=jnp.array(0, dtype=jnp.int32),
        angle_idx=empty_angle,
        angle_params=empty_angle_params,
        angle_mask=empty_angle_mask,
        n_angles=jnp.array(0, dtype=jnp.int32),
        dihedral_idx=empty_dihedral,
        dihedral_params=empty_dihedral_params,
        dihedral_mask=empty_dihedral_mask,
        n_dihedrals=jnp.array(0, dtype=jnp.int32),
        improper_idx=empty_improper,
        improper_params=empty_improper_params,
        improper_mask=empty_improper_mask,
        improper_is_periodic=jnp.array(False, dtype=jnp.bool_),
        n_impropers=jnp.array(0, dtype=jnp.int32),
        urey_bradley_idx=empty_ub,
        urey_bradley_params=empty_ub_params,
        urey_bradley_mask=empty_ub_mask,
        n_urey_bradley=jnp.array(0, dtype=jnp.int32),
        cmap_torsion_idx=jnp.zeros((8, 8), dtype=jnp.int32),
        cmap_energy_grids=empty_cmap,
        cmap_mask=empty_cmap_mask,
        n_cmap=jnp.array(0, dtype=jnp.int32),
        water_indices=empty_water,
        water_mask=empty_water_mask,
        n_waters=jnp.array(0, dtype=jnp.int32),
        excl_indices=empty_excl,
        excl_scales_vdw=empty_excl_vdw,
        excl_scales_elec=empty_excl_elec,
        excl_mask=empty_excl_mask,
        n_excl=jnp.array(0, dtype=jnp.int32),
        exception_pairs=empty_exc,
        exception_sigmas=empty_exc_sigma,
        exception_epsilons=empty_exc_epsilon,
        exception_chargeprods=empty_exc_charge,
        exception_mask=empty_exc_mask,
        n_exception_pairs=jnp.array(0, dtype=jnp.int32),
        pme_alpha=jnp.array(0.3, dtype=jnp.float32),
        cutoff_distance=jnp.array(9.0, dtype=jnp.float32),
        shape_spec=shape_spec,
    )


def _block_trajs(trajectories: list) -> None:
    """Force device sync so wall-clock includes XLA work."""
    for traj in trajectories:
        jax.block_until_ready(traj.positions)


@pytest.mark.slow
@pytest.mark.benchmark
class TestB1Smoke:
    """B1-smoke: Hetero-batch regression + AOT-ratio alert (prereg 260528).

    Tests EnsemblePlan.from_bundles(...).run() with B=4 varied-size systems to:
    1. Execute 10 steps without NaN/inf
    2. Catch wall-clock regressions (cold run < 300s on CPU)
    3. Alert if approximate AOT share of cold time is ≥ 0.5
    4. Validate output structure across varied N
    """

    def test_b1_smoke_b4_wall_clock(self):
        """B1-smoke: 4 varied-size bundles, cold/warm timing, AOT-ratio gate.

        Creates 4 bundles with 5, 10, 20, 35 atoms. Runs 10 steps of NVT via
        ``EnsemblePlan.from_bundles`` with ``dt=0.5`` **fs** (XR-VACUUM-DT).

        AOT proxy (prereg success criterion 3 / R4 alert):
        ``aot_ratio = (t_cold - t_warm) / t_cold`` must be ``< 0.5``.
        """
        bundles = [_make_bundle(n) for n in (5, 10, 20, 35)]
        plan = EnsemblePlan.from_bundles(bundles)

        t0 = time.perf_counter()
        trajectories = plan.run(n_steps=10, dt=0.5, kT=2.479e-3, seed=42)
        _block_trajs(trajectories)
        t_cold = time.perf_counter() - t0

        assert isinstance(trajectories, list)
        assert len(trajectories) == 4

        for i, traj in enumerate(trajectories):
            n_atoms = bundles[i].n_atoms.item()

            assert traj.n_steps == 10, f"Bundle {i}: expected 10 steps, got {traj.n_steps}"
            assert traj.positions.shape == (10, n_atoms, 3), (
                f"Bundle {i}: expected (10, {n_atoms}, 3), got {traj.positions.shape}"
            )
            assert isinstance(traj.observable_values, dict), (
                f"Bundle {i}: observable_values not a dict"
            )
            assert jnp.all(jnp.isfinite(traj.positions)), (
                f"Bundle {i}: non-finite positions detected (NaN or inf)"
            )
            assert not jnp.allclose(traj.positions[0], traj.positions[-1]), (
                f"Bundle {i}: positions unchanged after 10 steps"
            )

        t1 = time.perf_counter()
        warm = plan.run(n_steps=10, dt=0.5, kT=2.479e-3, seed=43)
        _block_trajs(warm)
        t_warm = time.perf_counter() - t1

        assert t_cold < 300.0, (
            f"B1-smoke timed out: t_cold={t_cold:.1f}s (expected < 300s on CPU)"
        )
        assert t_cold > 0.0
        aot_ratio = max(0.0, (t_cold - t_warm) / t_cold)
        print(
            f"\nB1-smoke t_cold={t_cold:.3f}s t_warm={t_warm:.3f}s "
            f"aot_ratio={aot_ratio:.3f} (B=4 x 10 steps)"
        )
        assert aot_ratio < 0.5, (
            f"B1-smoke AOT-ratio alert: aot_ratio={aot_ratio:.3f} ≥ 0.5 "
            f"(t_cold={t_cold:.3f}s, t_warm={t_warm:.3f}s). "
            "Escalate to R4 / AOT-budget before Claim-1 headline (prereg 260528)."
        )
