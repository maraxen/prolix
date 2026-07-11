"""Tests for EnsemblePlan stub."""

import pytest
import jax.numpy as jnp
from prolix.api import EnsemblePlan, Trajectory
from prolix.types.bundles import MolecularBundle, MolecularShapeSpec


class FakePlanner:
    """Stub planner for testing constructor."""

    def plan(self, bundles):
        """Return None (deferred behavior)."""
        return None


def _make_minimal_bundle(n_atoms=3) -> MolecularBundle:
    """Create a minimal MolecularBundle for testing (water molecule or similar).

    Args:
        n_atoms: Number of atoms (default 3 for water)

    Returns:
        MolecularBundle with minimal topology
    """
    # Create a minimal bundle directly with padded arrays
    # For simplicity: 3-atom system with no bonds, angles, dihedrals, etc.

    # Pad to smallest bucket sizes
    from prolix.types.bundles import ATOM_BUCKETS, _bucket_idx

    atom_bucket_idx = _bucket_idx(n_atoms, ATOM_BUCKETS)
    padded_n_atoms = ATOM_BUCKETS[atom_bucket_idx]

    positions = jnp.zeros((padded_n_atoms, 3), dtype=jnp.float32)
    positions = positions.at[:n_atoms].set(jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=jnp.float32))

    charges = jnp.zeros(padded_n_atoms, dtype=jnp.float32)
    charges = charges.at[:n_atoms].set(jnp.array([0.8, -0.4, -0.4], dtype=jnp.float32))

    sigmas = jnp.ones(padded_n_atoms, dtype=jnp.float32) * 3.15
    epsilons = jnp.zeros(padded_n_atoms, dtype=jnp.float32)
    radii = jnp.ones(padded_n_atoms, dtype=jnp.float32)
    scaled_radii = jnp.ones(padded_n_atoms, dtype=jnp.float32)

    atom_mask = jnp.zeros(padded_n_atoms, dtype=jnp.bool_)
    atom_mask = atom_mask.at[:n_atoms].set(True)

    # Empty topology arrays
    empty_bond = jnp.zeros((8, 2), dtype=jnp.int32)
    empty_bond_params = jnp.zeros((8, 2), dtype=jnp.float32)
    empty_bond_mask = jnp.zeros(8, dtype=jnp.bool_)

    empty_angle = jnp.zeros((32, 3), dtype=jnp.int32)
    empty_angle_params = jnp.zeros((32, 2), dtype=jnp.float32)
    empty_angle_mask = jnp.zeros(32, dtype=jnp.bool_)

    empty_dihedral = jnp.zeros((32, 4), dtype=jnp.int32)
    empty_dihedral_params = jnp.zeros((32, 4), dtype=jnp.float32)
    empty_dihedral_mask = jnp.zeros(32, dtype=jnp.bool_)

    empty_improper = jnp.zeros((32, 4), dtype=jnp.int32)
    empty_improper_params = jnp.zeros((32, 3), dtype=jnp.float32)
    empty_improper_mask = jnp.zeros(32, dtype=jnp.bool_)

    empty_ub = jnp.zeros((32, 3), dtype=jnp.int32)
    empty_ub_params = jnp.zeros((32, 2), dtype=jnp.float32)
    empty_ub_mask = jnp.zeros(32, dtype=jnp.bool_)

    empty_cmap = jnp.zeros((8, 24, 24), dtype=jnp.float32)
    empty_cmap_mask = jnp.zeros(8, dtype=jnp.bool_)

    empty_water = jnp.zeros((8, 3), dtype=jnp.int32)
    empty_water_mask = jnp.zeros(8, dtype=jnp.bool_)

    empty_excl = jnp.zeros((32, 2), dtype=jnp.int32)
    empty_excl_vdw = jnp.zeros(32, dtype=jnp.float32)
    empty_excl_elec = jnp.zeros(32, dtype=jnp.float32)
    empty_excl_mask = jnp.zeros(32, dtype=jnp.bool_)

    empty_exc = jnp.zeros((32, 2), dtype=jnp.int32)
    empty_exc_sigma = jnp.zeros(32, dtype=jnp.float32)
    empty_exc_epsilon = jnp.zeros(32, dtype=jnp.float32)
    empty_exc_charge = jnp.zeros(32, dtype=jnp.float32)
    empty_exc_mask = jnp.zeros(32, dtype=jnp.bool_)

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
        boundary_condition='free',
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


def test_ensemble_plan_construction():
    """Test EnsemblePlan can be constructed with empty bundle list."""
    ep = EnsemblePlan([])
    assert hasattr(ep, "bundles")
    assert ep.bundles == []


def test_ensemble_plan_run_requires_bundle():
    """Test EnsemblePlan.run() raises ValueError when no bundles provided."""
    ep = EnsemblePlan([])
    with pytest.raises(ValueError) as excinfo:
        ep.run(n_steps=10, dt=0.5, kT=2.479e-3)

    # Verify error message
    assert "at least one bundle" in str(excinfo.value)


def test_ensemble_plan_with_planner():
    """Test EnsemblePlan stores batch_plan when planner provided."""
    fake_planner = FakePlanner()
    ep = EnsemblePlan([], planner=fake_planner)
    assert ep.batch_plan is None


def test_ensemble_plan_with_none_planner():
    """Test EnsemblePlan sets batch_plan to None when planner is None."""
    ep = EnsemblePlan([], planner=None)
    assert ep.batch_plan is None


def test_ensemble_plan_run_returns_trajectory():
    """Test that EnsemblePlan.run() returns a Trajectory object."""
    # Create minimal bundle
    bundle = _make_minimal_bundle(n_atoms=3)
    ep = EnsemblePlan([bundle])

    # Run simulation
    trajectory = ep.run(n_steps=5, dt=0.5, kT=0.596, seed=42)

    # Check return type and shape
    assert isinstance(trajectory, Trajectory)
    assert hasattr(trajectory, 'positions')
    assert hasattr(trajectory, 'observable_values')
    assert hasattr(trajectory, 'n_steps')
    assert trajectory.n_steps == 5
    assert trajectory.positions.shape == (5, 3, 3)  # (steps, atoms, 3)


def test_ensemble_plan_run_single_bundle_parity():
    """Test that single-bundle run produces reasonable positions."""
    bundle = _make_minimal_bundle(n_atoms=3)
    ep = EnsemblePlan([bundle])

    # Run short simulation
    trajectory = ep.run(n_steps=5, dt=0.5, kT=0.596, seed=42)

    # Verify positions changed (not just returned initial)
    assert not jnp.allclose(trajectory.positions[0], trajectory.positions[-1])

    # Verify all positions are finite
    assert jnp.all(jnp.isfinite(trajectory.positions))


class TestFromBundleConstructor:
    """Tests for EnsemblePlan.from_bundle classmethod."""

    def test_from_bundle_creates_single_bundle_plan(self) -> None:
        """from_bundle(b) should set self.bundles = [b]."""
        b = _make_minimal_bundle()
        ep = EnsemblePlan.from_bundle(b)
        assert len(ep.bundles) == 1
        assert ep.bundles[0] is b
        assert ep.batch_plan is None

    def test_from_bundle_with_planner_calls_plan(self) -> None:
        """from_bundle with planner should call planner.plan([bundle])."""
        from unittest.mock import MagicMock
        b = _make_minimal_bundle()
        planner = MagicMock()
        planner.plan.return_value = "mock_plan"
        ep = EnsemblePlan.from_bundle(b, planner=planner)
        planner.plan.assert_called_once()
        # Verify the argument is a list with the bundle
        call_args = planner.plan.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0] is b
        assert ep.batch_plan == "mock_plan"


class TestFromBundlesConstructor:
    """Tests for EnsemblePlan.from_bundles classmethod."""

    def test_from_bundles_creates_multi_bundle_plan(self) -> None:
        """from_bundles([b1, b2]) should set self.bundles = [b1, b2]."""
        b1, b2 = _make_minimal_bundle(), _make_minimal_bundle()
        ep = EnsemblePlan.from_bundles([b1, b2])
        assert len(ep.bundles) == 2
        assert ep.bundles[0] is b1
        assert ep.bundles[1] is b2
        assert ep.batch_plan is not None

    def test_from_bundles_with_planner_calls_plan(self) -> None:
        """from_bundles with planner should call planner.plan(bundles)."""
        from unittest.mock import MagicMock
        b1, b2 = _make_minimal_bundle(), _make_minimal_bundle()
        planner = MagicMock()
        planner.plan.return_value = "mock_plan"
        ep = EnsemblePlan.from_bundles([b1, b2], planner=planner)
        planner.plan.assert_called_once()
        # Verify the argument is the list of bundles
        call_args = planner.plan.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0] is b1
        assert call_args[1] is b2
        assert ep.batch_plan == "mock_plan"

    def test_from_bundles_single_bundle(self) -> None:
        """from_bundles with one bundle should work."""
        b = _make_minimal_bundle()
        ep = EnsemblePlan.from_bundles([b])
        assert len(ep.bundles) == 1
        assert ep.bundles[0] is b
        assert ep.batch_plan is None


class TestMultiBundleRun:
    """Multi-bundle dispatch via xtrax-backed planner (#1842)."""

    def test_multi_bundle_run_returns_trajectory_list(self) -> None:
        b1, b2 = _make_minimal_bundle(), _make_minimal_bundle()
        ep = EnsemblePlan.from_bundles([b1, b2])
        assert ep.batch_plan is not None
        result = ep.run(n_steps=3, dt=0.5, kT=0.596, seed=42)
        assert isinstance(result, list)
        assert len(result) == 2
        for traj in result:
            assert traj.n_steps == 3
            assert traj.positions.shape[0] == 3
            assert jnp.all(jnp.isfinite(traj.positions))
