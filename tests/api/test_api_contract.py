"""A1 API contract acceptance test for prolix.api surface.

Comprehensive verification of the prolix.api contract including canonical exports,
observable protocol conformance, trajectory structure, ensemble plan construction,
and observable computation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from prolix.api import EnsemblePlan, Observable, Trajectory, Temperature, Energy
from prolix.types.bundles import MolecularBundle, MolecularShapeSpec
from prolix.typing import IntegratorState as LangevinState
from prolix.simulate import BOLTZMANN_KCAL


def _make_minimal_bundle(n_atoms: int = 3) -> MolecularBundle:
    """Create a minimal MolecularBundle for testing.

    Args:
        n_atoms: Number of atoms (default 3 for water)

    Returns:
        MolecularBundle with minimal topology
    """
    from prolix.types.bundles import ATOM_BUCKETS, _bucket_idx

    atom_bucket_idx = _bucket_idx(n_atoms, ATOM_BUCKETS)
    padded_n_atoms = ATOM_BUCKETS[atom_bucket_idx]

    positions = jnp.zeros((padded_n_atoms, 3), dtype=jnp.float32)
    positions = positions.at[:n_atoms].set(
        jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32
        )
    )

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
        boundary_condition="free",
    )

    return MolecularBundle(
        positions=positions,
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


class TestApiContract:
    """Comprehensive acceptance tests for prolix.api surface contract."""

    def test_canonical_imports(self):
        """Test that all expected symbols can be imported from prolix.api.

        Verifies the canonical export surface includes:
          - EnsemblePlan
          - Observable
          - Trajectory
          - Temperature
          - Energy
        """
        # All imports should succeed without error
        from prolix.api import EnsemblePlan, Observable, Trajectory, Temperature, Energy

        # All symbols should be non-None
        assert EnsemblePlan is not None
        assert Observable is not None
        assert Trajectory is not None
        assert Temperature is not None
        assert Energy is not None

        # Verify they are the correct types
        assert callable(EnsemblePlan), "EnsemblePlan should be callable"
        assert callable(Trajectory), "Trajectory should be callable"
        assert callable(Temperature), "Temperature should be callable"
        assert callable(Energy), "Energy should be callable"

    def test_observable_protocol_conformance(self):
        """Test that concrete observable classes implement the Observable protocol.

        Uses isinstance() with the @runtime_checkable Observable Protocol
        to verify conformance at runtime.
        """
        # Temperature should implement Observable
        temp = Temperature(dof=9)
        assert isinstance(
            temp, Observable
        ), "Temperature should implement Observable protocol"

        # Energy should implement Observable
        def mock_energy_fn(positions, bundle):
            return jnp.sum(positions**2)

        energy = Energy(energy_fn=mock_energy_fn, bundle={})
        assert isinstance(
            energy, Observable
        ), "Energy should implement Observable protocol"

        # Both should have compute method
        assert hasattr(temp, "compute"), "Temperature should have compute method"
        assert callable(
            temp.compute
        ), "Temperature.compute should be callable"

        assert hasattr(energy, "compute"), "Energy should have compute method"
        assert callable(
            energy.compute
        ), "Energy.compute should be callable"

    def test_trajectory_fields(self):
        """Test that Trajectory has correct structure and field types.

        Verifies:
          - positions field exists and has shape (n_steps, n_atoms, 3)
          - observable_values is a dict
          - n_steps is set and equals declared value
        """
        n_steps = 5
        n_atoms = 3
        positions = jnp.ones((n_steps, n_atoms, 3), dtype=jnp.float32)
        observable_values = {"energy": jnp.ones(n_steps), "temperature": jnp.ones(n_steps)}

        traj = Trajectory(
            positions=positions, observable_values=observable_values, n_steps=n_steps
        )

        # Check positions field
        assert hasattr(traj, "positions"), "Trajectory should have positions field"
        assert traj.positions.shape == (n_steps, n_atoms, 3), \
            f"positions shape should be ({n_steps}, {n_atoms}, 3), got {traj.positions.shape}"

        # Check observable_values field
        assert hasattr(
            traj, "observable_values"
        ), "Trajectory should have observable_values field"
        assert isinstance(
            traj.observable_values, dict
        ), "observable_values should be a dict"
        assert len(traj.observable_values) == 2
        assert "energy" in traj.observable_values
        assert "temperature" in traj.observable_values

        # Check n_steps field
        assert hasattr(traj, "n_steps"), "Trajectory should have n_steps field"
        assert traj.n_steps == n_steps, f"n_steps should be {n_steps}, got {traj.n_steps}"

    def test_trajectory_is_equinox_module(self):
        """Test that Trajectory is an equinox.Module."""
        positions = jnp.ones((5, 3, 3))
        observable_values = {}

        traj = Trajectory(positions=positions, observable_values=observable_values, n_steps=5)

        assert isinstance(
            traj, eqx.Module
        ), "Trajectory should be an equinox.Module instance"
        assert issubclass(Trajectory, eqx.Module), "Trajectory class should inherit from eqx.Module"

    def test_ensemble_plan_construction(self):
        """Test EnsemblePlan can be constructed with a minimal bundle.

        Verifies:
          - Construction without error
          - bundles attribute is set correctly
          - batch_plan is None when no planner provided
        """
        bundle = _make_minimal_bundle(n_atoms=3)
        ep = EnsemblePlan([bundle])

        assert hasattr(ep, "bundles"), "EnsemblePlan should have bundles attribute"
        assert len(ep.bundles) == 1, "bundles list should have 1 element"
        assert ep.bundles[0] is bundle, "first bundle should match input"

        assert hasattr(ep, "batch_plan"), "EnsemblePlan should have batch_plan attribute"
        assert ep.batch_plan is None, "batch_plan should be None when no planner provided"

    def test_ensemble_plan_run_returns_trajectory(self):
        """Test EnsemblePlan.run() returns a Trajectory with correct structure.

        Verifies:
          - Returns Trajectory instance
          - trajectory.n_steps matches requested n_steps
          - positions.shape[0] equals n_steps
          - All positions are finite (no NaN or inf)
        """
        bundle = _make_minimal_bundle(n_atoms=3)
        ep = EnsemblePlan([bundle])

        # Run simulation
        n_steps = 5
        trajectory = ep.run(n_steps=n_steps, dt=0.5, kT=2.479e-3, seed=42)

        # Check return type
        assert isinstance(
            trajectory, Trajectory
        ), "run() should return a Trajectory instance"

        # Check n_steps
        assert (
            trajectory.n_steps == n_steps
        ), f"trajectory.n_steps should be {n_steps}, got {trajectory.n_steps}"

        # Check positions shape
        assert trajectory.positions.shape[0] == n_steps, \
            f"positions.shape[0] should be {n_steps}, got {trajectory.positions.shape[0]}"

        assert trajectory.positions.shape[1] == 3, \
            f"positions.shape[1] should be 3 atoms, got {trajectory.positions.shape[1]}"

        assert trajectory.positions.shape[2] == 3, \
            f"positions.shape[2] should be 3 coords, got {trajectory.positions.shape[2]}"

        # Check finiteness
        assert jnp.all(jnp.isfinite(trajectory.positions)), \
            "All positions should be finite (no NaN or inf)"

    def test_ensemble_plan_run_trajectory_evolution(self):
        """Test that EnsemblePlan.run() produces evolving trajectory.

        Verifies that the returned trajectory shows position evolution
        (not just returning initial positions unchanged).
        """
        bundle = _make_minimal_bundle(n_atoms=3)
        ep = EnsemblePlan([bundle])

        trajectory = ep.run(n_steps=5, dt=0.5, kT=2.479e-3, seed=42)

        # Positions should evolve (not all the same)
        initial_pos = trajectory.positions[0]
        final_pos = trajectory.positions[-1]

        # They should not be identical (trajectory should evolve)
        # Use loose tolerance to allow for short integration times
        assert not jnp.allclose(
            initial_pos, final_pos, atol=1e-6
        ), "Trajectory positions should evolve over steps"

    def test_temperature_observable_compute(self):
        """Test Temperature observable compute() implementation.

        Verifies:
          - compute() returns a scalar
          - compute() result is finite for valid state
          - Formula T = (2*KE) / (dof * k_B) is correct
        """
        n_atoms = 3
        dof = 3 * n_atoms  # All atoms free

        # Create state with unit momentum and mass
        positions = jnp.zeros((n_atoms, 3))
        momentum = jnp.ones((n_atoms, 3))  # p_i = 1
        force = jnp.zeros((n_atoms, 3))
        mass = jnp.ones(n_atoms)
        key = jnp.zeros(2, dtype=jnp.uint32)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        # Compute temperature
        t_obs = Temperature(dof=dof)
        temp = t_obs.compute(state)

        # Check return type and shape
        assert isinstance(
            temp, jnp.ndarray
        ), "Temperature.compute() should return a JAX array"
        assert temp.shape == (), f"compute() should return a scalar, got shape {temp.shape}"

        # Check finiteness
        assert jnp.isfinite(temp), "Temperature should be finite for valid state"

        # Verify formula
        # KE = sum(p^2 / (2*m)) = 9 * (1 / 2) = 4.5
        # T = (2 * 4.5) / (9 * k_B)
        expected_temp = (2.0 * 4.5) / (dof * BOLTZMANN_KCAL)
        assert jnp.allclose(
            temp, expected_temp, rtol=1e-5
        ), f"Expected temp {expected_temp}, got {temp}"

    def test_energy_observable_compute(self):
        """Test Energy observable compute() implementation.

        Verifies:
          - compute() returns a scalar
          - compute() correctly passes state.positions to energy_fn
          - compute() result matches direct energy_fn call
        """
        def mock_energy_fn(positions, bundle):
            return jnp.sum(positions**2)

        bundle = {}
        energy = Energy(energy_fn=mock_energy_fn, bundle=bundle)

        # Create mock state
        class MockState:
            def __init__(self, positions):
                self.positions = positions

        positions = jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        state = MockState(positions)

        # Compute via observable
        e_obs = energy.compute(state)

        # Check return type
        assert isinstance(
            e_obs, jnp.ndarray
        ), "Energy.compute() should return a JAX array"
        assert e_obs.shape == (), f"compute() should return a scalar, got shape {e_obs.shape}"

        # Compute directly and compare
        e_direct = mock_energy_fn(positions, bundle)
        assert jnp.allclose(
            e_obs, e_direct
        ), f"Observable energy {e_obs} != direct {e_direct}"

    def test_observable_protocol_has_compute_method(self):
        """Test that Observable protocol requires compute() method."""
        # Create a class that implements the Observable protocol
        class CustomObservable(eqx.Module):
            def compute(self, state):
                return jnp.array(42.0)

        obs = CustomObservable()
        assert isinstance(
            obs, Observable
        ), "Custom class with compute() should satisfy Observable protocol"

    def test_trajectory_with_empty_observables(self):
        """Test Trajectory can be constructed with empty observable_values."""
        positions = jnp.ones((3, 2, 3))
        traj = Trajectory(positions=positions, observable_values={}, n_steps=3)

        assert traj.observable_values == {}
        assert traj.n_steps == 3
        assert traj.positions.shape == (3, 2, 3)

    def test_ensemble_plan_run_with_different_parameters(self):
        """Test EnsemblePlan.run() with various parameter combinations.

        Verifies robustness with different n_steps, dt, kT values.
        """
        bundle = _make_minimal_bundle(n_atoms=3)
        ep = EnsemblePlan([bundle])

        # Test with minimal steps
        traj1 = ep.run(n_steps=2, dt=0.1, kT=2.479e-3, seed=0)
        assert traj1.n_steps == 2
        assert traj1.positions.shape == (2, 3, 3)
        assert jnp.all(jnp.isfinite(traj1.positions))

        # Test with different dt
        traj2 = ep.run(n_steps=3, dt=0.25, kT=2.479e-3, seed=1)
        assert traj2.n_steps == 3
        assert jnp.all(jnp.isfinite(traj2.positions))

        # Test with different kT
        traj3 = ep.run(n_steps=3, dt=0.5, kT=1.0, seed=2)
        assert traj3.n_steps == 3
        assert jnp.all(jnp.isfinite(traj3.positions))

    def test_temperature_observable_is_equinox_module(self):
        """Test that Temperature is an equinox.Module."""
        t = Temperature(dof=9)
        assert isinstance(t, eqx.Module)
        assert issubclass(Temperature, eqx.Module)

    def test_energy_observable_is_equinox_module(self):
        """Test that Energy is an equinox.Module."""
        def mock_energy_fn(positions, bundle):
            return jnp.array(0.0)

        energy = Energy(energy_fn=mock_energy_fn, bundle={})
        assert isinstance(energy, eqx.Module)
        assert issubclass(Energy, eqx.Module)

    def test_ensemble_plan_raises_on_empty_bundle_list(self):
        """Test EnsemblePlan.run() raises ValueError with empty bundle list."""
        ep = EnsemblePlan([])

        with pytest.raises(ValueError) as excinfo:
            ep.run(n_steps=5, dt=0.5, kT=2.479e-3)

        assert "at least one bundle" in str(excinfo.value).lower()

    def test_observable_conformance_with_real_state(self):
        """Test Observable protocol with real LangevinState from prolix."""
        # Create a real integrator state
        n_atoms = 3
        dof = 3 * n_atoms

        positions = jnp.zeros((n_atoms, 3))
        momentum = jnp.ones((n_atoms, 3))
        force = jnp.zeros((n_atoms, 3))
        mass = jnp.ones(n_atoms)
        key = jax.random.PRNGKey(0)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        # Temperature should work with this state
        t_obs = Temperature(dof=dof)
        temp = t_obs.compute(state)
        assert jnp.isfinite(temp)
        assert temp.shape == ()
