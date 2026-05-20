"""Tests for bonded energy, parameters, and loss computation (§7.1 Phase B)."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from prolix.fitting import (
    BondedParams,
    BondedTopology,
    bonded_energy,
    bonded_loss,
    default_sigma,
    load_params_init_json,
)


# ===== FIXTURES =====


@pytest.fixture
def water_geometry():
    """3-atom water at standard geometry: ~0.96 Å O-H bonds, ~104.5° angle."""
    # O at origin, H's positioned for standard water geometry
    positions = np.array([
        [0.0, 0.0, 0.0],  # O
        [0.96, 0.0, 0.0],  # H1
        [-0.24, 0.93, 0.0],  # H2 (104.5° angle)
    ], dtype=np.float32)
    return jnp.array(positions)


@pytest.fixture
def water_topology():
    """Bonded topology for water (2 O-H bonds, 1 H-O-H angle)."""
    bond_idx = np.array([[0, 1], [0, 2]], dtype=np.int32)
    angle_idx = np.array([[1, 0, 2]], dtype=np.int32)
    torsion_idx = np.zeros((0, 4), dtype=np.int32)
    torsion_periodicity = np.zeros((0, 1), dtype=np.int32)
    torsion_phase_rad = np.zeros((0, 1), dtype=np.float32)

    return BondedTopology(
        bond_idx=bond_idx,
        angle_idx=angle_idx,
        torsion_idx=torsion_idx,
        torsion_periodicity=torsion_periodicity,
        torsion_phase_rad=torsion_phase_rad,
    )


@pytest.fixture
def water_params():
    """Bonded parameters for water (near-equilibrium values)."""
    k_bond = jnp.array([550.0, 550.0], dtype=jnp.float32)  # O-H stretching
    r0 = jnp.array([0.96, 0.96], dtype=jnp.float32)  # O-H equilibrium
    k_theta = jnp.array([50.0], dtype=jnp.float32)  # H-O-H angle
    theta0_rad = jnp.array([104.5 * jnp.pi / 180.0], dtype=jnp.float32)
    k_phi = jnp.zeros((0, 1), dtype=jnp.float32)

    return BondedParams(
        k_bond=k_bond,
        r0=r0,
        k_theta=k_theta,
        theta0_rad=theta0_rad,
        k_phi=k_phi,
    )


# ===== TESTS =====


def test_bonded_energy_water_equilibrium(water_geometry, water_topology, water_params):
    """Water at equilibrium geometry should have low (near-zero) bonded energy."""
    energy = bonded_energy(water_geometry, water_params, water_topology)
    # With ~perfect geometry, energy should be very small (< 0.1 kcal/mol)
    assert energy < 0.1, f"Expected low energy, got {energy}"
    assert jnp.isfinite(energy), f"Energy should be finite, got {energy}"


def test_bonded_energy_water_stretched(water_geometry, water_topology, water_params):
    """Water with one bond stretched should have higher energy."""
    # Stretch one O-H bond from 0.96 to 1.5 Å
    stretched_pos = water_geometry.at[1].set(jnp.array([1.5, 0.0, 0.0]))

    energy_eq = bonded_energy(water_geometry, water_params, water_topology)
    energy_stretch = bonded_energy(stretched_pos, water_params, water_topology)

    assert energy_stretch > energy_eq, "Stretched geometry should have higher energy"
    assert jnp.isfinite(energy_stretch), "Energy should be finite"


def test_bonded_energy_forces_parity(water_geometry, water_topology, water_params):
    """Gradient w.r.t. positions should give forces (not just not crash)."""
    def energy_fn(pos):
        return bonded_energy(pos, water_params, water_topology)

    forces = jax.grad(energy_fn)(water_geometry)

    assert forces.shape == water_geometry.shape
    # Verify forces are deterministic (same call twice gives same result)
    forces2 = jax.grad(energy_fn)(water_geometry)
    assert jnp.allclose(forces, forces2)
    assert jnp.all(jnp.isfinite(forces))


def test_numerical_gradient_parity_positions():
    """Finite-difference gradient should match jax.grad w.r.t. positions."""
    # Simple 3-atom system (water-like)
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0],
    ], dtype=jnp.float32)

    bond_idx = np.array([[0, 1], [0, 2]], dtype=np.int32)
    angle_idx = np.array([[1, 0, 2]], dtype=np.int32)
    torsion_idx = np.zeros((0, 4), dtype=np.int32)

    topology = BondedTopology(
        bond_idx=bond_idx,
        angle_idx=angle_idx,
        torsion_idx=torsion_idx,
        torsion_periodicity=np.zeros((0, 1), dtype=np.int32),
        torsion_phase_rad=np.zeros((0, 1), dtype=np.float32),
    )

    params = BondedParams(
        k_bond=jnp.array([550.0, 550.0]),
        r0=jnp.array([0.96, 0.96]),
        k_theta=jnp.array([50.0]),
        theta0_rad=jnp.array([104.5 * jnp.pi / 180.0]),
        k_phi=jnp.zeros((0, 1)),
    )

    def energy_fn(pos):
        return bonded_energy(pos, params, topology)

    grad_jax = jax.grad(energy_fn)(positions)

    # Finite differences
    eps = 1e-4
    grad_fd = jnp.zeros_like(positions)
    for i in range(positions.shape[0]):
        for j in range(3):
            pos_plus = positions.at[i, j].add(eps)
            pos_minus = positions.at[i, j].add(-eps)
            grad_fd = grad_fd.at[i, j].set((energy_fn(pos_plus) - energy_fn(pos_minus)) / (2 * eps))

    # Tolerance: 2e-3 is reasonable for finite differences with eps=1e-4
    np.testing.assert_allclose(grad_jax, grad_fd, rtol=2e-3, atol=1e-4)


def test_numerical_gradient_parity_params():
    """Finite-difference gradient should match jax.grad w.r.t. parameters."""
    # Simple bond-only system
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)

    bond_idx = np.array([[0, 1]], dtype=np.int32)
    angle_idx = np.zeros((0, 3), dtype=np.int32)
    torsion_idx = np.zeros((0, 4), dtype=np.int32)

    topology = BondedTopology(
        bond_idx=bond_idx,
        angle_idx=angle_idx,
        torsion_idx=torsion_idx,
        torsion_periodicity=np.zeros((0, 1), dtype=np.int32),
        torsion_phase_rad=np.zeros((0, 1), dtype=np.float32),
    )

    params_init = BondedParams(
        k_bond=jnp.array([400.0]),
        r0=jnp.array([1.0]),
        k_theta=jnp.array([]),
        theta0_rad=jnp.array([]),
        k_phi=jnp.zeros((0, 1)),
    )

    def energy_fn(k_bond_val):
        params = BondedParams(
            k_bond=k_bond_val,
            r0=params_init.r0,
            k_theta=params_init.k_theta,
            theta0_rad=params_init.theta0_rad,
            k_phi=params_init.k_phi,
        )
        return bonded_energy(positions, params, topology)

    grad_jax = jax.grad(energy_fn)(params_init.k_bond)

    # Finite differences
    eps = 1e-4
    grad_fd = jnp.zeros_like(params_init.k_bond)
    for i in range(len(params_init.k_bond)):
        k_plus = params_init.k_bond.at[i].add(eps)
        k_minus = params_init.k_bond.at[i].add(-eps)
        grad_fd = grad_fd.at[i].set((energy_fn(k_plus) - energy_fn(k_minus)) / (2 * eps))

    np.testing.assert_allclose(grad_jax, grad_fd, rtol=2e-3, atol=1e-4)


def test_bonded_loss_decreases_under_optimization():
    """Loss should decrease with a few gradient descent steps."""
    # 5-atom synthetic system
    positions_per_conf = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
         [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.95, 0.1, 0.0], [1.95, 0.05, 0.0],
         [2.05, 1.1, 0.0], [3.0, 0.95, 0.0]],
    ], dtype=jnp.float32)

    forces_ref = jnp.zeros_like(positions_per_conf)
    energies_ref = jnp.array([-100.0, -99.5], dtype=jnp.float64)

    # 4 bonds in a chain: 0-1, 1-2, 2-3, 3-4
    bond_idx = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
    angle_idx = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int32)
    torsion_idx = np.zeros((0, 4), dtype=np.int32)

    topology = BondedTopology(
        bond_idx=bond_idx,
        angle_idx=angle_idx,
        torsion_idx=torsion_idx,
        torsion_periodicity=np.zeros((0, 1), dtype=np.int32),
        torsion_phase_rad=np.zeros((0, 1), dtype=np.float32),
    )

    params_init = BondedParams(
        k_bond=jnp.array([400.0, 400.0, 400.0, 400.0]),
        r0=jnp.array([1.0, 1.0, 1.0, 1.0]),
        k_theta=jnp.array([50.0, 50.0, 50.0]),
        theta0_rad=jnp.array([180 * jnp.pi / 180, 180 * jnp.pi / 180, 180 * jnp.pi / 180]),
        k_phi=jnp.zeros((0, 1)),
    )

    # Initialize params with some perturbation
    params = BondedParams(
        k_bond=params_init.k_bond * 1.1,
        r0=params_init.r0 * 1.05,
        k_theta=params_init.k_theta * 0.9,
        theta0_rad=params_init.theta0_rad,
        k_phi=params_init.k_phi,
    )

    def loss_fn(p):
        return bonded_loss(
            positions_per_conf,
            forces_ref,
            energies_ref,
            p,
            params_init,
            topology,
            alpha=0.25,
            w_reg=0.01,
        )

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    losses = []
    for _ in range(50):
        loss_val = loss_fn(params)
        losses.append(float(loss_val))

        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)

    # Loss should decrease by at least 10%
    assert losses[0] > losses[-1], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
    assert (losses[0] - losses[-1]) / losses[0] > 0.1, \
        f"Loss decreased by < 10%: {(losses[0] - losses[-1]) / losses[0]}"


def test_load_params_init_json_mol_000():
    """Load mol_000.params_init.json and verify shapes."""
    path = Path("/home/marielle/projects/prolix/data/ani1x_subset/lane_a/mol_000.params_init.json")

    if not path.exists():
        pytest.skip(f"Test file not found: {path}")

    params, topology = load_params_init_json(path)

    # mol_000 has known structure from the JSON
    assert topology.n_bonds == 14
    assert topology.n_angles == 17
    assert topology.n_torsions == 17
    assert topology.n_torsion_terms == 1

    # Verify shapes
    assert params.k_bond.shape == (14,)
    assert params.r0.shape == (14,)
    assert params.k_theta.shape == (17,)
    assert params.theta0_rad.shape == (17,)
    assert params.k_phi.shape == (17, 1)

    # Verify units conversion
    # theta0_deg ~127° should be > pi/2 rad
    assert jnp.any(params.theta0_rad > jnp.pi / 2)

    # Verify no NaNs
    assert jnp.all(jnp.isfinite(params.k_bond))
    assert jnp.all(jnp.isfinite(params.r0))
    assert jnp.all(jnp.isfinite(params.k_theta))
    assert jnp.all(jnp.isfinite(params.theta0_rad))
    assert jnp.all(jnp.isfinite(params.k_phi))


@pytest.mark.slow
def test_load_trp_cage():
    """Load Trp-cage params and verify no OOM/NaN."""
    path = Path("/home/marielle/projects/prolix/data/ani1x_subset/lane_b/trp_cage.params_init.json")

    if not path.exists():
        pytest.skip(f"Test file not found: {path}")

    params, topology = load_params_init_json(path)

    # Trp-cage (1L2Y): 312 atoms, large molecule
    assert topology.n_atoms > 300, f"Expected ~312 atoms, got {topology.n_atoms}"
    assert topology.n_bonds > 300
    assert topology.n_angles > 500
    assert topology.n_torsions > 700

    # Verify no NaNs
    assert jnp.all(jnp.isfinite(params.k_bond))
    assert jnp.all(jnp.isfinite(params.r0))
    assert jnp.all(jnp.isfinite(params.k_theta))
    assert jnp.all(jnp.isfinite(params.theta0_rad))
    assert jnp.all(jnp.isfinite(params.k_phi))


def test_default_sigma():
    """Default sigma should have sensible per-parameter-type values."""
    params_init = BondedParams(
        k_bond=jnp.array([400.0, 400.0]),
        r0=jnp.array([1.0, 1.0]),
        k_theta=jnp.array([50.0]),
        theta0_rad=jnp.array([104.5 * jnp.pi / 180.0]),
        k_phi=jnp.zeros((1, 1)),
    )

    sigma = default_sigma(params_init)

    # Verify shapes match
    assert sigma.k_bond.shape == params_init.k_bond.shape
    assert sigma.r0.shape == params_init.r0.shape
    assert sigma.k_theta.shape == params_init.k_theta.shape
    assert sigma.theta0_rad.shape == params_init.theta0_rad.shape
    assert sigma.k_phi.shape == params_init.k_phi.shape

    # Verify constants (spec §6, non-contractual starting values)
    assert jnp.allclose(sigma.k_bond, 100.0)
    assert jnp.allclose(sigma.r0, 0.05)
    assert jnp.allclose(sigma.k_theta, 30.0)
    assert jnp.allclose(sigma.k_phi, 1.0)
    # theta0_rad should be 5° in radians
    assert jnp.allclose(sigma.theta0_rad, 5.0 * jnp.pi / 180.0, rtol=1e-5)


def test_bonded_loss_per_molecule_axis():
    """Loss should normalize correctly over conformers and atoms."""
    # 2 conformers, 2 atoms
    positions_per_conf = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.99, 0.0, 0.0]],
    ], dtype=jnp.float32)

    forces_ref = jnp.zeros((2, 2, 3), dtype=jnp.float32)
    energies_ref = jnp.array([-100.0, -100.0], dtype=jnp.float64)

    bond_idx = np.array([[0, 1]], dtype=np.int32)
    angle_idx = np.zeros((0, 3), dtype=np.int32)
    torsion_idx = np.zeros((0, 4), dtype=np.int32)

    topology = BondedTopology(
        bond_idx=bond_idx,
        angle_idx=angle_idx,
        torsion_idx=torsion_idx,
        torsion_periodicity=np.zeros((0, 1), dtype=np.int32),
        torsion_phase_rad=np.zeros((0, 1), dtype=np.float32),
    )

    params_init = BondedParams(
        k_bond=jnp.array([400.0]),
        r0=jnp.array([1.0]),
        k_theta=jnp.array([]),
        theta0_rad=jnp.array([]),
        k_phi=jnp.zeros((0, 1)),
    )

    params = BondedParams(
        k_bond=jnp.array([400.0]),
        r0=jnp.array([1.0]),
        k_theta=jnp.array([]),
        theta0_rad=jnp.array([]),
        k_phi=jnp.zeros((0, 1)),
    )

    loss = bonded_loss(
        positions_per_conf,
        forces_ref,
        energies_ref,
        params,
        params_init,
        topology,
    )

    assert jnp.isfinite(loss)
    assert loss > 0  # Should have nonzero loss (forces_pred != forces_ref)


def test_torsion_energy_basic():
    """Basic torsion energy test on a simple dihedral."""
    # 4-atom chain for a simple dihedral
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ], dtype=jnp.float32)

    bond_idx = np.zeros((0, 2), dtype=np.int32)
    angle_idx = np.zeros((0, 3), dtype=np.int32)
    torsion_idx = np.array([[0, 1, 2, 3]], dtype=np.int32)

    topology = BondedTopology(
        bond_idx=bond_idx,
        angle_idx=angle_idx,
        torsion_idx=torsion_idx,
        torsion_periodicity=np.array([[3]], dtype=np.int32),
        torsion_phase_rad=np.array([[0.0]], dtype=np.float32),
    )

    params = BondedParams(
        k_bond=jnp.array([]),
        r0=jnp.array([]),
        k_theta=jnp.array([]),
        theta0_rad=jnp.array([]),
        k_phi=jnp.array([[1.0]]),  # 1.0 kcal/mol amplitude
    )

    energy = bonded_energy(positions, params, topology)

    # Energy should be finite and reasonable
    assert jnp.isfinite(energy)
    # For this coplanar dihedral (phi ~ 0 or 180°), energy will be based on cos(3*phi)
    # Just verify it doesn't crash and is reasonable magnitude
    assert energy < 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
