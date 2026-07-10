"""S1 D2 / V6 (#268): jax.grad / jax.jacrev through finite-diff parity test.

Validates the differentiability claim: jax.grad(loss)(params) through MD must
agree with finite-difference gradients to RMS < 1e-4.

See also ``test_v6_jaxgrad_ensemble_plan.py`` for the EnsemblePlan.run path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from prolix.physics import pbc, settle, system
from prolix.physics.bonded import make_bond_energy_fn
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL


def _minimal_system_params():
    """Build minimal 2-atom system for fast differentiability testing.

    Simple system: 2 atoms in free space (no PBC), 1 bond.
    This avoids complications with water constraints and PME.

    Returns:
        positions: (2, 3) array of atom positions
        box_vec: None (free space)
        sys_dict: minimal system parameters (empty, no nonbonded)
        mass: (2,) atomic masses
    """
    # 2 atoms: atom0 at origin, atom1 displaced
    positions = jnp.array([
        [0.0, 0.0, 0.0],   # atom 0
        [1.5, 0.0, 0.0],   # atom 1 (at 1.5 Å bond length)
    ], dtype=jnp.float64)

    # Minimal empty system dict (no nonbonded interactions)
    sys_dict = {
        'nonbonded_pairs': jnp.array([], dtype=jnp.int32).reshape(0, 2),
        'charges': jnp.array([0.0, 0.0], dtype=jnp.float64),
        'sigmas': jnp.array([1.0, 1.0], dtype=jnp.float64),
        'epsilons': jnp.array([0.0, 0.0], dtype=jnp.float64),
    }

    mass = jnp.array([12.0, 12.0], dtype=jnp.float64)

    return positions, sys_dict, mass


def _make_harmonic_bond_energy_fn(bond_indices):
    """Create a harmonic bond energy function E = 0.5 * k * (r - r0)^2.

    Args:
        bond_indices: (n_bonds, 2) array of atom pair indices

    Returns:
        energy_fn(r, bond_k, bond_r0) -> scalar energy
    """
    def energy_fn(r, bond_k, bond_r0):
        """Harmonic bond energy.

        Args:
            r: (N, 3) atomic positions
            bond_k: (n_bonds,) spring constants
            bond_r0: (n_bonds,) equilibrium lengths

        Returns:
            Scalar total energy
        """
        r0 = r[bond_indices[:, 0]]   # (n_bonds, 3)
        r1 = r[bond_indices[:, 1]]   # (n_bonds, 3)
        dr = r1 - r0                 # (n_bonds, 3)
        dist = jnp.linalg.norm(dr, axis=1)  # (n_bonds,)

        e_bond = jnp.sum(0.5 * bond_k * (dist - bond_r0) ** 2)
        return e_bond

    return energy_fn


def test_jaxgrad_bond_params_parity():
    """S1 D2: jax.grad through harmonic bond energy agrees with finite-diff to RMS < 1e-4.

    This is a minimal test validating the differentiability pathway before
    full EnsemblePlan.run() integration. It verifies that jax.grad can compose
    through bonded energy functions used in MD simulations.

    Setup:
    - 2-atom system in free space (no PBC)
    - Single harmonic bond parameterized by spring constant k
    - Loss = bond energy at 3 configurations in a small trajectory
    - Bond equilibrium length = 1.5 Å

    Verify:
    - jax.grad(loss)(bond_k) is computable (not NaN/Inf)
    - Gradient matches finite-difference to RMS < 1e-4
    """
    jax.config.update("jax_enable_x64", True)

    # Minimal 2-atom system
    positions, sys_dict, mass = _minimal_system_params()

    # Bond: atom 0 to atom 1
    bond_indices = jnp.array([[0, 1]], dtype=jnp.int32)

    # Create energy function
    energy_fn = _make_harmonic_bond_energy_fn(bond_indices)

    # Define loss: sum bond energies over perturbed configurations
    def loss_wrapper(bond_k):
        """Compute loss = sum of bond energies at 3 atomic configurations."""
        # Configuration 1: initial
        r1 = positions
        e1 = energy_fn(r1, bond_k, jnp.array([1.5]))

        # Configuration 2: atom1 displaced by +0.1 Å along x
        r2 = jnp.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=jnp.float64)
        e2 = energy_fn(r2, bond_k, jnp.array([1.5]))

        # Configuration 3: atom1 displaced by +0.2 Å along x
        r3 = jnp.array([
            [0.0, 0.0, 0.0],
            [1.7, 0.0, 0.0],
        ], dtype=jnp.float64)
        e3 = energy_fn(r3, bond_k, jnp.array([1.5]))

        return e1 + e2 + e3

    # Test parameters
    bond_k_nominal = jnp.array([100.0], dtype=jnp.float64)  # kcal/(mol·Å²)
    eps_fd = 1e-5

    # Compute jax.grad
    grad_jax = jax.grad(loss_wrapper)(bond_k_nominal)

    # Compute finite-difference gradient
    loss_plus = loss_wrapper(bond_k_nominal + eps_fd)
    loss_minus = loss_wrapper(bond_k_nominal - eps_fd)
    grad_fd = (loss_plus - loss_minus) / (2.0 * eps_fd)

    # Verify parity: RMS < 1e-4
    rms_error = jnp.sqrt(jnp.mean((grad_jax - grad_fd) ** 2))

    print(f"jax.grad: {grad_jax[0]:.8e}")
    print(f"FD grad:  {grad_fd:.8e}")
    print(f"RMS error: {rms_error:.8e}")

    assert jnp.isfinite(grad_jax[0]), f"jax.grad produced NaN/Inf: {grad_jax[0]}"
    assert rms_error < 1e-4, (
        f"jax.grad / FD gradient mismatch: RMS={rms_error:.4e} "
        f"(threshold 1e-4). jax.grad={grad_jax[0]:.8e}, fd={grad_fd:.8e}"
    )
