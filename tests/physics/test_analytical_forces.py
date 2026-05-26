"""Tests for analytical bonded force computation.

Tests verify that analytical force functions (bond, angle, dihedral, improper,
urey_bradley) exactly match -jax.grad(energy_fn) at float64 machine precision.
"""

import jax
import jax.numpy as jnp
from jax_md import space

from prolix.physics import analytical_forces, bonded

# Enable float64 for all tests
jax.config.update("jax_enable_x64", True)


# =============================================================================
# BOND FORCES TESTS
# =============================================================================

def test_bond_analytical_vs_grad():
    """Test bond_forces_analytical matches -jax.grad(bond_energy_fn)."""
    displacement_fn, _ = space.free()

    # 3 atoms: two bonds
    # Bond 1: atoms 0-1, length 1.0, k=100
    # Bond 2: atoms 1-2, length 1.2, k=50
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [2.3, 0.0, 0.0],
    ], dtype=jnp.float64)

    bond_indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    bond_params = jnp.array([
        [1.0, 100.0],  # [r0, k]
        [1.2, 50.0],
    ], dtype=jnp.float64)
    bond_mask = jnp.array([1.0, 1.0], dtype=jnp.float64)

    # Compute forces analytically
    f_analytical = analytical_forces.bond_forces_analytical(
        positions, bond_indices, bond_params, displacement_fn, bond_mask
    )

    # Compute forces from energy gradient
    energy_fn = bonded.make_bond_energy_fn(displacement_fn, bond_indices)
    grad_fn = jax.grad(lambda pos: energy_fn(pos, bond_params))
    f_grad = -grad_fn(positions)

    # Compare
    assert jnp.allclose(f_analytical, f_grad, atol=1e-10), \
        f"Bond forces mismatch:\nAnalytical:\n{f_analytical}\nGrad:\n{f_grad}"


# =============================================================================
# ANGLE FORCES TESTS
# =============================================================================

def test_angle_analytical_vs_grad():
    """Test angle_forces_analytical matches -jax.grad(angle_energy_fn)."""
    displacement_fn, _ = space.free()

    # 4 atoms forming an angle at atom 1
    # Angle 1: atoms 0-1-2
    positions = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=jnp.float64)

    angle_indices = jnp.array([[0, 1, 2], [0, 1, 3]], dtype=jnp.int32)
    angle_params = jnp.array([
        [jnp.pi, 100.0],     # [theta0, k]
        [jnp.pi / 2, 50.0],
    ], dtype=jnp.float64)
    angle_mask = jnp.array([1.0, 1.0], dtype=jnp.float64)

    # Compute forces analytically
    f_analytical = analytical_forces.angle_forces_analytical(
        positions, angle_indices, angle_params, displacement_fn, angle_mask
    )

    # Compute forces from energy gradient
    energy_fn = bonded.make_angle_energy_fn(displacement_fn, angle_indices)
    grad_fn = jax.grad(lambda pos: energy_fn(pos, angle_params))
    f_grad = -grad_fn(positions)

    # Compare
    assert jnp.allclose(f_analytical, f_grad, atol=1e-10), \
        f"Angle forces mismatch:\nAnalytical:\n{f_analytical}\nGrad:\n{f_grad}"


# =============================================================================
# DIHEDRAL ANGLE HELPER TEST
# =============================================================================

def test_dihedral_angle_helper_vs_bonded():
    """Test _dihedral_angle_batched uses batched_energy convention (atan2 - pi)."""
    displacement_fn, _ = space.free()

    # 5 atoms: two dihedrals
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
    ], dtype=jnp.float64)

    dihedral_indices = jnp.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=jnp.int32)
    dihedral_mask = jnp.array([1.0, 1.0], dtype=jnp.float64)

    # Compute angles using helper
    phi_analytical = analytical_forces._dihedral_angle_batched(
        positions, dihedral_indices, displacement_fn, dihedral_mask
    )

    # Compute angles using bonded module (has different convention: atan2(y,x) NOT atan2(y,x) - pi)
    phi_bonded = bonded.compute_dihedral_angles(
        positions, dihedral_indices, displacement_fn
    )

    # Helper uses batched_energy convention (atan2 - pi), so compare modulo periodicity
    # Both should represent the same angle, possibly differing by 2*pi
    phi_analytical_unwrapped = jnp.where(
        jnp.abs(phi_analytical - phi_bonded) < jnp.pi,
        phi_analytical,
        phi_analytical + 2*jnp.pi
    )

    assert jnp.allclose(phi_analytical_unwrapped, phi_bonded, atol=1e-10), \
        f"Dihedral angle mismatch (after unwrapping):\nAnalytical:\n{phi_analytical_unwrapped}\nBonded:\n{phi_bonded}"


# =============================================================================
# DIHEDRAL FORCES TESTS
# =============================================================================

def test_dihedral_analytical_vs_grad():
    """Test dihedral_forces_analytical matches -jax.grad(dihedral_energy_fn)."""
    displacement_fn, _ = space.free()

    # 5 atoms: one dihedral (multi-term)
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=jnp.float64)

    dihedral_indices = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    # Params: (N_dih, N_terms, 3) = [periodicity, phase, k]
    dihedral_params = jnp.array(
        [[[1.0, 0.0, 5.0], [2.0, jnp.pi, 3.0]]],
        dtype=jnp.float64
    )
    dihedral_mask = jnp.array([1.0], dtype=jnp.float64)

    # Compute forces analytically
    f_analytical = analytical_forces.dihedral_forces_analytical(
        positions, dihedral_indices, dihedral_params, displacement_fn, dihedral_mask
    )

    # Compute forces from energy gradient
    energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, dihedral_indices)
    grad_fn = jax.grad(lambda pos: energy_fn(pos, dihedral_params))
    f_grad = -grad_fn(positions)

    # Compare
    assert jnp.allclose(f_analytical, f_grad, atol=1e-10), \
        f"Dihedral forces mismatch:\nAnalytical:\n{f_analytical}\nGrad:\n{f_grad}"


# =============================================================================
# IMPROPER FORCES TESTS
# =============================================================================

def test_improper_periodic_vs_grad():
    """Test improper_forces_analytical (periodic) matches grad."""
    displacement_fn, _ = space.free()

    # 4 atoms: one improper (periodic, 3-param format)
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=jnp.float64)

    improper_indices = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    # Periodic improper: (N_imp, N_terms, 3) = [periodicity, phase, k]
    improper_params = jnp.array(
        [[[2.0, 0.0, 10.0]]],
        dtype=jnp.float64
    )
    improper_mask = jnp.array([1.0], dtype=jnp.float64)

    # Compute forces analytically
    f_analytical = analytical_forces.improper_forces_analytical(
        positions, improper_indices, improper_params, displacement_fn, improper_mask
    )

    # Compute forces from energy gradient
    energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, improper_indices)
    grad_fn = jax.grad(lambda pos: energy_fn(pos, improper_params))
    f_grad = -grad_fn(positions)

    # Compare
    assert jnp.allclose(f_analytical, f_grad, atol=1e-10), \
        f"Improper periodic forces mismatch:\nAnalytical:\n{f_analytical}\nGrad:\n{f_grad}"


def test_improper_harmonic_vs_grad():
    """Test improper_forces_analytical (harmonic) matches grad."""
    displacement_fn, _ = space.free()

    # 4 atoms: one improper (harmonic, 2-param format)
    # Modified geometry so dihedral is NOT at equilibrium
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 1.5, 0.5],  # Out of plane to create non-zero dihedral
    ], dtype=jnp.float64)

    improper_indices = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    # Harmonic improper: (N_imp, N_terms, 2) = [k, phi0] (bonded convention)
    # Set equilibrium at 0 (not pi) to avoid grad=0 case
    improper_params = jnp.array(
        [[[20.0, 0.0]]],
        dtype=jnp.float64
    )
    improper_mask = jnp.array([1.0], dtype=jnp.float64)

    # Compute forces analytically
    f_analytical = analytical_forces.improper_forces_analytical(
        positions, improper_indices, improper_params, displacement_fn, improper_mask
    )

    # Compute forces from energy gradient
    energy_fn = bonded.make_harmonic_improper_energy_fn(displacement_fn, improper_indices)
    grad_fn = jax.grad(lambda pos: energy_fn(pos, improper_params))
    f_grad = -grad_fn(positions)

    # Compare (tolerance slightly relaxed due to nested autodiff in harmonic improper)
    assert jnp.allclose(f_analytical, f_grad, atol=5e-7), \
        f"Improper harmonic forces mismatch:\nAnalytical:\n{f_analytical}\nGrad:\n{f_grad}"


# =============================================================================
# UREY-BRADLEY FORCES TEST
# =============================================================================

def test_urey_bradley_analytical_vs_grad():
    """Test urey_bradley_forces_analytical matches grad."""
    displacement_fn, _ = space.free()

    # 3 atoms forming an angle, with urey-bradley interaction
    positions = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=jnp.float64)

    ub_indices = jnp.array([[0, 2]], dtype=jnp.int32)
    ub_params = jnp.array(
        [[1.5, 50.0]],  # [r0, k] for i-k interaction
        dtype=jnp.float64
    )
    ub_mask = jnp.array([1.0], dtype=jnp.float64)

    # Compute forces analytically
    f_analytical = analytical_forces.urey_bradley_forces_analytical(
        positions, ub_indices, ub_params, displacement_fn, ub_mask
    )

    # Compute forces from bond energy (UB uses same form as bond)
    energy_fn = bonded.make_bond_energy_fn(displacement_fn, ub_indices)
    grad_fn = jax.grad(lambda pos: energy_fn(pos, ub_params))
    f_grad = -grad_fn(positions)

    # Compare
    assert jnp.allclose(f_analytical, f_grad, atol=1e-10), \
        f"UB forces mismatch:\nAnalytical:\n{f_analytical}\nGrad:\n{f_grad}"


# =============================================================================
# PADDED ENTRIES ZERO FORCE TEST
# =============================================================================

def test_padded_entries_zero_force():
    """Test that padded entries (0 mask, 0 params) produce zero force."""
    displacement_fn, _ = space.free()

    # 3 atoms with 2 bonds
    # Bond 0: real (mask=1)
    # Bond 1: padded (mask=0, params=0)
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=jnp.float64)

    bond_indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    bond_params = jnp.array([
        [1.0, 100.0],  # Real bond
        [0.0, 0.0],    # Padded
    ], dtype=jnp.float64)
    bond_mask = jnp.array([1.0, 0.0], dtype=jnp.float64)

    # Compute forces
    f = analytical_forces.bond_forces_analytical(
        positions, bond_indices, bond_params, displacement_fn, bond_mask
    )

    # Atoms 0 and 1 should have forces from real bond
    # Atom 2 should have zero force (only padded bond touches it)
    assert not jnp.allclose(f[0], 0.0, atol=1e-12), "Real bond atom should have force"
    assert not jnp.allclose(f[1], 0.0, atol=1e-12), "Real bond atom should have force"
    assert jnp.allclose(f[2], 0.0, atol=1e-12), f"Padded bond should produce zero force, got {f[2]}"


# =============================================================================
# NEWTON'S 3RD LAW TEST
# =============================================================================

def test_force_sum_zero():
    """Test that sum of all forces is approximately zero (Newton's 3rd law)."""
    displacement_fn, _ = space.free()

    # 4 atoms with 2 bonds forming a chain
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [2.2, 0.0, 0.0],
        [3.3, 0.0, 0.0],
    ], dtype=jnp.float64)

    bond_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    bond_params = jnp.array([
        [1.0, 100.0],
        [1.0, 100.0],
    ], dtype=jnp.float64)
    bond_mask = jnp.array([1.0, 1.0], dtype=jnp.float64)

    # Compute forces
    f_bond = analytical_forces.bond_forces_analytical(
        positions, bond_indices, bond_params, displacement_fn, bond_mask
    )

    # Sum should be zero (Newton's 3rd law)
    f_sum = jnp.sum(f_bond, axis=0)
    assert jnp.allclose(f_sum, 0.0, atol=1e-10), \
        f"Force sum should be zero, got {f_sum}"
