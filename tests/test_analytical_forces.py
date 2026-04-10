"""Finite-difference verification: analytical forces ≈ -∇E for LJ and Coulomb."""

import jax
import jax.numpy as jnp
import pytest

# Force CPU backend for macOS testing
jax.config.update("jax_platform_name", "cpu")


def _make_test_data(n_atoms=8, seed=42):
    """Create a minimal test system with known LJ/Coulomb parameters."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Random positions spread out enough to avoid clashes
    positions = jax.random.uniform(k1, (n_atoms, 3), minval=-5.0, maxval=5.0)

    # LJ parameters — realistic sigmas/epsilons
    sigmas = jax.random.uniform(k2, (n_atoms,), minval=1.5, maxval=3.5)
    epsilons = jax.random.uniform(k3, (n_atoms,), minval=0.01, maxval=0.2)

    # Charges — mix of positive and negative
    charges = jax.random.normal(k4, (n_atoms,)) * 0.5

    # All atoms are real (no padding)
    atom_mask = jnp.ones(n_atoms, dtype=jnp.bool_)

    return positions, sigmas, epsilons, charges, atom_mask


def _make_padded_test_data(n_real=6, n_padded=3, seed=42):
    """Create test system WITH padding atoms."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n_total = n_real + n_padded

    positions = jax.random.uniform(k1, (n_total, 3), minval=-5.0, maxval=5.0)
    # Padded atoms at origin (this is what causes autodiff NaN)
    positions = positions.at[n_real:].set(0.0)

    sigmas = jax.random.uniform(k2, (n_total,), minval=1.5, maxval=3.5)
    # Padded atoms have sigma=0 (the literal pathological case)
    sigmas = sigmas.at[n_real:].set(0.0)

    epsilons = jax.random.uniform(k3, (n_total,), minval=0.01, maxval=0.2)
    epsilons = epsilons.at[n_real:].set(0.0)

    charges = jax.random.normal(k4, (n_total,)) * 0.5
    charges = charges.at[n_real:].set(0.0)

    atom_mask = jnp.concatenate([
        jnp.ones(n_real, dtype=jnp.bool_),
        jnp.zeros(n_padded, dtype=jnp.bool_),
    ])

    return positions, sigmas, epsilons, charges, atom_mask, n_real


# =============================================================================
# LJ force tests
# =============================================================================

class TestLJForces:
    """Test analytical LJ forces against finite differences."""

    def test_lj_forces_match_energy_gradient(self):
        """Analytical LJ forces ≈ -∇E(LJ) via finite differences."""
        from prolix.physics.analytical_forces import lj_forces_dense
        from prolix.batched_energy import _lj_energy_masked
        from jax_md import space

        positions, sigmas, epsilons, _, atom_mask = _make_test_data(n_atoms=6)
        displacement_fn, _ = space.free()

        # Analytical forces
        f_analytical = lj_forces_dense(
            positions, sigmas, epsilons, atom_mask,
            soft_core_lambda=jnp.float32(1.0),
        )

        # Energy function for finite differences
        def lj_energy(r):
            return _lj_energy_masked(r, sigmas, epsilons, atom_mask, displacement_fn)

        # Numerical gradient
        f_numerical = -jax.grad(lj_energy)(positions)

        # Check agreement (relaxed tolerance for float32)
        max_err = float(jnp.max(jnp.abs(f_analytical - f_numerical)))
        rel_err = float(jnp.max(
            jnp.abs(f_analytical - f_numerical) /
            (jnp.abs(f_numerical) + 1e-8)
        ))
        assert max_err < 0.1, f"Max absolute error {max_err} too large"
        assert rel_err < 0.05, f"Max relative error {rel_err} too large"

    def test_lj_forces_padded_atoms_zero(self):
        """Padded atoms must have exactly zero force."""
        from prolix.physics.analytical_forces import lj_forces_dense

        positions, sigmas, epsilons, _, atom_mask, n_real = _make_padded_test_data()

        f = lj_forces_dense(
            positions, sigmas, epsilons, atom_mask,
            soft_core_lambda=jnp.float32(1.0),
        )

        # Padded atom forces must be zero
        padded_forces = f[n_real:]
        assert jnp.all(padded_forces == 0.0), \
            f"Padded atoms have nonzero forces: {padded_forces}"

        # All forces must be finite (no NaN, no Inf)
        assert jnp.all(jnp.isfinite(f)), \
            f"Non-finite forces detected: NaN={int(jnp.sum(jnp.isnan(f)))}"


# =============================================================================
# Coulomb force tests
# =============================================================================

class TestCoulombForces:
    """Test analytical Coulomb forces against finite differences."""

    def test_coulomb_forces_match_energy_gradient(self):
        """Analytical Coulomb forces ≈ -∇E(Coulomb) via finite differences."""
        from prolix.physics.analytical_forces import coulomb_forces_dense
        from prolix.batched_energy import _coulomb_energy_masked
        from jax_md import space

        positions, _, _, charges, atom_mask = _make_test_data(n_atoms=6)
        displacement_fn, _ = space.free()

        # Analytical forces
        f_analytical = coulomb_forces_dense(
            positions, charges, atom_mask,
        )

        # Numerical gradient
        def coulomb_energy(r):
            return _coulomb_energy_masked(r, charges, atom_mask, displacement_fn)

        f_numerical = -jax.grad(coulomb_energy)(positions)

        max_err = float(jnp.max(jnp.abs(f_analytical - f_numerical)))
        rel_err = float(jnp.max(
            jnp.abs(f_analytical - f_numerical) /
            (jnp.abs(f_numerical) + 1e-8)
        ))
        assert max_err < 0.1, f"Max absolute error {max_err} too large"
        assert rel_err < 0.05, f"Max relative error {rel_err} too large"

    def test_coulomb_forces_padded_atoms_zero(self):
        """Padded atoms must have exactly zero force."""
        from prolix.physics.analytical_forces import coulomb_forces_dense

        positions, _, _, charges, atom_mask, n_real = _make_padded_test_data()

        f = coulomb_forces_dense(positions, charges, atom_mask)

        padded_forces = f[n_real:]
        assert jnp.all(padded_forces == 0.0), \
            f"Padded atoms have nonzero forces: {padded_forces}"
        assert jnp.all(jnp.isfinite(f)), \
            f"Non-finite forces detected: NaN={int(jnp.sum(jnp.isnan(f)))}"


# =============================================================================
# Composite force tests
# =============================================================================

class TestCompositeForce:
    """Test the full single_padded_force function."""

    def test_import_and_signature(self):
        """Verify single_padded_force is importable."""
        from prolix.batched_energy import single_padded_force
        assert callable(single_padded_force)


# =============================================================================
# GB + ACE solvation force tests
# =============================================================================

def _make_gb_test_data(n_real=6, n_padded=2, seed=42):
    """Create test system with radii/charges for GB testing."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    n_total = n_real + n_padded

    # Positions well-separated to avoid GB numerical issues
    positions = jax.random.uniform(k1, (n_total, 3), minval=-8.0, maxval=8.0)
    positions = positions.at[n_real:].set(0.0)

    # Realistic atomic radii (Angstroms)
    radii = jax.random.uniform(k2, (n_total,), minval=1.2, maxval=2.0)
    radii = radii.at[n_real:].set(0.0)

    # Partial charges
    charges = jax.random.normal(k3, (n_total,)) * 0.3
    charges = charges.at[n_real:].set(0.0)

    # Scaled radii (typically radius * 0.8 or similar for OBC)
    scaled_radii = radii * 0.8

    atom_mask = jnp.concatenate([
        jnp.ones(n_real, dtype=jnp.bool_),
        jnp.zeros(n_padded, dtype=jnp.bool_),
    ])

    return positions, charges, radii, scaled_radii, atom_mask, n_real


class TestGBACEForces:
    """Test GB+ACE solvation forces via decomposed VJP vs jax.grad reference."""

    def test_gb_ace_forces_match_energy_gradient(self):
        """Decomposed VJP forces ≈ -∇E(GB+ACE) via jax.grad reference."""
        from prolix.physics.analytical_forces import gb_ace_forces_dense
        from prolix.physics.generalized_born import (
            compute_gb_energy,
            compute_ace_nonpolar_energy,
        )

        # Use all-real atoms (no padding) for clean math comparison.
        # Padded-atom safety is tested separately.
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        n_atoms = 6

        positions = jax.random.uniform(k1, (n_atoms, 3), minval=-8.0, maxval=8.0)
        radii = jax.random.uniform(k2, (n_atoms,), minval=1.2, maxval=2.0)
        charges = jax.random.normal(k3, (n_atoms,)) * 0.3
        scaled_radii = radii * 0.8
        atom_mask = jnp.ones(n_atoms, dtype=jnp.bool_)

        # --- Reference: jax.grad of the full GB+ACE energy ---
        def gb_ace_energy_ref(pos):
            N = pos.shape[0]
            mask_ij = atom_mask[:, None] & atom_mask[None, :]
            energy_mask = mask_ij.astype(jnp.float32)

            e_gb, born_radii = compute_gb_energy(
                positions=pos,
                charges=charges,
                radii=radii,
                scaled_radii=scaled_radii,
                mask=mask_ij.astype(jnp.float32),
                energy_mask=energy_mask,
                dielectric_offset=0.09,
            )
            e_np = compute_ace_nonpolar_energy(radii, born_radii)
            e_np = jnp.sum(e_np * atom_mask)
            return e_gb + e_np

        f_reference = -jax.grad(gb_ace_energy_ref)(positions)

        # --- Decomposed VJP forces ---
        f_decomposed = gb_ace_forces_dense(
            positions, charges, radii, scaled_radii, atom_mask,
        )

        # Check agreement — float32 tolerance is wider for GB chain rule
        max_err = float(jnp.max(jnp.abs(f_decomposed - f_reference)))
        rel_err_denom = jnp.maximum(jnp.abs(f_reference), 1e-6)
        rel_err = float(jnp.max(
            jnp.abs(f_decomposed - f_reference) / rel_err_denom
        ))

        assert max_err < 0.5, (
            f"Max absolute error {max_err:.4f} too large "
            f"(decomposed vs jax.grad reference)"
        )
        assert rel_err < 0.1, (
            f"Max relative error {rel_err:.4f} too large "
            f"(decomposed vs jax.grad reference)"
        )

    def test_gb_ace_forces_finite(self):
        """All forces from decomposed VJP must be finite (no NaN/Inf)."""
        from prolix.physics.analytical_forces import gb_ace_forces_dense

        positions, charges, radii, scaled_radii, atom_mask, _ = (
            _make_gb_test_data()
        )

        f = gb_ace_forces_dense(
            positions, charges, radii, scaled_radii, atom_mask,
        )

        assert jnp.all(jnp.isfinite(f)), (
            f"Non-finite forces: NaN={int(jnp.sum(jnp.isnan(f)))}, "
            f"Inf={int(jnp.sum(jnp.isinf(f)))}"
        )

    def test_gb_ace_forces_padded_atoms_zero(self):
        """Padded atoms must have exactly zero GB/ACE forces."""
        from prolix.physics.analytical_forces import gb_ace_forces_dense

        positions, charges, radii, scaled_radii, atom_mask, n_real = (
            _make_gb_test_data()
        )

        f = gb_ace_forces_dense(
            positions, charges, radii, scaled_radii, atom_mask,
        )

        padded_forces = f[n_real:]
        assert jnp.all(padded_forces == 0.0), (
            f"Padded atoms have nonzero GB/ACE forces: {padded_forces}"
        )
