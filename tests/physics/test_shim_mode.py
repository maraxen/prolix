"""Test ShimMode: analytical force shim via custom_jvp for bonded terms.

ShimMode.ANALYTICAL registers a @jax.custom_jvp rule that computes bonded
forces analytically instead of tracing through AD. LJ/PME remain AD-traced.

Tests verify:
- ShimMode enum has correct values
- bond_forces analytical output matches AD to within 1e-4 kcal/mol/Å
- energy_with_analytical_shim wraps callable and registers custom_jvp
- Gradient of shim-wrapped energy uses analytical force path
"""

import sys
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from jax_md import space

from prolix.types.shims import ShimMode, energy_with_analytical_shim
from prolix.physics.analytical_forces import bond_forces

# Add tests directory to path for importing test utilities
sys.path.insert(0, str(Path(__file__).parent))


def test_shim_mode_enum():
    """ShimMode enum has correct string values."""
    assert ShimMode.AUTOGRAD.value == "autograd"
    assert ShimMode.ANALYTICAL.value == "analytical"
    assert ShimMode.AUTOGRAD != ShimMode.ANALYTICAL


def test_bond_forces_match_autograd():
    """Analytical bond forces must match AD within 1e-4 kcal/mol/Å."""
    disp_fn, _ = space.free()

    positions = jnp.array([
        [0., 0., 0.],
        [1.5, 0., 0.],
        [3.0, 0., 0.],
    ])

    bond_idx = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    bond_params = jnp.array([
        [100.0, 1.0],
        [100.0, 1.0],
    ])
    bond_mask = jnp.ones(2, dtype=bool)

    f_analytical = bond_forces(positions, bond_idx, bond_params, bond_mask, disp_fn)

    # Compute AD forces using harmonic potential
    def e_bonds(r):
        dvec = jax.vmap(lambda i, j: disp_fn(r[i], r[j]))(
            bond_idx[:, 0], bond_idx[:, 1]
        )
        dist = jnp.linalg.norm(dvec, axis=1)
        return jnp.sum(
            bond_mask * bond_params[:, 0] * (dist - bond_params[:, 1]) ** 2
        )

    f_ad = -jax.grad(e_bonds)(positions)

    max_diff = jnp.max(jnp.abs(f_analytical - f_ad))
    assert jnp.allclose(f_analytical, f_ad, atol=1e-4), (
        f"Bond forces mismatch. Max diff: {max_diff}"
    )


def test_energy_with_analytical_shim_returns_callable():
    """energy_with_analytical_shim returns a callable that works with bundles."""
    from test_molecular_bundle import _minimal_bundle

    def base_energy(bundle):
        return jnp.array(0.0)

    def bonded_forces(bundle):
        return jnp.zeros_like(bundle.positions)

    bundle = _minimal_bundle()
    wrapped = energy_with_analytical_shim(base_energy, bonded_forces)
    result = wrapped(bundle)

    assert isinstance(result, jnp.ndarray)
    assert result.shape == ()


def test_shim_grad_uses_analytical_path():
    """Gradient of shim-wrapped energy uses analytical forces, not AD through energy."""
    from test_molecular_bundle import _minimal_bundle

    call_log = []

    def base_energy(bundle):
        call_log.append("energy")
        return jnp.sum(bundle.positions ** 2)

    def bonded_forces(bundle):
        call_log.append("forces")
        return -2.0 * bundle.positions

    bundle = _minimal_bundle()
    wrapped = energy_with_analytical_shim(base_energy, bonded_forces)

    # Use eqx.filter_grad to handle eqx.Module with mixed field types
    grad_fn = eqx.filter_grad(wrapped)
    grad_bundle = grad_fn(bundle)

    # forces should have been called via custom_jvp
    assert "forces" in call_log
    assert grad_bundle.positions is not None
