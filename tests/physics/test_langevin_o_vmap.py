"""Constrained O-step maps waters via xtrax ``dispatch_n_waters`` (Vmap/SafeMap).

Replaces water-axis ``lax.scan`` (and a raw ``jax.vmap``) so 6×6 project + 3×3
Cholesky are batched. Keys: ``split(n_waters+1)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from prolix.physics import settle
from prolix.simulate import BOLTZMANN_KCAL


def _two_tip3p():
    r_oh = 0.9572
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [r_oh, 0.0, 0.0],
            [-0.2399, 0.9272, 0.0],
            [4.0, 0.0, 0.0],
            [4.0 + r_oh, 0.0, 0.0],
            [4.0 - 0.2399, 0.9272, 0.0],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array(
        [15.999, 1.008, 1.008, 15.999, 1.008, 1.008], dtype=jnp.float64
    )[:, None]
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    momentum = jnp.arange(18, dtype=jnp.float64).reshape(6, 3) * 0.01
    return positions, masses, water_indices, momentum


def _water_body(key_w, r_w, m_w, p_w, c1, c2, kT):
    p_rigid = settle._project_one_water_momentum_rigid(p_w, r_w, m_w)
    noise_w, _ = settle._ou_noise_one_water_rigid(key_w, r_w, m_w, kT)
    return c1 * p_rigid + c2 * noise_w


def test_langevin_o_constrained_matches_split_then_vmap():
    """O-step water momenta equal vmap(one_water) after split(n_waters+1)."""
    jax.config.update("jax_enable_x64", True)
    position, mass, idx, momentum = _two_tip3p()
    gamma = 10.0
    dt = 0.02
    kT = 300.0 * BOLTZMANN_KCAL
    rng = jax.random.key(7)

    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt(1.0 - c1**2)
    r_water = jnp.stack([position[idx[:, 0]], position[idx[:, 1]], position[idx[:, 2]]], axis=1)
    m_water = jnp.stack(
        [mass[idx[:, 0], 0], mass[idx[:, 1], 0], mass[idx[:, 2], 0]], axis=1
    )
    p_water_in = jnp.stack([momentum[idx[:, 0]], momentum[idx[:, 1]], momentum[idx[:, 2]]], axis=1)

    # Same first split as _langevin_step_o_constrained: leftover key seeds waters.
    key, _iso = jax.random.split(rng)
    n_w = int(idx.shape[0])
    splits = jax.random.split(key, n_w + 1)
    key_after, water_keys = splits[0], splits[1:]
    p_expected = jax.vmap(
        lambda k, r, m, p: _water_body(k, r, m, p, c1, c2, kT)
    )(water_keys, r_water, m_water, p_water_in)

    p_out, key_out = settle._langevin_step_o_constrained(
        momentum, position, mass, gamma, dt, kT, rng, idx
    )
    p_got = jnp.stack([p_out[idx[:, 0]], p_out[idx[:, 1]], p_out[idx[:, 2]]], axis=1)
    assert jnp.allclose(p_got, p_expected, rtol=0.0, atol=0.0)
    assert jnp.array_equal(key_out, key_after)


def test_langevin_o_constrained_uses_xtrax_n_waters_dispatch():
    import inspect

    from prolix.physics import settle as settle_mod

    src = inspect.getsource(settle_mod._langevin_step_o_constrained)
    assert "dispatch_n_waters" in src
    assert "jax.vmap(step_one_water)" not in src


def test_langevin_o_constrained_jaxpr_has_no_scan():
    jax.config.update("jax_enable_x64", True)
    position, mass, idx, momentum = _two_tip3p()
    kT = 300.0 * BOLTZMANN_KCAL
    rng = jax.random.key(0)
    closed = jax.make_jaxpr(settle._langevin_step_o_constrained)(
        momentum, position, mass, 10.0, 0.02, kT, rng, idx
    )
    text = str(closed)
    assert "scan" not in text, text[:2000]
    assert "vmap" in text or "batch" in text.lower()
