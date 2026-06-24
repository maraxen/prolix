"""V6 (#268): jax.grad finite-diff parity through EnsemblePlan.run."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from prolix.api.ensemble_plan import EnsemblePlan

_v1_path = Path(__file__).resolve().parent / "test_v1_ensemble_plan_parity.py"
_spec = importlib.util.spec_from_file_location("test_v1_ensemble_plan_parity", _v1_path)
assert _spec and _spec.loader
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)
_one_water_bundle = _v1._one_water_bundle


def test_v6_jaxgrad_ensemble_plan_dt_finite_diff_parity():
    """jax.grad through EnsemblePlan.run agrees with FD on dt (RMS < 1e-4)."""
    jax.config.update("jax_enable_x64", True)
    bundle = _one_water_bundle()

    def loss(dt: jnp.ndarray) -> jnp.ndarray:
        traj = EnsemblePlan.from_bundle(bundle).run(
            n_steps=10, dt=dt, kT=0.596, seed=42
        )
        return jnp.sum(traj.positions**2)

    dt0 = jnp.array(0.5, dtype=jnp.float64)
    grad_jax = jax.grad(loss)(dt0)
    eps = jnp.array(1e-5, dtype=jnp.float64)
    grad_fd = (loss(dt0 + eps) - loss(dt0 - eps)) / (2.0 * eps)
    rms = jnp.sqrt((grad_jax - grad_fd) ** 2)
    assert jnp.isfinite(grad_jax), f"jax.grad non-finite: {grad_jax}"
    assert rms < 1e-4, f"V6 dt grad parity failed: jax={grad_jax:.6e}, fd={grad_fd:.6e}, rms={rms:.3e}"
