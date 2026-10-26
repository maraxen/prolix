"""Locate the first step where EnsemblePlan.run and direct settle_langevin diverge (#6162)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from prolix.api.ensemble_plan import EnsemblePlan  # noqa: E402
from tests.api.test_v1_ensemble_plan_parity import (  # noqa: E402
    _reference_trajectory,
    _solvated_ake_bundle,
)

N_STEPS = 1000
DT, KT, SEED = 0.5, 0.596, 42

bundle = _solvated_ake_bundle()
ref = _reference_trajectory(bundle, n_steps=N_STEPS, dt=DT, kT=KT, seed=SEED)
out = EnsemblePlan.from_bundle(bundle).run(
    n_steps=N_STEPS, dt=DT, kT=KT, seed=SEED, dt_unit="akma"
).positions

per_step = jnp.sqrt(jnp.mean((out - ref) ** 2, axis=(1, 2)))
first = int(jnp.argmax(per_step > 1e-12))
print("backend", jax.default_backend(), "shape", out.shape, ref.shape)
print("max per-step rmsd", float(per_step.max()))
print("first step > 1e-12:", first if bool((per_step > 1e-12).any()) else None)
for s in (0, 1, 2, 5, 10, 50, 100, 200, 500, 999):
    print(s, float(per_step[s]))
print("ref finite", bool(jnp.all(jnp.isfinite(ref))), "out finite", bool(jnp.all(jnp.isfinite(out))))
print("ref span", float(jnp.ptp(ref)), "out span", float(jnp.ptp(out)))
