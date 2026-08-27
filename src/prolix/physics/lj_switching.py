"""OpenMM 8.3 Lennard-Jones switching function (Reference platform).

S(z) = 1 - 10 z^3 + 15 z^4 - 6 z^5
z = (r - r_switch) / (r_cut - r_switch) for r_switch < r < r_cut.

Applied to LJ only; PME/cutoff Coulomb is unswitched.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def lj_switch_weight(
    r: Array,
    cutoff: float,
    switch_width: float | None,
) -> Array:
    """Multiplicative LJ scale in ``[0, 1]``. ``switch_width is None`` or ``<= 0`` → 1."""
    if switch_width is None or float(switch_width) <= 0.0:
        return jnp.ones_like(r)
    r_cut = jnp.asarray(cutoff, dtype=r.dtype)
    width = jnp.asarray(switch_width, dtype=r.dtype)
    r_switch = r_cut - width
    z = (r - r_switch) / jnp.maximum(width, jnp.asarray(1e-12, dtype=r.dtype))
    s = 1.0 - 10.0 * z**3 + 15.0 * z**4 - 6.0 * z**5
    return jnp.where(
        r >= r_cut,
        jnp.asarray(0.0, dtype=r.dtype),
        jnp.where(r <= r_switch, jnp.asarray(1.0, dtype=r.dtype), s),
    )
