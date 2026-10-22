"""Free-particle drift invariants for the modular integrator (make_integrator).

For zero force and zero friction the exact solution is x(t+dt) = x(t) + dt*v, so
every correct splitting must reproduce it regardless of how it divides A-steps.

Pins two defects found 2026-10-01:

1. ``A_Step`` drifted ``fraction*dt*v`` without a ``shift_fn`` but
   ``fraction*dt*v/2`` with one (it delegated to ``_langevin_step_a``, which halves
   internally). Sequences only came out right because the builder left
   ``fraction=1.0`` and always passed ``shift_fn``.
2. The ``lfmiddle_langevin`` sequence had three ``a_step`` half drifts, advancing
   ``1.5*dt*v`` per step (measured 1.49999999998 before the fix).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import pbc
from prolix.physics.integrator_builder import make_integrator
from prolix.physics.step_system import A_Step

jax.config.update("jax_enable_x64", True)

DT = 0.02
U = np.array([0.01, -0.007, 0.013])


def _free_system(n=6):
    box = jnp.array([200.0] * 3)
    _, shift_fn = pbc.create_periodic_space(box)
    pos = jnp.asarray(np.random.default_rng(0).uniform(50.0, 150.0, (n, 3)))
    mass = jnp.ones((n, 1)) * 12.0
    return pos, mass, shift_fn


def _zero_energy(R, **_kwargs):
    return jnp.sum(R) * 0.0


@pytest.mark.parametrize("sequence_name", ["baoab_langevin", "lfmiddle_langevin"])
def test_sequence_advances_free_particle_by_dt(sequence_name):
    pos, mass, shift_fn = _free_system()
    init_fn, apply_fn = make_integrator(
        _zero_energy, shift_fn, mass, sequence_name=sequence_name, dt=DT, kT=0.6, gamma=0.0
    )
    state = init_fn(jax.random.key(0), pos)
    state = state.__replace__(momentum=mass * jnp.asarray(U), force=jnp.zeros_like(pos))
    dx = np.asarray(jax.jit(apply_fn)(state).positions - state.positions)
    ratio = dx / (DT * U)
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-8)


class _Params:
    dt = DT


@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_a_step_fraction_is_independent_of_shift_fn(fraction):
    """A_Step must drift fraction*dt*v whether or not a shift_fn is supplied."""
    pos, mass, shift_fn = _free_system()
    init_fn, _ = make_integrator(_zero_energy, shift_fn, mass, dt=DT, kT=0.6, gamma=0.0)
    state = init_fn(jax.random.key(0), pos).__replace__(momentum=mass * jnp.asarray(U))

    for sf in (shift_fn, None):
        out = A_Step(fraction=fraction, shift_fn=sf).apply(state, _Params())
        ratio = np.asarray(out.positions - state.positions) / (DT * U)
        np.testing.assert_allclose(ratio, fraction, rtol=1e-8, err_msg=f"shift_fn={sf}")
