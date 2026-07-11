"""XR-CARRY: JaxScanIterator dispatch + EnsemblePlan._run_single wiring."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest

from prolix.api import ensemble_dispatch, ensemble_plan
from prolix.api.ensemble_dispatch import dispatch_n_steps


def test_dispatch_n_steps_matches_lax_scan_toy():
    def step_fn(carry, x):
        new = carry + 1.0
        return new, new

    init = jnp.array(0.0)
    n_steps = 5
    final_c, ys = dispatch_n_steps(step_fn, init, n_steps)
    ref_c, ref_ys = jax.lax.scan(step_fn, init, jnp.arange(n_steps))
    assert int(final_c) == int(ref_c) == 5
    assert ys.shape == ref_ys.shape == (5,)
    assert jnp.allclose(ys, ref_ys)


def test_dispatch_n_steps_uses_make_axis_dispatch(monkeypatch):
    calls: list[object] = []
    real = ensemble_dispatch.make_axis_dispatch

    def spy(strategy, **kwargs):
        calls.append(strategy)
        return real(strategy, **kwargs)

    monkeypatch.setattr(ensemble_dispatch, "make_axis_dispatch", spy)

    def step_fn(carry, x):
        return carry + 1, carry + 1

    dispatch_n_steps(step_fn, jnp.array(0), 3)
    assert len(calls) == 1
    from xtrax.tiling.strategy import Scan

    assert isinstance(calls[0], Scan)


def test_ensemble_plan_run_single_uses_dispatch_n_steps():
    source = inspect.getsource(ensemble_plan.EnsemblePlan._run_single)
    assert "dispatch_n_steps" in source
    assert "jax.lax.scan" not in source
