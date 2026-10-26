"""P7: PME on/off must update PhysicsSystem static fields.

eqx.tree_at cannot replace pme_alpha / pme_grid_points (eqx.field(static=True));
cluster array 20707425 died with 'not a leaf of the pytree'.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from scripts.experiments.profile_b1_flash_vs_autodiff_forces import _apply_pme_config
from tests.physics.test_bundle_factory import _make_minimal_physics_system


def _sys_with_box():
    import dataclasses

    sys_obj = _make_minimal_physics_system()
    return dataclasses.replace(
        sys_obj,
        box_size=jnp.array([32.0, 32.0, 32.0]),
        pme_alpha=0.34,
        pme_grid_points=64,
    )


def test_pme_off_zeros_static_alpha():
    out, key = _apply_pme_config(_sys_with_box(), "off", 1.0)
    assert key == "na"
    assert out.pme_alpha == 0.0


def test_pme_on_sets_static_grid_points():
    out, key = _apply_pme_config(_sys_with_box(), "on", 1.0)
    assert key == "1.0"
    assert out.pme_grid_points == 32
    assert out.pme_alpha == pytest.approx(0.34)
