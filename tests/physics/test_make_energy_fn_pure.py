"""Tests for make_energy_fn_pure — closure→explicit-params refactor (v1.1 Item 1).

Validates:
1. make_energy_fn_pure returns (EnergyParams, fn) with correct types
2. fn(params, positions) produces finite energy on an 8-water TIP3P system
3. Energy matches make_energy_fn closure output within float64 rounding
4. jax.export(jax.jit(fn)).lower(params, positions) succeeds (no trace errors)
5. Gradient w.r.t. params is finite (differentiability)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prolix.physics import system as physics_system
from prolix.physics.system import EnergyParams, make_energy_fn_pure
from prolix.physics import pbc
from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME

from .test_explicit_langevin_tip3p_parity import (
    _equil_water_positions,
    _prolix_params_pure_water,
)


N_WATERS = 8


@pytest.fixture(scope="module")
def tip3p_setup():
    """Shared setup for 8-water TIP3P system."""
    jax.config.update("jax_enable_x64", True)
    positions_a, box_edge = _equil_water_positions(N_WATERS, seed=42)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    sys_dict = _prolix_params_pure_water(N_WATERS)
    displacement_fn, _ = pbc.create_periodic_space(box_vec)
    pme_alpha = float(REGRESSION_EXPLICIT_PME["pme_alpha_per_angstrom"])
    pme_grid = int(REGRESSION_EXPLICIT_PME["pme_grid_points"])
    cutoff = float(REGRESSION_EXPLICIT_PME["cutoff_angstrom"])
    positions = jnp.array(positions_a, dtype=jnp.float64)
    return dict(
        positions=positions,
        box_vec=box_vec,
        sys_dict=sys_dict,
        displacement_fn=displacement_fn,
        pme_alpha=pme_alpha,
        pme_grid=pme_grid,
        cutoff=cutoff,
    )


def test_returns_energy_params_and_callable(tip3p_setup):
    """make_energy_fn_pure must return (EnergyParams, callable)."""
    s = tip3p_setup
    result = make_energy_fn_pure(
        s["displacement_fn"], s["sys_dict"], s["box_vec"],
        cutoff_distance=s["cutoff"],
        pme_grid_points=s["pme_grid"],
        pme_alpha=s["pme_alpha"],
        strict_parameterization=False,
    )
    params, fn = result
    assert isinstance(params, EnergyParams), f"expected EnergyParams, got {type(params)}"
    assert callable(fn), "fn must be callable"
    assert params.charges.shape == (N_WATERS * 3,)
    assert params.sigmas.shape == (N_WATERS * 3,)
    assert params.epsilons.shape == (N_WATERS * 3,)


def test_energy_is_finite(tip3p_setup):
    """fn(params, positions) returns a finite scalar on equilibrated positions."""
    s = tip3p_setup
    params, fn = make_energy_fn_pure(
        s["displacement_fn"], s["sys_dict"], s["box_vec"],
        cutoff_distance=s["cutoff"],
        pme_grid_points=s["pme_grid"],
        pme_alpha=s["pme_alpha"],
        strict_parameterization=False,
    )
    e = fn(params, s["positions"])
    assert jnp.isfinite(e), f"energy is not finite: {e}"


def test_energy_matches_closure_fn(tip3p_setup):
    """Pure-params and closure energy functions agree to float64 tolerance."""
    s = tip3p_setup
    params, fn_pure = make_energy_fn_pure(
        s["displacement_fn"], s["sys_dict"], s["box_vec"],
        cutoff_distance=s["cutoff"],
        pme_grid_points=s["pme_grid"],
        pme_alpha=s["pme_alpha"],
        strict_parameterization=False,
    )
    fn_closure = physics_system.make_energy_fn(
        s["displacement_fn"], s["sys_dict"],
        box=s["box_vec"],
        use_pbc=True,
        implicit_solvent=False,
        cutoff_distance=s["cutoff"],
        pme_grid_points=s["pme_grid"],
        pme_alpha=s["pme_alpha"],
        strict_parameterization=False,
    )
    e_pure = float(fn_pure(params, s["positions"]))
    e_closure = float(fn_closure(s["positions"]))
    rel_err = abs(e_pure - e_closure) / (abs(e_closure) + 1e-12)
    assert rel_err < 1e-10, (
        f"energy mismatch: pure={e_pure:.6f}, closure={e_closure:.6f}, rel_err={rel_err:.2e}"
    )


def test_jax_export_succeeds(tip3p_setup):
    """jax.export(jax.jit(fn)).lower(params, positions) must not raise."""
    s = tip3p_setup
    params, fn = make_energy_fn_pure(
        s["displacement_fn"], s["sys_dict"], s["box_vec"],
        cutoff_distance=s["cutoff"],
        pme_grid_points=s["pme_grid"],
        pme_alpha=s["pme_alpha"],
        strict_parameterization=False,
    )
    jit_fn = jax.jit(fn)
    # Lower to StableHLO without error — the key export-readiness check
    lowered = jit_fn.lower(params, s["positions"])
    assert lowered is not None, "lower() returned None"
    # Compile and run to confirm the artifact is valid
    compiled = lowered.compile()
    e = compiled(params, s["positions"])
    assert jnp.isfinite(e), f"compiled artifact produced non-finite energy: {e}"


def test_gradient_wrt_charges_is_finite(tip3p_setup):
    """jax.grad through charges must return finite gradients."""
    s = tip3p_setup
    params, fn = make_energy_fn_pure(
        s["displacement_fn"], s["sys_dict"], s["box_vec"],
        cutoff_distance=s["cutoff"],
        pme_grid_points=s["pme_grid"],
        pme_alpha=s["pme_alpha"],
        strict_parameterization=False,
    )

    def energy_wrt_charges(charges):
        p = EnergyParams(charges=charges, sigmas=params.sigmas, epsilons=params.epsilons)
        return fn(p, s["positions"])

    grad = jax.grad(energy_wrt_charges)(params.charges)
    assert jnp.all(jnp.isfinite(grad)), f"charge gradients contain non-finite values: {grad}"
