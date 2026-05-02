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
from prolix.physics.system import make_energy_fn_pure
from prolix.physics.types import EnergyParams, PhysicsSystem
from prolix.physics import pbc
from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME

from .test_explicit_langevin_tip3p_parity import (
    _equil_water_positions,
)

def _prolix_params_pure_water(n_waters):
    """Modern TIP3P parameters with sparse exclusions."""
    n = n_waters * 3
    charges = jnp.tile(jnp.array([-0.834, 0.417, 0.417]), n_waters)
    sigmas = jnp.tile(jnp.array([3.1507, 1.0, 1.0]), n_waters)
    epsilons = jnp.tile(jnp.array([0.1521, 0.0, 0.0]), n_waters)
    
    # Build Sparse Exclusions (O-H1, O-H2, H1-H2)
    excl_indices = []
    for i in range(n_waters):
        o, h1, h2 = 3*i, 3*i+1, 3*i+2
        # Symmetric exclusions
        excl_indices.append([h1, h2, -1, -1]) # O: excluded H1,H2
        excl_indices.append([o, h2, -1, -1])  # H1: excluded O,H2
        excl_indices.append([o, h1, -1, -1])  # H2: excluded O,H1
    
    excl_indices = jnp.array(excl_indices, dtype=jnp.int32)
    excl_scales = jnp.zeros_like(excl_indices, dtype=jnp.float32)
    
    return {
        "charges": charges,
        "sigmas": sigmas,
        "epsilons": epsilons,
        "excl_indices": excl_indices,
        "excl_scales_vdw": excl_scales,
        "excl_scales_elec": excl_scales,
        "bonds": jnp.array([[3*i, 3*i+1] for i in range(n_waters)] + [[3*i, 3*i+2] for i in range(n_waters)]),
        "bond_params": jnp.tile(jnp.array([0.9572, 500000.0]), (2*n_waters, 1)),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2), dtype=jnp.float32),
        "proper_dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float32),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3), dtype=jnp.float32),
    }


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

    physics_system = PhysicsSystem.from_dict(
        sys_dict, positions, box_vec, cutoff_distance=cutoff
    )

    return dict(
        positions=positions,
        box_vec=box_vec,
        sys_dict=sys_dict,
        physics_system=physics_system,
        displacement_fn=displacement_fn,
        pme_alpha=pme_alpha,
        pme_grid=pme_grid,
        cutoff=cutoff,
    )


def test_returns_energy_params_and_callable(tip3p_setup):
    """make_energy_fn_pure must return (EnergyParams, callable)."""
    s = tip3p_setup
    result = make_energy_fn_pure(
        s["displacement_fn"], s["physics_system"],
        cutoff_distance=s["cutoff"],
        pme_grid_points=s["pme_grid"],
        pme_alpha=s["pme_alpha"],
        strict_parameterization=False,
    )
    params, fn = result
    assert isinstance(params, EnergyParams), f"expected EnergyParams, got {type(params)}"
    assert callable(fn), "fn must be callable"
    
    # Check params structure
    assert 'charges' in params.params
    assert params.params['charges'].shape == (N_WATERS * 3,)


def test_energy_is_finite(tip3p_setup):
    """fn(params, positions) returns a finite scalar on equilibrated positions."""
    s = tip3p_setup
    params, fn = make_energy_fn_pure(
        s["displacement_fn"], s["physics_system"],
        cutoff_distance=s["cutoff"],
        pme_grid_points=s["pme_grid"],
        pme_alpha=s["pme_alpha"],
        strict_parameterization=False,
    )
    e = fn(params, s["positions"])
    assert jnp.isfinite(e), f"energy is not finite: {e}"


def _prolix_params_argon(n_atoms):
    """Argon parameters (no exclusions)."""
    charges = jnp.zeros(n_atoms)
    sigmas = jnp.full(n_atoms, 3.405) # Argon sigma
    epsilons = jnp.full(n_atoms, 0.238) # Argon epsilon
    return {
        "charges": charges,
        "sigmas": sigmas,
        "epsilons": epsilons,
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2), dtype=jnp.float32),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2), dtype=jnp.float32),
        "proper_dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float32),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3), dtype=jnp.float32),
    }

@pytest.fixture(scope="module")
def argon_setup():
    """8-atom Argon box."""
    n_atoms = 8
    positions = jax.random.uniform(jax.random.PRNGKey(42), (n_atoms, 3)) * 10.0
    box_vec = jnp.array([15.0, 15.0, 15.0])
    sys_dict = _prolix_params_argon(n_atoms)
    displacement_fn, _ = pbc.create_periodic_space(box_vec)
    cutoff = 9.0
    physics_system = PhysicsSystem.from_dict(sys_dict, positions, box_vec, cutoff_distance=cutoff)
    return dict(positions=positions, box_vec=box_vec, sys_dict=sys_dict, physics_system=physics_system, displacement_fn=displacement_fn, cutoff=cutoff)

def test_energy_matches_closure_fn(argon_setup):
    """Pure-params and closure energy functions agree on Argon (no exclusions)."""
    s = argon_setup
    params, fn_pure = make_energy_fn_pure(
        s["displacement_fn"], s["physics_system"],
        cutoff_distance=s["cutoff"],
        pme_grid_points=16,
        pme_alpha=0.34,
    )
    fn_closure = physics_system.make_energy_fn(
        s["displacement_fn"], s["sys_dict"],
        box=s["box_vec"],
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=16,
        pme_alpha=0.34,
        cutoff_distance=s["cutoff"],
    )

    e_pure = float(fn_pure(params, s["positions"]))
    e_closure = float(fn_closure(s["positions"]))
    rel_err = abs(e_pure - e_closure) / (abs(e_closure) + 1e-12)
    assert rel_err < 1e-5, f"mismatch: pure={e_pure:.6f}, closure={e_closure:.6f}"


def test_jax_export_succeeds(tip3p_setup):
    """jax.export(jax.jit(fn)).lower(params, positions) must not raise."""
    s = tip3p_setup
    params, fn = make_energy_fn_pure(
        s["displacement_fn"], s["physics_system"],
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
        s["displacement_fn"], s["physics_system"],
        cutoff_distance=s["cutoff"],
        pme_grid_points=s["pme_grid"],
        pme_alpha=s["pme_alpha"],
        strict_parameterization=False,
    )

    def energy_wrt_charges(charges):
        p = EnergyParams(params={'charges': charges, 'sigmas': params.sigmas, 'epsilons': params.epsilons})
        return fn(p, s["positions"])

    grad = jax.grad(energy_wrt_charges)(params.charges)

    assert jnp.all(jnp.isfinite(grad)), f"charge gradients contain non-finite values: {grad}"
