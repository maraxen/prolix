"""Slow explicit-solvent validation: short NVE sanity check and extended OpenMM Reference parity.

Run with the rest of the suite using ``pytest -m slow``; exclude from fast CI with
``pytest -m \"not slow\"`` (see ``pyproject.toml``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import quantity, simulate as jmd_simulate

try:
    import openmm
    from openmm import unit

    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

from prolix.physics import pbc, system


def _mock_periodic(n: int, charges: list[float]) -> dict:
    return {
        "charges": jnp.array(charges, dtype=jnp.float64),
        "sigmas": jnp.ones(n, dtype=jnp.float64),
        "epsilons": jnp.zeros(n, dtype=jnp.float64),
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "exclusion_mask": jnp.ones((n, n), dtype=jnp.float64) - jnp.eye(n, dtype=jnp.float64),
    }


@pytest.mark.slow
def test_explicit_pbc_nve_short_run_finite():
    """Few-step NVE in a periodic box: positions, momenta, and energy stay sane."""
    jax.config.update("jax_enable_x64", True)
    n = 4
    box_size = 45.0
    box = jnp.array([box_size, box_size, box_size], dtype=jnp.float64)
    positions = jnp.array(
        [
            [15.0, 15.0, 15.0],
            [32.0, 15.0, 15.0],
            [15.0, 32.0, 15.0],
            [32.0, 32.0, 15.0],
        ],
        dtype=jnp.float64,
    )
    sys_dict = _mock_periodic(n, [1.0, -1.0, 0.2, -0.2])
    displacement_fn, shift_fn = pbc.create_periodic_space(box)
    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=box,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=32,
        pme_alpha=0.34,
        cutoff_distance=12.0,
        strict_parameterization=False,
    )

    init_fn, apply_fn = jmd_simulate.nve(energy_fn, shift_fn=shift_fn, dt=1e-3)
    key = jax.random.PRNGKey(42)
    mass = jnp.ones(n, dtype=jnp.float64) * 12.0
    state = init_fn(key, positions, mass=mass, kT=0.5)

    e0 = energy_fn(state.position) + quantity.kinetic_energy(
        momentum=state.momentum, mass=state.mass
    )

    def step(_i, s):
        return apply_fn(s)

    final = jax.lax.fori_loop(0, 8, step, state)
    e1 = energy_fn(final.position) + quantity.kinetic_energy(
        momentum=final.momentum, mass=final.mass
    )

    assert jnp.all(jnp.isfinite(final.position))
    assert jnp.all(jnp.isfinite(final.momentum))
    assert jnp.isfinite(e0) and jnp.isfinite(e1)
    assert float(jnp.abs(e1 - e0)) < 80.0, "Unexpected energy drift in short explicit NVE test"


@pytest.mark.slow
@pytest.mark.openmm
@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_openmm_four_particle_pme_reference_slow():
    """Extended anchor: four charges, PME, OpenMM Reference vs Prolix ``make_energy_fn``."""
    jax.config.update("jax_enable_x64", True)
    box_size = 40.0
    charges = [1.0, -1.0, 1.0, -1.0]
    positions = [
        [10.0, 10.0, 10.0],
        [30.0, 10.0, 10.0],
        [10.0, 30.0, 10.0],
        [30.0, 30.0, 10.0],
    ]
    alpha = 0.34
    grid = 32
    cutoff = 12.0

    omm_system = openmm.System()
    box_nm = box_size / 10.0
    omm_system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0, 0),
        openmm.Vec3(0, box_nm, 0),
        openmm.Vec3(0, 0, box_nm),
    )
    for _ in charges:
        omm_system.addParticle(1.0)

    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(cutoff / 10.0)
    nonbonded.setPMEParameters(alpha * 10.0, grid, grid, grid)
    nonbonded.setUseDispersionCorrection(False)

    for q in charges:
        nonbonded.addParticle(q, 0.1, 0.0)

    omm_system.addForce(nonbonded)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(omm_system, integrator, openmm.Platform.getPlatformByName("Reference"))
    pos_nm = [openmm.Vec3(p[0] / 10.0, p[1] / 10.0, p[2] / 10.0) for p in positions]
    context.setPositions(pos_nm)
    omm_energy = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalories_per_mole)
    )

    box_vec = jnp.array([box_size, box_size, box_size])
    sys_dict = _mock_periodic(len(charges), charges)
    displacement_fn, _ = pbc.create_periodic_space(box_vec)
    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=grid,
        pme_alpha=alpha,
        cutoff_distance=cutoff,
        strict_parameterization=False,
    )
    jax_energy = float(energy_fn(jnp.array(positions)))

    assert np.isclose(omm_energy, jax_energy, atol=1.0), (
        f"OpenMM vs Prolix: omm={omm_energy:.6f}, jax={jax_energy:.6f}"
    )
