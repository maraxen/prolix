"""NVT 216-water temperature stability test.

Phase 2b cross-validation: t2 validates SETTLE+Langevin temperature control
on a realistic 216-water TIP3P system (648 atoms) with periodic boundary conditions.

Production: 10000 steps (5 ps) at dt=0.5 fs after 5000-step burn-in.
Mean temperature must remain within ±5 K of 300 K target.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from .test_explicit_langevin_tip3p_parity import (
    _equil_water_positions,
    _proxide_params_pure_water,
)


def _dof_rigid_tip3p_waters(n_waters: int) -> float:
    """Degrees of freedom for rigid TIP3P waters (6 per water - 3 for COM)."""
    return float(6 * n_waters - 3)


@pytest.mark.slow
def test_nvt_216water_temperature_stability() -> None:
    """NVT 216-water temperature stability: mean T within ±5 K of 300 K.

    Validates SETTLE+Langevin thermostat on a realistic system:
    - 216 waters (648 atoms) in 30 Å periodic box
    - Explicit electrostatics (PME, grid=32, alpha=0.34, cutoff=9.0 Å)
    - dt=0.5 fs (maximum stable timestep for SETTLE+Langevin coupling)
    - 5000-step burn-in (2.5 ps) + 10000-step production (5 ps)
    - Target: T_mean = 300 K, |error| < 5 K
    """
    jax.config.update("jax_enable_x64", True)

    # Simulation parameters
    n_waters = 216
    dt_fs = 0.5
    steps = 15000
    burn = 5000
    seed = 42
    temperature_k = 300.0

    # Load equilibrated water positions (30 Å box from asset)
    positions_a, box_edge = _equil_water_positions(n_waters, seed=seed)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

    # Time/energy unit conversions
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_ps = 1.0  # friction coefficient (ps^-1)
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3

    # Load TIP3P forcefield parameters for pure water
    sys_dict = _proxide_params_pure_water(n_waters)

    # Set up periodic space and energy function
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)

    # PME grid sized for 30 Å box; use formula: grid ≈ box_edge / 1.0 Å (min 16)
    pme_grid = max(16, round(box_edge / 1.0))

    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=pme_grid,
        pme_alpha=0.34,
        cutoff_distance=9.0,
        strict_parameterization=False,
    )

    # Mass array: TIP3P masses [O=15.999, H=1.008, H=1.008] per water
    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters, dtype=jnp.float64).reshape(
        n_atoms
    )

    # Water indices for SETTLE constraint application
    water_indices = settle.get_water_indices(0, n_waters)

    # Initialize SETTLE+Langevin integrator
    init_s, apply_s = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma_reduced,
        mass=mass,
        water_indices=water_indices,
        box=box_vec,
        remove_linear_com_momentum=False,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
        settle_velocity_iters=10,
    )

    apply_j = jax.jit(apply_s)

    # Initialize state with equilibrated positions
    state = init_s(
        jax.random.key(seed),
        jnp.array(positions_a, dtype=jnp.float64),
        mass=mass,
    )

    # Run simulation and collect temperatures during production phase
    dof_rigid = _dof_rigid_tip3p_waters(n_waters)
    temps: list[float] = []

    for step in range(steps):
        state = apply_j(state)

        if step >= burn:
            # Compute kinetic energy from rigid-body decomposition
            ke_r = float(rigid_tip3p_box_ke_kcal(state.positions, state.momentum, state.mass, n_waters))

            # Temperature from equipartition: T = 2 * KE / (k_B * DOF)
            temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
            temps.append(temp)

    # Verify mean temperature within tolerance
    mean_t = float(np.mean(temps)) if temps else float("nan")
    assert abs(mean_t - 300.0) < 5.0, (
        f"216-water NVT: mean T={mean_t:.1f} K (target 300 K), "
        f"error {abs(mean_t - 300.0):.1f} K, required < 5 K"
    )
