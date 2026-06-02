"""NVT 895-water temperature stability test.

Phase 2b cross-validation: t2 validates SETTLE+Langevin temperature control
on a full 895-water TIP3P box (2685 atoms, liquid density) with PBC.

Production: 2000 steps (1 ps) at dt=0.5 fs after 1000-step burn-in.
Mean temperature must remain within ±5 K of 300 K target.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import pbc, settle, system
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from .test_explicit_langevin_tip3p_parity import (
    _equil_water_positions,
    _proxide_params_pure_water,
)


def _dof_rigid_tip3p_waters(n_waters: int) -> float:
    """Degrees of freedom for rigid TIP3P waters (6 per water - 3 for COM)."""
    return float(6 * n_waters - 3)


def _make_tip3p_excl_indices(n_waters: int) -> jnp.ndarray:
    """Per-atom sparse exclusion list for TIP3P: shape (N_atoms, 2).

    make_energy_fn reads excl_indices (N_atoms, max_excl), not the dense exclusion_mask.
    Without this, all intramolecular O-H/H-H Coulomb interactions are included (~121
    kcal/mol/Å per O-H pair), causing thermal runaway regardless of thermostat strength.
    """
    n_atoms = n_waters * 3
    excl = np.zeros((n_atoms, 2), dtype=np.int32)
    for w in range(n_waters):
        o, h1, h2 = w * 3, w * 3 + 1, w * 3 + 2
        excl[o] = [h1, h2]
        excl[h1] = [o, h2]
        excl[h2] = [o, h1]
    return jnp.array(excl, dtype=jnp.int32)


@pytest.mark.slow
def test_nvt_216water_temperature_stability() -> None:
    """NVT temperature stability on full equilibrated TIP3P water box: mean T within ±5 K.

    Validates SETTLE+Langevin thermostat on a realistic system:
    - 895 waters (2685 atoms) in 30 Å periodic box (liquid density)
    - Explicit electrostatics (PME, grid=30, alpha=0.34, cutoff=9.0 Å)
    - Sparse intramolecular exclusions via excl_indices (make_energy_fn format)
    - dt=0.5 fs (maximum stable timestep for SETTLE+Langevin coupling)
    - 1000-step burn-in (0.5 ps) + 2000-step production (1.0 ps)
    - Target: T_mean = 300 K, |error| < 5 K
    """
    jax.config.update("jax_enable_x64", True)

    # Simulation parameters — full equilibrated box (liquid density)
    n_waters = 895  # full 30 Å box
    dt_fs = 0.5
    steps = 3000   # 1.5 ps total (burn + production)
    burn = 1000    # 0.5 ps burn-in
    seed = 42
    temperature_k = 300.0

    # Load equilibrated water positions (30 Å box from asset)
    positions_a, box_edge = _equil_water_positions(n_waters, seed=seed)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

    # Time/energy unit conversions
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_ps = 10.0  # friction coefficient (ps^-1); strong coupling (τ=0.1 ps) to
    # equilibrate in 0.5 ps burn-in (5τ → 99.3% of offset removed). γ=1 ps⁻¹ (τ=1 ps)
    # fails to equilibrate the asset positions (different forcefield) within the burn.
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3

    # Build sys_dict: drop dense exclusion_mask (7.2 MB, 5-9 min to allocate at 895w),
    # inject sparse excl_indices instead. make_energy_fn reads excl_indices, not the mask.
    sys_dict = {k: v for k, v in _proxide_params_pure_water(n_waters).items() if k != "exclusion_mask"}
    sys_dict["excl_indices"] = _make_tip3p_excl_indices(n_waters)

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

    # Initialize state with equilibrated positions
    state = init_s(
        jax.random.key(seed),
        jnp.array(positions_a, dtype=jnp.float64),
        mass=mass,
    )

    # Run via scan: all steps on-device, single host transfer at end.
    # Simple Cartesian KE (Σ p²/2m) is position-independent — correct under PBC.
    dof = _dof_rigid_tip3p_waters(n_waters)

    def step_fn(state, _):
        state = apply_s(state)
        m_flat = state.mass.reshape(-1)
        ke = jnp.sum(jnp.sum(state.momentum**2, axis=-1) / (2.0 * m_flat))
        return state, ke

    _, kes = jax.lax.scan(step_fn, state, None, length=steps)
    mean_t = float(2.0 * jnp.mean(kes[burn:]) / (dof * BOLTZMANN_KCAL))

    assert abs(mean_t - 300.0) < 5.0, (
        f"NVT {n_waters}-water: mean T={mean_t:.1f} K (target 300 K), "
        f"error {abs(mean_t - 300.0):.1f} K, required < 5 K"
    )
