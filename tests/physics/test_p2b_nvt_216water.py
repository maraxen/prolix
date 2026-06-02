"""NVT temperature control tests for SETTLE+Langevin integrator.

Phase 2b cross-validation: t2 validates that the thermostat is active and
controlling temperature for TIP3P water with SETTLE constraints.

## Phase 5: Integration Order Fix (2026-06-02)

The settle_langevin integrator was updated to apply SETTLE position constraints
after each A-step (instead of once after both A-steps). This reduces the
per-step position excursion from 2×dt to dt and prevents large constraint impulses
from overwhelming the thermostat at liquid density. The integration order is now:

  B → A → SETTLE + RATTLE → O → A → SETTLE + RATTLE → Force → B → SETTLE_vel

This fixes the energy injection at liquid density: T_eq now remains ~300 K ±5 K
even at standard Langevin coupling (γ=10 ps⁻¹) and 895-water liquid density.
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
    _grid_water_positions,
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


def _run_nvt_scan(n_waters, positions_a, box_edge, steps, burn, dt_fs=0.5, gamma_ps=10.0,
                  seed=42, temperature_k=300.0, pme_grid=None, excl_indices=None):
    """Shared NVT scan runner. Returns mean T over production steps."""
    box_vec = jnp.array([box_edge]*3, dtype=jnp.float64)
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3

    sys_dict = {k: v for k, v in _proxide_params_pure_water(n_waters).items() if k != "exclusion_mask"}
    if excl_indices is not None:
        sys_dict["excl_indices"] = excl_indices

    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    grid = pme_grid or max(16, round(box_edge / 1.0))

    energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True,
        implicit_solvent=False, pme_grid_points=grid, pme_alpha=0.34, cutoff_distance=9.0,
        strict_parameterization=False)

    n_atoms = n_waters * 3
    mass = jnp.array([[15.999],[1.008],[1.008]]*n_waters, dtype=jnp.float64).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)

    init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT,
        gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec,
        remove_linear_com_momentum=False, project_ou_momentum_rigid=True,
        projection_site="post_o", settle_velocity_iters=10)

    state = init_s(jax.random.key(seed), jnp.array(positions_a, dtype=jnp.float64), mass=mass)

    dof = _dof_rigid_tip3p_waters(n_waters)

    def step_fn(state, _):
        state = apply_s(state)
        m_flat = state.mass.reshape(-1)
        ke = jnp.sum(jnp.sum(state.momentum**2, axis=-1) / (2.0 * m_flat))
        return state, ke

    _, kes = jax.lax.scan(step_fn, state, None, length=steps)
    return float(2.0 * jnp.mean(kes[burn:]) / (dof * BOLTZMANN_KCAL))


@pytest.mark.slow
def test_nvt_dilute_temperature_smoke() -> None:
    """NVT smoke: thermostat is active on dilute TIP3P; T_mean in [150, 450] K.

    Uses dilute grid positions (10 Å spacing, 8 waters) as a sanity check that
    settle_langevin thermostat is functioning. Even with negligible forces,
    the thermostat should maintain T in the broad [150, 450] K band.
    Same validation as test_proxide_settle_langevin_water_box_smoke.
    """
    jax.config.update("jax_enable_x64", True)

    n_waters = 8
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)

    mean_t = _run_nvt_scan(n_waters, positions_a, box_edge,
                           steps=500, burn=200, dt_fs=0.5, gamma_ps=10.0, pme_grid=16,
                           excl_indices=_make_tip3p_excl_indices(n_waters))

    assert 150.0 < mean_t < 450.0, (
        f"NVT dilute smoke: mean T={mean_t:.1f} K, expected [150, 450] K — "
        f"thermostat may be inactive"
    )


@pytest.mark.slow
def test_nvt_216water_temperature_stability() -> None:
    """NVT temperature stability: liquid-density 895-water, T_mean within ±5 K.

    Validates that the BAOA-SETTLE-B integration order fix (Phase 5) eliminates
    energy injection at liquid density, allowing stable T≈300 K ±5 K.
    """
    jax.config.update("jax_enable_x64", True)

    n_waters = 895
    positions_a, box_edge = _equil_water_positions(n_waters, seed=42)

    mean_t = _run_nvt_scan(n_waters, positions_a, box_edge,
                           steps=3000, burn=1000, dt_fs=0.5, gamma_ps=10.0,
                           excl_indices=_make_tip3p_excl_indices(n_waters))

    assert abs(mean_t - 300.0) < 5.0, (
        f"NVT {n_waters}-water: mean T={mean_t:.1f} K (target 300 K), "
        f"error {abs(mean_t - 300.0):.1f} K, required < 5 K"
    )
