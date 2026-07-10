"""SETTLE constraint violation quantification across trajectories.

Gate A2: Hard gate for v1.0 release validation.

Measures O-H bond length deviations from TIP3P ideal geometry (0.9572 Å) over:
1. Frozen 64-water snapshot (instantaneous constraint enforcement)
2. 100 ps NVT trajectory (time-averaged constraint satisfaction)
3. 10 ps NPT trajectory (constraint under pressure)

Success criteria:
- Frozen snapshot: O-H RMSD = 0 (SETTLE enforcement is perfect)
- 100 ps NVT: Time-averaged O-H RMSD < 0.01 Å
- 10 ps NPT: Time-averaged O-H RMSD < 0.01 Å (same threshold)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# XA-CI: heavy parity/compile — deselect from GitHub-faithful suite.
pytestmark = pytest.mark.slow

from prolix.physics import pbc, settle, system, pressure, stress
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.units import BAR_PER_AKMA_PRESSURE
from prolix.typing import NPTState
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from .test_explicit_langevin_tip3p_parity import (
    _grid_water_positions,
    _proxide_params_pure_water,
)


def _compute_oh_bond_distances(
    positions: jnp.ndarray, water_indices: jnp.ndarray, box: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Compute O-H bond distances for all waters.

    Args:
        positions: (N_atoms, 3) atomic positions
        water_indices: (N_waters, 3) indices for [O, H1, H2] in each water
        box: (3,) box dimensions for PBC, or None for no PBC

    Returns:
        (N_waters, 2) array of O-H distances for each water's two H atoms
    """
    oxygen_pos = positions[water_indices[:, 0]]  # (N_waters, 3)
    h1_pos = positions[water_indices[:, 1]]  # (N_waters, 3)
    h2_pos = positions[water_indices[:, 2]]  # (N_waters, 3)

    if box is not None:
        # Minimum image distances
        delta_oh1 = h1_pos - oxygen_pos
        delta_oh1 = delta_oh1 - jnp.round(delta_oh1 / box) * box
        delta_oh2 = h2_pos - oxygen_pos
        delta_oh2 = delta_oh2 - jnp.round(delta_oh2 / box) * box
    else:
        delta_oh1 = h1_pos - oxygen_pos
        delta_oh2 = h2_pos - oxygen_pos

    r_oh1 = jnp.linalg.norm(delta_oh1, axis=1)  # (N_waters,)
    r_oh2 = jnp.linalg.norm(delta_oh2, axis=1)  # (N_waters,)

    return jnp.stack([r_oh1, r_oh2], axis=1)  # (N_waters, 2)


def _constraint_rmsd_angstrom(
    positions: jnp.ndarray,
    water_indices: jnp.ndarray,
    ideal_roh: float = settle.TIP3P_ROH,
    box: jnp.ndarray | None = None,
) -> float:
    """Compute RMSD of O-H distances from ideal length.

    Returns scalar RMSD in Ångströms.
    """
    roh_actual = _compute_oh_bond_distances(positions, water_indices, box)
    roh_error = roh_actual - ideal_roh  # (N_waters, 2)
    rmsd = float(jnp.sqrt(jnp.mean(roh_error**2)))
    return rmsd


def test_settle_frozen_snapshot_constraint() -> None:
    """Gate A2.1: Frozen 64-water snapshot.

    After SETTLE projection, O-H distances should equal ideal (0.9572 Å) exactly.
    Tests that SETTLE constraint enforcement is numerically perfect in the absence
    of dynamics.

    Success: O-H RMSD = 0 (within floating-point tolerance ~1e-14)
    """
    jax.config.update("jax_enable_x64", True)
    n_waters = 64
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    positions_a = jnp.array(positions_a, dtype=jnp.float64)  # Convert to JAX array
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

    water_indices = settle.get_water_indices(0, n_waters)

    # Apply SETTLE to frozen snapshot
    positions_settled = settle.settle_positions(
        positions_unconstrained=positions_a,
        positions_old=positions_a,
        water_indices=water_indices,
        r_OH=settle.TIP3P_ROH,
        r_HH=settle.TIP3P_RHH,
        mass_oxygen=15.999,
        mass_hydrogen=1.008,
        box=box_vec,
    )

    rmsd = _constraint_rmsd_angstrom(
        positions_settled, water_indices, settle.TIP3P_ROH, box=box_vec
    )

    assert rmsd < 1e-10, (
        f"Frozen snapshot: O-H RMSD = {rmsd:.2e} Å, "
        f"expected <1e-10 Å (SETTLE constraint enforcement failure)"
    )


@pytest.mark.slow
def test_settle_nvt_100ps_constraint_satisfaction() -> None:
    """Gate A2.2: 100 ps NVT trajectory with SETTLE.

    Validates that SETTLE constraints remain satisfied over long NVT dynamics.
    Time-averaged O-H bond RMSD should stay below 0.01 Å, tracking how well
    the integrator maintains rigid-body geometry during thermostat coupling.

    Success: Time-averaged O-H RMSD < 0.01 Å, max deviation < 0.05 Å
    """
    jax.config.update("jax_enable_x64", True)
    n_waters = 64
    dt_fs = 0.5
    sim_ps = 100.0
    steps = int(sim_ps * 1000.0 / dt_fs)
    burn = max(100, steps // 3)
    measurement_interval = max(1, steps // 100)  # Sample ~100 frames

    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    positions_a = jnp.array(positions_a, dtype=jnp.float64)  # Convert to JAX array
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    temperature_k = 300.0
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_ps = 1.0
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3

    sys_dict = _proxide_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=32,
        pme_alpha=0.34,
        cutoff_distance=9.0,
        strict_parameterization=False,
    )

    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)

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
    state = init_s(jax.random.key(42), jnp.array(positions_a), mass=mass)

    rmsds: list[float] = []
    max_deviation = 0.0

    for step in range(steps):
        state = apply_j(state)

        if step >= burn and step % measurement_interval == 0:
            rmsd = _constraint_rmsd_angstrom(
                state.positions, water_indices, settle.TIP3P_ROH, box=box_vec
            )
            rmsds.append(rmsd)

            # Track maximum per-bond deviation
            roh_actual = _compute_oh_bond_distances(
                state.positions, water_indices, box=box_vec
            )
            roh_error = jnp.abs(roh_actual - settle.TIP3P_ROH)
            max_deviation = max(max_deviation, float(jnp.max(roh_error)))

    mean_rmsd = float(np.mean(rmsds)) if rmsds else float("nan")

    assert mean_rmsd < 0.01, (
        f"100 ps NVT: time-averaged O-H RMSD = {mean_rmsd:.4f} Å, "
        f"expected <0.01 Å (constraint violation during dynamics)"
    )

    assert max_deviation < 0.05, (
        f"100 ps NVT: maximum O-H deviation = {max_deviation:.4f} Å, "
        f"expected <0.05 Å (severe constraint violation)"
    )


@pytest.mark.slow
def test_settle_npt_10ps_constraint_satisfaction() -> None:
    """Gate A2.3: 10 ps NPT trajectory with SETTLE.

    Validates that SETTLE constraints are maintained under pressure control.
    Short NPT trajectory (10 ps, within stable operating range per CLAUDE.md).
    Time-averaged O-H bond RMSD should stay below 0.01 Å.

    Success: Time-averaged O-H RMSD < 0.01 Å, max deviation < 0.05 Å
    """
    jax.config.update("jax_enable_x64", True)
    n_waters = 64
    dt_fs = 0.5
    sim_ps = 10.0
    steps = int(sim_ps * 1000.0 / dt_fs)
    burn = max(50, steps // 3)
    measurement_interval = max(1, steps // 50)  # Sample ~50 frames

    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    positions_a = jnp.array(positions_a, dtype=jnp.float64)  # Convert to JAX array
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    temperature_k = 300.0
    pressure_bar = 1.0
    kT = float(temperature_k) * BOLTZMANN_KCAL
    tau_baro_akma = 2000.0
    tau_thermo_akma = 2000.0

    sys_dict = _proxide_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=32,
        pme_alpha=0.34,
        cutoff_distance=9.0,
        strict_parameterization=False,
    )

    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)

    init_s, apply_s = settle.settle_csvr_npt(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        target_pressure_bar=pressure_bar,
        tau_barostat_akma=tau_baro_akma,
        tau_thermostat_akma=tau_thermo_akma,
        mass=mass,
        water_indices=water_indices,
        box_init=box_vec,
    )

    apply_j = jax.jit(apply_s)
    state = init_s(jax.random.key(42), jnp.array(positions_a), mass=mass, box=box_vec)

    rmsds: list[float] = []
    max_deviation = 0.0

    for step in range(steps):
        state = apply_j(state, box=state.box)

        if step >= burn and step % measurement_interval == 0:
            rmsd = _constraint_rmsd_angstrom(
                state.positions, water_indices, settle.TIP3P_ROH, box=state.box
            )
            rmsds.append(rmsd)

            # Track maximum per-bond deviation
            roh_actual = _compute_oh_bond_distances(
                state.positions, water_indices, box=state.box
            )
            roh_error = jnp.abs(roh_actual - settle.TIP3P_ROH)
            max_deviation = max(max_deviation, float(jnp.max(roh_error)))

    mean_rmsd = float(np.mean(rmsds)) if rmsds else float("nan")

    assert mean_rmsd < 0.01, (
        f"10 ps NPT: time-averaged O-H RMSD = {mean_rmsd:.4f} Å, "
        f"expected <0.01 Å (constraint violation under pressure)"
    )

    assert max_deviation < 0.05, (
        f"10 ps NPT: maximum O-H deviation = {max_deviation:.4f} Å, "
        f"expected <0.05 Å (severe constraint violation under pressure)"
    )
