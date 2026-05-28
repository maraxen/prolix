#!/usr/bin/env python3
"""Probe NPT temperature at steps 0/1/200 (debug only, not bath-tracked)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from prolix.physics import pbc, rigid_water_ke, settle, system
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

# Test grid helpers
from tests.physics.test_explicit_langevin_tip3p_parity import (  # noqa: E402
    _grid_water_positions,
    _proxide_params_pure_water,
)


def _t_k(pos, mom, mass, n_waters: int) -> float:
    ndof = 6 * n_waters - 3
    ke = float(rigid_water_ke.rigid_tip3p_box_ke_kcal(pos, mom, mass, n_waters))
    return 2.0 * ke / (BOLTZMANN_KCAL * ndof)


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    n_waters = 16
    dt_akma = 0.5 / AKMA_TIME_UNIT_FS
    kT = 300.0 * BOLTZMANN_KCAL
    pos, box_edge = _grid_water_positions(n_waters, spacing_angstrom=3.1)
    box = jnp.array([box_edge] * 3, dtype=jnp.float64)
    sys_dict = _proxide_params_pure_water(n_waters)
    disp, shift = pbc.create_periodic_space(box)
    energy_fn = system.make_energy_fn(
        disp,
        sys_dict,
        box=box,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=32,
        pme_alpha=0.34,
        cutoff_distance=9.0,
        strict_parameterization=False,
    )
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters, dtype=jnp.float64).reshape(-1)
    water_idx = settle.get_water_indices(0, n_waters)

    init_l, apply_l = settle.settle_langevin(
        energy_fn,
        shift,
        dt=dt_akma,
        kT=kT,
        gamma=1.0 * AKMA_TIME_UNIT_FS * 1e-3,
        mass=mass,
        water_indices=water_idx,
        box=box,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
    )
    apply_l = jax.jit(apply_l)
    st = init_l(jax.random.PRNGKey(0), jnp.array(pos), mass=mass)
    for _ in range(4000):
        st = apply_l(st)
    print(f"after NVT 2ps: T={_t_k(st.positions, st.momentum, st.mass, n_waters):.1f} K")

    init_npt, apply_npt = settle.settle_csvr_npt(
        energy_fn,
        shift,
        dt=dt_akma,
        kT=kT,
        target_pressure_bar=1.0,
        tau_barostat_akma=2000.0,
        tau_thermostat_akma=2000.0,
        mass=mass,
        water_indices=water_idx,
        box_init=box,
    )
    apply_npt = jax.jit(apply_npt)
    cold = init_npt(jax.random.PRNGKey(1), st.positions, mass=mass, box=box)
    warm = init_npt(
        jax.random.PRNGKey(1), st.positions, mass=mass, box=box, momentum=st.momentum
    )
    print(f"init cold: T={_t_k(cold.positions, cold.momentum, cold.mass, n_waters):.1f} K")
    print(f"init warm: T={_t_k(warm.positions, warm.momentum, warm.mass, n_waters):.1f} K")
    warm = apply_npt(warm, box=warm.box)
    print(f"after 1 step warm: T={_t_k(warm.positions, warm.momentum, warm.mass, n_waters):.1f} K")


if __name__ == "__main__":
    main()
