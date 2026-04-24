"""Temperature control validation for SETTLE with Nosé-Hoover Chain thermostat.

Validates that settle_with_nhc (JAX MD's NHC thermostat + SETTLE constraints)
provides stable temperature control at production timesteps (dt=1.0, 2.0 fs).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water


def _dof_rigid_tip3p_waters(n_waters: int) -> float:
  return float(6 * n_waters - 3)


def _mean_rigid_t_nhc_after_burn(
    *, dt_fs: float, n_waters: int, seed: int, steps: int, burn: int
) -> float:
  """Compute mean temperature over NHC trajectory after burn-in."""
  jax.config.update("jax_enable_x64", True)

  temperature_k = 300.0
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL

  sys_dict = _prolix_params_pure_water(n_waters)
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

  # Use settle_with_nhc with default NHC parameters (chain_length=5)
  init_s, apply_s = settle.settle_with_nhc(
      energy_fn,
      shift_fn,
      dt=dt_akma,
      kT=kT,
      mass=mass,
      water_indices=water_indices,
      box=box_vec,
      remove_linear_com_momentum=True,
  )

  apply_j = jax.jit(apply_s)
  dof_rigid = _dof_rigid_tip3p_waters(n_waters)

  state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)
  temps: list[float] = []

  for step in range(steps):
    state = apply_j(state)
    if step >= burn:
      ke_r = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
      temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
      temps.append(temp)

  return float(np.mean(temps)) if temps else float("nan")


@pytest.mark.slow
def test_nhc_temperature_dt1fs_near_target() -> None:
  """dt=1.0 fs, 100 ps: mean T within 5K of 300K target.

  Validates Nosé-Hoover Chain thermostat provides stable temperature control.
  NHC is deterministic (not stochastic), so tighter tolerance (±5K) is expected.
  """
  n_waters = 2
  dt_fs = 1.0
  sim_ps = 100.0
  steps = int(sim_ps * 1000.0 / dt_fs)
  burn = max(100, steps // 3)
  seed = 601

  mean_t = _mean_rigid_t_nhc_after_burn(
      dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn
  )
  assert abs(mean_t - 300.0) < 5.0, f"dt={dt_fs} fs: T={mean_t:.1f} K, expected 300 ± 5 K"
  print(f"✓ NHC at dt={dt_fs} fs: T={mean_t:.1f} K (target 300 ± 5 K)")


@pytest.mark.slow
def test_nhc_temperature_dt2fs_near_target() -> None:
  """dt=2.0 fs, 100 ps: mean T within 5K of 300K target.

  Tests Nosé-Hoover Chain thermostat at production timestep (dt=2fs).
  This is the critical test: Phase 2B/Langevin failed at dt=2fs with temperature
  instability. NHC should handle this correctly.
  """
  n_waters = 2
  dt_fs = 2.0
  sim_ps = 100.0
  steps = int(sim_ps * 1000.0 / dt_fs)
  burn = max(100, steps // 3)
  seed = 602

  mean_t = _mean_rigid_t_nhc_after_burn(
      dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn
  )
  assert abs(mean_t - 300.0) < 5.0, f"dt={dt_fs} fs: T={mean_t:.1f} K, expected 300 ± 5 K"
  print(f"✓ NHC at dt={dt_fs} fs: T={mean_t:.1f} K (target 300 ± 5 K)")
