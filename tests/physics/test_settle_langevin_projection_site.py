"""Regression: ``projection_site=post_settle_vel`` must not skip post-O rigid OU projection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water


def _dof_rigid_tip3p_waters(n_waters: int) -> float:
  return float(6 * n_waters - 3)


def _mean_rigid_t_after_burn(
  *,
  projection_site: str,
  n_waters: int,
  seed: int,
  steps: int,
  burn: int,
) -> float:
  jax.config.update("jax_enable_x64", True)
  temperature_k = 300.0
  dt_fs = 2.0
  gamma_ps = 1.0
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
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
    projection_site=projection_site,
    settle_velocity_iters=10,
  )
  apply_j = jax.jit(apply_s)
  dof_rigid = _dof_rigid_tip3p_waters(n_waters)
  state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)
  acc: list[float] = []
  for step in range(steps):
    state = apply_j(state)
    if step >= burn:
      ke_r = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
      acc.append(2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL))
  return float(np.mean(acc)) if acc else float("nan")


def test_post_settle_vel_rigid_mean_t_near_post_o() -> None:
  """``post_settle_vel`` used to omit the post-O projection and run far too hot; keep aligned."""
  n_waters = 2
  seed = 701
  steps = 600
  burn = 200
  t_post_o = _mean_rigid_t_after_burn(
    projection_site="post_o", n_waters=n_waters, seed=seed, steps=steps, burn=burn
  )
  t_post_sv = _mean_rigid_t_after_burn(
    projection_site="post_settle_vel", n_waters=n_waters, seed=seed, steps=steps, burn=burn
  )
  assert abs(t_post_o - t_post_sv) < 25.0, f"post_o={t_post_o} post_settle_vel={t_post_sv}"
  assert t_post_sv < 450.0, f"expected post_settle_vel not blow up, got {t_post_sv}"
