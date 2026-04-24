"""Temperature control validation for SETTLE with Langevin dynamics.

Phase 2C: Constrained-subspace OU thermostat implementation.
Tests validate that noise is sampled in the 6D rigid-body subspace, providing
correct equipartition across the 6*N_w-3 translational and rotational DOF.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water

def _dof_rigid_tip3p_waters(n_waters: int) -> float:
  return float(6 * n_waters - 3)

def _mean_rigid_t_after_burn(*, dt_fs: float, n_waters: int, seed: int, steps: int, burn: int) -> float:
  jax.config.update("jax_enable_x64", True)
  temperature_k = 300.0
  gamma_ps = 1.0
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False, project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10)
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

def test_temperature_dt1fs_near_target() -> None:
  """dt=1.0 fs, 100 ps: mean T within 15K of 300K target.

  Validates constrained-subspace OU thermostat: noise is sampled directly in
  the 6D rigid-body subspace for each water, ensuring correct equipartition.

  Note: Tolerance is ±15K (5%) to account for chi-squared distribution variance
  in 6-DOF rigid-body kinetic energy. For dt=1fs with gamma=1ps⁻¹, equilibration
  is slower; the tighter ±5K tolerance is more appropriate for dt=2fs tests.
  """
  n_waters = 2
  dt_fs = 1.0
  sim_ps = 100.0
  steps = int(sim_ps * 1000.0 / dt_fs)
  burn = max(100, steps // 3)
  seed = 601
  mean_t = _mean_rigid_t_after_burn(dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn)
  assert abs(mean_t - 300.0) < 15.0, f"dt={dt_fs} fs: T={mean_t:.1f} K, expected 300 ± 15 K"

def test_temperature_dt2fs_near_target() -> None:
  """dt=2.0 fs, 100 ps: mean T within 5K of 300K target.

  Tests constrained-subspace thermostat at larger timestep (dt=2fs).
  Larger dt → stronger per-step Langevin coupling (c2≈0.421) → faster equilibration.
  Tighter ±5K tolerance is appropriate here since equilibration is faster.
  """
  n_waters = 2
  dt_fs = 2.0
  sim_ps = 100.0
  steps = int(sim_ps * 1000.0 / dt_fs)
  burn = max(100, steps // 3)
  seed = 602
  mean_t = _mean_rigid_t_after_burn(dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn)
  assert abs(mean_t - 300.0) < 5.0, f"dt={dt_fs} fs: T={mean_t:.1f} K, expected 300 ± 5 K"

def test_equipartition_chi2() -> None:
  """Equipartition: velocity distribution matches Maxwell-Boltzmann (KS p > 0.05).

  Validates that the constrained OU noise produces a velocity distribution
  consistent with the target Maxwell-Boltzmann ensemble.
  """
  jax.config.update("jax_enable_x64", True)
  n_waters = 2
  dt_fs = 1.0
  sim_ps = 50.0
  steps = int(sim_ps * 1000.0 / dt_fs)
  burn = max(50, steps // 4)
  seed = 603
  temperature_k = 300.0
  gamma_ps = 1.0
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False, project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10)
  apply_j = jax.jit(apply_s)
  state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)
  velocities: list[float] = []
  for step in range(steps):
    state = apply_j(state)
    if step >= burn:
      v_atoms = state.momentum / state.mass
      velocities.extend(v_atoms.flatten().tolist())
  velocities_arr = np.array(velocities)
  m_flat = jnp.asarray(mass).reshape(-1)
  sigma_v = np.sqrt(kT / m_flat)
  v_norm = [velocities[i] / sigma_v[i % len(m_flat)] for i in range(len(velocities))]
  v_norm_arr = np.array(v_norm)
  ks_stat, ks_pval = stats.kstest(v_norm_arr, 'norm')
  assert ks_pval > 0.05, f"Equipartition KS test failed: p={ks_pval:.4f}, expected > 0.05"


def _mean_rigid_t_csvr_after_burn(*, dt_fs: float, n_waters: int, seed: int, steps: int, burn: int) -> float:
  """Helper for CSVR temperature tests. Same as Langevin version but uses settle_csvr."""
  jax.config.update("jax_enable_x64", True)
  temperature_k = 300.0
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_csvr(energy_fn, shift_fn, dt=dt_akma, kT=kT, mass=mass, water_indices=water_indices, box=box_vec, remove_com=True)
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


def test_temperature_csvr_dt1fs_near_target() -> None:
  """CSVR: dt=1.0 fs, 100 ps: mean T within 15K of 300K target.

  Tolerance is ±15K (5%) to account for chi-squared distribution variance
  in 6-DOF rigid-body kinetic energy. For dt=1fs with CSVR, equilibration
  is slower; the tighter ±5K tolerance is more appropriate for dt=2fs tests.
  """
  n_waters = 2
  dt_fs = 1.0
  sim_ps = 100.0
  steps = int(sim_ps * 1000.0 / dt_fs)
  burn = max(100, steps // 3)
  seed = 701
  mean_t = _mean_rigid_t_csvr_after_burn(dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn)
  assert abs(mean_t - 300.0) < 15.0, f"CSVR dt={dt_fs} fs: T={mean_t:.1f} K, expected 300 ± 15 K"


def test_temperature_csvr_dt2fs_near_target() -> None:
  """CSVR: dt=2.0 fs, 100 ps: mean T within 5K of 300K target.

  This is the CRITICAL TEST: Langevin fails here (507K), CSVR must pass.
  Scalar rescaling commutes with SETTLE constraints, allowing larger timesteps.
  """
  n_waters = 2
  dt_fs = 2.0
  sim_ps = 100.0
  steps = int(sim_ps * 1000.0 / dt_fs)
  burn = max(100, steps // 3)
  seed = 702
  mean_t = _mean_rigid_t_csvr_after_burn(dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn)
  assert abs(mean_t - 300.0) < 5.0, f"CSVR dt={dt_fs} fs: T={mean_t:.1f} K, expected 300 ± 5 K"


def test_equipartition_csvr_dt2fs_chi2() -> None:
  """CSVR: Equipartition at dt=2.0 fs (KS p > 0.05).

  Validates velocity distribution matches Maxwell-Boltzmann at the
  critical timestep where Langevin fails. This is the actual proof
  that CSVR fixes the distributional problem (KS p=0.0000 in Langevin).
  """
  jax.config.update("jax_enable_x64", True)
  n_waters = 8
  dt_fs = 2.0
  sim_ps = 500.0
  steps = int(sim_ps * 1000.0 / dt_fs)
  burn = max(200, steps // 4)
  seed = 704
  temperature_k = 300.0
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_csvr(energy_fn, shift_fn, dt=dt_akma, kT=kT, mass=mass, water_indices=water_indices, box=box_vec, remove_com=True)
  apply_j = jax.jit(apply_s)
  state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)
  velocities: list[float] = []
  for step in range(steps):
    state = apply_j(state)
    if step >= burn:
      v_atoms = state.momentum / state.mass
      velocities.extend(v_atoms.flatten().tolist())
  velocities_arr = np.array(velocities)
  m_flat = jnp.asarray(mass).reshape(-1)
  sigma_v = np.sqrt(kT / m_flat)
  v_norm = [velocities[i] / sigma_v[i % len(m_flat)] for i in range(len(velocities))]
  v_norm_arr = np.array(v_norm)
  ks_stat, ks_pval = stats.kstest(v_norm_arr, 'norm')
  assert ks_pval > 0.05, f"CSVR Equipartition KS test failed at dt=2fs: p={ks_pval:.4f}, expected > 0.05"


def test_n_dof_thermostated_protein_only() -> None:
  """Unit test: _n_dof_thermostated with protein only (no water)."""
  from prolix.physics.settle import _n_dof_thermostated
  assert _n_dof_thermostated(n_atoms=10, n_waters=0, n_constraint_pairs=0, remove_com=True) == 27


def test_csvr_lambda_statistics() -> None:
  """Direct unit test: _csvr_compute_lambda draws lambda^2 consistent with Bussi 2007.

  At tau=dt (c1=exp(-1)), the Bussi formula draws lambda values that stochastically
  equilibrate kinetic energy toward the target. Tests that many samples are:
  (1) non-negative, (2) finite, and (3) have the correct central tendency.
  """
  from prolix.physics.settle import _csvr_compute_lambda
  jax.config.update("jax_enable_x64", True)
  kT = 300.0 * BOLTZMANN_KCAL  # kcal/mol
  n_dof = 9  # 2 waters, COM removed: 6*2 - 3
  target_ke = 0.5 * n_dof * kT
  ke_current = jnp.array(3.0 * target_ke, dtype=jnp.float64)  # running hot (2x)
  # tau = dt → c1 = exp(-1) ≈ 0.368 → significant coupling per step
  dt = 1.0  # arbitrary; tau/dt ratio is what matters
  tau = 1.0  # → c1 = exp(-1)

  n_samples = 5000
  key = jax.random.PRNGKey(42)
  lambdas = []
  for _ in range(n_samples):
    lam, key = _csvr_compute_lambda(key, ke_current, n_dof, kT, dt, tau)
    lambdas.append(float(lam))
  lambdas_arr = np.array(lambdas)
  mean_lam_sq = np.mean(lambdas_arr**2)

  # Bussi formula: lambda^2 = c1 + c2_corrected*(R^2 + S) + 2*R*sqrt(c1*c2_corrected)
  # where c1=exp(-1)≈0.368, c2_corrected≈0.105
  # E[lambda^2] ≈ c1 + c2_corrected*E[R^2+S] ≈ 0.368 + 0.105*10 ≈ 1.418 (rough)
  # Variance is significant due to random R and S; expect range ~0.5 to 2.5
  assert all(l >= 0.0 for l in lambdas_arr), "lambda must be non-negative"
  assert np.all(np.isfinite(lambdas_arr)), "lambda must be finite"
  assert 0.5 < mean_lam_sq < 2.5, f"Lambda^2 mean={mean_lam_sq:.3f} should be in typical equilibrium range [0.5, 2.5]"


def test_langevin_with_constraints_null_constraint() -> None:
  """langevin_with_constraints with constraint=None runs without error."""
  from prolix.physics.settle import langevin_with_constraints
  jax.config.update("jax_enable_x64", True)
  n_atoms = 3
  kT = 300.0 * BOLTZMANN_KCAL
  dt_akma = 0.5 / AKMA_TIME_UNIT_FS
  displacement_fn, shift_fn = pbc.create_periodic_space(
    jnp.array([20.0, 20.0, 20.0], dtype=jnp.float64)
  )
  # Simple harmonic energy (no periodic effects at origin)
  def energy_fn(R, **kwargs):
    return 0.5 * jnp.sum(R**2)
  mass = jnp.ones(n_atoms, dtype=jnp.float64)
  gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3
  init_s, apply_s = langevin_with_constraints(
    energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma, mass=mass,
    constraint=None, water_indices=None, project_ou_momentum_rigid=False
  )
  apply_j = jax.jit(apply_s)
  R0 = jnp.zeros((n_atoms, 3), dtype=jnp.float64) + 0.1
  state = init_s(jax.random.PRNGKey(99), R0, mass=mass)
  pos_init = state.position
  for _ in range(100):
    state = apply_j(state)
  assert not jnp.allclose(state.position, pos_init), "Position should advance"
  assert jnp.all(jnp.isfinite(state.position)), "Position must be finite"
  assert jnp.all(jnp.isfinite(state.momentum)), "Momentum must be finite"
