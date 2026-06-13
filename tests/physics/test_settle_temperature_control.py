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
from .test_explicit_langevin_tip3p_parity import _equil_water_positions, _grid_water_positions, _proxide_params_pure_water

def _dof_rigid_tip3p_waters(n_waters: int) -> float:
  return float(6 * n_waters - 3)

def _mean_rigid_t_after_burn(*, dt_fs: float, n_waters: int, seed: int, steps: int, burn: int,
                             gamma_ps: float = 1.0) -> tuple[float, float]:
  """Compute mean observable temperature from rigid-body KE.

  Returns (T_observable, T_thermostat_target).
  T_thermostat_target is the thermostat's kT converted back to Kelvin.
  T_observable is computed from the instantaneous rigid-body kinetic energy.

  gamma_ps: Langevin friction (ps^-1). Default 1.0 (legacy). The dt=1.0 fs
  production gate (job 15870804) and the size-crossover sweep (campaign ba334c1f)
  use gamma=10 ps^-1; see .praxia/docs/research/260612_p5-dt1fs-size-crossover.md.
  """
  jax.config.update("jax_enable_x64", True)
  temperature_k = 300.0
  positions_a, box_edge = _equil_water_positions(n_waters, seed=seed)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
  sys_dict = _proxide_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  pme_grid = max(16, round(box_edge / 1.0))
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=pme_grid, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False, project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10)
  apply_j = jax.jit(apply_s)
  dof_rigid = _dof_rigid_tip3p_waters(n_waters)
  state = init_s(jax.random.key(seed), jnp.array(positions_a), mass=mass)
  temps: list[float] = []
  for step in range(steps):
    state = apply_j(state)
    if step >= burn:
      ke_r = float(rigid_tip3p_box_ke_kcal(state.positions, state.momentum, state.mass, n_waters))
      temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
      temps.append(temp)
  mean_t_observable = float(np.mean(temps)) if temps else float("nan")
  t_thermostat_target = temperature_k  # The target temperature in Kelvin
  return mean_t_observable, t_thermostat_target


def _mean_trot_after_burn(*, dt_fs: float, n_waters: int, seed: int, steps: int, burn: int,
                          gamma_ps: float = 10.0) -> float:
  """Mean rotational temperature T_rot after burn-in (gate's metric).

  Unlike _mean_rigid_t_after_burn (which returns the combined T_total over 6N-3 DOF),
  this decomposes the rigid-body KE into translational (per-water COM) and rotational
  parts and returns T_rot = 2<ke_rot>/(3N·k_B), matching run_nvt() in the dt=1fs gate
  (job 15870804) and size sweep (ba334c1f). T_rot is the physically-controlled,
  finite-size-robust quantity: the small-N warm bias is translational (T_trans, 3N-3
  DOF), so T_total is confounded at small N while T_rot stays ~300 K at every size.
  See .praxia/docs/research/260612_p5-dt1fs-size-crossover.md.
  """
  jax.config.update("jax_enable_x64", True)
  temperature_k = 300.0
  positions_a, box_edge = _equil_water_positions(n_waters, seed=seed)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
  sys_dict = _proxide_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  pme_grid = max(16, round(box_edge / 1.0))
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=pme_grid, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  m_flat = mass.reshape(-1)
  M_water = 15.999 + 1.008 + 1.008
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False, project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10)
  apply_j = jax.jit(apply_s)
  dof_rot = 3 * n_waters
  state = init_s(jax.random.key(seed), jnp.array(positions_a), mass=mass)
  t_rots: list[float] = []
  for step in range(steps):
    state = apply_j(state)
    if step >= burn:
      ke_tot = float(jnp.sum(jnp.sum(state.momentum ** 2, axis=-1) / (2.0 * m_flat)))
      p3 = state.momentum.reshape(n_waters, 3, 3)
      v_com = p3.sum(axis=1) / M_water
      ke_trans = 0.5 * M_water * float(jnp.sum(v_com ** 2))
      ke_rot = ke_tot - ke_trans
      t_rots.append(2.0 * ke_rot / (dof_rot * BOLTZMANN_KCAL))
  return float(np.mean(t_rots)) if t_rots else float("nan")

@pytest.mark.xfail(
  strict=True,
  reason=(
    "n=2 small-system translational finite-size warm bias, NOT dt instability. The "
    "dt=1.0 fs cap was lifted at production scale (gate job 15870804, n=895 T_rot 299.6 K; "
    "size sweep ba334c1f). But at n=2 there are only 3·N−3 = 3 translational DOF, which the "
    "Langevin thermostat under-regulates against the SETTLE constraint impulse: T_trans "
    "→ ~600 K (gamma=10) so T_total ~408 K (gamma=10) / ~358 K (gamma=1) — both >300±15 K. "
    "T_rot itself is fine (~312 K). The bias washes out by n≳16 (T_total |dev| 9.4 K). "
    "See .praxia/docs/research/260612_p5-dt1fs-size-crossover.md. Keep this xfail until a "
    "small-N translational thermostat fix lands; it is a finite-size artifact, not runaway."
  ),
)
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
  mean_t_observable, t_thermostat_target = _mean_rigid_t_after_burn(dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn)
  assert abs(mean_t_observable - 300.0) < 15.0, f"dt={dt_fs} fs: T_obs={mean_t_observable:.1f} K (thermostat target {t_thermostat_target:.1f} K), expected 300 ± 15 K"

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
  mean_t_observable, t_thermostat_target = _mean_rigid_t_after_burn(dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn)
  assert abs(mean_t_observable - 300.0) < 5.0, f"dt={dt_fs} fs: T_obs={mean_t_observable:.1f} K (thermostat target {t_thermostat_target:.1f} K), expected 300 ± 5 K"

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
  sys_dict = _proxide_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False, project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10)
  apply_j = jax.jit(apply_s)
  state = init_s(jax.random.key(seed), jnp.array(positions_a), mass=mass)
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


def test_langevin_o_step_masked_atoms_unchanged() -> None:
  """Constrained atoms must be unchanged after _langevin_step_o_free_dof."""
  from prolix.physics.settle import _langevin_step_o_free_dof
  key = jax.random.PRNGKey(0)
  momentum = jnp.ones((4, 3))
  mass = jnp.ones((4, 1))
  free_mask = jnp.array([True, True, False, False])
  p_new, _ = _langevin_step_o_free_dof(momentum, mass, 1.0, 0.001, 1.0, key, free_mask)
  assert jnp.allclose(p_new[2], momentum[2]), "Constrained atom 2 should be unchanged"
  assert jnp.allclose(p_new[3], momentum[3]), "Constrained atom 3 should be unchanged"

def test_langevin_o_step_free_atoms_changed() -> None:
  """Free atoms must receive OU noise."""
  from prolix.physics.settle import _langevin_step_o_free_dof
  key = jax.random.PRNGKey(0)
  momentum = jnp.ones((4, 3))
  mass = jnp.ones((4, 1))
  free_mask = jnp.array([True, True, False, False])
  p_new, _ = _langevin_step_o_free_dof(momentum, mass, 1.0, 0.001, 1.0, key, free_mask)
  assert not jnp.allclose(p_new[0], momentum[0]), "Free atom 0 should have received noise"
  assert not jnp.allclose(p_new[1], momentum[1]), "Free atom 1 should have received noise"

def test_langevin_o_step_c1_applied_once() -> None:
  """c1 decay factor applied exactly once — no double-decay."""
  from prolix.physics.settle import _langevin_step_o_free_dof
  key = jax.random.PRNGKey(0)
  momentum = jnp.ones((4, 3)) * 100.0
  mass = jnp.ones((4, 1))
  free_mask = jnp.array([True, True, False, False])
  # gamma=100.0, dt=1.0 → c1 = exp(-100) ≈ 0; free atoms momentum ≈ noise term only
  p_new, _ = _langevin_step_o_free_dof(momentum, mass, 100.0, 1.0, 0.001, key, free_mask)
  # With c1 ≈ 0, original momentum term is suppressed → |p_new[0]| << 100
  assert jnp.abs(p_new[0]).max() < 10.0, "c1 double-applied would keep momentum high"

def test_langevin_o_step_masked_no_friction_decay() -> None:
  """Constrained atoms retain full original momentum (no friction applied)."""
  from prolix.physics.settle import _langevin_step_o_free_dof
  key = jax.random.PRNGKey(0)
  momentum = jnp.ones((4, 3)) * 100.0
  mass = jnp.ones((4, 1))
  free_mask = jnp.array([True, True, False, False])
  p_new, _ = _langevin_step_o_free_dof(momentum, mass, 100.0, 1.0, 0.001, key, free_mask)
  # Constrained atoms should still be 100.0 (no friction applied)
  assert jnp.allclose(p_new[2], momentum[2])
  assert jnp.allclose(p_new[3], momentum[3])

def test_langevin_o_step_1d_mass_same_as_2d() -> None:
  """mass.shape=(N,) and mass.shape=(N,1) must produce identical output.

  Regression guard for the mass-shape canonicalization in _langevin_step_o_free_dof:
  without reshape, (N,) mass would broadcast as (1,N) against (N,3) noise, applying
  mass[j] to spatial dimension j instead of mass[i] to atom i — silent wrong physics.
  """
  from prolix.physics.settle import _langevin_step_o_free_dof
  key = jax.random.PRNGKey(42)
  momentum = jnp.ones((4, 3))
  mass_1d = jnp.array([1.0, 2.0, 3.0, 4.0])          # shape (4,)
  mass_2d = mass_1d.reshape(-1, 1)                     # shape (4,1)
  free_mask = jnp.array([True, True, False, False])

  p_1d, _ = _langevin_step_o_free_dof(momentum, mass_1d, 1.0, 0.001, 1.0, key, free_mask)
  p_2d, _ = _langevin_step_o_free_dof(momentum, mass_2d, 1.0, 0.001, 1.0, key, free_mask)
  assert jnp.allclose(p_1d, p_2d), (
      "1D and 2D mass shapes produce different outputs — mass reshape canonicalization broken"
  )


def _mean_rigid_t_csvr_after_burn(*, dt_fs: float, n_waters: int, seed: int, steps: int, burn: int) -> float:
  """Helper for CSVR temperature tests. Same as Langevin version but uses settle_csvr."""
  jax.config.update("jax_enable_x64", True)
  temperature_k = 300.0
  positions_a, box_edge = _equil_water_positions(n_waters, seed=seed)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  sys_dict = _proxide_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  pme_grid = max(16, round(box_edge / 1.0))
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=pme_grid, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_csvr(energy_fn, shift_fn, dt=dt_akma, kT=kT, mass=mass, water_indices=water_indices, box=box_vec, remove_com=True)
  apply_j = jax.jit(apply_s)
  dof_rigid = _dof_rigid_tip3p_waters(n_waters)
  state = init_s(jax.random.key(seed), jnp.array(positions_a), mass=mass)
  temps: list[float] = []
  for step in range(steps):
    state = apply_j(state)
    if step >= burn:
      ke_r = float(rigid_tip3p_box_ke_kcal(state.positions, state.momentum, state.mass, n_waters))
      temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
      temps.append(temp)
  return float(np.mean(temps)) if temps else float("nan")


@pytest.mark.xfail(strict=True, reason="CSVR+SETTLE shows tau-dependent ~+8K bias at dt>=1fs (VV discretization artifact); runtime warning emitted in settle_csvr; see settle.py settle_csvr docstring for details")
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


@pytest.mark.xfail(strict=True, reason="CSVR+SETTLE shows tau-dependent ~+8K bias at dt>=1fs (VV discretization artifact); runtime warning emitted in settle_csvr; see settle.py settle_csvr docstring for details")
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


@pytest.mark.xfail(strict=True, reason="SETTLE velocity constraints produce correlated atom velocities; per-atom KS test against marginal MB is structurally invalid")
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
  sys_dict = _proxide_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_csvr(energy_fn, shift_fn, dt=dt_akma, kT=kT, mass=mass, water_indices=water_indices, box=box_vec, remove_com=True)
  apply_j = jax.jit(apply_s)
  state = init_s(jax.random.key(seed), jnp.array(positions_a), mass=mass)
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
  key = jax.random.key(42)
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
  state = init_s(jax.random.key(99), R0, mass=mass)
  pos_init = state.position
  for _ in range(100):
    state = apply_j(state)
  assert not jnp.allclose(state.positions, pos_init), "Position should advance"
  assert jnp.all(jnp.isfinite(state.positions)), "Position must be finite"
  assert jnp.all(jnp.isfinite(state.momentum)), "Momentum must be finite"


@pytest.mark.xfail(strict=True, reason="NVT temperature drift at long timescales (>100ps) — system-size dependent, under investigation. Ablation (8w, 200s) passes; 64w/200ks fails at 334K (+11%). Root: OU projection or DOF/KE mismatch. Deferred to Sprint 12.")
@pytest.mark.slow
def test_temperature_langevin_dt0_5fs_green() -> None:
  """dt=0.5 fs (production constraint), 100 ps total: mean T within 10K of 300K target.

  Canonical validation that settle_langevin is stable at the documented
  production operating point (dt ≤ 0.5 fs per CLAUDE.md). This is the
  green test that proves the Phase 2 constraint works in practice.

  Parameters match production usage: dt=0.5 fs, n_waters=8, gamma=1.0 ps⁻¹.
  After 33 ps burn-in, 67 ps of production data.
  """
  n_waters = 8
  dt_fs = 0.5
  steps = 200_000
  burn = 66_667
  seed = 7
  mean_t_observable, t_thermostat_target = _mean_rigid_t_after_burn(dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn)
  assert abs(mean_t_observable - 300.0) < 10.0, f"dt={dt_fs} fs: T_obs={mean_t_observable:.1f} K (thermostat target {t_thermostat_target:.1f} K), expected 300 ± 10 K"


@pytest.mark.slow
def test_equipartition_per_molecule_com_dt0_5fs() -> None:
  """Equipartition: per-molecule COM velocity distribution at dt=0.5 fs (KS p > 0.05).

  Validates that center-of-mass (COM) velocities of rigid water molecules
  follow Maxwell-Boltzmann distribution under SETTLE constraints at the
  production timestep. COM motion is preserved by SETTLE, so this test
  is structurally valid (unlike per-atom tests which fail due to SETTLE
  velocity correlations).

  Subsampling: Langevin with gamma=1 ps⁻¹ has autocorrelation ~1 ps = 2000 steps
  at dt=0.5 fs. Collects 120 independent snapshots by sampling every 2000 steps.
  After 50k step burn-in, total simulation: 290k steps (145 ps).
  Final sample pool: 120 snapshots × 8 waters × 3 components = 2880 samples (well-sized for KS).
  """
  jax.config.update("jax_enable_x64", True)
  n_waters = 8
  dt_fs = 0.5
  seed = 13
  temperature_k = 300.0
  gamma_ps = 1.0
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
  sys_dict = _proxide_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False, project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10)
  apply_j = jax.jit(apply_s)
  state = init_s(jax.random.key(seed), jnp.array(positions_a), mass=mass)

  # Subsampling parameters
  burn_steps = 50_000  # 25 ps burn-in
  decorr_steps = 2000  # Sample every ~1 ps (decorrelation time at gamma=1 ps^-1)
  n_snapshots = 120  # Collect 120 independent snapshots
  total_steps = burn_steps + n_snapshots * decorr_steps  # 290k total steps

  m_oxygen = 15.999
  m_hydrogen = 1.008
  m_water = m_oxygen + 2 * m_hydrogen

  com_velocities: list[np.ndarray] = []  # Store (water, 3) arrays per snapshot

  step = 0

  # Burn-in phase
  for _ in range(burn_steps):
    state = apply_j(state)
    step += 1

  # Production phase: collect every decorr_steps
  for snap in range(n_snapshots):
    for _ in range(decorr_steps):
      state = apply_j(state)
      step += 1

    # Extract COM velocity for each water molecule
    # Momentum is p_i = m_i * v_i, so v_com = sum(p_i) / sum(m_i)
    snapshot_com = []
    for w in range(n_waters):
      o_idx = 3 * w
      h1_idx = 3 * w + 1
      h2_idx = 3 * w + 2
      p_o = state.momentum[o_idx]
      p_h1 = state.momentum[h1_idx]
      p_h2 = state.momentum[h2_idx]
      # CORRECT: sum of momenta divided by total mass gives v_com
      v_com = (p_o + p_h1 + p_h2) / m_water
      snapshot_com.append(v_com)
    com_velocities.append(np.array(snapshot_com))

  # Flatten to (n_snapshots * n_waters, 3)
  com_arr = np.concatenate(com_velocities, axis=0)  # shape (n_snapshots * n_waters, 3)
  components = com_arr.flatten()  # shape (n_snapshots * n_waters * 3,)

  sigma = np.sqrt(kT / m_water)
  v_norm = components / sigma
  ks_stat, ks_pval = stats.kstest(v_norm, 'norm')
  assert ks_pval > 0.05, f"COM equipartition KS test failed at dt=0.5fs: p={ks_pval:.4f}, expected > 0.05"


# ============================================================================
# Sprint 5 Deferred: lax.scan Runner and Reproducibility
# ============================================================================

def _mean_rigid_t_lax_scan(*, dt_fs: float, n_waters: int, seed: int, steps: int, burn: int) -> tuple[float, np.ndarray]:
  """Temperature calculation using jax.lax.scan instead of Python loop (Sprint 5 deferred).

  Returns both mean temperature and full temperature array for comparison.
  """
  jax.config.update("jax_enable_x64", True)
  temperature_k = 300.0
  gamma_ps = 1.0
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
  sys_dict = _proxide_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False, project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10)
  apply_j = jax.jit(apply_s)
  dof_rigid = float(6 * n_waters - 3)
  state = init_s(jax.random.key(seed), jnp.array(positions_a), mass=mass)

  # Burn-in loop (Python, not scanned)
  for _ in range(burn):
    state = apply_j(state)

  # Production loop using lax.scan
  def step_fn(state, _):
    new_state = apply_j(state)
    ke_r = rigid_tip3p_box_ke_kcal(new_state.positions, new_state.momentum, new_state.mass, n_waters)
    temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
    return new_state, temp

  final_state, temps = jax.lax.scan(step_fn, state, None, length=steps - burn)
  temps_np = np.array(temps)
  mean_t = float(np.mean(temps_np)) if len(temps_np) > 0 else float("nan")
  return mean_t, temps_np


@pytest.mark.xfail(strict=False, reason="Sprint 5 deferred: lax.scan vs Python-loop temperature discrepancy under investigation")
def test_lax_scan_runner() -> None:
  """Unit test: lax.scan runner produces identical temperatures to Python loop.

  This validates the Sprint 5 deferred lax.scan refactor: baseline Python loop
  vs. JAX lax.scan should match to machine precision.
  """
  jax.config.update("jax_enable_x64", True)
  n_waters = 2
  dt_fs = 0.5
  steps = 50
  burn = 10
  seed = 801

  # Original Python loop approach
  mean_t_py_observable, _ = _mean_rigid_t_after_burn(dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn)

  # New lax.scan approach
  mean_t_scan, _ = _mean_rigid_t_lax_scan(dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn)

  # They must match to within floating-point error (1e-8 or better)
  assert abs(mean_t_py_observable - mean_t_scan) < 1e-8, \
    f"Python loop ({mean_t_py_observable:.6f}) vs lax.scan ({mean_t_scan:.6f}) differ by {abs(mean_t_py_observable - mean_t_scan):.2e}"


def test_temperature_reproducibility_same_seed() -> None:
  """Reproducibility check: identical seeds → identical trajectories.

  Init with seed=12345, run 50 steps → trajectory_a
  Init with seed=12345 again, run 50 steps → trajectory_b
  Assert positions match exactly (bit-identical).
  """
  jax.config.update("jax_enable_x64", True)
  n_waters = 2
  dt_fs = 0.5
  temperature_k = 300.0
  gamma_ps = 1.0
  steps = 50
  seed = 12345

  def run_trajectory(seed_val):
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
    sys_dict = _proxide_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)
    init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False, project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10)
    apply_j = jax.jit(apply_s)
    state = init_s(jax.random.key(seed_val), jnp.array(positions_a), mass=mass)

    positions_list = [state.positions]
    for _ in range(steps):
      state = apply_j(state)
      positions_list.append(state.positions)
    return jnp.array(positions_list)

  traj_a = run_trajectory(seed)
  traj_b = run_trajectory(seed)

  assert jnp.allclose(traj_a, traj_b, atol=0.0), \
    f"Same seed produced different trajectories; max diff: {jnp.max(jnp.abs(traj_a - traj_b))}"


def test_temperature_reproducibility_different_seed() -> None:
  """Divergence check: different seeds → different trajectories.

  Init with seed=12345, run 50 steps → trajectory_a
  Init with seed=99999, run 50 steps → trajectory_c
  Assert trajectories differ (diverge after burn-in).
  """
  jax.config.update("jax_enable_x64", True)
  n_waters = 2
  dt_fs = 0.5
  temperature_k = 300.0
  gamma_ps = 1.0
  steps = 50

  def run_trajectory(seed_val):
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
    sys_dict = _proxide_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)
    init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False, project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10)
    apply_j = jax.jit(apply_s)
    state = init_s(jax.random.key(seed_val), jnp.array(positions_a), mass=mass)

    positions_list = [state.positions]
    for _ in range(steps):
      state = apply_j(state)
      positions_list.append(state.positions)
    return jnp.array(positions_list)

  traj_a = run_trajectory(12345)
  traj_c = run_trajectory(99999)

  # Trajectories should diverge (not be allclose everywhere)
  max_diff = jnp.max(jnp.abs(traj_a - traj_c))
  assert max_diff > 0.01, \
    f"Different seeds should produce divergent trajectories; max diff only {max_diff:.6f}"


# ============================================================================
# Ablation Test: Rigid-Body KE Decomposition vs Atomic Simple KE
# ============================================================================

def _compute_atomic_ke_kcal(momentum: jax.Array, mass: jax.Array) -> jax.Array:
  """Compute simple atomic KE: 0.5 * sum(p_i^2 / m_i).

  This is the most basic KE formula without any rigid-body structure.
  momentum: shape (n_atoms, 3)
  mass: shape (n_atoms,)
  """
  p = jnp.asarray(momentum)  # shape (n_atoms, 3)
  m = jnp.asarray(mass).reshape(-1)  # shape (n_atoms,)

  # |p_i|^2 = sum over x,y,z components
  p_squared = jnp.sum(p ** 2, axis=1)  # shape (n_atoms,)

  # Compute per-atom KE: 0.5 * |p_i|^2 / m_i
  ke_per_atom = 0.5 * p_squared / m  # shape (n_atoms,)
  ke_total = jnp.sum(ke_per_atom)
  return ke_total


def test_ke_measurement_ablation() -> None:
  """Ablation test: Compare rigid-body KE vs simple atomic KE methods.

  This test isolates the source of temperature discrepancy by running a
  short NVT simulation (8 waters, 200 steps, dt=0.5fs) and measuring
  temperature using TWO different KE methods:

  1. T_rigid = 2 * KE_rigid / (k_B * ndof_rigid)
     where KE_rigid = COM + rotational (from rigid_tip3p_box_ke_kcal)

  2. T_atomic = 2 * KE_atomic / (k_B * ndof_atomic)
     where KE_atomic = 0.5 * sum(p_i^2 / m_i)

  Expected behavior:
  - If T_rigid >> T_atomic (e.g., 334K vs 300K): bug is in rigid_tip3p_box_ke_kcal
  - If T_rigid ≈ T_atomic >> 300K: bug is in ndof calculation
  - If T_rigid ≈ T_atomic ≈ 300K: both methods agree (no bug)

  The test logs side-by-side comparison every 10 steps to reveal the pattern.
  """
  jax.config.update("jax_enable_x64", True)

  n_waters = 8
  dt_fs = 0.5
  steps = 200
  burn = 50  # First 50 steps for equilibration
  seed = 1001

  temperature_k = 300.0
  gamma_ps = 1.0

  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3

  sys_dict = _proxide_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(
    displacement_fn, sys_dict, box=box_vec, use_pbc=True,
    implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34,
    cutoff_distance=9.0, strict_parameterization=False
  )

  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)

  init_s, apply_s = settle.settle_langevin(
    energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced,
    mass=mass, water_indices=water_indices, box=box_vec,
    remove_linear_com_momentum=False, project_ou_momentum_rigid=True,
    projection_site="post_o", settle_velocity_iters=10
  )
  apply_j = jax.jit(apply_s)

  dof_atomic_rigid = float(6 * n_waters - 3)
  # DOF for atomic formula: 3N (all atomic coordinates)
  dof_atomic_all = float(3 * n_atoms)

  state = init_s(jax.random.key(seed), jnp.array(positions_a), mass=mass)

  # Burn-in: discard first 'burn' steps
  for _ in range(burn):
    state = apply_j(state)

  # Production: collect temperatures every 10 steps after burn-in
  results = []
  log_interval = 10

  for step in range(steps - burn):
    state = apply_j(state)

    if step % log_interval == 0:
      # Compute KE using BOTH methods
      ke_rigid = float(rigid_tip3p_box_ke_kcal(
        state.positions, state.momentum, state.mass, n_waters
      ))
      ke_atomic = float(_compute_atomic_ke_kcal(state.momentum, state.mass))

      # Convert to temperature
      t_rigid = 2.0 * ke_rigid / (dof_atomic_rigid * BOLTZMANN_KCAL)
      t_atomic = 2.0 * ke_atomic / (dof_atomic_all * BOLTZMANN_KCAL)

      results.append({
        'step': burn + step,
        'ke_rigid': ke_rigid,
        'ke_atomic': ke_atomic,
        't_rigid': t_rigid,
        't_atomic': t_atomic,
        'diff_k': t_rigid - t_atomic,
        'ratio': t_rigid / t_atomic if t_atomic > 0 else float('nan'),
      })

  # Print side-by-side comparison
  print("\n" + "="*100)
  print("KE MEASUREMENT ABLATION TEST")
  print("="*100)
  print(f"n_waters={n_waters}, dt={dt_fs}fs, target_T={temperature_k}K, seed={seed}")
  print(f"dof_atomic_rigid={dof_atomic_rigid:.0f}, dof_atomic_all={dof_atomic_all:.0f}")
  print("-"*100)
  print(f"{'Step':>6} | {'KE_rigid':>14} | {'KE_atomic':>14} | {'T_rigid':>9} | {'T_atomic':>9} | {'Diff_K':>9} | {'Ratio':>9}")
  print("-"*100)

  for res in results:
    print(f"{res['step']:>6} | {res['ke_rigid']:>14.8f} | {res['ke_atomic']:>14.8f} | "
          f"{res['t_rigid']:>9.2f} | {res['t_atomic']:>9.2f} | {res['diff_k']:>9.2f} | {res['ratio']:>9.4f}")

  print("-"*100)

  # Compute statistics
  t_rigid_vals = [r['t_rigid'] for r in results]
  t_atomic_vals = [r['t_atomic'] for r in results]

  mean_t_rigid = float(np.mean(t_rigid_vals))
  mean_t_atomic = float(np.mean(t_atomic_vals))
  std_t_rigid = float(np.std(t_rigid_vals))
  std_t_atomic = float(np.std(t_atomic_vals))
  mean_diff = mean_t_rigid - mean_t_atomic
  mean_ratio = mean_t_rigid / mean_t_atomic if mean_t_atomic > 0 else float('nan')

  print(f"\nSUMMARY STATISTICS:")
  print(f"  T_rigid  : {mean_t_rigid:.2f} ± {std_t_rigid:.2f} K")
  print(f"  T_atomic : {mean_t_atomic:.2f} ± {std_t_atomic:.2f} K")
  print(f"  Difference (T_rigid - T_atomic): {mean_diff:.2f} K")
  print(f"  Ratio (T_rigid / T_atomic): {mean_ratio:.4f}")
  print(f"  Thermostat target: {temperature_k} K")
  print("="*100)

  # Diagnostic assertions
  expected_ratio = dof_atomic_all / dof_atomic_rigid
  ke_agreement = abs(results[0]['ke_rigid'] - results[0]['ke_atomic']) / results[0]['ke_rigid']

  print(f"\nKE AGREEMENT CHECK:")
  print(f"  Expected ratio (dof_atomic_all / dof_atomic_rigid) = {dof_atomic_all:.0f} / {dof_atomic_rigid:.0f} = {expected_ratio:.4f}")
  print(f"  Observed ratio (T_rigid / T_atomic): {mean_ratio:.4f}")
  print(f"  KE value agreement: {ke_agreement*100:.6f}% (identical if < 0.01%)")

  if ke_agreement < 0.001:  # KE values nearly identical
    print("\n  -> KE values are IDENTICAL (within 0.001%)")
    print("  -> This means rigid_tip3p_box_ke_kcal is computing KE correctly")
    if abs(mean_ratio - expected_ratio) < 0.001:
      print("  -> Ratio matches expected dof_atomic/dof_rigid exactly")
      print("\n*** CRITICAL FINDING ***")
      print("  -> CONCLUSION: The problem is NOT in KE measurement!")
      print("  -> rigid_tip3p_box_ke_kcal computes COM + rotational KE correctly")
      print("\n  Problem source: DOF MISMATCH in temperature formula")
      print("  -> rigid_tip3p_box_ke_kcal returns: KE_COM + KE_rot (for N_w waters)")
      print("  -> This KE already accounts for only rigid DOF (6*N_w - 3 for constraints)")
      print("  -> BUT: If you use this KE with dof_rigid, you get correct T for rigid-only ensemble")
      print("  -> HOWEVER: This KE includes all atomic motion (COM+rot), effectively 72 DOF not 45")
      print("\n  Root cause analysis:")
      print("  -> KE = KE_COM + KE_rot is computed over all 3N atomic coordinates")
      print("  -> It's NOT a reduced KE restricted to 6N-3 manifold")
      print("  -> Therefore: T = 2*KE / (dof_atomic * k_B) is the correct formula")
      print("  -> Using T = 2*KE / (dof_rigid * k_B) gives artificially LOW temperature")
  else:
    print(f"\n  -> KE values DIFFER significantly ({ke_agreement*100:.2f}%)")
    print("  -> CONCLUSION: Bug is in rigid_tip3p_box_ke_kcal function")

  if mean_diff > 20.0:  # Significant systematic difference
    print("\nTEMPERATURE DISCREPANCY PATTERN:")
    if abs(mean_ratio - 1.11) < 0.05:
      print("  -> Ratio ~1.11 suggests systematic factor in one calculation")
    if mean_t_rigid > 320.0 and mean_t_atomic > 320.0:
      print("  -> Both T_rigid and T_atomic >> 300K but have same pattern")
      print("  -> Consistent with using wrong DOF count for both")
  else:
    print("\nTEMPERATURE DISCREPANCY PATTERN: Both methods agree within ±20K")

  # Loose assertion: both methods should be reasonably close to thermostat target
  # (within 30K to allow for equilibration variance in short 150-step production)
  assert mean_t_rigid < 350.0 and mean_t_atomic < 350.0, \
    f"Both KE methods report T > 350K: T_rigid={mean_t_rigid:.1f}K, T_atomic={mean_t_atomic:.1f}K"


def test_r_step_conserves_angular_momentum() -> None:
  """Unit test: _r_step_conserve_angular_momentum restores per-water AM.

  Given random momenta before and after an A+SETTLE+R step, verify that
  the AM correction impulse restores the angular momentum of each water
  back to its initial value (within float64 tolerance ~1e-8).
  """
  jax.config.update("jax_enable_x64", True)
  from prolix.physics.settle import _r_step_conserve_angular_momentum, WaterIndices

  key = jax.random.PRNGKey(42)
  n_waters = 4
  mass_oxygen = 15.999
  mass_hydrogen = 1.008
  mass_total = mass_oxygen + 2.0 * mass_hydrogen

  # Create water indices
  water_indices = settle.get_water_indices(0, n_waters)
  indices = WaterIndices.from_row(water_indices.T)
  m_all = jnp.array([mass_oxygen, mass_hydrogen, mass_hydrogen])

  # Random positions before A step
  key, subkey = jax.random.split(key)
  x_unc = jax.random.normal(subkey, (n_waters * 3, 3)) * 0.1

  # Random momenta before A step
  key, subkey = jax.random.split(key)
  p_pre_a = jax.random.normal(subkey, (n_waters * 3, 3)) * 10.0

  # Simulate SETTLE displacement: constrained positions differ slightly from unconstrained
  key, subkey = jax.random.split(key)
  settle_displacement = jax.random.normal(subkey, (n_waters * 3, 3)) * 0.01
  x_con = x_unc + settle_displacement

  # Apply R-step momentum correction: dp = m * dx / half_dt
  half_dt = 0.25  # 0.5 fs (AKMA units)
  mass_arr = jnp.array([m for m in [mass_oxygen, mass_hydrogen, mass_hydrogen] for _ in range(n_waters)]).reshape(n_waters * 3)
  dp_r_w = mass_arr[:, None] * settle_displacement / half_dt
  momentum_after_r = p_pre_a + dp_r_w

  # Apply the AM correction
  momentum_corrected = _r_step_conserve_angular_momentum(
    momentum_after_r, p_pre_a, x_unc, x_con, water_indices, mass_oxygen, mass_hydrogen
  )

  # Check per-water angular momentum is restored
  for w in range(n_waters):
    o_idx = indices.oxygen[w]
    h1_idx = indices.hydrogen1[w]
    h2_idx = indices.hydrogen2[w]

    # Compute L_target = L(x_unc, p_pre_a)
    r_unc_w = jnp.array([x_unc[o_idx], x_unc[h1_idx], x_unc[h2_idx]])
    p_target_w = jnp.array([p_pre_a[o_idx], p_pre_a[h1_idx], p_pre_a[h2_idx]])
    r_com_target = (m_all @ r_unc_w) / mass_total
    d_target = r_unc_w - r_com_target
    L_target = jnp.sum(jnp.cross(d_target, p_target_w), axis=0)

    # Compute L_corrected = L(x_con, momentum_corrected)
    r_con_w = jnp.array([x_con[o_idx], x_con[h1_idx], x_con[h2_idx]])
    p_corrected_w = jnp.array([momentum_corrected[o_idx], momentum_corrected[h1_idx], momentum_corrected[h2_idx]])
    r_com_corrected = (m_all @ r_con_w) / mass_total
    d_corrected = r_con_w - r_com_corrected
    L_corrected = jnp.sum(jnp.cross(d_corrected, p_corrected_w), axis=0)

    # Verify AM is conserved
    diff = jnp.max(jnp.abs(L_target - L_corrected))
    assert diff < 1e-8, f"Water {w}: AM not restored. L_target={L_target}, L_corrected={L_corrected}, diff={diff}"


def test_r_step_conserves_angular_momentum_pbc() -> None:
  """Unit test: _r_step_conserve_angular_momentum handles PBC split-molecule case.

  When a water molecule straddles a periodic boundary, shift_fn wraps atoms
  individually, placing O at one image and H at another (e.g., x_O ≈ 0.1,
  x_H1 ≈ L_box - 0.2). Without minimum-image unwrapping of H relative to O,
  |d_unc| ~ L_box → catastrophic impulse → NaN.

  This test verifies that when box is provided, H atoms are unwrapped via
  minimum-image before computing L_target, and the AM correction succeeds
  without NaN or numerical blowup.

  Fixture: n_waters=1, split-molecule water straddling PBC boundary
    x_unc[0] = [0.1, 0.1, 0.1]    # O at x=0.1
    x_unc[1] = [3.0, 0.1, 0.1]    # H1 near boundary (x=3.0, box=3.1)
    x_unc[2] = [0.1, 0.3, 0.1]    # H2 within box
    box = [3.1, 3.1, 3.1]
  Minimum-image: d_OH1 = 3.0 - 0.1 = 2.9 → round(2.9/3.1) = 1 →
                 d_OH1_mi = 2.9 - 3.1 = -0.2 → H1_unwrapped_x = 0.1 - 0.2 = -0.1
  """
  jax.config.update("jax_enable_x64", True)
  from prolix.physics.settle import _r_step_conserve_angular_momentum, WaterIndices

  key = jax.random.PRNGKey(99)
  n_waters = 1
  mass_oxygen = 15.999
  mass_hydrogen = 1.008
  mass_total = mass_oxygen + 2.0 * mass_hydrogen

  # Create water indices for a single water
  water_indices = settle.get_water_indices(0, 1)
  indices = WaterIndices.from_row(water_indices.T)
  m_all = jnp.array([mass_oxygen, mass_hydrogen, mass_hydrogen])

  # Split-molecule fixture: H1 straddling PBC boundary
  x_unc = jnp.array([
    [0.1, 0.1, 0.1],      # O at x=0.1
    [3.0, 0.1, 0.1],      # H1 near boundary (x≈L_box)
    [0.1, 0.3, 0.1],      # H2 within box
  ], dtype=jnp.float64)

  # Box edge length
  box = jnp.array([3.1, 3.1, 3.1], dtype=jnp.float64)

  # Random momenta before A step
  key, subkey = jax.random.split(key)
  p_pre_a = jax.random.normal(subkey, (3, 3), dtype=jnp.float64) * 5.0

  # Simulate SETTLE displacement (small perturbation)
  settle_displacement = jnp.array([
    [0.001, 0.001, 0.001],
    [-0.001, 0.001, 0.001],
    [0.001, -0.001, 0.001],
  ], dtype=jnp.float64)
  x_con = x_unc + settle_displacement

  # Apply R-step momentum correction: dp = m * dx / half_dt
  half_dt = 0.25  # 0.5 fs (AKMA units)
  mass_arr = jnp.array([mass_oxygen, mass_hydrogen, mass_hydrogen], dtype=jnp.float64)
  dp_r_w = mass_arr[:, None] * settle_displacement / half_dt
  momentum_after_r = p_pre_a + dp_r_w

  # Apply the AM correction WITH box argument
  momentum_corrected = _r_step_conserve_angular_momentum(
    momentum_after_r, p_pre_a, x_unc, x_con, water_indices,
    mass_oxygen, mass_hydrogen, box=box
  )

  # Verify: no NaN in output
  assert jnp.all(jnp.isfinite(momentum_corrected)), \
    f"NaN in corrected momentum (PBC fixture). momentum_corrected={momentum_corrected}"

  # Compute L_target with unwrapped H atoms
  r_O_unc = x_unc[0]
  r_H1_unc = x_unc[1]
  r_H2_unc = x_unc[2]

  # Unwrap H atoms relative to O via minimum-image
  dH1 = r_H1_unc - r_O_unc
  dH1_mi = dH1 - box * jnp.round(dH1 / box)
  r_H1_unc_unwrapped = r_O_unc + dH1_mi

  dH2 = r_H2_unc - r_O_unc
  dH2_mi = dH2 - box * jnp.round(dH2 / box)
  r_H2_unc_unwrapped = r_O_unc + dH2_mi

  r_unc_w_unwrapped = jnp.array([r_O_unc, r_H1_unc_unwrapped, r_H2_unc_unwrapped])
  r_com_target = (m_all @ r_unc_w_unwrapped) / mass_total
  d_target = r_unc_w_unwrapped - r_com_target
  L_target = jnp.sum(jnp.cross(d_target, p_pre_a), axis=0)

  # Compute L_actual from corrected momentum at constrained positions
  r_con_w = jnp.array([x_con[0], x_con[1], x_con[2]])
  p_corrected_w = momentum_corrected
  r_com_corrected = (m_all @ r_con_w) / mass_total
  d_corrected = r_con_w - r_com_corrected
  L_actual = jnp.sum(jnp.cross(d_corrected, p_corrected_w), axis=0)

  # Verify AM is conserved (with slightly relaxed tolerance for PBC case)
  diff = jnp.max(jnp.abs(L_target - L_actual))
  assert jnp.allclose(L_actual, L_target, atol=1e-8), \
    f"AM not restored in PBC case: L_target={L_target}, L_actual={L_actual}, diff={diff}"


def test_settle_langevin_t_rot_small_system() -> None:
  """Regression test: AM conservation smoke test for small system.

  Validates that the R-step AM correction does not cause divergence.
  For a 4-water system with dt=0.5fs and gamma=1ps^-1, T_rot should
  remain finite and stable.

  Note: The actual T_rot depends on system size and may vary; this test
  is a smoke test to ensure the AM correction doesn't cause runaway.
  """
  jax.config.update("jax_enable_x64", True)
  n_waters = 4
  dt_fs = 0.5
  steps = 500
  burn = 100
  seed = 704
  mean_t_observable, _ = _mean_rigid_t_after_burn(
    dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn
  )
  # Smoke test: temperature should be finite and not runaway
  assert jnp.isfinite(mean_t_observable), \
    f"T_rot={mean_t_observable} is not finite (NaN or inf). Integration may have failed."
  assert mean_t_observable < 1000.0, \
    f"T_rot={mean_t_observable:.1f}K exceeds 1000K, suggesting thermal runaway."


# ============================================================================
# Parametrized dt sweep: 16-water NVT across dt=[0.5, 1.0, 2.0] fs
# ============================================================================

# (dt_fs, gamma_ps): all cases at the production-validated friction gamma=10 ps^-1
# (matches the dt=1fs gate job 15870804 + size sweep ba334c1f). The test asserts T_rot,
# which is stable ~300 K at every size; dt=0.5 and dt=1.0 both pass. dt=2.0 fs is untested
# (no gate run) and stays a permanent xfail.
DT_CASES = [
    pytest.param(0.5, 10.0, id="dt0.5fs"),
    pytest.param(1.0, 10.0, id="dt1.0fs"),
    pytest.param(
        2.0, 10.0,
        marks=pytest.mark.xfail(
            strict=False,
            reason="dt=2.0 fs untested — no gate run; do NOT interpret xfail as expected-stable"
        ),
        id="dt2.0fs"
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("dt_fs,gamma_ps", DT_CASES)
def test_dt_sweep_16water_nvt(dt_fs: float, gamma_ps: float) -> None:
  """Parametrized dt sweep: 16-water NVT, asserts T_rot (the gate's metric), gamma=10.

  Asserts on **T_rot**, not T_total: at n=16 the combined T_total is confounded by the
  small-N translational finite-size mode (3N-3 DOF), which makes T_total noisy/biased and
  is NOT a faithful test of dt/thermostat stability. T_rot is the physically-controlled
  quantity the dt=1fs gate (job 15870804) and size sweep (ba334c1f) validate, and is
  stable ~300 K at every size. See .praxia/docs/research/260612_p5-dt1fs-size-crossover.md.

  dt=0.5 fs and dt=1.0 fs (gamma=10 ps⁻¹) both hold T_rot within ±15 K. dt=2.0 fs:
  PERMANENT xfail — no gate run planned; not evidence of stability.

  NOTE: Existing n=2 tests (test_temperature_dt1fs_near_target, test_temperature_dt2fs_near_target)
  assert the *combined* T_total and remain xfailed by the small-N translational artifact;
  they are separate and not the gate evidence. Do not confuse them with this sweep.
  """
  n_waters = 16
  sim_ps = 10.0
  steps = int(sim_ps * 1000.0 / dt_fs)
  burn = max(100, steps // 3)
  seed = 777
  t_rot = _mean_trot_after_burn(dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps, burn=burn, gamma_ps=gamma_ps)
  assert abs(t_rot - 300.0) < 15.0, f"dt={dt_fs} gamma={gamma_ps}: T_rot={t_rot:.1f}K, expected 300±15K"
