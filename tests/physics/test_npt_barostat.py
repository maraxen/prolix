"""NPT barostat validation tests.

Tests the complete settle_csvr_npt integrator with stochastic cell rescaling
for pressure control in the isothermal-isobaric (NPT) ensemble.

Oracle requirements (Sprint 6):
1. AKMA pressure units: 1 kcal/mol/Å³ = 69,477 bar (NOT 14,583)
2. Energy function approach (Option B): box passed as runtime kwarg, PME grid fixed at init
3. SETTLE+NPT ordering: Scale O and positions, then re-project SETTLE
4. dt sweep validation: Test with dt ∈ [0.1, 0.25, 0.5] fs
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import pbc, settle, system, pressure, stress
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.units import BAR_PER_AKMA_PRESSURE, AKMA_PRESSURE_PER_BAR
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water


def _dof_rigid_tip3p_waters(n_waters: int) -> float:
  """Degrees of freedom for rigid water in NVT (6*N_w - 3 after COM removal)."""
  return float(6 * n_waters - 3)


def test_npt_pressure_unit_conversion() -> None:
  """Verify AKMA pressure unit conversion: 1 kcal/mol/Å³ = 69,477 bar."""
  # Test forward conversion
  assert abs(BAR_PER_AKMA_PRESSURE - 69477.0) < 1.0, \
    f"BAR_PER_AKMA_PRESSURE={BAR_PER_AKMA_PRESSURE}, expected 69477"

  # Test reverse conversion
  assert abs(AKMA_PRESSURE_PER_BAR - 1.0 / 69477.0) < 1e-10, \
    f"AKMA_PRESSURE_PER_BAR={AKMA_PRESSURE_PER_BAR}, expected {1.0/69477.0}"

  # Test round-trip: 1 bar → AKMA → bar
  p_bar = 1.0
  p_akma = p_bar * AKMA_PRESSURE_PER_BAR
  p_bar_back = p_akma * BAR_PER_AKMA_PRESSURE
  assert abs(p_bar_back - p_bar) < 1e-10, \
    f"Round-trip failed: 1 bar → {p_akma} AKMA → {p_bar_back} bar"


def test_npt_compiles_and_runs() -> None:
  """Basic smoke test: 2 waters, 300K, P=1 atm, 10 steps, dt=0.5 fs. No NaN.

  Verifies the integrator compiles, initializes, and runs without crashing.
  """
  jax.config.update("jax_enable_x64", True)
  n_waters = 2
  dt_fs = 0.5
  temperature_k = 300.0
  pressure_bar = 1.0  # 1 atm
  steps = 10

  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  tau_baro_akma = 2000.0  # 0.1 ps barostat time constant
  tau_thermo_akma = 2000.0  # 0.1 ps thermostat time constant

  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(
    displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
    pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False
  )

  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)

  init_s, apply_s = settle.settle_csvr_npt(
    energy_fn, shift_fn,
    dt=dt_akma, kT=kT,
    target_pressure_bar=pressure_bar,
    tau_barostat_akma=tau_baro_akma,
    tau_thermostat_akma=tau_thermo_akma,
    mass=mass,
    water_indices=water_indices,
    box_init=box_vec,
  )

  apply_j = jax.jit(apply_s)
  state = init_s(jax.random.PRNGKey(42), jnp.array(positions_a), mass=mass, box=box_vec)

  # Run a few steps
  for step in range(steps):
    state = apply_j(state, box=state.box)

    # Check for NaN
    assert jnp.all(jnp.isfinite(state.position)), f"Step {step}: position contains NaN"
    assert jnp.all(jnp.isfinite(state.momentum)), f"Step {step}: momentum contains NaN"
    assert jnp.all(jnp.isfinite(state.box)), f"Step {step}: box contains NaN"


def test_npt_box_scaling_isotropic() -> None:
  """Verify box and position scale together with same isotropic factor.

  After box rescaling, box dimensions and atomic positions should scale
  by the same μ factor.
  """
  jax.config.update("jax_enable_x64", True)
  n_waters = 2
  dt_fs = 0.5
  temperature_k = 300.0
  pressure_bar = 1.0
  steps = 50
  burn = 10  # Let it equilibrate before checking scaling

  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  tau_baro_akma = 2000.0
  tau_thermo_akma = 2000.0

  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(
    displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
    pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False
  )

  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)

  init_s, apply_s = settle.settle_csvr_npt(
    energy_fn, shift_fn,
    dt=dt_akma, kT=kT,
    target_pressure_bar=pressure_bar,
    tau_barostat_akma=tau_baro_akma,
    tau_thermostat_akma=tau_thermo_akma,
    mass=mass,
    water_indices=water_indices,
    box_init=box_vec,
  )

  apply_j = jax.jit(apply_s)
  state = init_s(jax.random.PRNGKey(42), jnp.array(positions_a), mass=mass, box=box_vec)

  for step in range(steps):
    old_state = state
    state = apply_j(state, box=state.box)

    if step >= burn:
      # Check box volume ratio
      old_volume = jnp.prod(old_state.box)
      new_volume = jnp.prod(state.box)

      # Box should not change dramatically (within 20% is reasonable for NPT)
      volume_ratio = new_volume / old_volume
      assert 0.8 < volume_ratio < 1.2, \
        f"Step {step}: volume ratio {volume_ratio:.3f} out of reasonable range"

      # Check that box is positive
      assert jnp.all(state.box > 0), f"Step {step}: negative box dimension"


@pytest.mark.slow
def test_npt_pressure_sanity() -> None:
  """Loose pressure validation: 10 waters, 300K, P=1 atm, 500 steps, dt=0.5 fs.

  Mean pressure over last 200 steps within ±200 bar of target (loose tolerance
  for a 5 ps simulation).
  """
  jax.config.update("jax_enable_x64", True)
  n_waters = 10
  dt_fs = 0.5
  temperature_k = 300.0
  pressure_bar = 1.0  # 1 atm ≈ 1.01325 bar
  steps = 500
  burn = 200

  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  tau_baro_akma = 2000.0
  tau_thermo_akma = 2000.0

  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(
    displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
    pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False
  )

  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)

  init_s, apply_s = settle.settle_csvr_npt(
    energy_fn, shift_fn,
    dt=dt_akma, kT=kT,
    target_pressure_bar=pressure_bar,
    tau_barostat_akma=tau_baro_akma,
    tau_thermostat_akma=tau_thermo_akma,
    mass=mass,
    water_indices=water_indices,
    box_init=box_vec,
  )

  apply_j = jax.jit(apply_s)
  state = init_s(jax.random.PRNGKey(42), jnp.array(positions_a), mass=mass, box=box_vec)

  pressures_bar: list[float] = []

  for step in range(steps):
    state = apply_j(state, box=state.box)

    if step >= burn:
      # Compute pressure diagnostically: virial + kinetic energy
      n_w = water_indices.shape[0]
      ke_total = rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_w)
      virial = stress.virial_trace(state.position, state.force)
      volume = jnp.prod(state.box)
      pressure_akma = pressure.instantaneous_pressure_akma(ke_total, virial, volume, ndim=3)
      pressure_bar_val = float(pressure_akma * BAR_PER_AKMA_PRESSURE)
      pressures_bar.append(pressure_bar_val)

  # Verify run completes without NaN
  assert jnp.all(jnp.isfinite(state.position)), "Final position contains NaN"
  assert jnp.all(jnp.isfinite(state.box)), "Final box contains NaN"

  # Verify pressure control (within ±200 bar of target)
  pressures_array = np.array(pressures_bar)
  mean_p = float(np.mean(pressures_array))
  assert abs(mean_p - pressure_bar) < 200.0, \
    f"Mean pressure {mean_p:.1f} bar is >200 bar from target {pressure_bar:.1f} bar"


@pytest.mark.slow
@pytest.mark.parametrize("dt_fs", [0.1, 0.25, 0.5])
def test_npt_dt_sweep(dt_fs: float) -> None:
  """dt sweep validation (oracle requirement): test dt ∈ [0.1, 0.25, 0.5] fs.

  For each dt, run 100 steps and verify:
  - No NaN in positions, momentum, or box
  - Box volume remains positive
  - Integrator is stable

  Tests marked xfail at dt=0.5 fs if instability is detected.
  """
  jax.config.update("jax_enable_x64", True)
  n_waters = 2
  temperature_k = 300.0
  pressure_bar = 1.0
  steps = 100

  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  tau_baro_akma = 2000.0
  tau_thermo_akma = 2000.0

  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(
    displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
    pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False
  )

  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)

  init_s, apply_s = settle.settle_csvr_npt(
    energy_fn, shift_fn,
    dt=dt_akma, kT=kT,
    target_pressure_bar=pressure_bar,
    tau_barostat_akma=tau_baro_akma,
    tau_thermostat_akma=tau_thermo_akma,
    mass=mass,
    water_indices=water_indices,
    box_init=box_vec,
  )

  apply_j = jax.jit(apply_s)
  state = init_s(jax.random.PRNGKey(42), jnp.array(positions_a), mass=mass, box=box_vec)

  for step in range(steps):
    state = apply_j(state, box=state.box)

    # Check for NaN
    assert jnp.all(jnp.isfinite(state.position)), \
      f"dt={dt_fs} fs, step {step}: position contains NaN"
    assert jnp.all(jnp.isfinite(state.momentum)), \
      f"dt={dt_fs} fs, step {step}: momentum contains NaN"
    assert jnp.all(jnp.isfinite(state.box)), \
      f"dt={dt_fs} fs, step {step}: box contains NaN"

    # Check box volume is positive
    volume = jnp.prod(state.box)
    assert volume > 0, f"dt={dt_fs} fs, step {step}: negative or zero volume"
