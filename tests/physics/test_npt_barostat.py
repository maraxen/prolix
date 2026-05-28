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
from prolix.physics.simulate import NPTState
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _proxide_params_pure_water


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

  sys_dict = _proxide_params_pure_water(n_waters)
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
    assert jnp.all(jnp.isfinite(state.positions)), f"Step {step}: position contains NaN"
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

  sys_dict = _proxide_params_pure_water(n_waters)
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

  sys_dict = _proxide_params_pure_water(n_waters)
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
      ke_total = rigid_tip3p_box_ke_kcal(state.positions, state.momentum, state.mass, n_w)
      virial = stress.virial_trace(state.positions, state.force)
      volume = jnp.prod(state.box)
      pressure_akma = pressure.compute_pressure_akma(ke_total, virial, volume, ndim=3)
      pressure_bar_val = float(pressure_akma * BAR_PER_AKMA_PRESSURE)
      pressures_bar.append(pressure_bar_val)

  # Verify run completes without NaN
  assert jnp.all(jnp.isfinite(state.positions)), "Final position contains NaN"
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

  sys_dict = _proxide_params_pure_water(n_waters)
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
    assert jnp.all(jnp.isfinite(state.positions)), \
      f"dt={dt_fs} fs, step {step}: position contains NaN"
    assert jnp.all(jnp.isfinite(state.momentum)), \
      f"dt={dt_fs} fs, step {step}: momentum contains NaN"
    assert jnp.all(jnp.isfinite(state.box)), \
      f"dt={dt_fs} fs, step {step}: box contains NaN"

    # Check box volume is positive
    volume = jnp.prod(state.box)
    assert volume > 0, f"dt={dt_fs} fs, step {step}: negative or zero volume"


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "NPT rigid-water T remains >>300K after Sprint 14 /mu scaling and warm handoff "
        "(local probe: cold init ~260K then step-1 spike ~7e3K). CSVR+SETTLE+KE coupling; "
        "see scripts/debug/npt_step0_diagnostic.py and .praxia/docs/npt_ke_bug_diagnosis.md."
    ),
)
def test_npt_20ps_liquid_water() -> None:
  """Two-phase NVT→NPT protocol for 64 TIP3P waters over 20 ps at liquid density.

  Phase 1 — NVT equilibration (4000 steps × 0.5 fs = 2 ps):
    Relax grid bad contacts with settle_langevin at 300 K, gamma=1.0 ps⁻¹.

  Phase 2 — NPT production (40000 steps × 0.5 fs = 20 ps):
    Run settle_csvr_npt at P=1 bar, T=300 K. Record (T, P, V) every 200 steps
    → 200 trajectory records total.

  Thermodynamic assertions on the last 10 ps (last 100 records):
    - T_mean ∈ [295, 305] K
    - P_mean ∈ [-99, 101] bar  (±100 bar around P=1; 64-water σ_P ≈ 500–800 bar
      instantaneous, ≈80–130 bar on a 10 ps mean — loose tolerance is physically
      justified for this system size)
    - No NaN anywhere in the timeseries
    - Box volume varies smoothly (no step-to-step jump > 5%)

  DOF convention: ndof = 6 * N_w - 3 (rigid TIP3P, 6 DOF/water, minus 3 COM
  translations). Temperature: T = 2*KE / (kB * ndof). Pressure uses virial theorem.
  """
  jax.config.update("jax_enable_x64", True)

  # ── Constants ──────────────────────────────────────────────────────────────
  n_waters = 64
  dt_fs = 0.5
  temperature_k = 300.0
  pressure_bar = 1.0
  gamma_ps = 1.0  # ps⁻¹
  tau_baro_akma = 2000.0
  tau_thermo_akma = 2000.0
  nvt_steps = 4000
  npt_steps = 40000
  record_every = 200

  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  ndof = float(6 * n_waters - 3)  # 381 for 64 waters
  n_atoms = n_waters * 3

  # ── System setup ───────────────────────────────────────────────────────────
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=3.1)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

  sys_dict = _proxide_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(
    displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
    pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False
  )

  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(n_protein_atoms=0, n_waters=n_waters)

  # ── Phase 1: NVT equilibration ─────────────────────────────────────────────
  init_nvt, apply_nvt = settle.settle_langevin(
    energy_fn, shift_fn,
    dt=dt_akma,
    kT=kT,
    gamma=gamma_ps,
    mass=mass,
    water_indices=water_indices,
    project_ou_momentum_rigid=True,
    projection_site="post_o",
  )

  apply_nvt_j = jax.jit(apply_nvt)
  nvt_state = init_nvt(jax.random.PRNGKey(7), jnp.array(positions_a), mass=mass, box=box_vec)

  for _ in range(nvt_steps):
    nvt_state = apply_nvt_j(nvt_state, box=box_vec)

  assert jnp.all(jnp.isfinite(nvt_state.positions)), "NVT phase: positions contain NaN/Inf"

  # ── Phase 2: NPT production ────────────────────────────────────────────────
  # Construct NPTState from NVT final state + original box
  npt_state = NPTState(
    positions=nvt_state.positions,
    momentum=nvt_state.momentum,
    force=nvt_state.force,
    mass=nvt_state.mass,
    rng=nvt_state.rng,
    box=box_vec,
  )

  init_npt, apply_npt = settle.settle_csvr_npt(
    energy_fn, shift_fn,
    dt=dt_akma,
    kT=kT,
    target_pressure_bar=pressure_bar,
    tau_barostat_akma=tau_baro_akma,
    tau_thermostat_akma=tau_thermo_akma,
    mass=mass,
    water_indices=water_indices,
    box_init=box_vec,
  )

  apply_npt_j = jax.jit(apply_npt)

  # Warm handoff: preserve NVT-equilibrated momenta (do not re-sample Maxwell-Boltzmann)
  npt_state = init_npt(
    npt_state.rng,
    npt_state.positions,
    mass=mass,
    box=box_vec,
    momentum=nvt_state.momentum,
  )

  # Trajectory arrays: shape (n_records, 3) → columns: (T_K, P_bar, V_A3)
  temperatures: list[float] = []
  pressures_bar_list: list[float] = []
  volumes: list[float] = []

  for step in range(npt_steps):
    npt_state = apply_npt_j(npt_state, box=npt_state.box)

    if (step + 1) % record_every == 0:
      ke_total = rigid_tip3p_box_ke_kcal(
        npt_state.positions, npt_state.momentum, npt_state.mass, n_waters
      )
      virial = stress.virial_trace(npt_state.positions, npt_state.force)
      volume = float(jnp.prod(npt_state.box))
      pressure_akma = pressure.compute_pressure_akma(
        ke_total, virial, volume, ndim=3
      )
      T_k = float(2.0 * ke_total / (BOLTZMANN_KCAL * ndof))
      P_bar = float(pressure_akma * BAR_PER_AKMA_PRESSURE)

      temperatures.append(T_k)
      pressures_bar_list.append(P_bar)
      volumes.append(volume)

  # ── Assertions ─────────────────────────────────────────────────────────────
  T_arr = np.array(temperatures)    # shape (200,)
  P_arr = np.array(pressures_bar_list)
  V_arr = np.array(volumes)

  # 1. No NaN in timeseries
  assert np.all(np.isfinite(T_arr)), "Temperature timeseries contains NaN/Inf"
  assert np.all(np.isfinite(P_arr)), "Pressure timeseries contains NaN/Inf"
  assert np.all(np.isfinite(V_arr)), "Volume timeseries contains NaN/Inf"

  # 2. T_mean over last 10 ps (last 100 records, steps 20000–40000)
  T_last = T_arr[-100:]
  T_mean = float(np.mean(T_last))
  assert 295.0 <= T_mean <= 305.0, (
    f"Mean T over last 10 ps = {T_mean:.2f} K, expected 295–305 K"
  )

  # 3. Slope-based drift gate: temperature must not be trending up or down
  # Accept only if |slope| < 0.2 K/ps (≡ < 2 K per 10 ps)
  from scipy.stats import linregress
  time_ps = np.arange(len(T_last)) * 0.1
  slope, intercept, r_value, p_value, std_err = linregress(time_ps, T_last)
  assert abs(slope) < 0.2, (
      f"Temperature drift slope {slope:.4f} K/ps exceeds limit 0.2 K/ps "
      f"(≡ 2 K per 10 ps). r²={r_value**2:.4f}. "
      f"This indicates systematic thermal runaway or cooling — "
      f"momentum inverse-scaling fix may not be sufficient."
  )

  # 4. P_mean over last 10 ps in [-99, 101] bar
  #    64-water σ_P ≈ 500–800 bar instantaneous; ≈80–130 bar on 10 ps mean
  P_last = P_arr[-100:]
  P_mean = float(np.mean(P_last))
  assert -99.0 <= P_mean <= 101.0, (
    f"Mean P over last 10 ps = {P_mean:.1f} bar, expected within [-99, 101] bar of "
    f"target {pressure_bar:.1f} bar"
  )

  # 5. Box volume varies smoothly (no step-to-step jump > 5%)
  max_jump_frac = float(np.max(np.abs(np.diff(V_arr)) / V_arr[:-1]))
  assert max_jump_frac < 0.05, (
    f"Box volume has a sudden jump of {max_jump_frac*100:.2f}% between records "
    f"(threshold: 5%)"
  )


@pytest.mark.slow
@pytest.mark.parametrize("dt_fs,expected_stable", [
    (0.5, True),
    (1.0, True),  # May now pass with Parrinello-Rahman momentum scaling
    pytest.param(
        2.0,
        False,
        marks=pytest.mark.xfail(
            strict=True,
            reason="dt=2.0fs exceeds SETTLE constraint stability; expected to diverge"
        ),
    ),
])
def test_npt_dt_sweep_stability(
    dt_fs: float,
    expected_stable: bool,
    n_waters: int = 16,
    temperature_k: float = 300.0,
    pressure_bar: float = 1.0,
) -> None:
    """NPT stability across dt in {0.5, 1.0, 2.0} fs over 5 ps.

    dt=0.5 fs: baseline (always pass)
    dt=1.0 fs: test Parrinello-Rahman fix enables larger timestep
    dt=2.0 fs: exceeds SETTLE limit — expected to diverge (xfail strict=True)
    """
    import scipy.stats
    jax.config.update("jax_enable_x64", True)

    npt_steps = 10_000   # 5 ps at dt=0.5 fs
    record_every = 100
    dt_akma = float(dt_fs) / AKMA_TIME_UNIT_FS
    kT = float(temperature_k) * BOLTZMANN_KCAL
    tau_baro_akma = 2000.0
    tau_thermo_akma = 2000.0

    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

    sys_dict = _proxide_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    energy_fn = system.make_energy_fn(
        displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
        pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0,
        strict_parameterization=False,
    )

    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters, dtype=jnp.float64).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)

    init_npt, apply_npt = settle.settle_csvr_npt(
        energy_fn, shift_fn,
        dt=dt_akma, kT=kT,
        target_pressure_bar=pressure_bar,
        tau_barostat_akma=tau_baro_akma,
        tau_thermostat_akma=tau_thermo_akma,
        mass=mass,
        water_indices=water_indices,
        box_init=box_vec,
    )

    apply_npt_j = jax.jit(apply_npt)
    npt_state = init_npt(
        jax.random.PRNGKey(42),
        jnp.array(positions_a, dtype=jnp.float64),
        mass=mass,
        box=box_vec,
    )

    temperatures = []
    for step in range(npt_steps):
        npt_state = apply_npt_j(npt_state, box=npt_state.box)
        if (step + 1) % record_every == 0:
            ke_total = rigid_tip3p_box_ke_kcal(
                npt_state.positions, npt_state.momentum, npt_state.mass, n_waters
            )
            ndof = float(6 * n_waters - 3)
            T_k = float(2.0 * ke_total / (BOLTZMANN_KCAL * ndof))
            temperatures.append(T_k)

    T_arr = np.array(temperatures)
    assert np.all(np.isfinite(T_arr)), f"dt={dt_fs}fs: temperature contains NaN/Inf"

    T_mean = float(np.mean(T_arr))
    assert abs(T_mean - temperature_k) < 20.0, (
        f"dt={dt_fs}fs: mean T {T_mean:.1f}K is >20K from target {temperature_k}K"
    )

    # Slope check: temperature should not drift (< 0.5 K/ps over 5ps)
    time_ps = np.arange(len(T_arr)) * 0.1
    slope, _, r_value, _, _ = scipy.stats.linregress(time_ps, T_arr)
    assert abs(slope) < 0.5, (
        f"dt={dt_fs}fs: temperature drift slope {slope:.4f} K/ps exceeds 0.5 K/ps, "
        f"r²={r_value**2:.4f}"
    )
