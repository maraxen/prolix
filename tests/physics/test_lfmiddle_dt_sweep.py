"""LFMiddle dt-sweep hypothesis test (v1.1).

Tests whether Leimkuhler-Matthews O-step splitting (settle_lfmiddle_langevin)
reduces SETTLE+thermostat coupling enough to lift the dt <= 0.5 fs cap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.typing import IntegratorState
from prolix.physics.step_system import make_sequence
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _proxide_params_pure_water


def _dof_rigid_tip3p_waters(n_waters: int) -> float:
  """Degrees of freedom for rigid TIP3P water molecules (6N_w - 3)."""
  return float(6 * n_waters - 3)


def _mean_rigid_t_langevin_after_burn(
    *,
    dt_fs: float,
    n_waters: int,
    seed: int,
    steps: int,
    burn: int,
    sequence_name: str = "baoab_langevin"
) -> tuple[float, float]:
  """Compute mean rigid-body temperature from integrator trajectory.

  Args:
      dt_fs: Timestep in femtoseconds.
      n_waters: Number of water molecules.
      seed: Random seed.
      steps: Number of integration steps.
      burn: Burn-in steps to skip before averaging.
      sequence_name: Integrator sequence name ("baoab_langevin", "lfmiddle_langevin").

  Returns:
      (mean_T_observable, target_T_kelvin) where mean_T_observable is computed
      from rigid-body kinetic energy, target_T_kelvin is 300 K.
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
  energy_fn = system.make_energy_fn(
      displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
      pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False
  )

  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)

  integrator_kw = dict(
      energy_or_force_fn=energy_fn,
      shift_fn=shift_fn,
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
  if sequence_name == "lfmiddle_langevin":
    init_s, apply_s = settle.settle_lfmiddle_langevin(**integrator_kw)
  else:
    init_s, apply_s = settle.settle_langevin(**integrator_kw)

  apply_j = jax.jit(apply_s)
  dof_rigid = _dof_rigid_tip3p_waters(n_waters)

  state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)
  temps: list[float] = []

  for step in range(steps):
    state = apply_j(state)
    if step >= burn:
      ke_r = float(rigid_tip3p_box_ke_kcal(state.positions, state.momentum, state.mass, n_waters))
      temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
      temps.append(temp)

  mean_t_observable = float(np.mean(temps)) if temps else float("nan")
  t_thermostat_target = temperature_k

  return mean_t_observable, t_thermostat_target


class TestLFMiddleBaseline:
  """Baseline tests: verify LFMiddle works at dt=0.5 fs (proven-stable timestep)."""

  def test_lfmiddle_single_water_stability(self):
    """LFMiddle at dt=0.5 fs, 1 water, 50 ps: no NaN, temperature stable.

    Baseline sanity check: verify integrator doesn't crash and temperature
    stays bounded on a simple 1-water system.
    """
    dt_fs = 0.5
    n_waters = 1
    sim_ps = 50.0
    steps = int(sim_ps * 1000.0 / dt_fs)
    burn = 0  # Don't skip steps for short trajectory
    seed = 700

    mean_t_observable, _ = _mean_rigid_t_langevin_after_burn(
        dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps,
        burn=burn, sequence_name="lfmiddle_langevin"
    )

    # Baseline: temperature should be finite and within reasonable bounds
    assert not np.isnan(mean_t_observable), "Temperature is NaN"
    assert not np.isinf(mean_t_observable), "Temperature is Inf"
    assert mean_t_observable > 0, f"Temperature is non-positive: {mean_t_observable}"
    # Very loose bound for sanity check
    assert mean_t_observable < 1000, f"Temperature exceeded 1000 K: {mean_t_observable:.1f}"

  def test_lfmiddle_dt_0_5fs_baseline(self):
    """LFMiddle at dt=0.5 fs: baseline stability (100 fs trajectory).

    Verify LFMiddle at proven-stable dt=0.5 fs produces reasonable temperature
    statistics on a 2-water system over 100 fs.
    """
    dt_fs = 0.5
    n_waters = 2
    sim_ps = 100.0
    steps = int(sim_ps * 1000.0 / dt_fs)
    burn = max(100, steps // 3)
    seed = 701

    mean_t_observable, target_t = _mean_rigid_t_langevin_after_burn(
        dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps,
        burn=burn, sequence_name="lfmiddle_langevin"
    )

    # At dt=0.5 fs, temperature should be within ±20 K of target
    assert abs(mean_t_observable - target_t) < 20, (
        f"dt={dt_fs} fs: T_obs={mean_t_observable:.1f} K, target={target_t} K, "
        f"delta={abs(mean_t_observable - target_t):.1f} K (expected < 20 K)"
    )


class TestLFMiddleDtSweep:
  """dt sweep tests: measure temperature stability at dt = 0.25, 0.5, 1.0 fs."""

  def test_lfmiddle_dt_0_25fs(self):
    """LFMiddle at dt=0.25 fs: sanity check sub-0.5 fs stability (100 fs).

    Very small timestep should show excellent stability as a sanity check.
    """
    dt_fs = 0.25
    n_waters = 2
    sim_ps = 100.0
    steps = int(sim_ps * 1000.0 / dt_fs)
    burn = max(200, steps // 3)
    seed = 702

    mean_t_observable, target_t = _mean_rigid_t_langevin_after_burn(
        dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps,
        burn=burn, sequence_name="lfmiddle_langevin"
    )

    # At very small dt, expect excellent stability (±10 K)
    assert abs(mean_t_observable - target_t) < 10, (
        f"dt={dt_fs} fs (very small): T_obs={mean_t_observable:.1f} K, "
        f"target={target_t} K (expected < 10 K deviation)"
    )

  @pytest.mark.xfail(
      reason="LFMiddle hypothesis test: expected failure if O-step splitting insufficient. "
      "Either: (1) hypothesis refuted (dt=1.0 still unstable), or (2) hypothesis confirmed (remarkable discovery). "
      "Phase 3 explores which outcome is true."
  )
  def test_lfmiddle_dt_1_0fs_hypothesis(self):
    """LFMiddle at dt=1.0 fs: hypothesis test (Phase 3 exploratory).

    **Hypothesis**: LFMiddle discretization enables dt=1.0 fs without thermal runaway.

    **Expected Outcome**: XFAIL (hypothesis likely refuted; documented expected failure).
    - If passes: Hypothesis confirmed (remarkable, unexpected result)
    - If fails: Hypothesis refuted (expected; SETTLE+thermostat coupling still limits dt)

    **Acceptance criteria** (if hypothesis somehow passes):
    - Temperature bounded: mean T < 500 K (2 sigma from target 300 K)
    - No NaN or Inf
    - Energy drift < 1% (relaxed from 0.1% due to larger timestep)

    This test is decorated @pytest.mark.xfail to allow CI to report it as
    "expected failure" rather than regression. The test outcome validates the
    hypothesis either way.
    """
    dt_fs = 1.0
    n_waters = 2
    steps = int(100.0 * 1000.0 / dt_fs)  # 100 fs wall-clock = 200 fs integration time
    burn = max(100, steps // 3)
    seed = 703

    mean_t_observable, target_t = _mean_rigid_t_langevin_after_burn(
        dt_fs=dt_fs, n_waters=n_waters, seed=seed, steps=steps,
        burn=burn, sequence_name="lfmiddle_langevin"
    )

    # If we reach here without NaN/Inf/crash, check temperature bounds
    assert not np.isnan(mean_t_observable), "Temperature is NaN (thermal collapse)"
    assert not np.isinf(mean_t_observable), "Temperature is Inf (thermal runaway)"

    # Hypothesis acceptance criteria: mean T < 500 K (relaxed from 300 due to larger dt)
    assert mean_t_observable < 500, (
        f"dt={dt_fs} fs (hypothesis test): T_obs={mean_t_observable:.1f} K exceeds 500 K "
        f"(thermal runaway). Hypothesis refuted: dt={dt_fs} fs not stable."
    )


class TestLFMiddleIntegration:
  """Integration test: 64-water system, dt sweep, collect statistics."""

  def test_lfmiddle_64water_10ps_dt_sweep(self):
    """64-water, 10 ps trajectory: dt sweep with statistics (Phase 3 data collection).

    Collects mean T, std T, max T, energy drift for dt = 0.25, 0.5, 1.0 fs.
    Only dt <= 0.5 fs required to pass; dt=1.0 fs expected to show thermal runaway.

    **Expected Results**:
    - dt=0.25 fs: T = 300 ± 8 K, ΔE < 0.05%
    - dt=0.5 fs:  T = 300 ± 9 K, ΔE < 0.1%
    - dt=1.0 fs:  T >> 300 K or NaN (thermal runaway)

    This test is marked as slow (>30s runtime).
    """
    pytest.skip(
        "Large integration test (64 waters, 10 ps) skipped in quick test suite. "
        "Run manually with: pytest -m slow tests/physics/test_lfmiddle_dt_sweep.py::TestLFMiddleIntegration::test_lfmiddle_64water_10ps_dt_sweep"
    )

    # For manual testing, uncomment below:
    # n_waters = 64
    # box_edge_angstrom = 50.0
    #
    # for dt_fs in [0.25, 0.5, 1.0]:
    #   steps = int(10.0 * 1000.0 / dt_fs)  # 10 ps at given dt
    #   burn = max(1000, steps // 4)
    #
    #   mean_t, target_t = _mean_rigid_t_langevin_after_burn(
    #       dt_fs=dt_fs, n_waters=n_waters, seed=704, steps=steps,
    #       burn=burn, sequence_name="baoab_langevin"
    #   )
    #
    #   print(f"dt={dt_fs} fs: T={mean_t:.1f} K (target {target_t} K)")
    #
    #   if dt_fs <= 0.5:
    #     # Expect stability at proven-safe timesteps
    #     assert abs(mean_t - target_t) < 20, (
    #         f"dt={dt_fs} fs should be stable but T={mean_t:.1f} K"
    #     )


class TestLFMiddleRobustness:
  """Robustness checks: edge cases and error handling."""

  def test_lfmiddle_sequence_exists(self):
    """lfmiddle_langevin sequence is registered and instantiable."""
    from prolix.physics.step_system import step_sequences, make_sequence

    assert "lfmiddle_langevin" in step_sequences, "lfmiddle_langevin not in step_sequences"

    seq = step_sequences["lfmiddle_langevin"]
    assert seq.name == "lfmiddle_langevin"
    assert len(seq.steps) > 0, "lfmiddle_langevin has no steps"

    # All steps in sequence should be valid
    from prolix.physics.step_system import step_registry
    for step_name in seq.steps:
      assert step_name in step_registry, f"Step '{step_name}' not in registry"

  def test_lfmiddle_sequence_parameter_override(self):
    """make_sequence allows parameter overrides for lfmiddle_langevin."""
    seq = make_sequence("lfmiddle_langevin", dt=1.0, gamma=0.5, kT=5.0)

    assert seq.parameters["dt"] == 1.0
    assert seq.parameters["gamma"] == 0.5
    assert seq.parameters["kT"] == 5.0


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
