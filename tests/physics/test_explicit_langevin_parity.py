"""P2a baseline: Langevin distribution-level validation on the n=4 explicit PME mock box.

Validates ``jax_md.simulate.nvt_langevin`` against Prolix ``make_energy_fn`` using
ensemble statistics (window-mean temperature), not bitwise trajectories.

OpenMM Langevin reference trajectories are intentionally **not** asserted on this
fixture: empirically, ``LangevinMiddleIntegrator`` does not yield a stable 300 K
ensemble at timesteps compatible with JAX-MD stability on this pathological PME
mock system. See ``docs/source/explicit_solvent/l2_dynamics_protocol.md`` and
``scripts/benchmarks/openmm_langevin_temperature_stats.py`` for OpenMM-only stats on
a different tiny cell.

Tolerances and rationale: ``docs/source/explicit_solvent/l2_dynamics_protocol.md``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import quantity, simulate as jmd_simulate

from prolix.physics import pbc, system
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL


def _mock_periodic(n: int, charges: list[float]) -> dict:
  return {
    "charges": jnp.array(charges, dtype=jnp.float64),
    "sigmas": jnp.ones(n, dtype=jnp.float64),
    "epsilons": jnp.zeros(n, dtype=jnp.float64),
    "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
    "bond_params": jnp.zeros((0, 2), dtype=jnp.float64),
    "angles": jnp.zeros((0, 3), dtype=jnp.int32),
    "angle_params": jnp.zeros((0, 2), dtype=jnp.float64),
    "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float64),
    "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
    "improper_params": jnp.zeros((0, 3), dtype=jnp.float64),
    "exclusion_mask": jnp.ones((n, n), dtype=jnp.float64) - jnp.eye(n, dtype=jnp.float64),
  }


def _output_dir(tmp_path: Path) -> Path:
  ws = os.environ.get("GITHUB_WORKSPACE")
  if ws:
    p = Path(ws) / "artifacts" / "l2_parity"
    p.mkdir(parents=True, exist_ok=True)
    return p
  p = tmp_path / "l2_parity"
  p.mkdir(parents=True, exist_ok=True)
  return p


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
  if not rows:
    return
  fields = list(rows[0].keys())
  with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)


@pytest.mark.slow
def test_explicit_langevin_baseline_parity_csv(tmp_path, regression_pme_params):
  """Baseline (N=4): stable AKMA timestep with PME + ``nvt_langevin`` (matches slow NVT test scale).

  **Timestep note:** ``jax_md.simulate.nvt_langevin`` on this explicit PME mock charge system is
  only stable at the same ~``1e-3`` AKMA step used in ``test_explicit_pbc_nvt_mean_temperature_targets_spec``
  (≈0.049 fs). Larger steps (e.g. 20 fs) blow up the stochastic dynamics.
  """
  jax.config.update("jax_enable_x64", True)

  temperature_k = 300.0
  n = 4
  dof = float(3 * n)
  box_size = 45.0
  charges = [1.0, -1.0, 0.2, -0.2]
  positions = jnp.array(
    [
      [15.0, 15.0, 15.0],
      [32.0, 15.0, 15.0],
      [15.0, 32.0, 15.0],
      [32.0, 32.0, 15.0],
    ],
    dtype=jnp.float64,
  )

  alpha = float(regression_pme_params["pme_alpha_per_angstrom"])
  grid = int(regression_pme_params["pme_grid_points"])
  cutoff = float(regression_pme_params["cutoff_angstrom"])

  dt_akma = 1e-3
  dt_fs = float(dt_akma * AKMA_TIME_UNIT_FS)
  n_steps = 3000
  burn = 1000

  box = jnp.array([box_size, box_size, box_size], dtype=jnp.float64)
  sys_dict = _mock_periodic(n, charges)
  displacement_fn, shift_fn = pbc.create_periodic_space(box)
  energy_fn = system.make_energy_fn(
    displacement_fn,
    sys_dict,
    box=box,
    use_pbc=True,
    implicit_solvent=False,
    pme_grid_points=grid,
    pme_alpha=alpha,
    cutoff_distance=cutoff,
    strict_parameterization=False,
  )

  kT = temperature_k * BOLTZMANN_KCAL
  gamma_reduced = 1.0 * AKMA_TIME_UNIT_FS * 1e-3

  init_fn, apply_fn = jmd_simulate.nvt_langevin(
    energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced
  )
  key = jax.random.PRNGKey(123)
  mass = jnp.ones(n, dtype=jnp.float64) * 12.0

  @jax.jit
  def _prolix_chain(k: jax.Array) -> tuple[jax.Array, jax.Array]:
    state = init_fn(k, positions, mass=mass)

    def _body(carry, _i):
      s = carry
      ns = apply_fn(s)
      ke = quantity.kinetic_energy(momentum=ns.momentum, mass=ns.mass)
      pe = energy_fn(ns.position)
      return ns, jnp.stack([ke, pe])

    _final, out = jax.lax.scan(_body, state, None, length=n_steps)
    kes = out[:, 0]
    pes = out[:, 1]
    return kes, pes

  kes, pes = _prolix_chain(key)
  kes_np = np.asarray(kes)
  pes_np = np.asarray(pes)
  t_inst_np = 2.0 * kes_np / (dof * BOLTZMANN_KCAL)

  prolix_rows: list[dict[str, float | int]] = []
  for step in range(n_steps):
    ke = float(kes_np[step])
    pe = float(pes_np[step])
    t_inst = float(t_inst_np[step])
    time_ps = (step + 1) * dt_fs / 1000.0
    prolix_rows.append(
      {
        "step": step + 1,
        "time_ps": time_ps,
        "T_inst": t_inst,
        "K_kcalmol": ke,
        "U_kcalmol": pe,
        "Etot_kcalmol": ke + pe,
      }
    )

  t_samples = t_inst_np[burn:].astype(np.float64)
  mean_t_prolix = float(np.mean(t_samples))
  assert abs(mean_t_prolix - temperature_k) <= 0.10 * temperature_k, (
    f"Prolix mean T={mean_t_prolix:.2f} K vs target {temperature_k} K (baseline ±10% gate)"
  )
  assert float(np.std(t_samples)) > 1e-6

  init_nve, apply_nve = jmd_simulate.nve(energy_fn, shift_fn=shift_fn, dt=dt_akma)
  key_nve = jax.random.PRNGKey(456)
  st0 = init_nve(key_nve, positions, mass=mass, kT=kT)
  e0 = float(energy_fn(st0.position) + quantity.kinetic_energy(momentum=st0.momentum, mass=st0.mass))
  st = st0
  nve_steps = 800
  for _ in range(nve_steps):
    st = apply_nve(st)
  e1 = float(energy_fn(st.position) + quantity.kinetic_energy(momentum=st.momentum, mass=st.mass))
  dt_ps_nve = nve_steps * dt_fs / 1000.0
  rel = abs(e1 - e0) / max(abs(e0), 1e-12)
  assert rel <= 0.01 * max(dt_ps_nve, 1e-6), (
    f"NVE relative energy drift {rel:.3e} exceeds ~1%/ps × {dt_ps_nve:.3f} ps budget"
  )

  out_dir = _output_dir(tmp_path)
  _write_csv(out_dir / "prolix_langevin_parity.csv", prolix_rows)
