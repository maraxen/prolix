"""Explicit solvent: OpenMM Reference vs Prolix ``make_energy_fn`` (solvated-style systems).

Requires optional dependency ``openmm`` (``pytest -m openmm`` / ``pip install -e '.[dev,openmm]'``).

Layer A extension: two TIP3P waters (6 sites) with intramolecular Coulomb exclusions
mirrored in OpenMM via ``NonbondedForce.addException``. Compares total potential energy
and force RMSE to the anchor tolerances in ``test_openmm_explicit_anchor``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

try:
  import openmm
  from openmm import unit

  HAS_OPENMM = True
except ImportError:
  HAS_OPENMM = False

from prolix.physics import pbc, system
from prolix.physics.water_models import WaterModelType, get_water_params


def _tip3p_two_waters_positions() -> np.ndarray:
  """Two TIP3P waters in a periodic box (Å), separated to avoid minimum-image ambiguity."""
  tip = get_water_params(WaterModelType.TIP3P)
  r_oh = float(tip.r_OH)
  r_hh = float(tip.r_hh)
  # Water 1: O at origin-ish; H2O in xy plane
  w1_o = np.array([5.0, 5.0, 5.0])
  w1_h1 = w1_o + np.array([r_oh, 0.0, 0.0])
  w1_h2 = w1_o + np.array([r_oh * np.cos(1.911), r_oh * np.sin(1.911), 0.0])
  # Adjust H2 to match H–H distance (simplified placement)
  w1_h2 = w1_o + np.array([0.585, 0.757, 0.0])  # ~TIP3P geometry
  # Water 2: translated
  shift = np.array([12.0, 0.0, 0.0])
  w2_o = w1_o + shift
  w2_h1 = w1_h1 + shift
  w2_h2 = w1_h2 + shift
  return np.array([w1_o, w1_h1, w1_h2, w2_o, w2_h1, w2_h2], dtype=np.float64)


def _prolix_params_two_waters() -> dict:
  tip = get_water_params(WaterModelType.TIP3P)
  q_o = float(tip.charge_O)
  q_h = float(tip.charge_H)
  sig_o = float(tip.sigma_O)
  eps_o = float(tip.epsilon_O)
  n = 6
  charges = jnp.array([q_o, q_h, q_h, q_o, q_h, q_h], dtype=jnp.float64)
  sigmas = jnp.array([sig_o, 1.0, 1.0, sig_o, 1.0, 1.0], dtype=jnp.float64)
  epsilons = jnp.array([eps_o, 0.0, 0.0, eps_o, 0.0, 0.0], dtype=jnp.float64)
  # Include pair (1) if nonbonded interaction applies; exclude intra-water full exclusions
  mask = jnp.ones((n, n), dtype=jnp.float64) - jnp.eye(n, dtype=jnp.float64)
  intra = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
  for i, j in intra:
    mask = mask.at[i, j].set(0.0)
    mask = mask.at[j, i].set(0.0)
  return {
    "charges": charges,
    "sigmas": sigmas,
    "epsilons": epsilons,
    "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
    "bond_params": jnp.zeros((0, 2), dtype=jnp.float64),
    "angles": jnp.zeros((0, 3), dtype=jnp.int32),
    "angle_params": jnp.zeros((0, 2), dtype=jnp.float64),
    "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
    "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float64),
    "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
    "improper_params": jnp.zeros((0, 3), dtype=jnp.float64),
    "exclusion_mask": mask,
  }


def _openmm_two_waters_energy_forces(
  positions_angstrom: np.ndarray,
  box_angstrom: float,
    *,
    alpha_per_angstrom: float,
    grid: int,
    cutoff_angstrom: float,
) -> tuple[float, np.ndarray]:
  """OpenMM Reference: TIP3P two-water system, PME, kcal/mol and kcal/mol/Å."""
  tip = get_water_params(WaterModelType.TIP3P)
  q_o = tip.charge_O
  q_h = tip.charge_H
  sig_o_nm = tip.sigma_O / 10.0
  eps_o = tip.epsilon_O

  box_nm = box_angstrom / 10.0
  omm_system = openmm.System()
  omm_system.setDefaultPeriodicBoxVectors(
    openmm.Vec3(box_nm, 0, 0),
    openmm.Vec3(0, box_nm, 0),
    openmm.Vec3(0, 0, box_nm),
  )
  for _ in range(6):
    omm_system.addParticle(1.0)

  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.PME)
  nb.setCutoffDistance(cutoff_angstrom / 10.0)
  nb.setPMEParameters(alpha_per_angstrom * 10.0, grid, grid, grid)
  nb.setUseDispersionCorrection(False)
  nb.setUseSwitchingFunction(False)

  for q in [q_o, q_h, q_h, q_o, q_h, q_h]:
    nb.addParticle(q, sig_o_nm if abs(q) > 0.5 else 1e-6, eps_o if abs(q) > 0.5 else 0.0)

  # Intra-water exclusions (replace pair with zero)
  for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
    nb.addException(i, j, 0.0, 1.0, 0.0)

  omm_system.addForce(nb)
  integrator = openmm.VerletIntegrator(0.001)
  context = openmm.Context(omm_system, integrator, openmm.Platform.getPlatformByName("Reference"))
  pos_nm = [openmm.Vec3(p[0] / 10.0, p[1] / 10.0, p[2] / 10.0) for p in positions_angstrom]
  context.setPositions(pos_nm)
  state = context.getState(getEnergy=True, getForces=True)
  e = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
  f = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole / unit.angstrom)
  return float(e), np.asarray(f, dtype=np.float64)


@pytest.mark.openmm
@pytest.mark.slow
@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_two_water_tip3p_explicit_pme_matches_openmm_reference():
  """Six-site TIP3P dimer: energy and forces vs OpenMM Reference (PME)."""
  positions = _tip3p_two_waters_positions()
  box_size = 40.0
  alpha = 0.34
  grid = 32
  cutoff = 12.0

  omm_e, omm_f = _openmm_two_waters_energy_forces(
    positions,
    box_size,
    alpha_per_angstrom=alpha,
    grid=grid,
    cutoff_angstrom=cutoff,
  )

  box_vec = jnp.array([box_size, box_size, box_size], dtype=jnp.float64)
  sys_dict = _prolix_params_two_waters()
  displacement_fn, _ = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(
    displacement_fn,
    sys_dict,
    box=box_vec,
    use_pbc=True,
    implicit_solvent=False,
    pme_grid_points=grid,
    pme_alpha=alpha,
    cutoff_distance=cutoff,
    strict_parameterization=False,
  )
  r = jnp.array(positions)
  jax_e = float(energy_fn(r))

  def grad_e(rr):
    return jax.grad(lambda x: energy_fn(x))(rr)

  jax_f = np.array(grad_e(r))

  assert np.isclose(omm_e, jax_e, atol=2.0), f"energy omm={omm_e} jax={jax_e}"
  rmse = float(np.sqrt(np.mean((omm_f - jax_f) ** 2)))
  assert rmse < 0.15, f"force RMSE {rmse}"
