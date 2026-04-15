"""OpenMM-pinned checks for optional reaction-field electrostatics (user guide §19.6.3)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

try:
  import openmm

  HAS_OPENMM = True
except ImportError:
  HAS_OPENMM = False

from prolix.physics import pbc, system
from prolix.physics.electrostatic_methods import ElectrostaticMethod, openmm_reaction_field_coefficients


def _dict_two_particles(
    charges: tuple[float, float],
) -> dict:
    return {
        "charges": jnp.array(charges, dtype=jnp.float64),
        "sigmas": jnp.ones(2, dtype=jnp.float64) * 0.1,
        "epsilons": jnp.zeros(2, dtype=jnp.float64),
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "exclusion_mask": jnp.ones((2, 2), dtype=jnp.float64) - jnp.eye(2, dtype=jnp.float64),
    }


@pytest.mark.openmm
@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_reaction_field_energy_matches_openmm_reference():
  """Two charges, periodic box, CutoffPeriodic + reaction field; dense path."""
  from openmm import unit

  box_a = 50.0
  cutoff = 10.0
  eps_solv = 78.3
  charges = [1.0, -1.0]
  pos_a = np.array([[20.0, 25.0, 25.0], [30.0, 25.0, 25.0]], dtype=np.float64)

  box_nm = box_a / 10.0
  omm_system = openmm.System()
  omm_system.setDefaultPeriodicBoxVectors(
    openmm.Vec3(box_nm, 0, 0),
    openmm.Vec3(0, box_nm, 0),
    openmm.Vec3(0, 0, box_nm),
  )
  for _ in charges:
    omm_system.addParticle(1.0)
  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
  nb.setCutoffDistance(cutoff / 10.0)
  nb.setReactionFieldDielectric(eps_solv)
  nb.setUseDispersionCorrection(False)
  nb.setUseSwitchingFunction(False)
  for q in charges:
    nb.addParticle(q, 0.1, 0.0)
  omm_system.addForce(nb)
  integrator = openmm.VerletIntegrator(0.001)
  context = openmm.Context(omm_system, integrator, openmm.Platform.getPlatformByName("Reference"))
  pos_nm = [openmm.Vec3(p[0] / 10.0, p[1] / 10.0, p[2] / 10.0) for p in pos_a]
  context.setPositions(pos_nm)
  state = context.getState(getEnergy=True)
  e_omm = float(
    state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
  )

  box = jnp.array([box_a, box_a, box_a], dtype=jnp.float64)
  r = jnp.asarray(pos_a)
  displacement_fn, _ = pbc.create_periodic_space(box)
  energy_fn = system.make_energy_fn(
      displacement_fn,
      _dict_two_particles((charges[0], charges[1])),
      box=box,
      use_pbc=True,
      implicit_solvent=False,
      cutoff_distance=cutoff,
      pme_alpha=0.34,
      strict_parameterization=False,
      electrostatic_method=ElectrostaticMethod.REACTION_FIELD,
      reaction_field_dielectric=eps_solv,
  )
  e_px = float(energy_fn(r))

  assert np.isclose(e_px, e_omm, rtol=1e-4, atol=0.02), f"Prolix={e_px} OpenMM={e_omm}"


def test_openmm_reaction_field_coefficients_match_literature_form():
  k_rf, c_rf = openmm_reaction_field_coefficients(10.0, 78.3)
  rc = 10.0
  eps = 78.3
  assert np.isclose(k_rf, (1.0 / rc**3) * ((eps - 1.0) / (2.0 * eps + 1.0)))
  assert np.isclose(c_rf, (1.0 / rc) * (3.0 * eps / (2.0 * eps + 1.0)))
