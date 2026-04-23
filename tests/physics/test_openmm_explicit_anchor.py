"""Anchor parity: Prolix ``physics.system.make_energy_fn`` vs OpenMM Reference (PME).

This is the recommended **first** explicit-solvent validation: two charges in a
periodic box, **dense** electrostatics (no neighbor list), matched PME grid and
Ewald splitting.

**OpenMM alignment**
  - ``setPMEParameters(alpha_nm, nx, ny, nz)`` with ``alpha_nm = alpha_A * 10``
    (OpenMM uses nm⁻¹; Prolix uses Å⁻¹).
  - ``setUseDispersionCorrection(False)`` so OpenMM does not add an automatic
    LJ long-range tail; Prolix adds an explicit isotropic tail in
    ``explicit_corrections.lj_dispersion_tail_energy`` when LJ is active. This
    anchor uses **zero LJ** (electrostatics-only) so the tail is irrelevant.

**Tolerances**
  SPME is mesh-dependent; defaults here match ``test_pbc_end_to_end`` (~0.5
  kcal/mol energy, 0.1 kcal/mol/Å force RMSE). Tighten after locking grid policy.

**Follow-on tests** (not here): neighbor-list path, FlashMD ``flash_explicit``,
solvated protein, trajectory parity — see ``docs/source/explicit_solvent/explicit_solvent_progress.md``.
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


def _minimal_params(charges: list[float]) -> dict:
  n = len(charges)
  return {
    "charges": jnp.array(charges, dtype=jnp.float64),
    "sigmas": jnp.ones(n, dtype=jnp.float64),
    "epsilons": jnp.zeros(n, dtype=jnp.float64),
    "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
    "bond_params": jnp.zeros((0, 2), dtype=jnp.float64),
    "angles": jnp.zeros((0, 3), dtype=jnp.int32),
    "angle_params": jnp.zeros((0, 2), dtype=jnp.float64),
    "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
    "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float64),
    "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
    "improper_params": jnp.zeros((0, 3), dtype=jnp.float64),
    "exclusion_mask": jnp.ones((n, n), dtype=jnp.float64) - jnp.eye(n, dtype=jnp.float64),
  }


def _openmm_pme_energy_forces(
  *,
  positions_angstrom: np.ndarray,
  box_angstrom: float,
  charges: list[float],
  cutoff_angstrom: float,
  alpha_per_angstrom: float,
  grid: int,
  platform_name: str,
  use_dispersion_correction: bool,
) -> tuple[float, np.ndarray]:
  """OpenMM: energy (kcal/mol) and forces (kcal/mol/Å) on the named platform."""
  box_nm = box_angstrom / 10.0
  omm_system = openmm.System()
  omm_system.setDefaultPeriodicBoxVectors(
    openmm.Vec3(box_nm, 0, 0),
    openmm.Vec3(0, box_nm, 0),
    openmm.Vec3(0, 0, box_nm),
  )
  for _ in charges:
    omm_system.addParticle(1.0)

  nonbonded = openmm.NonbondedForce()
  nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
  nonbonded.setCutoffDistance(cutoff_angstrom / 10.0)
  nonbonded.setPMEParameters(alpha_per_angstrom * 10.0, grid, grid, grid)
  nonbonded.setUseDispersionCorrection(bool(use_dispersion_correction))
  nonbonded.setUseSwitchingFunction(False)
  for q in charges:
    nonbonded.addParticle(q, 0.1, 0.0)

  omm_system.addForce(nonbonded)
  integrator = openmm.VerletIntegrator(0.001)
  context = openmm.Context(omm_system, integrator, openmm.Platform.getPlatformByName(platform_name))
  pos_nm = [
    openmm.Vec3(p[0] / 10.0, p[1] / 10.0, p[2] / 10.0) for p in positions_angstrom
  ]
  context.setPositions(pos_nm)
  state = context.getState(getEnergy=True, getForces=True)
  e = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
  f = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole / unit.angstrom)
  return float(e), np.asarray(f, dtype=np.float64)


def _prolix_pme_energy_forces(
  *,
  positions_angstrom: jnp.ndarray,
  box_vec: jnp.ndarray,
  charges: list[float],
  cutoff_angstrom: float,
  alpha_per_angstrom: float,
  grid: int,
) -> tuple[float, np.ndarray]:
  params = _minimal_params(charges)
  displacement_fn, _ = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(
    displacement_fn,
    params,
    box=box_vec,
    use_pbc=True,
    implicit_solvent=False,
    pme_grid_points=grid,
    pme_alpha=alpha_per_angstrom,
    cutoff_distance=cutoff_angstrom,
    strict_parameterization=False,
  )
  pos = jnp.asarray(positions_angstrom, dtype=jnp.float64)
  e = float(energy_fn(pos))
  g = jax.grad(energy_fn)(pos)
  f = -np.asarray(g, dtype=np.float64)
  return e, f


# Tolerances (see module docstring)
ATOL_ENERGY_KCAL = 0.5
RTOL_ENERGY = 1e-5
MAX_RMSE_FORCE = 0.1  # kcal/mol/Å


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_anchor_two_particle_pme_energy_and_forces(regression_pme_params):
  """Minimal periodic two-charge system: energy and forces vs OpenMM."""
  box_size = 30.0
  charges = [1.0, -1.0]
  positions = np.array([[5.0, 5.0, 5.0], [20.0, 5.0, 5.0]], dtype=np.float64)
  alpha = float(regression_pme_params["pme_alpha_per_angstrom"])
  grid = int(regression_pme_params["pme_grid_points"])
  cutoff = float(regression_pme_params["cutoff_angstrom"])
  platform_name = str(regression_pme_params["openmm_platform"])
  use_dispersion_correction = bool(regression_pme_params["use_dispersion_correction"])

  box_vec = jnp.array([box_size, box_size, box_size], dtype=jnp.float64)

  omm_e, omm_f = _openmm_pme_energy_forces(
    positions_angstrom=positions,
    box_angstrom=box_size,
    charges=charges,
    cutoff_angstrom=cutoff,
    alpha_per_angstrom=alpha,
    grid=grid,
    platform_name=platform_name,
    use_dispersion_correction=use_dispersion_correction,
  )
  jax_e, jax_f = _prolix_pme_energy_forces(
    positions_angstrom=jnp.asarray(positions),
    box_vec=box_vec,
    charges=charges,
    cutoff_angstrom=cutoff,
    alpha_per_angstrom=alpha,
    grid=grid,
  )

  assert np.isfinite(omm_e) and np.isfinite(jax_e)
  assert np.all(np.isfinite(jax_f))
  assert np.isclose(omm_e, jax_e, rtol=RTOL_ENERGY, atol=ATOL_ENERGY_KCAL), (
    f"Energy: OpenMM={omm_e:.6f} Prolix={jax_e:.6f} kcal/mol"
  )

  diff = omm_f - jax_f
  rmse = float(np.sqrt(np.mean(diff**2)))
  assert rmse < MAX_RMSE_FORCE, f"Force RMSE {rmse:.6f} >= {MAX_RMSE_FORCE}"
