#!/usr/bin/env python3
"""Optional OpenMM reference: LangevinMiddleIntegrator mean T / variance on a tiny explicit PME system.

Compare **distribution-level** statistics to Prolix (not bitwise trajectories). Uses the same
four-charge periodic layout as ``tests/physics/test_explicit_slow_validation.py`` (OpenMM half).

Requires ``openmm``. Intended for local/cluster runs when wheels install cleanly.
"""

from __future__ import annotations

import argparse
import statistics
import sys


def _build_four_charge_system():
  import openmm
  from openmm import unit as u

  box_size = 40.0
  charges = [1.0, -1.0, 1.0, -1.0]
  positions = [
    [10.0, 10.0, 10.0],
    [30.0, 10.0, 10.0],
    [10.0, 30.0, 10.0],
    [30.0, 30.0, 10.0],
  ]
  alpha = 0.34
  grid = 32
  cutoff = 12.0

  omm_system = openmm.System()
  box_nm = box_size / 10.0
  omm_system.setDefaultPeriodicBoxVectors(
    openmm.Vec3(box_nm, 0, 0),
    openmm.Vec3(0, box_nm, 0),
    openmm.Vec3(0, 0, box_nm),
  )
  for _ in charges:
    omm_system.addParticle(1.0)

  nonbonded = openmm.NonbondedForce()
  nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
  nonbonded.setCutoffDistance(cutoff / 10.0)
  nonbonded.setPMEParameters(alpha * 10.0, grid, grid, grid)
  nonbonded.setUseDispersionCorrection(False)

  for q in charges:
    nonbonded.addParticle(q, 0.1, 0.0)

  omm_system.addForce(nonbonded)
  pos_nm = [openmm.Vec3(p[0] / 10.0, p[1] / 10.0, p[2] / 10.0) for p in positions]
  return omm_system, pos_nm, u


def main() -> int:
  try:
    import openmm
  except ImportError:
    print("openmm is not installed.", file=sys.stderr)
    return 2

  parser = argparse.ArgumentParser(description="OpenMM LangevinMiddleIntegrator temperature statistics.")
  parser.add_argument("--temperature", type=float, default=300.0, help="Target temperature (K).")
  parser.add_argument("--friction", type=float, default=1.0, help="Friction coefficient (1/ps).")
  parser.add_argument("--timestep-fs", type=float, default=1.0, help="Integrator timestep (fs).")
  parser.add_argument("--steps", type=int, default=8000, help="Production steps after burn-in.")
  parser.add_argument("--burn-in", type=int, default=2000, help="Discarded steps.")
  parser.add_argument("--sample-every", type=int, default=10, help="Record KE every N steps.")
  args = parser.parse_args()

  omm_system, pos_nm, u = _build_four_charge_system()
  integrator = openmm.LangevinMiddleIntegrator(
    args.temperature * u.kelvin,
    args.friction / u.picoseconds,
    args.timestep_fs * u.femtoseconds,
  )
  platform = openmm.Platform.getPlatformByName("Reference")
  context = openmm.Context(omm_system, integrator, platform)
  context.setPositions(pos_nm)

  temps_k: list[float] = []
  # k_B in kcal/(mol·K); same scale as Prolix ``BOLTZMANN_KCAL``.
  k_b = 0.0019872041
  dof = 3 * omm_system.getNumParticles() - 3

  total = args.burn_in + args.steps
  for step in range(total):
    integrator.step(1)
    if step < args.burn_in:
      continue
    if (step - args.burn_in) % args.sample_every != 0:
      continue
    ke = context.getState(getEnergy=True).getKineticEnergy().value_in_unit(u.kilocalories_per_mole)
    t_inst = 2.0 * ke / (dof * k_b)
    temps_k.append(t_inst)

  if not temps_k:
    print("No samples collected.", file=sys.stderr)
    return 2

  mean_t = statistics.fmean(temps_k)
  std_t = statistics.pstdev(temps_k) if len(temps_k) > 1 else 0.0
  print(f"samples={len(temps_k)} mean_T_K={mean_t:.3f} std_T_K={std_t:.3f} target_K={args.temperature:.3f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
