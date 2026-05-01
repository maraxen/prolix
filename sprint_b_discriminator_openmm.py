#!/usr/bin/env python3
"""Sprint B Discriminator: OpenMM cross-validation.

Runs the EXACT same setup as the prolix Sprint B baseline:
  - n_waters=64, TIP3P rigid, 10Å grid spacing (gas-like, NO pre-equilibration)
  - Langevin γ=1/ps, dt=0.5fs, T_target=300K
  - 50ps total (33ps burn, 17ps production)
  - Reports mean and std temperature from the same DOF (6*N - 3 rigid-body)

If OpenMM gives ~300K: prolix has a bug (thermostat or estimator).
If OpenMM also gives ~411K: the offset is initialization artifact (10Å grid needs more time).
"""
from __future__ import annotations
import numpy as np

from openmm import app, unit, LangevinMiddleIntegrator, Platform
from openmm.app import Modeller, ForceField, Simulation

n_waters = 64
spacing_ang = 10.0
dt_fs = 0.5
gamma_ps = 1.0
T_target_K = 300.0
sim_ps = 50.0
burn_fraction = 1.0 / 3.0

# Build cubic grid of water molecules at 10 Å spacing (same as prolix)
positions_nm = []
n_side = round(n_waters ** (1.0 / 3.0))
for ix in range(n_side):
    for iy in range(n_side):
        for iz in range(n_side):
            cx = ix * spacing_ang * 0.1  # nm
            cy = iy * spacing_ang * 0.1
            cz = iz * spacing_ang * 0.1
            # TIP3P geometry: O at center, H1 and H2 at ~0.9572 Å, angle 104.52°
            import math
            r_oh = 0.09572  # nm
            angle_hoh = math.radians(104.52)
            hx = r_oh * math.sin(angle_hoh / 2)
            hy = r_oh * math.cos(angle_hoh / 2)
            positions_nm.append([cx, cy, cz])              # O
            positions_nm.append([cx + hx, cy + hy, cz])   # H1
            positions_nm.append([cx - hx, cy + hy, cz])   # H2

box_nm = n_side * spacing_ang * 0.1
n_atoms = n_waters * 3

# Build topology
topology = app.Topology()
chain = topology.addChain()
for _ in range(n_waters):
    res = topology.addResidue("HOH", chain)
    o = topology.addAtom("O", app.Element.getBySymbol("O"), res)
    h1 = topology.addAtom("H1", app.Element.getBySymbol("H"), res)
    h2 = topology.addAtom("H2", app.Element.getBySymbol("H"), res)
    topology.addBond(o, h1)
    topology.addBond(o, h2)

from openmm import Vec3
topology.setPeriodicBoxVectors(
    (Vec3(box_nm, 0, 0), Vec3(0, box_nm, 0), Vec3(0, 0, box_nm)) * unit.nanometer
)

ff = ForceField("tip3p.xml")
system = ff.createSystem(
    topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=0.9 * unit.nanometer,
    constraints=app.HBonds,
    rigidWater=True,
    ewaldErrorTolerance=1e-4,
)

integrator = LangevinMiddleIntegrator(
    T_target_K * unit.kelvin,
    gamma_ps / unit.picosecond,
    dt_fs * unit.femtosecond,
)
integrator.setRandomNumberSeed(7)

platform = Platform.getPlatformByName("CPU")
sim = Simulation(topology, system, integrator, platform)
sim.context.setPositions(np.array(positions_nm) * unit.nanometer)
sim.context.setVelocitiesToTemperature(T_target_K * unit.kelvin, 7)

n_steps_total = int(sim_ps * 1000.0 / dt_fs)
n_burn = int(n_steps_total * burn_fraction)
n_prod = n_steps_total - n_burn

print(f"OpenMM TIP3P rigid, n_waters={n_waters}, 10Å grid, γ={gamma_ps}/ps, dt={dt_fs}fs")
print(f"Burn: {n_burn} steps ({n_burn*dt_fs/1000:.1f} ps)")
print(f"Production: {n_prod} steps ({n_prod*dt_fs/1000:.1f} ps)")
print("Running burn...", flush=True)
sim.step(n_burn)

print("Running production...", flush=True)
temps_K = []
# Sample every 100 steps to reduce autocorrelation
sample_interval = 100
for _ in range(n_prod // sample_interval):
    sim.step(sample_interval)
    state = sim.context.getState(getEnergy=True)
    ke = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    # DOF: same as prolix, 6*n_waters - 3 rigid-body
    dof = 6 * n_waters - 3
    R = 8.314472e-3  # kJ/mol/K
    T_inst = 2.0 * ke / (dof * R)
    temps_K.append(T_inst)

temps_K = np.array(temps_K)
t_mean = float(np.mean(temps_K))
t_std = float(np.std(temps_K))
print(f"\nOpenMM result: T_mean={t_mean:.2f} K  T_std={t_std:.2f} K  ΔT={t_mean-T_target_K:+.2f} K")
print(f"n_samples={len(temps_K)}")
print()
if abs(t_mean - 300.0) < 15.0:
    print("VERDICT: OpenMM ≈ 300K → Prolix has a bug (not an initialization issue)")
elif t_mean > 350.0:
    print("VERDICT: OpenMM also hot → Initialization artifact (10Å grid needs more burn-in or pre-equilibration)")
else:
    print(f"VERDICT: UNCLEAR — OpenMM at {t_mean:.1f}K, prolix at ~411K, both somewhat elevated")
