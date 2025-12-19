
import openmm
from openmm import app, unit


def verify_physics():
    print("Verifying OpenMM PeriodicTorsionForce Physics...")

    # 1. Create a 4-particle system defining a 0 degree dihedral
    # Atoms at (1,0,0), (0,0,0), (0,0,1), (1,0,1)?
    # i=(1,0,0), j=(0,0,0), k=(0,0,1).
    # b0 = j-i = (-1,0,0).
    # b1 = k-j = (0,0,1).
    # b2 = l-k.
    # We want phi=0 (cis).
    # l should be at (1,0,1).
    # b2 = (1,0,0).
    # Normal to b0,b1: cross((-1,0,0), (0,0,1)) = (0,1,0).
    # Normal to b1,b2: cross((0,0,1), (1,0,0)) = (0,1,0).
    # Parallel -> 0 degrees.
    # Check definition: IUPAC 0 is cis.

    positions = [
        openmm.Vec3(1,0,0),
        openmm.Vec3(0,0,0),
        openmm.Vec3(0,0,1),
        openmm.Vec3(1,0,1)
    ]

    system = openmm.System()
    for _ in range(4): system.addParticle(1.0)

    # Add Periodic Torsion
    force = openmm.PeriodicTorsionForce()
    # k=1.0 kJ/mol, n=1, phase=0
    # Expected E = 1.0 * (1 + cos(0)) = 2.0 kJ/mol
    force.addTorsion(0, 1, 2, 3, 1, 0.0 * unit.radians, 1.0 * unit.kilojoules_per_mole)

    # Test phase=Pi (180 deg)
    # Expected E = 1.0 * (1 + cos(0 - Pi)) = 1.0 * (1 - 1) = 0.0
    force.addTorsion(0, 1, 2, 3, 1, 3.141592653589793 * unit.radians, 1.0 * unit.kilojoules_per_mole)

    system.addForce(force)

    integrator = openmm.VerletIntegrator(1.0)
    sim = app.Simulation(app.Topology(), system, integrator, openmm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions(positions)

    state = sim.context.getState(getEnergy=True)
    e_tot = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"Total Energy (0 deg): {e_tot} kJ/mol")
    print("Expected: Term1(2.0) + Term2(0.0) = 2.0")

    if abs(e_tot - 2.0) < 1e-4:
        print("PASS: OpenMM matches k(1+cos).")
    else:
        print("FAIL: OpenMM mismatch.")

    # Test 180 degrees (trans)
    # Move atom 0 to (-1, 0, 0)
    positions_trans = [
        openmm.Vec3(-1,0,0),
        openmm.Vec3(0,0,0),
        openmm.Vec3(0,0,1),
        openmm.Vec3(1,0,1)
    ]
    sim.context.setPositions(positions_trans)
    e_trans = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"Total Energy (180 deg): {e_trans} kJ/mol")
    # Term 1: 1*(1+cos(180)) = 0
    # Term 2: 1*(1+cos(180-180)) = 1*(1+1) = 2
    print("Expected: Term1(0.0) + Term2(2.0) = 2.0")

    if abs(e_trans - 2.0) < 1e-4:
        print("PASS: OpenMM matches k(1+cos) at 180.")
    else:
        print("FAIL: OpenMM mismatch at 180.")

if __name__ == "__main__":
    verify_physics()
