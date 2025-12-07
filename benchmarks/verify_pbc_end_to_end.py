"""
End-to-End PBC/PME Physics Verification Script.

This script validates the JAX MD pipeline with Periodic Boundary Conditions
and PME electrostatics against OpenMM. It ensures that the implementation
independently produces the same physics as OpenMM for periodic systems.
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space, energy
from termcolor import colored

# Enable x64
jax.config.update("jax_enable_x64", True)

# OpenMM Imports
try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    print(colored("Error: OpenMM not found. Please install it.", "red"))
    sys.exit(1)

# Prolix Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from prolix.physics import system, pbc, pme
from priox.physics import force_fields, constants
from priox.md import jax_md_bridge
from priox.io.parsing import biotite as parsing_biotite
import biotite.structure.io.pdb as pdb_io

# Configuration
PDB_PATH = "data/pdb/1UAO.pdb"  # Small protein for testing
FF_EQX_PATH = "data/force_fields/protein19SB.eqx"
OPENMM_XMLS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml")),
]

# PBC Settings
BOX_PADDING = 10.0  # Angstroms
PME_CUTOFF = 9.0    # Angstroms
PME_ALPHA = 0.34    # 1/Angstrom (3.4 1/nm for OpenMM)
PME_GRID = 32       # Grid points per dimension

# Tolerances
TOL_ENERGY = 1.0    # kcal/mol (Relaxed for PME approximation)
TOL_FORCE_RMSE = 0.1  # kcal/mol/A


def setup_box(positions_angstrom):
    """Create a periodic box around positions with padding."""
    min_coords = np.min(positions_angstrom, axis=0)
    max_coords = np.max(positions_angstrom, axis=0)
    box_size = (max_coords - min_coords) + 2 * BOX_PADDING
    
    # Shift positions to center in box
    center = (max_coords + min_coords) / 2
    box_center = box_size / 2
    shift = box_center - center
    centered_positions = positions_angstrom + shift
    
    return centered_positions, box_size


def run_pbc_verification(pdb_code="1UAO", return_results=False):
    print(colored("===========================================================", "cyan"))
    print(colored(f"   PBC/PME End-to-End Verification: {pdb_code} + ff19SB", "cyan"))
    print(colored("===========================================================", "cyan"))

    # -------------------------------------------------------------------------
    # 1. Load Structure
    # -------------------------------------------------------------------------
    pdb_path = f"data/pdb/{pdb_code}.pdb"
    print(colored(f"\n[1] Loading Structure {pdb_path}...", "yellow"))
    
    if not os.path.exists(pdb_path):
        import biotite.database.rcsb as rcsb
        os.makedirs("data/pdb", exist_ok=True)
        try:
            rcsb.fetch(pdb_code, "pdb", "data/pdb")
        except Exception as e:
            print(colored(f"Error fetching {pdb_code}: {e}", "red"))
            if return_results: return None
            sys.exit(1)
        
    atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)
    
    # -------------------------------------------------------------------------
    # 2. Setup OpenMM System with PME (Ground Truth)
    # -------------------------------------------------------------------------
    print(colored("\n[2] Setting up OpenMM System with PME...", "yellow"))
    
    # Convert to OpenMM Topology/Positions via temporary PDB
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp:
        pdb_file_bio = pdb_io.PDBFile()
        pdb_file_bio.set_structure(atom_array)
        pdb_file_bio.write(tmp)
        tmp.flush()
        tmp.seek(0)
        pdb_file = app.PDBFile(tmp.name)
        topology = pdb_file.topology
        positions_nm = pdb_file.positions

    positions_angstrom = np.array(positions_nm.value_in_unit(unit.angstrom))
    
    # Create box
    centered_positions, box_size_angstrom = setup_box(positions_angstrom)
    box_size_nm = box_size_angstrom / 10.0  # Convert to nm
    
    print(f"  Box Size: {box_size_angstrom[0]:.2f} x {box_size_angstrom[1]:.2f} x {box_size_angstrom[2]:.2f} A")
    
    # Set box vectors on topology
    topology.setPeriodicBoxVectors([
        openmm.Vec3(box_size_nm[0], 0, 0),
        openmm.Vec3(0, box_size_nm[1], 0),
        openmm.Vec3(0, 0, box_size_nm[2])
    ])
    
    omm_ff = app.ForceField(*OPENMM_XMLS)
    omm_system = omm_ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=PME_CUTOFF / 10.0 * unit.nanometer,  # Convert A to nm
        constraints=None,
        rigidWater=False,
        removeCMMotion=False
    )
    
    # Set PME parameters explicitly
    for force in omm_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            # alpha in nm^-1 = alpha_A * 10
            force.setPMEParameters(PME_ALPHA * 10.0, PME_GRID, PME_GRID, PME_GRID)
            force.setUseDispersionCorrection(False)
            print(f"  Set PME: alpha={PME_ALPHA * 10.0:.2f} nm^-1, grid={PME_GRID}")
    
    # Set box on system
    omm_system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_size_nm[0], 0, 0),
        openmm.Vec3(0, box_size_nm[1], 0),
        openmm.Vec3(0, 0, box_size_nm[2])
    )
    
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(topology, omm_system, integrator, platform)
    
    # Set centered positions
    centered_positions_nm = [(p / 10.0) * unit.nanometer for p in centered_positions]
    simulation.context.setPositions(centered_positions_nm)
    simulation.context.setPeriodicBoxVectors(
        openmm.Vec3(box_size_nm[0], 0, 0),
        openmm.Vec3(0, box_size_nm[1], 0),
        openmm.Vec3(0, 0, box_size_nm[2])
    )
    
    state = simulation.context.getState(getEnergy=True, getForces=True)
    omm_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    omm_forces = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole / unit.angstrom)
    
    print(f"OpenMM Total Energy (PME): {omm_energy:.4f} kcal/mol")
    
    # Breakdown OpenMM Energy by force group
    omm_components = {}
    for i, force in enumerate(omm_system.getForces()):
        force.setForceGroup(i)
        
    # Recreate context with groups
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    simulation = app.Simulation(topology, omm_system, integrator, platform)
    simulation.context.setPositions(centered_positions_nm)
    simulation.context.setPeriodicBoxVectors(
        openmm.Vec3(box_size_nm[0], 0, 0),
        openmm.Vec3(0, box_size_nm[1], 0),
        openmm.Vec3(0, 0, box_size_nm[2])
    )
    
    print("\n  OpenMM Energy Breakdown:")
    for i, force in enumerate(omm_system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups=1<<i)
        e = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        fname = force.__class__.__name__
        omm_components[fname] = omm_components.get(fname, 0.0) + e
        print(f"    {fname}: {e:.4f} kcal/mol")
    
    # -------------------------------------------------------------------------
    # 3. Setup JAX MD System with PBC/PME
    # -------------------------------------------------------------------------
    print(colored("\n[3] Setting up JAX MD System with PBC/PME...", "yellow"))
    
    ff = force_fields.load_force_field(FF_EQX_PATH)
    
    # Extract topology info
    residues = []
    atom_names = []
    atom_counts = []
    
    for i, chain in enumerate(topology.chains()):
        for j, res in enumerate(chain.residues()):
            residues.append(res.name)
            count = 0
            for atom in res.atoms():
                name = atom.name
                if i == 0 and j == 0 and name == "H":
                    name = "H1"
                atom_names.append(name)
                count += 1
            atom_counts.append(count)
            
    # Rename terminals
    if residues:
        residues[0] = "N" + residues[0]
        residues[-1] = "C" + residues[-1]
            
    # Parameterize
    system_params = jax_md_bridge.parameterize_system(
        ff, residues, atom_names, atom_counts
    )
    
    # Setup PBC displacement function
    box_vec = jnp.array(box_size_angstrom)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    
    jax_positions = jnp.array(centered_positions)
    
    # Create energy function with PBC/PME
    energy_fn = system.make_energy_fn(
        displacement_fn,
        system_params,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,  # No GBSA for explicit solvent
        cutoff_distance=PME_CUTOFF,
        pme_grid_points=PME_GRID,
        pme_alpha=PME_ALPHA,
    )
    
    jax_energy = float(energy_fn(jax_positions))
    jax_forces = -jax.grad(energy_fn)(jax_positions)
    
    print(f"JAX MD Total Energy (PME): {jax_energy:.4f} kcal/mol")
    
    # -------------------------------------------------------------------------
    # 4. Compare Results
    # -------------------------------------------------------------------------
    print(colored("\n[4] Comparing Results...", "yellow"))
    
    energy_diff = abs(omm_energy - jax_energy)
    force_diff = np.array(omm_forces) - np.array(jax_forces)
    force_rmse = np.sqrt(np.mean(force_diff**2))
    
    print(f"\n  Energy Difference: {energy_diff:.4f} kcal/mol (Tolerance: {TOL_ENERGY})")
    print(f"  Force RMSE: {force_rmse:.6f} kcal/mol/A (Tolerance: {TOL_FORCE_RMSE})")
    
    # Verdict
    energy_pass = energy_diff < TOL_ENERGY
    force_pass = force_rmse < TOL_FORCE_RMSE
    
    print("\n" + "="*60)
    if energy_pass and force_pass:
        print(colored("  RESULT: PASS", "green"))
    elif energy_pass:
        print(colored("  RESULT: PARTIAL PASS (Energy OK, Forces differ)", "yellow"))
    else:
        print(colored("  RESULT: FAIL", "red"))
    print("="*60)
    
    results = {
        "pdb_code": pdb_code,
        "omm_energy": omm_energy,
        "jax_energy": jax_energy,
        "energy_diff": energy_diff,
        "force_rmse": force_rmse,
        "energy_pass": energy_pass,
        "force_pass": force_pass,
        "omm_components": omm_components,
    }
    
    if return_results:
        return results
    
    return energy_pass and force_pass


def run_simple_pme_verification():
    """Quick verification with synthetic 2-particle system."""
    print(colored("\n" + "="*60, "cyan"))
    print(colored("  Simple PME Verification (2 Particles)", "cyan"))
    print(colored("="*60, "cyan"))
    
    # Setup
    box_size = 30.0  # Angstroms
    box_vec = jnp.array([box_size, box_size, box_size])
    charges = [1.0, -1.0]
    positions = [[5.0, 5.0, 5.0], [20.0, 5.0, 5.0]]  # 15A separation
    
    # OpenMM
    omm_system = openmm.System()
    omm_system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_size/10.0, 0, 0),
        openmm.Vec3(0, box_size/10.0, 0),
        openmm.Vec3(0, 0, box_size/10.0)
    )
    
    for q in charges:
        omm_system.addParticle(1.0)
        
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(0.9)  # 9A in nm
    nonbonded.setPMEParameters(PME_ALPHA * 10.0, PME_GRID, PME_GRID, PME_GRID)
    nonbonded.setUseDispersionCorrection(False)
    
    for q in charges:
        nonbonded.addParticle(q, 0.1, 0.0)  # sigma 1A, epsilon 0
        
    omm_system.addForce(nonbonded)
    
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(omm_system, integrator, openmm.Platform.getPlatformByName('Reference'))
    context.setPositions([openmm.Vec3(p[0]/10, p[1]/10, p[2]/10) for p in positions])
    
    omm_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    print(f"  OpenMM Energy: {omm_energy:.4f} kcal/mol")
    
    # JAX MD
    mock_system_params = {
        "charges": jnp.array(charges),
        "sigmas": jnp.array([1.0, 1.0]),
        "epsilons": jnp.array([0.0, 0.0]),
        "bonds": jnp.zeros((0, 2), dtype=int),
        "bond_params": jnp.zeros((0, 2)),
        "angles": jnp.zeros((0, 3), dtype=int),
        "angle_params": jnp.zeros((0, 2)),
        "dihedrals": jnp.zeros((0, 4), dtype=int),
        "dihedral_params": jnp.zeros((0, 3)),
        "impropers": jnp.zeros((0, 4), dtype=int),
        "improper_params": jnp.zeros((0, 3)),
        "exclusion_mask": jnp.ones((2, 2)) - jnp.eye(2),
    }
    
    displacement_fn, _ = pbc.create_periodic_space(box_vec)
    
    energy_fn = system.make_energy_fn(
        displacement_fn,
        mock_system_params,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=PME_GRID,
        pme_alpha=PME_ALPHA,
    )
    
    jax_energy = float(energy_fn(jnp.array(positions)))
    print(f"  JAX MD Energy: {jax_energy:.4f} kcal/mol")
    
    diff = abs(omm_energy - jax_energy)
    print(f"  Difference: {diff:.4f} kcal/mol")
    
    if diff < 0.5:
        print(colored("  PASS", "green"))
        return True
    else:
        print(colored("  FAIL", "red"))
        return False


if __name__ == "__main__":
    # Run simple verification first
    simple_pass = run_simple_pme_verification()
    
    # Run full protein verification
    if len(sys.argv) > 1 and sys.argv[1] != "--simple":
        pdb_code = sys.argv[1]
    else:
        pdb_code = "1UAO"
    
    if "--simple" not in sys.argv:
        full_pass = run_pbc_verification(pdb_code)
    else:
        full_pass = simple_pass
        
    sys.exit(0 if (simple_pass and full_pass) else 1)
