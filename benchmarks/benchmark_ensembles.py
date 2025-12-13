
import time
import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from termcolor import colored
from prolix.physics import simulate, system
from proxide.physics import force_fields
from proxide.md import jax_md_bridge
from proxide.physics import constants
from jax_md import space

# Configuration
PDB_PATH = "data/pdb/1UAO.pdb"
FF_EQX_PATH = "data/force_fields/protein19SB.eqx"
STEPS = 1000
DT_FS = 2.0

def benchmark_ensembles():
    print(colored("===========================================================", "cyan"))
    print(colored(f"   Benchmark Ensembles: Prolix vs OpenMM (1UAO)", "cyan"))
    print(colored("===========================================================", "cyan"))

    # 1. Setup Input
    import biotite.structure.io.pdb as pdb
    import biotite.structure as struc
    from proxide.io.parsing import biotite as parsing_biotite
    
    atom_array = parsing_biotite.load_structure_with_hydride(PDB_PATH, model=1)
    
    # 2. Setup OpenMM Verification Baseline
    print(colored("\n[OpenMM] Setting up...", "yellow"))
    # (Simplified OpenMM setup similar to verification script)
    # Convert to OpenMM Topology/Positions via temporary PDB
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp:
        pdb_file_bio = pdb.PDBFile()
        pdb_file_bio.set_structure(atom_array)
        pdb_file_bio.write(tmp)
        tmp.flush()
        tmp.seek(0)
        pdb_file = app.PDBFile(tmp.name)
        topology = pdb_file.topology
        positions = pdb_file.positions
    
    omm_ff = app.ForceField('amber14-all.xml', 'implicit/obc2.xml') # Using amber14 for standard benchmark
    omm_system = omm_ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    
    # OpenMM NVT (Langevin)
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, DT_FS*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference') # Use Reference for fairness? No, CPU.
    # Actually use CPU for fair comparison with JAX CPU (or CUDA if available)
    try:
        platform = openmm.Platform.getPlatformByName('CPU')
    except:
        platform = openmm.Platform.getPlatformByName('Reference')
        
    simulation = app.Simulation(topology, omm_system, integrator, platform)
    simulation.context.setPositions(positions)
    
    print(colored(f"[OpenMM] Running {STEPS} steps NVT (Langevin)...", "green"))
    start = time.time()
    simulation.step(STEPS)
    end = time.time()
    omm_fps = STEPS / (end - start)
    print(f"OpenMM Speed: {omm_fps:.2f} steps/sec")

    # 3. Setup Prolix
    print(colored("\n[Prolix] Setting up...", "yellow"))
    ff = force_fields.load_force_field(FF_EQX_PATH)
    
    residues = []
    atom_names = []
    atom_counts = []
    for chain in topology.chains():
        for res in chain.residues():
            residues.append(res.name)
            count = 0
            for atom in res.atoms():
                name = atom.name
                if name == "H" and res.index == 0: name = "H1"
                atom_names.append(name)
                count += 1
            atom_counts.append(count)
    if residues: residues[0] = "N" + residues[0]; residues[-1] = "C" + residues[-1]
    
    system_params = jax_md_bridge.parameterize_system(ff, residues, atom_names, atom_counts)
    
    jax_positions = jnp.array(positions.value_in_unit(unit.angstrom))
    
    displacement_fn, shift_fn = space.free()
    energy_fn = system.make_energy_fn(displacement_fn, system_params, implicit_solvent=True)
    
    # Warmup / JIT
    print(colored("[Prolix] JIT Compiling...", "yellow"))
    spec = simulate.SimulationSpec(total_time_ns=STEPS * 1e-6 * DT_FS, 
                                   step_size_fs=DT_FS, 
                                   ensemble="nvt_langevin",
                                   accumulate_steps=STEPS)
                                   
    run_fn = jax.jit(lambda r: simulate.run_simulation(system_params, r, key=jax.random.PRNGKey(0), sim_spec=spec))
    _ = run_fn(jax_positions).block_until_ready()
    
    # Run Benchmark
    print(colored(f"[Prolix] Running {STEPS} steps NVT (Langevin)...", "green"))
    start = time.time()
    _ = run_fn(jax_positions).block_until_ready()
    end = time.time()
    jax_fps = STEPS / (end - start)
    print(f"Prolix Speed: {jax_fps:.2f} steps/sec")
    
    # NVE Benchmark
    print(colored(f"[Prolix] Running {STEPS} steps NVE...", "green"))
    spec_nve = simulate.SimulationSpec(total_time_ns=STEPS * 1e-6 * DT_FS,
                                       step_size_fs=DT_FS,
                                       ensemble="nve",
                                       accumulate_steps=STEPS)
    run_nve_fn = jax.jit(lambda r: simulate.run_simulation(system_params, r, key=jax.random.PRNGKey(0), sim_spec=spec_nve))
    _ = run_nve_fn(jax_positions).block_until_ready() # Warmup
    
    start = time.time()
    _ = run_nve_fn(jax_positions).block_until_ready()
    end = time.time()
    jax_nve_fps = STEPS / (end - start)
    print(f"Prolix NVE Speed: {jax_nve_fps:.2f} steps/sec")
    
    # NVT Nose-Hoover Benchmark
    print(colored(f"[Prolix] Running {STEPS} steps NVT (Nose-Hoover)...", "green"))
    spec_nh = simulate.SimulationSpec(total_time_ns=STEPS * 1e-6 * DT_FS,
                                      step_size_fs=DT_FS,
                                      ensemble="nvt_nose_hoover",
                                      accumulate_steps=STEPS)
    run_nh_fn = jax.jit(lambda r: simulate.run_simulation(system_params, r, key=jax.random.PRNGKey(0), sim_spec=spec_nh))
    _ = run_nh_fn(jax_positions).block_until_ready() # Warmup
    
    start = time.time()
    _ = run_nh_fn(jax_positions).block_until_ready()
    end = time.time()
    jax_nh_fps = STEPS / (end - start)
    print(f"Prolix NVT (NH) Speed: {jax_nh_fps:.2f} steps/sec")
    
    # Brownian Benchmark
    print(colored(f"[Prolix] Running {STEPS} steps Brownian...", "green"))
    spec_br = simulate.SimulationSpec(total_time_ns=STEPS * 1e-6 * DT_FS,
                                      step_size_fs=DT_FS,
                                      ensemble="brownian",
                                      accumulate_steps=STEPS)
    run_br_fn = jax.jit(lambda r: simulate.run_simulation(system_params, r, key=jax.random.PRNGKey(0), sim_spec=spec_br))
    _ = run_br_fn(jax_positions).block_until_ready() # Warmup
    
    start = time.time()
    _ = run_br_fn(jax_positions).block_until_ready()
    end = time.time()
    jax_br_fps = STEPS / (end - start)
    print(f"Prolix Brownian Speed: {jax_br_fps:.2f} steps/sec")

    print(colored("\nSummary:", "white"))
    print(f"  OpenMM (CPU): {omm_fps:.2f} steps/sec")
    print(f"  Prolix (Langevin): {jax_fps:.2f} steps/sec")
    print(f"  Prolix (NVE): {jax_nve_fps:.2f} steps/sec")
    print(f"  Prolix (Nose-Hoover): {jax_nh_fps:.2f} steps/sec")
    print(f"  Prolix (Brownian): {jax_br_fps:.2f} steps/sec")

if __name__ == "__main__":
    benchmark_ensembles()
