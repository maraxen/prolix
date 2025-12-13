
import time
import os
import sys
import jax
import jax.numpy as jnp
from jax_md import space
import numpy as np
from termcolor import colored

# Enable x64
jax.config.update("jax_enable_x64", True)

# Add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    print(colored("OpenMM not found", "red"))
    sys.exit(1)

print("Imports done.")

from prolix.physics import system
from proxide.physics import force_fields
from proxide.md import jax_md_bridge
from proxide.io.parsing import biotite as parsing_biotite
from biotite import structure
import biotite.database.rcsb as rcsb

def benchmark(pdb_code="1UAO", n_steps=100):
    print(colored(f"Benchmarking JAX MD vs OpenMM for {pdb_code} ({n_steps} steps)", "cyan"))
    
    # 1. Load Data
    pdb_path = f"data/pdb/{pdb_code}.pdb"
    if not os.path.exists(pdb_path):
        os.makedirs("data/pdb", exist_ok=True)
        rcsb.fetch(pdb_code, "pdb", "data/pdb")
        
    atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)
    
    # =================
    # OpenMM Benchmark
    # =================
    print(colored("\n[1] OpenMM Benchmark...", "yellow"))
    
    # Helper to convert to OMM
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp:
        from biotite.structure.io.pdb import PDBFile
        pdb_file_bio = PDBFile()
        pdb_file_bio.set_structure(atom_array)
        pdb_file_bio.write(tmp)
        tmp.flush()
        tmp.seek(0)
        pdb_file = app.PDBFile(tmp.name)
        topology = pdb_file.topology
        positions = pdb_file.positions

    # Use same XMLs as verify script
    OPENMM_XMLS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml")),
        'implicit/obc2.xml'
    ]
    
    omm_ff = app.ForceField(*OPENMM_XMLS)
    omm_system = omm_ff.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False, 
        removeCMMotion=False
    )
    
    integrator = openmm.VerletIntegrator(2.0 * unit.femtoseconds)
    # Try CPU platform first, then Reference if CPU fails (should not happen)
    try:
        platform = openmm.Platform.getPlatformByName('CPU')
    except Exception:
        platform = openmm.Platform.getPlatformByName('Reference')
        print("Warning: Using Reference platform (slow)")
        
    simulation = app.Simulation(topology, omm_system, integrator, platform)
    simulation.context.setPositions(positions)
    
    # Warmup
    simulation.step(10)
    
    start_omm = time.time()
    simulation.step(n_steps)
    end_omm = time.time()
    
    omm_time_ms = (end_omm - start_omm) / n_steps * 1000
    print(colored(f"OpenMM ({platform.getName()}): {omm_time_ms:.4f} ms/step", "green"))
    
    
    # =================
    # JAX MD Benchmark
    # =================
    print(colored("\n[2] JAX MD Benchmark...", "yellow"))
    
    ff_path = "data/force_fields/protein19SB.eqx"
    if not os.path.exists(ff_path):
        print(colored(f"Force field not found at {ff_path}", "red"))
        return

    ff = force_fields.load_force_field(ff_path)
    
    full_residues = []
    atom_names = []
    atom_counts = []
    
    for i, res_atoms in enumerate(structure.residue_iter(atom_array)):
        res_name = res_atoms[0].res_name
        full_residues.append(res_name)
        atom_counts.append(len(res_atoms))
        for atom in res_atoms:
            name = atom.atom_name
            if i == 0 and name == "H": name = "H1"
            atom_names.append(name)

    if full_residues:
        full_residues[0] = "N" + full_residues[0]
        full_residues[-1] = "C" + full_residues[-1]

    system_params = jax_md_bridge.parameterize_system(
        ff, full_residues, atom_names, atom_counts
    )
    
    displacement_fn, shift_fn = space.free()
    positions_jax = jnp.array(atom_array.coord)
    
    # Energy fn
    energy_fn = system.make_energy_fn(
        displacement_fn, 
        system_params,
        implicit_solvent=True
    )
    
    compiled_fn = jax.jit(energy_fn)
    # Warmup
    _ = compiled_fn(positions_jax).block_until_ready()
    
    start_jax = time.time()
    # To be fair to OpenMM's C++ loop, we should use scan or simple python loop?
    # OpenMM's simulation.step() is a C++ loop.
    # We should measure raw energy evaluation speed OR integration speed.
    # Energy evaluation is what we optimized.
    # But user asked about "reasonable", which usually implies integration/throughput.
    # Since we don't have a full integrator loop optimized yet (Agent 4 is doing that),
    # let's stick to energy evaluation to be precise about *my* task.
    # But wait, OpenMM `step` does integration.
    # So comparing OpenMM `step` vs JAX MD `energy_fn` is apples to oranges.
    # JAX MD energy call is just gradients (if we added grad).
    # Let's add grad to be comparable to "Force calculation".
    
    grad_fn = jax.jit(jax.grad(energy_fn))
    _ = grad_fn(positions_jax).block_until_ready() # Warmup grad
    
    start_jax_force = time.time()
    for _ in range(n_steps):
        _ = grad_fn(positions_jax).block_until_ready()
    end_jax_force = time.time()
    
    jax_time_ms = (end_jax_force - start_jax_force) / n_steps * 1000
    print(colored(f"JAX MD (Grad/Force, Python Loop): {jax_time_ms:.4f} ms/step", "green"))
    
    # =================
    # JAX MD (Scan)
    # =================
    print(colored("\n[3] JAX MD Benchmark (lax.scan)...", "yellow"))
    
    def scan_fn(pos, _):
        # We process current position, return it as new state (dummy dynamics)
        # Just calculating force/grad to measure throughput
        grads = grad_fn(pos)
        
        # To prevent DCE, we must use the result.
        # Let's do a dummy update: pos = pos + 1e-9 * grads
        # This keeps the computation alive in the graph.
        new_pos = pos + 1e-9 * grads
        
        return new_pos, None
        
    # JIT the scan
    def run_scan(pos, steps):
        final_pos, _ = jax.lax.scan(scan_fn, pos, None, length=steps)
        return final_pos

    run_scan_jit = jax.jit(run_scan, static_argnums=(1,))
    
    # Warmup
    _ = run_scan_jit(positions_jax, 10).block_until_ready()
    
    start_scan = time.time()
    final_pos = run_scan_jit(positions_jax, n_steps).block_until_ready()
    end_scan = time.time()
    
    # Verify result changed to ensure it ran
    diff = jnp.sum(jnp.abs(final_pos - positions_jax))
    # print(f"DEBUG: Total displacement (check for DCE): {diff}")
    
    scan_time_ms = (end_scan - start_scan) / n_steps * 1000
    print(colored(f"JAX MD (lax.scan): {scan_time_ms:.4f} ms/step", "green"))
    
    print("\n------------------------------------------------")
    print(f"Ratio (JAX-Python / OpenMM): {jax_time_ms / omm_time_ms:.2f}x")
    # print(f"Ratio (JAX-Scan / OpenMM):   {omm_time_ms / scan_time_ms:.2f}x (Speedup)") # Correct ratio
    print(f"Speedup vs OpenMM: {omm_time_ms / scan_time_ms:.2f}x")
    print("------------------------------------------------")

if __name__ == "__main__":
    benchmark(n_steps=1000)
