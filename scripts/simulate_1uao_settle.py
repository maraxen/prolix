"""Run 1UAO simulation in explicit solvent with SETTLE constraints.

Uses SETTLE for rigid water, enabling 2 fs timesteps.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt
from matplotlib import animation

from proxide.io.parsing.backend import parse_structure
from proxide import OutputSpec, CoordFormat

# Prolix imports
from prolix.physics import system, solvation, settle
from prolix.physics.solvation import fix_water_geometry
from prolix.physics.neighbor_list import ExclusionSpec, make_neighbor_list_fn
from jax_md import space, partition

# Enable f64 for physics precision
jax.config.update("jax_enable_x64", True)


def main():
    # 1. Load and parameterize using Rust parser
    pdb_path = "data/pdb/1UAO.pdb"
    ff_path = "proxide/src/proxide/assets/protein.ff19SB.xml"
    
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Missing {pdb_path}")
    if not os.path.exists(ff_path):
        raise FileNotFoundError(f"Force field not found at {ff_path}")

    print(f"Loading {pdb_path}...")
    print(f"Using force field: {ff_path}")
    
    # Parse with MD parameterization
    spec = OutputSpec()
    spec.coord_format = CoordFormat.Full
    spec.parameterize_md = True
    spec.force_field = ff_path
    spec.add_hydrogens = False  # Use existing hydrogens or add later
    spec.remove_solvent = True
    
    protein = parse_structure(pdb_path, spec)
    
    # Extract coordinates - filter by mask
    coords = np.array(protein.coordinates)
    mask = np.array(protein.atom_mask)
    
    if coords.ndim == 3:
        flat_coords = coords.reshape(-1, 3)
        flat_mask = mask.reshape(-1)
        valid_indices = np.where(flat_mask > 0.5)[0]
        positions = jnp.array(flat_coords[valid_indices])
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
    else:
        positions = jnp.array(coords)
        valid_indices = np.arange(len(coords))
        old_to_new = {i: i for i in range(len(coords))}
    
    n_atoms = len(positions)
    print(f"Structure loaded: {n_atoms} atoms")
    
    # Remap bonds
    orig_bonds = np.array(protein.bonds) if protein.bonds is not None else np.zeros((0, 2), dtype=np.int32)
    remapped_bonds = []
    for b in orig_bonds:
        if b[0] in old_to_new and b[1] in old_to_new:
            remapped_bonds.append([old_to_new[b[0]], old_to_new[b[1]]])
    bonds = np.array(remapped_bonds, dtype=np.int32) if remapped_bonds else np.zeros((0, 2), dtype=np.int32)
    print(f"Remapped {len(remapped_bonds)} bonds")
    
    angles = np.array(protein.angles) if protein.angles is not None else np.zeros((0, 3), dtype=np.int32)
    
    # Get params for valid atoms only
    n_valid = len(valid_indices)
    charges = np.array(protein.charges)[:n_valid] if protein.charges is not None else np.zeros(n_valid)
    sigmas = np.array(protein.sigmas)[:n_valid] if protein.sigmas is not None else np.ones(n_valid) * 3.0
    epsilons = np.array(protein.epsilons)[:n_valid] if protein.epsilons is not None else np.ones(n_valid) * 0.1
    
    # FIX: Some atoms have sigma=0 from parser bug - use minimum value
    # This prevents LJ singularities at short distances
    n_zero_sigma = np.sum(sigmas == 0)
    if n_zero_sigma > 0:
        print(f"WARNING: {n_zero_sigma} atoms have sigma=0, setting to 1.0 Å")
        sigmas = np.where(sigmas == 0, 1.0, sigmas)
    
    system_params = {
        "charges": jnp.array(charges),
        "masses": jnp.ones(n_atoms) * 12.0,
        "sigmas": jnp.array(sigmas),
        "epsilons": jnp.array(epsilons),
        "bonds": jnp.array(bonds),
        "bond_params": jnp.array(protein.bond_params)[:len(bonds)] if protein.bond_params is not None else jnp.zeros((len(bonds), 2)),
        "angles": jnp.array(angles),
        "angle_params": jnp.array(protein.angle_params) if protein.angle_params is not None else jnp.zeros((0, 2)),
        "dihedrals": jnp.array(protein.proper_dihedrals) if protein.proper_dihedrals is not None else jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.array(protein.dihedral_params) if protein.dihedral_params is not None else jnp.zeros((0, 3)),
        "impropers": jnp.array(protein.impropers) if protein.impropers is not None else jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.array(protein.improper_params) if protein.improper_params is not None else jnp.zeros((0, 3)),
    }
    
    # Solvate
    print("\n--- Adding Explicit Solvent ---")
    solute_radii = sigmas * 0.5
    
    solvated_positions, box_size = solvation.solvate(
        solute_positions=positions,
        solute_radii=solute_radii,
        padding=8.0,
    )
    n_waters = (len(solvated_positions) - n_atoms) // 3
    print(f"Added {n_waters} water molecules")
    print(f"Total atoms: {len(solvated_positions)} (protein: {n_atoms}, water: {n_waters * 3})")
    print(f"Box size: {box_size} Å")
    
    positions = solvated_positions
    n_total = len(positions)
    
    # FIX: Ensure all water molecules are whole (unwrapped internal geometry)
    # The solvate() tiling can leave H/O on opposite sides of PBC.
    positions = fix_water_geometry(positions, box_size, n_atoms, n_waters)
    
    # TIP3P parameters (FIXED: sigma_h = 0.0001 not 0.0)
    water_charge_o = -0.834
    water_charge_h = 0.417
    water_sigma_o = 3.15061
    water_epsilon_o = 0.1521
    water_sigma_h = 0.4     # CHARMM/Modified TIP3P value for stability
    water_epsilon_h = 0.046 # Small repulsion to prevent overlap
    
    water_charges = [water_charge_o, water_charge_h, water_charge_h] * n_waters
    water_sigmas = [water_sigma_o, water_sigma_h, water_sigma_h] * n_waters
    water_epsilons = [water_epsilon_o, water_epsilon_h, water_epsilon_h] * n_waters
    water_masses = [15.999, 1.008, 1.008] * n_waters
    
    system_params["charges"] = jnp.concatenate([system_params["charges"], jnp.array(water_charges)])
    system_params["sigmas"] = jnp.concatenate([system_params["sigmas"], jnp.array(water_sigmas)])
    system_params["epsilons"] = jnp.concatenate([system_params["epsilons"], jnp.array(water_epsilons)])
    system_params["masses"] = jnp.concatenate([system_params["masses"], jnp.array(water_masses)])
    
    # Create TWO sets of system parameters:
    # 1. Minimization: Stiff harmonic bonds/angles for water (mimic rigid constraints)
    # 2. Dynamics: Zero bonds/angles for water (SETTLE handles geometry)
    
    import copy
    system_params_min = copy.deepcopy(system_params)
    system_params_dyn = copy.deepcopy(system_params)
    
    # Add water bonds (same for both min and dyn, needed for topology/exclusions)
    water_bonds = []
    for w in range(n_waters):
        base = n_atoms + w * 3
        water_bonds.append([base, base + 1])  # O-H1
        water_bonds.append([base, base + 2])  # O-H2
    water_bonds = np.array(water_bonds, dtype=np.int32)
    n_water_bonds = len(water_bonds)

    # Add water angles (same for both min and dyn, needed for topology/exclusions)
    water_angles = []
    for w in range(n_waters):
        base = n_atoms + w * 3
        water_angles.append([base + 1, base, base + 2])  # H1-O-H2
    water_angles = np.array(water_angles, dtype=np.int32)
    n_water_angles = len(water_angles)

    # Ensure protein bonds/angles are int32, then concatenate
    protein_bonds = np.array(system_params["bonds"], dtype=jnp.int32)
    protein_angles = np.array(system_params["angles"], dtype=jnp.int32)
    
    # MINIMIZATION PARAMS: Stiff (K=500.0)
    # OpenMM uses constraint solver, we approximate with stiff springs
    # Correct Order: [equilibrium_length, spring_constant]
    w_bonds_min = np.tile([0.9572, 500.0], (n_water_bonds, 1))
    # Correct Order: [equilibrium_angle, spring_constant]
    w_angles_min = np.tile([1.8242, 100.0], (n_water_angles, 1))
    
    system_params_min["bonds"] = jnp.concatenate([protein_bonds, jnp.array(water_bonds, dtype=jnp.int32)])
    system_params_min["bond_params"] = jnp.concatenate([system_params["bond_params"], jnp.array(w_bonds_min)])
    system_params_min["angles"] = jnp.concatenate([protein_angles, jnp.array(water_angles, dtype=jnp.int32)])
    system_params_min["angle_params"] = jnp.concatenate([system_params["angle_params"], jnp.array(w_angles_min)])

    # DYNAMICS PARAMS: Zero (K=0.0)
    # Bonds must exist in topology for exclusions, but K=0 ensures no double-counting with SETTLE
    w_bonds_dyn = np.zeros((n_water_bonds, 2), dtype=np.float32)
    w_angles_dyn = np.zeros((n_water_angles, 2), dtype=np.float32)
    
    system_params_dyn["bonds"] = jnp.concatenate([protein_bonds, jnp.array(water_bonds, dtype=jnp.int32)])
    system_params_dyn["bond_params"] = jnp.concatenate([system_params["bond_params"], jnp.array(w_bonds_dyn)])
    system_params_dyn["angles"] = jnp.concatenate([protein_angles, jnp.array(water_angles, dtype=jnp.int32)])
    system_params_dyn["angle_params"] = jnp.concatenate([system_params["angle_params"], jnp.array(w_angles_dyn)])
    
    # Setup energy function
    print("\n--- Setting Up Energy Function ---")
    displacement_fn, shift_fn = space.periodic(np.array(box_size))
    
    pme_grid_size = max(32, int(np.ceil(np.max(box_size) / 1.0)))
    cutoff = 9.0
    print(f"PME grid: {pme_grid_size}, cutoff: {cutoff}Å")
    
    exclusion_spec = ExclusionSpec.from_system_params(system_params_dyn) # Topology is same
    print(f"Exclusions: {len(exclusion_spec.idx_12_13)} 1-2/1-3, {len(exclusion_spec.idx_14)} 1-4")
    
    neighbor_fn = make_neighbor_list_fn(displacement_fn, np.array(box_size), cutoff)
    neighbor = neighbor_fn.allocate(positions)
    print(f"Neighbor list: {neighbor.idx.shape}")
    
    # MINIMIZATION Energy Function (Stiff Waters)
    energy_fn_min = system.make_energy_fn(
        displacement_fn,
        system_params_min,
        exclusion_spec=exclusion_spec,
        box=jnp.array(box_size),
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=pme_grid_size,
        pme_alpha=0.34,
        cutoff_distance=cutoff,
    )

    # DYNAMICS Energy Function (Zero Waters)
    energy_fn_dyn = system.make_energy_fn(
        displacement_fn,
        system_params_dyn,
        exclusion_spec=exclusion_spec,
        box=jnp.array(box_size),
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=pme_grid_size,
        pme_alpha=0.34,
        cutoff_distance=cutoff,
    )
    
    # Wrap energy function for integrator
    def wrapped_energy_fn(R, **kwargs):
        """Energy with neighbor list management."""
        nbr = kwargs.get("neighbor", neighbor)
        nbr = nbr.update(R)
        return energy_fn(R, neighbor=nbr)
    
    # Minimization uses energy_fn_min
    e_initial = float(energy_fn_min(positions, neighbor=neighbor))
    print(f"Initial energy: {e_initial:.2e} kcal/mol")
    
    # Minimization with molecule-aware PBC wrapping
    print("\n--- Energy Minimization ---")
    
    def minimize_step(pos, nbr):
        nbr = nbr.update(pos)
        grad_fn = jax.grad(lambda r: energy_fn_min(r, neighbor=nbr))
        g = grad_fn(pos)
        g_norm = jnp.linalg.norm(g)
        g = jnp.where(g_norm > 100.0, g * 100.0 / g_norm, g)
        new_pos = pos - 0.001 * g
        # Use molecule-aware wrapping to keep water molecules intact
        new_pos = fix_water_geometry(new_pos, box_size, n_atoms, n_waters)
        return new_pos, nbr
    
    @jax.jit
    def minimize_batch(pos, nbr, n_steps):
        def body_fn(i, carry):
            pos, nbr = carry
            return minimize_step(pos, nbr)
        return jax.lax.fori_loop(0, n_steps, body_fn, (pos, nbr))
    
    print("Running 2000 minimization steps...")
    current_pos = positions
    for batch in range(4):
        current_pos, neighbor = minimize_batch(current_pos, neighbor, 500)
        neighbor = neighbor.update(current_pos)
        e = float(energy_fn_min(current_pos, neighbor=neighbor))
        print(f"  Step {(batch+1)*500}: E = {e:.2e} kcal/mol")
        if not jnp.isfinite(e):
            print("  Energy exploded!")
            break
    
    minimized_pos = current_pos
    e_min = float(energy_fn_min(minimized_pos, neighbor=neighbor))
    print(f"Final minimized energy: {e_min:.2e} kcal/mol")
    
    # MD with SETTLE - Staged Heating Protocol
    print("\n--- MD Simulation with SETTLE (Staged Heating) ---")
    
    water_indices = settle.get_water_indices(n_atoms, n_waters)
    print(f"SETTLE water indices: {water_indices.shape}")
    
    from proxide.physics.constants import BOLTZMANN_KCAL
    from prolix.physics.simulate import NVTLangevinState
    import functools
    
    # Staged heating protocol: gradually increase temperature
    # (Temp, dt, steps, gamma)
    heating_schedule = [
        # Stage 1: Strong friction to drain potential energy release
        (1.0, 0.0002, 500, 100.0),    # 1K, 0.2 fs, 500 steps, gamma=100
        # Stage 2: Moderate friction
        (10.0, 0.0005, 500, 20.0),    # 10K, 0.5 fs, 500 steps, gamma=20
        # Stage 3: Standard heating
        (50.0, 0.001, 100, 1.0),      # 50K, 1.0 fs, 100 steps, gamma=1
        (100.0, 0.001, 100, 1.0),     # 100K, 1.0 fs, 100 steps
        (200.0, 0.002, 100, 1.0),     # 200K, 2.0 fs, 100 steps
        (300.0, 0.002, 500, 1.0),     # 300K, 2.0 fs, 500 steps (production)
    ]
    
    key = random.PRNGKey(42)
    trajectory = [np.array(minimized_pos)]
    energies = []
    temperatures = []
    
    # Initialize with zero momentum (perfect 0K)
    # Initialize with zero momentum (perfect 0K)
    neighbor = neighbor.update(minimized_pos)
    
    # FIX: Convert masses from AMU to internal units
    # 1 kcal/mol = 418.4 J/mol
    # Internal mass unit = AMU / 418.4 to make (kcal/mol, A, ps) consistent
    masses = system_params["masses"] / 418.4
    
    mass_arr = jnp.array(masses)
    if mass_arr.ndim == 0:
        mass_arr = jnp.ones((len(minimized_pos),)) * mass_arr
    mass_state = mass_arr[:, None]
    
    # Dynamics uses energy_fn_dyn
    initial_force = jax.grad(lambda r: -energy_fn_dyn(r, neighbor=neighbor))(minimized_pos)
    
    state = NVTLangevinState(
        position=minimized_pos,
        momentum=jnp.zeros_like(minimized_pos),  # Start at 0K
        force=initial_force,
        mass=mass_state,
        rng=key,
    )
    
    print(f"Starting with zero momentum (0K)")
    
    for stage_idx, (temp_K, dt, n_steps, gamma) in enumerate(heating_schedule):
        kT = BOLTZMANN_KCAL * temp_K
        
        print(f"\n--- Stage {stage_idx + 1}: T={temp_K:.0f}K, dt={dt*1000:.2f}fs, gamma={gamma}, {n_steps} steps ---")
        
        # Create integrator for this temperature/timestep
        # Create integrator with energy_fn_dyn
        _, apply_fn = settle.settle_langevin(
            energy_fn_dyn,
            shift_fn,
            dt=dt,
            kT=kT,
            gamma=gamma,
            mass=masses,
            water_indices=water_indices,
            box=jnp.array(box_size),  # Enable PBC-aware SETTLE
        )
        
        # Use functools.partial to properly bind apply_fn
        apply_step = functools.partial(apply_fn)
        apply_jit = jax.jit(lambda s, nbr: apply_step(s, neighbor=nbr))
        
        # Run equilibration
        for step in range(1, n_steps + 1):
            # Update neighbor list every 10 steps
            if step % 10 == 0:
                neighbor = neighbor.update(state.position)
            
            try:
                state = apply_jit(state, neighbor)
            except Exception as e:
                print(f"  Step {step}: Error - {e}")
                break
            
            if step % 25 == 0 or step == n_steps:
                neighbor = neighbor.update(state.position)
                # Compute Energy using Dynamics function (K=0)
                e = float(energy_fn_dyn(state.position, neighbor=neighbor))
                
                # Compute instantaneous temperature
                ke = 0.5 * jnp.sum(state.momentum**2 / state.mass)
                # Correct DOF for rigid water (6 DOF per water, not 9)
                n_dof = 3 * n_atoms + 6 * n_waters - 3
                inst_temp = float(2 * ke / (n_dof * BOLTZMANN_KCAL))
                
                energies.append(e)
                temperatures.append(inst_temp)
                
                # Check for problems
                if not jnp.all(jnp.isfinite(state.position)):
                    print(f"  Step {step}: NaN/Inf detected!")
                    break
                
                if e > 1e10:
                    print(f"  Step {step}: E={e:.2e} - Energy explosion!")
                    break
                    
                print(f"  Step {step}: E={e:.2f} kcal/mol, T={inst_temp:.1f}K")
        
        # Save trajectory frame
        trajectory.append(np.array(state.position))
        
        # Check for simulation failure
        if not jnp.all(jnp.isfinite(state.position)) or (len(energies) > 0 and energies[-1] > 1e10):
            print("Simulation failed - stopping.")
            break
    
    print(f"\nSimulation complete! {len(trajectory)} frames saved.")
    
    # Check final water geometry
    print("\n--- Water Geometry Check ---")
    final_pos = state.position
    n_check = min(10, n_waters)
    max_oh_err = 0.0
    max_hh_err = 0.0
    for i in range(n_check):
        base = n_atoms + i * 3
        O, H1, H2 = final_pos[base], final_pos[base+1], final_pos[base+2]
        r_OH1 = float(jnp.linalg.norm(H1 - O))
        r_OH2 = float(jnp.linalg.norm(H2 - O))
        r_HH = float(jnp.linalg.norm(H2 - H1))
        max_oh_err = max(max_oh_err, abs(r_OH1 - settle.TIP3P_ROH), abs(r_OH2 - settle.TIP3P_ROH))
        max_hh_err = max(max_hh_err, abs(r_HH - settle.TIP3P_RHH))
    
    print(f"Max O-H error: {max_oh_err:.4f} Å (target: 0.9572)")
    print(f"Max H-H error: {max_hh_err:.4f} Å (target: 1.5139)")
    
    if max_oh_err < 0.02:
        print("✅ Water geometry preserved by SETTLE!")
    else:
        print("⚠️ Water geometry drifted - SETTLE may need refinement")
    
    # Save trajectory for visualization
    np.savez(
        "1uao_explicit_settle_traj.npz",
        positions=np.array(trajectory),
        energies=np.array(energies),
        temperatures=np.array(temperatures),
        box_size=np.array(box_size),
        n_protein_atoms=n_atoms,
        n_waters=n_waters,
    )
    print("\nTrajectory saved to 1uao_explicit_settle_traj.npz")


if __name__ == "__main__":
    main()
