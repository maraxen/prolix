"""Run 1CRN simulation in explicit solvent and generate GIF.

Uses the new Rust-based parsing and parameterization via proxide backend.
Includes energy minimization and molecule-type visualization.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from matplotlib import animation
import matplotlib.pyplot as plt

from proxide.io.parsing.backend import parse_structure
from proxide import OutputSpec, CoordFormat

# Prolix imports
from prolix import simulate
from prolix.physics import system, solvation
from prolix.physics.neighbor_list import ExclusionSpec, make_neighbor_list_fn
from jax_md import space, minimize, partition


def main():
    # 1. Load and parameterize using Rust parser
    pdb_path = "data/pdb/1CRN.pdb"
    ff_path = "proxide/src/proxide/assets/protein.ff19SB.xml"
    
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Missing {pdb_path}")
    if not os.path.exists(ff_path):
        raise FileNotFoundError(f"Force field not found at {ff_path}")

    print(f"Loading {pdb_path}...")
    print(f"Using force field: {ff_path}")
    
    # Parse with MD parameterization and hydrogen addition
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        parameterize_md=True,
        force_field=ff_path,
        add_hydrogens=True,  # Kabsch rotation fix now places H at correct ~1.0 Å distances
        remove_solvent=True,  # Protein only for now
    )
    
    protein = parse_structure(pdb_path, spec)
    
    # Extract coordinates - protein uses flat coordinates
    coords = np.array(protein.coordinates)
    mask = np.array(protein.atom_mask)
    
    # Get valid atom positions and create index mapping
    if coords.ndim == 3:
        flat_coords = coords.reshape(-1, 3)
        flat_mask = mask.reshape(-1)
        valid_indices = np.where(flat_mask > 0.5)[0]
        positions = jnp.array(flat_coords[valid_indices])
        
        # Create old->new index mapping for bonds
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
    else:
        positions = jnp.array(coords)
        valid_indices = np.arange(len(coords))
        old_to_new = {i: i for i in range(len(coords))}
    
    n_atoms = len(positions)
    print(f"Structure loaded: {n_atoms} atoms")
    
    # Get elements for hydrogens (for sizing)
    elements = protein.elements if protein.elements is not None else ['C'] * n_atoms
    if len(elements) > n_atoms:
        elements = [elements[i] for i in valid_indices]
    
    # Build system params - remap bonds to valid indices
    orig_bonds = np.array(protein.bonds) if protein.bonds is not None else np.zeros((0, 2), dtype=np.int32)
    
    # Remap bonds to new indices (filter out bonds with invalid atoms)
    remapped_bonds = []
    for b in orig_bonds:
        if b[0] in old_to_new and b[1] in old_to_new:
            remapped_bonds.append([old_to_new[b[0]], old_to_new[b[1]]])
    bonds = np.array(remapped_bonds, dtype=np.int32) if remapped_bonds else np.zeros((0, 2), dtype=np.int32)
    print(f"Remapped {len(remapped_bonds)} bonds (from {len(orig_bonds)} original)")
    
    angles = np.array(protein.angles) if protein.angles is not None else np.zeros((0, 3), dtype=np.int32)
    
    system_params = {
        "charges": jnp.array(protein.charges),
        "masses": jnp.ones(n_atoms) * 12.0,
        "sigmas": jnp.array(protein.sigmas),
        "epsilons": jnp.array(protein.epsilons),
        "bonds": jnp.array(bonds),
        "bond_params": jnp.array(protein.bond_params)[:len(bonds)] if protein.bond_params is not None else jnp.zeros((len(bonds), 2)),
        "angles": jnp.array(angles),
        "angle_params": jnp.array(protein.angle_params) if protein.angle_params is not None else jnp.zeros((0, 2)),
        "dihedrals": jnp.array(protein.proper_dihedrals) if protein.proper_dihedrals is not None else jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.array(protein.dihedral_params) if protein.dihedral_params is not None else jnp.zeros((0, 3)),
        "impropers": jnp.array(protein.impropers) if protein.impropers is not None else jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.array(protein.improper_params) if protein.improper_params is not None else jnp.zeros((0, 3)),
        # NOTE: We use ExclusionSpec now instead of dense exclusion_mask for neighbor list efficiency
    }
    
    # Get VDW radii from sigmas for solvation (sigma is roughly 2*radius for LJ)
    solute_radii = np.array(protein.sigmas) * 0.5  # sigma/2 as VDW radius
    
    # Solvate with water using our fixed tiling algorithm
    print("\n--- Adding Explicit Solvent ---")
    
    try:
        solvated_positions, box_size = solvation.solvate(
            solute_positions=positions,
            solute_radii=solute_radii,
            padding=8.0,  # Reduced padding to avoid OOM
        )
        n_waters = (len(solvated_positions) - n_atoms) // 3
        print(f"Added {n_waters} water molecules")
        print(f"Total atoms: {len(solvated_positions)} (protein: {n_atoms}, water: {n_waters * 3})")
        print(f"Box size: {box_size} Å")
        
        # Update positions to include waters
        positions = solvated_positions
        n_total = len(positions)
        
        # Extend elements list for waters (O, H, H pattern)
        water_elements = ['O', 'H', 'H'] * n_waters
        elements = list(elements) + water_elements
        
        # Extend system params for waters
        # TIP3P parameters
        water_charge_o = -0.834
        water_charge_h = 0.417
        water_sigma_o = 3.15061  # Angstrom
        water_epsilon_o = 0.1521  # kcal/mol
        water_sigma_h = 0.0  # No LJ on H
        water_epsilon_h = 0.0
        
        water_charges = [water_charge_o, water_charge_h, water_charge_h] * n_waters
        water_sigmas = [water_sigma_o, water_sigma_h, water_sigma_h] * n_waters
        water_epsilons = [water_epsilon_o, water_epsilon_h, water_epsilon_h] * n_waters
        
        # Merge parameters
        system_params["charges"] = jnp.concatenate([system_params["charges"], jnp.array(water_charges)])
        system_params["sigmas"] = jnp.concatenate([system_params["sigmas"], jnp.array(water_sigmas)])
        system_params["epsilons"] = jnp.concatenate([system_params["epsilons"], jnp.array(water_epsilons)])
        system_params["masses"] = jnp.concatenate([system_params["masses"], jnp.ones(n_waters * 3) * 18.0 / 3])
        
        # Add water bonds for exclusion generation (O-H1, O-H2 bonds per water)
        water_bonds = []
        for w in range(n_waters):
            base = n_atoms + w * 3
            water_bonds.append([base, base + 1])  # O-H1
            water_bonds.append([base, base + 2])  # O-H2
        all_bonds = np.concatenate([bonds, np.array(water_bonds, dtype=np.int32)]) if len(water_bonds) > 0 else bonds
        system_params["bonds"] = jnp.array(all_bonds)
        # Pad bond_params for water (water bonds are rigid, but we need params for exclusion logic)
        n_water_bonds = len(water_bonds)
        water_bond_params = np.zeros((n_water_bonds, 2), dtype=np.float32)
        system_params["bond_params"] = jnp.concatenate([system_params["bond_params"], jnp.array(water_bond_params)])
        
        use_explicit_water = True
    except Exception as e:
        print(f"Solvation failed: {e}")
        print("Falling back to vacuum simulation")
        # Compute box size with larger padding (50 Å on each side)
        min_coords = positions.min(axis=0)
        max_coords = positions.max(axis=0)
        box_size = max_coords - min_coords + 100.0
        use_explicit_water = False
    
    print(f"Final box size: {box_size} Å")
    
    # 2. Energy Minimization
    print("\n--- Energy Minimization ---")
    displacement_fn, shift_fn = space.periodic(np.array(box_size))
    
    # Use PME for explicit solvent (efficient O(N log N) electrostatics)
    pme_grid_size = max(32, int(np.ceil(np.max(box_size) / 1.0)))  # ~1Å grid spacing
    cutoff = 9.0  # Angstroms
    print(f"Using PME with grid size: {pme_grid_size}, cutoff: {cutoff}Å")
    
    # Build sparse ExclusionSpec from bonds (replaces N×N exclusion_mask for memory efficiency)
    exclusion_spec = ExclusionSpec.from_system_params(system_params)
    print(f"ExclusionSpec: {len(exclusion_spec.idx_12_13)} 1-2/1-3 pairs, {len(exclusion_spec.idx_14)} 1-4 pairs")
    
    # Create neighbor list for O(N*K) non-bonded computation
    neighbor_fn = make_neighbor_list_fn(displacement_fn, np.array(box_size), cutoff)
    neighbor = neighbor_fn.allocate(positions)
    print(f"Neighbor list allocated: shape {neighbor.idx.shape}")
    
    energy_fn = system.make_energy_fn(
        displacement_fn,
        system_params,
        exclusion_spec=exclusion_spec,  # Use sparse exclusions
        box=jnp.array(box_size),
        use_pbc=True,
        implicit_solvent=False,  # Explicit water, no GBSA
        pme_grid_points=pme_grid_size,
        pme_alpha=0.34,  # Standard Ewald parameter
        cutoff_distance=cutoff,
    )
    
    # Update neighbor list and compute energy
    neighbor = neighbor.update(positions)
    e_initial = float(energy_fn(positions, neighbor=neighbor))
    print(f"Initial energy: {e_initial:.2e} kcal/mol")
    
    # JIT-compiled minimization with neighbor list as part of carried state
    # The neighbor list shape is fixed (N, K), so it can be carried through fori_loop
    
    def minimize_step(pos, nbr):
        """One step of gradient descent with neighbor list update."""
        # Update neighbor list (this is JIT-compatible when shape doesn't change)
        nbr = nbr.update(pos)
        # Compute gradient
        grad_fn = jax.grad(lambda r: energy_fn(r, neighbor=nbr))
        g = grad_fn(pos)
        # Clip gradient for stability
        g_norm = jnp.linalg.norm(g)
        g = jnp.where(g_norm > 100.0, g * 100.0 / g_norm, g)
        step_size = 0.001  # Small step for stability
        new_pos = pos - step_size * g
        # Wrap to PBC
        new_pos = jnp.mod(new_pos, jnp.array(box_size))
        return new_pos, nbr
    
    @jax.jit
    def minimize_n_steps(pos, nbr, n_steps):
        """Run n_steps of minimization in a JIT-compiled loop."""
        def body_fn(i, carry):
            pos, nbr = carry
            new_pos, new_nbr = minimize_step(pos, nbr)
            return (new_pos, new_nbr)
        return jax.lax.fori_loop(0, n_steps, body_fn, (pos, nbr))
    
    print("Running minimization (5000 steps in JIT-compiled batches)...")
    current_pos = positions
    best_energy = e_initial
    best_positions = positions
    
    # Run in batches of 500 to check progress
    for batch in range(10):
        current_pos, neighbor = minimize_n_steps(current_pos, neighbor, 500)
        neighbor = neighbor.update(current_pos)  # Ensure consistent state
        e = float(energy_fn(current_pos, neighbor=neighbor))
        print(f"  Step {(batch+1)*500}: E = {e:.2e} kcal/mol")
        
        if e < best_energy and jnp.isfinite(e):
            best_energy = e
            best_positions = current_pos
        
        # Early stopping if energy explodes
        if e > 1e10 or not jnp.isfinite(e):
            print(f"  Early stopping at batch {batch} due to explosion (reverting to best: {best_energy:.2e})")
            break
    
    minimized_positions = best_positions
    neighbor = neighbor.update(minimized_positions)
    e_final = float(energy_fn(minimized_positions, neighbor=neighbor))
    print(f"Final energy after minimization: {e_final:.2e} kcal/mol")
    
    # 3. Setup MD Simulation
    key = random.PRNGKey(42)
    traj_path = "1crn_explicit_traj.array_record"
    
    sim_spec = simulate.SimulationSpec(
        total_time_ns=0.005,  # 5 ps
        step_size_fs=0.1,  # Reduced to 0.1 fs for stability (no constraints on water)
        save_interval_ns=0.0005,
        save_path=traj_path,
        temperature_k=300.0,
        gamma=1.0,
        box=box_size,
        use_pbc=True,
        pme_grid_size=32,
        # Use neighbor list for O(N*K) efficiency
        use_neighbor_list=True,
        neighbor_cutoff=9.0,
    )
    
    print(f"\nRunning simulation for {sim_spec.total_time_ns*1000:.1f} ps...")
    
    final_state = simulate.run_simulation(
        system_params,  # Pass as positional arg (new API)
        r_init=minimized_positions,
        spec=sim_spec,
        key=key
    )
    print("Simulation complete!")
    print(f"Final Energy: {final_state.potential_energy:.2f} kcal/mol")
    
    # 4. Generate GIF with molecule-type coloring
    print("\nGenerating GIF...")
    generate_enhanced_gif(
        traj_path,
        "outputs/1crn_explicit_movie.gif",
        bonds=bonds,
        elements=elements,
        box_size=np.array(box_size),
    )
    print("GIF saved to 1crn_explicit_movie.gif")


def generate_enhanced_gif(
    traj_path: str,
    output_path: str,
    bonds: np.ndarray,
    elements: list,
    box_size: np.ndarray | None = None,
    fps: int = 15
):
    """Generate an enhanced GIF with bonds and molecule-type coloring."""
    from prolix.visualization import TrajectoryReader
    
    if not os.path.exists(traj_path):
        print(f"Trajectory not found: {traj_path}")
        return
    
    reader = TrajectoryReader(traj_path)
    positions = reader.get_positions()
    n_frames = len(positions)
    n_atoms = positions[0].shape[0]
    
    print(f"Loaded {n_frames} frames, {n_atoms} atoms")
    
    # Determine protein vs water atoms
    # First n_protein atoms are protein, rest are water (O, H, H pattern)
    n_protein = len([e for e in elements if e not in ['O', 'H'] or elements.index(e) < 550])  # Approximate
    # Actually, use the pattern: protein atoms come first, then waters
    # Water starts after protein atoms (first 550 in our case)
    
    colors = []
    sizes = []
    alphas = []
    for i in range(n_atoms):
        elem = elements[i].upper() if i < len(elements) else 'C'
        # Protein atoms (first ~550)
        if i < 550:
            if elem == 'H':
                colors.append('#FFB6C1')  # Light pink for protein H
                sizes.append(8)
                alphas.append(0.95)
            else:
                colors.append('#FF6347')  # Tomato/salmon for protein heavy atoms
                sizes.append(25)
                alphas.append(0.95)
        else:
            # Water atoms - blue with low alpha
            if elem == 'O':
                colors.append('#4169E1')  # Royal blue for water O
                sizes.append(15)
                alphas.append(0.15)
            else:
                colors.append('#87CEEB')  # Sky blue for water H
                sizes.append(5)
                alphas.append(0.1)
    
    # Setup figure with dark background
    fig = plt.figure(figsize=(12, 10), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection="3d", facecolor='#1a1a2e')
    
    # Style axes
    ax.set_xlabel('X (Å)', color='white', fontsize=10)
    ax.set_ylabel('Y (Å)', color='white', fontsize=10)
    ax.set_zlabel('Z (Å)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    
    # Compute bounds - use box_size if provided for proper visualization
    if box_size is not None:
        max_range = np.max(box_size) / 2 * 1.1  # 10% padding
        mid = box_size / 2  # Center on box
    else:
        # Fallback to trajectory-based bounds
        all_pos = np.array(positions)
        min_bound = all_pos.min(axis=(0, 1))
        max_bound = all_pos.max(axis=(0, 1))
        mid = (min_bound + max_bound) / 2
        max_range = (max_bound - min_bound).max() / 2 * 1.2
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_title("1CRN Protein MD Simulation", color='white', fontsize=14, fontweight='bold')
    
    pos = positions[0]
    
    # Draw bonds first (as lines)
    bond_lines = []
    valid_bonds = bonds[(bonds[:, 0] < n_atoms) & (bonds[:, 1] < n_atoms)]
    print(f"Drawing {len(valid_bonds)} bonds")
    
    for b in valid_bonds:
        line = ax.plot(
            [pos[b[0], 0], pos[b[1], 0]],
            [pos[b[0], 1], pos[b[1], 1]],
            [pos[b[0], 2], pos[b[1], 2]],
            color='#4a4a6a', linewidth=0.8, alpha=0.7
        )[0]
        bond_lines.append(line)
    
    # Draw atoms (use RGBA for per-atom alpha)
    import matplotlib.colors as mcolors
    rgba_colors = [mcolors.to_rgba(c, a) for c, a in zip(colors, alphas)]
    scat = ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=rgba_colors, s=sizes, edgecolors='none'
    )
    
    # Frame text and energy
    txt = fig.text(0.02, 0.95, f"Frame 0/{n_frames}", color='white', fontsize=12)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6347', label='Protein (heavy atoms)'),
        Patch(facecolor='#FFB6C1', label='Protein (hydrogen)'),
        Patch(facecolor='#4169E1', alpha=0.3, label='Water'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
              facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    
    def update(frame_idx):
        pos = positions[frame_idx]
        
        # Update atoms
        scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        
        # Update bonds
        for i, b in enumerate(valid_bonds):
            bond_lines[i].set_data_3d(
                [pos[b[0], 0], pos[b[1], 0]],
                [pos[b[0], 1], pos[b[1], 1]],
                [pos[b[0], 2], pos[b[1], 2]]
            )
        
        txt.set_text(f"Frame {frame_idx + 1}/{n_frames}")
        return [scat, txt] + bond_lines
    
    print("Rendering animation...")
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/fps)
    writer = animation.PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
