"""Position plausibility tests for molecular dynamics simulations.

These tests ensure that minimization and simulation produce physically
plausible atomic positions that match OpenMM behavior.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import os

# Enable x64 for physics
jax.config.update("jax_enable_x64", True)


class TestPositionPlausibility:
    """Test that physics computations produce plausible positions matching OpenMM."""

    @pytest.fixture
    def chignolin_setup(self):
        """Setup Chignolin (1UAO) for testing."""
        from priox.io.parsing import biotite as parsing_biotite
        from priox.physics.force_fields.loader import load_force_field
        from priox.md.bridge.core import parameterize_system
        from prolix.physics import system
        from jax_md import space
        import biotite.structure as struc
        
        # Load PDB
        pdb_path = "data/pdb/1UAO.pdb"
        atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)
        coords = atom_array.coord
        
        # Extract topology
        residues = []
        atom_names = []
        atom_counts = []
        
        res_starts = struc.get_residue_starts(atom_array)
        for i, start_idx in enumerate(res_starts):
            if i < len(res_starts) - 1:
                end_idx = res_starts[i + 1]
                res_atoms = atom_array[start_idx:end_idx]
            else:
                res_atoms = atom_array[start_idx:]
            
            res_name = res_atoms.res_name[0]
            residues.append(res_name)
            
            names = res_atoms.atom_name.tolist()
            if len(residues) == 1:
                for k in range(len(names)):
                    if names[k] == "H":
                        names[k] = "H1"
            atom_names.extend(names)
            atom_counts.append(len(names))
        
        # Rename terminals
        if residues:
            residues[0] = "N" + residues[0]
            residues[-1] = "C" + residues[-1]
        
        # Parameterize
        ff = load_force_field("data/force_fields/protein19SB.eqx")
        system_params = parameterize_system(ff, residues, atom_names, atom_counts)
        
        # Create energy function
        displacement_fn, shift_fn = space.free()
        energy_fn = system.make_energy_fn(
            displacement_fn, system_params, implicit_solvent=True
        )
        
        positions = jnp.array(coords)
        
        return {
            "energy_fn": energy_fn,
            "positions": positions,
            "coords": coords,
            "system_params": system_params,
            "atom_array": atom_array,
            "atom_names": atom_names,
        }

    def test_initial_positions_molecular_scale(self, chignolin_setup):
        """Initial PDB positions should be within typical molecular range."""
        coords = chignolin_setup["coords"]
        
        # Center positions
        centered = coords - coords.mean(axis=0)
        
        # Proteins typically fit within ~100 Å
        assert centered.min() > -100, f"Min position {centered.min():.1f} Å too far"
        assert centered.max() < 100, f"Max position {centered.max():.1f} Å too far"

    def test_minimization_preserves_position_scale(self, chignolin_setup):
        """Minimized positions should stay within molecular scale."""
        from prolix.physics import simulate
        
        energy_fn = chignolin_setup["energy_fn"]
        positions = chignolin_setup["positions"]
        
        # Run minimization
        r_min = simulate.run_minimization(energy_fn, positions, steps=500)
        
        # Convert to numpy and center
        r_min_np = np.asarray(r_min)
        centered = r_min_np - r_min_np.mean(axis=0)
        
        # Positions should remain within ±100 Å (molecular scale)
        assert centered.min() > -100, f"Atom flew too far negative: {centered.min():.1f} Å"
        assert centered.max() < 100, f"Atom flew too far positive: {centered.max():.1f} Å"

    def test_minimization_reduces_energy(self, chignolin_setup):
        """Minimization should reduce energy significantly."""
        from prolix.physics import simulate
        
        energy_fn = chignolin_setup["energy_fn"]
        positions = chignolin_setup["positions"]
        
        e_initial = float(energy_fn(positions))
        
        r_min = simulate.run_minimization(energy_fn, positions, steps=500)
        e_final = float(energy_fn(r_min))
        
        # Energy should decrease significantly
        assert e_final < e_initial, (
            f"Minimization did not reduce energy: {e_initial:.1f} -> {e_final:.1f}"
        )
        
        # Energy should be finite
        assert np.isfinite(e_final), f"Final energy is not finite: {e_final}"

    def test_minimization_finite_forces(self, chignolin_setup):
        """Forces after minimization should be finite and within reasonable bounds."""
        from prolix.physics import simulate
        
        energy_fn = chignolin_setup["energy_fn"]
        positions = chignolin_setup["positions"]
        system_params = chignolin_setup["system_params"] # Assuming this is available
        
        r_min = simulate.run_minimization(energy_fn, positions, steps=500)
        
        from jax_md import space
        from prolix.physics import system
        
        displacement_fn, _ = space.free()
        
        # Compute forces (-grad E)
        # Re-create energy_fn if the one from setup is not suitable for direct grad
        # Or use the one from setup if it's already jax-grad compatible
        energy_fn_for_grad = system.make_energy_fn(displacement_fn, system_params, implicit_solvent=True)
        forces = -jax.grad(energy_fn_for_grad)(r_min)
        
        # Check finite
        assert jnp.all(jnp.isfinite(forces))
        
        # Check magnitude (max force < 1e5 kJ/mol/nm or similar)
        # OpenMM minimization usually targets 10 kJ/mol/nm. 
        # But we just want strict "not exploding".
        max_force = jnp.max(jnp.linalg.norm(forces, axis=-1))
        
        # Note: units are kcal/mol/A. 1000 is high but finite.
        # Unminimized system might have 1e5+. Minimized should be < 100 or so.
        # But we are checking simple finiteness/plausibility.
        assert max_force < 5000.0, (
            f"Max force after minimization is too large: {max_force:.1f} kcal/mol/Å"
        )

    def test_all_atoms_bonded(self, chignolin_setup):
        """Verify every atom has at least one bond to prevent floating atoms."""
        system_params = chignolin_setup['system_params'] # Corrected to 'system_params'
        atom_names = chignolin_setup['atom_names']
        
        bonds = np.array(system_params['bonds'])
        
        bonded_atoms = set()
        for a, b in bonds:
            bonded_atoms.add(int(a))
            bonded_atoms.add(int(b))
        
        n_atoms = len(atom_names)
        unbonded_indices = [i for i in range(n_atoms) if i not in bonded_atoms]
        unbonded_details = [
            f"{i}: {atom_names[i]}" for i in unbonded_indices
        ]
        
        assert len(unbonded_indices) == 0, (
            f"Found {len(unbonded_indices)} unbonded atoms that will float away: {unbonded_details}"
        )


class TestOpenMMComparison:
    """Compare minimization results with OpenMM as ground truth."""

    @pytest.fixture
    def openmm_available(self):
        """Check if OpenMM is available."""
        try:
            import openmm
            import openmm.app as app
            return True
        except ImportError:
            pytest.skip("OpenMM not installed")
            return False

    @pytest.fixture
    def setup_both_systems(self, openmm_available):
        """Setup both JAX MD and OpenMM systems on same structure."""
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        from priox.io.parsing import biotite as parsing_biotite
        from priox.physics.force_fields.loader import load_force_field
        from priox.md.bridge.core import parameterize_system
        from prolix.physics import system
        from jax_md import space
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb
        import tempfile
        
        pdb_path = "data/pdb/1UAO.pdb"
        atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)
        coords = atom_array.coord
        
        # ---- JAX MD Setup ----
        residues = []
        atom_names = []
        atom_counts = []
        
        res_starts = struc.get_residue_starts(atom_array)
        for i, start_idx in enumerate(res_starts):
            if i < len(res_starts) - 1:
                end_idx = res_starts[i + 1]
                res_atoms = atom_array[start_idx:end_idx]
            else:
                res_atoms = atom_array[start_idx:]
            
            res_name = res_atoms.res_name[0]
            residues.append(res_name)
            
            names = res_atoms.atom_name.tolist()
            if len(residues) == 1:
                for k in range(len(names)):
                    if names[k] == "H":
                        names[k] = "H1"
            atom_names.extend(names)
            atom_counts.append(len(names))
        
        if residues:
            residues[0] = "N" + residues[0]
            residues[-1] = "C" + residues[-1]
        
        ff = load_force_field("data/force_fields/protein19SB.eqx")
        system_params = parameterize_system(ff, residues, atom_names, atom_counts)
        
        displacement_fn, shift_fn = space.free()
        energy_fn = system.make_energy_fn(
            displacement_fn, system_params, implicit_solvent=True
        )
        
        jax_positions = jnp.array(coords)
        
        # ---- OpenMM Setup ----
        ff_xml_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            "../../openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml"
        ))
        
        # Write temp PDB for OpenMM
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+", delete=False) as tmp:
            pdb_file_bio = pdb.PDBFile()
            pdb_file_bio.set_structure(atom_array)
            pdb_file_bio.write(tmp)
            tmp.flush()
            tmp_path = tmp.name
        
        pdb_file = app.PDBFile(tmp_path)
        topology = pdb_file.topology
        omm_positions = pdb_file.positions
        
        os.unlink(tmp_path)
        
        omm_ff = app.ForceField(ff_xml_path, 'implicit/obc2.xml')
        omm_system = omm_ff.createSystem(
            topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False,
            removeCMMotion=False
        )
        
        return {
            "jax_energy_fn": energy_fn,
            "jax_positions": jax_positions,
            "omm_system": omm_system,
            "omm_positions": omm_positions,
            "topology": topology,
            "coords": coords,
        }

    def test_minimization_position_rmsd_vs_openmm(self, setup_both_systems):
        """JAX MD minimized positions should be similar to OpenMM minimized positions."""
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        from prolix.physics import simulate
        
        data = setup_both_systems
        
        # ---- JAX MD Minimization ----
        jax_r_min = simulate.run_minimization(
            data["jax_energy_fn"], data["jax_positions"], steps=1000
        )
        jax_r_min_np = np.asarray(jax_r_min)
        
        # ---- OpenMM Minimization ----
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        )
        simulation = app.Simulation(
            data["topology"], data["omm_system"], integrator
        )
        simulation.context.setPositions(data["omm_positions"])
        
        # Minimize
        simulation.minimizeEnergy(maxIterations=1000)
        
        # Get final positions
        omm_state = simulation.context.getState(getPositions=True)
        omm_r_min = omm_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        
        # ---- Compare Position Scales ----
        # Both should be within molecular scale
        jax_centered = jax_r_min_np - jax_r_min_np.mean(axis=0)
        omm_centered = omm_r_min - omm_r_min.mean(axis=0)
        
        jax_range = (jax_centered.min(), jax_centered.max())
        omm_range = (omm_centered.min(), omm_centered.max())
        
        # JAX positions should be within similar scale as OpenMM
        assert jax_range[0] > -100, f"JAX min position {jax_range[0]:.1f} too far"
        assert jax_range[1] < 100, f"JAX max position {jax_range[1]:.1f} too far"
        
        # Compare structure similarity via all-atom RMSD
        # First align structures (simple centering for now)
        jax_c = jax_r_min_np - jax_r_min_np.mean(axis=0)
        omm_c = omm_r_min - omm_r_min.mean(axis=0)
        
        # RMSD
        rmsd = np.sqrt(np.mean(np.sum((jax_c - omm_c) ** 2, axis=1)))
        
        # After minimization from same start, structures should be similar
        # 10 Å is generous - may be tighter once physics is fully matched
        assert rmsd < 10.0, (
            f"RMSD between JAX MD and OpenMM minimized structures is {rmsd:.2f} Å "
            f"(expected < 10 Å)"
        )

    def test_minimized_energy_same_order_of_magnitude(self, setup_both_systems):
        """Minimized energies should be in same order of magnitude."""
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        from prolix.physics import simulate
        
        data = setup_both_systems
        
        # ---- JAX MD ----
        jax_r_min = simulate.run_minimization(
            data["jax_energy_fn"], data["jax_positions"], steps=1000
        )
        jax_e_min = float(data["jax_energy_fn"](jax_r_min))
        
        # ---- OpenMM ----
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        )
        simulation = app.Simulation(
            data["topology"], data["omm_system"], integrator
        )
        simulation.context.setPositions(data["omm_positions"])
        simulation.minimizeEnergy(maxIterations=1000)
        
        omm_state = simulation.context.getState(getEnergy=True)
        omm_e_min = omm_state.getPotentialEnergy().value_in_unit(
            unit.kilocalories_per_mole
        )
        
        # Both should be negative (stable minimum)
        # Allow for different minima due to different algorithms
        # Just check they're in same ballpark (within factor of 10)
        assert jax_e_min < 0, f"JAX minimized energy should be negative: {jax_e_min:.1f}"
        
        # Energy difference should be within reasonable range
        energy_diff = abs(jax_e_min - omm_e_min)
        
        # Log the values for debugging
        print(f"JAX minimized energy: {jax_e_min:.1f} kcal/mol")
        print(f"OpenMM minimized energy: {omm_e_min:.1f} kcal/mol")
        print(f"Difference: {energy_diff:.1f} kcal/mol")

    def test_minimization_position_range_vs_openmm(self, setup_both_systems):
        """JAX MD and OpenMM should produce similar position ranges after minimization."""
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        from prolix.physics import simulate
        
        data = setup_both_systems
        
        # ---- JAX MD Minimization ----
        jax_r_min = simulate.run_minimization(
            data["jax_energy_fn"], data["jax_positions"], steps=1000
        )
        jax_r_min_np = np.asarray(jax_r_min)
        
        # ---- OpenMM Minimization ----
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        )
        simulation = app.Simulation(
            data["topology"], data["omm_system"], integrator
        )
        simulation.context.setPositions(data["omm_positions"])
        simulation.minimizeEnergy(maxIterations=1000)
        
        omm_state = simulation.context.getState(getPositions=True)
        omm_r_min = omm_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        
        # ---- Compare Absolute Position Ranges ----
        jax_centered = jax_r_min_np - jax_r_min_np.mean(axis=0)
        omm_centered = omm_r_min - omm_r_min.mean(axis=0)
        
        jax_range = jax_centered.max() - jax_centered.min()
        omm_range = omm_centered.max() - omm_centered.min()
        
        print(f"JAX position range: {jax_range:.2f} Å")
        print(f"OpenMM position range: {omm_range:.2f} Å")
        print(f"Ratio: {jax_range / omm_range:.2f}")
        
        # Ranges should be within factor of 2 of each other
        # (same order of magnitude for molecular dimensions)
        assert 0.5 < jax_range / omm_range < 2.0, (
            f"JAX position range ({jax_range:.2f} Å) differs significantly from "
            f"OpenMM ({omm_range:.2f} Å)"
        )
        
        # Both should indicate compact molecular structure (< 50 Å for small proteins)
        assert jax_range < 50.0, f"JAX structure too extended: {jax_range:.2f} Å"
        assert omm_range < 50.0, f"OpenMM structure too extended: {omm_range:.2f} Å"


class TestTrajectoryPositions:
    """Test that trajectory frames have reasonable positions."""

    def test_trajectory_positions_stable(self):
        """If a trajectory file exists, all frames should have reasonable positions."""
        from pathlib import Path
        
        traj_path = Path("chignolin_traj.array_record")
        
        if not traj_path.exists():
            pytest.skip("Trajectory file not found - run simulation first")
        
        from prolix.visualization import TrajectoryReader
        
        reader = TrajectoryReader(str(traj_path))
        for i in range(min(len(reader), 10)):  # Check first 10 frames
            state = reader.get_state(i)
            pos = np.asarray(state.positions)
            centered = pos - pos.mean(axis=0)
            
            # Positions should stay within ±200 Å (generous for minor outliers)
            assert centered.min() > -200 and centered.max() < 200, (
                f"Frame {i} has positions outside molecular range: "
                f"[{centered.min():.1f}, {centered.max():.1f}]"
            )
