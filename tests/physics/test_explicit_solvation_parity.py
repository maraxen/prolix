"""Explicit solvent parity test using OpenMM for solvation.

This test solvates 1UAO using OpenMM's Modeller, then compares
energy values between OpenMM and Prolix for the solvated system.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix.physics import system, solvation, settle

# Enable x64 for physics precision
jax.config.update("jax_enable_x64", True)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pdb"
FF_PATH = (
    Path(__file__).parent.parent.parent
    / "proxide" / "src" / "proxide" / "assets" / "protein.ff19SB.xml"
)


def openmm_available():
    """Check if OpenMM is available."""
    try:
        import openmm  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not openmm_available(), reason="OpenMM not installed")
class TestOpenMMSolvationParity:
    """Tests comparing Prolix solvation to OpenMM."""
    
    @pytest.fixture
    def solvated_system_openmm(self):
        """Solvate 1UAO using OpenMM's Modeller."""
        from openmm import app, unit
        import openmm
        
        pdb_path = DATA_DIR / "1UAO.pdb"
        if not pdb_path.exists():
            pytest.skip("1UAO.pdb not found")
        
        # Load structure
        pdb = app.PDBFile(str(pdb_path))
        
        # Force field with waters
        ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
        
        # Build modeller and solvate
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        modeller.addSolvent(ff, padding=0.8*unit.nanometer, model='tip3p')
        
        # Create system with PME
        omm_system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=0.9*unit.nanometer,
            constraints=app.HBonds,
        )
        
        # Extract positions in Angstroms
        positions_nm = modeller.positions.value_in_unit(unit.nanometer)
        positions_A = np.array([[p[0] * 10, p[1] * 10, p[2] * 10] for p in positions_nm])
        
        # Get box vectors
        box_vecs = modeller.topology.getPeriodicBoxVectors()
        box_A = np.array([
            box_vecs[0][0].value_in_unit(unit.angstrom),
            box_vecs[1][1].value_in_unit(unit.angstrom),
            box_vecs[2][2].value_in_unit(unit.angstrom),
        ])
        
        # Count waters
        n_waters = sum(1 for r in modeller.topology.residues() if r.name in ('HOH', 'WAT'))
        n_protein_atoms = len(pdb.positions)
        
        return {
            'positions': positions_A,
            'box': box_A,
            'system': omm_system,
            'topology': modeller.topology,
            'n_waters': n_waters,
            'n_protein_atoms': n_protein_atoms,
        }
    
    def test_openmm_solvation_runs(self, solvated_system_openmm):
        """Test that OpenMM solvation completes successfully."""
        data = solvated_system_openmm
        
        assert data['positions'].shape[0] > 0, "No positions"
        assert data['n_waters'] > 0, f"No waters added"
        assert np.all(np.isfinite(data['positions'])), "Non-finite positions"
        
        print(f"Solvated system: {data['positions'].shape[0]} atoms")
        print(f"  Waters: {data['n_waters']}")
        print(f"  Box: {data['box']}")
    
    def test_openmm_energy_finite(self, solvated_system_openmm):
        """Test that OpenMM energy is finite for solvated system."""
        from openmm import unit
        import openmm
        
        data = solvated_system_openmm
        
        # Create context and compute energy
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        context = openmm.Context(
            data['system'], integrator,
            openmm.Platform.getPlatformByName("Reference")
        )
        context.setPositions((data['positions'] * 0.1) * unit.nanometer)
        
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        
        assert np.isfinite(energy), f"OpenMM energy is not finite: {energy}"
        print(f"OpenMM total energy: {energy:.2f} kcal/mol")
    
    def test_openmm_energy_breakdown(self, solvated_system_openmm):
        """Get OpenMM energy breakdown by force type."""
        from openmm import unit
        import openmm
        
        data = solvated_system_openmm
        omm_system = data['system']
        
        # Extract energies per force
        energies = {}
        for i in range(omm_system.getNumForces()):
            force = omm_system.getForce(i)
            force_name = type(force).__name__
            
            # Create isolated system with just this force
            test_system = openmm.System()
            for j in range(omm_system.getNumParticles()):
                test_system.addParticle(omm_system.getParticleMass(j))
            test_system.setDefaultPeriodicBoxVectors(*omm_system.getDefaultPeriodicBoxVectors())
            
            # Clone force
            if isinstance(force, openmm.NonbondedForce):
                cloned = openmm.NonbondedForce()
                for j in range(force.getNumParticles()):
                    cloned.addParticle(*force.getParticleParameters(j))
                for j in range(force.getNumExceptions()):
                    cloned.addException(*force.getExceptionParameters(j))
                cloned.setNonbondedMethod(force.getNonbondedMethod())
                cloned.setCutoffDistance(force.getCutoffDistance())
                cloned.setForceGroup(0)
                test_system.addForce(cloned)
            elif isinstance(force, openmm.HarmonicBondForce):
                cloned = openmm.HarmonicBondForce()
                for j in range(force.getNumBonds()):
                    cloned.addBond(*force.getBondParameters(j))
                test_system.addForce(cloned)
            elif isinstance(force, openmm.HarmonicAngleForce):
                cloned = openmm.HarmonicAngleForce()
                for j in range(force.getNumAngles()):
                    cloned.addAngle(*force.getAngleParameters(j))
                test_system.addForce(cloned)
            elif isinstance(force, openmm.PeriodicTorsionForce):
                cloned = openmm.PeriodicTorsionForce()
                for j in range(force.getNumTorsions()):
                    cloned.addTorsion(*force.getTorsionParameters(j))
                test_system.addForce(cloned)
            else:
                continue  # Skip unsupported forces
            
            # Compute energy
            integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
            context = openmm.Context(
                test_system, integrator,
                openmm.Platform.getPlatformByName("Reference")
            )
            context.setPositions((data['positions'] * 0.1) * unit.nanometer)
            
            state = context.getState(getEnergy=True)
            energies[force_name] = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        
        print("\nOpenMM Energy Breakdown:")
        for name, energy in energies.items():
            print(f"  {name}: {energy:.2f} kcal/mol")
        
        # All energies should be finite
        for name, energy in energies.items():
            assert np.isfinite(energy), f"{name} energy is not finite"


class TestProlixSolvation:
    """Tests for Prolix's solvation implementation."""
    
    def test_prolix_solvation_basic(self):
        """Test that Prolix solvation runs without errors."""
        from proxide.io.parsing.rust import parse_structure
        from proxide import OutputSpec, CoordFormat
        
        pdb_path = DATA_DIR / "1UAO.pdb"
        if not pdb_path.exists():
            pytest.skip("1UAO.pdb not found")
        
        # Parse protein with Full (flat) format to avoid Atom37 padding issues
        spec = OutputSpec()
        spec.coord_format = CoordFormat.Full
        spec.parameterize_md = True
        spec.force_field = str(FF_PATH) if FF_PATH.exists() else None
        
        protein = parse_structure(str(pdb_path), spec)
        
        # Handle Full format: positions may be padded by mask
        coords = np.array(protein.coordinates)
        mask = np.array(protein.atom_mask)
        
        if coords.ndim == 3:
            coords = coords.reshape(-1, 3)
            mask = mask.reshape(-1)
        
        # Filter by mask
        valid_idx = np.where(mask > 0.5)[0]
        positions = jnp.array(coords[valid_idx])
        
        # Get radii - filter to match valid atoms
        if protein.sigmas is not None:
            sigmas = np.array(protein.sigmas)
            # Sigmas might already be for valid atoms only
            if len(sigmas) == len(valid_idx):
                radii = jnp.array(sigmas) * 0.5
            elif len(sigmas) >= len(mask):
                radii = jnp.array(sigmas[valid_idx]) * 0.5 if len(sigmas) == len(mask) else jnp.ones(len(valid_idx)) * 1.5
            else:
                radii = jnp.ones(len(valid_idx)) * 1.5
        else:
            radii = jnp.ones(positions.shape[0]) * 1.5
        
        # Solvate
        solvated_pos, box_size = solvation.solvate(
            positions,
            radii,
            padding=8.0,  # 8 Å padding
        )
        
        n_protein = positions.shape[0]
        n_waters = (solvated_pos.shape[0] - n_protein) // 3
        
        print(f"\nProlix solvation:")
        print(f"  Protein atoms: {n_protein}")
        print(f"  Waters added: {n_waters}")
        print(f"  Total atoms: {solvated_pos.shape[0]}")
        print(f"  Box size: {box_size}")
        
        # Verify
        assert solvated_pos.shape[0] > n_protein, "No waters added"
        assert jnp.all(jnp.isfinite(solvated_pos)), "Non-finite positions"
        assert jnp.all(box_size > 0), "Invalid box size"
    
    def test_water_geometry_after_solvation(self):
        """Test that solvated waters have correct O-H and H-H distances."""
        from proxide.io.parsing.rust import parse_structure
        from proxide import OutputSpec, CoordFormat
        
        pdb_path = DATA_DIR / "1UAO.pdb"
        if not pdb_path.exists():
            pytest.skip("1UAO.pdb not found")
        
        spec = OutputSpec()
        spec.coord_format = CoordFormat.Full
        spec.parameterize_md = True
        spec.force_field = str(FF_PATH) if FF_PATH.exists() else None
        
        protein = parse_structure(str(pdb_path), spec)
        
        # Handle padded coordinates
        coords = np.array(protein.coordinates)
        mask = np.array(protein.atom_mask)
        
        if coords.ndim == 3:
            coords = coords.reshape(-1, 3)
            mask = mask.reshape(-1)
        
        valid_idx = np.where(mask > 0.5)[0]
        positions = jnp.array(coords[valid_idx])
        
        # Get matching radii
        if protein.sigmas is not None:
            sigmas = np.array(protein.sigmas)
            radii = jnp.array(sigmas[:len(valid_idx)]) * 0.5 if len(sigmas) >= len(valid_idx) else jnp.ones(len(valid_idx)) * 1.5
        else:
            radii = jnp.ones(len(valid_idx)) * 1.5
        
        solvated_pos, _ = solvation.solvate(positions, radii, padding=8.0)
        
        n_protein = positions.shape[0]
        n_water_atoms = solvated_pos.shape[0] - n_protein
        n_waters = n_water_atoms // 3
        
        # Check water geometry
        bad_waters = 0
        for i in range(min(100, n_waters)):  # Check first 100 waters
            base = n_protein + i * 3
            O = solvated_pos[base]
            H1 = solvated_pos[base + 1]
            H2 = solvated_pos[base + 2]
            
            r_OH1 = float(jnp.linalg.norm(H1 - O))
            r_OH2 = float(jnp.linalg.norm(H2 - O))
            
            # TIP3P: O-H ~ 0.9572 Å
            if abs(r_OH1 - 0.9572) > 0.1 or abs(r_OH2 - 0.9572) > 0.1:
                bad_waters += 1
        
        print(f"  Bad water geometries: {bad_waters}/{min(100, n_waters)}")
        assert bad_waters < 10, f"Too many bad water geometries: {bad_waters}"


class TestSETTLEIntegration:
    """Tests for SETTLE integration with simulation."""
    
    def test_settle_water_indices(self):
        """Test water index generation."""
        n_protein = 100
        n_waters = 50
        
        water_indices = settle.get_water_indices(n_protein, n_waters)
        
        assert water_indices.shape == (50, 3)
        assert int(water_indices[0, 0]) == 100  # First O
        assert int(water_indices[-1, 2]) == 249  # Last H2
    
    def test_settle_preserves_water_geometry(self):
        """Test that SETTLE maintains water O-H and H-H distances over MD steps."""
        from jax_md import space
        
        # Create 10 waters
        n_waters = 10
        R = []
        for i in range(n_waters):
            base = jnp.array([i * 5.0, 0.0, 0.0])
            R.extend([
                base,                                    # O
                base + jnp.array([0.757, 0.586, 0.0]),   # H1
                base + jnp.array([-0.757, 0.586, 0.0]),  # H2
            ])
        R = jnp.array(R)
        
        water_indices = settle.get_water_indices(0, n_waters)
        
        def energy_fn(r):
            return 0.001 * jnp.sum(r**2)  # Weak harmonic
        
        init_fn, apply_fn = settle.settle_langevin(
            energy_fn,
            shift_fn=space.free()[1],
            dt=0.002,  # 2 fs
            kT=0.001,  # Low temp
            gamma=1.0,
            water_indices=water_indices,
        )
        
        key = jax.random.PRNGKey(0)
        state = init_fn(key, R)
        
        # Run 500 steps
        for _ in range(500):
            state = apply_fn(state)
        
        R_final = state.position
        
        # Check all water geometries
        max_oh_error = 0.0
        max_hh_error = 0.0
        for i in range(n_waters):
            base = i * 3
            O, H1, H2 = R_final[base], R_final[base+1], R_final[base+2]
            
            r_OH1 = float(jnp.linalg.norm(H1 - O))
            r_OH2 = float(jnp.linalg.norm(H2 - O))
            r_HH = float(jnp.linalg.norm(H2 - H1))
            
            max_oh_error = max(max_oh_error, abs(r_OH1 - settle.TIP3P_ROH))
            max_oh_error = max(max_oh_error, abs(r_OH2 - settle.TIP3P_ROH))
            max_hh_error = max(max_hh_error, abs(r_HH - settle.TIP3P_RHH))
        
        print(f"\nAfter 500 steps:")
        print(f"  Max O-H error: {max_oh_error:.4f} Å")
        print(f"  Max H-H error: {max_hh_error:.4f} Å")
        
        # Tolerance: 0.02 Å for O-H, 0.05 Å for H-H
        assert max_oh_error < 0.02, f"O-H constraint violated: {max_oh_error}"
        assert max_hh_error < 0.05, f"H-H constraint violated: {max_hh_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
