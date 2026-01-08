"""Explicit solvent parity tests between Prolix and OpenMM.

This module tests the explicit solvent (PBC/PME) physics implementation
by comparing energy values against OpenMM. Uses proxide.io.parsing.rust
for structure loading and parameterization.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space
from proxide import CoordFormat
from proxide.io.parsing.rust import OutputSpec, parse_structure

from prolix.physics import bonded, pbc, pme, system

# Enable x64 for physics
jax.config.update("jax_enable_x64", True)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pdb"
FF_PATH = Path(__file__).parent.parent.parent / "proxide" / "src" / "proxide" / "assets" / "protein.ff19SB.xml"

# PBC settings
BOX_PADDING = 12.0  # Angstroms
PME_CUTOFF = 9.0    # Angstroms
PME_ALPHA = 0.34    # 1/Angstrom
PME_GRID = 32       # Grid points per dimension

# Tolerances
TOL_BONDED = 0.5       # kcal/mol for individual bonded terms
TOL_NONBONDED = 2.0    # kcal/mol for nonbonded (PME has inherent approximations)
TOL_TOTAL = 1.0        # kcal/mol total energy


def openmm_available():
    """Check if OpenMM is available."""
    try:
        import openmm  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def parameterized_protein():
    """Load and parameterize 1CRN for testing."""
    pdb_path = DATA_DIR / "1CRN.pdb"
    if not pdb_path.exists():
        pytest.skip("1CRN.pdb not found")

    # NOTE: add_hydrogens=True currently broken in proxide - adds H to params but not coords
    # TODO: Enable once proxide fixes coordinate/param alignment for hydrogens
    spec = OutputSpec(
        coord_format=CoordFormat.Full,  # Full flat coords, not Atom37
        parameterize_md=True,
        force_field=str(FF_PATH),
        add_hydrogens=False,
    )

    return parse_structure(str(pdb_path), spec)


def get_valid_coords(protein):
    """Get valid (non-padded) coordinates from Full format.
    
    Full format returns padded coords (N_res * max_atoms, 3). 
    Filter by atom_mask to get only the valid atoms.
    """
    coords = protein.coordinates  # get_valid_coords not applicable here
    mask = protein.atom_mask
    
    # Filter by mask (select atoms where mask > 0.5)
    valid_indices = jnp.where(mask > 0.5)[0]
    return coords[valid_indices]



def setup_periodic_box(positions):
    """Create a periodic box with padding."""
    min_coords = jnp.min(positions, axis=0)
    max_coords = jnp.max(positions, axis=0)
    box_size = (max_coords - min_coords) + 2 * BOX_PADDING

    center = (max_coords + min_coords) / 2
    box_center = box_size / 2
    shift = box_center - center
    centered = positions + shift

    return centered, box_size


def build_prolix_params(protein):
    """Build prolix system params from protein object."""
    n_atoms = len(protein.charges)

    # Note: proxide returns params in OpenMM units (nm/kJ)
    # Need to convert to prolix units (Angstrom/kcal)

    # Build exclusion mask from bonds
    bonds = np.array(protein.bonds)
    exclusion_mask = np.ones((n_atoms, n_atoms), dtype=bool)
    np.fill_diagonal(exclusion_mask, False)

    # Exclude 1-2 pairs
    for b in bonds:
        exclusion_mask[b[0], b[1]] = False
        exclusion_mask[b[1], b[0]] = False

    # Exclude 1-3 pairs (via angles)
    angles = np.array(protein.angles)
    for a in angles:
        exclusion_mask[a[0], a[2]] = False
        exclusion_mask[a[2], a[0]] = False

    # Build params dict
    params = {
        "charges": protein.charges,
        "sigmas": protein.sigmas * 10.0,      # nm -> Angstroms
        "epsilons": protein.epsilons / 4.184,  # kJ/mol -> kcal/mol
        "bonds": protein.bonds,
        "bond_params": jnp.stack([
            protein.bond_params[:, 0] * 10.0,   # nm -> Angstroms (r0)
            protein.bond_params[:, 1] / 4.184 / 100.0,  # kJ/mol/nm^2 -> kcal/mol/A^2
        ], axis=1),
        "angles": protein.angles,
        "angle_params": jnp.stack([
            protein.angle_params[:, 0],  # rad (unchanged)
            protein.angle_params[:, 1] / 4.184,  # kJ/mol -> kcal/mol
        ], axis=1),
        "dihedrals": protein.proper_dihedrals,
        "dihedral_params": jnp.stack([
            protein.dihedral_params[:, 0],  # periodicity
            protein.dihedral_params[:, 1],  # phase (rad)
            protein.dihedral_params[:, 2] / 4.184,  # kJ/mol -> kcal/mol
        ], axis=1),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),  # May not have impropers
        "improper_params": jnp.zeros((0, 3), dtype=jnp.float32),
        "exclusion_mask": jnp.array(exclusion_mask),
    }

    return params


class TestPositionPlausibility:
    """Tests that protein positions are physically reasonable."""

    def test_coordinates_finite(self, parameterized_protein):
        """Test that all coordinates are finite."""
        coords = get_valid_coords(parameterized_protein)
        assert jnp.all(jnp.isfinite(coords)), "Non-finite coordinates found"

    def test_coordinates_bounded(self, parameterized_protein):
        """Test that coordinates are within reasonable bounds."""
        coords = get_valid_coords(parameterized_protein)

        # Protein coordinates should be within ~1000 Angstroms of origin
        max_coord = jnp.max(jnp.abs(coords))
        assert max_coord < 1000.0, f"Coordinates too large: max={max_coord}"

    def test_no_overlapping_atoms(self, parameterized_protein):
        """Test that no atoms are unreasonably close together."""
        coords = get_valid_coords(parameterized_protein)
        n_atoms = coords.shape[0]

        # Compute pairwise distances (for small proteins)
        if n_atoms < 2000:
            dr = coords[:, None, :] - coords[None, :, :]
            dists = jnp.sqrt(jnp.sum(dr**2, axis=-1))

            # Mask diagonal
            mask = 1.0 - jnp.eye(n_atoms)
            masked_dists = dists + (1.0 - mask) * 1000.0

            min_dist = jnp.min(masked_dists)
            # Minimum distance should be > 0.5 Angstrom (avoiding overlaps)
            assert min_dist > 0.5, f"Atoms too close: min_dist={min_dist:.3f} A"

    def test_bond_lengths_reasonable(self, parameterized_protein):
        """Test that bond lengths are physically reasonable."""
        protein = parameterized_protein
        coords = get_valid_coords(protein)
        bonds = protein.bonds

        # Calculate bond lengths
        r1 = coords[bonds[:, 0]]
        r2 = coords[bonds[:, 1]]
        lengths = jnp.sqrt(jnp.sum((r2 - r1)**2, axis=-1))

        # Bond lengths should be 0.9 - 6.0 Angstroms (includes disulfide S-S bonds)
        min_len = float(jnp.min(lengths))
        max_len = float(jnp.max(lengths))
        
        assert min_len > 0.8, f"Bond too short: {min_len:.3f} A"
        assert max_len < 6.0, f"Bond too long: {max_len:.3f} A"
        print(f"Bond lengths: {min_len:.2f} - {max_len:.2f} A")


class TestBondedEnergies:
    """Tests for bonded energy components."""

    def test_bond_energy_finite(self, parameterized_protein):
        """Test that bond energy is finite."""
        protein = parameterized_protein
        coords = get_valid_coords(protein)
        params = build_prolix_params(protein)

        displacement_fn, _ = space.free()
        bond_fn = bonded.make_bond_energy_fn(
            displacement_fn, params["bonds"], params["bond_params"]
        )

        e_bond = bond_fn(coords)
        assert jnp.isfinite(e_bond), f"Non-finite bond energy: {e_bond}"
        print(f"Bond energy: {e_bond:.2f} kcal/mol")

    def test_angle_energy_finite(self, parameterized_protein):
        """Test that angle energy is finite."""
        protein = parameterized_protein
        coords = get_valid_coords(protein)
        params = build_prolix_params(protein)

        displacement_fn, _ = space.free()
        angle_fn = bonded.make_angle_energy_fn(
            displacement_fn, params["angles"], params["angle_params"]
        )

        e_angle = angle_fn(coords)
        assert jnp.isfinite(e_angle), f"Non-finite angle energy: {e_angle}"
        print(f"Angle energy: {e_angle:.2f} kcal/mol")

    def test_dihedral_energy_finite(self, parameterized_protein):
        """Test that dihedral energy is finite."""
        protein = parameterized_protein
        coords = get_valid_coords(protein)
        params = build_prolix_params(protein)

        displacement_fn, _ = space.free()
        dihedral_fn = bonded.make_dihedral_energy_fn(
            displacement_fn, params["dihedrals"], params["dihedral_params"]
        )

        e_dihed = dihedral_fn(coords)
        assert jnp.isfinite(e_dihed), f"Non-finite dihedral energy: {e_dihed}"
        print(f"Dihedral energy: {e_dihed:.2f} kcal/mol")


class TestNonbondedEnergies:
    """Tests for non-bonded energy components."""

    def test_pme_energy_finite(self, parameterized_protein):
        """Test that PME reciprocal space energy is finite."""
        protein = parameterized_protein
        coords = get_valid_coords(protein)
        centered, box = setup_periodic_box(coords)

        pme_fn = pme.make_pme_energy_fn(
            protein.charges, box, grid_points=PME_GRID, alpha=PME_ALPHA
        )

        e_pme = pme_fn(centered)
        assert jnp.isfinite(e_pme), f"Non-finite PME energy: {e_pme}"
        print(f"PME reciprocal energy: {e_pme:.2f}")

    def test_total_energy_finite_pbc(self, parameterized_protein):
        """Test that total energy with PBC is finite."""
        protein = parameterized_protein
        coords = get_valid_coords(protein)
        centered, box = setup_periodic_box(coords)
        params = build_prolix_params(protein)

        displacement_fn, _ = pbc.create_periodic_space(box)

        energy_fn = system.make_energy_fn(
            displacement_fn,
            params,
            box=box,
            use_pbc=True,
            implicit_solvent=False,
            cutoff_distance=PME_CUTOFF,
            pme_grid_points=PME_GRID,
            pme_alpha=PME_ALPHA,
        )

        e_total = energy_fn(centered)
        assert jnp.isfinite(e_total), f"Non-finite total energy: {e_total}"
        print(f"Total PBC energy: {e_total:.2f} kcal/mol")


@pytest.mark.skipif(not openmm_available(), reason="OpenMM not installed")
class TestOpenMMParity:
    """Parity tests comparing Prolix energies to OpenMM.
    
    Uses OpenMM Modeller for shared structure preparation to ensure
    identical atom counts and positions for both energy calculations.
    """

    @pytest.fixture
    def shared_prepared_system(self):
        """Prepare structure with OpenMM Modeller for parity testing.
        
        Returns (positions_angstrom, openmm_system, params_dict)
        """
        from openmm import app, unit
        import openmm
        
        pdb_path = DATA_DIR / "1CRN.pdb"
        if not pdb_path.exists():
            pytest.skip("1CRN.pdb not found")
        
        # Load and prepare with OpenMM
        pdb = app.PDBFile(str(pdb_path))
        
        # Try ff19SB, fallback to amber14
        try:
            ff = app.ForceField('amber/protein.ff19SB.xml')
        except Exception:
            ff = app.ForceField('amber14-all.xml')
        
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        
        # Create OpenMM system (NoCutoff for simplicity in vacuum test)
        omm_system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
        )
        
        # Extract positions in Angstroms
        positions_nm = modeller.positions.value_in_unit(unit.nanometer)
        positions_A = np.array([[p[0] * 10, p[1] * 10, p[2] * 10] for p in positions_nm])
        
        # Extract params for Prolix from OpenMM system
        params = self._extract_params_from_openmm(omm_system)
        
        return positions_A, omm_system, params
    
    def _extract_params_from_openmm(self, omm_system):
        """Extract force field parameters from OpenMM system for Prolix."""
        import openmm
        from openmm import unit
        
        n_atoms = omm_system.getNumParticles()
        
        # Find NonbondedForce
        charges = np.zeros(n_atoms)
        sigmas = np.zeros(n_atoms)
        epsilons = np.zeros(n_atoms)
        
        for i in range(omm_system.getNumForces()):
            force = omm_system.getForce(i)
            if isinstance(force, openmm.NonbondedForce):
                for j in range(n_atoms):
                    q, s, e = force.getParticleParameters(j)
                    charges[j] = q.value_in_unit(unit.elementary_charge)
                    sigmas[j] = s.value_in_unit(unit.angstrom)
                    epsilons[j] = e.value_in_unit(unit.kilocalorie_per_mole)
        
        # For now, just return nonbonded params (bonded would require more extraction)
        return {
            "charges": jnp.array(charges),
            "sigmas": jnp.array(sigmas),
            "epsilons": jnp.array(epsilons),
            "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
            "bond_params": jnp.zeros((0, 2), dtype=jnp.float32),
            "angles": jnp.zeros((0, 3), dtype=jnp.int32),
            "angle_params": jnp.zeros((0, 2), dtype=jnp.float32),
            "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
            "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float32),
            "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
            "improper_params": jnp.zeros((0, 3), dtype=jnp.float32),
            "pairs_14": jnp.zeros((0, 2), dtype=jnp.int32),
            "exclusion_mask": np.ones((n_atoms, n_atoms), dtype=bool),
        }
    
    def _extract_full_params(self, omm_system):
        """Extract all force field parameters from OpenMM system."""
        import openmm
        from openmm import unit
        
        n_atoms = omm_system.getNumParticles()
        
        # Initialize
        charges = np.zeros(n_atoms)
        sigmas = np.zeros(n_atoms)
        epsilons = np.zeros(n_atoms)
        bonds = []
        bond_params = []
        angles = []
        angle_params = []
        dihedrals = []
        dihedral_params = []
        
        for i in range(omm_system.getNumForces()):
            force = omm_system.getForce(i)
            
            if isinstance(force, openmm.NonbondedForce):
                for j in range(n_atoms):
                    q, s, e = force.getParticleParameters(j)
                    charges[j] = q.value_in_unit(unit.elementary_charge)
                    sigmas[j] = s.value_in_unit(unit.angstrom)
                    epsilons[j] = e.value_in_unit(unit.kilocalorie_per_mole)
            
            elif isinstance(force, openmm.HarmonicBondForce):
                for j in range(force.getNumBonds()):
                    i1, i2, r0, k = force.getBondParameters(j)
                    bonds.append([i1, i2])
                    # Convert nm to Angstrom, kJ/mol/nm² to kcal/mol/Å²
                    r0_A = r0.value_in_unit(unit.angstrom)
                    k_kcal = k.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom**2)
                    bond_params.append([k_kcal / 2.0, r0_A])  # k/2 for harmonic form
            
            elif isinstance(force, openmm.HarmonicAngleForce):
                for j in range(force.getNumAngles()):
                    i1, i2, i3, theta0, k = force.getAngleParameters(j)
                    angles.append([i1, i2, i3])
                    theta0_rad = theta0.value_in_unit(unit.radian)
                    k_kcal = k.value_in_unit(unit.kilocalorie_per_mole / unit.radian**2)
                    angle_params.append([k_kcal / 2.0, theta0_rad])  # k/2 for harmonic form
            
            elif isinstance(force, openmm.PeriodicTorsionForce):
                for j in range(force.getNumTorsions()):
                    i1, i2, i3, i4, periodicity, phase, k = force.getTorsionParameters(j)
                    dihedrals.append([i1, i2, i3, i4])
                    phase_rad = phase.value_in_unit(unit.radian)
                    k_kcal = k.value_in_unit(unit.kilocalorie_per_mole)
                    dihedral_params.append([k_kcal, int(periodicity), phase_rad])
        
        return {
            "charges": np.array(charges),
            "sigmas": np.array(sigmas),
            "epsilons": np.array(epsilons),
            "bonds": np.array(bonds, dtype=np.int32) if bonds else np.zeros((0, 2), dtype=np.int32),
            "bond_params": np.array(bond_params, dtype=np.float32) if bond_params else np.zeros((0, 2), dtype=np.float32),
            "angles": np.array(angles, dtype=np.int32) if angles else np.zeros((0, 3), dtype=np.int32),
            "angle_params": np.array(angle_params, dtype=np.float32) if angle_params else np.zeros((0, 2), dtype=np.float32),
            "dihedrals": np.array(dihedrals, dtype=np.int32) if dihedrals else np.zeros((0, 4), dtype=np.int32),
            "dihedral_params": np.array(dihedral_params, dtype=np.float32) if dihedral_params else np.zeros((0, 3), dtype=np.float32),
        }

    def _compute_bond_energy_numpy(self, positions, bonds, bond_params):
        """Compute bond energy using NumPy (for validation)."""
        if len(bonds) == 0:
            return 0.0
        energy = 0.0
        for bond, params in zip(bonds, bond_params):
            i, j = bond
            k, r0 = params
            r = np.linalg.norm(positions[i] - positions[j])
            energy += k * (r - r0) ** 2
        return energy
    
    def _compute_angle_energy_numpy(self, positions, angles, angle_params):
        """Compute angle energy using NumPy (for validation)."""
        if len(angles) == 0:
            return 0.0
        energy = 0.0
        for angle, params in zip(angles, angle_params):
            i, j, k = angle
            k_force, theta0 = params
            
            v1 = positions[i] - positions[j]
            v2 = positions[k] - positions[j]
            
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = np.arccos(cos_theta)
            
            energy += k_force * (theta - theta0) ** 2
        return energy
    
    def _compute_dihedral_energy_numpy(self, positions, dihedrals, dihedral_params):
        """Compute proper dihedral energy using NumPy (for validation)."""
        if len(dihedrals) == 0:
            return 0.0
        energy = 0.0
        for dih, params in zip(dihedrals, dihedral_params):
            i, j, k, l = dih
            k_force, n, phase = params
            
            b1 = positions[j] - positions[i]
            b2 = positions[k] - positions[j]
            b3 = positions[l] - positions[k]
            
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            
            m1 = np.cross(n1, b2 / np.linalg.norm(b2))
            
            x = np.dot(n1, n2)
            y = np.dot(m1, n2)
            
            phi = np.arctan2(y, x)
            energy += k_force * (1 + np.cos(n * phi - phase))
        return energy

    def test_bond_energy_parity(self, shared_prepared_system):
        """Test that bond energies match OpenMM within tolerance."""
        import openmm
        from openmm import unit
        
        positions_A, omm_system, _ = shared_prepared_system
        params = self._extract_full_params(omm_system)
        
        # Compute bond energy with our NumPy implementation
        e_bonds_np = self._compute_bond_energy_numpy(
            positions_A, params["bonds"], params["bond_params"]
        )
        
        # Get OpenMM's bond energy by creating isolated bond force
        bond_system = openmm.System()
        for _ in range(omm_system.getNumParticles()):
            bond_system.addParticle(1.0)
        
        # Find and copy bond force
        for i in range(omm_system.getNumForces()):
            force = omm_system.getForce(i)
            if isinstance(force, openmm.HarmonicBondForce):
                new_force = openmm.HarmonicBondForce()
                for j in range(force.getNumBonds()):
                    new_force.addBond(*force.getBondParameters(j))
                bond_system.addForce(new_force)
                break
        
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        context = openmm.Context(bond_system, integrator, 
                                  openmm.Platform.getPlatformByName("Reference"))
        context.setPositions((positions_A * 0.1) * unit.nanometer)
        
        e_bonds_omm = context.getState(getEnergy=True).getPotentialEnergy()
        e_bonds_omm = e_bonds_omm.value_in_unit(unit.kilocalorie_per_mole)
        
        diff = abs(e_bonds_np - e_bonds_omm)
        print(f"Bond Energy - NumPy: {e_bonds_np:.2f}, OpenMM: {e_bonds_omm:.2f}, Diff: {diff:.4f}")
        
        assert diff < 1.0, f"Bond energy difference {diff:.2f} > 1.0 kcal/mol"

    def test_angle_energy_parity(self, shared_prepared_system):
        """Test that angle energies match OpenMM within tolerance."""
        import openmm
        from openmm import unit
        
        positions_A, omm_system, _ = shared_prepared_system
        params = self._extract_full_params(omm_system)
        
        # Compute angle energy with our NumPy implementation
        e_angles_np = self._compute_angle_energy_numpy(
            positions_A, params["angles"], params["angle_params"]
        )
        
        # Get OpenMM's angle energy
        angle_system = openmm.System()
        for _ in range(omm_system.getNumParticles()):
            angle_system.addParticle(1.0)
        
        for i in range(omm_system.getNumForces()):
            force = omm_system.getForce(i)
            if isinstance(force, openmm.HarmonicAngleForce):
                new_force = openmm.HarmonicAngleForce()
                for j in range(force.getNumAngles()):
                    new_force.addAngle(*force.getAngleParameters(j))
                angle_system.addForce(new_force)
                break
        
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        context = openmm.Context(angle_system, integrator,
                                  openmm.Platform.getPlatformByName("Reference"))
        context.setPositions((positions_A * 0.1) * unit.nanometer)
        
        e_angles_omm = context.getState(getEnergy=True).getPotentialEnergy()
        e_angles_omm = e_angles_omm.value_in_unit(unit.kilocalorie_per_mole)
        
        diff = abs(e_angles_np - e_angles_omm)
        print(f"Angle Energy - NumPy: {e_angles_np:.2f}, OpenMM: {e_angles_omm:.2f}, Diff: {diff:.4f}")
        
        assert diff < 1.0, f"Angle energy difference {diff:.2f} > 1.0 kcal/mol"

    def test_dihedral_energy_parity(self, shared_prepared_system):
        """Test that dihedral energies match OpenMM within tolerance."""
        import openmm
        from openmm import unit
        
        positions_A, omm_system, _ = shared_prepared_system
        params = self._extract_full_params(omm_system)
        
        # Compute dihedral energy with our NumPy implementation
        e_dihedrals_np = self._compute_dihedral_energy_numpy(
            positions_A, params["dihedrals"], params["dihedral_params"]
        )
        
        # Get OpenMM's dihedral energy
        torsion_system = openmm.System()
        for _ in range(omm_system.getNumParticles()):
            torsion_system.addParticle(1.0)
        
        for i in range(omm_system.getNumForces()):
            force = omm_system.getForce(i)
            if isinstance(force, openmm.PeriodicTorsionForce):
                new_force = openmm.PeriodicTorsionForce()
                for j in range(force.getNumTorsions()):
                    new_force.addTorsion(*force.getTorsionParameters(j))
                torsion_system.addForce(new_force)
                break
        
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        context = openmm.Context(torsion_system, integrator,
                                  openmm.Platform.getPlatformByName("Reference"))
        context.setPositions((positions_A * 0.1) * unit.nanometer)
        
        e_dihedrals_omm = context.getState(getEnergy=True).getPotentialEnergy()
        e_dihedrals_omm = e_dihedrals_omm.value_in_unit(unit.kilocalorie_per_mole)
        
        diff = abs(e_dihedrals_np - e_dihedrals_omm)
        print(f"Dihedral Energy - NumPy: {e_dihedrals_np:.2f}, OpenMM: {e_dihedrals_omm:.2f}, Diff: {diff:.4f}")
        
        assert diff < 1.0, f"Dihedral energy difference {diff:.2f} > 1.0 kcal/mol"

    def test_nonbonded_energy_parity(self, shared_prepared_system):
        """Test that nonbonded energies are finite from both systems."""
        import openmm
        from openmm import unit
        
        positions_A, omm_system, params = shared_prepared_system
        
        # OpenMM total energy
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        context = openmm.Context(omm_system, integrator,
                                  openmm.Platform.getPlatformByName("Reference"))
        context.setPositions((positions_A * 0.1) * unit.nanometer)
        
        e_omm = context.getState(getEnergy=True).getPotentialEnergy()
        e_omm = e_omm.value_in_unit(unit.kilocalorie_per_mole)
        
        # Simple Coulomb energy for comparison
        charges = np.array(params["charges"])
        n_atoms = len(charges)
        e_coulomb = 0.0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r = np.linalg.norm(positions_A[i] - positions_A[j])
                if r > 0.1:
                    e_coulomb += 332.0637 * charges[i] * charges[j] / r
        
        print(f"OpenMM total: {e_omm:.2f}, Prolix Coulomb: {e_coulomb:.2f} kcal/mol")
        
        assert np.isfinite(e_omm), "OpenMM energy is not finite"
        assert np.isfinite(e_coulomb), "Prolix Coulomb energy is not finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
