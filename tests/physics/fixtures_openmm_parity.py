"""OpenMM reference fixtures for bonded-energy parity tests.

Provides:
- build_ala_dip_openmm_system(): Construct OpenMM alanine dipeptide with ForceGroups
- extract_bonded_params(omm_system): Extract bonded parameters from OpenMM System
- get_openmm_per_term_energies(omm_system, positions_ang): Per-ForceGroup energies
- get_openmm_forces(omm_system, positions_ang): Per-atom forces
- build_prolix_bonded_system(params, positions_ang): Bridge to PhysicsSystem

Unit conversions: OpenMM SI → AKMA-like (Å, kcal/mol, rad)
"""

import numpy as np
import pytest

try:
    import openmm
    from openmm import app, unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

from prolix.physics.types import PhysicsSystem
from prolix.physics import bonded
from jax_md import space
import jax.numpy as jnp
import jax


def _kj_to_kcal(x):
    """Convert kJ/mol to kcal/mol."""
    return x / 4.184


def _nm_to_ang(x):
    """Convert nm to Angstrom."""
    return x * 10.0


def build_ala_dip_openmm_system():
    """Build OpenMM system for alanine dipeptide (ACE-ALA-NME).

    Constructs the system by loading a simple PDB or creating from scratch.

    Returns:
        omm_system: OpenMM System object with ForceGroups assigned
        positions_ang: (N, 3) float64 array in Angstroms
        topology: OpenMM Topology
    """
    if not HAS_OPENMM:
        raise ImportError("OpenMM not available")

    # Create a simple alanine dipeptide structure as a PDB string
    # This avoids topology/bonding issues from manual construction
    pdb_content = """HEADER
REMARK Alanine dipeptide (ACE-ALA-NME)
ATOM      1  C   ACE A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  O   ACE A   1       1.200   0.000   0.000  1.00  0.00           O
ATOM      3  CH3 ACE A   1      -0.600   1.200   0.000  1.00  0.00           C
ATOM      4  N   ALA A   2      -0.600  -1.200   0.000  1.00  0.00           N
ATOM      5  CA  ALA A   2      -2.000  -1.200   0.000  1.00  0.00           C
ATOM      6  C   ALA A   2      -2.600  -2.400   0.000  1.00  0.00           C
ATOM      7  O   ALA A   2      -3.800  -2.400   0.000  1.00  0.00           O
ATOM      8  CB  ALA A   2      -2.600  -0.000   0.000  1.00  0.00           C
ATOM      9  N   NME A   3      -1.800  -3.600   0.000  1.00  0.00           N
ATOM     10  CH3 NME A   3      -2.400  -4.800   0.000  1.00  0.00           C
END
"""

    # Parse PDB string
    from io import StringIO
    pdb_file = app.PDBFile(StringIO(pdb_content))

    ff = app.ForceField('amber14-all.xml')
    modeller = app.Modeller(pdb_file.topology, pdb_file.positions)

    # Add hydrogens
    modeller.addHydrogens(ff)

    # Create system with ForceGroups for bonded terms
    omm_system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )

    # Assign ForceGroups: 0=bonds, 1=angles, 2=dihedrals, 3+=other (nonbonded)
    force_group_map = {
        openmm.HarmonicBondForce: 0,
        openmm.HarmonicAngleForce: 1,
        openmm.PeriodicTorsionForce: 2,  # Both proper and improper for v1
        openmm.NonbondedForce: 3,  # Isolate LJ + Coulomb
    }

    for force in omm_system.getForces():
        force_type = type(force)
        if force_type in force_group_map:
            force.setForceGroup(force_group_map[force_type])

    # Minimize to get sensible starting geometry
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(omm_system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(modeller.positions)

    # Energy minimization
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=100, tolerance=1.0)

    state = context.getState(getPositions=True)
    positions_omm = state.getPositions()
    positions_ang = np.array(positions_omm.value_in_unit(unit.angstrom), dtype=np.float64)

    return omm_system, positions_ang, modeller.topology


def extract_bonded_params(omm_system):
    """Extract bonded parameters from OpenMM System.

    Returns dict with keys:
        bonds, bond_params
        angles, angle_params
        dihedrals, dihedral_params
        impropers, improper_params (all PeriodicTorsion entries for v1)
    All parameters converted to AKMA-like units (Å, kcal/mol, rad).
    """
    if not HAS_OPENMM:
        raise ImportError("OpenMM not available")

    data = {
        'bonds': [], 'bond_params': [],
        'angles': [], 'angle_params': [],
        'dihedrals': [], 'dihedral_params': [],
        'impropers': [], 'improper_params': [],
    }

    for force in omm_system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, l, k = force.getBondParameters(i)
                data['bonds'].append([p1, p2])
                # length: nm -> Å, k: kJ/mol/nm² -> kcal/mol/Å²
                # Unit conversion: 1 nm = 10 Å, so 1 nm² = 100 Å²
                # X kJ/mol/nm² = X / (4.184 * 100) kcal/mol/Å² = X / 418.4
                l_ang = _nm_to_ang(l.value_in_unit(unit.nanometer))
                k_kj_nm2 = k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
                k_kcal_ang2 = k_kj_nm2 / 418.4
                data['bond_params'].append([l_ang, k_kcal_ang2])

        elif isinstance(force, openmm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                p1, p2, p3, a, k = force.getAngleParameters(i)
                data['angles'].append([p1, p2, p3])
                # angle: rad (no conversion), k: kJ/mol/rad² -> kcal/mol/rad²
                a_rad = a.value_in_unit(unit.radian)
                k_kj_rad2 = k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
                k_kcal_rad2 = _kj_to_kcal(k_kj_rad2)
                data['angle_params'].append([a_rad, k_kcal_rad2])

        elif isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = force.getTorsionParameters(i)
                # For v1: treat all as one "torsion" group; no proper/improper split
                data['dihedrals'].append([p1, p2, p3, p4])
                # per: periodicity (int), phase: rad, k: kJ/mol -> kcal/mol
                per_f = float(per)
                phase_rad = phase.value_in_unit(unit.radian)
                k_kcal = _kj_to_kcal(k.value_in_unit(unit.kilojoule_per_mole))
                data['dihedral_params'].append([per_f, phase_rad, k_kcal])

    # Convert to numpy arrays
    for key in data:
        if len(data[key]) > 0:
            data[key] = np.array(data[key], dtype=np.float64)
        else:
            # Empty array with correct shape
            if 'param' in key:
                if 'bond' in key:
                    data[key] = np.zeros((0, 2), dtype=np.float64)
                elif 'angle' in key:
                    data[key] = np.zeros((0, 2), dtype=np.float64)
                else:  # dihedral/improper
                    data[key] = np.zeros((0, 3), dtype=np.float64)
            else:  # indices
                if 'bond' in key:
                    data[key] = np.zeros((0, 2), dtype=np.int32)
                elif 'angle' in key:
                    data[key] = np.zeros((0, 3), dtype=np.int32)
                else:
                    data[key] = np.zeros((0, 4), dtype=np.int32)

    return data


def get_openmm_per_term_energies(omm_system, positions_ang):
    """Get per-term (per-ForceGroup) energies from OpenMM.

    Returns dict with keys: 'bonds', 'angles', 'dihedrals', 'impropers' (all zeros for v1).
    Values in kcal/mol.
    """
    if not HAS_OPENMM:
        raise ImportError("OpenMM not available")

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(omm_system, integrator, openmm.Platform.getPlatformByName("Reference"))

    # Set positions in nm
    context.setPositions(positions_ang / 10.0 * unit.nanometer)

    energies = {}

    # ForceGroup 0: bonds
    state = context.getState(getEnergy=True, groups={0})
    e_bonds_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    energies['bonds'] = _kj_to_kcal(e_bonds_kj)

    # ForceGroup 1: angles
    state = context.getState(getEnergy=True, groups={1})
    e_angles_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    energies['angles'] = _kj_to_kcal(e_angles_kj)

    # ForceGroup 2: dihedrals (both proper and improper for v1)
    state = context.getState(getEnergy=True, groups={2})
    e_dih_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    energies['dihedrals'] = _kj_to_kcal(e_dih_kj)

    # ForceGroup 3: improper (separate if assigned; for v1 all in group 2)
    energies['impropers'] = 0.0

    return energies


def get_openmm_forces(omm_system, positions_ang):
    """Get per-atom forces from OpenMM.

    Returns (N, 3) array in kcal/mol/Å.
    """
    if not HAS_OPENMM:
        raise ImportError("OpenMM not available")

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(omm_system, integrator, openmm.Platform.getPlatformByName("Reference"))

    # Set positions in nm
    context.setPositions(positions_ang / 10.0 * unit.nanometer)

    state = context.getState(getForces=True)
    forces_omm = state.getForces()

    # Convert from kJ/mol/nm to kcal/mol/Å
    forces_kj_nm = np.array(forces_omm.value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    forces_kcal_ang = forces_kj_nm * 0.1 / 4.184  # nm^-1 -> Å^-1, kJ -> kcal

    return forces_kcal_ang


def build_prolix_bonded_system(bonded_params, positions_ang):
    """Build minimal prolix PhysicsSystem with bonded terms only.

    Args:
        bonded_params: dict from extract_bonded_params()
        positions_ang: (N, 3) float64 array in Angstroms

    Returns:
        system: PhysicsSystem with bonded fields populated, nonbonded zeroed
        displacement_fn: jax_md displacement function
    """
    n_atoms = positions_ang.shape[0]

    # Create displacement function for free boundary
    displacement_fn, _ = space.free()

    # Convert positions to JAX array (enable x64 for float64)
    jax.config.update("jax_enable_x64", True)
    positions = jnp.array(positions_ang, dtype=jnp.float64)

    # Create system with bonded terms, nonbonded zeroed
    system = PhysicsSystem(
        positions=positions,
        # Nonbonded fields zeroed
        charges=jnp.zeros(n_atoms, dtype=jnp.float64),
        sigmas=jnp.ones(n_atoms, dtype=jnp.float64) * 1e-6,  # ~zero
        epsilons=jnp.ones(n_atoms, dtype=jnp.float64) * 1e-6,  # ~zero
        # Bonded terms from extraction
        bonds=jnp.array(bonded_params['bonds'], dtype=jnp.int32),
        bond_params=jnp.array(bonded_params['bond_params'], dtype=jnp.float64),
        angles=jnp.array(bonded_params['angles'], dtype=jnp.int32),
        angle_params=jnp.array(bonded_params['angle_params'], dtype=jnp.float64),
        dihedrals=jnp.array(bonded_params['dihedrals'], dtype=jnp.int32),
        dihedral_params=jnp.array(bonded_params['dihedral_params'], dtype=jnp.float64),
        impropers=jnp.array(bonded_params.get('impropers', np.zeros((0, 4))), dtype=jnp.int32),
        improper_params=jnp.array(bonded_params.get('improper_params', np.zeros((0, 3))), dtype=jnp.float64),
        # Mandatory fields
        radii=jnp.ones(n_atoms, dtype=jnp.float64),
        scaled_radii=jnp.ones(n_atoms, dtype=jnp.float64),
        masses=jnp.ones(n_atoms, dtype=jnp.float64),
        element_ids=jnp.zeros(n_atoms, dtype=jnp.int32),
        atom_mask=jnp.ones(n_atoms, dtype=bool),
        is_hydrogen=jnp.zeros(n_atoms, dtype=bool),
        is_backbone=jnp.zeros(n_atoms, dtype=bool),
        is_heavy=jnp.ones(n_atoms, dtype=bool),
        protein_atom_mask=jnp.ones(n_atoms, dtype=bool),
        water_atom_mask=jnp.zeros(n_atoms, dtype=bool),
        bond_mask=jnp.ones(bonded_params['bonds'].shape[0], dtype=bool),
        angle_mask=jnp.ones(bonded_params['angles'].shape[0], dtype=bool),
        dihedral_mask=jnp.ones(bonded_params['dihedrals'].shape[0], dtype=bool),
        improper_mask=jnp.ones(bonded_params.get('impropers', np.zeros((0, 4))).shape[0], dtype=bool),
    )

    return system, displacement_fn


def get_prolix_per_term_energies(system, displacement_fn, positions_ang):
    """Get per-term prolix bonded energies.

    Uses the direct-path bonded energy factories (params at call time).

    Returns dict with keys: 'bonds', 'angles', 'dihedrals', 'impropers'.
    Values in kcal/mol.
    """
    # Enable x64 for float64 support
    jax.config.update("jax_enable_x64", True)

    positions = jnp.array(positions_ang, dtype=jnp.float64)
    energies = {}

    # Bonds
    if system.bonds is not None and system.bonds.shape[0] > 0:
        bond_fn = bonded.make_bond_energy_fn(displacement_fn, system.bonds)
        energies['bonds'] = float(bond_fn(positions, system.bond_params))
    else:
        energies['bonds'] = 0.0

    # Angles
    if system.angles is not None and system.angles.shape[0] > 0:
        angle_fn = bonded.make_angle_energy_fn(displacement_fn, system.angles)
        energies['angles'] = float(angle_fn(positions, system.angle_params))
    else:
        energies['angles'] = 0.0

    # Dihedrals (proper + improper for v1)
    if system.dihedrals is not None and system.dihedrals.shape[0] > 0:
        dih_fn = bonded.make_dihedral_energy_fn(displacement_fn, system.dihedrals)
        energies['dihedrals'] = float(dih_fn(positions, system.dihedral_params))
    else:
        energies['dihedrals'] = 0.0

    # Impropers (v1: all in dihedrals)
    energies['impropers'] = 0.0

    return energies


@pytest.fixture(scope="module")
def ala_dip_reference():
    """Module-scoped fixture: OpenMM alanine dipeptide reference.

    Returns:
        {
            'omm_system': OpenMM System,
            'positions_ang': (N, 3) float64 Angstroms,
            'bonded_params': dict from extract_bonded_params(),
            'energies': dict from get_openmm_per_term_energies(),
            'forces': (N, 3) float64 kcal/mol/Å,
            'prolix_system': PhysicsSystem,
            'displacement_fn': jax_md displacement_fn,
        }
    """
    omm_system, positions_ang, _ = build_ala_dip_openmm_system()
    bonded_params = extract_bonded_params(omm_system)
    energies = get_openmm_per_term_energies(omm_system, positions_ang)
    forces = get_openmm_forces(omm_system, positions_ang)
    prolix_system, displacement_fn = build_prolix_bonded_system(bonded_params, positions_ang)

    return {
        'omm_system': omm_system,
        'positions_ang': positions_ang,
        'bonded_params': bonded_params,
        'energies': energies,
        'forces': forces,
        'prolix_system': prolix_system,
        'displacement_fn': displacement_fn,
    }
