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

from prolix.typing import PhysicsSystem
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
    """Get per-atom forces from OpenMM (bonded terms only).

    Returns (N, 3) array in kcal/mol/Å. Only includes forces from ForceGroups
    0 (bonds), 1 (angles), and 2 (dihedrals), excluding nonbonded interactions.
    """
    if not HAS_OPENMM:
        raise ImportError("OpenMM not available")

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(omm_system, integrator, openmm.Platform.getPlatformByName("Reference"))

    # Set positions in nm
    context.setPositions(positions_ang / 10.0 * unit.nanometer)

    # Get forces from bonded ForceGroups only (0, 1, 2)
    state = context.getState(getForces=True, groups={0, 1, 2})
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
        dihedral_params=jnp.expand_dims(jnp.array(bonded_params['dihedral_params'], dtype=jnp.float64), axis=1),
        impropers=jnp.array(bonded_params.get('impropers', np.zeros((0, 4))), dtype=jnp.int32),
        improper_params=jnp.expand_dims(jnp.array(bonded_params.get('improper_params', np.zeros((0, 3))), dtype=jnp.float64), axis=1),
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


def extract_nonbonded_params(omm_system):
    """Extract nonbonded parameters from OpenMM System.

    Iterates over all forces, finds NonbondedForce, and extracts:
    - Per-atom: charges (e), sigmas (Å), epsilons (kcal/mol)
    - Exception data: pairs, sigmas (Å), epsilons (kcal/mol), charge products (e²)

    Returns dict with keys:
        charges, sigmas, epsilons, exception_pairs, exception_sigmas,
        exception_epsilons, exception_chargeprods
    All parameters converted to AKMA-like units (Å, kcal/mol, e).
    """
    if not HAS_OPENMM:
        raise ImportError("OpenMM not available")

    charges = []
    sigmas = []
    epsilons = []
    exception_pairs = []
    exception_sigmas = []
    exception_epsilons = []
    exception_chargeprods = []

    for force in omm_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            # Extract per-atom parameters
            for i in range(force.getNumParticles()):
                charge, sigma_nm, epsilon_kj = force.getParticleParameters(i)
                charges.append(charge.value_in_unit(unit.elementary_charge))
                sigmas.append(_nm_to_ang(sigma_nm.value_in_unit(unit.nanometer)))
                epsilons.append(_kj_to_kcal(epsilon_kj.value_in_unit(unit.kilojoule_per_mole)))

            # Extract exception data
            for k in range(force.getNumExceptions()):
                i, j, chargeProd_kj, sigma_nm, epsilon_kj = force.getExceptionParameters(k)
                exception_pairs.append([i, j])
                exception_sigmas.append(_nm_to_ang(sigma_nm.value_in_unit(unit.nanometer)))
                exception_epsilons.append(_kj_to_kcal(epsilon_kj.value_in_unit(unit.kilojoule_per_mole)))
                exception_chargeprods.append(chargeProd_kj.value_in_unit(unit.elementary_charge**2))

    # Convert to numpy arrays
    data = {
        'charges': np.array(charges, dtype=np.float64) if charges else np.zeros(0, dtype=np.float64),
        'sigmas': np.array(sigmas, dtype=np.float64) if sigmas else np.zeros(0, dtype=np.float64),
        'epsilons': np.array(epsilons, dtype=np.float64) if epsilons else np.zeros(0, dtype=np.float64),
        'exception_pairs': np.array(exception_pairs, dtype=np.int32) if exception_pairs else np.zeros((0, 2), dtype=np.int32),
        'exception_sigmas': np.array(exception_sigmas, dtype=np.float64) if exception_sigmas else np.zeros(0, dtype=np.float64),
        'exception_epsilons': np.array(exception_epsilons, dtype=np.float64) if exception_epsilons else np.zeros(0, dtype=np.float64),
        'exception_chargeprods': np.array(exception_chargeprods, dtype=np.float64) if exception_chargeprods else np.zeros(0, dtype=np.float64),
    }

    return data


def get_openmm_nonbonded_energies(omm_system, positions_ang):
    """Get nonbonded energies from OpenMM (LJ + Coulomb) via charge zeroing.

    Two-pass approach:
    1. Full nonbonded energy (LJ + Coulomb)
    2. Zero all per-atom charges, get LJ-only energy
    3. Restore charges
    4. E_Coulomb = E_nb - E_LJ

    NOTE: `setParticleParameters` only zeroes per-atom charges; exception entries
    (getException) are NOT modified. Therefore E_LJ (from the zeroed-charge pass)
    includes both 1-4 LJ AND 1-4 Coulomb contributions carried via exception_chargeprods.
    E_Coul = E_nb - E_LJ is therefore 1-5+ Coulomb only (no 1-4 contribution).
    This matches prolix's chunked_coulomb_energy semantics where 1-4 Coulomb is
    routed separately through exception_chargeprods.

    Returns dict with keys: 'total_nb', 'lj', 'coulomb' (all in kcal/mol).
    """
    if not HAS_OPENMM:
        raise ImportError("OpenMM not available")

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(omm_system, integrator, openmm.Platform.getPlatformByName("Reference"))

    # Set positions in nm
    context.setPositions(positions_ang / 10.0 * unit.nanometer)

    # Find NonbondedForce
    nonbonded_force = None
    for force in omm_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
            break

    if nonbonded_force is None:
        raise ValueError("No NonbondedForce found in system")

    # Pass 1: Get full nonbonded energy (ForceGroup 3)
    state = context.getState(getEnergy=True, groups={3})
    e_nb_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    e_nb = _kj_to_kcal(e_nb_kj)

    # Save original charges
    original_charges = []
    for i in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        original_charges.append(charge)

    # Pass 2: Zero all charges and get LJ-only energy
    for i in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        nonbonded_force.setParticleParameters(i, 0.0 * unit.elementary_charge, sigma, epsilon)

    context.reinitialize(preserveState=True)
    state = context.getState(getEnergy=True, groups={3})
    e_lj_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    e_lj = _kj_to_kcal(e_lj_kj)

    # Pass 3: Restore original charges
    for i in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        nonbonded_force.setParticleParameters(i, original_charges[i], sigma, epsilon)

    context.reinitialize(preserveState=True)

    # Compute Coulomb energy as difference
    e_coulomb = e_nb - e_lj

    return {
        'total_nb': e_nb,
        'lj': e_lj,
        'coulomb': e_coulomb,
    }


def get_openmm_nonbonded_forces(omm_system, positions_ang):
    """Get per-atom nonbonded forces from OpenMM (ForceGroup 3 only).

    Returns (N, 3) array in kcal/mol/Å.
    """
    if not HAS_OPENMM:
        raise ImportError("OpenMM not available")

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(omm_system, integrator, openmm.Platform.getPlatformByName("Reference"))

    # Set positions in nm
    context.setPositions(positions_ang / 10.0 * unit.nanometer)

    # Get forces from nonbonded ForceGroup only (3)
    state = context.getState(getForces=True, groups={3})
    forces_omm = state.getForces()

    # Convert from kJ/mol/nm to kcal/mol/Å
    forces_kj_nm = np.array(forces_omm.value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    forces_kcal_ang = forces_kj_nm * 0.1 / 4.184  # nm^-1 -> Å^-1, kJ -> kcal

    return forces_kcal_ang


def build_exclusion_spec(omm_system, n_atoms):
    """Build ExclusionSpec from OpenMM NonbondedForce exceptions.

    Classifies exceptions:
    - If epsilon ≈ 0 AND chargeProd ≈ 0: add to idx_12_13 (fully excluded)
    - Else: add to exception arrays (scaled 1-4 pairs)

    Returns ExclusionSpec with prolix.physics.neighbor_list.ExclusionSpec structure.
    """
    from prolix.physics.neighbor_list import ExclusionSpec

    # Enable x64 for float64 support if needed
    jax.config.update("jax_enable_x64", True)

    idx_12_13 = []
    exception_pairs = []
    exception_sigmas = []
    exception_epsilons = []
    exception_chargeprods = []

    # Find NonbondedForce
    nonbonded_force = None
    for force in omm_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
            break

    if nonbonded_force is None:
        raise ValueError("No NonbondedForce found in system")

    # Process exceptions
    for k in range(nonbonded_force.getNumExceptions()):
        i, j, chargeProd_kj, sigma_nm, epsilon_kj = nonbonded_force.getExceptionParameters(k)

        # Strip units
        chargeProd_val = chargeProd_kj.value_in_unit(unit.elementary_charge**2)
        sigma_val = sigma_nm.value_in_unit(unit.nanometer)
        epsilon_val = epsilon_kj.value_in_unit(unit.kilojoule_per_mole)

        # Classify: fully excluded (1-2/1-3) or scaled (1-4)
        if abs(epsilon_val) < 1e-12 and abs(chargeProd_val) < 1e-12:
            idx_12_13.append([i, j])
        else:
            exception_pairs.append([i, j])
            exception_sigmas.append(_nm_to_ang(sigma_val))
            exception_epsilons.append(_kj_to_kcal(epsilon_val))
            exception_chargeprods.append(chargeProd_val)

    # Validate: no atom should have >=32 exclusions
    atom_excl_count = [0] * n_atoms
    for i, j in idx_12_13:
        atom_excl_count[i] += 1
        atom_excl_count[j] += 1
    for i, j in exception_pairs:
        atom_excl_count[i] += 1
        atom_excl_count[j] += 1

    max_excl = max(atom_excl_count) if atom_excl_count else 0
    assert max_excl < 32, f"Atom has {max_excl} exclusions, limit is 32"

    # CRITICAL: exception_pairs (1-4 pairs) must be excluded from the main pairwise
    # sum in chunked_lj_energy / chunked_coulomb_energy, because map_exclusions_to_dense_padded
    # only processes idx_12_13 and idx_14 — not exception_pairs.  If we omit 1-4 pairs
    # from idx_12_13, they appear in both the main pairwise sum AND the exception energy
    # function, causing double-counting.  The fix is to include ALL exception pairs
    # (regardless of epsilon/chargeProd) in idx_12_13 so they are zeroed out from the
    # main sum, then added back with correct parameters via exception_pairs.
    all_idx_12_13 = idx_12_13 + exception_pairs  # type: list[list[int]]

    return ExclusionSpec(
        idx_12_13=jnp.array(all_idx_12_13, dtype=jnp.int32) if all_idx_12_13 else jnp.zeros((0, 2), dtype=jnp.int32),
        idx_14=jnp.zeros((0, 2), dtype=jnp.int32),
        scale_14_elec=0.0,
        scale_14_vdw=0.0,
        n_atoms=n_atoms,
        exception_pairs=jnp.array(exception_pairs, dtype=jnp.int32) if exception_pairs else jnp.zeros((0, 2), dtype=jnp.int32),
        exception_sigmas=jnp.array(exception_sigmas, dtype=jnp.float32) if exception_sigmas else jnp.zeros(0, dtype=jnp.float32),
        exception_epsilons=jnp.array(exception_epsilons, dtype=jnp.float32) if exception_epsilons else jnp.zeros(0, dtype=jnp.float32),
        exception_chargeprods=jnp.array(exception_chargeprods, dtype=jnp.float32) if exception_chargeprods else jnp.zeros(0, dtype=jnp.float32),
    )


def build_prolix_nonbonded_system(nb_params, bonded_params, positions_ang):
    """Build prolix PhysicsSystem with nonbonded parameters added to bonded system.

    Uses build_prolix_bonded_system to create the base system, then augments with
    nonbonded fields (charges, sigmas, epsilons). Reuses dihedral expand_dims logic.

    Args:
        nb_params: dict from extract_nonbonded_params()
        bonded_params: dict from extract_bonded_params()
        positions_ang: (N, 3) float64 array in Angstroms

    Returns:
        system: PhysicsSystem with bonded and nonbonded fields
        displacement_fn: jax_md displacement function
    """
    # Build bonded system first (handles positions, displacement_fn, bonds, angles, dihedrals, impropers)
    bonded_sys, displacement_fn = build_prolix_bonded_system(bonded_params, positions_ang)

    # Enable x64 for float64 support
    jax.config.update("jax_enable_x64", True)

    # Add nonbonded fields using dataclasses.replace
    import dataclasses
    system = dataclasses.replace(
        bonded_sys,
        charges=jnp.array(nb_params['charges'], dtype=jnp.float64),
        sigmas=jnp.array(nb_params['sigmas'], dtype=jnp.float64),
        epsilons=jnp.array(nb_params['epsilons'], dtype=jnp.float64),
    )

    return system, displacement_fn


def get_prolix_nonbonded_energies(system, displacement_fn, positions_ang, exclusion_spec):
    """Get per-term prolix nonbonded energies (LJ, Coulomb, 1-4 exceptions).

    Uses make_energy_fn with return_decomposed=True and exclusion_spec to
    separate LJ, Coulomb, and exception pair contributions.

    Returns dict with keys: 'lj', 'coulomb', 'exception_14' (all in kcal/mol).
    """
    from prolix.physics.system import make_energy_fn

    # Enable x64 for float64 support
    jax.config.update("jax_enable_x64", True)

    positions = jnp.array(positions_ang, dtype=jnp.float64)

    # Build decomposed energy functions
    energy_fns = make_energy_fn(
        displacement_fn,
        system,
        cutoff_distance=0,  # No cutoff for parity test
        pme_alpha=0.0,
        use_pbc=False,
        return_decomposed=True,
        exclusion_spec=exclusion_spec,
    )

    # Evaluate each term.
    # NOTE: 'lj' here is the COMBINED LJ + exception_14 energy, matching OpenMM's
    # charge-zeroed E_LJ decomposition which includes both 1-5+ LJ and all exception
    # pair contributions (1-4 LJ + 1-4 Coulomb from exception_chargeprods).
    # 'coulomb' is the 1-5+ Coulomb only (bare pairwise, excluding exception pairs),
    # matching OpenMM's E_Coul = E_nb - E_LJ_zeroed.
    # 'exception_14' is the raw exception pair energy (1-4 LJ + 1-4 Coulomb), returned
    # separately for the self-consistency gate in test_exception_14_energy_parity.
    e_lj_1_5plus = float(energy_fns['lj'](positions))
    e_exception_14 = float(energy_fns['exception'](positions))
    return {
        'lj': e_lj_1_5plus + e_exception_14,  # combined: matches OpenMM charge-zeroed E_LJ
        'coulomb': float(energy_fns['electrostatics'](positions)),
        'exception_14': e_exception_14,
    }


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
