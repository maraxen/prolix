import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space, partition
from pathlib import Path
import openmm
from openmm import app, unit

from prolix.physics.system import make_energy_fn_pure, PhysicsSystem

# Use float64 for high-precision parity if possible
jax.config.update("jax_enable_x64", True)

@pytest.fixture
def openmm_1uao_system():
    # Use a small PDB from the repo
    pdb_path = Path("data/pdb/1UAO.pdb")
    if not pdb_path.exists():
        pytest.skip("Data file 1UAO.pdb not found")
    
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)
    modeller.addSolvent(ff, padding=0.8*unit.nanometer, model='tip3p')
    
    omm_system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9*unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
        # Explicitly disable dispersion correction for exact parity if requested,
        # but here we just note it as a source of energy diff.
    )
    
    pos_A = np.array(modeller.positions.value_in_unit(unit.angstrom))
    box_vecs = modeller.topology.getPeriodicBoxVectors()
    box_A = np.array([box_vecs[0][0].value_in_unit(unit.angstrom),
                      box_vecs[1][1].value_in_unit(unit.angstrom),
                      box_vecs[2][2].value_in_unit(unit.angstrom)])
    
    return omm_system, pos_A, box_A

def extract_omm_params(omm_system):
    n = omm_system.getNumParticles()
    data = {'charges': np.zeros(n), 'sigmas': np.zeros(n), 'epsilons': np.zeros(n),
            'bonds': [], 'bond_params': [], 'angles': [], 'angle_params': [],
            'dihedrals': [], 'dihedral_params': [], 'impropers': [], 'improper_params': [],
            'exclusion_mask_vdw': np.ones((n, n), dtype=np.float32),
            'exclusion_mask_elec': np.ones((n, n), dtype=np.float32)}
    
    for f in omm_system.getForces():
        if isinstance(f, openmm.NonbondedForce):
            for i in range(n):
                q, s, e = f.getParticleParameters(i)
                data['charges'][i] = q.value_in_unit(unit.elementary_charge)
                data['sigmas'][i] = s.value_in_unit(unit.angstrom)
                data['epsilons'][i] = e.value_in_unit(unit.kilocalorie_per_mole)
            for i in range(f.getNumExceptions()):
                p1, p2, q, s, e = f.getExceptionParameters(i)
                q1, s1, e1 = f.getParticleParameters(p1); q2, s2, e2 = f.getParticleParameters(p2)
                q1v, q2v = q1.value_in_unit(unit.elementary_charge), q2.value_in_unit(unit.elementary_charge)
                q_full = q1v * q2v; q_exc = q.value_in_unit(unit.elementary_charge**2)
                se = q_exc / q_full if abs(q_full) > 1e-10 else 0.0
                e1v, e2v = e1.value_in_unit(unit.kilocalorie_per_mole), e2.value_in_unit(unit.kilocalorie_per_mole)
                e_full = np.sqrt(e1v * e2v); e_exc = e.value_in_unit(unit.kilocalorie_per_mole)
                sv = e_exc / e_full if e_full > 1e-10 else 0.0
                data['exclusion_mask_vdw'][p1, p2] = data['exclusion_mask_vdw'][p2, p1] = sv
                data['exclusion_mask_elec'][p1, p2] = data['exclusion_mask_elec'][p2, p1] = se
        elif isinstance(f, openmm.HarmonicBondForce):
            for i in range(f.getNumBonds()):
                p1, p2, l, k = f.getBondParameters(i)
                data['bonds'].append([p1, p2])
                data['bond_params'].append([l.value_in_unit(unit.angstrom), k.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom**2)])
        elif isinstance(f, openmm.HarmonicAngleForce):
            for i in range(f.getNumAngles()):
                p1, p2, p3, a, k = f.getAngleParameters(i)
                data['angles'].append([p1, p2, p3])
                data['angle_params'].append([a.value_in_unit(unit.radian), k.value_in_unit(unit.kilocalorie_per_mole / unit.radian**2)])
        elif isinstance(f, openmm.PeriodicTorsionForce):
            for i in range(f.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = f.getTorsionParameters(i)
                data['dihedrals'].append([p1, p2, p3, p4])
                data['dihedral_params'].append([float(per), phase.value_in_unit(unit.radian), k.value_in_unit(unit.kilocalorie_per_mole)])
    for k in ['bonds', 'bond_params', 'angles', 'angle_params', 'dihedrals', 'dihedral_params']:
        data[k] = np.array(data[k]) if len(data[k]) > 0 else np.zeros((0, 2 if 'bond' in k else (3 if 'angle' in k else 4)))
    return data

@pytest.mark.slow
def test_openmm_energy_force_parity(openmm_1uao_system):
    omm_sys, pos_A, box_A = openmm_1uao_system
    
    # Get OMM reference data
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(omm_sys, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(pos_A * 0.1 * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    omm_e = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    omm_f = state.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)
    
    # Extract parameters
    params = extract_omm_params(omm_sys)
    
    # Create Prolix system
    phys_sys = PhysicsSystem.from_dict({
            'charges': params['charges'], 'sigmas': params['sigmas'], 'epsilons': params['epsilons'],
            'bonds': params['bonds'], 'bond_params': params['bond_params'],
            'angles': params['angles'], 'angle_params': params['angle_params'],
            'proper_dihedrals': params['dihedrals'], 'dihedral_params': params['dihedral_params'],
            'exclusion_mask': params['exclusion_mask_vdw'],
        }, positions=jnp.array(pos_A), box_size=jnp.array(box_A), cutoff_distance=9.0)
    
    # Set dense exclusion scales
    phys_sys = phys_sys.replace(dense_excl_scale_vdw=jnp.array(params['exclusion_mask_vdw']),
                                dense_excl_scale_elec=jnp.array(params['exclusion_mask_elec']))
    
    disp_fn, _ = space.periodic(box_A)
    # Match OpenMM default PME settings for 1UAO cell
    pme_alpha = 0.292 
    grid_spacing = 1.09
    
    e_params, energy_fn = make_energy_fn_pure(disp_fn, phys_sys, pme_alpha=pme_alpha, pme_grid_spacing=grid_spacing, tile_size=64)
    
    # Use Neighbor List for parity check
    neighbor_fn = partition.neighbor_list(disp_fn, box_A, r_cutoff=9.5, dr_threshold=1.0)
    nb = neighbor_fn.allocate(jnp.array(pos_A))
    
    prolix_e = float(energy_fn(e_params, jnp.array(pos_A), neighbor=nb))
    prolix_f = -jax.grad(energy_fn, argnums=1)(e_params, jnp.array(pos_A), neighbor=nb)
    
    print(f"\nOpenMM Energy: {omm_e:.4f}")
    print(f"Prolix Energy: {prolix_e:.4f}")
    print(f"Energy Diff: {prolix_e - omm_e:.4f}")
    
    # Energy Parity Gate
    assert abs(prolix_e - omm_e) < 5.0, f"Energy discrepancy too large: {prolix_e - omm_e:.4f}"
    
    # Force Parity Gate: RMSE < 1.0 kcal/mol/A
    f_rmse = np.sqrt(np.mean((np.array(prolix_f) - omm_f)**2))
    print(f"Force RMSE: {f_rmse:.4e}")
    assert f_rmse < 1.0, f"Force RMSE too large: {f_rmse:.4e}"
    
    # Verify NL vs Dense internal consistency
    prolix_e_dense = float(energy_fn(e_params, jnp.array(pos_A), neighbor=None))
    assert abs(prolix_e - prolix_e_dense) < 1e-5, "NL and Dense energies inconsistent"
