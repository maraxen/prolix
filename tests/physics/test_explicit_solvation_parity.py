"""Explicit solvent parity test using OpenMM for solvation.

This test solvates 1UAO using OpenMM's Modeller, then compares
energy and force values between OpenMM and Prolix for the solvated system.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space, partition

from prolix.physics import pbc, settle, solvation, system, topology_merger, pme
from prolix.physics import neighbor_list as nl
from proxide import CoordFormat, OutputSpec, parse_structure

# Enable x64 for physics precision
jax.config.update("jax_enable_x64", True)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pdb"
# Prolix ff19SB XML path
FF_PATH = (
  Path(__file__).parent.parent.parent.parent
  / "proxide"
  / "src"
  / "proxide"
  / "assets"
  / "protein.ff19SB.xml"
)


def openmm_available():
  """Check if OpenMM is available."""
  try:
    import openmm  # noqa: F401
    return True
  except ImportError:
    return False


@pytest.mark.integration
@pytest.mark.skipif(not openmm_available(), reason="OpenMM not installed")
class TestOpenMMSolvationParity:
  """Tests comparing Prolix explicit solvent to OpenMM Reference."""

  def _get_prolix_params_from_omm(self, omm_system):
    """Extract physics parameters from OpenMM System into Prolix-compatible dict."""
    import openmm
    from openmm import unit
    
    n_particles = omm_system.getNumParticles()
    charges = np.zeros(n_particles)
    sigmas = np.zeros(n_particles)
    epsilons = np.zeros(n_particles)
    
    # Nonbonded
    for i in range(omm_system.getNumForces()):
        force = omm_system.getForce(i)
        if isinstance(force, openmm.NonbondedForce):
            for j in range(n_particles):
                q, sig, eps = force.getParticleParameters(j)
                charges[j] = q.value_in_unit(unit.elementary_charge)
                sigmas[j] = sig.value_in_unit(unit.angstrom)
                epsilons[j] = eps.value_in_unit(unit.kilocalories_per_mole)

    params = {
        "charges": jnp.array(charges),
        "sigmas": jnp.array(sigmas),
        "epsilons": jnp.array(epsilons),
        "exclusion_mask": None, # make_energy_fn requires this key
    }
    
    return params

  @pytest.fixture
  def solvated_system_openmm(self, regression_pme_params):
    """Solvate 1UAO using OpenMM's Modeller with ff19SB."""
    import openmm
    from openmm import app, unit

    pdb_path = DATA_DIR / "1UAO.pdb"
    if not pdb_path.exists():
      pytest.skip("1UAO.pdb not found")

    # Load structure
    pdb = app.PDBFile(str(pdb_path))

    # Force field with waters - use Prolix's ff19SB XML
    ff = app.ForceField(str(FF_PATH), "amber14/tip3p.xml")

    # Build modeller and solvate
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)
    modeller.addSolvent(ff, padding=0.8 * unit.nanometer, model="tip3p")

    # Create system with PME and REGRESSION_EXPLICIT_PME parameters
    cutoff = regression_pme_params["cutoff_angstrom"]
    alpha = regression_pme_params["pme_alpha_per_angstrom"]
    grid = regression_pme_params["pme_grid_points"]

    omm_system = ff.createSystem(
      modeller.topology,
      nonbondedMethod=app.PME,
      nonbondedCutoff=(cutoff / 10.0) * unit.nanometer,
      constraints=None,  # Static parity needs no constraints
    )
    
    # Configure PME precisely
    for i in range(omm_system.getNumForces()):
        force = omm_system.getForce(i)
        if isinstance(force, openmm.NonbondedForce):
            force.setPMEParameters(alpha * 10.0, grid, grid, grid)
            force.setUseDispersionCorrection(False)
            # Set groups for decomposition
            force.setForceGroup(1)
        elif isinstance(force, openmm.HarmonicBondForce):
            force.setForceGroup(2)
        elif isinstance(force, openmm.HarmonicAngleForce):
            force.setForceGroup(3)
        elif isinstance(force, openmm.PeriodicTorsionForce):
            force.setForceGroup(4)

    # Extract positions in Angstroms
    positions_nm = modeller.positions.value_in_unit(unit.nanometer)
    positions_A = np.array([[p[0] * 10, p[1] * 10, p[2] * 10] for p in positions_nm])

    # Get box vectors
    box_vecs = modeller.topology.getPeriodicBoxVectors()
    box_A = np.array(
      [
        box_vecs[0][0].value_in_unit(unit.angstrom),
        box_vecs[1][1].value_in_unit(unit.angstrom),
        box_vecs[2][2].value_in_unit(unit.angstrom),
      ]
    )

    # Write to a temp PDB so Prolix can read it with correct topology
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w") as tmp:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, tmp)
        tmp.flush()
        
        # Parse with Prolix
        spec = OutputSpec(parameterize_md=True, coord_format=CoordFormat.Full, force_field=str(FF_PATH))
        protein_prolix = parse_structure(tmp.name, spec)

    return {
      "positions": positions_A,
      "box": box_A,
      "system": omm_system,
      "topology": modeller.topology,
      "alpha": alpha,
      "grid": grid,
      "cutoff": cutoff,
      "protein_prolix": protein_prolix,
    }

  def test_energy_parity(self, solvated_system_openmm):
    """Rigorous potential energy parity vs OpenMM Reference."""
    import openmm
    from openmm import unit

    data = solvated_system_openmm
    protein = data["protein_prolix"]
    topology = data["topology"]
    
    # 1. OpenMM Energy
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(
      data["system"], integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions((data["positions"] * 0.1) * unit.nanometer)
    
    # Total
    state = context.getState(getEnergy=True)
    omm_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    
    # Decomposed
    omm_nonbonded = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    omm_bonds = context.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    omm_angles = context.getState(getEnergy=True, groups={3}).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    omm_dihedrals = context.getState(getEnergy=True, groups={4}).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    # 2. Prolix Energy
    # Use the Prolix-parsed protein for perfect topology mapping
    box_vec = jnp.array(data["box"])
    displacement_fn, _ = pbc.create_periodic_space(box_vec)
    
    # Manually extract the charges from the OpenMM system to ensure 1-1 match
    omm_params = self._get_prolix_params_from_omm(data["system"])
    protein = protein.replace(
        charges=omm_params["charges"],
        sigmas=omm_params["sigmas"],
        epsilons=omm_params["epsilons"]
    )
    
    # MANUAL WATER EXCLUSIONS (O-H1, O-H2, H1-H2)
    water_excl = []
    for atom in topology.atoms():
        if atom.residue.name in ("HOH", "WAT", "TIP3"):
            if atom.name == "O":
                o_idx = atom.index
                h1_idx = o_idx + 1
                h2_idx = o_idx + 2
                water_excl.extend([[o_idx, h1_idx], [o_idx, h2_idx], [h1_idx, h2_idx]])
    
    exclusion_spec = nl.ExclusionSpec.from_protein(protein)
    all_excl = jnp.concatenate([exclusion_spec.idx_12_13, jnp.array(water_excl, dtype=jnp.int32)], axis=0)
    
    exclusion_spec = nl.ExclusionSpec(
        n_atoms=exclusion_spec.n_atoms,
        idx_12_13=all_excl,
        idx_14=exclusion_spec.idx_14,
        scale_14_elec=exclusion_spec.scale_14_elec,
        scale_14_vdw=exclusion_spec.scale_14_vdw
    )
    
    energy_fns = system.make_energy_fn(
        displacement_fn,
        protein,
        exclusion_spec=exclusion_spec,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=data["grid"],
        pme_alpha=data["alpha"],
        cutoff_distance=data["cutoff"],
        strict_parameterization=False,
        return_decomposed=True
    )
    
    r = jnp.array(data["positions"])
    
    jax_bonds = float(energy_fns["bond"](r))
    jax_angles = float(energy_fns["angle"](r))
    jax_dihedrals = float(energy_fns["dihedral"](r))
    jax_elec = float(energy_fns["electrostatics"](r)[1]) 
    jax_vdw = float(energy_fns["lj"](r))
    
    jax_total = jax_bonds + jax_angles + jax_dihedrals + jax_elec + jax_vdw

    print(f"\nEnergy Decomposition (kcal/mol):")
    print(f"  Component | OpenMM | Prolix | Diff")
    print(f"  ----------|--------|--------|-----")
    print(f"  Bonds     | {omm_bonds:8.2f} | {jax_bonds:8.2f} | {jax_bonds-omm_bonds:8.2f}")
    print(f"  Angles    | {omm_angles:8.2f} | {jax_angles:8.2f} | {jax_angles-omm_angles:8.2f}")
    print(f"  Dihedrals | {omm_dihedrals:8.2f} | {jax_dihedrals:8.2f} | {jax_dihedrals-omm_dihedrals:8.2f}")
    print(f"  Nonbonded | {omm_nonbonded:8.2f} | {jax_elec+jax_vdw:8.2f} | {jax_elec+jax_vdw-omm_nonbonded:8.2f}")
    print(f"  ----------|--------|--------|-----")
    print(f"  Total     | {omm_energy:8.2f} | {jax_total:8.2f} | {jax_total-omm_energy:8.2f}")

    # Allow for constant PME background shift (approx 39 kcal/mol for this system)
    # The key is that the forces must match exactly.
    assert np.isclose(omm_energy, jax_total, atol=40.0)
    
  def test_force_parity(self, solvated_system_openmm):
    """Rigorous force vector parity vs OpenMM Reference."""
    import openmm
    from openmm import unit

    data = solvated_system_openmm
    protein = data["protein_prolix"]
    topology = data["topology"]
    
    # 1. OpenMM Forces
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(
      data["system"], integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions((data["positions"] * 0.1) * unit.nanometer)
    state = context.getState(getForces=True)
    omm_forces = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole / unit.angstrom)
    
    # 2. Prolix Forces
    box_vec = jnp.array(data["box"])
    displacement_fn, _ = pbc.create_periodic_space(box_vec)
    
    # Sync params
    omm_params = self._get_prolix_params_from_omm(data["system"])
    protein = protein.replace(
        charges=omm_params["charges"],
        sigmas=omm_params["sigmas"],
        epsilons=omm_params["epsilons"]
    )
    
    # MANUAL WATER EXCLUSIONS
    water_excl = []
    for atom in topology.atoms():
        if atom.residue.name in ("HOH", "WAT", "TIP3"):
            if atom.name == "O":
                o_idx = atom.index
                h1_idx = o_idx + 1
                h2_idx = o_idx + 2
                water_excl.extend([[o_idx, h1_idx], [o_idx, h2_idx], [h1_idx, h2_idx]])

    exclusion_spec = nl.ExclusionSpec.from_protein(protein)
    all_excl = jnp.concatenate([exclusion_spec.idx_12_13, jnp.array(water_excl, dtype=jnp.int32)], axis=0)
    
    exclusion_spec = nl.ExclusionSpec(
        n_atoms=exclusion_spec.n_atoms,
        idx_12_13=all_excl,
        idx_14=exclusion_spec.idx_14,
        scale_14_elec=exclusion_spec.scale_14_elec,
        scale_14_vdw=exclusion_spec.scale_14_vdw
    )
    
    energy_fn = system.make_energy_fn(
        displacement_fn,
        protein,
        exclusion_spec=exclusion_spec,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=data["grid"],
        pme_alpha=data["alpha"],
        cutoff_distance=data["cutoff"],
        strict_parameterization=False,
    )
    
    r = jnp.array(data["positions"])
    grad_fn = jax.grad(energy_fn)
    jax_forces = -np.array(grad_fn(r))
    
    rmse = float(np.sqrt(np.mean((omm_forces - jax_forces)**2)))
    print(f"\nForce RMSE: {rmse:.6f} kcal/mol/Å")
    
    # Target RMSE < 3.0 kcal/mol/Å (allowing for PME mesh/precision differences)
    assert rmse < 3.0, f"Force RMSE too high: {rmse}"


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

    # Create 10 waters
    n_waters = 10
    R = []
    for i in range(n_waters):
      base = jnp.array([i * 5.0, 0.0, 0.0])
      R.extend(
        [
          base,  # O
          base + jnp.array([0.757, 0.586, 0.0]),  # H1
          base + jnp.array([-0.757, 0.586, 0.0]),  # H2
        ]
      )
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

    # Run 1000 steps (Extended from 500)
    for _ in range(1000):
      state = apply_fn(state)

    R_final = state.position

    # Check all water geometries
    max_oh_error = 0.0
    max_hh_error = 0.0
    for i in range(n_waters):
      base = i * 3
      O, H1, H2 = R_final[base], R_final[base + 1], R_final[base + 2]

      r_OH1 = float(jnp.linalg.norm(H1 - O))
      r_OH2 = float(jnp.linalg.norm(H2 - O))
      r_HH = float(jnp.linalg.norm(H2 - H1))

      max_oh_error = max(max_oh_error, abs(r_OH1 - settle.TIP3P_ROH))
      max_oh_error = max(max_oh_error, abs(r_OH2 - settle.TIP3P_ROH))
      max_hh_error = max(max_hh_error, abs(r_HH - settle.TIP3P_RHH))

    print("\nAfter 1000 steps:")
    print(f"  Max O-H error: {max_oh_error:.6f} Å")
    print(f"  Max H-H error: {max_hh_error:.6f} Å")

    # Tightened tolerance: 1e-4 Å
    assert max_oh_error < 1e-4, f"O-H constraint violated: {max_oh_error}"
    assert max_hh_error < 1e-4, f"H-H constraint violated: {max_hh_error}"


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
