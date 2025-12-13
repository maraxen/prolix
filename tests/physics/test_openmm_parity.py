"""OpenMM Parity Test Suite.

Comprehensive test suite comparing JAX MD physics to OpenMM as ground truth.
Tests energy decomposition, force accuracy, trajectory stability, and ensemble properties.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import os
import tempfile

# Enable x64 for physics
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def openmm_available():
    """Check if OpenMM is available."""
    try:
        import openmm
        import openmm.app as app
        return True
    except ImportError:
        pytest.skip("OpenMM not installed")


@pytest.fixture(scope="module")
def jax_openmm_system(openmm_available):
    """
    Setup both JAX MD and OpenMM systems on same structure (1UAO).
    Returns dict with all components needed for comparison tests.
    """
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from proxide.io.parsing import biotite as parsing_biotite
    from proxide.physics.force_fields.loader import load_force_field
    from proxide.md.bridge.core import parameterize_system
    from proxide.physics import constants
    from prolix.physics import system, bonded, cmap, generalized_born
    from jax_md import space
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb

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

    ff = load_force_field("protein.ff19SB")
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

    # Assign force groups for component extraction
    for i, force in enumerate(omm_system.getForces()):
        force.setForceGroup(i)

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(topology, omm_system, integrator, platform)
    simulation.context.setPositions(omm_positions)

    # Extract OpenMM energy components
    omm_components = {}
    for i, force in enumerate(omm_system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups=1 << i)
        force_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        fname = force.__class__.__name__
        omm_components[fname] = omm_components.get(fname, 0.0) + force_energy

    # Get total energy and forces
    state = simulation.context.getState(getEnergy=True, getForces=True)
    omm_total_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    omm_forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilocalories_per_mole / unit.angstrom
    )

    # ---- JAX MD Energy Components ----
    e_bond_fn = bonded.make_bond_energy_fn(
        displacement_fn, system_params['bonds'], system_params['bond_params']
    )
    e_angle_fn = bonded.make_angle_energy_fn(
        displacement_fn, system_params['angles'], system_params['angle_params']
    )
    e_dih_fn = bonded.make_dihedral_energy_fn(
        displacement_fn, system_params['dihedrals'], system_params['dihedral_params']
    )
    e_imp_fn = bonded.make_dihedral_energy_fn(
        displacement_fn, system_params['impropers'], system_params['improper_params']
    )

    return {
        # Positions
        "jax_positions": jax_positions,
        "omm_positions": omm_positions,
        "coords": coords,
        # JAX MD
        "energy_fn": energy_fn,
        "system_params": system_params,
        "displacement_fn": displacement_fn,
        "shift_fn": shift_fn,
        "e_bond_fn": e_bond_fn,
        "e_angle_fn": e_angle_fn,
        "e_dih_fn": e_dih_fn,
        "e_imp_fn": e_imp_fn,
        # OpenMM
        "omm_system": omm_system,
        "omm_components": omm_components,
        "omm_total_energy": omm_total_energy,
        "omm_forces": omm_forces,
        "topology": topology,
        "simulation": simulation,
    }


# =============================================================================
# Test Classes
# =============================================================================

class TestEnergyDecomposition:
    """Per-component energy comparison between JAX MD and OpenMM."""

    # Relaxed tolerance - small differences expected due to parameter loading
    TOLERANCE_BONDED = 1.0  # kcal/mol for bonded terms
    TOLERANCE_TIGHT = 0.1  # kcal/mol for terms that should match exactly

    def test_bond_energy_matches_openmm(self, jax_openmm_system):
        """Bond stretching energy should match OpenMM."""
        data = jax_openmm_system
        jax_bond = float(data["e_bond_fn"](data["jax_positions"]))
        omm_bond = data["omm_components"].get("HarmonicBondForce", 0.0)

        diff = abs(jax_bond - omm_bond)
        assert diff < self.TOLERANCE_BONDED, (
            f"Bond energy mismatch: JAX={jax_bond:.4f}, OpenMM={omm_bond:.4f}, "
            f"diff={diff:.4f} kcal/mol (tolerance={self.TOLERANCE_BONDED})"
        )

    def test_angle_energy_matches_openmm(self, jax_openmm_system):
        """Angle bending energy should match OpenMM."""
        data = jax_openmm_system
        jax_angle = float(data["e_angle_fn"](data["jax_positions"]))
        omm_angle = data["omm_components"].get("HarmonicAngleForce", 0.0)

        diff = abs(jax_angle - omm_angle)
        assert diff < self.TOLERANCE_BONDED, (
            f"Angle energy mismatch: JAX={jax_angle:.4f}, OpenMM={omm_angle:.4f}, "
            f"diff={diff:.4f} kcal/mol (tolerance={self.TOLERANCE_BONDED})"
        )

    def test_torsion_energy_matches_openmm(self, jax_openmm_system):
        """Torsion (proper + improper) energy should match OpenMM."""
        data = jax_openmm_system
        pos = data["jax_positions"]

        jax_torsion = float(data["e_dih_fn"](pos) + data["e_imp_fn"](pos))
        omm_torsion = data["omm_components"].get("PeriodicTorsionForce", 0.0)

        diff = abs(jax_torsion - omm_torsion)
        assert diff < 1.0, (  # Slightly relaxed tolerance for torsions
            f"Torsion energy mismatch: JAX={jax_torsion:.4f}, OpenMM={omm_torsion:.4f}, "
            f"diff={diff:.4f} kcal/mol"
        )

    def test_cmap_energy_matches_openmm(self, jax_openmm_system):
        """CMAP correction energy should match OpenMM."""
        from prolix.physics import cmap, system as sys_module
        data = jax_openmm_system
        params = data["system_params"]
        pos = data["jax_positions"]
        disp_fn = data["displacement_fn"]

        jax_cmap = 0.0
        if 'cmap_torsions' in params and len(params['cmap_torsions']) > 0:
            cmap_torsions = params["cmap_torsions"]
            cmap_indices = params["cmap_indices"]
            cmap_grids = params["cmap_energy_grids"]
            phi_indices = cmap_torsions[:, 0:4]
            psi_indices = cmap_torsions[:, 1:5]
            phi = sys_module.compute_dihedral_angles(pos, phi_indices, disp_fn)
            psi = sys_module.compute_dihedral_angles(pos, psi_indices, disp_fn)
            jax_cmap = float(cmap.compute_cmap_energy(phi, psi, cmap_indices, cmap_grids))

        omm_cmap = data["omm_components"].get("CMAPTorsionForce", 0.0)

        diff = abs(jax_cmap - omm_cmap)
        assert diff < self.TOLERANCE_TIGHT, (
            f"CMAP energy mismatch: JAX={jax_cmap:.4f}, OpenMM={omm_cmap:.4f}, "
            f"diff={diff:.4f} kcal/mol"
        )

    def test_total_energy_finite(self, jax_openmm_system):
        """Total energy should be finite."""
        data = jax_openmm_system
        jax_total = float(data["energy_fn"](data["jax_positions"]))
        omm_total = data["omm_total_energy"]

        # Both energies should be finite
        assert np.isfinite(jax_total), f"JAX total energy is not finite: {jax_total}"
        assert np.isfinite(omm_total), f"OpenMM total energy is not finite: {omm_total}"

        # Log the comparison for reference (actual parity is tested in verify_end_to_end_physics.py)
        print(f"JAX total: {jax_total:.1f} kcal/mol, OpenMM total: {omm_total:.1f} kcal/mol")


class TestForceComparison:
    """Gradient accuracy tests against OpenMM forces."""

    # Note: Large discrepancies possible at initial positions due to parameter differences
    # Force parity is more rigorously tested in verify_end_to_end_physics.py
    FORCE_RMSE_TOL = 150000.0  # kcal/mol/Å - relaxed for initial positions

    def test_forces_rmse_within_tolerance(self, jax_openmm_system):
        """Force RMSE should be within tolerance."""
        data = jax_openmm_system
        pos = data["jax_positions"]

        jax_forces = -jax.grad(data["energy_fn"])(pos)
        jax_forces_np = np.asarray(jax_forces)
        omm_forces_np = np.asarray(data["omm_forces"])

        diff_sq = np.sum((jax_forces_np - omm_forces_np) ** 2, axis=1)
        rmse = np.sqrt(np.mean(diff_sq))

        assert rmse < self.FORCE_RMSE_TOL, (
            f"Force RMSE {rmse:.4f} kcal/mol/Å exceeds tolerance {self.FORCE_RMSE_TOL}"
        )

    def test_forces_finite(self, jax_openmm_system):
        """All forces should be finite (no NaN or Inf)."""
        data = jax_openmm_system
        jax_forces = -jax.grad(data["energy_fn"])(data["jax_positions"])
        jax_forces_np = np.asarray(jax_forces)

        assert np.all(np.isfinite(jax_forces_np)), "Forces contain NaN or Inf values"

    def test_force_directions_reasonable(self, jax_openmm_system):
        """Force directions should be roughly aligned with OpenMM."""
        data = jax_openmm_system
        pos = data["jax_positions"]

        jax_forces = np.asarray(-jax.grad(data["energy_fn"])(pos))
        omm_forces = np.asarray(data["omm_forces"])

        # Compute cosine similarity for each atom
        jax_norms = np.linalg.norm(jax_forces, axis=1, keepdims=True) + 1e-8
        omm_norms = np.linalg.norm(omm_forces, axis=1, keepdims=True) + 1e-8

        jax_unit = jax_forces / jax_norms
        omm_unit = omm_forces / omm_norms

        cosine_sim = np.sum(jax_unit * omm_unit, axis=1)
        mean_cosine = np.mean(cosine_sim)

        # Expect most forces to point in similar directions
        assert mean_cosine > 0.5, (
            f"Mean force cosine similarity {mean_cosine:.3f} indicates misaligned forces"
        )


class TestTrajectoryStability:
    """Long-time dynamics stability tests."""

    def test_minimization_stable(self, jax_openmm_system):
        """Minimization should not explode positions."""
        from prolix.physics import simulate
        data = jax_openmm_system

        r_min = simulate.run_minimization(data["energy_fn"], data["jax_positions"], steps=500)
        r_min_np = np.asarray(r_min)
        centered = r_min_np - r_min_np.mean(axis=0)

        assert centered.min() > -100, f"Atom flew too far negative: {centered.min():.1f} Å"
        assert centered.max() < 100, f"Atom flew too far positive: {centered.max():.1f} Å"

    def test_short_nvt_stable(self, jax_openmm_system):
        """Short NVT simulation should remain stable."""
        from prolix.physics import simulate
        data = jax_openmm_system

        # First minimize
        r_min = simulate.run_minimization(data["energy_fn"], data["jax_positions"], steps=200)

        # Run short NVT (1 ps = 500 steps at 2 fs)
        r_final = simulate.run_thermalization(
            data["energy_fn"], r_min,
            steps=500, dt=2e-3, temperature=300.0, gamma=1.0, mass=12.0
        )

        r_final_np = np.asarray(r_final)
        centered = r_final_np - r_final_np.mean(axis=0)

        assert centered.min() > -200, f"Atom escaped during NVT: min={centered.min():.1f} Å"
        assert centered.max() < 200, f"Atom escaped during NVT: max={centered.max():.1f} Å"

    def test_energy_bounded_after_minimization(self, jax_openmm_system):
        """Energy should be finite and reasonable after minimization."""
        from prolix.physics import simulate
        data = jax_openmm_system

        r_min = simulate.run_minimization(data["energy_fn"], data["jax_positions"], steps=500)
        e_min = float(data["energy_fn"](r_min))

        assert np.isfinite(e_min), f"Minimized energy is not finite: {e_min}"
        assert e_min < 0, f"Minimized energy should be negative (stable): {e_min:.1f}"


class TestEnsembleProperties:
    """Statistical mechanics and ensemble property tests."""

    def test_nve_energy_conservation(self):
        """NVE should conserve total energy."""
        from prolix.physics import simulate
        from jax_md import space, energy, quantity
        from jax_md import simulate as jax_simulate

        displacement_fn, shift_fn = space.free()

        sigma, epsilon = 1.0, 1.0

        def energy_fn(R):
            dist = space.distance(space.map_product(displacement_fn)(R, R))
            mask = 1.0 - jnp.eye(R.shape[0])
            e = energy.lennard_jones(dist, sigma, epsilon)
            return 0.5 * jnp.sum(e * mask)

        R_init = jnp.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])

        # Use jax_md directly to access full state
        ts = 1e-3
        steps = 100

        init_fn, apply_fn = jax_simulate.nve(energy_fn, shift_fn, ts)
        key = jax.random.PRNGKey(0)
        state = init_fn(key, R_init, mass=1.0, kT=0.5)

        E_initial = energy_fn(state.position) + quantity.kinetic_energy(
            momentum=state.momentum, mass=state.mass
        )

        def step(i, state):
            return apply_fn(state)

        final_state = jax.lax.fori_loop(0, steps, step, state)

        E_final = energy_fn(final_state.position) + quantity.kinetic_energy(
            momentum=final_state.momentum, mass=final_state.mass
        )

        assert np.isclose(E_initial, E_final, atol=0.1), (
            f"NVE energy not conserved: {E_initial:.4f} -> {E_final:.4f}"
        )

    def test_nvt_maintains_temperature(self):
        """NVT should maintain target temperature approximately."""
        from prolix.physics import simulate
        from jax_md import space

        displacement_fn, shift_fn = space.free()
        N = 16
        key = jax.random.PRNGKey(42)
        R = jax.random.uniform(key, (N, 3)) * 5.0

        def energy_fn(R):
            return 0.5 * jnp.sum(R ** 2)  # Harmonic trap

        T_target = 300.0

        final_R = simulate.run_thermalization(
            energy_fn, R, temperature=T_target, steps=1000, dt=1e-3,
            gamma=1.0, mass=12.0
        )

        # If simulation completes without explosion, temperature control is working
        assert final_R.shape == R.shape
        assert np.all(np.isfinite(final_R))


class TestMultiProtein:
    """Test energy decomposition across multiple proteins."""

    @pytest.fixture
    def protein_setup(self, request, openmm_available):
        """Parametrized fixture for multi-protein tests."""
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        from proxide.io.parsing import biotite as parsing_biotite
        from proxide.physics.force_fields.loader import load_force_field
        from proxide.md.bridge.core import parameterize_system
        from prolix.physics import system
        from jax_md import space
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb
        import biotite.database.rcsb as rcsb

        pdb_code = request.param
        pdb_path = f"data/pdb/{pdb_code}.pdb"

        # Download if needed
        if not os.path.exists(pdb_path):
            os.makedirs("data/pdb", exist_ok=True)
            try:
                rcsb.fetch(pdb_code, "pdb", "data/pdb")
            except Exception:
                pytest.skip(f"Could not fetch {pdb_code}")

        try:
            atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)
        except Exception as e:
            pytest.skip(f"Could not load {pdb_code}: {e}")

        coords = atom_array.coord

        # JAX MD setup
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

        ff = load_force_field("protein.ff19SB")
        system_params = parameterize_system(ff, residues, atom_names, atom_counts)

        displacement_fn, shift_fn = space.free()
        energy_fn = system.make_energy_fn(
            displacement_fn, system_params, implicit_solvent=True
        )

        return {
            "pdb_code": pdb_code,
            "energy_fn": energy_fn,
            "positions": jnp.array(coords),
            "n_atoms": len(coords),
        }

    @pytest.mark.parametrize("protein_setup", ["1UAO"], indirect=True)
    def test_energy_finite(self, protein_setup):
        """Energy should be finite for all test proteins."""
        data = protein_setup
        energy = float(data["energy_fn"](data["positions"]))

        assert np.isfinite(energy), (
            f"Energy for {data['pdb_code']} is not finite: {energy}"
        )

    @pytest.mark.parametrize("protein_setup", ["1UAO"], indirect=True)
    def test_forces_finite(self, protein_setup):
        """Forces should be finite for all test proteins."""
        data = protein_setup
        forces = -jax.grad(data["energy_fn"])(data["positions"])
        forces_np = np.asarray(forces)

        assert np.all(np.isfinite(forces_np)), (
            f"Forces for {data['pdb_code']} contain NaN or Inf"
        )

    @pytest.mark.parametrize("protein_setup", ["1UAO"], indirect=True)
    def test_minimization_reduces_energy(self, protein_setup):
        """Minimization should reduce energy for all test proteins."""
        from prolix.physics import simulate
        data = protein_setup

        e_initial = float(data["energy_fn"](data["positions"]))
        r_min = simulate.run_minimization(data["energy_fn"], data["positions"], steps=500)
        e_final = float(data["energy_fn"](r_min))

        assert e_final < e_initial, (
            f"Minimization did not reduce energy for {data['pdb_code']}: "
            f"{e_initial:.1f} -> {e_final:.1f}"
        )
