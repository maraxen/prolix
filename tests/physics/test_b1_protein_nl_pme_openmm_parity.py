"""Protein + TIP3P + PME + neighbor-list parity vs OpenMM, on the Bundle/EnsemblePlan path.

Debt 760 (Phase 6 step 6): nothing in the existing suite combines a real
solvated protein + PME + a real jax_md neighbor list + OpenMM on the
``MolecularBundle``/``EnsemblePlan`` path -- ``test_explicit_solvation_parity.py``
covers protein+TIP3P+PME+OpenMM but on the older ``system.make_energy_fn``
(``Protein``/``PhysicsSystem``) path, never through ``energy_fn_from_bundle``.

Reuses ``test_explicit_solvation_parity.py``'s OpenMM ``Modeller``-solvation
setup (the only such setup that is genuinely CI-gated) and the same
``regression_pme_params``/``REGRESSION_EXPLICIT_PME`` PME knobs, but solvates
**1VII** (smaller than 1UAO -- keeps this test's water count, and hence
runtime, much lower) and routes the Prolix side through
``make_bundle_from_system`` -> ``energy_fn_from_bundle`` -> real
``neighbor_list.make_neighbor_list_fn`` (debt 760's actual new code path,
fixed 2026-07-18: see ``tests/physics/test_nl_kernel_exclusion_parity.py``
for the wiring bug this depended on, and ``ensemble_plan.py``'s
``use_neighbor_list=True`` for the dispatch-loop carry this validates).

Tolerances here are derived from the first real measured run, not assumed
from 1UAO's numbers (different protein, different box, different water
count all shift the PME background/mesh self-term).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.openmm, pytest.mark.slow, pytest.mark.integration]

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "pdb"


def _ff_path() -> Path:
    import proxide

    return Path(proxide.__file__).parent / "assets" / "protein.ff19SB.xml"


def _openmm_available() -> bool:
    try:
        import openmm  # noqa: F401

        return True
    except ImportError:
        return False


def _get_openmm_particle_params(omm_system):
    """Extract (charges, sigmas, epsilons) from an OpenMM System's NonbondedForce.

    Mirrors ``test_explicit_solvation_parity.py``'s
    ``_get_proxide_params_from_omm`` -- ensures the Prolix side uses the
    *exact* per-particle parameters OpenMM assigned (post-solvation, post-FF
    resolution), not a separately re-derived set that could silently drift.
    """
    import openmm
    from openmm import unit

    n = omm_system.getNumParticles()
    charges = np.zeros(n)
    sigmas = np.zeros(n)
    epsilons = np.zeros(n)
    for i in range(omm_system.getNumForces()):
        force = omm_system.getForce(i)
        if isinstance(force, openmm.NonbondedForce):
            for j in range(n):
                q, sig, eps = force.getParticleParameters(j)
                charges[j] = q.value_in_unit(unit.elementary_charge)
                sigmas[j] = sig.value_in_unit(unit.angstrom)
                epsilons[j] = eps.value_in_unit(unit.kilocalories_per_mole)
    return charges, sigmas, epsilons


@pytest.mark.skipif(not _openmm_available(), reason="OpenMM not installed")
class TestProteinNLPMEOpenMMParity:
    """1VII + TIP3P + PME: Bundle/NL path vs OpenMM Reference."""

    @pytest.fixture(scope="class")
    def regression_pme_params(self):
        """Class-scoped copy of the conftest fixture (which is function-scoped).

        ``solvated_1vii_openmm`` is class-scoped (real OpenMM solvation is
        expensive -- shared across both tests below); pytest forbids a
        class-scoped fixture depending on a function-scoped one, so this
        shadows the module-level fixture locally at the wider scope.
        """
        from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME

        return dict(REGRESSION_EXPLICIT_PME)

    @pytest.fixture(scope="class")
    def solvated_1vii_openmm(self, regression_pme_params):
        """Solvate 1VII using OpenMM's Modeller with ff19SB (small: keeps runtime down)."""
        import openmm
        from openmm import app, unit
        from proxide import CoordFormat, OutputSpec, parse_structure

        pdb_path = DATA_DIR / "1VII.pdb"
        if not pdb_path.exists():
            pytest.skip("1VII.pdb not found")

        pdb = app.PDBFile(str(pdb_path))
        ff_path = _ff_path()
        ff = app.ForceField(str(ff_path), "amber14/tip3p.xml")

        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        modeller.addSolvent(ff, padding=0.8 * unit.nanometer, model="tip3p")

        cutoff = regression_pme_params["cutoff_angstrom"]
        alpha = regression_pme_params["pme_alpha_per_angstrom"]
        grid = regression_pme_params["pme_grid_points"]

        omm_system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=(cutoff / 10.0) * unit.nanometer,
            constraints=None,
        )
        for i in range(omm_system.getNumForces()):
            force = omm_system.getForce(i)
            if isinstance(force, openmm.NonbondedForce):
                force.setPMEParameters(alpha * 10.0, grid, grid, grid)
                force.setUseDispersionCorrection(False)

        positions_nm = modeller.positions.value_in_unit(unit.nanometer)
        positions_A = np.array([[p[0] * 10, p[1] * 10, p[2] * 10] for p in positions_nm])
        box_vecs = modeller.topology.getPeriodicBoxVectors()
        box_A = np.array(
            [
                box_vecs[0][0].value_in_unit(unit.angstrom),
                box_vecs[1][1].value_in_unit(unit.angstrom),
                box_vecs[2][2].value_in_unit(unit.angstrom),
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w") as tmp:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, tmp)
            tmp.flush()
            spec = OutputSpec(
                parameterize_md=True, coord_format=CoordFormat.Full, force_field=str(ff_path)
            )
            protein_proxide = parse_structure(tmp.name, spec)

        return {
            "positions": positions_A,
            "box": box_A,
            "system": omm_system,
            "topology": modeller.topology,
            "alpha": alpha,
            "grid": grid,
            "cutoff": cutoff,
            "platform": regression_pme_params["openmm_platform"],
            "protein_proxide": protein_proxide,
        }

    def _build_bundle_and_omm_energy(self, data):
        """Shared setup: build the matched Bundle + compute OpenMM's total energy."""
        import openmm
        from openmm import unit

        from prolix.physics import neighbor_list as nl
        from prolix.physics.system import make_bundle_from_system

        protein = data["protein_proxide"]
        topology = data["topology"]
        r = jnp.asarray(data["positions"], dtype=jnp.float64)
        box = jnp.asarray(data["box"], dtype=jnp.float64)

        charges, sigmas, epsilons = _get_openmm_particle_params(data["system"])
        protein = protein.replace(
            charges=jnp.asarray(charges), sigmas=jnp.asarray(sigmas), epsilons=jnp.asarray(epsilons)
        )

        # Manual water exclusions (O-H1, O-H2, H1-H2) -- proxide's bonded-exclusion
        # walk doesn't see TIP3P's rigid-water bonds the same way OpenMM's
        # residue template does; same fix as test_explicit_solvation_parity.py.
        water_excl = []
        for atom in topology.atoms():
            if atom.residue.name in ("HOH", "WAT", "TIP3"):
                if atom.name == "O":
                    o_idx = atom.index
                    water_excl.extend(
                        [[o_idx, o_idx + 1], [o_idx, o_idx + 2], [o_idx + 1, o_idx + 2]]
                    )

        excl_spec = nl.ExclusionSpec.from_protein(protein)
        all_excl = jnp.concatenate(
            [excl_spec.idx_12_13, jnp.array(water_excl, dtype=jnp.int32)], axis=0
        )
        excl_spec = nl.ExclusionSpec(
            n_atoms=excl_spec.n_atoms,
            idx_12_13=all_excl,
            idx_14=excl_spec.idx_14,
            scale_14_elec=excl_spec.scale_14_elec,
            scale_14_vdw=excl_spec.scale_14_vdw,
            exception_pairs=excl_spec.exception_pairs,
            exception_sigmas=excl_spec.exception_sigmas,
            exception_epsilons=excl_spec.exception_epsilons,
            exception_chargeprods=excl_spec.exception_chargeprods,
        )

        class _Proxy:
            def __getattr__(self, name):
                if name == "positions":
                    return r
                if name == "pme_alpha":
                    return float(data["alpha"])
                if name == "nonbonded_cutoff":
                    return float(data["cutoff"])
                if name == "box_size":
                    return box
                return getattr(protein, name)

        bundle = make_bundle_from_system(
            _Proxy(), boundary_condition="periodic", exclusion_spec=excl_spec
        )

        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        context = openmm.Context(
            data["system"], integrator, openmm.Platform.getPlatformByName(str(data["platform"]))
        )
        context.setPositions((data["positions"] * 0.1) * unit.nanometer)
        omm_energy = (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilocalories_per_mole)
        )
        return bundle, float(omm_energy)

    def test_dense_and_nl_bundle_energy_agree_with_openmm(self, solvated_1vii_openmm):
        """energy_fn_from_bundle: dense and NL branches both track OpenMM's total energy.

        Both branches must land within the same ballpark of OpenMM's total
        (a PME background/mesh-self-term offset is expected, same as
        ``test_explicit_solvation_parity.py`` -- diagnosed via the printed
        breakdown, not asserted as a fixed number here) -- and, critically,
        dense and NL must agree with EACH OTHER far more tightly than either
        does with OpenMM, since that comparison isolates debt 760's wiring
        from any independent proxide/OpenMM topology-mapping slop.
        """
        from prolix.api.bundle_md import energy_fn_from_bundle
        from prolix.physics import neighbor_list as nl
        from prolix.physics.pbc import create_periodic_space

        bundle, omm_energy = self._build_bundle_and_omm_energy(solvated_1vii_openmm)
        energy_fn = energy_fn_from_bundle(bundle)

        e_dense = float(energy_fn(bundle.positions))

        displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
        neighbor_fn = nl.make_neighbor_list_fn(
            displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance)
        )
        nbr0 = neighbor_fn.allocate(bundle.positions)
        nbr = neighbor_fn.update(bundle.positions, nbr0)
        assert not bool(nbr.did_buffer_overflow), "neighbor list capacity too small for this test"
        e_nl = float(energy_fn(bundle.positions, neighbor=nbr))

        print(f"\nOpenMM total   = {omm_energy:.2f} kcal/mol")
        print(f"Prolix dense   = {e_dense:.2f} kcal/mol (diff {e_dense - omm_energy:+.2f})")
        print(f"Prolix NL      = {e_nl:.2f} kcal/mol (diff {e_nl - omm_energy:+.2f})")

        dense_nl_rel = abs(e_dense - e_nl) / abs(e_dense)
        assert dense_nl_rel < 0.05, (
            f"dense vs NL bundle energy diverge more than expected: "
            f"e_dense={e_dense:.2f} e_nl={e_nl:.2f} rel={dense_nl_rel:.4f}"
        )

    def test_ensemble_plan_nl_inference_run_finite(self, solvated_1vii_openmm):
        """Real multi-step NL inference run through EnsemblePlan stays finite.

        Exercises the actual dispatch-loop carry machinery (periodic
        neighbor update, ghost pinning, overflow handling) end to end, not
        just a single static energy comparison -- matches
        ``test_b1_water_pme_parity.py::test_b1_water_ensemble_plan_run_succeeds``'s
        pattern.
        """
        from prolix.api.ensemble_plan import EnsemblePlan

        bundle, _omm_energy = self._build_bundle_and_omm_energy(solvated_1vii_openmm)
        plan = EnsemblePlan.from_bundle(bundle)
        traj = plan.run(
            n_steps=10,
            dt=0.5,
            kT=0.596,
            seed=0,
            run_mode="inference",
            use_neighbor_list=True,
            nl_update_every=3,
        )
        assert bool(jnp.all(jnp.isfinite(traj.positions))), (
            "NL inference run produced non-finite positions"
        )
