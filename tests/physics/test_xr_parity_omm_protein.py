"""XR-PARITY-OMM-PROTEIN: vacuum 2GB1 OpenMM static ΔE/ΔF on bundle energy path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PDB = Path("data/pdb/2GB1.pdb")

pytestmark = [pytest.mark.openmm, pytest.mark.slow, pytest.mark.integration]  # XA-CI


@pytest.mark.slow
def test_2gb1_vacuum_openmm_static_energy_force_parity():
    """AC1: shared amber14 params → |ΔE|≤0.1 and force_rmse<3.0."""
    if not PDB.exists():
        pytest.skip(f"missing fixture {PDB}")

    import jax
    import jax.numpy as jnp
    import openmm as mm
    from openmm import app, unit

    from prolix.api.bundle_md import energy_fn_from_bundle
    from prolix.physics.system import make_bundle_from_system

    from .fixtures_openmm_parity import (
        build_exclusion_spec,
        build_prolix_nonbonded_system,
        extract_bonded_params,
        extract_nonbonded_params,
    )

    jax.config.update("jax_enable_x64", True)

    pdb = app.PDBFile(str(PDB))
    ff = app.ForceField("amber14-all.xml")
    omm_system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        removeCMMotion=False,
    )
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    bonded = extract_bonded_params(omm_system)
    nb = extract_nonbonded_params(omm_system)
    sys, _disp = build_prolix_nonbonded_system(nb, bonded, positions)
    excl = build_exclusion_spec(omm_system, positions.shape[0])
    bundle = make_bundle_from_system(
        sys, boundary_condition="free", exclusion_spec=excl
    )

    pos = jnp.asarray(positions, dtype=jnp.float64)
    e_fn = energy_fn_from_bundle(bundle)
    e_p, g_p = jax.value_and_grad(e_fn)(pos)
    f_p = -np.asarray(g_p)

    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = mm.Context(
        omm_system, integrator, mm.Platform.getPlatformByName("Reference")
    )
    ctx.setPositions(pdb.positions)
    state = ctx.getState(getEnergy=True, getForces=True)
    kj_to_kcal = 1.0 / 4.184
    e_omm = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * kj_to_kcal
    f_omm = (
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        * kj_to_kcal
        / 10.0
    )

    delta_e = abs(float(e_p) - float(e_omm))
    force_rmse = float(np.sqrt(np.mean((f_p - f_omm) ** 2)))
    assert delta_e <= 0.1, f"|ΔE|={delta_e:.4f} kcal/mol exceeds 0.1"
    assert force_rmse < 3.0, f"force_rmse={force_rmse:.4f} exceeds 3.0"
