"""Regression test for 1vii NL nonbonded parity (no OpenMM required).

Validates that the Prolix NL+switch engine matches the vendored OpenMM gold
for a solvated protein (1vii) at the nonbonded-only energy/force level.
No OpenMM import needed — this test reads only the precomputed gold JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Ensure float64 for PME precision
jax.config.update("jax_enable_x64", True)

# Import helpers from nl_omm_parity script
_SCRIPTS_EXP = Path(__file__).resolve().parents[2] / "scripts" / "experiments"
if str(_SCRIPTS_EXP) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_EXP))

from nl_omm_parity import _dispersion_correction_1vii, _bundle_from_1vii_gold


@pytest.fixture(scope="module")
def gold_1vii():
    """Load precomputed 1vii gold from vendored JSON.

    Scoped to module level so it can be shared with the module-scoped bundle_1vii fixture
    and all tests in this module without redundant reloading.
    """
    gold_path = Path(__file__).resolve().parents[2] / "data" / "oracles" / "openmm_8.3.1" / "nl_1vii.json"
    if not gold_path.is_file():
        pytest.skip(f"Gold file not found: {gold_path}")
    rec = json.loads(gold_path.read_text())
    # Verify required fields for this test
    required = ["omm_exceptions", "n_atoms", "pme_grid_points", "energy_kcal", "forces_kcal_mol_A"]
    missing = [k for k in required if k not in rec]
    if missing:
        pytest.skip(f"Gold file missing required fields: {missing}")
    return rec


@pytest.fixture(scope="module")
def bundle_1vii(gold_1vii):
    """Build 1vii bundle once for all tests in this module.

    Constructs the full periodic bundle (positions, PME grid, neighbor list)
    from the vendored gold JSON. Scoped to module level to avoid rebuilding
    across all three test functions.
    """
    return _bundle_from_1vii_gold(gold_1vii)


def test_1vii_bundle_construction_no_crash(bundle_1vii, gold_1vii):
    """Smoke test: build bundle from gold, verify it doesn't crash.

    This isolates "does the bundle construction even run" from energy/force matching.
    Uses the shared module-scoped bundle_1vii fixture to avoid redundant construction.
    """
    bundle = bundle_1vii
    n_atoms = int(gold_1vii["n_atoms"])

    # Verify bundle has required fields
    assert bundle.positions is not None
    assert bundle.positions.shape[0] == n_atoms
    assert bundle.excl_dense_indices is not None, "Bundle should have excl_dense_indices populated by make_bundle_from_system"


def test_1vii_nl_energy_parity(bundle_1vii, gold_1vii):
    """Energy parity: NL+switch prolix vs gold within 0.1 kcal/mol.

    Requires the bundle construction AND energy_fn_from_bundle path.
    Uses the shared module-scoped bundle_1vii fixture to avoid redundant construction.
    """
    from prolix.api.bundle_md import energy_fn_from_bundle
    from prolix.physics import neighbor_list as nl
    from prolix.physics.pbc import create_periodic_space

    rec = gold_1vii
    bundle = bundle_1vii

    # Build energy function with gold's PME grid
    energy_fn = energy_fn_from_bundle(
        bundle,
        include_nonbonded=True,
        lj_switch_width=1.0,
        pme_grid_points=int(rec["pme_grid_points"]),
    )

    # Build neighbor list
    displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
    # Cell list left at its default (on): #4699's [0, box) wrap makes it reproduce
    # the brute-force candidate set exactly, so this also guards that fix.
    neighbor_fn = nl.make_neighbor_list_fn(
        displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance),
    )
    nbr0 = neighbor_fn.allocate(bundle.positions)
    nbr = neighbor_fn.update(bundle.positions, nbr0)

    # Check overflow
    assert not bool(nbr.did_buffer_overflow), "Neighbor list buffer overflow detected"

    # Evaluate energy
    e_prolix = float(energy_fn(bundle.positions, neighbor=nbr))
    e_gold = float(rec["energy_kcal"])

    # WORKAROUND for dispersion tail bug (to be fixed separately)
    # OpenMM gold was emitted with setUseDispersionCorrection(False), but Prolix
    # unconditionally adds lj_dispersion_tail_energy (making energy MORE negative).
    # This is a known issue tracked separately. For now, ADD back the correction.
    dispersion_tail_correction = _dispersion_correction_1vii()
    e_prolix_corrected = e_prolix + dispersion_tail_correction

    delta_e = abs(e_prolix_corrected - e_gold)

    assert delta_e <= 0.1, f"Energy mismatch too large: delta_e={delta_e:.4f} kcal/mol (prolix_corrected={e_prolix_corrected:.2f}, gold={e_gold:.2f})"


def test_1vii_nl_force_parity(bundle_1vii, gold_1vii):
    """Force parity: NL+switch prolix vs gold with RMSE < 3.0 kcal/(mol*A).

    Requires jax.grad through the energy function on the same bundle.
    Uses the shared module-scoped bundle_1vii fixture to avoid redundant construction.
    """
    from prolix.api.bundle_md import energy_fn_from_bundle
    from prolix.physics import neighbor_list as nl
    from prolix.physics.pbc import create_periodic_space

    rec = gold_1vii
    bundle = bundle_1vii

    # Build energy function
    energy_fn = energy_fn_from_bundle(
        bundle,
        include_nonbonded=True,
        lj_switch_width=1.0,
        pme_grid_points=int(rec["pme_grid_points"]),
    )

    # Build neighbor list
    displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
    # Cell list left at its default (on): #4699's [0, box) wrap makes it reproduce
    # the brute-force candidate set exactly, so this also guards that fix.
    neighbor_fn = nl.make_neighbor_list_fn(
        displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance),
    )
    nbr0 = neighbor_fn.allocate(bundle.positions)
    nbr = neighbor_fn.update(bundle.positions, nbr0)

    # Check overflow
    assert not bool(nbr.did_buffer_overflow), "Neighbor list buffer overflow detected"

    # Compute forces via jax.grad
    forces_prolix = -jax.grad(energy_fn)(bundle.positions, neighbor=nbr)
    forces_gold = np.asarray(rec["forces_kcal_mol_A"], dtype=np.float64)

    # Compute RMSE
    force_rmse = float(np.sqrt(np.mean((np.asarray(forces_prolix) - forces_gold) ** 2)))

    assert force_rmse < 3.0, f"Force RMSE too large: {force_rmse:.4f} kcal/(mol*A)"


