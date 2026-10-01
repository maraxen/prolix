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

from nl_omm_parity import _bundle_from_1vii_gold


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


@pytest.mark.xfail(
    strict=True,
    reason="Real residual, previously masked: with the dispersion tail removed exactly "
    "(use_dispersion_correction=False) prolix is -9799.57 vs gold -9790.92 kcal/mol, "
    "|dE|=8.65 kcal/mol (titanix CPU, 2026-10-01). The earlier PASS (|dE|=0.034) added "
    "a hand-set +156.8 kcal/mol 'dispersion correction' whereas the actual 1vii tail is "
    "~-148.2 kcal/mol, so the constant absorbed this residual. Same class as backlog "
    "#5020 (NL bundle energy vs OpenMM residual). Band 0.1 kcal/mol is pre-registered in "
    "nl_omm_parity.bth.toml and is NOT loosened here.",
)
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
        use_dispersion_correction=False,  # gold: setUseDispersionCorrection(False)
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

    delta_e = abs(e_prolix - e_gold)

    assert delta_e <= 0.1, f"Energy mismatch too large: delta_e={delta_e:.4f} kcal/mol (prolix={e_prolix:.2f}, gold={e_gold:.2f})"


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
        use_dispersion_correction=False,  # gold: setUseDispersionCorrection(False)
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




def test_1vii_dispersion_correction_flag_removes_only_the_tail(bundle_1vii, gold_1vii):
    """use_dispersion_correction toggles exactly the isotropic LJ tail.

    The difference must equal ``explicit_corrections.lj_dispersion_tail_energy`` on
    the same system. (nl_omm_parity.py used to add back a hand-set +156.8 kcal/mol
    instead, which is not this tail -- see the xfail on test_1vii_nl_energy_parity.)
    """
    from prolix.api.bundle_md import energy_fn_from_bundle, physics_system_from_bundle
    from prolix.physics import explicit_corrections
    from prolix.physics import neighbor_list as nl
    from prolix.physics.pbc import create_periodic_space

    rec = gold_1vii
    bundle = bundle_1vii
    kw = {"include_nonbonded": True, "lj_switch_width": 1.0,
          "pme_grid_points": int(rec["pme_grid_points"])}
    e_on_fn = energy_fn_from_bundle(bundle, **kw)
    e_off_fn = energy_fn_from_bundle(bundle, use_dispersion_correction=False, **kw)

    displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
    neighbor_fn = nl.make_neighbor_list_fn(
        displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance),
    )
    nbr = neighbor_fn.update(bundle.positions, neighbor_fn.allocate(bundle.positions))

    diff = float(e_on_fn(bundle.positions, neighbor=nbr)) - float(
        e_off_fn(bundle.positions, neighbor=nbr)
    )
    sys = physics_system_from_bundle(bundle, bundle.positions,
                                     pme_grid_points=int(rec["pme_grid_points"]))
    safe_sig = jnp.where(sys.atom_mask, sys.sigmas, 1.0)
    safe_eps = jnp.where(sys.atom_mask, sys.epsilons, 0.0)
    tail = float(explicit_corrections.lj_dispersion_tail_energy(
        jnp.asarray(sys.box_size), safe_sig, safe_eps, float(sys.nonbonded_cutoff),
        sys.atom_mask,
    ))
    assert tail < -100.0, f"1vii tail should be ~-157 kcal/mol, got {tail:.2f}"
    assert diff == pytest.approx(tail, abs=1e-6)
