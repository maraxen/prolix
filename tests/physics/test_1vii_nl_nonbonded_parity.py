"""Regression test for 1vii NL nonbonded-only energy/force parity vs vendored OpenMM gold.

This test reads the vendored gold JSON (no OpenMM required at test time) and validates
that the zero-bonded-topology MolecularBundle + neighbor_list + energy_fn_from_bundle
path matches OpenMM's nonbonded-only energy/forces within tight bands.

Per #4383 spec AC6: read only the vendored gold, no OpenMM required.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Always enable x64 precision for PME/Coulomb accuracy (per spec precision lock)
jax.config.update("jax_enable_x64", True)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "oracles" / "openmm_8.3.1"


def _load_1vii_gold() -> dict:
    """Load the vendored 1vii NL gold record (no OpenMM needed)."""
    gold_path = DATA_DIR / "nl_1vii.json"
    if not gold_path.exists():
        pytest.skip(f"1vii gold not found at {gold_path}")
    return json.loads(gold_path.read_text())


def _bundle_from_1vii_gold(rec: dict):
    """Build MolecularBundle from gold record (imported from emit script for consistency)."""
    from scripts.experiments.nl_omm_parity import _bundle_from_1vii_gold as helper

    return helper(rec)


class TestOneViiNLNonbondedParity:
    """1vii NL nonbonded-only energy/force parity (no bonded topology)."""

    @pytest.fixture
    def gold(self):
        """Vendored 1vii gold record."""
        return _load_1vii_gold()

    @pytest.fixture
    def bundle(self, gold):
        """MolecularBundle built from gold via make_bundle_from_system."""
        return _bundle_from_1vii_gold(gold)

    def test_bundle_structure(self, gold, bundle):
        """Verify bundle has correct structure: zero bonded, populated exclusions."""
        assert bundle.n_atoms == gold["n_atoms"]
        assert bundle.positions.shape[0] == gold["n_atoms"]
        assert bundle.charges.shape[0] == gold["n_atoms"]

        # Zero bonded topology (per spec decomposition method)
        assert bundle.n_bonds == 0
        assert bundle.n_angles == 0
        assert bundle.n_dihedrals == 0
        assert bundle.n_impropers == 0

        # Populated exclusion spec (per spec C1)
        assert bundle.excl_dense_indices is not None
        assert bundle.excl_dense_scales_vdw is not None
        assert bundle.excl_dense_scales_elec is not None
        assert bundle.exception_pairs.shape[0] > 0  # Should have nonzero exceptions

    def test_neighbor_list_no_overflow(self, gold, bundle):
        """Verify neighbor list does not overflow (mandatory gate per spec C4)."""
        from prolix.physics import neighbor_list as nl
        from prolix.physics.pbc import create_periodic_space

        displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
        neighbor_fn = nl.make_neighbor_list_fn(
            displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance)
        )
        nbr0 = neighbor_fn.allocate(bundle.positions)
        nbr = neighbor_fn.update(bundle.positions, nbr0)

        assert not bool(nbr.did_buffer_overflow), "neighbor list capacity overflow"

    def test_nonbonded_energy_parity(self, gold, bundle):
        """Energy matches gold within band (AC2).

        AC2 band: ABS(delta_e_kcal) <= 0.1 (spec §37)
        """
        from prolix.api.bundle_md import energy_fn_from_bundle
        from prolix.physics import neighbor_list as nl
        from prolix.physics.pbc import create_periodic_space

        # Build neighbor list
        displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
        neighbor_fn = nl.make_neighbor_list_fn(
            displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance)
        )
        nbr0 = neighbor_fn.allocate(bundle.positions)
        nbr = neighbor_fn.update(bundle.positions, nbr0)
        assert not bool(nbr.did_buffer_overflow)

        # Evaluate energy via NL branch
        energy_fn = energy_fn_from_bundle(
            bundle,
            include_nonbonded=True,
            lj_switch_width=1.0,
            pme_grid_points=int(gold["pme_grid_points"]),
        )
        e_prolix = float(energy_fn(bundle.positions, neighbor=nbr))

        # Compare to gold (nonbonded-only, ForceGroup(0) from OpenMM)
        e_gold = float(gold["energy_kcal"])
        delta_e = e_prolix - e_gold

        print(f"\nEnergy parity:")
        print(f"  Gold (OpenMM NL)      = {e_gold:12.2f} kcal/mol")
        print(f"  Prolix (NL branch)    = {e_prolix:12.2f} kcal/mol")
        print(f"  Delta                 = {delta_e:+12.4f} kcal/mol")

        assert abs(delta_e) <= 0.1, (
            f"Energy mismatch outside band: delta={delta_e:.4f} kcal/mol "
            f"(gold={e_gold:.2f}, prolix={e_prolix:.2f})"
        )

    def test_force_parity(self, gold, bundle):
        """Forces match gold within band (AC3).

        AC3 band: force_rmse_kcal_mol_A < 3.0 (spec §37)
        """
        from prolix.api.bundle_md import energy_fn_from_bundle
        from prolix.physics import neighbor_list as nl
        from prolix.physics.pbc import create_periodic_space

        # Build neighbor list
        displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
        neighbor_fn = nl.make_neighbor_list_fn(
            displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance)
        )
        nbr0 = neighbor_fn.allocate(bundle.positions)
        nbr = neighbor_fn.update(bundle.positions, nbr0)
        assert not bool(nbr.did_buffer_overflow)

        # Evaluate energy and forces
        energy_fn = energy_fn_from_bundle(
            bundle,
            include_nonbonded=True,
            lj_switch_width=1.0,
            pme_grid_points=int(gold["pme_grid_points"]),
        )

        grad_fn = jax.grad(energy_fn)
        f_prolix = grad_fn(bundle.positions, neighbor=nbr)

        # Compare to gold forces
        f_gold = jnp.asarray(gold["forces_kcal_mol_A"], dtype=jnp.float32)
        force_rmse = float(jnp.sqrt(jnp.mean((f_prolix - f_gold) ** 2)))

        # Per-atom max deviation for diagnostics
        dev_per_atom = jnp.sqrt(jnp.sum((f_prolix - f_gold) ** 2, axis=1))
        max_atom_dev = float(jnp.max(dev_per_atom))
        max_atom_idx = int(jnp.argmax(dev_per_atom))

        print(f"\nForce parity:")
        print(f"  RMSE                  = {force_rmse:.4f} kcal/mol/Å")
        print(f"  Max per-atom dev      = {max_atom_dev:.4f} kcal/mol/Å (atom {max_atom_idx})")

        assert force_rmse < 3.0, (
            f"Force RMSE outside band: {force_rmse:.4f} kcal/mol/Å "
            f"(max per-atom: {max_atom_dev:.4f} at atom {max_atom_idx})"
        )
