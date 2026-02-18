"""Tests for the OutputSpec → Protein → make_energy_fn migration.

Validates:
1. Full-format Protein has consistent shapes (coords ↔ physics params)
2. ExclusionSpec builds correctly from Protein
3. Energy computation produces finite, non-zero values
4. Legacy system_params_to_protein converter works correctly
5. RATTLE constraints are correctly passed through the converter
"""

import pytest
import jax.numpy as jnp
from pathlib import Path
from proxide import OutputSpec, parse_structure, CoordFormat
from prolix.physics import system as physics_system
from prolix.physics import neighbor_list as nl
from jax_md import space

# Path to test data and force field
DATA_DIR = Path(__file__).parent.parent / "data" / "pdb"
FF_PATH = (
    Path(__file__).parent.parent.parent
    / "proxide"
    / "src"
    / "proxide"
    / "assets"
    / "protein.ff19SB.xml"
)


@pytest.fixture
def crambin_protein():
    """Load 1CRN with CoordFormat.Full and MD parameterization."""
    pdb_path = DATA_DIR / "1CRN.pdb"
    spec = OutputSpec()
    spec.parameterize_md = True
    spec.force_field = str(FF_PATH)
    spec.coord_format = CoordFormat.Full
    return parse_structure(str(pdb_path), spec)


class TestOutputSpecProteinPipeline:
    """Tests for the new OutputSpec → Protein pipeline."""

    def test_full_format_shape_consistency(self, crambin_protein):
        """Coords N must match charges N (the Padding Paradox fix)."""
        protein = crambin_protein
        assert protein.format == "Full"
        n_atoms = protein.coordinates.shape[0]
        assert protein.charges.shape == (n_atoms,)
        assert protein.sigmas.shape == (n_atoms,)
        assert protein.epsilons.shape == (n_atoms,)

    def test_exclusion_spec_from_protein(self, crambin_protein):
        """ExclusionSpec builds correctly from Protein with bonds."""
        protein = crambin_protein
        assert protein.bonds is not None
        assert protein.bonds.shape[0] > 0

        excl = nl.ExclusionSpec.from_protein(protein)
        assert excl.idx_12_13.shape[0] > 0
        assert excl.idx_14.shape[0] > 0
        assert excl.n_atoms == protein.charges.shape[0]

    def test_energy_computation_finite(self, crambin_protein):
        """Full pipeline: parse → ExclusionSpec → make_energy_fn → energy."""
        protein = crambin_protein
        displacement_fn, _ = space.free()

        exclusion_spec = nl.ExclusionSpec.from_protein(protein)
        energy_fn = physics_system.make_energy_fn(
            displacement_fn, protein,
            exclusion_spec=exclusion_spec,
            implicit_solvent=False,
        )

        coords = protein.coordinates
        energy = energy_fn(coords)
        assert jnp.isfinite(energy), f"Energy is not finite: {energy}"
        assert energy != 0.0, "Energy should be non-zero for a real protein"

    def test_energy_with_implicit_solvent(self, crambin_protein):
        """Energy computation with implicit solvent (GBSA)."""
        protein = crambin_protein

        # Implicit solvent requires radii — skip if not available
        if protein.radii is None:
            pytest.skip("Protein does not have radii for implicit solvent")

        displacement_fn, _ = space.free()
        exclusion_spec = nl.ExclusionSpec.from_protein(protein)
        energy_fn = physics_system.make_energy_fn(
            displacement_fn, protein,
            exclusion_spec=exclusion_spec,
            implicit_solvent=True,
        )

        coords = protein.coordinates
        energy = energy_fn(coords)
        assert jnp.isfinite(energy), f"Energy is not finite: {energy}"


class TestLegacyCompatShim:
    """Tests for system_params_to_protein backward compatibility."""

    def test_system_params_to_protein_basic(self):
        """Converter creates a valid Protein from a legacy dict."""
        from prolix.compat import system_params_to_protein

        charges = jnp.ones((5,))
        legacy = {
            "charges": charges,
            "bonds": jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
            "bond_params": jnp.zeros((2, 2)),
            "angles": jnp.zeros((0, 3), dtype=jnp.int32),
            "angle_params": jnp.zeros((0, 2)),
            "sigmas": jnp.ones((5,)) * 3.0,
            "epsilons": jnp.ones((5,)) * 0.1,
        }

        with pytest.warns(DeprecationWarning):
            protein = system_params_to_protein(legacy)

        assert protein.charges.shape == (5,)
        assert protein.bonds.shape == (2, 2)

    def test_rattle_constraints_passed(self):
        """constrained_bonds and constrained_bond_lengths round-trip."""
        from prolix.compat import system_params_to_protein

        legacy = {
            "charges": jnp.zeros((2,)),
            "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
            "bond_params": jnp.zeros((0, 2)),
            "angles": jnp.zeros((0, 3), dtype=jnp.int32),
            "angle_params": jnp.zeros((0, 2)),
            "sigmas": jnp.ones((2,)) * 3.0,
            "epsilons": jnp.ones((2,)) * 0.1,
            "constrained_bonds": jnp.array([[0, 1]], dtype=jnp.int32),
            "constrained_bond_lengths": jnp.array([1.0], dtype=jnp.float32),
        }

        with pytest.warns(DeprecationWarning):
            protein = system_params_to_protein(legacy)

        assert protein.constrained_bonds is not None
        assert protein.constrained_bonds.shape == (1, 2)
        assert protein.constrained_bond_lengths.shape == (1,)
        assert protein.constrained_bond_lengths[0] == 1.0
