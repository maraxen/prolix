"""Charge injection self-consistency: prolix consumes arrays; assignment lives in proxide."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import pytest
from jax_md import space
from proxide import CoordFormat, OutputSpec, parse_structure

from prolix.physics import neighbor_list as nl
from prolix.physics import system as physics_system

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
  pdb_path = DATA_DIR / "1CRN.pdb"
  spec = OutputSpec()
  spec.parameterize_md = True
  spec.force_field = str(FF_PATH)
  spec.coord_format = CoordFormat.Full
  return parse_structure(str(pdb_path), spec)


def test_scaled_charges_change_explicit_energy(crambin_protein):
  """Same geometry, different charge vectors → different electrostatic energy (no espaloma dep)."""
  displacement_fn, _ = space.free()
  exclusion_spec = nl.ExclusionSpec.from_protein(crambin_protein)
  coords = crambin_protein.coordinates

  efn_base = physics_system.make_energy_fn(
    displacement_fn,
    crambin_protein,
    exclusion_spec=exclusion_spec,
    implicit_solvent=False,
  )
  e0 = efn_base(coords)

  perturbed = dataclasses.replace(
    crambin_protein,
    charges=jnp.asarray(crambin_protein.charges) * 1.1,
  )
  efn_pert = physics_system.make_energy_fn(
    displacement_fn,
    perturbed,
    exclusion_spec=exclusion_spec,
    implicit_solvent=False,
  )
  e1 = efn_pert(coords)

  assert jnp.isfinite(e0) and jnp.isfinite(e1)
  assert not bool(jnp.allclose(e0, e1, rtol=1e-6))
