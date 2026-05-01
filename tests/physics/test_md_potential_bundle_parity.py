"""Parity: md_potential_bundle vs jax_md.quantity for scalar energies."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax_md import quantity

from prolix.physics import pbc, system
from prolix.physics.md_potential_bundle import (
  make_force_fn_like_canonicalize,
  sum_real_atoms_per_batch,
  value_energy_and_forces,
)
from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water


@pytest.fixture(scope="module")
def _float64():
  jax.config.update("jax_enable_x64", True)
  yield


def test_value_energy_matches_canonicalize_force(_float64):
  n_waters = 2
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  displacement_fn, _ = pbc.create_periodic_space(box_vec)
  sys_dict = _prolix_params_pure_water(n_waters)
  energy_fn = system.make_energy_fn(
    displacement_fn,
    sys_dict,
    box=box_vec,
    use_pbc=True,
    implicit_solvent=False,
    pme_grid_points=32,
    pme_alpha=0.34,
    cutoff_distance=9.0,
    strict_parameterization=False,
  )
  R = jnp.asarray(positions_a, dtype=jnp.float64)
  f_canon = quantity.canonicalize_force(energy_fn)(R, box=box_vec)
  bundle = value_energy_and_forces(energy_fn, R, box=box_vec)
  assert bundle.forces.shape == f_canon.shape
  assert jnp.allclose(bundle.forces, f_canon, rtol=1e-10, atol=1e-10)


def test_make_force_fn_like_canonicalize_matches(_float64):
  n_waters = 2
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  displacement_fn, _ = pbc.create_periodic_space(box_vec)
  sys_dict = _prolix_params_pure_water(n_waters)
  energy_fn = system.make_energy_fn(
    displacement_fn,
    sys_dict,
    box=box_vec,
    use_pbc=True,
    implicit_solvent=False,
    pme_grid_points=32,
    pme_alpha=0.34,
    cutoff_distance=9.0,
    strict_parameterization=False,
  )
  n_atoms = n_waters * 3
  template_R = jnp.zeros((n_atoms, 3), dtype=jnp.float64)
  fn = make_force_fn_like_canonicalize(
    energy_fn, template_R=template_R, template_kwargs={"box": box_vec}
  )
  R = jnp.asarray(positions_a, dtype=jnp.float64)
  f_wrap = fn(R, box=box_vec)
  f_ref = quantity.canonicalize_force(energy_fn)(R, box=box_vec)
  assert jnp.allclose(f_wrap, f_ref, rtol=1e-10, atol=1e-10)


def test_sum_real_atoms_per_batch(_float64):
  q = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
  m = jnp.array([True, True, False])
  s = sum_real_atoms_per_batch(q, m)
  assert jnp.allclose(s, jnp.array([3.0, 0.0, 0.0]))
