"""Two-stage runtime consumption audit (FR-2) and MolecularShapeSpec bucket tests."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from prolix.api.bundle_md import displacement_fn_for_bundle, pbc_box_for_bundle, physics_system_from_bundle
from prolix.api.bundle_stack import can_stack_molecular_bundles
from prolix.batched_energy import single_padded_energy, single_padded_force
from prolix.governance.pending_knobs import DOCUMENTED_INERT_KNOBS, PENDING_EPIC_KNOBS
from prolix.physics import cmap, pbc
from prolix.physics.field_audit import FieldAuditProxy
from prolix.physics.system import make_bundle_from_system
from prolix.types.bundles import (
  CMAP_MAP_BUCKETS,
  CONSTRAINT_BUCKETS,
  MolecularShapeSpec,
  cmap_map_pad_length,
  constraint_pad_length,
)


def _make_minimal_protein_namespace(*, with_pbc: bool = True, is_triclinic: bool = False):
  from types import SimpleNamespace

  positions = jnp.array(
    [
      [1.0, 1.0, 1.0],
      [2.5, 1.0, 1.0],
      [2.5, 2.5, 1.0],
      [4.0, 2.5, 1.0],
      [5.5, 2.5, 2.5],
      [7.0, 4.0, 2.5],
    ],
    dtype=jnp.float32,
  )
  masses = jnp.array([12.0, 14.0, 12.0, 16.0, 15.999, 1.008], dtype=jnp.float32)
  charges = jnp.array([0.1, -0.1, 0.2, -0.2, -0.834, 0.834], dtype=jnp.float32)
  sigmas = jnp.full((6,), 3.4, dtype=jnp.float32)
  epsilons = jnp.full((6,), 0.1, dtype=jnp.float32)
  radii = jnp.full((6,), 1.5, dtype=jnp.float32)
  scaled_radii = jnp.full((6,), 1.2, dtype=jnp.float32)
  bonds = jnp.array([[0, 1], [1, 2], [2, 3]], dtype=jnp.int32)
  bond_params = jnp.array([[1.5, 300.0], [1.5, 300.0], [1.2, 500.0]], dtype=jnp.float32)
  angles = jnp.array([[0, 1, 2], [1, 2, 3]], dtype=jnp.int32)
  angle_params = jnp.array([[1.9, 50.0], [2.0, 50.0]], dtype=jnp.float32)
  dihedrals = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
  dihedral_params = jnp.array([[2.0, 0.0, 1.5]], dtype=jnp.float32)
  impropers = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
  improper_params = jnp.array([[2.0, 3.14159, 1.0]], dtype=jnp.float32)
  if not with_pbc:
    box_size = None
  elif is_triclinic:
    box_size = pbc.triclinic_box_from_lengths_angles(24.0, 24.0, 24.0, 60.0, 60.0, 60.0)
  else:
    box_size = jnp.array([24.0, 24.0, 24.0], dtype=jnp.float32)

  return SimpleNamespace(
    positions=positions,
    masses=masses,
    charges=charges,
    sigmas=sigmas,
    epsilons=epsilons,
    radii=radii,
    scaled_radii=scaled_radii,
    bonds=bonds,
    bond_params=bond_params,
    angles=angles,
    angle_params=angle_params,
    dihedrals=dihedrals,
    dihedral_params=dihedral_params,
    impropers=impropers,
    improper_params=improper_params,
    box_size=box_size,
  )


def test_cmap_map_buckets_and_constraint_buckets():
  """Verify CB-1 and CB-6: CMAP_MAP_BUCKETS minimum is 1 and MolecularShapeSpec fields stack."""
  assert CMAP_MAP_BUCKETS == (1, 8, 16)
  assert CMAP_MAP_BUCKETS[0] == 1
  assert CONSTRAINT_BUCKETS == (0, 256, 512, 1024, 2048, 4096, 8192)

  b1 = make_bundle_from_system(_make_minimal_protein_namespace(with_pbc=True))
  b2 = make_bundle_from_system(_make_minimal_protein_namespace(with_pbc=True))
  assert cmap_map_pad_length(b1.shape_spec) == 1
  assert constraint_pad_length(b1.shape_spec) == 0
  assert can_stack_molecular_bundles([b1, b2])

  # Changing is_triclinic or cmap_map_bucket_idx prevents invalid stacking
  spec_tri = dataclasses.replace(b2.shape_spec, is_triclinic=True)
  b2_tri = dataclasses.replace(b2, shape_spec=spec_tri)
  assert not can_stack_molecular_bundles([b1, b2_tri])

  # pbc_box_for_bundle returns (3,) when is_triclinic=False and (3, 3) when is_triclinic=True
  assert pbc_box_for_bundle(b1).shape == (3,)
  assert pbc_box_for_bundle(b2_tri).shape == (3, 3)


def test_cmap_safe_indexing_on_single_padded_map():
  """Verify CB-1: compute_cmap_energies safely clamps map_indices on (1, 24, 24, 16) coeffs."""
  dummy_coeffs = jnp.zeros((1, 24, 24, 16), dtype=jnp.float32)
  phi = jnp.zeros((8,), dtype=jnp.float32)
  psi = jnp.zeros((8,), dtype=jnp.float32)
  # Even if padded map_indices has out-of-range index 5, clamping to [0, M-1] prevents crash/NaN
  map_idx = jnp.array([0, 0, 5, -1, 0, 0, 0, 0], dtype=jnp.int32)
  energies = cmap.compute_cmap_energies(phi, psi, map_idx, dummy_coeffs)
  assert energies.shape == (8,)
  assert bool(jnp.all(jnp.isfinite(energies)))


def test_two_stage_field_audit_proxy_consumption():
  """Verify FR-2: Two-stage FieldAuditProxy tracks MolecularBundle and PhysicsSystem field consumption."""
  ns = _make_minimal_protein_namespace(with_pbc=True)
  bundle = make_bundle_from_system(ns, boundary_condition="periodic")

  # Stage 1: Wrap MolecularBundle in FieldAuditProxy and lower to PhysicsSystem
  bundle_proxy = FieldAuditProxy(bundle)
  sys_pbc = physics_system_from_bundle(bundle_proxy, bundle.positions)
  stage1_read = bundle_proxy.get_accessed_fields()
  for req in ("positions", "charges", "sigmas", "epsilons", "atom_mask", "bond_idx", "angle_idx", "dihedral_idx"):
    assert req in stage1_read, f"Stage 1 missed required bundle field {req!r}"

  # Stage 2: Wrap PhysicsSystem in FieldAuditProxy across PBC+PME and Implicit GBSA
  disp_pbc, _ = displacement_fn_for_bundle(bundle)
  sys_pbc_proxy = FieldAuditProxy(sys_pbc)
  e_pbc = single_padded_energy(sys_pbc_proxy, disp_pbc, implicit_solvent=False)
  f_pbc = single_padded_force(sys_pbc_proxy, disp_pbc, implicit_solvent=False, explicit_solvent=True)
  assert bool(jnp.isfinite(e_pbc))
  assert bool(jnp.all(jnp.isfinite(f_pbc)))
  pbc_read = sys_pbc_proxy.get_accessed_fields()

  # Implicit GBSA mode
  ns_gb = _make_minimal_protein_namespace(with_pbc=False)
  bundle_gb = make_bundle_from_system(ns_gb, boundary_condition="free", has_implicit_solvent=True)
  sys_gb = physics_system_from_bundle(bundle_gb, bundle_gb.positions)
  disp_gb, _ = displacement_fn_for_bundle(bundle_gb)
  sys_gb_proxy = FieldAuditProxy(sys_gb)
  e_gb = single_padded_energy(sys_gb_proxy, disp_gb)
  f_gb = single_padded_force(sys_gb_proxy, disp_gb)
  assert bool(jnp.isfinite(e_gb))
  assert bool(jnp.all(jnp.isfinite(f_gb)))
  gb_read = sys_gb_proxy.get_accessed_fields()

  union_read = pbc_read | gb_read
  for core_field in (
    "positions",
    "charges",
    "sigmas",
    "epsilons",
    "atom_mask",
    "bonds",
    "bond_params",
    "angles",
    "angle_params",
    "dihedrals",
    "dihedral_params",
    "impropers",
    "improper_params",
    "radii",
    "scaled_radii",
    "box_size",
    "pme_alpha",
  ):
    assert core_field in union_read, f"Stage 2 union missed core PhysicsSystem field {core_field!r}"

  # Verify replace() propagation on FieldAuditProxy preserves access log without falsely marking mutated keys as read
  fresh_proxy = FieldAuditProxy(sys_pbc)
  replaced_proxy = fresh_proxy.replace(pme_alpha=0.35)
  assert "pme_alpha" not in fresh_proxy.get_accessed_fields()
  assert float(replaced_proxy.pme_alpha) == 0.35
  assert "pme_alpha" in fresh_proxy.get_accessed_fields()
  assert isinstance(PENDING_EPIC_KNOBS, frozenset)
  assert isinstance(DOCUMENTED_INERT_KNOBS, frozenset)
