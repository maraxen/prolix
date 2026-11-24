"""Tests for triclinic (3, 3) periodic boxes across PBC, NeighborList, SPME, and StableHLO."""

from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import neighbor_list as nl
from prolix.physics import pbc, pme
from prolix.simulate import SimulationSpec


def test_no_lapack_det_or_inv_in_pbc_or_pme():
  """Enforce CB-9 / R-3: pbc.py and pme.py must never call jnp.linalg.det or jnp.linalg.inv."""
  repo_root = Path(__file__).resolve().parents[2]
  for rel in ("src/prolix/physics/pbc.py", "src/prolix/physics/pme.py"):
    tree = ast.parse((repo_root / rel).read_text())
    for node in ast.walk(tree):
      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {"det", "inv", "solve"}:
          pytest.fail(f"Forbidden LAPACK call .{node.func.attr}() found in {rel}")


def test_closed_form_box_volume_and_inverse():
  """Verify closed-form lower-triangular volume and inverse against exact algebra."""
  box = pbc.triclinic_box_from_lengths_angles(32.0, 32.0, 32.0, 60.0, 60.0, 60.0)
  assert box.shape == (3, 3)
  # Check OpenMM reduced-basis invariants
  assert float(box[0, 0]) > 0.0
  assert float(box[1, 1]) > 0.0
  assert float(box[2, 2]) > 0.0
  assert abs(float(box[1, 0])) <= 0.5 * float(box[0, 0]) + 1e-5
  assert abs(float(box[2, 0])) <= 0.5 * float(box[0, 0]) + 1e-5
  assert abs(float(box[2, 1])) <= 0.5 * float(box[1, 1]) + 1e-5

  vol = pbc.box_volume(box)
  # Rhombic dodecahedron with edge d has volume d^3 / sqrt(2)
  expected_vol = (32.0**3) / np.sqrt(2.0)
  np.testing.assert_allclose(vol, expected_vol, rtol=1e-5)

  B = pbc.invert_box_lower_triangular(box)
  np.testing.assert_allclose(box @ B, np.eye(3), atol=1e-6)
  np.testing.assert_allclose(B @ box, np.eye(3), atol=1e-6)


def test_orthorhombic_1d_and_2d_equivalence():
  """Verify diag(L) (3, 3) matches L (3,) across PBC and SPME."""
  L_1d = jnp.array([24.0, 26.0, 28.0], dtype=jnp.float32)
  L_2d = jnp.diag(L_1d)

  key = jax.random.PRNGKey(42)
  k1, k2 = jax.random.split(key)
  pos = jax.random.uniform(k1, (16, 3), minval=-10.0, maxval=35.0, dtype=jnp.float32)
  charges = jax.random.normal(k2, (16,), dtype=jnp.float32)
  charges = charges - jnp.mean(charges)
  mask = jnp.ones((16,), dtype=jnp.bool_)

  disp_1d = pbc.minimum_image_displacement(pos[:, None, :], pos[None, :, :], L_1d)
  disp_2d = pbc.minimum_image_displacement(pos[:, None, :], pos[None, :, :], L_2d)
  np.testing.assert_allclose(disp_1d, disp_2d, atol=1e-5)

  wrap_1d = pbc.wrap_positions(pos, L_1d)
  wrap_2d = pbc.wrap_positions(pos, L_2d)
  np.testing.assert_allclose(wrap_1d, wrap_2d, atol=1e-5)

  grid_dims = (16, 16, 16)
  e_1d, g_1d = jax.value_and_grad(pme.spme_energy_with_forces)(
    pos, charges, mask, L_1d, grid_dims, 0.34, 4, False
  )
  e_2d, g_2d = jax.value_and_grad(pme.spme_energy_with_forces)(
    pos, charges, mask, L_2d, grid_dims, 0.34, 4, False
  )
  np.testing.assert_allclose(e_1d, e_2d, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(g_1d, g_2d, rtol=1e-5, atol=1e-5)


def test_triclinic_spme_lattice_translation_invariance_and_custom_vjp():
  """Verify CA-5: m = B @ n gives lattice-invariant energy & forces and exact custom_vjp."""
  box = pbc.triclinic_box_from_lengths_angles(24.0, 25.0, 26.0, 75.0, 80.0, 70.0)
  key = jax.random.PRNGKey(7)
  k1, k2, k3 = jax.random.split(key, 3)
  pos = jax.random.uniform(k1, (12, 3), minval=2.0, maxval=18.0, dtype=jnp.float32)
  charges = jax.random.normal(k2, (12,), dtype=jnp.float32)
  charges = charges - jnp.mean(charges)
  mask = jnp.ones((12,), dtype=jnp.bool_)
  grid_dims = (16, 16, 16)

  # Integer lattice shift n_a * a + n_b * b + n_c * c
  n_int = jax.random.randint(k3, (12, 3), minval=-2, maxval=3).astype(jnp.float32)
  pos_shifted = pos + n_int @ box

  e_base, g_base = jax.value_and_grad(pme.spme_energy_with_forces)(
    pos, charges, mask, box, grid_dims, 0.34, 4, False
  )
  e_shift, g_shift = jax.value_and_grad(pme.spme_energy_with_forces)(
    pos_shifted, charges, mask, box, grid_dims, 0.34, 4, False
  )
  np.testing.assert_allclose(e_base, e_shift, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(g_base, g_shift, rtol=1e-5, atol=1e-5)

  # Compare custom_vjp gradient against primal autodiff of spme_reciprocal_energy
  g_autodiff = jax.grad(
    lambda r: pme.spme_reciprocal_energy(r, charges, mask, box, grid_dims, 0.34, 4, False)
  )(pos)
  np.testing.assert_allclose(g_base, g_autodiff, rtol=1e-4, atol=1e-4)


def test_simulation_spec_and_neighbor_list_triclinic_validation():
  """Verify CB-5 and CB-6: SimulationSpec and NeighborList accept lower-triangular (3, 3) box."""
  box = pbc.triclinic_box_from_lengths_angles(30.0, 30.0, 30.0, 60.0, 60.0, 60.0)
  spec = SimulationSpec(total_time_ns=0.001, box=box, use_pbc=True)
  assert spec.box.shape == (3, 3)

  cap = nl.compute_nl_capacity(512, box, cutoff=9.0)
  assert cap >= 256

  with pytest.raises(ValueError, match="lower-triangular"):
    bad_box = np.array([[20.0, 1.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]], dtype=np.float32)
    SimulationSpec(total_time_ns=0.001, box=bad_box, use_pbc=True)


def test_spme_energy_with_forces_box_gradient_finite():
  """Verify _spme_bwd and _spme_cached_bwd return a finite non-None dE_dbox cotangent."""
  box = pbc.triclinic_box_from_lengths_angles(24.0, 25.0, 26.0, 75.0, 80.0, 70.0)
  key = jax.random.PRNGKey(19)
  k1, k2 = jax.random.split(key)
  pos = jax.random.uniform(k1, (8, 3), minval=2.0, maxval=18.0, dtype=jnp.float32)
  charges = jax.random.normal(k2, (8,), dtype=jnp.float32)
  charges = charges - jnp.mean(charges)
  mask = jnp.ones((8,), dtype=jnp.bool_)
  grid_dims = (16, 16, 16)

  g_box = jax.grad(
    lambda b: pme.spme_energy_with_forces(pos, charges, mask, b, grid_dims, 0.34, 4, False)
  )(box)
  g_box_ref = jax.grad(
    lambda b: pme.spme_reciprocal_energy(pos, charges, mask, b, grid_dims, 0.34, 4, False)
  )(box)
  assert g_box.shape == (3, 3)
  assert bool(jnp.all(jnp.isfinite(g_box)))
  np.testing.assert_allclose(g_box, g_box_ref, rtol=1e-4, atol=1e-4)
