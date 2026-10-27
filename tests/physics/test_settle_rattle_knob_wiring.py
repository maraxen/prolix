"""Tests for SETTLE/RATTLE propagator knob wiring, R-step SHAKE impulse, and masked CSVR."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from jax_md import space

from prolix.physics import pbc, settle
from prolix.physics.constraint_projection import (
  project_momenta,
  project_positions,
  unpack_constraints,
)


def test_custom_water_geometry_spce_langevin_and_lfmiddle():
  """Custom r_OH=1.0, r_HH=1.633 (SPC/E geometry) is preserved in settle_langevin and settle_lfmiddle_langevin."""
  r_OH_spce = 1.0
  r_HH_spce = 1.633
  half_hh = r_HH_spce / 2.0
  h_y = float(jnp.sqrt(r_OH_spce**2 - half_hh**2))

  R_init = jnp.array(
    [
      [0.0, 0.0, 0.0],
      [half_hh, h_y, 0.0],
      [-half_hh, h_y, 0.0],
    ],
    dtype=jnp.float64,
  )
  water_indices = jnp.array([[0, 1, 2]], dtype=jnp.int32)
  masses = jnp.array([15.999, 1.008, 1.008], dtype=jnp.float64)

  def energy_fn(R):
    return 0.05 * jnp.sum(R**2)

  for factory in (settle.settle_langevin, settle.settle_lfmiddle_langevin):
    init_fn, apply_fn = factory(
      energy_fn,
      shift_fn=space.free()[1],
      dt=0.001,
      kT=0.596,
      gamma=1.0,
      mass=masses,
      water_indices=water_indices,
      r_OH=r_OH_spce,
      r_HH=r_HH_spce,
    )
    state = init_fn(random.key(0), R_init)
    step_fn = jax.jit(apply_fn)
    for _ in range(15):
      state = step_fn(state)

    R_final = state.positions
    d_oh1 = jnp.linalg.norm(R_final[1] - R_final[0])
    d_oh2 = jnp.linalg.norm(R_final[2] - R_final[0])
    d_hh = jnp.linalg.norm(R_final[2] - R_final[1])
    assert jnp.allclose(d_oh1, r_OH_spce, atol=1e-4)
    assert jnp.allclose(d_oh2, r_OH_spce, atol=1e-4)
    assert jnp.allclose(d_hh, r_HH_spce, atol=1e-4)


def test_constraint_projection_padded_mask_and_pbc_displacement():
  """Padded constraints (mask=False, pairs=[0, 0]) do not corrupt atom 0, and PBC bonds use displacement_fn."""
  box = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float64)
  displacement_fn, shift_fn = pbc.create_periodic_space(box)

  # Atom 0 is an uninvolved atom; atoms 1 and 2 form a constrained bond across x=0/x=10 PBC boundary.
  # Unwrapped distance is |9.8 - 10.4| = 0.6 A, target length is 1.0 A.
  R = jnp.array(
    [
      [5.0, 5.0, 5.0],
      [9.8, 5.0, 5.0],
      [0.4, 5.0, 5.0],
      [0.0, 0.0, 0.0],  # padded dummy atom with mass 0
    ],
    dtype=jnp.float64,
  )
  masses = jnp.array([12.0, 12.0, 1.0, 0.0], dtype=jnp.float64)[:, None]
  pairs = jnp.array([[1, 2], [0, 0], [3, 3]], dtype=jnp.int32)
  lengths = jnp.array([1.0, 1.5, 1.5], dtype=jnp.float64)
  mask = jnp.array([True, False, False])

  # Also test 3-tuple unpacking
  u_pairs, u_lengths, u_mask = unpack_constraints((pairs, lengths, mask))
  assert u_mask is not None

  R_proj = project_positions(
    R,
    u_pairs,
    u_lengths,
    masses,
    shift_fn,
    constraint_mask=u_mask,
    max_iter=25,
    displacement_fn=displacement_fn,
  )
  assert not jnp.any(jnp.isnan(R_proj))
  # Atom 0 and padded atom 3 must be untouched
  assert jnp.allclose(R_proj[0], R[0], atol=1e-12)
  assert jnp.allclose(R_proj[3], R[3], atol=1e-12)
  # Minimum-image distance between atom 1 and atom 2 must equal target 1.0 A
  d_12 = jnp.linalg.norm(displacement_fn(R_proj[1], R_proj[2]))
  assert jnp.allclose(d_12, 1.0, atol=1e-4)

  # Test RATTLE momentum projection with padded [0, 0] and PBC displacement_fn
  P = jnp.array(
    [
      [1.5, -0.5, 0.2],
      [2.0, 0.0, 0.0],
      [-1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ],
    dtype=jnp.float64,
  )
  P_proj = project_momenta(
    P,
    R_proj,
    u_pairs,
    masses,
    shift_fn,
    constraint_mask=u_mask,
    max_iter=25,
    displacement_fn=displacement_fn,
  )
  assert not jnp.any(jnp.isnan(P_proj))
  assert jnp.allclose(P_proj[0], P[0], atol=1e-12)
  assert jnp.allclose(P_proj[3], P[3], atol=1e-12)

  r_12 = displacement_fn(R_proj[1], R_proj[2])
  v_12 = P_proj[1] / masses[1, 0] - P_proj[2] / masses[2, 0]
  assert jnp.allclose(jnp.dot(v_12, r_12), 0.0, atol=1e-5)


def test_r_step_shake_impulse_and_mixed_water_solute_settle_langevin():
  """Verify R-step SHAKE positional correction contributes to momentum (x_before_constraints not overwritten)."""
  box = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float64)
  displacement_fn, shift_fn = pbc.create_periodic_space(box)

  # 0,1,2: real water; 3,4: solute constrained bond; 5: padded atom
  R_init = jnp.array(
    [
      [2.0, 2.0, 2.0],
      [2.9572, 2.0, 2.0],
      [1.7601, 2.9266, 2.0],
      [19.7, 5.0, 5.0],
      [0.8, 5.0, 5.0],  # MIC distance is 1.1 A; target constraint is 1.0 A -> SHAKE must pull them together
      [0.0, 0.0, 0.0],
    ],
    dtype=jnp.float64,
  )
  water_indices = jnp.array([[0, 1, 2], [0, 0, 0]], dtype=jnp.int32)
  water_mask = jnp.array([True, False])
  atom_mask = jnp.array([True, True, True, True, True, False])
  masses = jnp.array([15.999, 1.008, 1.008, 12.0, 1.008, 1.0], dtype=jnp.float64)

  pairs = jnp.array([[3, 4], [0, 0]], dtype=jnp.int32)
  lengths = jnp.array([1.0, 1.0], dtype=jnp.float64)
  c_mask = jnp.array([True, False])

  # Zero potential energy and zero friction/temperature: any momentum change on atoms 3,4
  # in a single step starting from zero momentum comes strictly from the R-step SHAKE impulse!
  def zero_energy_fn(R):
    return jnp.sum(R * 0.0)

  init_fn, apply_fn = settle.settle_langevin(
    zero_energy_fn,
    shift_fn=shift_fn,
    dt=0.01,
    kT=0.0,
    gamma=0.0,
    mass=masses,
    water_indices=water_indices,
    water_mask=water_mask,
    atom_mask=atom_mask,
    box=box,
    constraints=(pairs, lengths, c_mask),
    displacement_fn=displacement_fn,
  )
  state0 = init_fn(random.key(7), R_init)
  # Padded atom 5 momentum must be zero
  assert jnp.allclose(state0.momentum[5], 0.0)

  # Give atom 4 an initial velocity along +x to stretch the bond during A-step, then check R-step impulse
  p_init = jnp.zeros_like(R_init).at[4, 0].set(0.5)
  state0 = state0.replace(momentum=p_init)
  state1 = jax.jit(apply_fn)(state0)

  # Bond length (3, 4) across PBC must be constrained to 1.0 A
  d_34 = jnp.linalg.norm(displacement_fn(state1.positions[3], state1.positions[4]))
  assert jnp.allclose(d_34, 1.0, atol=1e-4)
  # Atom 3 started with zero momentum and zero force; because R-step SHAKE moved atom 3 toward atom 4
  # (+x across the boundary), atom 3 MUST have acquired a positive x-momentum from dx_1 / half_dt!
  assert jnp.abs(state1.momentum[3, 0]) > 1e-3
  # Padded atom 5 momentum remains strictly zero
  assert jnp.allclose(state1.momentum[5], 0.0)


def test_settle_csvr_and_npt_mixed_solute_water_temperature():
  """settle_csvr and settle_csvr_npt handle mixed solute+water systems with masks and constraints."""
  box = jnp.array([18.0, 18.0, 18.0], dtype=jnp.float64)
  displacement_fn, shift_fn = pbc.create_periodic_space(box)

  # 2 real waters (6 atoms) + 2 solute atoms with 1 constrained bond + 1 padded atom = 9 atoms
  R_init = jnp.array(
    [
      [2.0, 2.0, 2.0],
      [2.9572, 2.0, 2.0],
      [1.7601, 2.9266, 2.0],
      [6.0, 6.0, 6.0],
      [6.9572, 6.0, 6.0],
      [5.7601, 6.9266, 6.0],
      [10.0, 10.0, 10.0],
      [11.1, 10.0, 10.0],
      [0.0, 0.0, 0.0],
    ],
    dtype=jnp.float64,
  )
  water_indices = jnp.array([[0, 1, 2], [3, 4, 5], [0, 0, 0]], dtype=jnp.int32)
  water_mask = jnp.array([True, True, False])
  atom_mask = jnp.array([True, True, True, True, True, True, True, True, False])
  masses = jnp.array([15.999, 1.008, 1.008, 15.999, 1.008, 1.008, 12.0, 1.008, 1.0], dtype=jnp.float64)
  constraints = (
    jnp.array([[6, 7], [0, 0]], dtype=jnp.int32),
    jnp.array([1.1, 1.0], dtype=jnp.float64),
    jnp.array([True, False]),
  )

  def harmonic_energy(R, **kwargs):
    del kwargs
    return 0.01 * jnp.sum((R - R_init) ** 2)

  kT_target = 0.59616  # ~300 K
  dt = 0.005  # ~0.25 fs in AKMA

  # 1. NVT settle_csvr
  init_csvr, apply_csvr = settle.settle_csvr(
    harmonic_energy,
    shift_fn=shift_fn,
    dt=dt,
    kT=kT_target,
    tau=0.05,
    mass=masses,
    water_indices=water_indices,
    water_mask=water_mask,
    atom_mask=atom_mask,
    constraints=constraints,
    box=box,
    displacement_fn=displacement_fn,
  )
  state_csvr = init_csvr(random.key(11), R_init)
  step_csvr = jax.jit(apply_csvr)
  for _ in range(40):
    state_csvr = step_csvr(state_csvr)

  assert not jnp.any(jnp.isnan(state_csvr.positions))
  assert not jnp.any(jnp.isnan(state_csvr.momentum))
  assert jnp.allclose(state_csvr.momentum[8], 0.0)
  d_solute = jnp.linalg.norm(displacement_fn(state_csvr.positions[6], state_csvr.positions[7]))
  assert jnp.allclose(d_solute, 1.1, atol=1e-4)

  # 2. NPT settle_csvr_npt
  init_npt, apply_npt = settle.settle_csvr_npt(
    harmonic_energy,
    shift_fn=shift_fn,
    dt=dt,
    kT=kT_target,
    target_pressure_bar=1.0,
    tau_barostat_akma=10.0,
    tau_thermostat_akma=0.05,
    mass=masses,
    water_indices=water_indices,
    water_mask=water_mask,
    atom_mask=atom_mask,
    constraints=constraints,
    box_init=box,
    displacement_fn=displacement_fn,
  )
  state_npt = init_npt(random.key(12), R_init)
  step_npt = jax.jit(apply_npt)
  for _ in range(20):
    state_npt = step_npt(state_npt)

  assert not jnp.any(jnp.isnan(state_npt.positions))
  assert not jnp.any(jnp.isnan(state_npt.momentum))
  assert jnp.allclose(state_npt.momentum[8], 0.0)
  d_solute_npt = jnp.linalg.norm(displacement_fn(state_npt.positions[6], state_npt.positions[7]))
  assert jnp.allclose(d_solute_npt, 1.1, atol=1e-4)
