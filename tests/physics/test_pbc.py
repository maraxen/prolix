"""Tests for PBC module."""

import jax.numpy as jnp

from prolix.physics import pbc


def test_periodic_displacement_wraps_correctly():
  """Test that displacement function respects minimum image convention."""
  box = jnp.array([10.0, 10.0, 10.0])
  displacement_fn, _ = pbc.create_periodic_space(box)

  # Points at 1.0 and 9.0 (dist 8.0 or 2.0 wrapped?)
  # 1.0 - 9.0 = -8.0 -> nearest image is +2.0
  r1 = jnp.array([1.0, 0.0, 0.0])
  r2 = jnp.array([9.0, 0.0, 0.0])

  dr = displacement_fn(r1, r2)
  assert jnp.allclose(dr, jnp.array([2.0, 0.0, 0.0]))

  # Distance
  dist = jnp.linalg.norm(dr)
  assert jnp.isclose(dist, 2.0)


def test_wrap_positions_stays_in_box():
  """Test that positions outside box are wrapped correctly."""
  box = jnp.array([10.0, 10.0, 10.0])

  r = jnp.array([[11.0, -1.0, 5.0], [25.0, 10.0, 0.0]])

  wrapped = pbc.wrap_positions(r, box)

  expected = jnp.array(
    [
      [1.0, 9.0, 5.0],
      [5.0, 0.0, 0.0],  # 10.0 % 10.0 == 0.0 usually, or 10.0 depending on implementation
    ]
  )

  # Jax modulo handling: 10.0 % 10.0 -> 0.0
  assert jnp.allclose(wrapped, expected)
  assert jnp.all(wrapped >= 0.0)
  assert jnp.all(wrapped < box)


def test_minimum_image_distance():
  """Test explicit minimum image distance calculation."""
  box = jnp.array([10.0, 10.0, 10.0])
  r1 = jnp.array([1.0, 1.0, 1.0])
  r2 = jnp.array([9.0, 9.0, 9.0])

  # Dist vector: (-8, -8, -8) -> (2, 2, 2)
  # Norm: sqrt(12)

  dist = pbc.minimum_image_distance(r1, r2, box)
  expected = jnp.sqrt(12.0)

  assert jnp.isclose(dist, expected)


def test_round_half_away_from_zero_differs_from_bankers_rounding():
  """Test the tie-breaking convention itself (backlog #4232).

  `minimum_image_distance`'s scalar output can't distinguish the two
  conventions at an exact box/2 tie: both bankers'-rounding and
  round-half-away-from-zero wrap to a mirror-symmetric `dr`, so the
  resulting *distance* is identical either way (verified: dr=[-5,0,0] vs
  [5,0,0] for box=10, both norm 5.0). The only way to actually verify the
  fix landed -- and guard against a future revert to `jnp.round` -- is to
  test the tie-breaking helper directly against known half-integer ties.
  """
  # jnp.round uses round-half-to-even (banker's rounding): both +0.5 and
  # -0.5 round to 0.0, since 0 is the nearest even integer to both.
  assert jnp.round(0.5) == 0.0
  assert jnp.round(-0.5) == 0.0
  assert jnp.round(1.5) == 2.0  # 2 is even -> rounds up here instead

  # round-half-away-from-zero must break every tie outward, unlike jnp.round.
  assert pbc._round_half_away_from_zero(jnp.array(0.5)) == 1.0
  assert pbc._round_half_away_from_zero(jnp.array(-0.5)) == -1.0
  assert pbc._round_half_away_from_zero(jnp.array(1.5)) == 2.0
  assert pbc._round_half_away_from_zero(jnp.array(-1.5)) == -2.0
  # Non-tie values must round normally (same as jnp.round).
  assert pbc._round_half_away_from_zero(jnp.array(0.3)) == 0.0
  assert pbc._round_half_away_from_zero(jnp.array(0.7)) == 1.0


def test_minimum_image_distance_at_box_half_boundary():
  """Test PBC wrap at an exact box/2 tie (backlog #4232).

  At this exact boundary the wrap is genuinely ambiguous (two equally
  valid periodic images), so both tie-breaking conventions happen to
  produce the same *distance* here -- see the dedicated tie-breaking
  test above for the real discriminating check. This test just pins
  down that `minimum_image_distance` itself is deterministic and stays
  well-defined (finite, correct magnitude) at the tie point.

  Note: `create_periodic_space`'s `displacement_fn` wraps jax_md's own
  `space.periodic`, a separate implementation this fix does not touch --
  do not use it here to probe `minimum_image_distance`'s behavior.
  """
  box = jnp.array([10.0, 10.0, 10.0])

  # Place atoms exactly box/2 apart in the x-dimension.
  r1 = jnp.array([2.5, 0.0, 0.0])  # center of 0-5 cell
  r2 = jnp.array([7.5, 0.0, 0.0])  # center of 5-10 cell

  dist = pbc.minimum_image_distance(r1, r2, box)
  assert jnp.isclose(dist, 5.0)

  # Verify determinism: running it again should give the same result.
  dist2 = pbc.minimum_image_distance(r1, r2, box)
  assert jnp.isclose(dist2, dist)
