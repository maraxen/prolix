"""Test that SETTLE preserves molecular rotation.

This test suite validates that the SETTLE constraint algorithm is rotation-preserving,
i.e., it correctly projects constraint violations without artificially damping rotation.
"""

import jax
import jax.numpy as jnp
import pytest

from prolix.physics.settle import _settle_water_batch, settle_positions

# Enable x64 for numerical precision
jax.config.update("jax_enable_x64", True)

# TIP3P water model constants
TIP3P_ROH = 0.9572  # Å
TIP3P_RHH = 1.5136  # Å
MASS_O = 15.999  # amu
MASS_H = 1.008  # amu


def build_ideal_water_geometry():
    """Construct an ideal TIP3P water in the standard body-frame coordinates.

    Returns:
        Tuple of (pos_oxygen, pos_h1, pos_h2) in the COM frame, (N_waters=1, 3).
    """
    # Body-frame canonical coordinates (COM-centered)
    dist_oh_mid = jnp.sqrt(TIP3P_ROH**2 - (TIP3P_RHH / 2) ** 2)
    mass_total = MASS_O + 2 * MASS_H

    dist_O_to_COM = 2 * MASS_H * dist_oh_mid / mass_total
    dist_H_mid_to_COM = MASS_O * dist_oh_mid / mass_total
    half_hh = TIP3P_RHH / 2

    # Canonical body-frame template (single water: shape (1, 3))
    b_O = jnp.array([[0.0, dist_O_to_COM, 0.0]])
    b_H1 = jnp.array([[half_hh, -dist_H_mid_to_COM, 0.0]])
    b_H2 = jnp.array([[-half_hh, -dist_H_mid_to_COM, 0.0]])

    return b_O, b_H1, b_H2


def rotation_matrix_x(angle):
    """3x3 rotation matrix about the x-axis (angle in radians)."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ])


def rotation_matrix_from_axis_angle(axis, angle):
    """3x3 rotation matrix via axis-angle (Rodrigues formula).

    Args:
        axis: (3,) unit vector
        angle: rotation angle in radians

    Returns:
        (3, 3) rotation matrix
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    omc = 1.0 - c

    x, y, z = axis[0], axis[1], axis[2]

    return jnp.array([
        [c + x*x*omc,     x*y*omc - z*s,  x*z*omc + y*s],
        [y*x*omc + z*s,   c + y*y*omc,    y*z*omc - x*s],
        [z*x*omc - y*s,   z*y*omc + x*s,  c + z*z*omc]
    ])


@pytest.mark.order(1)
def test_already_rigid_invariance_about_x():
    """Test that an already-rigid water rotated about x is returned unchanged.

    This is THE decisive test: if the water is perfectly rigid (bonds correct),
    a correct constraint projection MUST return it unchanged. If it doesn't,
    the implementation is snapping the orientation.
    """
    # Build ideal geometry
    b_O, b_H1, b_H2 = build_ideal_water_geometry()

    # Rotate by 5 degrees about x-axis
    R_x = rotation_matrix_x(jnp.radians(5.0))

    # Apply rotation to body-frame points
    pos_O_rotated = b_O @ R_x.T  # (1, 3)
    pos_H1_rotated = b_H1 @ R_x.T
    pos_H2_rotated = b_H2 @ R_x.T

    # Call SETTLE with:
    #   - pos_*_new = rotated (already perfect)
    #   - pos_*_old = unrotated ideal (for unwrap only)
    pos_O_c, pos_H1_c, pos_H2_c = _settle_water_batch(
        pos_oxygen_old=b_O,
        pos_h1_old=b_H1,
        pos_h2_old=b_H2,
        pos_oxygen_new=pos_O_rotated,
        pos_h1_new=pos_H1_rotated,
        pos_h2_new=pos_H2_rotated,
        r_OH=TIP3P_ROH,
        r_HH=TIP3P_RHH,
        mass_oxygen=MASS_O,
        mass_hydrogen=MASS_H,
        box=None,
    )

    # Assert output ≈ rotated input (rotation preserved)
    assert jnp.allclose(pos_O_c, pos_O_rotated, atol=1e-6), (
        f"O moved {jnp.linalg.norm(pos_O_c - pos_O_rotated)} Å "
        f"(rotated input at {pos_O_rotated}, got {pos_O_c})"
    )
    assert jnp.allclose(pos_H1_c, pos_H1_rotated, atol=1e-6), (
        f"H1 moved {jnp.linalg.norm(pos_H1_c - pos_H1_rotated)} Å"
    )
    assert jnp.allclose(pos_H2_c, pos_H2_rotated, atol=1e-6), (
        f"H2 moved {jnp.linalg.norm(pos_H2_c - pos_H2_rotated)} Å"
    )


@pytest.mark.order(2)
def test_full_3d_rotation_preserved():
    """Test that a water rotated by ~17° about a generic 3D axis is preserved."""
    b_O, b_H1, b_H2 = build_ideal_water_geometry()

    # Rotate by 17 degrees about axis (1,1,1) normalized
    axis = jnp.array([1.0, 1.0, 1.0])
    axis = axis / jnp.linalg.norm(axis)

    R = rotation_matrix_from_axis_angle(axis, jnp.radians(17.0))

    pos_O_rotated = b_O @ R.T
    pos_H1_rotated = b_H1 @ R.T
    pos_H2_rotated = b_H2 @ R.T

    pos_O_c, pos_H1_c, pos_H2_c = _settle_water_batch(
        pos_oxygen_old=b_O,
        pos_h1_old=b_H1,
        pos_h2_old=b_H2,
        pos_oxygen_new=pos_O_rotated,
        pos_h1_new=pos_H1_rotated,
        pos_h2_new=pos_H2_rotated,
        r_OH=TIP3P_ROH,
        r_HH=TIP3P_RHH,
        mass_oxygen=MASS_O,
        mass_hydrogen=MASS_H,
        box=None,
    )

    assert jnp.allclose(pos_O_c, pos_O_rotated, atol=1e-6)
    assert jnp.allclose(pos_H1_c, pos_H1_rotated, atol=1e-6)
    assert jnp.allclose(pos_H2_c, pos_H2_rotated, atol=1e-6)


@pytest.mark.order(3)
def test_bond_angle_restoration():
    """Test that SETTLE restores ideal bond lengths when they are violated.

    Take the ideal water and push one H radially outward by 0.1 Å (violates O-H).
    Assert that SETTLE restores |O-H| and |H1-H2| to their target values.
    """
    b_O, b_H1, b_H2 = build_ideal_water_geometry()

    # Perturb: push H1 outward (increase O-H1 distance)
    delta_perturbation = jnp.array([[0.1, 0.0, 0.0]])  # Push radially
    pos_H1_perturbed = b_H1 + delta_perturbation

    pos_O_c, pos_H1_c, pos_H2_c = _settle_water_batch(
        pos_oxygen_old=b_O,
        pos_h1_old=b_H1,
        pos_h2_old=b_H2,
        pos_oxygen_new=b_O,  # O unchanged
        pos_h1_new=pos_H1_perturbed,
        pos_h2_new=b_H2,  # H2 unchanged
        r_OH=TIP3P_ROH,
        r_HH=TIP3P_RHH,
        mass_oxygen=MASS_O,
        mass_hydrogen=MASS_H,
        box=None,
    )

    # Verify bond lengths
    dist_OH1 = jnp.linalg.norm(pos_H1_c - pos_O_c)
    dist_OH2 = jnp.linalg.norm(pos_H2_c - pos_O_c)
    dist_H1H2 = jnp.linalg.norm(pos_H2_c - pos_H1_c)

    assert jnp.abs(dist_OH1 - TIP3P_ROH) < 1e-6, f"O-H1 = {dist_OH1}, expected {TIP3P_ROH}"
    assert jnp.abs(dist_OH2 - TIP3P_ROH) < 1e-6, f"O-H2 = {dist_OH2}, expected {TIP3P_ROH}"
    assert jnp.abs(dist_H1H2 - TIP3P_RHH) < 1e-6, f"H1-H2 = {dist_H1H2}, expected {TIP3P_RHH}"


@pytest.mark.order(4)
def test_com_preserved():
    """Test that mass-weighted COM is preserved."""
    b_O, b_H1, b_H2 = build_ideal_water_geometry()

    # Perturb all atoms
    delta = jnp.array([[0.05, 0.03, -0.02]])
    pos_O_perturbed = b_O + delta
    pos_H1_perturbed = b_H1 + delta * 0.5
    pos_H2_perturbed = b_H2 + delta * 0.5

    pos_O_c, pos_H1_c, pos_H2_c = _settle_water_batch(
        pos_oxygen_old=b_O,
        pos_h1_old=b_H1,
        pos_h2_old=b_H2,
        pos_oxygen_new=pos_O_perturbed,
        pos_h1_new=pos_H1_perturbed,
        pos_h2_new=pos_H2_perturbed,
        r_OH=TIP3P_ROH,
        r_HH=TIP3P_RHH,
        mass_oxygen=MASS_O,
        mass_hydrogen=MASS_H,
        box=None,
    )

    mass_total = MASS_O + 2 * MASS_H
    com_input = (MASS_O * pos_O_perturbed + MASS_H * pos_H1_perturbed + MASS_H * pos_H2_perturbed) / mass_total
    com_output = (MASS_O * pos_O_c + MASS_H * pos_H1_c + MASS_H * pos_H2_c) / mass_total

    assert jnp.allclose(com_output, com_input, atol=1e-9), (
        f"COM moved: input {com_input}, output {com_output}"
    )


@pytest.mark.order(5)
def test_batched_consistency():
    """Test that batched results match single-water calls (vmap correctness)."""
    b_O, b_H1, b_H2 = build_ideal_water_geometry()

    # Create a batch of 3 waters with different rotations
    angles = jnp.array([5.0, 17.0, 42.0])

    # Rotate each by its angle about x
    pos_O_batch = jnp.vstack([
        (b_O @ rotation_matrix_x(jnp.radians(angle)).T)
        for angle in angles
    ])
    pos_H1_batch = jnp.vstack([
        (b_H1 @ rotation_matrix_x(jnp.radians(angle)).T)
        for angle in angles
    ])
    pos_H2_batch = jnp.vstack([
        (b_H2 @ rotation_matrix_x(jnp.radians(angle)).T)
        for angle in angles
    ])

    # Repeat old positions for all waters (for unwrap)
    pos_O_old_batch = jnp.repeat(b_O, 3, axis=0)
    pos_H1_old_batch = jnp.repeat(b_H1, 3, axis=0)
    pos_H2_old_batch = jnp.repeat(b_H2, 3, axis=0)

    # Call batched SETTLE
    pos_O_c_batch, pos_H1_c_batch, pos_H2_c_batch = _settle_water_batch(
        pos_oxygen_old=pos_O_old_batch,
        pos_h1_old=pos_H1_old_batch,
        pos_h2_old=pos_H2_old_batch,
        pos_oxygen_new=pos_O_batch,
        pos_h1_new=pos_H1_batch,
        pos_h2_new=pos_H2_batch,
        r_OH=TIP3P_ROH,
        r_HH=TIP3P_RHH,
        mass_oxygen=MASS_O,
        mass_hydrogen=MASS_H,
        box=None,
    )

    # Verify each water matches the expected rotation
    for i, angle in enumerate(angles):
        expected_O = b_O @ rotation_matrix_x(jnp.radians(angle)).T
        expected_H1 = b_H1 @ rotation_matrix_x(jnp.radians(angle)).T
        expected_H2 = b_H2 @ rotation_matrix_x(jnp.radians(angle)).T

        assert jnp.allclose(pos_O_c_batch[i:i+1], expected_O, atol=1e-6), (
            f"Water {i} O mismatch"
        )
        assert jnp.allclose(pos_H1_c_batch[i:i+1], expected_H1, atol=1e-6), (
            f"Water {i} H1 mismatch"
        )
        assert jnp.allclose(pos_H2_c_batch[i:i+1], expected_H2, atol=1e-6), (
            f"Water {i} H2 mismatch"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
