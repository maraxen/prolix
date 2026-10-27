"""Tests for multi-component cosolvent Halton placement and repulsive potential (FR-7)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import pbc
from prolix.physics.cosolvent import (
    AVOGADRO_ANGSTROM3_FACTOR,
    CosolventSpeciesSpec,
    concentration_to_copies,
    cosolvent_repulsive_energy,
    halton_sequence,
    openmm_quintic_switch,
    place_cosolvents_halton,
    random_rotation_matrices,
)


def test_halton_sequence_deterministic_and_exact_values() -> None:
    """Halton sequence with bases (2, 3, 5) produces exact radical inverses."""
    pts = halton_sequence(4, bases=(2, 3, 5), seed_offset=1)
    assert pts.shape == (4, 3)
    expected = np.array(
        [
            [1.0 / 2.0, 1.0 / 3.0, 1.0 / 5.0],
            [1.0 / 4.0, 2.0 / 3.0, 2.0 / 5.0],
            [3.0 / 4.0, 1.0 / 9.0, 3.0 / 5.0],
            [1.0 / 8.0, 4.0 / 9.0, 4.0 / 5.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(pts, expected, rtol=1e-12, atol=1e-12)
    assert np.all((pts >= 0.0) & (pts < 1.0))

    # Deterministic across repeated calls
    pts2 = halton_sequence(4, bases=(2, 3, 5), seed_offset=1)
    np.testing.assert_array_equal(pts, pts2)


def test_random_rotation_matrices_so3_invariants() -> None:
    """Sampled rotation matrices are orthogonal with determinant +1."""
    rots = random_rotation_matrices(jax.random.PRNGKey(42), 32)
    assert rots.shape == (32, 3, 3)
    identities = jnp.einsum("nij,nkj->nik", rots, rots)
    eye_batch = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (32, 3, 3))
    np.testing.assert_allclose(np.asarray(identities), np.asarray(eye_batch), atol=1e-5)

    dets = np.linalg.det(np.asarray(rots))
    np.testing.assert_allclose(dets, np.ones(32), atol=1e-5)


def test_concentration_to_copies_formula() -> None:
    """Verify N_copies = round(C_molar * (V_box - V_protein) * 6.02214076e-4)."""
    v_box = 60.0 ** 3  # 216,000 A^3
    v_prot = 16000.0   # 16,000 A^3 -> V_solvent = 200,000 A^3
    c_molar = 0.25     # 0.25 M -> 0.25 * 200000 * 6.02214076e-4 = 30.1107 -> 30
    n_copies = concentration_to_copies(c_molar, v_box, v_prot)
    expected = round(c_molar * (v_box - v_prot) * AVOGADRO_ANGSTROM3_FACTOR)
    assert n_copies == expected == 30
    assert concentration_to_copies(0.0, v_box, v_prot) == 0


def test_two_stage_halton_placement_and_all_atom_group_ids() -> None:
    """Verify two-stage clash-free placement, distances, and all-atom group/mol IDs."""
    # Protein cluster near box center (25, 25, 25)
    protein_pos = jnp.array(
        [
            [25.0, 25.0, 25.0],
            [26.2, 25.0, 25.0],
            [25.0, 26.2, 25.0],
            [25.0, 25.0, 26.2],
        ],
        dtype=jnp.float32,
    )
    box = jnp.array([50.0, 50.0, 50.0], dtype=jnp.float32)

    # Multi-atom templates (e.g., 3-atom and 2-atom cosolvents + 2-atom non-repulsive species)
    ben_template = jnp.array(
        [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.9, 0.0]], dtype=jnp.float32
    )
    ipa_template = jnp.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=jnp.float32)
    wat_template = jnp.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)

    specs = [
        CosolventSpeciesSpec(
            name="BEN",
            copies=5,
            atom_positions_template=ben_template,
            repulsive=True,
        ),
        CosolventSpeciesSpec(
            name="IPA",
            copies=4,
            atom_positions_template=ipa_template,
            repulsive=True,
        ),
        CosolventSpeciesSpec(
            name="NRP",
            copies=3,
            atom_positions_template=wat_template,
            repulsive=False,
        ),
    ]

    res = place_cosolvents_halton(
        protein_pos,
        box,
        specs,
        protein_radius_angstrom=5.5,
        cosolvent_placement_radius_angstrom=2.5,
        post_placement_clash_angstrom=1.8,
        seed=7,
    )

    n_prot = protein_pos.shape[0]
    n_ben_atoms = 5 * 3
    n_ipa_atoms = 4 * 2
    n_nrp_atoms = 3 * 2
    n_total = n_prot + n_ben_atoms + n_ipa_atoms + n_nrp_atoms

    assert res.positions.shape == (n_total, 3)
    assert res.cosolvent_group_id.shape == (n_total,)
    assert res.cosolvent_mol_id.shape == (n_total,)
    assert res.copies_per_species == (5, 4, 3)

    # Check group IDs: 0 for protein, 1 for all atoms of BEN, 2 for all atoms of IPA, 0 for NRP
    gid = np.asarray(res.cosolvent_group_id)
    np.testing.assert_array_equal(gid[:n_prot], 0)
    np.testing.assert_array_equal(gid[n_prot : n_prot + n_ben_atoms], 1)
    np.testing.assert_array_equal(
        gid[n_prot + n_ben_atoms : n_prot + n_ben_atoms + n_ipa_atoms], 2
    )
    np.testing.assert_array_equal(gid[n_prot + n_ben_atoms + n_ipa_atoms :], 0)

    # Check molecule IDs: -1 for protein, 0..11 for the 12 placed cosolvent molecules
    mid = np.asarray(res.cosolvent_mol_id)
    np.testing.assert_array_equal(mid[:n_prot], -1)
    assert len(np.unique(mid[n_prot:])) == 5 + 4 + 3

    # Verify distances to protein >= 5.5 A (and >= 1.8 A)
    d_prot_cosolv = pbc.minimum_image_distance(
        res.cosolvent_positions[:, None, :], protein_pos[None, :, :], box
    )
    assert float(jnp.min(d_prot_cosolv)) >= 5.5 - 1e-5

    # Verify inter-cosolvent center distances >= 2.5 A and all-atom distances >= 1.8 A
    d_centers = pbc.minimum_image_distance(
        res.cosolvent_centers[:, None, :], res.cosolvent_centers[None, :, :], box
    )
    n_mols = res.cosolvent_centers.shape[0]
    off_diag = ~jnp.eye(n_mols, dtype=bool)
    assert float(jnp.min(jnp.where(off_diag, d_centers, jnp.inf))) >= 2.5 - 1e-5
    assert res.min_post_placement_distance >= 1.8


def test_triclinic_halton_placement() -> None:
    """Verify Halton placement works with a lower-triangular (3, 3) triclinic box."""
    box_triclinic = pbc.triclinic_box_from_lengths_angles(
        45.0, 45.0, 45.0, 60.0, 60.0, 60.0
    )
    protein_pos = jnp.array([[10.0, 10.0, 10.0]], dtype=jnp.float32)
    specs = [
        CosolventSpeciesSpec(
            name="BEN",
            copies=4,
            atom_positions_template=jnp.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=jnp.float32),
            repulsive=True,
        )
    ]
    res = place_cosolvents_halton(protein_pos, box_triclinic, specs, seed=11)
    assert res.positions.shape == (1 + 4 * 2, 3)
    assert res.min_post_placement_distance >= 1.8


def test_quintic_switch_smoothness() -> None:
    """Verify r_switch = r_cutoff - 1.0 A quintic switch value, 1st, and 2nd derivatives."""
    r_cutoff = 9.0
    r_switch = r_cutoff - 1.0  # 8.0 A

    # Boundary values
    assert float(openmm_quintic_switch(jnp.array(7.5), r_switch, r_cutoff)) == pytest.approx(1.0)
    assert float(openmm_quintic_switch(jnp.array(8.0), r_switch, r_cutoff)) == pytest.approx(1.0)
    assert float(openmm_quintic_switch(jnp.array(8.5), r_switch, r_cutoff)) == pytest.approx(0.5)
    assert float(openmm_quintic_switch(jnp.array(9.0), r_switch, r_cutoff)) == pytest.approx(0.0)
    assert float(openmm_quintic_switch(jnp.array(9.5), r_switch, r_cutoff)) == pytest.approx(0.0)

    # First and second derivatives vanish at r_switch and r_cutoff (C^2 continuity)
    grad_fn = jax.grad(lambda r: openmm_quintic_switch(r, r_switch, r_cutoff))
    hess_fn = jax.grad(grad_fn)

    for r_pt in (8.0 + 1e-5, 9.0 - 1e-5):
        assert float(grad_fn(jnp.array(r_pt, dtype=jnp.float32))) == pytest.approx(0.0, abs=1e-3)
        assert float(hess_fn(jnp.array(r_pt, dtype=jnp.float32))) == pytest.approx(0.0, abs=1e-2)


def test_repulsive_energy_intramolecular_and_cross_species_exclusions() -> None:
    """Intramolecular pairs (same mol_id) and cross-species pairs contribute zero energy."""
    box = jnp.array([30.0, 30.0, 30.0], dtype=jnp.float32)
    displacement_fn, _ = pbc.create_periodic_space(box)

    # Atoms 0, 1 belong to molecule 0 (species 1) at distance 1.4 A -> intramolecular, must be excluded!
    # Atom 2 belongs to molecule 1 (species 2) at distance 4.0 A -> different species, must be excluded!
    # Atom 3 is protein (group_id=0, mol_id=-1) at distance 4.0 A -> must be excluded!
    positions = jnp.array(
        [
            [5.0, 5.0, 5.0],
            [6.4, 5.0, 5.0],
            [5.0, 9.0, 5.0],
            [5.0, 5.0, 9.0],
        ],
        dtype=jnp.float32,
    )
    group_id = jnp.array([1, 1, 2, 0], dtype=jnp.int32)
    mol_id = jnp.array([0, 0, 1, -1], dtype=jnp.int32)

    e = cosolvent_repulsive_energy(
        positions,
        group_id,
        mol_id,
        displacement_fn,
        deduplicate_interaction_groups=True,
    )
    assert float(e) == pytest.approx(0.0, abs=1e-10)


def test_repulsive_energy_and_force_exact_2x_deduplicate_ratio() -> None:
    """Verify exact 2x ratio between deduplicate_interaction_groups=False and True on energy & forces."""
    box = jnp.array([40.0, 40.0, 40.0], dtype=jnp.float32)
    displacement_fn, _ = pbc.create_periodic_space(box)

    # Two 2-atom molecules of species 1 (all 4 atoms have group_id=1; mol 0 vs mol 1)
    # Plus one protein atom (group_id=0, mol_id=-1)
    positions = jnp.array(
        [
            [10.0, 10.0, 10.0],  # mol 0, atom 0
            [11.2, 10.0, 10.0],  # mol 0, atom 1
            [16.5, 10.2, 10.0],  # mol 1, atom 0 (within < 8 A of mol 0)
            [17.7, 10.2, 10.0],  # mol 1, atom 1 (in switching region 8..9 A of atom 0)
            [2.0, 2.0, 2.0],     # protein atom
        ],
        dtype=jnp.float32,
    )
    group_id = jnp.array([1, 1, 1, 1, 0], dtype=jnp.int32)
    mol_id = jnp.array([0, 0, 1, 1, -1], dtype=jnp.int32)

    def energy_fn(r: jax.Array, dedup: bool) -> jax.Array:
        return cosolvent_repulsive_energy(
            r,
            group_id,
            mol_id,
            displacement_fn,
            eps_kcal=0.01,
            r0_angstrom=7.0,
            cutoff_angstrom=9.0,
            deduplicate_interaction_groups=dedup,
        )

    e_dedup_true = energy_fn(positions, True)
    e_dedup_false = energy_fn(positions, False)

    assert float(e_dedup_true) > 0.0
    assert float(e_dedup_false) == pytest.approx(2.0 * float(e_dedup_true), rel=1e-6, abs=1e-8)

    f_dedup_true = -jax.grad(lambda r: energy_fn(r, True))(positions)
    f_dedup_false = -jax.grad(lambda r: energy_fn(r, False))(positions)

    np.testing.assert_allclose(
        np.asarray(f_dedup_false),
        2.0 * np.asarray(f_dedup_true),
        rtol=1e-6,
        atol=1e-7,
    )
    # Protein atom (index 4) has zero repulsive force
    np.testing.assert_allclose(np.asarray(f_dedup_false[4]), np.zeros(3), atol=1e-10)
