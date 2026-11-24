"""Multi-component cosolvent Halton placement and all-atom per-species repulsive potential.

Implements ``cosolvkit`` / ``naurmalade`` parity for multi-component cosolvent
systems (FR-7):
1. Concentration-to-copy conversion from solvent-accessible box volume.
2. Deterministic 3D Halton low-discrepancy sequence and uniform SO(3) random
   quaternion rotations.
3. Two-stage PBC-aware clash-free cosolvent placement:
   - Stage 1 (placement): enforces minimum distance >= ``protein_radius_angstrom``
     (5.5 Å) from protein atoms and >= ``cosolvent_placement_radius_angstrom``
     (2.5 Å) between cosolvent molecules.
   - Stage 2 (post-placement clash check): verifies all inter-molecular atom-atom
     distances >= ``post_placement_clash_angstrom`` (1.8 Å) under PBC minimum image.
4. All-atom per-species ``cosolvent_group_id`` (``g >= 1`` for all atoms of
   repulsive cosolvent species ``g``, ``0`` otherwise) and ``cosolvent_mol_id``
   (unique molecule index ``m >= 0`` per placed cosolvent molecule, ``-1`` for
   non-cosolvent atoms).
5. Differentiable all-atom per-species $r^{-12}$ repulsive potential with OpenMM
   quintic switching function over $[r_{\\text{cutoff}} - 1.0, r_{\\text{cutoff}}]$
   and ``deduplicate_interaction_groups`` control (default ``False`` reproduces
   ``cosolvkit``'s $2\\times$ ``addInteractionGroup(ag, ag)`` prefactor; ``True``
   evaluates the $1\\times$ physical unordered pair sum).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space, util

from prolix.physics import pbc

Array = util.Array

# Avogadro's number * 1e-27 L/Å^3 = 6.02214076e23 * 1e-27 = 6.02214076e-4
AVOGADRO_ANGSTROM3_FACTOR: float = 6.02214076e-4


@dataclass(frozen=True)
class CosolventSpeciesSpec:
    """Specification for a single cosolvent species.

    Attributes:
        name: Residue / species identifier (e.g., ``"BEN"``, ``"IPA"``).
        smiles: Optional SMILES string for provenance.
        concentration_molar: Target molar concentration (mol/L). Ignored when
            ``copies`` is explicitly provided.
        copies: Explicit number of cosolvent molecule copies to place. If
            ``None``, computed from ``concentration_molar`` and solvent volume.
        atom_positions_template: Optional ``(N_atoms_mol, 3)`` template Cartesian
            coordinates in Å for one molecule of this species. If ``None``, a
            single-atom site at the origin ``(1, 3)`` is used.
        repulsive: Whether this cosolvent species participates in the
            inter-cosolvent $r^{-12}$ repulsive potential.
        repulsive_eps_kcal: Repulsive well depth $\\epsilon_{\\text{rep}}$ in
            kcal/mol (default ``0.01`` kcal/mol = ``0.04184`` kJ/mol).
        repulsive_r0_angstrom: Repulsive reference distance $r_0$ ($\\sigma$) in
            Å (default ``7.0`` Å).
        repulsive_cutoff_angstrom: Cutoff distance $r_{\\text{cutoff}}$ in Å
            (default ``9.0`` Å, with switch at $r_{\\text{cutoff}} - 1.0$ Å).
    """

    name: str
    smiles: str = ""
    concentration_molar: float = 0.0
    copies: int | None = None
    atom_positions_template: Array | None = None
    repulsive: bool = True
    repulsive_eps_kcal: float = 0.01
    repulsive_r0_angstrom: float = 10.0
    repulsive_cutoff_angstrom: float = 9.0


class CosolventPlacementResult(NamedTuple):
    """Result of two-stage Halton cosolvent placement.

    Attributes:
        positions: ``(N_protein + N_cosolv_atoms, 3)`` combined coordinates (Å).
        cosolvent_group_id: ``(N_protein + N_cosolv_atoms,)`` int32 array; ``0``
            for protein/non-repulsive atoms and ``g >= 1`` for all atoms of
            repulsive cosolvent species ``g``.
        cosolvent_mol_id: ``(N_protein + N_cosolv_atoms,)`` int32 array; ``-1``
            for protein/non-cosolvent atoms and unique ``m >= 0`` for each placed
            cosolvent molecule.
        cosolvent_positions: ``(N_cosolv_atoms, 3)`` placed cosolvent atom
            coordinates (Å).
        cosolvent_centers: ``(N_placed_molecules, 3)`` placed cosolvent molecule
            centroids (Å).
        copies_per_species: Tuple of placed molecule counts for each species in
            ``species_specs``.
        min_post_placement_distance: Minimum inter-molecular atom-atom distance
            (Å) observed in the post-placement clash verification check.
    """

    positions: Array
    cosolvent_group_id: Array
    cosolvent_mol_id: Array
    cosolvent_positions: Array
    cosolvent_centers: Array
    copies_per_species: tuple[int, ...]
    min_post_placement_distance: float


def concentration_to_copies(
    concentration_molar: float,
    box_volume_angstrom3: float | Array,
    protein_volume_angstrom3: float | Array = 0.0,
) -> int:
    """Convert molar concentration (M) to integer copy count in solvent volume.

    Implements:
        $$N_{\\text{copies}} = \\text{round}\\!\\left(
            C_{\\text{molar}} \\cdot V_{\\text{solvent, \\AA}^3}
            \\cdot 6.02214076 \\times 10^{-4}
        \\right)$$
    where $V_{\\text{solvent, \\AA}^3} = \\max(V_{\\text{box, \\AA}^3} - V_{\\text{protein, \\AA}^3}, 0)$.

    Args:
        concentration_molar: Target molarity in mol/L.
        box_volume_angstrom3: Total periodic box volume in $\\text{\\AA}^3$.
        protein_volume_angstrom3: Solute/protein excluded volume in $\\text{\\AA}^3$.

    Returns:
        Non-negative integer number of cosolvent copies.
    """
    if concentration_molar <= 0.0:
        return 0
    v_box = float(np.asarray(box_volume_angstrom3))
    v_prot = float(np.asarray(protein_volume_angstrom3))
    v_solvent = max(v_box - v_prot, 0.0)
    copies = int(round(float(concentration_molar) * v_solvent * AVOGADRO_ANGSTROM3_FACTOR))
    return max(copies, 0)


def _van_der_corput(indices: np.ndarray, base: int) -> np.ndarray:
    """Evaluate 1D radical-inverse (van der Corput) sequence for given integer indices."""
    if base < 2:
        msg = f"Halton base must be >= 2, got {base}"
        raise ValueError(msg)
    result = np.zeros(indices.shape, dtype=np.float64)
    denom = float(base)
    n = indices.astype(np.int64).copy()
    while np.any(n > 0):
        result += (n % base) / denom
        n //= base
        denom *= base
    return result


def halton_sequence(
    n_samples: int,
    bases: tuple[int, int, int] = (2, 3, 5),
    seed_offset: int = 1,
) -> np.ndarray:
    """Generate deterministic 3D Halton low-discrepancy points in $[0, 1)^3$.

    Args:
        n_samples: Number of 3D points to generate.
        bases: Tuple of three coprime prime bases (default ``(2, 3, 5)``).
        seed_offset: Starting 1-based index offset into the Halton sequence
            (must be ``>= 0``; default ``1`` skips the origin ``(0, 0, 0)``).

    Returns:
        NumPy float64 array of shape ``(n_samples, 3)`` with values in $[0, 1)$.
    """
    if n_samples < 0:
        msg = f"n_samples must be non-negative, got {n_samples}"
        raise ValueError(msg)
    if len(bases) != 3:
        msg = f"bases must have length 3, got {bases}"
        raise ValueError(msg)
    if seed_offset < 0:
        msg = f"seed_offset must be >= 0, got {seed_offset}"
        raise ValueError(msg)
    if n_samples == 0:
        return np.zeros((0, 3), dtype=np.float64)

    indices = np.arange(seed_offset, seed_offset + n_samples, dtype=np.int64)
    coords = [_van_der_corput(indices, int(b)) for b in bases]
    return np.stack(coords, axis=-1)


def random_rotation_matrices(key: Array, n: int) -> Array:
    """Sample ``n`` uniform $SO(3)$ rotation matrices via Shoemake's quaternion method.

    Args:
        key: JAX PRNGKey.
        n: Number of $3 \\times 3$ rotation matrices to sample.

    Returns:
        JAX float32 array of shape ``(n, 3, 3)`` containing orthogonal matrices
        with determinant $+1$.
    """
    if n < 0:
        msg = f"n must be non-negative, got {n}"
        raise ValueError(msg)
    if n == 0:
        return jnp.zeros((0, 3, 3), dtype=jnp.float32)

    u = jax.random.uniform(key, shape=(n, 3), minval=0.0, maxval=1.0, dtype=jnp.float32)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]

    sqrt_1_minus_u1 = jnp.sqrt(1.0 - u1)
    sqrt_u1 = jnp.sqrt(u1)
    two_pi_u2 = 2.0 * jnp.pi * u2
    two_pi_u3 = 2.0 * jnp.pi * u3

    qx = sqrt_1_minus_u1 * jnp.sin(two_pi_u2)
    qy = sqrt_1_minus_u1 * jnp.cos(two_pi_u2)
    qz = sqrt_u1 * jnp.sin(two_pi_u3)
    qw = sqrt_u1 * jnp.cos(two_pi_u3)

    # Convert unit quaternion (qw, qx, qy, qz) to 3x3 rotation matrix
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)

    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)

    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    row0 = jnp.stack([r00, r01, r02], axis=-1)
    row1 = jnp.stack([r10, r11, r12], axis=-1)
    row2 = jnp.stack([r20, r21, r22], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def _fractional_to_cartesian(u: Array, box: Array) -> Array:
    """Map fractional coordinates in $[0, 1)^3$ into Cartesian box coordinates."""
    box_arr = jnp.asarray(box, dtype=jnp.float32)
    u_arr = jnp.asarray(u, dtype=jnp.float32)
    if box_arr.ndim == 1:
        return u_arr * box_arr
    if box_arr.ndim == 2:
        return u_arr @ box_arr
    msg = f"box must be 1D (3,) or 2D (3, 3), got shape {box_arr.shape}"
    raise ValueError(msg)


def _pairwise_min_pbc_distance(pos_a: Array, pos_b: Array, box: Array) -> float:
    """Compute the minimum PBC distance between two point sets (or +inf if either is empty)."""
    if pos_a.shape[0] == 0 or pos_b.shape[0] == 0:
        return float("inf")
    dists = pbc.minimum_image_distance(pos_a[:, None, :], pos_b[None, :, :], box)
    return float(jnp.min(dists))


def place_cosolvents_halton(
    protein_positions: Array,
    box: Array,
    species_specs: Sequence[CosolventSpeciesSpec],
    *,
    protein_radius_angstrom: float = 5.5,
    cosolvent_placement_radius_angstrom: float = 2.5,
    post_placement_clash_angstrom: float = 1.8,
    protein_volume_angstrom3: float = 0.0,
    seed: int = 0,
    max_candidates_multiplier: int = 64,
) -> CosolventPlacementResult:
    """Place multi-component cosolvent molecules on a Halton sequence with two-stage clash gating.

    Stage 1 (Halton placement):
        Iterates over Halton candidate points inside the periodic box (orthorhombic
        ``(3,)`` or lower-triangular ``(3, 3)``), applies a deterministic SO(3)
        rotation to the centered ``atom_positions_template``, and accepts a
        placement iff:
        - Center-to-protein and all-atom-to-protein minimum image distances are
          ``>= protein_radius_angstrom`` (default ``5.5`` Å) and
          ``>= post_placement_clash_angstrom`` (default ``1.8`` Å).
        - Center-to-cosolvent-center and all-atom-to-cosolvent-atom minimum image
          distances to all previously placed cosolvent molecules are
          ``>= cosolvent_placement_radius_angstrom`` (default ``2.5`` Å) and
          ``>= post_placement_clash_angstrom`` (default ``1.8`` Å).

    Stage 2 (Post-placement all-atom clash verification):
        Computes all inter-molecular atom-atom minimum-image distances (cosolvent vs.
        protein and cosolvent molecule $m_1$ vs. cosolvent molecule $m_2 \\neq m_1$)
        and verifies every pair distance is ``>= post_placement_clash_angstrom``
        (default ``1.8`` Å).

    Args:
        protein_positions: ``(N_protein, 3)`` solute atom coordinates in Å.
        box: ``(3,)`` orthogonal box lengths or ``(3, 3)`` lower-triangular
            triclinic box matrix in Å.
        species_specs: Sequence of :class:`CosolventSpeciesSpec` definitions.
        protein_radius_angstrom: Minimum distance (Å) from protein atoms during
            placement (default ``5.5`` Å).
        cosolvent_placement_radius_angstrom: Minimum distance (Å) between
            cosolvent molecules during placement (default ``2.5`` Å).
        post_placement_clash_angstrom: Post-placement all-atom steric clash
            threshold (Å) (default ``1.8`` Å).
        protein_volume_angstrom3: Excluded protein volume ($\\text{\\AA}^3$) used when
            converting ``concentration_molar`` to copy counts.
        seed: Integer seed controlling the Halton offset and SO(3) rotations.
        max_candidates_multiplier: Candidate pool multiplier per requested copy.

    Returns:
        :class:`CosolventPlacementResult` containing combined positions,
        ``cosolvent_group_id``, ``cosolvent_mol_id``, placed cosolvent positions,
        centers, copies per species, and minimum post-placement distance.

    Raises:
        RuntimeError: If the requested number of cosolvent copies cannot be
            placed within the candidate budget or if Stage 2 clash verification fails.
    """
    prot_pos = jnp.asarray(protein_positions, dtype=jnp.float32).reshape(-1, 3)
    box_arr = jnp.asarray(box, dtype=jnp.float32)
    n_prot = prot_pos.shape[0]
    v_box = float(pbc.box_volume(box_arr))

    # Determine copy count and 1-based repulsive group ID for each species
    copies_list: list[int] = []
    species_group_ids: list[int] = []
    next_repulsive_group_id = 1
    for spec in species_specs:
        if spec.copies is not None:
            n_copies = max(int(spec.copies), 0)
        else:
            n_copies = concentration_to_copies(
                spec.concentration_molar,
                box_volume_angstrom3=v_box,
                protein_volume_angstrom3=protein_volume_angstrom3,
            )
        copies_list.append(n_copies)
        if spec.repulsive:
            species_group_ids.append(next_repulsive_group_id)
            next_repulsive_group_id += 1
        else:
            species_group_ids.append(0)

    total_copies = sum(copies_list)
    if total_copies == 0:
        return CosolventPlacementResult(
            positions=prot_pos,
            cosolvent_group_id=jnp.zeros((n_prot,), dtype=jnp.int32),
            cosolvent_mol_id=jnp.full((n_prot,), -1, dtype=jnp.int32),
            cosolvent_positions=jnp.zeros((0, 3), dtype=jnp.float32),
            cosolvent_centers=jnp.zeros((0, 3), dtype=jnp.float32),
            copies_per_species=tuple(copies_list),
            min_post_placement_distance=float("inf"),
        )

    n_candidates = max(total_copies * max_candidates_multiplier, 256)
    seed_offset = max(1, int(abs(seed)) * 997 + 1)
    halton_pts = halton_sequence(n_candidates, bases=(2, 3, 5), seed_offset=seed_offset)
    candidate_centers = _fractional_to_cartesian(jnp.asarray(halton_pts, dtype=jnp.float32), box_arr)
    rot_matrices = random_rotation_matrices(jax.random.PRNGKey(int(seed)), n_candidates)

    placed_mol_atoms: list[Array] = []
    placed_centers_list: list[Array] = []
    placed_group_ids: list[Array] = []
    placed_mol_ids: list[Array] = []

    prot_cutoff = max(float(protein_radius_angstrom), float(post_placement_clash_angstrom))
    cosolv_cutoff = max(float(cosolvent_placement_radius_angstrom), float(post_placement_clash_angstrom))

    candidate_cursor = 0
    mol_counter = 0

    for spec, n_copies, group_id in zip(species_specs, copies_list, species_group_ids):
        if n_copies == 0:
            continue

        if spec.atom_positions_template is None:
            template = jnp.zeros((1, 3), dtype=jnp.float32)
        else:
            template = jnp.asarray(spec.atom_positions_template, dtype=jnp.float32).reshape(-1, 3)
            template = template - jnp.mean(template, axis=0, keepdims=True)

        n_atoms_mol = template.shape[0]
        placed_for_species = 0

        while placed_for_species < n_copies:
            if candidate_cursor >= n_candidates:
                # Expand Halton candidate pool deterministically if needed
                extra = n_candidates
                more_halton = halton_sequence(
                    extra,
                    bases=(2, 3, 5),
                    seed_offset=seed_offset + n_candidates,
                )
                more_centers = _fractional_to_cartesian(
                    jnp.asarray(more_halton, dtype=jnp.float32), box_arr
                )
                more_rots = random_rotation_matrices(
                    jax.random.PRNGKey(int(seed) + n_candidates), extra
                )
                candidate_centers = jnp.concatenate([candidate_centers, more_centers], axis=0)
                rot_matrices = jnp.concatenate([rot_matrices, more_rots], axis=0)
                n_candidates += extra
                if n_candidates > max(total_copies * max_candidates_multiplier * 8, 4096):
                    msg = (
                        f"Could not place {n_copies} copies of cosolvent '{spec.name}' "
                        f"(placed {placed_for_species}) within {n_candidates} Halton candidates."
                    )
                    raise RuntimeError(msg)

            center = candidate_centers[candidate_cursor]
            rot = rot_matrices[candidate_cursor]
            candidate_cursor += 1

            # Check center distance to protein
            if n_prot > 0:
                d_center_prot = _pairwise_min_pbc_distance(center[None, :], prot_pos, box_arr)
                if d_center_prot < prot_cutoff:
                    continue

            # Check center distance to already-placed cosolvent centers
            if placed_centers_list:
                prev_centers = jnp.stack(placed_centers_list, axis=0)
                d_center_cosolv = _pairwise_min_pbc_distance(center[None, :], prev_centers, box_arr)
                if d_center_cosolv < cosolv_cutoff:
                    continue

            # Wrap center into the primary periodic cell and construct candidate coordinates
            center = pbc.wrap_positions(center[None, :], box_arr)[0]
            mol_coords = (template @ rot.T) + center[None, :]

            # All-atom distance check against protein
            if n_prot > 0:
                d_atoms_prot = _pairwise_min_pbc_distance(mol_coords, prot_pos, box_arr)
                if d_atoms_prot < prot_cutoff:
                    continue

            # All-atom distance check against already-placed cosolvent atoms
            if placed_mol_atoms:
                prev_atoms = jnp.concatenate(placed_mol_atoms, axis=0)
                d_atoms_cosolv = _pairwise_min_pbc_distance(mol_coords, prev_atoms, box_arr)
                if d_atoms_cosolv < cosolv_cutoff:
                    continue

            placed_mol_atoms.append(mol_coords)
            placed_centers_list.append(center)
            placed_group_ids.append(jnp.full((n_atoms_mol,), group_id, dtype=jnp.int32))
            placed_mol_ids.append(jnp.full((n_atoms_mol,), mol_counter, dtype=jnp.int32))
            mol_counter += 1
            placed_for_species += 1

    cosolv_positions = jnp.concatenate(placed_mol_atoms, axis=0)
    cosolv_centers = jnp.stack(placed_centers_list, axis=0)
    cosolv_group_id_arr = jnp.concatenate(placed_group_ids, axis=0)
    cosolv_mol_id_arr = jnp.concatenate(placed_mol_ids, axis=0)

    # Stage 2: Post-placement all-atom clash verification across all distinct molecules
    min_observed_dist = float("inf")
    if n_prot > 0 and cosolv_positions.shape[0] > 0:
        min_observed_dist = min(
            min_observed_dist,
            _pairwise_min_pbc_distance(cosolv_positions, prot_pos, box_arr),
        )
    if cosolv_positions.shape[0] > 1:
        dists_cc = pbc.minimum_image_distance(
            cosolv_positions[:, None, :], cosolv_positions[None, :, :], box_arr
        )
        diff_mol_mask = cosolv_mol_id_arr[:, None] != cosolv_mol_id_arr[None, :]
        if bool(jnp.any(diff_mol_mask)):
            masked_dists = jnp.where(diff_mol_mask, dists_cc, jnp.float32(jnp.inf))
            min_observed_dist = min(min_observed_dist, float(jnp.min(masked_dists)))

    if min_observed_dist < float(post_placement_clash_angstrom):
        msg = (
            f"Post-placement steric clash check failed: minimum inter-molecular distance "
            f"{min_observed_dist:.4f} Å < {post_placement_clash_angstrom:.4f} Å."
        )
        raise RuntimeError(msg)

    combined_positions = jnp.concatenate([prot_pos, cosolv_positions], axis=0)
    combined_group_id = jnp.concatenate(
        [jnp.zeros((n_prot,), dtype=jnp.int32), cosolv_group_id_arr], axis=0
    )
    combined_mol_id = jnp.concatenate(
        [jnp.full((n_prot,), -1, dtype=jnp.int32), cosolv_mol_id_arr], axis=0
    )

    return CosolventPlacementResult(
        positions=combined_positions,
        cosolvent_group_id=combined_group_id,
        cosolvent_mol_id=combined_mol_id,
        cosolvent_positions=cosolv_positions,
        cosolvent_centers=cosolv_centers,
        copies_per_species=tuple(copies_list),
        min_post_placement_distance=min_observed_dist,
    )


def openmm_quintic_switch(
    r: Array,
    r_switch: float | Array,
    r_cutoff: float | Array,
) -> Array:
    """Evaluate OpenMM's $C^2$-continuous quintic switching function $S_{\\text{OpenMM}}(r)$.

    For $x = (r - r_{\\text{switch}}) / (r_{\\text{cutoff}} - r_{\\text{switch}})$:
    - $S(r) = 1$ for $r \\le r_{\\text{switch}}$
    - $S(r) = 1 - 10 x^3 + 15 x^4 - 6 x^5$ for $r_{\\text{switch}} < r < r_{\\text{cutoff}}$
    - $S(r) = 0$ for $r \\ge r_{\\text{cutoff}}$
    """
    r_sw = jnp.asarray(r_switch, dtype=r.dtype)
    r_cut = jnp.asarray(r_cutoff, dtype=r.dtype)
    width = jnp.maximum(r_cut - r_sw, jnp.asarray(1e-6, dtype=r.dtype))
    x = jnp.clip((r - r_sw) / width, jnp.asarray(0.0, dtype=r.dtype), jnp.asarray(1.0, dtype=r.dtype))
    poly = 1.0 - 10.0 * (x ** 3) + 15.0 * (x ** 4) - 6.0 * (x ** 5)
    return jnp.where(r <= r_sw, jnp.ones_like(r), jnp.where(r >= r_cut, jnp.zeros_like(r), poly))


def compute_ionic_neutralization_counts(
    net_charge_e: float,
    box_volume_angstrom3: float | Array,
    ionic_strength_molar: float = 0.15,
    protein_volume_angstrom3: float | Array = 0.0,
) -> tuple[int, int]:
    """Compute monovalent (Na+, Cl-) ion counts for charge neutralization and target ionic strength."""
    n_salt = concentration_to_copies(
        max(float(ionic_strength_molar), 0.0),
        box_volume_angstrom3=box_volume_angstrom3,
        protein_volume_angstrom3=protein_volume_angstrom3,
    )
    q_int = int(round(float(net_charge_e)))
    if q_int < 0:
        n_pos = n_salt + abs(q_int)
        n_neg = n_salt
    elif q_int > 0:
        n_pos = n_salt
        n_neg = n_salt + q_int
    else:
        n_pos = n_salt
        n_neg = n_salt
    return int(n_pos), int(n_neg)


def cosolvent_repulsive_energy(
    positions: Array,
    cosolvent_group_id: Array,
    cosolvent_mol_id: Array,
    displacement_fn: space.DisplacementFn,
    *,
    eps_kcal: float = 0.01,
    r0_angstrom: float = 10.0,
    cutoff_angstrom: float = 9.0,
    deduplicate_interaction_groups: bool = False,
    atom_mask: Array | None = None,
    repulsive_mode: str = "r12",
    include_inter_species: bool = False,
) -> Array:
    """Compute all-atom per-species inter-cosolvent repulsive potential energy (CA-4 / FR-7).

    Evaluates:
        $$U_{\\text{rep}}(r) = \\text{mult} \\cdot \\epsilon_{\\text{rep}}
        \\left(\\frac{r_0}{r}\\right)^{12}
        S_{\\text{OpenMM}}(r; r_{\\text{switch}}, r_{\\text{cutoff}})$$
    over unordered pairs of atoms $(i, j)$ satisfying:
    - ``cosolvent_group_id[i] == cosolvent_group_id[j]``
    - ``cosolvent_group_id[i] > 0`` (repulsive cosolvent species)
    - ``cosolvent_mol_id[i] != cosolvent_mol_id[j]`` (distinct cosolvent molecules)

    where $r_{\\text{switch}} = r_{\\text{cutoff}} - 1.0\\text{ \\AA}$, and:
    - ``mult = 2.0`` when ``deduplicate_interaction_groups=False`` (default),
      reproducing ``cosolvkit``'s $2\\times$ energy/force factor caused by calling
      ``CustomNonbondedForce.addInteractionGroup(ag, ag)`` twice in its nested loop.
    - ``mult = 1.0`` when ``deduplicate_interaction_groups=True``, evaluating the
      deduplicated $1\\times$ physical unordered pair potential.

    Args:
        positions: ``(N, 3)`` atomic positions in Å.
        cosolvent_group_id: ``(N,)`` int32 species group IDs (``0`` for
            protein/water/non-repulsive, ``g >= 1`` for repulsive cosolvent species).
        cosolvent_mol_id: ``(N,)`` int32 molecule IDs (``-1`` for non-cosolvent,
            unique ``m >= 0`` per cosolvent molecule).
        displacement_fn: JAX-MD periodic or free-space displacement function.
        eps_kcal: Repulsive energy scale $\\epsilon_{\\text{rep}}$ in kcal/mol
            (default ``0.01``).
        r0_angstrom: Repulsive reference distance $r_0$ in Å (default ``7.0``).
        cutoff_angstrom: Repulsive cutoff $r_{\\text{cutoff}}$ in Å (default ``9.0``;
            switch onset is $r_{\\text{switch}} = r_{\\text{cutoff}} - 1.0$ Å).
        deduplicate_interaction_groups: If ``False`` (default), applies the $2\\times$
            ``cosolvkit`` interaction-group duplication factor; if ``True``, uses $1\\times$.
        atom_mask: Optional ``(N,)`` boolean mask for real (non-padded) atoms.

    Returns:
        Scalar repulsive potential energy in kcal/mol.
    """
    r = jnp.asarray(positions)
    gid = jnp.asarray(cosolvent_group_id, dtype=jnp.int32)
    mid = jnp.asarray(cosolvent_mol_id, dtype=jnp.int32)
    n = r.shape[0]
    if n <= 1:
        return jnp.zeros((), dtype=r.dtype)

    if include_inter_species:
        species_ok = (gid[:, None] > 0) & (gid[None, :] > 0)
    else:
        species_ok = (gid[:, None] == gid[None, :]) & (gid[:, None] > 0)
    diff_molecule = (mid[:, None] != mid[None, :]) & (mid[:, None] >= 0) & (mid[None, :] >= 0)
    no_self = ~jnp.eye(n, dtype=bool)
    pair_mask = species_ok & diff_molecule & no_self
    if atom_mask is not None:
        amask = jnp.asarray(atom_mask, dtype=bool)
        pair_mask = pair_mask & amask[:, None] & amask[None, :]

    dr = space.map_product(displacement_fn)(r, r)
    # Gradient-safe distance: substitute r0_angstrom for masked-out pairs before (r0 / r)^12
    dist_raw = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + jnp.asarray(1e-12, dtype=r.dtype))
    r0 = jnp.asarray(r0_angstrom, dtype=r.dtype)
    r_cut = jnp.asarray(cutoff_angstrom, dtype=r.dtype)
    r_sw = r_cut - jnp.asarray(1.0, dtype=r.dtype)
    eps = jnp.asarray(eps_kcal, dtype=r.dtype)

    dist_safe = jnp.where(pair_mask, jnp.maximum(dist_raw, jnp.asarray(1e-4, dtype=r.dtype)), r0)
    if str(repulsive_mode).lower() == "harmonic":
        delta_r = jnp.maximum(r0 - dist_safe, jnp.asarray(0.0, dtype=r.dtype))
        e_pair = 0.5 * eps * (delta_r * delta_r)
    else:
        inv_r6 = (r0 / dist_safe) ** 6
        inv_r12 = inv_r6 * inv_r6
        sw = openmm_quintic_switch(dist_safe, r_sw, r_cut)
        e_pair = eps * inv_r12 * sw
    e_pair = jnp.where(pair_mask & (dist_raw < r_cut), e_pair, jnp.zeros_like(e_pair))

    mult = jnp.asarray(1.0 if deduplicate_interaction_groups else 2.0, dtype=r.dtype)
    # 0.5 accounts for symmetric (i, j) and (j, i) matrix sum over unordered pairs
    return jnp.asarray(0.5, dtype=r.dtype) * mult * jnp.sum(e_pair)
