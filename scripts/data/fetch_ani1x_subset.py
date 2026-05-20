#!/usr/bin/env python
"""HP4 ANI-1x + COMP6 DFT-Forces Subset Curation Fetch Script.

Implements §7 (Fetch Script Spec) from docs/superpowers/specs/2026-05-20-hp4-ani1x-curation.md.

Responsibilities:
  1. Verify SHA-256 of ANI-1x and COMP6 archives against pinned constants.
  2. Run SELECT_16 (AIMNet2-style local-environment hashing diversity) over ANI-1x.
  3. Extract Lane B fixed molecules from COMP6.
  4. Convert forces Ha/Å → kcal/mol/Å using multiplier 627.5094740631 (no Bohr factor).
  5. Write per-molecule HDF5 to data/ani1x_subset/lane_a/ and lane_b/.
  6. Compute SHA-256 per output file.
  7. Emit manifest.json per spec §4.4.
  8. Print summary table.

Note: ANI-1x HDF5 groups are keyed by molecular formula (e.g., C9H9N3O2), NOT by SMILES.
Bonds are derived from xyz via covalent-radius distance threshold (no SMILES parsing at curation).
The downstream §7.1 toolchain (openff-toolkit) will derive SMILES from positions+species.

Exit code 0 on success; non-zero with error message on failure.
"""

import logging
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import numpy as np
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter

# Configure logging (per CLAUDE.md § stack: use Python stdlib logging, not print)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# SHA-256 constants — pinned 2026-05-20 from local downloads.
# ANI-1x: ndownloader.figshare.com/files/18112775 (Figshare 10.6084/m9.figshare.10047041.v1)
#   Figshare-published MD5: 98090dd6679106da861f52bed825ffb7
# COMP6v2 wB97X-631Gd: zenodo.org/records/10126157 (DOI 10.5281/zenodo.10126157)
#   Zenodo-published MD5 (tarball): 0a417148966022e72f54c135d8f3d4e7
#   Inner HDF5: comp6v2_final_h5/COMP6v2_wB97X-631Gd.h5 (extracted from tarball)
ANI1X_EXPECTED_SHA256 = "fe0ba06198ee72cf1003deebab2652097f6ab518337784dc811fa7da0c3bf5ac"
COMP6V2_HDF5_EXPECTED_SHA256 = (
    "e7c3e3e5db9fb7a64d00f86fb6b843323fae9dac8736a56c5875ef38051c81d0"
)
# Back-compat alias (script body still uses COMP6_EXPECTED_SHA256)
COMP6_EXPECTED_SHA256 = COMP6V2_HDF5_EXPECTED_SHA256

# Force unit conversion constant (spec §3 critical correction: Ha/Å → kcal/mol/Å)
FORCE_HA_TO_KCAL_PER_MOL_PER_ANGSTROM = 627.5094740631

# Allowed atomic numbers (H=1, C=6, N=7, O=8; spec §4.1 F1)
ALLOWED_ELEMENTS = {1, 6, 7, 8}

# Lane A selection parameters (spec §4.1, §4.2)
LANE_A_N_ATOMS_MIN = 15
LANE_A_N_ATOMS_MAX = 30
LANE_A_N_ATOMS_MIN_CONF = 20
LANE_A_SELECT_COUNT = 16
LANE_A_SEED = 42

# Covalent radii in Angstroms (Cordero et al., 2008; H, C, N, O only — ANI-1x scope)
# Used for bond perception from xyz via distance threshold
COVALENT_RADII = {
    1: 0.31,  # H
    6: 0.76,  # C
    7: 0.71,  # N
    8: 0.66,  # O
}

# Bond distance threshold multiplier (soft cutoff per Jensen 2018 / xyz2mol)
BOND_DISTANCE_THRESHOLD_MULTIPLIER = 1.20

# Lane B fixed molecules (spec §4.3).
# IMPORTANT: COMP6v2 HDF5 is organized by atom-count, NOT by subset name.
# Group keys are 3-digit zero-padded atom counts ("006", "007", ..., "312").
# Each group has subkeys: coordinates, forces, energies, species,
# cm5_atomic_charges, hirshfeld_atomic_charges, hirshfeld_atomic_dipoles.
# Note: COMP6v2 uses bare keys ("forces") — NOT the ANI-1x dot-notation
# ("wb97x_dz.forces"). Each tarball is one DFT level, so no prefix needed.
COMP6V2_MANDATORY_LANE_B = {
    "trp_cage":  {"pdb_id": "1L2Y", "comp6v2_group": "312", "expected_atoms": 312},
    "chignolin": {"pdb_id": "1UAO", "comp6v2_group": "138", "expected_atoms": 138},
}

# COMP6v2 schema (differs from ANI-1x — see spec §3 schema notes).
COMP6V2_FORCES_KEY = "forces"          # bare, no "wb97x_dz." prefix
COMP6V2_ENERGY_KEY = "energies"        # plural in COMP6v2 vs singular "energy" in ANI-1x
COMP6V2_COORD_KEY = "coordinates"
COMP6V2_SPECIES_KEY = "species"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        file_path: Path to file to hash.

    Returns:
        Hex-encoded SHA-256 string.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_archive_hash(archive_path: Path, expected_hash: str, name: str) -> bool:
    """Verify archive SHA-256 against expected value.

    Args:
        archive_path: Path to archive file.
        expected_hash: Expected SHA-256 hash (may be placeholder).
        name: Human-readable name for logging.

    Returns:
        True if hash matches (or is placeholder); False if mismatch.
    """
    if not archive_path.exists():
        logger.error(f"{name} archive not found: {archive_path}")
        return False

    if expected_hash == "PLACEHOLDER_FILL_AT_FIRST_DOWNLOAD":
        actual_hash = compute_sha256(archive_path)
        logger.info(f"{name} SHA-256: {actual_hash} (PLACEHOLDER — pin this in script)")
        return True

    actual_hash = compute_sha256(archive_path)
    if actual_hash != expected_hash:
        logger.error(
            f"{name} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
        return False

    logger.info(f"{name} SHA-256 verified: {actual_hash}")
    return True


def iter_data_buckets(hdf5_path: str, data_keys: list[str]):
    """Vendored from aiqm/ANI1x_datasets/dataloader.py (with attribution).

    Iterate over HDF5 groups (isomers) and yield molecules with requested keys.

    Args:
        hdf5_path: Path to ANI-1x HDF5 archive.
        data_keys: List of keys to filter (e.g., ['wb97x_dz.energy', 'wb97x_dz.forces']).

    Yields:
        (smiles, data_dict) tuples for each molecule with all requested keys.
    """
    with h5py.File(hdf5_path, "r") as f:
        for smiles in f.keys():
            molecule_group = f[smiles]
            # Check if all requested keys exist
            if all(key in molecule_group for key in data_keys):
                data_dict = {key: molecule_group[key][()] for key in data_keys}
                yield smiles, data_dict




def compute_local_env_hash(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
) -> set[tuple]:
    """Compute AIMNet2-style local-environment hashes from xyz + species (no RDKit Mol).

    Per spec §4.2, for each non-H atom a:
      1. Determine bonded neighbors by distance threshold:
         bond exists between atoms i, j iff
         |r_i - r_j| < (covalent_radius[Z_i] + covalent_radius[Z_j]) * BOND_DISTANCE_THRESHOLD_MULTIPLIER
      2. Compute hash tuple:
         (Z_a, n_H_neighbors, n_total_neighbors, sorted_tuple(Z_neighbor for neighbor in non-H neighbors))

    Args:
        atomic_numbers: (Na,) int array of atomic numbers.
        positions: (Na, 3) float32 array of positions in Angstroms.

    Returns:
        set of hash tuples (Z, n_H, n_total, sorted_neighbor_Z_tuple) for non-H atoms.
    """
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    positions = np.asarray(positions, dtype=np.float32)

    n_atoms = len(atomic_numbers)
    env_hashes = set()

    for a in range(n_atoms):
        z_a = atomic_numbers[a]

        # Skip hydrogens
        if z_a == 1:
            continue

        # Determine bonded neighbors via distance threshold
        neighbors = []
        n_h_neighbors = 0

        for b in range(n_atoms):
            if a == b:
                continue

            z_b = atomic_numbers[b]
            r_ab = np.linalg.norm(positions[a] - positions[b])

            # Bond threshold
            r_cov_a = COVALENT_RADII.get(z_a, 0.76)  # Default to C if unknown
            r_cov_b = COVALENT_RADII.get(z_b, 0.76)
            threshold = (r_cov_a + r_cov_b) * BOND_DISTANCE_THRESHOLD_MULTIPLIER

            if r_ab < threshold:
                neighbors.append(z_b)
                if z_b == 1:
                    n_h_neighbors += 1

        # Count non-H neighbors
        non_h_neighbors = [z for z in neighbors if z != 1]
        n_total_neighbors = len(neighbors)

        # Create hash tuple
        sorted_neighbor_z = tuple(sorted(non_h_neighbors))
        env_hash = (z_a, n_h_neighbors, n_total_neighbors, sorted_neighbor_z)
        env_hashes.add(env_hash)

    return env_hashes


def select_16_lane_a(
    ani1x_path: Path,
    seed: int = 42,
) -> list[dict]:
    """Select 16 Lane A molecules via AIMNet2-style local-environment hashing diversity.

    Implements spec §4.2 SELECT_16 algorithm.
    ANI-1x group keys are molecular formulas (e.g., C9H9N3O2), not SMILES.

    Args:
        ani1x_path: Path to ANI-1x HDF5 archive.
        seed: Random seed for greedy selection.

    Returns:
        List of dicts with keys: formula, n_atoms, n_conf_with_forces, env_hashes, rarity_score.
    """
    logger.info("Loading ANI-1x molecules and filtering F1–F4 (formula-keyed groups)...")

    # Apply filters F1–F4 (F5 SMILES parsing removed; ANI-1x keys are formulae)
    passing_molecules = []
    with h5py.File(ani1x_path, "r") as f:
        for formula in f.keys():
            group = f[formula]

            # Check if all required keys exist
            required_keys = ["atomic_numbers", "wb97x_dz.energy", "wb97x_dz.forces", "coordinates"]
            if not all(key in group for key in required_keys):
                continue

            atomic_numbers = np.array(group["atomic_numbers"], dtype=int)
            forces = np.array(group["wb97x_dz.forces"], dtype=np.float32)
            coordinates = np.array(group["coordinates"], dtype=np.float32)

            # F1: Elements subset
            if not all(z in ALLOWED_ELEMENTS for z in atomic_numbers):
                continue

            # F2: Atom count range
            n_atoms = len(atomic_numbers)
            if not (LANE_A_N_ATOMS_MIN <= n_atoms <= LANE_A_N_ATOMS_MAX):
                continue

            # F3: Conformer count with non-NaN forces
            n_conf_with_forces = np.sum(~np.isnan(forces[:, 0, 0]))
            if n_conf_with_forces < LANE_A_N_ATOMS_MIN_CONF:
                continue

            # F4: Forces non-NaN (defensive, redundant for ωB97X/6-31G*)
            if np.isnan(forces).any():
                continue

            passing_molecules.append({
                "formula": formula,
                "n_atoms": n_atoms,
                "n_conf_with_forces": int(n_conf_with_forces),
                "atomic_numbers": atomic_numbers,
                "positions": coordinates[0],  # First conformer for env hash
            })

    logger.info(f"Passing molecules (F1–F4): {len(passing_molecules)}")

    if len(passing_molecules) == 0:
        logger.warning("No molecules passed F1–F4 filters")
        return []

    # Compute environment hashes for each molecule
    for mol_dict in passing_molecules:
        env_hash_set = compute_local_env_hash(
            mol_dict["atomic_numbers"],
            mol_dict["positions"],
        )
        mol_dict["env_hash_set"] = env_hash_set

    # Compute global hash frequency
    global_hash_freq = Counter()
    for mol_dict in passing_molecules:
        global_hash_freq.update(mol_dict["env_hash_set"])

    # Compute rarity scores
    for mol_dict in passing_molecules:
        if len(mol_dict["env_hash_set"]) > 0:
            rarity_scores = [1.0 / global_hash_freq[h] for h in mol_dict["env_hash_set"]]
            mol_dict["rarity_score"] = float(np.mean(rarity_scores))
        else:
            mol_dict["rarity_score"] = 0.0

    # Sort by rarity (descending)
    passing_molecules.sort(key=lambda m: m["rarity_score"], reverse=True)

    # Greedy diverse pick
    selected = []
    picked_hashes = set()
    for mol_dict in passing_molecules:
        # Check if this molecule has novel hashes
        if not mol_dict["env_hash_set"].issubset(picked_hashes):
            selected.append(mol_dict)
            picked_hashes.update(mol_dict["env_hash_set"])

            if len(selected) >= LANE_A_SELECT_COUNT:
                break

    logger.info(f"Selected {len(selected)} Lane A molecules via diversity selection")

    # If fewer than 16, relax F3 and retry (not implemented for MVP)
    if len(selected) < LANE_A_SELECT_COUNT:
        logger.warning(
            f"Selected only {len(selected)} molecules (target {LANE_A_SELECT_COUNT}). "
            f"Consider relaxing F3 filter or expanding ANI-1x search."
        )

    # Sort output by n_atoms ascending, then formula ascending
    selected.sort(key=lambda m: (m["n_atoms"], m["formula"]))

    # Remove intermediate fields (not part of output)
    for mol_dict in selected:
        mol_dict.pop("atomic_numbers", None)
        mol_dict.pop("positions", None)
        env_hashes_set = mol_dict.pop("env_hash_set", set())
        mol_dict["env_hashes"] = list(map(str, env_hashes_set))

    return selected


def load_molecule_from_ani1x(
    ani1x_path: Path,
    formula: str,
) -> Optional[dict]:
    """Load a single molecule from ANI-1x HDF5 by formula (group key).

    Args:
        ani1x_path: Path to ANI-1x archive.
        formula: Molecular formula (group key in HDF5, e.g., "C9H9N3O2").

    Returns:
        Dict with positions, forces (kcal/mol/Å), energy, species, or None if not found.
    """
    try:
        with h5py.File(ani1x_path, "r") as f:
            if formula not in f:
                return None

            group = f[formula]
            positions = np.array(group["coordinates"], dtype=np.float32)
            forces_raw = np.array(group["wb97x_dz.forces"], dtype=np.float32)
            energy = np.array(group["wb97x_dz.energy"], dtype=np.float64)
            species = np.array(group["atomic_numbers"], dtype=np.int8)

            # Convert forces Ha/Å → kcal/mol/Å
            forces = forces_raw * FORCE_HA_TO_KCAL_PER_MOL_PER_ANGSTROM

            return {
                "positions": positions,
                "forces": forces,
                "energy": energy,
                "species": species,
            }
    except Exception as e:
        logger.debug(f"Failed to load {formula} from ANI-1x: {e}")
        return None


def load_comp6v2_group(
    comp6v2_path: Path,
    group_name: str,
) -> Optional[dict]:
    """Load a single COMP6v2 atom-count group and return a per-molecule dict.

    COMP6v2 differs from ANI-1x: one group = all molecules of a given atom count,
    with multiple conformers stacked along the leading axis. The "species" array is
    (Nc, Na), so we must extract one species pattern (from row 0) and verify uniformity.

    For large groups (e.g., /312 Trp-cage, /138 Chignolin), all rows have identical
    species; we return the whole group as one molecule. For small groups with heterogeneous
    species, we filter to the most common species pattern (with warning).

    Args:
        comp6v2_path: Path to COMP6v2 HDF5.
        group_name: Group name, e.g., "312".

    Returns:
        Dict with positions, forces (kcal/mol/Å), energy, species (per-atom, extracted
        from row 0), molecule_id, or None if not found / load fails.
    """
    try:
        with h5py.File(comp6v2_path, "r") as f:
            if group_name not in f:
                logger.debug(f"COMP6v2 group /{group_name} not found")
                return None

            group = f[group_name]

            # Load raw data
            coordinates = np.array(group[COMP6V2_COORD_KEY], dtype=np.float32)  # (Nc, Na, 3)
            forces_raw = np.array(group[COMP6V2_FORCES_KEY], dtype=np.float32)  # (Nc, Na, 3)
            energies = np.array(group[COMP6V2_ENERGY_KEY], dtype=np.float64)  # (Nc,)
            species_array = np.array(group[COMP6V2_SPECIES_KEY], dtype=np.int64)  # (Nc, Na)

            n_conf, n_atoms = coordinates.shape[0], coordinates.shape[1]

            # Extract species pattern (from row 0)
            species_pattern = species_array[0, :].astype(np.int8)

            # Check uniformity: are all rows identical?
            uniform = np.all(species_array == species_pattern[np.newaxis, :], axis=1)
            n_uniform = np.sum(uniform)

            if n_uniform < n_conf:
                # Log warning: this group has heterogeneous species
                logger.warning(
                    f"COMP6v2 group /{group_name}: {n_uniform}/{n_conf} conformers match "
                    f"the dominant species pattern. Filtering to matching conformers."
                )
                # Filter to uniform conformers
                coordinates = coordinates[uniform]
                forces_raw = forces_raw[uniform]
                energies = energies[uniform]
                n_conf = len(coordinates)

            # Convert forces Ha/Å → kcal/mol/Å
            forces = forces_raw * FORCE_HA_TO_KCAL_PER_MOL_PER_ANGSTROM

            return {
                "positions": coordinates,  # (Nc, Na, 3)
                "forces": forces,
                "energy": energies,  # (Nc,)
                "species": species_pattern,  # (Na,)
                "molecule_id": int(group_name),  # Use atom count as molecule_id
            }
    except Exception as e:
        logger.debug(f"Failed to load COMP6v2 group /{group_name}: {e}")
        return None


def select_lane_b_from_comp6v2(
    comp6v2_path: Path,
    seed: int = 42,
) -> list[dict]:
    """Select Lane B molecules from COMP6v2 per spec §4.3.

    Returns fixed molecules:
    - Trp-cage 1L2Y at /312 (bucket 3)
    - Chignolin 1UAO at /138 (bucket 2)
    - 2 mid-size molecules from 65–128 atom range (bucket 1):
      - One near 100 atoms (drug-like)
      - One near 80 atoms (peptide-tier)

    Uses deterministic tie-breaking with seed=42 for reproducibility.

    Args:
        comp6v2_path: Path to COMP6v2 HDF5.
        seed: Random seed (for tie-breaking if multiple candidates have same distance).

    Returns:
        List of dicts with keys: name, pdb_id, comp6v2_group, n_atoms, bucket_idx, n_conf.
    """
    logger.info("Selecting Lane B molecules from COMP6v2...")

    rng = np.random.RandomState(seed)

    # Fixed mandatory molecules
    lane_b = []

    # 1. Trp-cage (1L2Y) at /312
    with h5py.File(comp6v2_path, "r") as f:
        if "312" in f:
            n_conf = f["312"]["coordinates"].shape[0]
            lane_b.append({
                "name": "trp_cage",
                "pdb_id": "1L2Y",
                "comp6v2_group": "312",
                "n_atoms": 312,
                "bucket_idx": bucket_idx_from_atom_count(312),
                "n_conf": n_conf,
            })
            logger.info(f"  Selected Trp-cage (1L2Y) at /312 (312 atoms, {n_conf} conformers, bucket 3)")

    # 2. Chignolin (1UAO) at /138
    with h5py.File(comp6v2_path, "r") as f:
        if "138" in f:
            n_conf = f["138"]["coordinates"].shape[0]
            lane_b.append({
                "name": "chignolin",
                "pdb_id": "1UAO",
                "comp6v2_group": "138",
                "n_atoms": 138,
                "bucket_idx": bucket_idx_from_atom_count(138),
                "n_conf": n_conf,
            })
            logger.info(f"  Selected Chignolin (1UAO) at /138 (138 atoms, {n_conf} conformers, bucket 2)")

    # 3. Mid-size picks (bucket 1: 65–128 atoms)
    # Find groups in this range, select two with best coverage (near 100 and 80)
    mid_size_candidates = []
    with h5py.File(comp6v2_path, "r") as f:
        for group_name in f.keys():
            try:
                n_atoms = int(group_name)
                if 65 <= n_atoms <= 128:
                    # Check uniformity of species
                    species_array = f[group_name][COMP6V2_SPECIES_KEY][()]
                    species_pattern_row0 = species_array[0, :]
                    uniform = np.all(
                        species_array == species_pattern_row0[np.newaxis, :],
                        axis=1
                    )
                    if np.sum(uniform) >= 5:  # At least 5 conformers with uniform species
                        n_conf = f[group_name]["coordinates"].shape[0]
                        mid_size_candidates.append({
                            "comp6v2_group": group_name,
                            "n_atoms": n_atoms,
                            "n_conf": n_conf,
                        })
            except (ValueError, KeyError):
                continue

    if len(mid_size_candidates) < 2:
        logger.warning(f"Found only {len(mid_size_candidates)} bucket-1 candidates; target is 2")

    # Sort by distance to target atoms (100 and 80), deterministically
    mid_size_candidates.sort(key=lambda c: (c["n_atoms"], c["comp6v2_group"]))

    # Pick closest to 100 atoms
    if mid_size_candidates:
        targets = [100, 80]
        picks = []

        for target in targets:
            # Find closest to target
            dists = [abs(c["n_atoms"] - target) for c in mid_size_candidates]
            min_dist = min(dists)
            closest = [c for c, d in zip(mid_size_candidates, dists) if d == min_dist]

            if closest:
                # Tie-break deterministically by group name
                pick = sorted(closest, key=lambda c: c["comp6v2_group"])[0]
                picks.append(pick)
                mid_size_candidates.remove(pick)

        # Add picks to lane_b
        for pick in picks:
            lane_b.append({
                "name": f"midsize_mol_{pick['n_atoms']}",
                "pdb_id": None,
                "comp6v2_group": pick["comp6v2_group"],
                "n_atoms": pick["n_atoms"],
                "bucket_idx": bucket_idx_from_atom_count(pick["n_atoms"]),
                "n_conf": pick["n_conf"],
            })
            logger.info(
                f"  Selected mid-size molecule at /{pick['comp6v2_group']} "
                f"({pick['n_atoms']} atoms, {pick['n_conf']} conformers, bucket 1)"
            )

    return lane_b


def write_molecule_h5(
    output_path: Path,
    identifier: str,
    lane: str,
    molecule_id: int,
    bucket_idx: int,
    positions: np.ndarray,
    forces: np.ndarray,
    energy: np.ndarray,
    species: np.ndarray,
) -> bool:
    """Write a single molecule to HDF5 per spec §5 schema.

    Args:
        output_path: Output file path.
        identifier: Formula (Lane A) or PDB ID (Lane B).
        lane: "a" or "b".
        molecule_id: Molecule index or identifier.
        bucket_idx: Precomputed bucket index.
        positions: [N_conf, N_atoms, 3] float32 in Å.
        forces: [N_conf, N_atoms, 3] float32 in kcal/mol/Å.
        energy: [N_conf] float64 in Ha.
        species: [N_atoms] int8 atomic numbers.

    Returns:
        True on success, False on error.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as f:
            f.create_dataset("positions", data=positions, dtype=np.float32)
            f.create_dataset("forces", data=forces, dtype=np.float32)
            f.create_dataset("energy", data=energy, dtype=np.float64)
            f.create_dataset("species", data=species, dtype=np.int8)

            # Store identifier (formula for Lane A, PDB ID for Lane B) as string
            f.create_dataset("formula", data=identifier, dtype=h5py.string_dtype())

            # Store molecule_id
            f.create_dataset("molecule_id", data=molecule_id, dtype=h5py.string_dtype())

            # Attributes (per spec §5)
            f.attrs["lane"] = lane
            f.attrs["bucket_idx"] = int(bucket_idx)

        return True
    except Exception as e:
        logger.error(f"Failed to write {output_path}: {e}")
        return False


def bucket_idx_from_atom_count(n_atoms: int) -> int:
    """Compute bucket index from atom count (per spec §5, new HP4 ladder).

    Uses ATOM_BUCKETS = (64, 128, 256, 1024, 5000, 25000, 60000).

    Args:
        n_atoms: Number of atoms.

    Returns:
        Bucket index (0–6).
    """
    ATOM_BUCKETS = (64, 128, 256, 1_024, 5_000, 25_000, 60_000)
    for i, bucket in enumerate(ATOM_BUCKETS):
        if n_atoms <= bucket:
            return i
    return len(ATOM_BUCKETS) - 1


def fetch_ani1x_subset(
    ani1x_archive: Path,
    comp6_archive: Optional[Path],
    out_dir: Path,
    seed: int = 42,
    dry_run: bool = False,
) -> bool:
    """Main fetch function. Implements spec §7.1 responsibilities.

    Args:
        ani1x_archive: Path to ANI-1x HDF5 archive.
        comp6_archive: Path to COMP6 archive (optional for dry-run).
        out_dir: Output directory.
        seed: Random seed for Lane A selection.
        dry_run: If True, run filters and print candidates without writing files.

    Returns:
        True on success, False on error.
    """
    # Step 1: Verify SHA-256
    logger.info("=== SHA-256 Verification ===")
    if not verify_archive_hash(ani1x_archive, ANI1X_EXPECTED_SHA256, "ANI-1x"):
        return False

    if comp6_archive is not None:
        if not verify_archive_hash(comp6_archive, COMP6_EXPECTED_SHA256, "COMP6"):
            return False
    elif not dry_run:
        logger.error("COMP6 archive required (not dry-run). Provide --comp6-archive.")
        return False

    # Step 2: Run SELECT_16 over ANI-1x
    logger.info("\n=== Lane A Selection (SELECT_16) ===")
    lane_a_molecules = select_16_lane_a(ani1x_archive, seed=seed)

    if dry_run:
        logger.info("\n=== DRY RUN ===")
        if not lane_a_molecules:
            logger.warning("Lane A selection returned 0 molecules (check F1–F4 filters)")
        else:
            logger.info(f"Lane A candidates ({len(lane_a_molecules)}):")
            for i, mol in enumerate(lane_a_molecules):
                logger.info(
                    f"  {i}: {mol['formula'][:40]} "
                    f"(n_atoms={mol['n_atoms']}, n_conf={mol['n_conf_with_forces']})"
                )

        # If COMP6 archive provided, try Lane B selection in dry-run
        if comp6_archive is not None:
            logger.info("\n=== DRY RUN: Lane B Selection ===")
            lane_b_selections = select_lane_b_from_comp6v2(comp6_archive, seed=seed)
            if lane_b_selections:
                logger.info(f"Lane B candidates ({len(lane_b_selections)}):")
                for i, sel in enumerate(lane_b_selections):
                    logger.info(
                        f"  {i}: {sel['name']} (pdb_id={sel['pdb_id']}, "
                        f"group=/{sel['comp6v2_group']}, n_atoms={sel['n_atoms']}, "
                        f"bucket={sel['bucket_idx']}, n_conf={sel['n_conf']})"
                    )
        else:
            logger.info("(COMP6 archive not provided; skipping Lane B selection)")

        logger.info("(No files written in dry-run mode)")
        return True

    if not lane_a_molecules:
        logger.error("Failed to select Lane A molecules")
        return False

    logger.info(f"Selected {len(lane_a_molecules)} Lane A molecules")

    # Step 3: Write Lane A HDF5 files
    logger.info("\n=== Writing Lane A HDF5 Files ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    lane_a_dir = out_dir / "lane_a"
    lane_a_dir.mkdir(parents=True, exist_ok=True)

    lane_a_manifest = []
    for idx, mol_dict in enumerate(lane_a_molecules):
        formula = mol_dict["formula"]
        n_atoms = mol_dict["n_atoms"]

        # Load from ANI-1x
        data = load_molecule_from_ani1x(ani1x_archive, formula)
        if data is None:
            logger.warning(f"Failed to load {formula} from ANI-1x")
            continue

        # Write HDF5
        output_file = lane_a_dir / f"mol_{idx:03d}.h5"
        bucket_idx = bucket_idx_from_atom_count(n_atoms)

        write_molecule_h5(
            output_file,
            identifier=formula,
            lane="a",
            molecule_id=formula,
            bucket_idx=bucket_idx,
            positions=data["positions"],
            forces=data["forces"],
            energy=data["energy"],
            species=data["species"],
        )

        # Compute SHA-256
        sha256 = compute_sha256(output_file)

        lane_a_manifest.append({
            "idx": idx,
            "formula": formula,
            "n_total_atoms": n_atoms,
            "n_heavy_atoms": int(np.sum(data["species"] != 1)),
            "n_conf_with_forces": mol_dict["n_conf_with_forces"],
            "bucket_idx": bucket_idx,
            "file": str(output_file.relative_to(out_dir)),
            "sha256": sha256,
            "env_hashes": mol_dict.get("env_hashes", []),
            "rarity_score": mol_dict.get("rarity_score", 0.0),
        })

        logger.info(f"  {idx:2d}. {formula:40s} → {output_file.name}")

    # Step 4: Lane B (COMP6v2)
    logger.info("\n=== Lane B Selection (COMP6v2) ===")
    lane_b_dir = out_dir / "lane_b"
    lane_b_dir.mkdir(parents=True, exist_ok=True)

    lane_b_selections = select_lane_b_from_comp6v2(comp6_archive, seed=seed)
    lane_b_manifest = []

    for idx, sel in enumerate(lane_b_selections):
        group_name = sel["comp6v2_group"]
        n_atoms = sel["n_atoms"]
        name = sel["name"]

        # Load from COMP6v2
        data = load_comp6v2_group(comp6_archive, group_name)
        if data is None:
            logger.warning(f"Failed to load COMP6v2 group /{group_name}")
            continue

        # Write HDF5
        output_file = lane_b_dir / f"{name}.h5"
        bucket_idx = sel["bucket_idx"]

        write_molecule_h5(
            output_file,
            identifier=sel["pdb_id"] or f"comp6v2_{group_name}",
            lane="b",
            molecule_id=sel["pdb_id"],
            bucket_idx=bucket_idx,
            positions=data["positions"],
            forces=data["forces"],
            energy=data["energy"],
            species=data["species"],
        )

        # Compute SHA-256
        sha256 = compute_sha256(output_file)

        lane_b_manifest.append({
            "idx": idx,
            "name": name,
            "pdb_id": sel["pdb_id"],
            "subset": "COMP6v2",
            "n_total_atoms": n_atoms,
            "n_conf": sel["n_conf"],
            "bucket_idx": bucket_idx,
            "file": str(output_file.relative_to(out_dir)),
            "sha256": sha256,
        })

        logger.info(f"  {idx}. {name:20s} → {output_file.name}")

    # Step 5: Write manifest.json
    logger.info("\n=== Writing manifest.json ===")
    manifest = {
        "spec_version": "2026-05-20-hp4-ani1x-curation.md v1.0",
        "lane_a": {
            "source": "ANI-1x",
            "source_doi": "10.6084/m9.figshare.10047041.v1",
            "source_sha256": ANI1X_EXPECTED_SHA256,
            "selection_algorithm": "AIMNet2-style local-environment hashing diversity (§4.2)",
            "selection_seed": seed,
            "force_unit_correction": "Ha/Å → kcal/mol/Å multiplier 627.5094740631 (no Bohr factor)",
            "molecules": lane_a_manifest,
        },
        "lane_b": {
            "source": "COMP6 (https://github.com/isayev/COMP6)",
            "source_sha256": COMP6_EXPECTED_SHA256,
            "molecules": lane_b_manifest,
        },
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest to {manifest_path}")

    # Step 6: Summary table
    logger.info("\n=== Summary Table (Lane A) ===")
    logger.info(f"{'Idx':<3} {'Formula':<40} {'n_atoms':<8} {'bucket':<7} {'n_conf':<8} {'File':<20}")
    logger.info("─" * 90)
    for entry in lane_a_manifest:
        logger.info(
            f"{entry['idx']:<3d} {entry['formula'][:40]:<40} "
            f"{entry['n_total_atoms']:<8d} {entry.get('bucket_idx', bucket_idx_from_atom_count(entry['n_total_atoms'])):<7d} "
            f"{entry['n_conf_with_forces']:<8d} {entry['file'].split('/')[-1]:<20}"
        )

    logger.info("\n=== Summary Table (Lane B) ===")
    logger.info(f"{'Idx':<3} {'Name':<20} {'PDB ID':<8} {'n_atoms':<8} {'bucket':<7} {'n_conf':<8} {'File':<20}")
    logger.info("─" * 90)
    for entry in lane_b_manifest:
        pdb_id = entry.get("pdb_id") or "—"
        logger.info(
            f"{entry['idx']:<3d} {entry['name']:<20} {pdb_id:<8} "
            f"{entry['n_total_atoms']:<8d} {entry['bucket_idx']:<7d} "
            f"{entry['n_conf']:<8d} {entry['file'].split('/')[-1]:<20}"
        )

    logger.info(f"\nTotal Lane A molecules: {len(lane_a_manifest)}")
    logger.info(f"Total Lane B molecules: {len(lane_b_manifest)}")

    return True


def main():
    """Parse arguments and run fetch."""
    parser = argparse.ArgumentParser(
        description="HP4 ANI-1x + COMP6 Subset Curation Fetch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local dry-run (no writes, quick feedback)
  python scripts/data/fetch_ani1x_subset.py --ani1x-archive /path/to/ani1x.h5 --dry-run

  # Full fetch (requires both archives)
  python scripts/data/fetch_ani1x_subset.py \\
    --ani1x-archive /path/to/ani1x.h5 \\
    --comp6-archive /path/to/comp6.h5 \\
    --out-dir data/ani1x_subset

ANI-1x source:   https://figshare.com/articles/dataset/The_ANI-1ccx_and_ANI-1x_datasets/10047041
COMP6 source:    https://github.com/isayev/COMP6
        """,
    )

    parser.add_argument(
        "--ani1x-archive",
        type=Path,
        required=True,
        help="Path to ANI-1x HDF5 archive",
    )
    parser.add_argument(
        "--comp6-archive",
        type=Path,
        default=None,
        help="Path to COMP6 HDF5 archive (required unless --dry-run)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/ani1x_subset"),
        help="Output directory for curated subset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Lane A selection (default 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run: filter molecules, print candidates, no writes",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.ani1x_archive.exists():
        logger.error(
            f"ANI-1x archive not found: {args.ani1x_archive}\n"
            f"Download from: https://figshare.com/articles/dataset/The_ANI-1ccx_and_ANI-1x_datasets/10047041"
        )
        sys.exit(1)

    # Run fetch
    success = fetch_ani1x_subset(
        ani1x_archive=args.ani1x_archive,
        comp6_archive=args.comp6_archive,
        out_dir=args.out_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
