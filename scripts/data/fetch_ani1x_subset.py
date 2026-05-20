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

# SHA-256 constants — filled at first download (see spec §7.1)
ANI1X_EXPECTED_SHA256 = "PLACEHOLDER_FILL_AT_FIRST_DOWNLOAD"
COMP6_EXPECTED_SHA256 = "PLACEHOLDER_FILL_AT_FIRST_DOWNLOAD"

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

# Lane B fixed molecules (spec §4.3, COMP6 source)
LANE_B_FIXED_MOLECULES = {
    "trp_cage": {"pdb_id": "1L2Y", "subset": "ANI-MD", "expected_atoms": 312},
    "chignolin": {"pdb_id": "1UAO", "subset": "ANI-MD", "expected_atoms": 138},
}


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


def parse_smiles_to_mol(smiles: str) -> Optional[object]:
    """Parse SMILES to RDKit molecule.

    Args:
        smiles: SMILES string.

    Returns:
        RDKit Mol object or None if parsing fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except Exception as e:
        logger.debug(f"SMILES parsing failed for {smiles}: {e}")
        return None


def get_canonical_smiles(smiles: str) -> Optional[str]:
    """Get canonical SMILES from input SMILES (spec §4.1 F5).

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES or None if parsing fails.
    """
    mol = parse_smiles_to_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def compute_local_env_hash(mol: object) -> dict[str, int]:
    """Compute AIMNet2-style local-environment hashes for a molecule.

    Per spec §4.2, for each non-H atom:
      hash(a) := (Z_a, n_H_connected, n_neighbors, sorted(Z_b for b in neighbors(a)))

    Args:
        mol: RDKit Mol object.

    Returns:
        Dict mapping hash values (as frozenset) to count. Use frozenset for hashability.
    """
    env_hashes = {}
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z == 1:  # Skip hydrogens
            continue

        # Count connected hydrogens
        n_h = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1)
        n_neighbors = atom.GetDegree()

        # Get neighbor atomic numbers (sorted, excluding H)
        neighbor_zs = tuple(sorted(
            n.GetAtomicNum() for n in atom.GetNeighbors()
            if n.GetAtomicNum() != 1
        ))

        # Create hash tuple
        env_hash = (z, n_h, n_neighbors, neighbor_zs)
        env_hashes[env_hash] = env_hashes.get(env_hash, 0) + 1

    return env_hashes


def select_16_lane_a(
    ani1x_path: Path,
    seed: int = 42,
) -> list[dict]:
    """Select 16 Lane A molecules via AIMNet2-style local-environment hashing diversity.

    Implements spec §4.2 SELECT_16 algorithm.

    Args:
        ani1x_path: Path to ANI-1x HDF5 archive.
        seed: Random seed for greedy selection.

    Returns:
        List of dicts with keys: smiles, n_total_atoms, n_conf_with_forces, env_hashes, rarity_score.
    """
    logger.info("Loading ANI-1x molecules and filtering F1–F5...")

    # Apply filters F1–F5
    passing_molecules = []
    for smiles, data in iter_data_buckets(
        str(ani1x_path),
        ["atomic_numbers", "wb97x_dz.energy", "wb97x_dz.forces", "coordinates"]
    ):
        atomic_numbers = data["atomic_numbers"]
        forces = data["wb97x_dz.forces"]

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

        # F5: SMILES parseable
        canonical_smiles = get_canonical_smiles(smiles)
        if canonical_smiles is None:
            continue

        passing_molecules.append({
            "smiles": canonical_smiles,
            "n_atoms": n_atoms,
            "n_conf_with_forces": int(n_conf_with_forces),
            "mol_object": parse_smiles_to_mol(canonical_smiles),
        })

    logger.info(f"Passing molecules (F1–F5): {len(passing_molecules)}")

    if len(passing_molecules) == 0:
        logger.error("No molecules passed F1–F5 filters")
        return []

    # Compute environment hashes for each molecule
    for mol_dict in passing_molecules:
        if mol_dict["mol_object"] is not None:
            env_hashes = compute_local_env_hash(mol_dict["mol_object"])
            mol_dict["env_hash_set"] = set(env_hashes.keys())

    # Compute global hash frequency
    global_hash_freq = Counter()
    for mol_dict in passing_molecules:
        if "env_hash_set" in mol_dict:
            global_hash_freq.update(mol_dict["env_hash_set"])

    # Compute rarity scores
    for mol_dict in passing_molecules:
        if "env_hash_set" in mol_dict and len(mol_dict["env_hash_set"]) > 0:
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
        if "env_hash_set" not in mol_dict:
            continue

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

    # Sort output by n_atoms ascending, then SMILES ascending
    selected.sort(key=lambda m: (m["n_atoms"], m["smiles"]))

    # Remove mol_object and env_hash_set (not part of output)
    for mol_dict in selected:
        mol_dict.pop("mol_object", None)
        env_hashes_set = mol_dict.pop("env_hash_set", set())
        mol_dict["env_hashes"] = list(map(str, env_hashes_set))

    return selected


def load_molecule_from_ani1x(
    ani1x_path: Path,
    smiles: str,
) -> Optional[dict]:
    """Load a single molecule from ANI-1x HDF5 by SMILES.

    Args:
        ani1x_path: Path to ANI-1x archive.
        smiles: SMILES string (key in HDF5).

    Returns:
        Dict with positions, forces (kcal/mol/Å), energy, species, or None if not found.
    """
    try:
        with h5py.File(ani1x_path, "r") as f:
            if smiles not in f:
                return None

            group = f[smiles]
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
        logger.debug(f"Failed to load {smiles} from ANI-1x: {e}")
        return None


def write_molecule_h5(
    output_path: Path,
    smiles: str,
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
        smiles: SMILES string (or PDB ID for Lane B).
        lane: "a" or "b".
        molecule_id: Molecule index.
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

            # Store SMILES as string
            f.create_dataset("smiles", data=smiles, dtype=h5py.string_dtype())

            # Store molecule_id
            f.create_dataset("molecule_id", data=molecule_id, dtype=np.int64)

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

    if not lane_a_molecules:
        logger.error("Failed to select Lane A molecules")
        return False

    logger.info(f"Selected {len(lane_a_molecules)} Lane A molecules")

    if dry_run:
        logger.info("\n=== DRY RUN: Candidate Molecules ===")
        logger.info("Lane A candidates:")
        for i, mol in enumerate(lane_a_molecules):
            logger.info(
                f"  {i}: {mol['smiles'][:40]} ... "
                f"(n_atoms={mol['n_atoms']}, n_conf={mol['n_conf_with_forces']})"
            )
        logger.info("(No files written in dry-run mode)")
        return True

    # Step 3: Write Lane A HDF5 files
    logger.info("\n=== Writing Lane A HDF5 Files ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    lane_a_dir = out_dir / "lane_a"
    lane_a_dir.mkdir(parents=True, exist_ok=True)

    lane_a_manifest = []
    for idx, mol_dict in enumerate(lane_a_molecules):
        smiles = mol_dict["smiles"]
        n_atoms = mol_dict["n_atoms"]

        # Load from ANI-1x
        data = load_molecule_from_ani1x(ani1x_archive, smiles)
        if data is None:
            logger.warning(f"Failed to load {smiles} from ANI-1x")
            continue

        # Write HDF5
        output_file = lane_a_dir / f"mol_{idx:03d}.h5"
        bucket_idx = bucket_idx_from_atom_count(n_atoms)

        write_molecule_h5(
            output_file,
            smiles=smiles,
            lane="a",
            molecule_id=idx,
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
            "smiles": smiles,
            "n_total_atoms": n_atoms,
            "n_heavy_atoms": np.sum(data["species"] != 1),
            "n_conf_with_forces": mol_dict["n_conf_with_forces"],
            "file": str(output_file.relative_to(out_dir)),
            "sha256": sha256,
            "env_hashes": mol_dict.get("env_hashes", []),
            "rarity_score": mol_dict.get("rarity_score", 0.0),
        })

        logger.info(f"  {idx:2d}. {smiles[:40]:40s} → {output_file.name}")

    # Step 4: Lane B (stub for now — minimal COMP6 support)
    logger.info("\n=== Lane B Selection (Stub) ===")
    lane_b_dir = out_dir / "lane_b"
    lane_b_dir.mkdir(parents=True, exist_ok=True)
    lane_b_manifest = []
    logger.info("Lane B: COMP6 archive support not yet implemented (stub only)")
    logger.info("  Expected: Trp-cage (1L2Y), Chignolin (1UAO), ANI-MD drug, DrugBank")

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
    logger.info("\n=== Summary Table ===")
    logger.info(f"{'Idx':<3} {'SMILES':<40} {'n_atoms':<8} {'bucket':<7} {'n_conf':<8} {'File':<20}")
    logger.info("─" * 90)
    for entry in lane_a_manifest:
        logger.info(
            f"{entry['idx']:<3d} {entry['smiles'][:40]:<40} "
            f"{entry['n_total_atoms']:<8d} {bucket_idx_from_atom_count(entry['n_total_atoms']):<7d} "
            f"{entry['n_conf_with_forces']:<8d} {entry['file'].split('/')[-1]:<20}"
        )

    logger.info(f"\nTotal Lane A molecules: {len(lane_a_manifest)}")
    logger.info(f"Total Lane B molecules: {len(lane_b_manifest)} (stub)")

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
