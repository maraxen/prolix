#!/usr/bin/env python3
"""
Phase A: OpenFF bonded parameter initialization toolchain for §7.1.

Reads HP4 ANI-1x/COMP6 per-molecule HDF5 files and writes params_init.json
with initial bonded parameters (bond k/r0, angle k/θ0, torsions).

Spec: docs/superpowers/specs/2026-05-20-hp4-ani1x-curation.md §10.2

Usage:
  uv run python scripts/data/build_params_init.py --subset-dir data/ani1x_subset --dry-run
  uv run python scripts/data/build_params_init.py --subset-dir data/ani1x_subset --lane a
  uv run python scripts/data/build_params_init.py --subset-dir data/ani1x_subset --lane b
"""

import argparse
import json
import logging
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BondType

# OpenFF imports
HAS_OPENFF = False
try:
    from openff.toolkit import Molecule as OFFMolecule
    from openff.toolkit import ForceField
    HAS_OPENFF = True
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# OpenFF force field version pinned by spec §10.2
OPENFF_FF_NAME = "openff_unconstrained-2.2.0"
OPENFF_FF_PATH = f"{OPENFF_FF_NAME}.offxml"

# Unit conversion: Ha/Å to kcal/mol/Å (spec §3, §8)
HA_TO_KCAL_PER_MOL = 627.5094740631


@dataclass
class AtomType:
    """Atom typing information from OpenFF."""

    idx: int
    Z: int
    smirks_type: str
    gaff_type: str | None = None


@dataclass
class BondParam:
    """Bond parameter entry."""

    i: int
    j: int
    k: float  # kcal/mol/Å²
    r0: float  # Å
    smirks_pattern: str


@dataclass
class AngleParam:
    """Angle parameter entry."""

    i: int
    j: int
    k: int
    k_theta: float  # kcal/mol/rad²
    theta0_deg: float  # degrees
    smirks_pattern: str


@dataclass
class ProperTorsionParam:
    """Proper dihedral parameter entry."""

    i: int
    j: int
    k: int
    l: int
    periodicity: list[int]
    phase_deg: list[float]
    k_phi: list[float]
    smirks_pattern: str


@dataclass
class ImproperTorsionParam:
    """Improper dihedral parameter entry."""

    i: int
    j: int
    k: int
    l: int
    periodicity: list[int]
    phase_deg: list[float]
    k_phi: list[float]
    smirks_pattern: str


def xyz2mol_perceive(
    positions: np.ndarray,
    species: np.ndarray,
    charge: int = 0,
) -> Chem.Mol:
    """
    Build RDKit Mol from xyz coordinates using bond perception.

    Args:
        positions: [N_atoms, 3] coordinates in Ångströms
        species: [N_atoms] atomic numbers
        charge: molecular charge (default 0)

    Returns:
        RDKit Mol with perceived bonds
    """
    mol = Chem.RWMol()

    # Add atoms with correct atomic numbers
    for z in species:
        atom = Chem.Atom(int(z))
        mol.AddAtom(atom)

    # Compute distance matrix
    distances = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
        axis=2,
    )

    # Perceive bonds using RDKit's bond-determination heuristics
    # This uses covalent radii + distance thresholds
    AllChem.EmbedMolecule(mol, useRandomCoords=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    # Use OpenBabel-style bond detection (RDKit's DetermineBonds)
    from rdkit.Chem import rdDetermineBonds

    rdDetermineBonds.DetermineBonds(mol, charge=charge)

    return mol.GetMol()


def get_openff_params_from_molecule(off_mol: "OFFMolecule") -> dict[str, Any]:
    """
    Extract bonded parameters from OpenFF Molecule via SMIRNOFF parametrization.

    Args:
        off_mol: OpenFF Molecule with SMILES/structure

    Returns:
        Dictionary with atom_types, bonds, angles, proper_torsions, improper_torsions
    """
    logger.debug(f"Loading OpenFF force field: {OPENFF_FF_NAME}")
    try:
        ff = ForceField(OPENFF_FF_PATH)
    except Exception as e:
        logger.error(f"Failed to load OpenFF FF {OPENFF_FF_NAME}: {e}")
        raise

    logger.debug(f"Assigning parameters to molecule with {off_mol.n_atoms} atoms")

    # Get labeled molecule (with parameter assignments)
    try:
        labels = ff.label_molecules(off_mol)[0]
    except Exception as e:
        logger.error(f"Failed to label molecule: {e}")
        raise

    # Extract atom types
    atom_types = []
    for atom_idx, atom_label in enumerate(labels["Atoms"]):
        atom_type_smirks = atom_label.atom_type
        atom_types.append(
            AtomType(
                idx=atom_idx,
                Z=off_mol.atoms[atom_idx].atomic_number,
                smirks_type=atom_type_smirks,
            )
        )

    # Extract bonds
    bonds = []
    for bond_label in labels.get("Bonds", []):
        atom_indices = bond_label.atom_indices
        bond_param = bond_label.parameter_type
        # Parameters are in OpenFF's native units (kcal/mol/Å² and Å)
        bonds.append(
            BondParam(
                i=atom_indices[0],
                j=atom_indices[1],
                k=float(bond_param.k.to("kcal/mol/angstrom**2").magnitude),
                r0=float(bond_param.length.to("angstrom").magnitude),
                smirks_pattern=bond_label.smirks,
            )
        )

    # Extract angles
    angles = []
    for angle_label in labels.get("Angles", []):
        atom_indices = angle_label.atom_indices
        angle_param = angle_label.parameter_type
        angles.append(
            AngleParam(
                i=atom_indices[0],
                j=atom_indices[1],
                k=atom_indices[2],
                k_theta=float(angle_param.k.to("kcal/mol/radian**2").magnitude),
                theta0_deg=float(angle_param.angle.to("degree").magnitude),
                smirks_pattern=angle_label.smirks,
            )
        )

    # Extract proper torsions
    proper_torsions = []
    for torsion_label in labels.get("ProperTorsions", []):
        atom_indices = torsion_label.atom_indices
        torsion_param = torsion_label.parameter_type

        # Torsions can have multiple terms (periodicity)
        periodicity = []
        phase_deg = []
        k_phi = []

        for term in torsion_param.parameters:
            periodicity.append(int(term.periodicity))
            phase_deg.append(float(term.phase.to("degree").magnitude))
            k_phi.append(float(term.k.to("kcal/mol").magnitude))

        proper_torsions.append(
            ProperTorsionParam(
                i=atom_indices[0],
                j=atom_indices[1],
                k=atom_indices[2],
                l=atom_indices[3],
                periodicity=periodicity,
                phase_deg=phase_deg,
                k_phi=k_phi,
                smirks_pattern=torsion_label.smirks,
            )
        )

    # Extract improper torsions
    improper_torsions = []
    for torsion_label in labels.get("ImproperTorsions", []):
        atom_indices = torsion_label.atom_indices
        torsion_param = torsion_label.parameter_type

        periodicity = []
        phase_deg = []
        k_phi = []

        for term in torsion_param.parameters:
            periodicity.append(int(term.periodicity))
            phase_deg.append(float(term.phase.to("degree").magnitude))
            k_phi.append(float(term.k.to("kcal/mol").magnitude))

        improper_torsions.append(
            ImproperTorsionParam(
                i=atom_indices[0],
                j=atom_indices[1],
                k=atom_indices[2],
                l=atom_indices[3],
                periodicity=periodicity,
                phase_deg=phase_deg,
                k_phi=k_phi,
                smirks_pattern=torsion_label.smirks,
            )
        )

    return {
        "atom_types": atom_types,
        "bonds": bonds,
        "angles": angles,
        "proper_torsions": proper_torsions,
        "improper_torsions": improper_torsions,
    }


def load_molecule_from_hdf5(h5_path: Path) -> tuple[np.ndarray, np.ndarray, str, str]:
    """
    Load a single molecule from HDF5 file.

    Returns:
        (positions, species, formula, lane) where positions is [N_conf, N_atoms, 3]
    """
    with h5py.File(h5_path, "r") as f:
        positions = np.array(f["positions"])  # [N_conf, N_atoms, 3]
        species = np.array(f["species"])  # [N_atoms]
        formula = f["formula"][()].decode() if isinstance(f["formula"][()], bytes) else f["formula"][()]
        lane = f.attrs.get("lane", "a")

    return positions, species, formula, lane


def build_params_for_molecule(h5_path: Path, dry_run: bool = False) -> dict[str, Any] | None:
    """
    Build params_init.json for a single molecule.

    Lane A path: use RDKit xyz2mol perception
    Lane B path: use formula/PDB ID (if available)

    Returns:
        params dict or None if failed
    """
    logger.info(f"Processing: {h5_path.name}")

    try:
        positions, species, formula, lane = load_molecule_from_hdf5(h5_path)
    except Exception as e:
        logger.error(f"Failed to load HDF5 {h5_path}: {e}")
        return None

    first_conf = positions[0]  # Use first conformer only
    n_atoms = len(species)
    logger.debug(f"  Formula: {formula}, N_atoms: {n_atoms}, Lane: {lane}")

    # Try to build RDKit molecule from xyz + species
    logger.debug("  Attempting xyz2mol perception...")
    try:
        rdkit_mol = xyz2mol_perceive(first_conf, species, charge=0)
        logger.debug(f"  xyz2mol success: {rdkit_mol.GetNumAtoms()} atoms, {rdkit_mol.GetNumBonds()} bonds")
    except Exception as e:
        logger.error(f"  xyz2mol failed: {e}")
        return None

    # Convert RDKit → OpenFF
    logger.debug("  Converting RDKit → OpenFF...")
    try:
        off_mol = OFFMolecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)
        logger.debug(f"  OpenFF molecule created: {off_mol.n_atoms} atoms")
    except Exception as e:
        logger.error(f"  RDKit → OpenFF conversion failed: {e}")
        return None

    if dry_run:
        logger.info(f"  [DRY-RUN] Would parameterize {formula}")
        return None

    # Get OpenFF parameters
    logger.debug("  Extracting OpenFF parameters...")
    try:
        param_data = get_openff_params_from_molecule(off_mol)
    except Exception as e:
        logger.error(f"  OpenFF parameter extraction failed: {e}")
        return None

    # Build params_init.json structure
    atom_types_dict = [
        {
            "idx": at.idx,
            "Z": at.Z,
            "smirks_type": at.smirks_type,
        }
        for at in param_data["atom_types"]
    ]

    bonds_dict = [
        asdict(b) for b in param_data["bonds"]
    ]

    angles_dict = [
        asdict(a) for a in param_data["angles"]
    ]

    proper_torsions_dict = [
        asdict(p) for p in param_data["proper_torsions"]
    ]

    improper_torsions_dict = [
        asdict(im) for im in param_data["improper_torsions"]
    ]

    # Compute hash of bonded parameters for reproducibility
    hash_input = json.dumps(
        {
            "bonds": bonds_dict,
            "angles": angles_dict,
            "proper_torsions": proper_torsions_dict,
            "improper_torsions": improper_torsions_dict,
        },
        sort_keys=True,
    )
    params_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    params_dict = {
        "molecule_id": formula,
        "lane": lane,
        "n_atoms": n_atoms,
        "atom_bucket_idx": 0,  # Will be read from HDF5 attrs
        "force_field": OPENFF_FF_NAME,
        "openff_version": "0.16.0",  # Minimum pinned version
        "atom_types": atom_types_dict,
        "bonds": bonds_dict,
        "angles": angles_dict,
        "proper_torsions": proper_torsions_dict,
        "improper_torsions": improper_torsions_dict,
        "params_init_sha256": params_hash,
    }

    # Add bucket_idx from HDF5 attrs if present
    with h5py.File(h5_path, "r") as f:
        if "bucket_idx" in f.attrs:
            params_dict["atom_bucket_idx"] = int(f.attrs["bucket_idx"])

    logger.info(
        f"  Success: {len(bonds_dict)} bonds, {len(angles_dict)} angles, "
        f"{len(proper_torsions_dict)} proper torsions, {len(improper_torsions_dict)} improper"
    )

    return params_dict


def main():
    parser = argparse.ArgumentParser(
        description="Build OpenFF bonded parameter initialization files for HP4 molecules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=Path("data/ani1x_subset"),
        help="Path to ani1x_subset directory (default: data/ani1x_subset)",
    )
    parser.add_argument(
        "--lane",
        type=str,
        choices=["a", "b", "both"],
        default="both",
        help="Process only lane a, lane b, or both (default: both)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be parameterized without writing files",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first parameterization failure (default: log and continue)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Check for OpenFF toolkit dependency
    if not HAS_OPENFF:
        logger.error(
            "openff-toolkit is required but not installed. "
            "Install with: pip install 'openff-toolkit>=0.16' openmmforcefields"
        )
        return 1

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    subset_dir = args.subset_dir.resolve()
    if not subset_dir.exists():
        logger.error(f"Subset directory not found: {subset_dir}")
        return 1

    lanes_to_process = ["a", "b"] if args.lane == "both" else [args.lane]
    total_processed = 0
    total_failed = 0

    for lane in lanes_to_process:
        lane_dir = subset_dir / f"lane_{lane}"
        if not lane_dir.exists():
            logger.warning(f"Lane directory not found: {lane_dir}")
            continue

        logger.info(f"Processing lane {lane.upper()}...")
        h5_files = sorted(lane_dir.glob("*.h5"))
        logger.info(f"  Found {len(h5_files)} HDF5 files")

        for h5_path in h5_files:
            try:
                params = build_params_for_molecule(h5_path, dry_run=args.dry_run)
                if params is None:
                    total_failed += 1
                    if args.fail_fast:
                        logger.error("Stopping due to --fail-fast")
                        return 1
                    continue

                if not args.dry_run:
                    params_path = h5_path.with_suffix(".params_init.json")
                    with open(params_path, "w") as f:
                        json.dump(params, f, indent=2)
                    logger.info(f"  Wrote: {params_path.name}")

                total_processed += 1
            except Exception as e:
                logger.error(f"Unexpected error processing {h5_path.name}: {e}")
                total_failed += 1
                if args.fail_fast:
                    return 1

    logger.info(
        f"\nSummary: {total_processed} molecules processed successfully, "
        f"{total_failed} failed"
    )

    if total_failed > 0 and args.fail_fast:
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
