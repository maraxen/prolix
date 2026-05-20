#!/usr/bin/env python3
"""Phase A: Geometry+MMFF94-typed bonded parameter initialization for §7.1.

Reads HP4 ANI-1x/COMP6v2 per-molecule HDF5 files and writes a sibling
``<name>.params_init.json`` with initial bonded parameters that Phase B's
differentiable loss will use as θ_init for the harmonic prior.

**Toolchain pivot from spec §10.2** (2026-05-20):
- Spec originally pinned openff-toolkit >= 0.16. PyPI only has the yanked
  0.18.0 release; conda-forge install via micromamba killed the local
  resolver (OOM on the dep graph).
- Pivoted to RDKit MMFF94 (already a project dep, pip-installable) for atom
  typing + bond perception, plus geometry-derived r₀ / θ₀.
- Force constants use approximate uniform defaults by bond/angle type — the
  §7.1 figure's load-bearing claim is "batched vmap converges faster than
  the looped baseline," not "initial-param accuracy." Gradient descent
  updates k/r₀/θ₀ from these starts during training (Phase C).

Schema written (per molecule):

    {
      "molecule_id": "C2H5N5O3" | "trp_cage" | ...,
      "lane": "a" | "b",
      "n_atoms": 15,
      "atom_bucket_idx": 0,
      "param_source": "geometry+MMFF94",
      "force_constants_default": {"bond": 400, "angle": 50},
      "atom_types": [{"idx": 0, "Z": 6, "mmff_type": 1}, ...],
      "bonds": [{"i": 0, "j": 1, "k": 400.0, "r0": 1.40, "type_pair": [1, 5]}, ...],
      "angles": [{"i":0,"j":1,"k":2,"k_theta": 50.0, "theta0_deg": 109.5,
                  "type_triple": [1, 1, 5]}, ...],
      "proper_torsions": [],  # uniform-zero for v0; gradient descent learns them
      "params_init_sha256": "..."
    }

Units: bond k in kcal/mol/Å²; r₀ in Å; angle k_theta in kcal/mol/rad²;
theta0 in degrees. Conventions match Phase B's loss formulation
(spec §6).

Usage:
    # Local L1 dry-run
    uv run python scripts/data/build_params_init.py \\
        --subset-dir data/ani1x_subset --dry-run

    # Full run
    uv run python scripts/data/build_params_init.py \\
        --subset-dir data/ani1x_subset

    # One lane only (e.g., debug Trp-cage)
    uv run python scripts/data/build_params_init.py \\
        --subset-dir data/ani1x_subset --lane b
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Uniform initial force constants (Phase B harmonic prior centers θ_init = these).
DEFAULT_K_BOND = 400.0          # kcal/mol/Å²
DEFAULT_K_ANGLE = 50.0          # kcal/mol/rad²

PARAM_SOURCE_TAG = "geometry+MMFF94"

# ATOM_BUCKETS must match src/prolix/types/bundles.py (HP4 finer ladder).
ATOM_BUCKETS = (64, 128, 256, 1024, 5000, 25000, 60000)


def bucket_idx_from_atom_count(n_atoms: int) -> int:
    """Smallest bucket index i such that ATOM_BUCKETS[i] >= n_atoms."""
    for i, threshold in enumerate(ATOM_BUCKETS):
        if n_atoms <= threshold:
            return i
    raise ValueError(f"n_atoms={n_atoms} exceeds largest ATOM_BUCKETS slot")


def perceive_mol_from_xyz(species: np.ndarray, positions: np.ndarray) -> Chem.Mol:
    """Build an RDKit Mol with perceived bonds from atomic numbers + coordinates.

    Uses rdDetermineBonds.DetermineBonds (Jensen xyz2mol algorithm).
    Falls back to DetermineConnectivity (bond perception only, no order assignment)
    if full bond-order perception fails — adequate for our purpose since we only
    need the bonded topology, not aromaticity or formal charges.
    """
    xyz_lines = [f"{len(species)}", ""]
    for z, pos in zip(species, positions, strict=True):
        symbol = Chem.GetPeriodicTable().GetElementSymbol(int(z))
        xyz_lines.append(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    xyz_block = "\n".join(xyz_lines)

    raw_mol = Chem.MolFromXYZBlock(xyz_block)
    if raw_mol is None:
        raise RuntimeError("MolFromXYZBlock returned None")

    mol = Chem.RWMol(raw_mol)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except (ValueError, RuntimeError) as e:
        log.warning("DetermineBonds(charge=0) failed (%s); falling back to DetermineConnectivity", e)
        rdDetermineBonds.DetermineConnectivity(mol)

    # Initialize RingInfo + valence cache so downstream calls
    # (MMFF typing, atom neighbor iteration) don't trip RingInfo pre-condition.
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)
    return mol


def mmff_atom_types(mol: Chem.Mol) -> list[int] | None:
    """Return MMFF94 atom types per atom, or None if MMFF doesn't recognize the molecule.

    MMFF94 supports H, C, N, O, F, P, S, Cl, Br, I plus a few specialty types.
    Trp-cage etc. should be covered. If unsupported, returns None and callers
    fall back to the atomic-number-only type pair.
    """
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    n = mol.GetNumAtoms()
    types = [int(props.GetMMFFAtomType(i)) for i in range(n)]
    # MMFFGetMoleculeProperties returns valid props even when some types are 0
    # (unrecognized atoms). Treat 0 as "unknown" but don't fail outright.
    return types


def find_angles(mol: Chem.Mol) -> list[tuple[int, int, int]]:
    """All bonded i-j-k triplets where j is the central atom."""
    triplets = []
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        for a, b in itertools.combinations(neighbors, 2):
            triplets.append((a, j, b))
    return triplets


def find_proper_torsions(mol: Chem.Mol) -> list[tuple[int, int, int, int]]:
    """All proper i-j-k-l torsions on bonded paths."""
    out = []
    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        i_candidates = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
        l_candidates = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
        for i in i_candidates:
            for l_idx in l_candidates:
                if i == l_idx:
                    continue
                out.append((i, j, k, l_idx))
    return out


def vec_angle_deg(p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray) -> float:
    """Angle i-j-k in degrees (j is the vertex)."""
    v1 = p_i - p_j
    v2 = p_k - p_j
    cos_theta = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def build_params_dict(
    molecule_id: str,
    lane: str,
    species: np.ndarray,
    positions: np.ndarray,
) -> dict:
    """Compute bonded params for one molecule. Pure function — no I/O."""
    mol = perceive_mol_from_xyz(species, positions)
    n = mol.GetNumAtoms()
    if n != positions.shape[0]:
        raise RuntimeError(f"Atom count mismatch: mol={n} positions={positions.shape[0]}")

    mmff_types = mmff_atom_types(mol)
    atom_types = [
        {
            "idx": i,
            "Z": int(species[i]),
            "mmff_type": int(mmff_types[i]) if mmff_types else None,
        }
        for i in range(n)
    ]

    # Bonds
    bonds_out = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        r0 = float(np.linalg.norm(positions[i] - positions[j]))
        type_pair = sorted([atom_types[i]["mmff_type"] or atom_types[i]["Z"],
                            atom_types[j]["mmff_type"] or atom_types[j]["Z"]])
        bonds_out.append(
            {"i": i, "j": j, "k": DEFAULT_K_BOND, "r0": r0, "type_pair": type_pair},
        )

    # Angles
    angles_out = []
    for i, j, k in find_angles(mol):
        theta0 = vec_angle_deg(positions[i], positions[j], positions[k])
        type_triple = [
            atom_types[i]["mmff_type"] or atom_types[i]["Z"],
            atom_types[j]["mmff_type"] or atom_types[j]["Z"],
            atom_types[k]["mmff_type"] or atom_types[k]["Z"],
        ]
        angles_out.append(
            {
                "i": i, "j": j, "k": k,
                "k_theta": DEFAULT_K_ANGLE,
                "theta0_deg": theta0,
                "type_triple": type_triple,
            },
        )

    # Proper torsions — uniform zero for v0; Phase C gradient descent learns them.
    proper_torsions = find_proper_torsions(mol)

    params = {
        "molecule_id": molecule_id,
        "lane": lane,
        "n_atoms": n,
        "atom_bucket_idx": bucket_idx_from_atom_count(n),
        "param_source": PARAM_SOURCE_TAG,
        "force_constants_default": {
            "bond_kcal_per_mol_per_A2": DEFAULT_K_BOND,
            "angle_kcal_per_mol_per_rad2": DEFAULT_K_ANGLE,
        },
        "n_bonds": len(bonds_out),
        "n_angles": len(angles_out),
        "n_proper_torsions": len(proper_torsions),
        "atom_types": atom_types,
        "bonds": bonds_out,
        "angles": angles_out,
        "proper_torsions": [
            {"i": i, "j": j, "k": k, "l": l_idx, "periodicity": [], "phase_deg": [], "k_phi": []}
            for (i, j, k, l_idx) in proper_torsions
        ],
        "improper_torsions": [],
    }

    # Reproducibility hash on the bonded params alone.
    hash_payload = json.dumps(
        {
            "bonds": [(b["i"], b["j"], round(b["r0"], 6)) for b in bonds_out],
            "angles": [(a["i"], a["j"], a["k"], round(a["theta0_deg"], 4)) for a in angles_out],
            "torsions": [(t["i"], t["j"], t["k"], t["l"]) for t in params["proper_torsions"]],
        },
        sort_keys=True,
    ).encode()
    params["params_init_sha256"] = hashlib.sha256(hash_payload).hexdigest()
    return params


def process_one(h5_path: Path, dry_run: bool, fail_fast: bool) -> tuple[bool, str]:
    """Process one per-molecule HDF5 file. Returns (success, message)."""
    try:
        with h5py.File(h5_path) as f:
            positions = np.asarray(f["positions"][0], dtype=np.float32)
            species = np.asarray(f["species"][:], dtype=np.int32)
            # Lane A: 'formula' dataset (e.g., "C2H5N5O3").
            # Lane B: 'molecule_id' dataset (e.g., "trp_cage").
            if "formula" in f:
                molecule_id = f["formula"][()].decode() if hasattr(f["formula"][()], "decode") else str(f["formula"][()])
                lane = "a"
            elif "molecule_id" in f:
                molecule_id = f["molecule_id"][()].decode() if hasattr(f["molecule_id"][()], "decode") else str(f["molecule_id"][()])
                lane = "b"
            else:
                molecule_id = h5_path.stem
                lane = h5_path.parent.name.replace("lane_", "")
    except (OSError, KeyError) as e:
        msg = f"Failed to read {h5_path.name}: {e}"
        log.error(msg)
        if fail_fast:
            raise
        return False, msg

    try:
        params = build_params_dict(molecule_id, lane, species, positions)
    except (RuntimeError, ValueError) as e:
        msg = f"{h5_path.name}: parameterization failed ({e})"
        log.error(msg)
        if fail_fast:
            raise
        return False, msg

    log.info(
        "%s: id=%s n_atoms=%d bonds=%d angles=%d torsions=%d bucket=%d",
        h5_path.name, molecule_id, params["n_atoms"],
        params["n_bonds"], params["n_angles"], params["n_proper_torsions"],
        params["atom_bucket_idx"],
    )

    if not dry_run:
        out_path = h5_path.with_suffix(".params_init.json")
        out_path.write_text(json.dumps(params, indent=2))
        log.info("  → wrote %s (%d bytes)", out_path.name, out_path.stat().st_size)
    return True, ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset-dir", type=Path, default=Path("data/ani1x_subset"),
                   help="HP4 curated subset root (contains lane_a/ and lane_b/)")
    p.add_argument("--lane", choices=("a", "b", "both"), default="both")
    p.add_argument("--dry-run", action="store_true", help="No file writes; report only")
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop on first parameterization failure (default: log + continue)")
    args = p.parse_args()

    if not args.subset_dir.exists():
        log.error("Subset dir not found: %s. Run scripts/data/fetch_ani1x_subset.py first.",
                  args.subset_dir)
        return 1

    lanes = ("a", "b") if args.lane == "both" else (args.lane,)
    files = []
    for ln in lanes:
        ldir = args.subset_dir / f"lane_{ln}"
        if not ldir.exists():
            log.warning("Lane dir missing: %s (skip)", ldir)
            continue
        files.extend(sorted(ldir.glob("*.h5")))

    log.info("Processing %d per-molecule HDF5 files (dry_run=%s)", len(files), args.dry_run)

    n_ok, failures = 0, []
    for h5_path in files:
        ok, msg = process_one(h5_path, args.dry_run, args.fail_fast)
        if ok:
            n_ok += 1
        else:
            failures.append((h5_path.name, msg))

    log.info("=== Summary === ok=%d/%d", n_ok, len(files))
    for name, msg in failures:
        log.error("FAIL %s: %s", name, msg)

    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
