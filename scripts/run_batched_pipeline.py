"""Benchmark entrypoint: PDB catalog, Amber FF parameterization, padded batches.

Used by ``scripts/benchmark_nlvsdense.py`` and GPU/parity test drivers under
``tests/``. Resolve ``protein.ff19SB.xml`` from an in-repo ``proxide/`` checkout,
a sibling ``../proxide`` tree, or the installed ``proxide`` package.
"""

from __future__ import annotations

import dataclasses as dc
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp

from prolix.padding import PaddedSystem, collate_batch, pad_protein, select_bucket

if TYPE_CHECKING:
  from proxide.core.containers import Protein

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PDB = _REPO_ROOT / "data" / "pdb"


def _resolve_ff_xml() -> str:
  candidates = [
    _REPO_ROOT / "proxide" / "src" / "proxide" / "assets" / "protein.ff19SB.xml",
    _REPO_ROOT.parent / "proxide" / "src" / "proxide" / "assets" / "protein.ff19SB.xml",
  ]
  try:
    import proxide

    candidates.append(Path(proxide.__file__).resolve().parent / "assets" / "protein.ff19SB.xml")
  except ImportError:
    pass
  omm = (
    _REPO_ROOT
    / "openmmforcefields"
    / "openmmforcefields"
    / "ffxml"
    / "amber"
    / "protein.ff19SB.xml"
  )
  candidates.append(omm)
  for p in candidates:
    if p.is_file():
      return str(p)
  raise FileNotFoundError(
    "Could not find protein.ff19SB.xml. Install proxide, clone proxide next to prolix, "
    "or vendor openmmforcefields under prolix/openmmforcefields/."
  )


def _dhfr_path() -> Path:
  """Prefer DHFR crystal structure when present; else a large in-repo protein proxy."""
  preferred = _PDB / "4m8j.pdb"
  if preferred.is_file():
    return preferred
  return _PDB / "1UBQ.pdb"


# Keys used by tests and benchmarks. ``1X2G`` is a legacy name: we ship ``1CRN.pdb`` instead.
SYSTEM_CATALOG: dict[str, Path] = {
  "1X2G": _PDB / "1CRN.pdb",
  "1Y57": _PDB / "1UBQ.pdb",
  "CHIGNOLIN": _PDB / "1VII.pdb",
  "1UAO": _PDB / "1UAO.pdb",
  "DHFR": _dhfr_path(),
}


def load_and_parameterize(pdb_path: str | Path) -> Protein:
  from proxide import CoordFormat, OutputSpec, parse_structure
  from proxide import assign_mbondi2_radii, assign_obc2_scaling_factors

  pdb_path = Path(pdb_path)
  if not pdb_path.is_file():
    raise FileNotFoundError(f"PDB not found: {pdb_path}")

  spec = OutputSpec(
    coord_format=CoordFormat.Full,
    parameterize_md=True,
    force_field=_resolve_ff_xml(),
    add_hydrogens=False,
    remove_solvent=True,
    remove_hetatm=True,
  )
  protein = parse_structure(str(pdb_path), spec)

  if protein.radii is None:
    _radii = assign_mbondi2_radii(list(protein.atom_names), protein.bonds)
    _scaled = assign_obc2_scaling_factors(list(protein.atom_names))
    protein = dc.replace(
      protein,
      radii=jnp.asarray(_radii),
      scaled_radii=jnp.asarray(_scaled),
    )
  return protein


def prepare_batches(
  proteins: list,
  names: list[str],
  n_replicas: int = 1,
) -> dict[int, tuple[PaddedSystem, list[str]]]:
  """Replicate systems, bucket by ``ATOM_BUCKETS`` padding, and collate each bucket."""
  if len(proteins) != len(names):
    raise ValueError("proteins and names must have the same length")

  expanded_p: list = []
  expanded_n: list[str] = []
  for p, n in zip(proteins, names, strict=True):
    for _ in range(n_replicas):
      expanded_p.append(p)
      expanded_n.append(n)

  groups: dict[int, list[tuple[object, str]]] = {}
  for p, n in zip(expanded_p, expanded_n, strict=True):
    pos = jnp.asarray(p.coordinates).reshape(-1, 3)
    n_atoms = int(pos.shape[0])
    bucket = select_bucket(n_atoms)
    groups.setdefault(bucket, []).append((p, n))

  out: dict[int, tuple[PaddedSystem, list[str]]] = {}
  for bucket_size, items in groups.items():
    max_bonds = int(1.2 * bucket_size)
    max_angles = int(2.2 * bucket_size)
    max_dihedrals = int(3.5 * bucket_size)
    max_impropers = int(0.5 * bucket_size)
    max_cmaps = int(0.3 * bucket_size)
    max_constraints = int(0.7 * bucket_size)

    padded_list: list[PaddedSystem] = []
    name_list: list[str] = []
    for p, n in items:
      padded_list.append(
        pad_protein(
          p,
          bucket_size,
          target_bonds=max_bonds,
          target_angles=max_angles,
          target_dihedrals=max_dihedrals,
          target_impropers=max_impropers,
          target_cmaps=max_cmaps,
          target_constraints=max_constraints,
        )
      )
      name_list.append(n)

    out[bucket_size] = (collate_batch(padded_list), name_list)
  return out
