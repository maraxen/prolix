"""Load a protein PDB as a parameterised MolecularBundle (spec section 4, D5).

A copy of ``scripts/benchmarks/_b1_paramize.paramize_pdb_to_bundle``. A parity test (slow) keeps
the copy equal to the script. Always the free boundary (C10): the copy sets no box. Heavy imports
(proxide, jax, prolix.physics) stay inside the functions.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

DEFAULT_FF = "protein.ff19SB.xml"


def resolve_ff_path(ff_name: str) -> str:
  """Resolve a bundled proxide force-field XML by name, or accept an absolute path.

  Raises ``FileNotFoundError`` when neither exists (the ops layer maps it to ``InputError``).
  """
  if os.path.isabs(ff_name) and os.path.exists(ff_name):
    return ff_name
  import proxide

  assets = os.path.join(os.path.dirname(proxide.__file__), "assets")
  for candidate in (os.path.join(assets, ff_name), os.path.join(assets, "amber", ff_name)):
    if os.path.exists(candidate):
      return candidate
  raise FileNotFoundError(
    f"force field {ff_name!r} not found under proxide assets ({assets}); "
    "pass an absolute path or a bundled name such as 'protein.ff19SB.xml'"
  )


def count_pdb_atoms(pdb_path: str | os.PathLike[str]) -> tuple[int, int]:
  """Return ``(atom_records, hydrogen_records)`` counted from ATOM/HETATM lines.

  Element comes from columns 77-78 when present, else the first non-digit character of the
  atom name (columns 13-16).
  """
  n_atoms = 0
  n_h = 0
  with open(pdb_path, encoding="utf-8", errors="replace") as fh:
    for line in fh:
      if not line.startswith(("ATOM", "HETATM")):
        continue
      n_atoms += 1
      element = line[76:78].strip() if len(line.rstrip("\n")) >= 78 else ""
      if not element:
        element = line[12:16].strip().lstrip("0123456789")[:1]
      if element.upper() == "H":
        n_h += 1
  return n_atoms, n_h


def load_system_bundle(pdb_path: str, ff_name: str = DEFAULT_FF) -> Any:
  """Parameterise a PDB with proxide (ff19SB by default) and build a free-boundary bundle.

  Passes ``ExclusionSpec.from_protein`` so 1-2/1-3/1-4 exclusions populate the bundle.
  """
  import numpy as np
  import proxide
  from proxide import CoordFormat, OutputSpec, parse_structure

  from prolix.physics.neighbor_list import ExclusionSpec
  from prolix.physics.system import make_bundle_from_system

  ff_path = resolve_ff_path(ff_name)
  spec = OutputSpec(parameterize_md=True, force_field=ff_path, coord_format=CoordFormat.Full)
  protein = parse_structure(pdb_path, spec)

  masses = np.asarray(proxide.assign_masses(list(protein.atom_names)), dtype=np.float64)

  def _arr(name: str) -> Any:
    v = getattr(protein, name, None)
    return None if v is None else np.asarray(v)

  system = SimpleNamespace(
    positions=_arr("coordinates"),
    masses=masses,
    bonds=_arr("bonds"),
    bond_params=_arr("bond_params"),
    angles=_arr("angles"),
    angle_params=_arr("angle_params"),
    dihedrals=_arr("proper_dihedrals"),
    dihedral_params=_arr("dihedral_params"),
    impropers=_arr("impropers"),
    improper_params=_arr("improper_params"),
    charges=_arr("charges"),
    sigmas=_arr("sigmas"),
    epsilons=_arr("epsilons"),
  )
  exclusion_spec = ExclusionSpec.from_protein(protein)
  return make_bundle_from_system(system, boundary_condition="free", exclusion_spec=exclusion_spec)


__all__ = ["DEFAULT_FF", "count_pdb_atoms", "load_system_bundle", "resolve_ff_path"]
