#!/usr/bin/env python3
"""Fetch the canonical JAC/DHFR explicit-solvent benchmark fixture (#295).

Materializes the standard Joint Amber CHARMM (JAC) DHFR / TIP3P benchmark
system (23,558 atoms — protein + explicit TIP3P water + PME electrostatics)
via ``openmmtools.testsystems.DHFRExplicit``, which bundles the AMBER
prmtop/inpcrd for this system internally. No manual PDB fetch or solvation
is required — openmmtools has already done that work.

Outputs (both required for Phase 2's OpenMM comparator + prolix ingestion):
  - ``data/pdb/dhfr_jac_benchmark.pdb``            topology + positions
  - ``data/pdb/dhfr_jac_benchmark_system.xml``      serialized OpenMM System
    (preserves the exact PME/constraint/box configuration DHFRExplicit
    configures, so Phase 2 doesn't need to reconstruct it from scratch)

Import note: ``openmmtools/__init__.py`` eagerly imports its full multistate
replica-exchange sampler stack (``openmmtools.multistate``), which pulls in
``mpiplus`` (not on PyPI — conda-forge only) and, if worked around, further
transitively requires ``numba`` and ``pymbar`` (whose import emits a warning
that it will silently flip JAX to 64-bit mode on first use — a real footgun
in a JAX-based repo). None of that machinery is needed to read a pre-built
test system. This script therefore loads only the two submodules
``DHFRExplicit`` actually needs (``openmmtools.constants``,
``openmmtools.testsystems``) directly from the installed package's files,
without executing ``openmmtools/__init__.py``. See
``.praxia/docs/research/`` (task 260813_triage_next_sprint) for the full
investigation trail.

Usage:
  uv run python scripts/data_prep/fetch_dhfr_benchmark.py
  uv run python scripts/data_prep/fetch_dhfr_benchmark.py --force
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "pdb"
PDB_PATH = OUT_DIR / "dhfr_jac_benchmark.pdb"
SYSTEM_XML_PATH = OUT_DIR / "dhfr_jac_benchmark_system.xml"

EXPECTED_ATOM_COUNT = 23558

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_dhfr_explicit_cls():
  """Import ``DHFRExplicit`` without executing ``openmmtools/__init__.py``.

  See module docstring for why: the real package init drags in an MPI-based
  replica-exchange stack (``mpiplus``, ``numba``, ``pymbar``) that has
  nothing to do with reading a bundled prmtop/inpcrd test system, and one of
  those (``pymbar``) mutates global JAX precision state on import/use.
  """
  # find_spec (unlike `import openmmtools`) locates the installed package
  # without executing its __init__.py.
  located_spec = importlib.util.find_spec("openmmtools")
  if located_spec is None or located_spec.origin is None:
    msg = "openmmtools is not installed (expected as a project dependency)"
    raise ModuleNotFoundError(msg)
  pkg_dir = Path(located_spec.origin).resolve().parent
  init_path = pkg_dir / "__init__.py"

  # A real package spec (submodule_search_locations set) is required so that
  # openmmtools' internal ``importlib.resources.files("openmmtools")`` data
  # lookups (used to find data/dhfr/JAC.prmtop) resolve correctly — but we
  # deliberately never execute __init__.py's body.
  pkg_spec = importlib.util.spec_from_file_location(
      "openmmtools", init_path, submodule_search_locations=[str(pkg_dir)]
  )
  pkg_module = importlib.util.module_from_spec(pkg_spec)
  sys.modules["openmmtools"] = pkg_module

  def _load_submodule(name: str, relpath: str) -> types.ModuleType:
    sub_spec = importlib.util.spec_from_file_location(
        f"openmmtools.{name}", pkg_dir / relpath
    )
    mod = importlib.util.module_from_spec(sub_spec)
    sys.modules[f"openmmtools.{name}"] = mod
    sub_spec.loader.exec_module(mod)
    setattr(pkg_module, name, mod)
    return mod

  _load_submodule("constants", "constants.py")
  testsystems_mod = _load_submodule("testsystems", "testsystems.py")
  return testsystems_mod.DHFRExplicit


def fetch_dhfr_benchmark(force: bool = False) -> int:
  """Materialize the DHFR JAC benchmark fixture. Returns the atom count."""
  OUT_DIR.mkdir(parents=True, exist_ok=True)

  if PDB_PATH.exists() and SYSTEM_XML_PATH.exists() and not force:
    logger.info("Both output files already exist; skipping regeneration (use --force to redo).")
    logger.info("  %s", PDB_PATH)
    logger.info("  %s", SYSTEM_XML_PATH)
    import openmm
    from openmm import app

    with PDB_PATH.open() as fh:
      pdb = app.PDBFile(fh)
    atom_count = pdb.topology.getNumAtoms()
    logger.info("Atom count (from existing PDB): %d", atom_count)
    return atom_count

  DHFRExplicit = _load_dhfr_explicit_cls()

  logger.info("Instantiating openmmtools.testsystems.DHFRExplicit() (defaults: PME, HBonds, rigid water, 10.0 A cutoff)...")
  test_system = DHFRExplicit()

  import openmm
  from openmm import app

  atom_count = test_system.topology.getNumAtoms()
  logger.info("Atom count: %d", atom_count)

  logger.info("Writing PDB -> %s", PDB_PATH)
  with PDB_PATH.open("w") as fh:
    app.PDBFile.writeFile(test_system.topology, test_system.positions, fh)

  logger.info("Serializing OpenMM System -> %s", SYSTEM_XML_PATH)
  with SYSTEM_XML_PATH.open("w") as fh:
    fh.write(openmm.XmlSerializer.serialize(test_system.system))

  return atom_count


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--force", action="store_true", help="Regenerate outputs even if they already exist."
  )
  args = parser.parse_args()

  atom_count = fetch_dhfr_benchmark(force=args.force)

  if atom_count != EXPECTED_ATOM_COUNT:
    logger.error(
        "Atom count mismatch: got %d, expected %d. STOPPING — do not silently"
        " adjust downstream assumptions to match a different number.",
        atom_count,
        EXPECTED_ATOM_COUNT,
    )
    return 1

  logger.info("OK: atom count matches expected JAC/DHFR benchmark size (%d).", EXPECTED_ATOM_COUNT)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
