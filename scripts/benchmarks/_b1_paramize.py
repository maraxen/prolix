"""B1 benchmark keystone: PDB -> MolecularBundle with REAL ff19SB parameters.

The B1 Claim-1 benchmark needs prolix to pay a genuine force-field
parametrization cost (fair vs OpenMM's ``createSystem``). ``MolecularBundle.from_pdb``
only infers connectivity and leaves force constants zero, so it is unusable here.

This module uses proxide's native residue-template pipeline
(``parse_structure`` with ``OutputSpec(parameterize_md=True, ...)``) — the same
path prolix's ``test_system_parity`` suite exercises — to produce a fully
parametrized ``Protein``, then maps it onto ``make_bundle_from_system`` to get a
``MolecularBundle`` with real bond/angle/dihedral/nonbonded parameters.

Usage (self-test):
    uv run python scripts/benchmarks/_b1_paramize.py data/pdb/2GB1.pdb
"""

from __future__ import annotations

import argparse
import logging
import os
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_FF = "protein.ff19SB.xml"


def _resolve_ff_path(ff_name: str) -> str:
    """Resolve a bundled proxide force-field XML by name, or accept a full path."""
    if os.path.isabs(ff_name) and os.path.exists(ff_name):
        return ff_name
    import proxide

    assets = os.path.join(os.path.dirname(proxide.__file__), "assets")
    for candidate in (os.path.join(assets, ff_name), os.path.join(assets, "amber", ff_name)):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Force field '{ff_name}' not found under proxide assets ({assets}). "
        "Pass an absolute path or a bundled name like 'protein.ff19SB.xml'."
    )


def paramize_pdb_to_bundle(
    pdb_path: str,
    ff_name: str = DEFAULT_FF,
    periodic: bool = False,
):
    """Load a protein PDB and build a MolecularBundle with real ff19SB parameters.

    Args:
        pdb_path: Path to the PDB file.
        ff_name: Bundled proxide FF XML name (or absolute path). Default ff19SB.
        periodic: If True, build a periodic bundle; else free (vacuum).

    Returns:
        MolecularBundle with populated bonded + nonbonded parameters.
    """
    import proxide
    from proxide import CoordFormat, OutputSpec, parse_structure

    from prolix.physics.system import make_bundle_from_system

    ff_path = _resolve_ff_path(ff_name)
    logger.info("Parametrizing %s with force field %s", pdb_path, ff_path)

    spec = OutputSpec(
        parameterize_md=True,
        force_field=ff_path,
        coord_format=CoordFormat.Full,
    )
    protein = parse_structure(pdb_path, spec)

    masses = np.asarray(proxide.assign_masses(list(protein.atom_names)), dtype=np.float64)

    def _arr(name):
        v = getattr(protein, name, None)
        return None if v is None else np.asarray(v)

    # Protein carries dihedral indices under ``proper_dihedrals``; the ``dihedrals``
    # attribute is None. dihedral_params / improper_params are multi-term (N, T, P);
    # make_bundle_from_system flattens them via _flatten_multi_term_torsions.
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

    boundary = "periodic" if periodic else "free"
    bundle = make_bundle_from_system(system, boundary_condition=boundary)
    logger.info(
        "Built MolecularBundle: real_atoms=%d padded=%d bonds=%d",
        int(system.positions.shape[0]),
        bundle.positions.shape[0],
        int(system.bonds.shape[0]) if system.bonds is not None else 0,
    )
    return bundle


def _self_test(pdb_path: str, ff_name: str) -> int:
    import jax.numpy as jnp

    from prolix.api import EnsemblePlan

    bundle = paramize_pdb_to_bundle(pdb_path, ff_name=ff_name)

    bp_nonzero = bool(jnp.any(bundle.bond_params != 0))
    ap_nonzero = bool(jnp.any(bundle.angle_params != 0))
    q_nonzero = bool(jnp.any(bundle.charges != 0))
    real_atoms = int(bundle.atom_mask.sum())
    logger.info(
        "params: bond=%s angle=%s charge=%s | real_atoms=%d",
        bp_nonzero, ap_nonzero, q_nonzero, real_atoms,
    )
    if not (bp_nonzero and ap_nonzero and q_nonzero):
        logger.error("FAIL: force-field parameters are not all populated (zero params).")
        return 1

    # Runnability: 2 steps through the ensemble path must not NaN.
    traj = EnsemblePlan.from_bundles([bundle]).run(n_steps=2, dt=0.5, kT=2.5, seed=0)
    final = traj[0] if isinstance(traj, list) else traj
    pos = np.asarray(final.positions if hasattr(final, "positions") else final)
    finite = bool(np.all(np.isfinite(pos)))
    logger.info("2-step EnsemblePlan run finite=%s", finite)
    if not finite:
        logger.error("FAIL: trajectory produced non-finite positions.")
        return 1
    logger.info("PASS: real params + finite 2-step run.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Parametrize a PDB into a MolecularBundle (self-test).")
    parser.add_argument("pdb", help="Path to PDB file")
    parser.add_argument("--ff", default=DEFAULT_FF, help="Force field XML name or path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return _self_test(args.pdb, args.ff)


if __name__ == "__main__":
    raise SystemExit(main())
