"""B1 keystone: PDB -> MolecularBundle with real ff19SB parameters (XR-A2A3 / A2).

Uses proxide ``parse_structure(..., parameterize_md=True)`` then
``make_bundle_from_system(..., exclusion_spec=ExclusionSpec.from_protein(...))``.

Force-scale gate (C1): after build, median |grad| on real atoms must be < 1e3
kcal/mol/Å (exclusions prevent bonded-neighbor LJ clashes).
"""

from __future__ import annotations

import argparse
import logging
import os
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_FF = "protein.ff19SB.xml"
MEDIAN_GRAD_MAX = 1e3  # kcal/mol/Å — XR-A2A3 C1 lock


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

    Passes ``ExclusionSpec.from_protein`` so 1-2/1-3/1-4 exclusions populate
    ``excl_*`` on the bundle (XR-A2A3 C4).
    """
    import proxide
    from proxide import CoordFormat, OutputSpec, parse_structure

    from prolix.physics.neighbor_list import ExclusionSpec
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
    boundary = "periodic" if periodic else "free"
    bundle = make_bundle_from_system(
        system,
        boundary_condition=boundary,
        exclusion_spec=exclusion_spec,
    )
    logger.info(
        "Built MolecularBundle: real_atoms=%d padded=%d bonds=%d excl_pairs=%d",
        int(system.positions.shape[0]),
        bundle.positions.shape[0],
        int(system.bonds.shape[0]) if system.bonds is not None else 0,
        int(np.asarray(bundle.excl_mask).sum()),
    )
    return bundle


def bundle_force_stats(bundle) -> dict[str, float]:
    """Energy + median/max |grad| on real atoms (host)."""
    import jax
    import jax.numpy as jnp

    from prolix.api.bundle_md import active_positions, energy_fn_from_bundle

    energy_fn = energy_fn_from_bundle(bundle)
    pos = active_positions(bundle)
    e, g = jax.value_and_grad(energy_fn)(pos)
    norms = jnp.linalg.norm(g, axis=-1)
    mask = jnp.asarray(bundle.atom_mask[: pos.shape[0]])
    real_norms = norms[mask]
    return {
        "energy": float(e),
        "median_abs_grad": float(jnp.median(real_norms)),
        "max_abs_grad": float(jnp.max(real_norms)),
        "n_real": int(mask.sum()),
    }


def assert_force_scale_ok(bundle, *, max_median: float = MEDIAN_GRAD_MAX) -> dict[str, float]:
    """C1 gate: median |grad| < max_median and finite energy."""
    stats = bundle_force_stats(bundle)
    if not np.isfinite(stats["energy"]):
        raise AssertionError(f"non-finite energy: {stats}")
    if stats["median_abs_grad"] >= max_median:
        raise AssertionError(
            f"median |grad|={stats['median_abs_grad']:.3e} >= {max_median:.3e} "
            f"(exclusions likely missing); stats={stats}"
        )
    return stats


def _self_test(pdb_path: str, ff_name: str) -> int:
    import jax.numpy as jnp

    from prolix.api import EnsemblePlan

    bundle = paramize_pdb_to_bundle(pdb_path, ff_name=ff_name)

    bp_nonzero = bool(jnp.any(bundle.bond_params != 0))
    ap_nonzero = bool(jnp.any(bundle.angle_params != 0))
    q_nonzero = bool(jnp.any(bundle.charges != 0))
    excl_n = int(np.asarray(bundle.excl_mask).sum())
    real_atoms = int(bundle.atom_mask.sum())
    logger.info(
        "params: bond=%s angle=%s charge=%s excl_pairs=%d | real_atoms=%d",
        bp_nonzero, ap_nonzero, q_nonzero, excl_n, real_atoms,
    )
    if not (bp_nonzero and ap_nonzero and q_nonzero):
        logger.error("FAIL: force-field parameters are not all populated (zero params).")
        return 1
    if excl_n <= 0:
        logger.error("FAIL: no exclusion pairs on bundle (ExclusionSpec not wired).")
        return 1

    try:
        stats = assert_force_scale_ok(bundle)
    except AssertionError as e:
        logger.error("FAIL force-scale gate: %s", e)
        return 1
    logger.info("force-scale OK: %s", stats)

    traj = EnsemblePlan.from_bundles([bundle]).run(n_steps=100, dt=0.5, kT=0.596, seed=0)
    final = traj[0] if isinstance(traj, list) else traj
    pos = np.asarray(final.positions if hasattr(final, "positions") else final)
    finite = bool(np.all(np.isfinite(pos)))
    logger.info("100-step EnsemblePlan run finite=%s", finite)
    if not finite:
        logger.error("FAIL: trajectory produced non-finite positions.")
        return 1
    logger.info("PASS: real params + exclusions + force-scale + finite 100-step run.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parametrize a PDB into a MolecularBundle (XR-A2A3 keystone)."
    )
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
