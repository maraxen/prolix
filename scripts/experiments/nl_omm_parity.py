#!/usr/bin/env python3
"""NL vs vendored OpenMM 8.3.1 Reference gold (campaign nonbonded-omm-parity).

Emit (requires OpenMM extra)::

    uv run --extra openmm python scripts/experiments/nl_omm_parity.py \\
      --emit-oracle --probe water --out /tmp/emit.json

Parity vs committed gold (no OpenMM needed once gold exists)::

    uv run bth run python scripts/experiments/nl_omm_parity.py \\
      --tag nonbonded-omm-parity --campaign <id> \\
      --out outputs/nonbonded_omm_parity/nl.json -- --probe water
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parent
_ORACLE = _REPO / "data" / "oracles" / "openmm_8.3.1"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from nonbonded_omm_parity.lj_oracle import (  # noqa: E402
    compare_nl_two_particle,
    emit_two_particle_lj,
    missing_gold_row,
    result_row,
    sha256_file,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", choices=("water", "1vii", "all"), default="water")
    p.add_argument("--emit-oracle", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _gold_path(probe: str) -> Path:
    return _ORACLE / f"nl_{probe}.json"


def _emit_1vii_gold() -> dict:
    """Solvated 1VII, OpenMM Reference, NL cutoff + LJ switch, nonbonded-only."""
    try:
        import openmm
        from openmm import app, unit
    except ImportError as e:
        raise SystemExit(f"OpenMM required for --emit-oracle: {e}") from e

    import numpy as np
    from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME

    pdb_path = _REPO / "data" / "pdb" / "1VII.pdb"
    if not pdb_path.is_file():
        raise SystemExit(f"missing {pdb_path}")

    import proxide

    ff_path = Path(proxide.__file__).parent / "assets" / "protein.ff19SB.xml"
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(str(ff_path), "amber14/tip3p.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)
    modeller.addSolvent(ff, padding=0.8 * unit.nanometer, model="tip3p")

    cutoff = float(REGRESSION_EXPLICIT_PME["cutoff_angstrom"])
    alpha = float(REGRESSION_EXPLICIT_PME["pme_alpha_per_angstrom"])
    grid = int(REGRESSION_EXPLICIT_PME["pme_grid_points"])
    omm_system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=(cutoff / 10.0) * unit.nanometer,
        constraints=None,
    )
    switch_a = cutoff - 1.0
    for i in range(omm_system.getNumForces()):
        force = omm_system.getForce(i)
        if isinstance(force, openmm.NonbondedForce):
            force.setPMEParameters(alpha * 10.0, grid, grid, grid)
            force.setUseDispersionCorrection(False)
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance((switch_a / 10.0) * unit.nanometer)
            force.setForceGroup(0)
        else:
            force.setForceGroup(1)

    integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = openmm.Context(
        omm_system, integ, openmm.Platform.getPlatformByName("Reference")
    )
    ctx.setPositions(modeller.positions)
    e = (
        ctx.getState(getEnergy=True, groups={0})
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalories_per_mole)
    )
    f = (
        ctx.getState(getForces=True, groups={0})
        .getForces(asNumpy=True)
        .value_in_unit(unit.kilocalories_per_mole / unit.angstrom)
    )
    positions_nm = modeller.positions.value_in_unit(unit.nanometer)
    positions_A = np.array([[p[0] * 10, p[1] * 10, p[2] * 10] for p in positions_nm])
    box_vecs = modeller.topology.getPeriodicBoxVectors()
    box_A = [
        box_vecs[0][0].value_in_unit(unit.angstrom),
        box_vecs[1][1].value_in_unit(unit.angstrom),
        box_vecs[2][2].value_in_unit(unit.angstrom),
    ]
    charges, sigmas, epsilons = [], [], []
    omm_exceptions = []
    for i in range(omm_system.getNumForces()):
        force = omm_system.getForce(i)
        if isinstance(force, openmm.NonbondedForce):
            for j in range(omm_system.getNumParticles()):
                q, sig, eps = force.getParticleParameters(j)
                charges.append(q.value_in_unit(unit.elementary_charge))
                sigmas.append(sig.value_in_unit(unit.angstrom))
                epsilons.append(eps.value_in_unit(unit.kilocalories_per_mole))
            # Extract exception parameters (LJ and Coulomb modifications for bonded pairs)
            for exc_idx in range(force.getNumExceptions()):
                i_atom, j_atom, chargeProd, sigma_exc, epsilon_exc = force.getExceptionParameters(exc_idx)
                omm_exceptions.append([
                    int(i_atom),
                    int(j_atom),
                    float(chargeProd.value_in_unit(unit.elementary_charge**2)),
                    float(sigma_exc.value_in_unit(unit.angstrom)),
                    float(epsilon_exc.value_in_unit(unit.kilocalories_per_mole)),
                ])
    return {
        "probe": "1vii",
        "openmm_version": getattr(openmm, "__version__", "unknown"),
        "openmm_platform": "Reference",
        "nonbonded_method": "PME",
        "cutoff_angstrom": cutoff,
        "switch_distance_angstrom": switch_a,
        "pme_alpha_per_angstrom": alpha,
        "pme_grid_points": grid,
        "energy_kcal": float(e),
        "forces_kcal_mol_A": np.asarray(f, dtype=float).tolist(),
        "positions_A": positions_A.tolist(),
        "box_A": box_A,
        "charges": charges,
        "sigmas_A": sigmas,
        "epsilons_kcal": epsilons,
        "n_atoms": int(positions_A.shape[0]),
        "omm_exceptions": omm_exceptions,
        "compare_status": "gold_only",
    }


def _compare_probe(probe: str, gold: Path) -> dict:
    rec = json.loads(gold.read_text())
    if probe == "1vii" or rec.get("probe") == "1vii":
        # Gold is vendored for later EnsemblePlan/NL compare; this cut gates the
        # two-particle switch window only.
        return {
            "campaign_slug": "nonbonded-omm-parity",
            "probe": "1vii",
            "engine": "neighbor_list",
            "oracle_path": str(gold.relative_to(_REPO)),
            "oracle_sha256": sha256_file(gold),
            "openmm_version": rec.get("openmm_version", ""),
            "gate_pass": 0,
            "delta_e_kcal": 1e9,
            "force_rmse_kcal_mol_A": 1e9,
            "cutoff_angstrom": float(rec.get("cutoff_angstrom", 9.0)),
            "switch_distance_angstrom": float(rec.get("switch_distance_angstrom") or 0.0),
            "skin_angstrom": 1.0,
            "compare_status": rec.get("compare_status", "gold_only"),
        }
    delta_e, force_rmse = compare_nl_two_particle(rec)
    e_p = float(rec["energy_kcal"]) + delta_e
    return result_row(
        probe=rec.get("probe", probe),
        engine="neighbor_list",
        gold=gold,
        rec=rec,
        repo=_REPO,
        delta_e=delta_e,
        force_rmse=force_rmse,
        e_prolix=e_p,
    )


def main() -> int:
    args = _parse_args()
    probes = ("water", "1vii") if args.probe == "all" else (args.probe,)
    last: dict | None = None
    rc = 0
    if args.emit_oracle:
        for probe in probes:
            if probe == "water":
                rec = emit_two_particle_lj(
                    probe="water",
                    cutoff_angstrom=9.0,
                    switch_distance_angstrom=8.0,
                    nonbonded_method="CutoffPeriodic",
                )
            else:
                rec = _emit_1vii_gold()
            gold = _gold_path(probe)
            write_json(gold, rec)
            last = {
                "campaign_slug": "nonbonded-omm-parity",
                "probe": probe,
                "engine": "neighbor_list",
                "oracle_path": str(gold.relative_to(_REPO)),
                "oracle_sha256": sha256_file(gold),
                "openmm_version": rec["openmm_version"],
                "gate_pass": 0,
                "delta_e_kcal": 0.0,
                "force_rmse_kcal_mol_A": 0.0,
                "cutoff_angstrom": rec["cutoff_angstrom"],
                "switch_distance_angstrom": rec.get("switch_distance_angstrom") or 0.0,
                "skin_angstrom": 1.0,
                "emit_only": 1,
            }
        # After emit, compare water gold immediately so emit cannot claim pass.
        if "water" in probes:
            last = _compare_probe("water", _gold_path("water"))
            rc = 0 if last["gate_pass"] else 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(last, indent=2) + "\n")
        return rc

    gold = _gold_path(probes[0])
    if not gold.is_file():
        last = missing_gold_row(probes[0], "neighbor_list", gold)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(last, indent=2) + "\n")
        return 1
    last = _compare_probe(probes[0], gold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(last, indent=2) + "\n")
    return 0 if last["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
