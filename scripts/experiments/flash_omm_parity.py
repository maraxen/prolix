#!/usr/bin/env python3
"""Flash vs vendored OpenMM 8.3.1 Reference gold (campaign nonbonded-omm-parity).

Flash gold uses MIC-diameter cutoff (half cubic box), not the 9 Å NL cutoff.

Emit (requires OpenMM extra)::

    uv run --extra openmm python scripts/experiments/flash_omm_parity.py \\
      --emit-oracle --probe water --out /tmp/emit_flash.json
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
    BOX_A,
    compare_flash_two_particle,
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
    return _ORACLE / f"flash_{probe}.json"


def _compare_probe(probe: str, gold: Path) -> dict:
    rec = json.loads(gold.read_text())
    if probe == "1vii" or rec.get("probe") == "1vii":
        return {
            "campaign_slug": "nonbonded-omm-parity",
            "probe": "1vii",
            "engine": "flash",
            "oracle_path": str(gold.relative_to(_REPO)),
            "oracle_sha256": sha256_file(gold),
            "openmm_version": rec.get("openmm_version", ""),
            "gate_pass": 0,
            "delta_e_kcal": 1e9,
            "force_rmse_kcal_mol_A": 1e9,
            "cutoff_angstrom": float(rec.get("cutoff_angstrom", 10.0)),
            "switch_distance_angstrom": 0.0,
            "skin_angstrom": 1.0,
            "compare_status": "gold_only",
        }
    delta_e, force_rmse = compare_flash_two_particle(rec)
    e_p = float(rec["energy_kcal"]) + delta_e
    return result_row(
        probe=rec.get("probe", probe),
        engine="flash",
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
    mic = BOX_A / 2.0
    if args.emit_oracle:
        for probe in probes:
            if probe == "1vii":
                raise SystemExit(
                    "flash 1vii emit is not in this cut; emit --probe water "
                    "(MIC-diameter cutoff gold)"
                )
            rec = emit_two_particle_lj(
                probe="water",
                cutoff_angstrom=mic,
                switch_distance_angstrom=None,
                nonbonded_method="CutoffPeriodic",
            )
            gold = _gold_path(probe)
            write_json(gold, rec)
            last = {
                "campaign_slug": "nonbonded-omm-parity",
                "probe": probe,
                "engine": "flash",
                "oracle_path": str(gold.relative_to(_REPO)),
                "oracle_sha256": sha256_file(gold),
                "openmm_version": rec["openmm_version"],
                "gate_pass": 0,
                "delta_e_kcal": 0.0,
                "force_rmse_kcal_mol_A": 0.0,
                "cutoff_angstrom": rec["cutoff_angstrom"],
                "switch_distance_angstrom": 0.0,
                "skin_angstrom": 1.0,
                "emit_only": 1,
            }
        if "water" in probes:
            last = _compare_probe("water", _gold_path("water"))
            rc = 0 if last["gate_pass"] else 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(last, indent=2) + "\n")
        return rc

    gold = _gold_path(probes[0])
    if not gold.is_file():
        last = missing_gold_row(probes[0], "flash", gold)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(last, indent=2) + "\n")
        return 1
    last = _compare_probe(probes[0], gold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(last, indent=2) + "\n")
    return 0 if last["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
