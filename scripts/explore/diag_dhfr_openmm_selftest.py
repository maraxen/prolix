"""Does OpenMM ALONE also blow up on the DHFR fixture? (task 260903_dhfr_divergence_parity)

Why this exists
---------------
T0 (``diag_dhfr_force_parity.py``, job 21967097) showed prolix's nonbonded forces match
OpenMM's to 0.065% on this exact configuration -- cosine 0.9999999999, magnitude ratio
1.00065, at 23558 atoms through the production NL+PME path. That exonerates the whole
nonbonded surface, which was the entire suspect list for #4953.

But it also surfaced something neither engine disputes: at step 0 this configuration has
an RMS force of ~4158 kcal/mol/A and a potential energy of ~+198,339 kcal/mol. An
equilibrated solvated protein should sit at a strongly NEGATIVE potential energy (order
-70,000 kcal/mol for ~7000 waters). Both engines agree on the number, so it is a property
of the configuration, not of either force implementation.

That inverts the question. Instead of "why does prolix explode on this fixture", ask
"does anything integrate this fixture stably?" This script answers it with the one
engine nobody suspects, running entirely on its own: OpenMM, its own System XML, its own
integrator, no prolix in the process at all.

Reading:
  OpenMM also blows up      -> the fixture's starting configuration is unusable as-is
                               (needs minimization/equilibration), and prolix has been
                               faithfully integrating a broken input. #4953 is not a
                               prolix force bug.
  OpenMM stays stable       -> the configuration is fine and the fault is in prolix's
                               integrator (settle.py), since its forces are now proven
                               correct. Either way this is decisive.

``bench_dhfr_parity.py::_run_openmm_comparator`` already runs this fixture for 5000 steps
and reports ns/day -- but it asserts nothing about the trajectory, so it would report a
throughput number for a diverging system without complaint. That is exactly the blind
spot this script fills.

OpenMM only: no JAX, no prolix import, CPU platform. Safe to run locally.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("diag_dhfr_openmm_selftest")

_ROOT = Path(__file__).resolve().parents[2]
DHFR_PDB = _ROOT / "data" / "pdb" / "dhfr_jac_benchmark.pdb"
DHFR_SYSTEM_XML = _ROOT / "data" / "pdb" / "dhfr_jac_benchmark_system.xml"


def _min_pair_distance(pos_ang: np.ndarray, box_edge: float, sample: int = 3000, seed: int = 0):
    """Minimum minimum-image separation over a random atom subset.

    Sampled rather than exhaustive: the full 23558^2 matrix is 4.4 GB, and the question
    here ("has anything collapsed onto anything") is answered just as well by a subset.
    Minimum-image is mandatory -- under PBC a raw difference reports a boundary crossing
    as a box-sized separation.
    """
    rng = np.random.default_rng(seed)
    n = pos_ang.shape[0]
    idx = rng.choice(n, size=min(sample, n), replace=False)
    p = pos_ang[idx]
    d = p[:, None, :] - p[None, :, :]
    d -= box_edge * np.round(d / box_edge)
    r = np.sqrt((d**2).sum(-1))
    np.fill_diagonal(r, np.inf)
    return float(r.min())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt-fs", type=float, default=1.0)
    ap.add_argument("--chunk-steps", type=int, default=20)
    ap.add_argument("--n-chunks", type=int, default=5)
    ap.add_argument("--platform", default="CPU")
    ap.add_argument(
        "--minimize",
        action="store_true",
        help="Energy-minimize before integrating. The discriminating comparison is "
        "running this script with and without it.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    import openmm
    from openmm import XmlSerializer, unit as u
    from openmm.app import PDBFile

    pdb = PDBFile(str(DHFR_PDB))
    system = XmlSerializer.deserialize(DHFR_SYSTEM_XML.read_text())

    integrator = openmm.LangevinMiddleIntegrator(
        300 * u.kelvin, 10.0 / u.picosecond, args.dt_fs * u.femtosecond
    )
    platform = openmm.Platform.getPlatformByName(args.platform)
    sim_ctx = openmm.Context(system, integrator, platform)
    sim_ctx.setPositions(pdb.positions)

    box = np.asarray(
        system.getDefaultPeriodicBoxVectors().value_in_unit(u.nanometer)
        if hasattr(system.getDefaultPeriodicBoxVectors(), "value_in_unit")
        else [[v.x, v.y, v.z] for v in system.getDefaultPeriodicBoxVectors()]
    )
    box_edge = float(box[0][0]) * 10.0  # nm -> Angstrom
    logger.info("box_edge=%.4f A  n_atoms=%d  minimize=%s", box_edge, system.getNumParticles(), args.minimize)

    if args.minimize:
        logger.info("minimizing ...")
        openmm.LocalEnergyMinimizer.minimize(sim_ctx, maxIterations=500)

    rows = []
    prev = None
    for c in range(args.n_chunks + 1):
        st = sim_ctx.getState(getPositions=True, getEnergy=True, getForces=True)
        pos = np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer)) * 10.0
        pe = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / 4.184
        ke = st.getKineticEnergy().value_in_unit(u.kilojoule_per_mole) / 4.184
        f = np.asarray(
            st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)
        ) / 41.84
        # 3N-Nconstraints degrees of freedom; T = 2*KE / (dof * kB)
        dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
        temp = 2.0 * ke / (dof * 0.0019872041)

        disp = None
        if prev is not None:
            d = pos - prev
            d -= box_edge * np.round(d / box_edge)  # minimum image, mandatory under PBC
            disp = float(np.mean(np.linalg.norm(d, axis=1))) / args.chunk_steps

        row = {
            "step": c * args.chunk_steps,
            "finite": bool(np.all(np.isfinite(pos))),
            "potential_kcal": float(pe),
            "temperature_K": float(temp),
            "force_rms": float(np.sqrt(np.mean(np.sum(f**2, axis=1)))),
            "min_dist_sampled": _min_pair_distance(pos, box_edge),
            "mean_disp_per_step": disp,
        }
        rows.append(row)
        logger.info(
            "step %5d  PE=%12.1f  T=%9.1f K  Frms=%9.2f  min_dist=%.4f  disp/step=%s",
            row["step"], row["potential_kcal"], row["temperature_K"],
            row["force_rms"], row["min_dist_sampled"],
            "n/a" if disp is None else f"{disp:.4f}",
        )
        prev = pos
        if c < args.n_chunks:
            integrator.step(args.chunk_steps)

    report = {
        "engine": "openmm",
        "task": "260903_dhfr_divergence_parity",
        "issue": 4953,
        "platform": args.platform,
        "dt_fs": args.dt_fs,
        "minimized": args.minimize,
        "n_atoms": int(system.getNumParticles()),
        "n_constraints": int(system.getNumConstraints()),
        "box_edge_angstrom": box_edge,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
