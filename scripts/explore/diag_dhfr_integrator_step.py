"""Does one integrator step move atoms by the amount Newton says? (#4953)

Why this exists
---------------
Everything upstream of the integrator is now verified:

  * forces at step 0 match OpenMM to 0.039% at 23558 atoms through the production
    NL+PME path (job 21967760: cosine 0.99999993, magnitude ratio 1.0000)
  * potential energy at step 0 is -71959.5 kcal/mol, against OpenMM's -73104.0,
    the whole gap being the known HBonds bonded difference
  * the box bug that inflated forces 166x is fixed (61de3d2)

And yet potential energy reaches 2.7e15 kcal/mol within 20 steps, with per-atom
displacement of 0.24 A/step rising to 1.5 A/step against ~0.005 A/step expected at
300 K. Correct forces in, catastrophe out. That localises the fault to the
integrator -- but "the integrator" is not an answer, and six mechanism guesses have
already been falsified on this bug.

So this measures an invariant instead, against ground truth that needs no second
engine. Start from rest, switch the thermostat off entirely (kT=0 so there is no
noise, gamma=0 so there is no friction), and take a single step. Newton then fixes
the answer exactly:

    dx = 0.5 * (F/m) * dt^2

Any deviation is a pure number, and that number names the defect:

    ratio ~ 1                -> the integrator is fine; look at the thermostat
    ratio ~ m (mass-valued)  -> a momentum/velocity confusion (p = mv dropped)
    ratio ~ 1/m              -> the inverse: a mass divided in twice
    ratio ~ constant != 1    -> a dt or unit-conversion factor; the value says which
                                (2, 0.5, 48.888 = AKMA_TIME_UNIT_FS, 4.184, ...)

Reporting the ratio grouped by atomic mass is what separates the mass-valued cases
from the constant ones: a single scalar ratio cannot tell "12x for carbon" from
"12x for everything".

This is the BATHOS "verify against synthetic ground truth" discipline applied to the
integrator itself -- the one component this sprint has never checked directly.

Cluster only (JAX). Never run locally for prolix.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

logger = logging.getLogger("diag_integrator")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force-path", choices=("nl", "flash"), default="nl")
    p.add_argument("--dt", type=float, default=1.0, help="fs")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import bench_dhfr_parity as B  # noqa: N812  (matches tests/bench/test_bench_dhfr_backend_split.py)

    B._init_jax()
    import jax
    import jax.numpy as jnp
    import numpy as np

    from prolix.api import EnsemblePlan
    from prolix.api.bundle_md import energy_fn_from_bundle
    from prolix.physics import neighbor_list as nl_mod
    from prolix.physics.pbc import create_periodic_space

    sys_data = B._build_dhfr_system_dict(B._resolve_ff_path())
    bundle = B._build_bundle_from_sys_data(sys_data)
    box_vec = jnp.diag(bundle.box)
    n_atoms = int(bundle.positions.shape[0])

    # Reference force at the starting configuration, via the same production path.
    energy_fn = energy_fn_from_bundle(bundle)
    disp_fn, _ = create_periodic_space(box_vec)
    nfn = nl_mod.make_neighbor_list_fn(disp_fn, box_vec, float(bundle.cutoff_distance))
    nbr = nfn.update(bundle.positions, nfn.allocate(bundle.positions))
    e0, grad = jax.value_and_grad(lambda r: energy_fn(r, neighbor=nbr))(bundle.positions)
    force = -np.asarray(grad, dtype=np.float64)
    mass = np.asarray(bundle.masses, dtype=np.float64)
    logger.info("E0=%.1f kcal/mol  n_atoms=%d  F_rms=%.3f", float(e0), n_atoms,
                float(np.sqrt(np.mean((force**2).sum(-1)))))

    # kT=0 and gamma=0: no noise, no friction, and momenta start at rest, so the
    # step is purely Newtonian and has a closed form.
    path_kwargs = (
        {"use_neighbor_list": True} if args.force_path == "nl" else {"use_flash_forces": True}
    )
    plan = EnsemblePlan.from_bundle(bundle)
    traj = plan.run(
        n_steps=args.steps, dt=args.dt, kT=0.0, seed=0, gamma=0.0,
        run_mode="inference", **path_kwargs,
    )
    t = traj[0] if isinstance(traj, list) else traj
    new_pos = np.asarray(t.positions[-1], dtype=np.float64)

    raw = new_pos - np.asarray(bundle.positions, dtype=np.float64)
    box_np = np.asarray(box_vec, dtype=np.float64)
    actual = raw - box_np[None, :] * np.round(raw / box_np[None, :])  # minimum image

    # AKMA: with dt in fs converted by AKMA_TIME_UNIT_FS, a = F/m is A/akma^2.
    dt_akma = args.dt / B.AKMA_TIME_UNIT_FS if hasattr(B, "AKMA_TIME_UNIT_FS") else None
    if dt_akma is None:
        from prolix.simulate import AKMA_TIME_UNIT_FS

        dt_akma = args.dt / AKMA_TIME_UNIT_FS
    expected = 0.5 * (force / mass[:, None]) * (dt_akma**2) * (args.steps**2)

    a_norm = np.linalg.norm(actual, axis=1)
    e_norm = np.linalg.norm(expected, axis=1)
    ok = e_norm > 1e-12
    ratio = np.full(n_atoms, np.nan)
    ratio[ok] = a_norm[ok] / e_norm[ok]

    logger.info(
        "displacement  actual_rms=%.6g  expected_rms=%.6g  ratio median=%.6g mean=%.6g",
        float(np.sqrt(np.mean(a_norm**2))), float(np.sqrt(np.mean(e_norm**2))),
        float(np.nanmedian(ratio)), float(np.nanmean(ratio)),
    )

    # Grouped by mass: this is what separates a mass-valued error from a constant one.
    groups = []
    for m in sorted({round(float(x), 3) for x in mass})[:12]:
        sel = np.isclose(mass, m) & ok
        if sel.sum() == 0:
            continue
        r = float(np.nanmedian(ratio[sel]))
        groups.append({"mass": m, "n": int(sel.sum()), "ratio_median": r})
        logger.info("  mass %7.3f  n=%6d  ratio_median=%.6g  (ratio*m=%.4g  ratio/m=%.4g)",
                    m, int(sel.sum()), r, r * m, r / m)

    report = {
        "task": "260903_dhfr_divergence_parity", "issue": 4953,
        "force_path": args.force_path, "dt_fs": args.dt, "steps": args.steps,
        "n_atoms": n_atoms, "energy0_kcal": float(e0),
        "actual_rms": float(np.sqrt(np.mean(a_norm**2))),
        "expected_rms": float(np.sqrt(np.mean(e_norm**2))),
        "ratio_median": float(np.nanmedian(ratio)),
        "by_mass": groups,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        logger.info("wrote %s", args.out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
