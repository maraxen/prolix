"""Is the DHFR NVT trajectory physical at all? (task 260903_dhfr_nl_capacity)

bench_dhfr_parity has never checked. ``_block()`` only waits on the array, so a
diverged run still produces a plausible wall-clock number, and every NL+PME parity
test in the repo runs on 1VII -- chosen explicitly to keep the water count small.
So the DHFR force path has no stability evidence at its own scale, on either
branch, and the previously-reported 110.9 ms/step may have been timing a system
that had already fallen apart.

Job 21896491/21896938 forced the question: the neighbor list needs <=768 slots at
t=0 (jax_md's own allocate() measured it on the real configuration) and >5887
after ~200 fs. That is a ~7.7x local density increase -- a collapse, not a
capacity mis-estimate. It reproduced with all 1442 ghost/padding atoms removed, so
it is the REAL atoms piling up.

This script runs the same bundle in short chunks and reports, per chunk, the three
quantities that separate the candidate explanations:

  * finite        -- did positions go non-finite (divergence)
  * min_dist      -- closest approach over a sampled subset. Collapsing matter
                     drives this toward 0; a healthy TIP3P/protein box holds
                     around ~0.9 A (the O-H bond) and no closer.
  * max_neighbors -- largest neighbor count within the cutoff over the same
                     sample. This is the quantity that overflowed, measured
                     directly rather than inferred from an allocation failure.

Run it on both force paths and both timesteps. The readings discriminate:

  flash stable + NL collapses   -> NL-specific force bug (exclusions/PME on the
                                   NL branch); dense vs NL parity at DHFR scale
                                   is the follow-up, since it only exists for 1VII
  both collapse, dt=0.5 fine    -> genuine timestep/stability limit at this scale,
                                   not a code defect; CLAUDE.md's dt<=1.0 fs cap
                                   was validated on pure water, not this bundle
  both collapse at both dt      -> the system build itself is wrong (box, water
                                   parameter override, or the bonded filtering)

Usage (cluster only -- never run JAX locally for prolix):
    uv run --extra cuda python scripts/explore/diag_dhfr_nvt_stability.py \
        --force-path nl --dt 1.0 --chunk-steps 20 --n-chunks 12
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

logger = logging.getLogger("diag_dhfr")


def _sample_geometry(positions, box_vec, cutoff: float, n_sample: int, key):
    """Min pair distance and max neighbor count over a random sample of atoms.

    Sampled rather than exhaustive: the full (N, N) distance matrix for 23,558
    atoms is the very allocation this whole sprint exists to avoid. A few hundred
    rows is ample to detect a 7.7x density change.
    """
    import jax
    import jax.numpy as jnp

    n = positions.shape[0]
    idx = jax.random.choice(key, n, shape=(min(n_sample, n),), replace=False)
    sample = positions[idx]

    # Minimum-image displacement against the full set.
    d = sample[:, None, :] - positions[None, :, :]
    d = d - box_vec[None, None, :] * jnp.round(d / box_vec[None, None, :])
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))

    # Mask each sampled atom's self-pair (exactly r == 0 at its own column).
    self_mask = jnp.zeros_like(r, dtype=bool).at[jnp.arange(sample.shape[0]), idx].set(True)
    r_no_self = jnp.where(self_mask, jnp.inf, r)

    min_dist = float(jnp.min(r_no_self))
    max_neighbors = int(jnp.max(jnp.sum(r_no_self < cutoff, axis=1)))
    mean_neighbors = float(jnp.mean(jnp.sum(r_no_self < cutoff, axis=1)))
    return min_dist, max_neighbors, mean_neighbors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force-path", choices=("nl", "flash"), default="nl")
    p.add_argument("--dt", type=float, default=1.0, help="fs")
    p.add_argument("--chunk-steps", type=int, default=20)
    p.add_argument("--n-chunks", type=int, default=12)
    p.add_argument("--n-sample", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import bench_dhfr_parity as B

    B._init_jax()
    import jax
    import jax.numpy as jnp

    from prolix.api import EnsemblePlan
    from prolix.simulate import BOLTZMANN_KCAL

    ff_path = B._resolve_ff_path()
    sys_data = B._build_dhfr_system_dict(ff_path)
    bundle = B._build_bundle_from_sys_data(sys_data)

    box_vec = jnp.diag(bundle.box)
    cutoff = B.NONBONDED_CUTOFF
    kT = B.TARGET_KELVIN * BOLTZMANN_KCAL
    key = jax.random.PRNGKey(args.seed)

    # NL and flash are mutually exclusive in the resolver -- set exactly one.
    path_kwargs = (
        {"use_neighbor_list": True} if args.force_path == "nl" else {"use_flash_forces": True}
    )

    logger.info(
        "force_path=%s dt=%.2f fs chunk_steps=%d n_chunks=%d n_atoms=%d cutoff=%.1f",
        args.force_path, args.dt, args.chunk_steps, args.n_chunks,
        int(bundle.positions.shape[0]), cutoff,
    )

    key, sub = jax.random.split(key)
    min_d, max_n, mean_n = _sample_geometry(
        bundle.positions, box_vec, cutoff, args.n_sample, sub
    )
    logger.info(
        "step=0 finite=True min_dist=%.3f max_neighbors=%d mean_neighbors=%.1f",
        min_d, max_n, mean_n,
    )
    rows = [{
        "step": 0, "finite": True, "min_dist": min_d,
        "max_neighbors": max_n, "mean_neighbors": mean_n,
    }]

    positions = bundle.positions
    failure = None
    for chunk in range(1, args.n_chunks + 1):
        step = chunk * args.chunk_steps
        cur = dataclasses.replace(bundle, positions=positions)
        plan = EnsemblePlan.from_bundle(cur)
        try:
            traj = plan.run(
                n_steps=args.chunk_steps, dt=args.dt, kT=kT, seed=args.seed + chunk,
                gamma=B.GAMMA_PS, run_mode="inference", **path_kwargs,
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            logger.error("step=%d raised: %s", step, failure)
            rows.append({"step": step, "raised": failure})
            break

        t = traj[0] if isinstance(traj, list) else traj
        positions = t.positions[-1]
        finite = bool(jnp.all(jnp.isfinite(positions)))
        if not finite:
            logger.error("step=%d NON-FINITE positions -- trajectory diverged", step)
            rows.append({"step": step, "finite": False})
            failure = "non-finite positions"
            break

        key, sub = jax.random.split(key)
        min_d, max_n, mean_n = _sample_geometry(positions, box_vec, cutoff, args.n_sample, sub)
        logger.info(
            "step=%d finite=True min_dist=%.3f max_neighbors=%d mean_neighbors=%.1f",
            step, min_d, max_n, mean_n,
        )
        rows.append({
            "step": step, "finite": True, "min_dist": min_d,
            "max_neighbors": max_n, "mean_neighbors": mean_n,
        })

    result = {
        "force_path": args.force_path,
        "dt_fs": args.dt,
        "n_atoms": int(bundle.positions.shape[0]),
        "cutoff": cutoff,
        "chunk_steps": args.chunk_steps,
        "n_chunks": args.n_chunks,
        "seed": args.seed,
        "failure": failure,
        "rows": rows,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
