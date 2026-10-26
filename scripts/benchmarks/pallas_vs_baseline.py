"""Apples-to-apples Pallas vs XLA-baseline scaling harness (Phase 3.1/3.2).

Compares the fused Pallas neighbor-list nonbonded kernel against the chunked XLA
reference (``chunked_lj_energy_nl`` + ``chunked_coulomb_energy_nl``) on identical
inputs at water-like density, across system sizes. Records for each size:

* steady-state ``value_and_grad`` wall time (compile excluded, median of repeats),
* compile + first-call time,
* peak device memory delta when the backend reports it,
* parity (max |dE|, max |dF|) between the two paths,
* a least-squares scaling exponent for time vs N for each path.

Both paths receive the same positions, neighbor list, exclusions and parameters.
Results are written as JSON so runs on titanix / engaging are comparable.

Usage::

    uv run python scripts/benchmarks/pallas_vs_baseline.py --sizes 512,2048,8192,32768 \
        --out outputs/bench/pallas_vs_baseline.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from prolix.pallas_kernels import _triton_gpu_available, pallas_nonbonded_nl
from prolix.physics.optimization import chunked_coulomb_energy_nl, chunked_lj_energy_nl

CUTOFF = 9.0
SWITCH_WIDTH = 1.5
PME_ALPHA = 0.34
DENSITY = 0.1  # atoms / A^3 (liquid water)


def build_system(n: int, seed: int = 0):
    """Random periodic system at water density with a padded neighbor list."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    # Jittered cubic lattice: avoids the overlapping atoms of uniform-random
    # placement, whose ~1e13 energies make float32 parity meaningless.
    spacing = (1.0 / DENSITY) ** (1.0 / 3.0)
    m = int(np.ceil(n ** (1.0 / 3.0)))
    box_len = float(m * spacing)
    grid = np.stack(np.meshgrid(*[np.arange(m)] * 3, indexing="ij"), -1).reshape(-1, 3)
    grid = grid[rng.permutation(len(grid))[:n]]
    pos = ((grid + 0.5) * spacing + rng.uniform(-0.1, 0.1, (n, 3)) * spacing).astype(np.float32)
    pos %= box_len
    tree = cKDTree(pos, boxsize=box_len)
    pairs = tree.query_ball_point(pos, r=CUTOFF + 1.0)
    k_max = max(len(p) for p in pairs) - 1
    k = int(np.ceil(k_max / 8.0) * 8)
    nbr = np.full((n, k), n, dtype=np.int32)
    for i, p in enumerate(pairs):
        p = [j for j in p if j != i]
        nbr[i, : len(p)] = p
    sigma = rng.uniform(1.6, 2.0, size=n).astype(np.float32)
    eps = rng.uniform(0.05, 0.2, size=n).astype(np.float32)
    q = rng.uniform(-0.6, 0.6, size=n).astype(np.float32)
    q -= q.mean()
    return {
        "box": jnp.full((3,), box_len, dtype=jnp.float32),
        "pos": jnp.asarray(pos),
        "nbr": jnp.asarray(nbr),
        "sigma": jnp.asarray(sigma),
        "eps": jnp.asarray(eps),
        "q": jnp.asarray(q),
        "mask": jnp.ones((n,), dtype=bool),
        "excl_idx": jnp.full((n, 1), n, dtype=jnp.int32),
        "excl_vdw": jnp.ones((n, 1), dtype=jnp.float32),
        "excl_elec": jnp.ones((n, 1), dtype=jnp.float32),
        "k": k,
    }


def make_fns(s, interpret: bool):
    disp, _ = space.periodic(s["box"])

    def xla_fn(p):
        e_lj = chunked_lj_energy_nl(
            p, s["sigma"], s["eps"], s["excl_idx"], s["excl_vdw"], s["nbr"], disp,
            cutoff=CUTOFF, tile_size=256, switch_width=SWITCH_WIDTH,
        )
        e_c = chunked_coulomb_energy_nl(
            p, s["q"], s["excl_idx"], s["excl_elec"], s["nbr"], disp,
            pme_alpha=PME_ALPHA, cutoff=CUTOFF, tile_size=256,
        )
        return e_lj + e_c

    def pallas_fn(p):
        return pallas_nonbonded_nl(
            p, s["q"], s["sigma"], s["eps"], s["mask"], s["nbr"], s["excl_idx"],
            s["excl_vdw"], s["excl_elec"], box=s["box"], pme_alpha=PME_ALPHA,
            cutoff=CUTOFF, lj_switch_width=SWITCH_WIDTH, soft_core_lambda=1.0,
            include_pme_excl_erf=True, interpret=interpret,
        )

    return jax.jit(jax.value_and_grad(xla_fn)), jax.jit(jax.value_and_grad(pallas_fn))


def make_pme_fns(s, interpret: bool):
    """SPME reciprocal+self energy ``value_and_grad``: XLA vs Pallas spread/gather."""
    from prolix.physics.pme import compute_pme_grid_dims, spme_energy_with_forces

    grid_dims = compute_pme_grid_dims(np.asarray(s["box"]), grid_spacing=1.0)

    def make(use_pallas: bool):
        def fn(p):
            return spme_energy_with_forces(
                p, s["q"], s["mask"], s["box"], grid_dims, PME_ALPHA, 4, use_pallas
            )

        return jax.jit(jax.value_and_grad(fn))

    return make(False), make(True)


def _peak_mem() -> int | None:
    try:
        st = jax.devices()[0].memory_stats()
        return None if st is None else int(st.get("peak_bytes_in_use", 0))
    except Exception:  # noqa: BLE001 - backend without memory stats
        return None


def time_fn(fn, pos, repeats: int):
    t0 = time.perf_counter()
    out = fn(pos)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(pos))
        times.append(time.perf_counter() - t0)
    return out, compile_s, statistics.median(times)


def fit_exponent(ns, ts):
    if len(ns) < 2:
        return None
    return float(np.polyfit(np.log(ns), np.log(ts), 1)[0])


def bench_nl_build(sizes: list[int], repeats: int, interpret: bool) -> list[dict]:
    """Neighbor-list (jax_md cell list) rebuild cost vs one Pallas nonbonded step.

    Answers whether a Pallas NL builder could matter: the rebuild is amortized over
    ``steps_per_rebuild`` force evaluations, so ``amortized_ratio`` is the fraction of
    step time a perfect (zero-cost) builder could remove.
    """
    from prolix.physics.neighbor_list import make_neighbor_list_fn

    rows = []
    for n in sizes:
        s = build_system(n)
        disp, _ = space.periodic(s["box"])
        nl_fns = make_neighbor_list_fn(disp, s["box"], CUTOFF)
        nbrs = nl_fns.allocate(s["pos"])
        upd = jax.jit(nl_fns.update)
        # jax_md's update() only rebuilds (lax.cond) when some atom moved more than
        # dr_threshold/2 from the reference positions; at the allocation positions it
        # takes the cheap no-op branch. Shift every atom by 0.4 A (> 0.25 A) per axis
        # (0.69 A total) so the timed call is a genuine cell-list rebuild.
        moved = s["pos"] + jnp.float32(0.4)
        out, _, t_upd = time_fn(lambda p: upd(p, nbrs).idx, moved, repeats)
        _, _, t_chk = time_fn(lambda p: upd(p, nbrs).idx, s["pos"], repeats)
        _, pal_vg = make_fns(s, interpret)
        _, _, t_nb = time_fn(pal_vg, s["pos"], repeats)
        row = {
            "n_atoms": n,
            "nl_capacity_k": int(nbrs.idx.shape[1]),
            "nl_update_median_s": t_upd,
            "nl_skin_check_only_median_s": t_chk,
            "pallas_nonbonded_median_s": t_nb,
            "rebuild_over_step": t_upd / t_nb,
        }
        for steps in (10, 20, 50):
            row[f"amortized_ratio_{steps}"] = (t_upd / steps) / (t_nb + t_upd / steps)
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,2048,8192,32768")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="outputs/bench/pallas_vs_baseline.json")
    ap.add_argument("--interpret", action="store_true", help="force Pallas interpret mode")
    ap.add_argument(
        "--kernel", choices=("nonbonded", "pme", "nl_build"), default="nonbonded",
        help="nonbonded: NL LJ+Coulomb kernel; pme: SPME spread/gather kernels; "
        "nl_build: jax_md neighbor-list rebuild cost vs the Pallas nonbonded step",
    )
    args = ap.parse_args()
    maker = make_pme_fns if args.kernel == "pme" else make_fns

    interpret = args.interpret or not _triton_gpu_available()
    if args.kernel == "nl_build":
        sizes = [int(x) for x in args.sizes.split(",")]
        rows = bench_nl_build(sizes, args.repeats, interpret)
        payload = {
            "meta": {
                "backend": jax.default_backend(),
                "device_kind": getattr(jax.devices()[0], "device_kind", "unknown"),
                "pallas_interpret": interpret,
                "kernel": "nl_build",
                "jax": jax.__version__,
            },
            "rows": rows,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        return

    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        s = build_system(n)
        xla_vg, pal_vg = maker(s, interpret)
        row = {"n_atoms": n, "k": s["k"]}
        out_x, c_x, t_x = time_fn(xla_vg, s["pos"], args.repeats)
        row.update(xla_compile_s=c_x, xla_median_s=t_x)
        mem_x = _peak_mem()
        out_p, c_p, t_p = time_fn(pal_vg, s["pos"], args.repeats)
        row.update(pallas_compile_s=c_p, pallas_median_s=t_p, speedup=t_x / t_p)
        row["peak_mem_after_xla_bytes"] = mem_x
        row["peak_mem_after_pallas_bytes"] = _peak_mem()
        row["max_abs_dE"] = float(abs(out_x[0] - out_p[0]))
        row["rel_dE"] = float(abs(out_x[0] - out_p[0]) / max(abs(float(out_x[0])), 1e-12))
        row["max_abs_dF"] = float(jnp.max(jnp.abs(out_x[1] - out_p[1])))
        print(json.dumps(row), flush=True)
        rows.append(row)

    ns = [r["n_atoms"] for r in rows]
    summary = {
        "xla_scaling_exponent": fit_exponent(ns, [r["xla_median_s"] for r in rows]),
        "pallas_scaling_exponent": fit_exponent(ns, [r["pallas_median_s"] for r in rows]),
    }
    payload = {
        "meta": {
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "device_kind": getattr(jax.devices()[0], "device_kind", "unknown"),
            "pallas_interpret": interpret,
            "kernel": args.kernel,
            "jax": jax.__version__,
            "host": platform.node(),
            "cutoff": CUTOFF,
            "density": DENSITY,
        },
        "rows": rows,
        "summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
