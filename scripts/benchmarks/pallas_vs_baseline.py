"""Apples-to-apples Pallas vs XLA-baseline scaling harness (Phase 3.1/3.2 + Backlog #6167).

Compares Prolix's Pallas kernels against their XLA references on identical
inputs at water-like density across system sizes:

* ``--kernel nonbonded``: fused Pallas neighbor-list nonbonded kernel
  (``pallas_nonbonded_nl``) vs chunked XLA NL reference
  (``chunked_lj_energy_nl`` + ``chunked_coulomb_energy_nl``).
* ``--kernel flash``: 2D-tiled Pallas FlashMD kernel
  (``pallas_flash_explicit_nonbonded``) vs tiled XLA ``jax.checkpoint`` scan
  (``chunked_explicit_nonbonded_energy(..., use_pallas=False)``).
* ``--kernel pme``: SPME B-spline spread/gather Pallas vs XLA.
* ``--kernel nl_build``: ``jax_md`` cell-list neighbor-list rebuild cost vs one
  Pallas nonbonded step.
* ``--kernel nl_skin``: neighbor-list skin (``dr_threshold``) and
  ``capacity_multiplier`` sweep measuring capacity ``K``, rebuild time, Pallas
  nonbonded step time, and net amortized step time.
* ``--kernel mem_scaling``: isolated **per-process** device memory scaling
  (spawning a fresh subprocess with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` for
  every ``(path, N)`` pair) comparing ``dense_xla`` ($O(N^2)$ un-tiled),
  ``flash_xla`` ($O(N\\cdot T)$ remat), ``flash_pallas`` ($O(N)$),
  ``nl_xla`` ($O(N\\cdot K)$ chunked), and ``nl_pallas`` ($O(N\\cdot K)$ fused
  forward-force Pallas).

Pass ``--isolate-memory`` with ``nonbonded``, ``flash``, or ``pme`` to measure
``peak_mem_after_xla_bytes`` and ``peak_mem_after_pallas_bytes`` (plus working
and activation memory deltas) in fresh per-path subprocesses instead of a
single cumulative process.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from prolix.pallas_kernels import (
    _triton_gpu_available,
    pallas_flash_explicit_nonbonded,
    pallas_nonbonded_nl,
)
from prolix.physics.flash_explicit import chunked_explicit_nonbonded_energy
from prolix.physics.optimization import chunked_coulomb_energy_nl, chunked_lj_energy_nl

CUTOFF = 9.0
SWITCH_WIDTH = 1.5
PME_ALPHA = 0.34
DENSITY = 0.1  # atoms / A^3 (liquid water)


def build_system(n: int, seed: int = 0, include_nl: bool = True):
    """Random periodic system at water density with optional padded neighbor list."""
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

    nbr_jnp = None
    k = 0
    if include_nl:
        from scipy.spatial import cKDTree

        tree = cKDTree(pos, boxsize=box_len)
        pairs = tree.query_ball_point(pos, r=CUTOFF + 1.0)
        k_max = max(len(p) for p in pairs) - 1
        k = int(np.ceil(max(k_max, 1) / 8.0) * 8)
        nbr = np.full((n, k), n, dtype=np.int32)
        for i, p in enumerate(pairs):
            p = [j for j in p if j != i]
            nbr[i, : len(p)] = p
        nbr_jnp = jnp.asarray(nbr)

    sigma = rng.uniform(1.6, 2.0, size=n).astype(np.float32)
    eps = rng.uniform(0.05, 0.2, size=n).astype(np.float32)
    q = rng.uniform(-0.6, 0.6, size=n).astype(np.float32)
    q -= q.mean()
    return {
        "box": jnp.full((3,), box_len, dtype=jnp.float32),
        "pos": jnp.asarray(pos),
        "nbr": nbr_jnp,
        "sigma": jnp.asarray(sigma),
        "eps": jnp.asarray(eps),
        "q": jnp.asarray(q),
        "mask": jnp.ones((n,), dtype=bool),
        "excl_idx": jnp.full((n, 1), -1, dtype=jnp.int32),
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


def make_flash_fns(s, interpret: bool):
    """FlashMD dense O(N^2) tiled nonbonded: XLA lax.scan+remat vs Pallas 2D-tiled."""
    sys_stub = SimpleNamespace(
        excl_indices=s["excl_idx"],
        excl_scales_vdw=s["excl_vdw"],
        excl_scales_elec=s["excl_elec"],
        box_size=s["box"],
    )

    def xla_fn(p):
        return chunked_explicit_nonbonded_energy(
            p, s["q"], s["sigma"], s["eps"], s["mask"], sys_stub,
            T=256, soft_core_lambda=1.0, pme_alpha=PME_ALPHA,
            remat=True, use_pallas=False,
        )

    def pallas_fn(p):
        return pallas_flash_explicit_nonbonded(
            p, s["q"], s["sigma"], s["eps"], s["mask"],
            s["excl_idx"], s["excl_vdw"], s["excl_elec"],
            box=s["box"], pme_alpha=PME_ALPHA, soft_core_lambda=1.0,
            interpret=interpret,
        )

    return jax.jit(jax.value_and_grad(xla_fn)), jax.jit(jax.value_and_grad(pallas_fn))


def make_dense_xla_fn(s):
    """Un-tiled O(N^2) dense pairwise LJ + damped Coulomb XLA reference (for memory scaling)."""
    from jax.scipy.special import erfc
    from proxide.physics.constants import COULOMB_CONSTANT

    n = s["pos"].shape[0]
    box = s["box"]

    def dense_fn(p):
        dr = p[:, None, :] - p[None, :, :]
        scaled = dr / box
        wrap_idx = jnp.sign(scaled) * jnp.floor(jnp.abs(scaled) + 0.5)
        dr = dr - box * wrap_idx
        dist_sq = jnp.sum(dr * dr, axis=-1) + 1e-10
        dist = jnp.sqrt(dist_sq)
        not_self = 1.0 - jnp.eye(n, dtype=p.dtype)
        sig_ij = 0.5 * (s["sigma"][:, None] + s["sigma"][None, :])
        eps_ij = jnp.sqrt(s["eps"][:, None] * s["eps"][None, :])
        u6 = (sig_ij / dist) ** 6
        e_lj = 4.0 * eps_ij * (u6 * u6 - u6) * not_self
        qq = s["q"][:, None] * s["q"][None, :] * COULOMB_CONSTANT
        e_coul = qq * erfc(PME_ALPHA * dist) / dist * not_self
        return 0.5 * jnp.sum(e_lj + e_coul)

    return jax.jit(jax.value_and_grad(dense_fn))


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


def _mem_stats() -> dict[str, int] | None:
    try:
        st = jax.devices()[0].memory_stats()
        if not st:
            return None
        return {
            "bytes_in_use": int(st.get("bytes_in_use", 0)),
            "peak_bytes_in_use": int(st.get("peak_bytes_in_use", 0)),
        }
    except Exception:  # noqa: BLE001 - backend without memory stats
        return None


def _peak_mem() -> int | None:
    st = _mem_stats()
    return None if st is None else st["peak_bytes_in_use"]


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
    pairs = [(n, t) for n, t in zip(ns, ts) if n is not None and t is not None and t > 0]
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def _run_isolated_worker(spec: str, repeats: int, interpret: bool) -> dict:
    """Execute one (path, n_atoms) case in a fresh process for uncontaminated peak memory."""
    path_name, n_str = spec.split(":")
    n = int(n_str)
    # Touch device before building tensors to record baseline runtime memory.
    _ = jnp.zeros((1,), dtype=jnp.float32).block_until_ready()
    m0 = _mem_stats()
    base_bytes = m0["bytes_in_use"] if m0 else None

    needs_nl = path_name in ("nl_xla", "nl_pallas", "nonbonded_xla", "nonbonded_pallas")
    s = build_system(n, include_nl=needs_nl)
    jax.block_until_ready(s["pos"])
    if s["nbr"] is not None:
        jax.block_until_ready(s["nbr"])
    m1 = _mem_stats()
    input_bytes = m1["bytes_in_use"] if m1 else None

    if path_name in ("nl_xla", "nonbonded_xla"):
        fn, _ = make_fns(s, interpret)
    elif path_name in ("nl_pallas", "nonbonded_pallas"):
        _, fn = make_fns(s, interpret)
    elif path_name == "flash_xla":
        fn, _ = make_flash_fns(s, interpret)
    elif path_name == "flash_pallas":
        _, fn = make_flash_fns(s, interpret)
    elif path_name == "pme_xla":
        fn, _ = make_pme_fns(s, interpret)
    elif path_name == "pme_pallas":
        _, fn = make_pme_fns(s, interpret)
    elif path_name == "dense_xla":
        fn = make_dense_xla_fn(s)
    else:
        raise ValueError(f"Unknown worker path: {path_name}")

    out, compile_s, median_s = time_fn(fn, s["pos"], repeats)
    m2 = _mem_stats()
    peak_bytes = m2["peak_bytes_in_use"] if m2 else None
    working_peak = (peak_bytes - base_bytes) if (peak_bytes is not None and base_bytes is not None) else None
    activation_peak = (peak_bytes - input_bytes) if (peak_bytes is not None and input_bytes is not None) else None
    return {
        "path": path_name,
        "n_atoms": n,
        "k": int(s["k"]),
        "compile_s": compile_s,
        "median_s": median_s,
        "baseline_bytes": base_bytes,
        "input_bytes": input_bytes,
        "peak_bytes": peak_bytes,
        "working_peak_bytes": working_peak,
        "activation_peak_bytes": activation_peak,
        "energy": float(out[0]),
    }


def _spawn_worker(path_name: str, n: int, repeats: int, interpret: bool) -> dict:
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Bypass BFC 64 MiB arena rounding so peak_bytes_in_use reflects exact tensor/activation HBM.
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        f"{path_name}:{n}",
        "--repeats",
        str(repeats),
    ]
    if interpret:
        cmd.append("--interpret")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {
            "path": path_name,
            "n_atoms": n,
            "error": (proc.stderr or proc.stdout)[-400:],
            "peak_bytes": None,
            "working_peak_bytes": None,
            "activation_peak_bytes": None,
            "median_s": None,
        }
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    return json.loads(lines[-1])


def bench_mem_scaling(sizes: list[int], repeats: int, interpret: bool) -> tuple[list[dict], dict]:
    """Per-process memory scaling across dense O(N^2), Flash O(N*T)/O(N), and NL O(N*K)."""
    rows = []
    paths = ("dense_xla", "flash_xla", "flash_pallas", "nl_xla", "nl_pallas")
    for n in sizes:
        row: dict = {"n_atoms": n}
        for p in paths:
            # Skip un-tiled dense O(N^2) past 16k and O(N^2)-compute flash past 32k so runs stay fast.
            if p == "dense_xla" and n > 16384:
                continue
            if p in ("flash_xla", "flash_pallas") and n > 32768:
                continue
            res = _spawn_worker(p, n, repeats, interpret)
            if "k" in res and res["k"]:
                row["k"] = res["k"]
            row[p] = {
                "median_s": res.get("median_s"),
                "compile_s": res.get("compile_s"),
                "peak_bytes": res.get("peak_bytes"),
                "working_peak_bytes": res.get("working_peak_bytes"),
                "activation_peak_bytes": res.get("activation_peak_bytes"),
                "error": res.get("error"),
            }
        print(json.dumps(row), flush=True)
        rows.append(row)

    summary = {}
    for p in paths:
        ns_p = [r["n_atoms"] for r in rows if p in r and r[p].get("working_peak_bytes")]
        wp_p = [r[p]["working_peak_bytes"] for r in rows if p in r and r[p].get("working_peak_bytes")]
        ap_p = [r[p]["activation_peak_bytes"] for r in rows if p in r and r[p].get("activation_peak_bytes")]
        ts_p = [r[p]["median_s"] for r in rows if p in r and r[p].get("median_s")]
        summary[f"{p}_working_mem_exponent"] = fit_exponent(ns_p, wp_p)
        summary[f"{p}_activation_mem_exponent"] = fit_exponent(ns_p, ap_p)
        summary[f"{p}_time_exponent"] = fit_exponent(ns_p, ts_p)
    return rows, summary


def bench_nl_build(sizes: list[int], repeats: int, interpret: bool) -> list[dict]:
    """Neighbor-list (jax_md cell list) rebuild cost vs one Pallas nonbonded step."""
    from prolix.physics.neighbor_list import make_neighbor_list_fn

    rows = []
    for n in sizes:
        s = build_system(n)
        disp, _ = space.periodic(s["box"])
        nl_fns = make_neighbor_list_fn(disp, s["box"], CUTOFF)
        nbrs = nl_fns.allocate(s["pos"])
        upd = jax.jit(nl_fns.update)
        moved = s["pos"] + jnp.float32(0.4)
        _, _, t_upd = time_fn(lambda p: upd(p, nbrs).idx, moved, repeats)
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


def bench_nl_skin(sizes: list[int], repeats: int, interpret: bool) -> list[dict]:
    """Sweep NL skin (dr_threshold) and capacity_multiplier vs amortized step cost."""
    from prolix.physics.neighbor_list import make_neighbor_list_fn

    # Typical thermal RMS max-atom displacement per 2 fs step in 300 K water (~0.015 A/step).
    step_disp_a = 0.015
    rows = []
    for n in sizes:
        s_base = build_system(n, include_nl=False)
        disp, _ = space.periodic(s_base["box"])
        for dr_threshold in (0.5, 1.0, 1.5, 2.0):
            for cap_mult in (1.15, 1.25, 1.50):
                nl_fns = make_neighbor_list_fn(
                    disp, s_base["box"], CUTOFF,
                    dr_threshold=dr_threshold, capacity_multiplier=cap_mult,
                )
                nbrs = nl_fns.allocate(s_base["pos"])
                k_raw = int(nbrs.idx.shape[1])
                k_pad = int(np.ceil(max(k_raw, 1) / 8.0) * 8)
                nbr_pad = jnp.pad(
                    nbrs.idx, ((0, 0), (0, k_pad - k_raw)), constant_values=n
                )
                s = {**s_base, "nbr": nbr_pad, "k": k_pad}
                upd = jax.jit(nl_fns.update)
                shift = jnp.float32(dr_threshold * 0.6)
                moved = s["pos"] + shift
                _, _, t_upd = time_fn(lambda p: upd(p, nbrs).idx, moved, repeats)
                _, pal_vg = make_fns(s, interpret)
                _, _, t_nb = time_fn(pal_vg, s["pos"], repeats)
                est_steps_between_rebuilds = max(1.0, (dr_threshold / 2.0) / step_disp_a)
                amortized_step_s = t_nb + (t_upd / est_steps_between_rebuilds)
                row = {
                    "n_atoms": n,
                    "dr_threshold": dr_threshold,
                    "capacity_multiplier": cap_mult,
                    "nl_capacity_k": k_pad,
                    "est_steps_per_rebuild": round(est_steps_between_rebuilds, 1),
                    "nl_update_median_s": t_upd,
                    "pallas_nonbonded_median_s": t_nb,
                    "amortized_step_median_s": amortized_step_s,
                }
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
        "--isolate-memory",
        action="store_true",
        help="measure peak device memory for each path in a fresh subprocess",
    )
    ap.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    ap.add_argument(
        "--kernel",
        choices=("nonbonded", "flash", "pme", "nl_build", "nl_skin", "mem_scaling"),
        default="nonbonded",
        help="kernel/experiment family to benchmark",
    )
    args = ap.parse_args()

    interpret = args.interpret or not _triton_gpu_available()
    if args._worker:
        res = _run_isolated_worker(args._worker, args.repeats, interpret)
        print(json.dumps(res), flush=True)
        return

    sizes = [int(x) for x in args.sizes.split(",")]
    meta = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "device_kind": getattr(jax.devices()[0], "device_kind", "unknown"),
        "pallas_interpret": interpret,
        "kernel": args.kernel,
        "jax": jax.__version__,
        "host": platform.node(),
        "cutoff": CUTOFF,
        "density": DENSITY,
    }

    if args.kernel in ("nl_build", "nl_skin", "mem_scaling"):
        if args.kernel == "nl_build":
            rows = bench_nl_build(sizes, args.repeats, interpret)
            summary = {}
        elif args.kernel == "nl_skin":
            rows = bench_nl_skin(sizes, args.repeats, interpret)
            summary = {}
        else:
            rows, summary = bench_mem_scaling(sizes, args.repeats, interpret)
        payload = {"meta": meta, "rows": rows, "summary": summary}
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        if summary:
            print(json.dumps(summary), flush=True)
        return

    if args.kernel == "pme":
        maker = make_pme_fns
        worker_prefix = "pme"
        needs_nl = False
    elif args.kernel == "flash":
        maker = make_flash_fns
        worker_prefix = "flash"
        needs_nl = False
    else:
        maker = make_fns
        worker_prefix = "nonbonded"
        needs_nl = True

    rows = []
    for n in sizes:
        s = build_system(n, include_nl=needs_nl)
        xla_vg, pal_vg = maker(s, interpret)
        row = {"n_atoms": n, "k": s["k"]}
        out_x, c_x, t_x = time_fn(xla_vg, s["pos"], args.repeats)
        row.update(xla_compile_s=c_x, xla_median_s=t_x)
        mem_x = _peak_mem()
        out_p, c_p, t_p = time_fn(pal_vg, s["pos"], args.repeats)
        row.update(pallas_compile_s=c_p, pallas_median_s=t_p, speedup=t_x / t_p)
        mem_p = _peak_mem()
        if args.isolate_memory:
            wx = _spawn_worker(f"{worker_prefix}_xla", n, args.repeats, interpret)
            wp = _spawn_worker(f"{worker_prefix}_pallas", n, args.repeats, interpret)
            row["peak_mem_after_xla_bytes"] = wx.get("peak_bytes")
            row["peak_mem_after_pallas_bytes"] = wp.get("peak_bytes")
            row["xla_working_peak_bytes"] = wx.get("working_peak_bytes")
            row["pallas_working_peak_bytes"] = wp.get("working_peak_bytes")
            row["xla_activation_peak_bytes"] = wx.get("activation_peak_bytes")
            row["pallas_activation_peak_bytes"] = wp.get("activation_peak_bytes")
        else:
            row["peak_mem_after_xla_bytes"] = mem_x
            row["peak_mem_after_pallas_bytes"] = mem_p
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
    if args.isolate_memory:
        summary["xla_working_mem_exponent"] = fit_exponent(
            ns, [r.get("xla_working_peak_bytes") for r in rows]
        )
        summary["pallas_working_mem_exponent"] = fit_exponent(
            ns, [r.get("pallas_working_peak_bytes") for r in rows]
        )
    payload = {
        "meta": meta,
        "rows": rows,
        "summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
