"""Parity, overhead and speed of the SPME gather behind xtrax.op (task 261027_xtrax-op-prolix-spike).

For each atom count, compares three jitted callables on one random system and
potential mesh:

- ``reference``: ``prolix.physics.pme_op.gather_reference`` (plain JAX);
- ``raw``: ``gather_pallas`` called directly (compiled on GPU, interpret mode on CPU);
- ``op``: ``make_gather_op(..., context=Context(kernels=True))``.

Parity is the scale-relative error ``max|op - ref| / max(1, max|ref|)`` over both
outputs. A corruption control perturbs one force by 1% of ``max|F|`` and must
exceed the tolerance. Timing is the median steady-state call (``block_until_ready``)
over ``--repeats`` after one compile call.

Writes one JSON result to ``--out`` and to ``$BTH_RESULTS_PATH`` when set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xtrax
from xtrax.intent import Context
from xtrax.testing.kernel_parity import KernelParityError, assert_kernel_ran

from prolix.physics.pme_op import gather_reference, make_gather_op
from prolix.physics.pme_pallas import gather_pallas

log = logging.getLogger("pme_gather_xtrax_op")

TOLERANCE = 1e-4
GHOST_FRACTION = 0.02


def make_system(n_atoms: int, grid: int, density: float, seed: int):
    """Random system at ``density`` atoms/A^3 with ~2% ghost atoms and a random mesh."""
    k_pos, k_q, k_t = jax.random.split(jax.random.PRNGKey(seed), 3)
    edge = (n_atoms / density) ** (1.0 / 3.0)
    box = jnp.full((3,), edge, jnp.float32)
    positions = jax.random.uniform(k_pos, (n_atoms, 3), jnp.float32) * box
    charges = jax.random.normal(k_q, (n_atoms,), jnp.float32)
    charges = charges - jnp.mean(charges)
    n_ghost = max(1, int(n_atoms * GHOST_FRACTION))
    mask = jnp.arange(n_atoms) < n_atoms - n_ghost
    theta = jax.random.normal(k_t, (grid, grid, grid), jnp.float32)
    return positions, charges, mask, box, theta


def scale_relative_error(got, want) -> float:
    """Worst ``max|got - want| / max(1, max|want|)`` over matching leaves."""
    errs = []
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        g, w = np.asarray(g, np.float64), np.asarray(w, np.float64)
        errs.append(float(np.max(np.abs(g - w)) / max(1.0, float(np.max(np.abs(w))))))
    return max(errs)


def time_call(fn, args, repeats: int) -> tuple[float, float]:
    """Return ``(first_call_s, median_steady_s)`` for a jitted ``fn``."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t0
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return first, statistics.median(samples)


def run_size(n_atoms: int, grid: int, density: float, repeats: int, seed: int) -> dict:
    """Measure one atom count."""
    dims = (grid, grid, grid)
    interpret = jax.default_backend() == "cpu"
    args = make_system(n_atoms, grid, density, seed)

    reference = jax.jit(lambda *a: gather_reference(*a, grid_dims=dims))
    raw = jax.jit(lambda *a: gather_pallas(*a, grid_dims=dims, interpret=interpret))
    op_fn = make_gather_op(dims, context=Context(kernels=True))
    op = jax.jit(lambda *a: op_fn(*a))

    try:
        assert_kernel_ran(op_fn, *args)
        kernel_ran = True
    except KernelParityError:
        kernel_ran = False

    want = reference(*args)
    got = op(*args)
    err = scale_relative_error(got, want)
    forces, potentials = got
    bump = 0.01 * float(jnp.max(jnp.abs(want[0])))
    control_err = scale_relative_error((forces.at[0, 0].add(bump), potentials), want)

    ref_first, ref_med = time_call(reference, args, repeats)
    raw_first, raw_med = time_call(raw, args, repeats)
    op_first, op_med = time_call(op, args, repeats)
    row = {
        "n_atoms": n_atoms,
        "max_rel_err": err,
        "control_rel_err": control_err,
        "kernel_ran": kernel_ran,
        "reference_first_s": ref_first,
        "reference_median_s": ref_med,
        "raw_first_s": raw_first,
        "raw_median_s": raw_med,
        "op_first_s": op_first,
        "op_median_s": op_med,
        "op_overhead_ratio": op_med / raw_med,
        "speedup_vs_reference": ref_med / op_med,
    }
    log.info("%s", json.dumps(row))
    return row


def main() -> None:
    """Parse arguments, measure every size, write the result JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="512,2048,8192,23558")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--density", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = [run_size(n, args.grid, args.density, args.repeats, args.seed) for n in sizes]
    largest = max(rows, key=lambda r: r["n_atoms"])
    result = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "xtrax_version": xtrax.__version__,
        "grid": args.grid,
        "sizes": args.sizes,
        "tolerance": TOLERANCE,
        "max_rel_err": max(r["max_rel_err"] for r in rows),
        "kernel_ran": all(r["kernel_ran"] for r in rows),
        "control_red": all(r["control_rel_err"] > TOLERANCE for r in rows),
        "op_overhead_ratio": largest["op_overhead_ratio"],
        "speedup_largest_n": largest["speedup_vs_reference"],
        "per_size": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    results_path = os.environ.get("BTH_RESULTS_PATH")
    if results_path:
        Path(results_path).write_text(json.dumps(result))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
