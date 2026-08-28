#!/usr/bin/env python3
"""Compile-once inner-loop timing for EnsemblePlan inference.

The Campaign E harness fits wall time vs n_steps for *separate* ``plan.run()``
calls. Each call rebuilds ``settle_langevin`` and ``jax.jit``s a new
``while_loop`` closed over that ``n_steps``, so t_ss is ~compile+init (seconds)
almost independent of step count. The reported slope/r² are residuals on that
floor, not Flash/NL kernel quality.

This probe: one integrator setup, one jit of a *fixed* ``fori_loop(n=N)``,
warmup, then time the second execution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.prof_ensemble_step import (  # noqa: E402
    _build_1vii,
    _run_kwargs,
)


def _time_compiled(bundle, *, n_inner: int, seed: int, dt: float, kT: float, gamma: float, run_kw: dict):
    import jax
    import jax.numpy as jnp

    from prolix.api import EnsemblePlan
    from prolix.api.ensemble_plan import _integrator_positions

    plan = EnsemblePlan.from_bundle(bundle)
    use_nl = bool(run_kw.get("use_neighbor_list"))
    use_flash = bool(run_kw.get("use_flash_forces"))
    prefix = int(bundle.positions.shape[0]) if use_nl else None

    nbr0 = None
    if use_nl:
        from prolix.api.bundle_md import (
            _host_float,
            displacement_fn_for_bundle,
            positions_with_prefix,
        )
        from prolix.physics import neighbor_list as nl_mod

        displacement_fn, _ = displacement_fn_for_bundle(bundle)
        box_vec = jnp.diag(bundle.box)
        cutoff = _host_float(bundle.cutoff_distance, 9.0)
        neighbor_fn = nl_mod.make_neighbor_list_fn(displacement_fn, box_vec, cutoff)
        nbr0 = neighbor_fn.allocate(positions_with_prefix(bundle, prefix))

    state, apply_fn, dt_s, kT_s = plan._setup_integrator(
        bundle,
        dt,
        kT,
        seed,
        integration_prefix=prefix,
        dt_unit="fs",
        gamma=gamma,
        use_flash_forces=use_flash,
        initial_neighbor=nbr0,
    )

    n_inner = int(n_inner)
    if use_nl:
        import equinox as eqx

        pad_mask_3d = bundle.atom_mask[:, None]
        ghost = _integrator_positions(state)
        update_every = 20

        def body(i, carry):
            langevin_state, neighbor = carry
            should_update = (i % update_every) == 0
            new_neighbor = jax.lax.cond(
                should_update,
                lambda n: n.update(_integrator_positions(langevin_state)),
                lambda n: n,
                neighbor,
            )
            new_state = apply_fn(langevin_state, neighbor=new_neighbor, kT=kT_s, dt=dt_s)
            new_R = jnp.where(pad_mask_3d, _integrator_positions(new_state), ghost)
            new_P = jnp.where(pad_mask_3d, new_state.momentum, 0.0)
            if hasattr(new_state, "positions"):
                new_state = eqx.tree_at(
                    lambda s: (s.positions, s.momentum), new_state, (new_R, new_P)
                )
            else:
                new_state = new_state.set(position=new_R, momentum=new_P)
            return (new_state, new_neighbor)

        init = (state, nbr0)

        @jax.jit
        def many(carry):
            return jax.lax.fori_loop(0, n_inner, body, carry)

    else:

        def body(i, s):
            return apply_fn(s, kT=kT_s, dt=dt_s)

        init = state

        @jax.jit
        def many(s):
            return jax.lax.fori_loop(0, n_inner, body, s)

    t0 = time.perf_counter()
    warmed = many(init)
    jax.block_until_ready(warmed)
    t_compile = time.perf_counter() - t0

    t1 = time.perf_counter()
    out = many(warmed)
    jax.block_until_ready(out)
    t_exec = time.perf_counter() - t1
    return t_compile, t_exec, t_exec / float(n_inner)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-inner", type=int, default=256)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cuda")

    import jax
    import jax.numpy as jnp

    from prolix.simulate import BOLTZMANN_KCAL

    print("jax.default_backend=", jax.default_backend())
    print("jax.devices=", jax.devices())

    kT = 300.0 * BOLTZMANN_KCAL
    cells = (
        ("flash", "on"),
        ("flash", "off"),
        ("nl", "on"),
        ("nl", "off"),
        ("dense", "on"),
    )
    rows = []
    for nonbonded, buckets in cells:
        bundle = _build_1vii(buckets == "on", jnp.float32)
        t_compile, t_exec, per = _time_compiled(
            bundle,
            n_inner=args.n_inner,
            seed=0,
            dt=1.0,
            kT=kT,
            gamma=10.0,
            run_kw=_run_kwargs(nonbonded),
        )
        row = {
            "system": "1vii",
            "nonbonded": nonbonded,
            "use_size_buckets": buckets == "on",
            "n_real": int(jnp.asarray(bundle.n_atoms)),
            "n_padded": int(bundle.positions.shape[0]),
            "n_inner": args.n_inner,
            "compile_plus_first_s": t_compile,
            "exec_s": t_exec,
            "per_step_s": per,
            "backend": str(jax.default_backend()),
        }
        print(json.dumps(row))
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cells": rows}, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
