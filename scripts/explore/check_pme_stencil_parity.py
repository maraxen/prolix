#!/usr/bin/env python3
"""GPU/host dump: vectorized SPME stencil vs Python 4³ oracle.

Writes JSON with max abs errors on Q, energy, and finite-difference forces.
Explore / ungated — follow-up to the stencil rewrite, not a ranking cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _spread_loop_oracle(positions, charges, atom_mask, box_size, grid_dims, order=4):
    import jax.numpy as jnp

    from prolix.physics.pme import _bspline4

    Kx, Ky, Kz = grid_dims
    work_dtype = positions.dtype
    K_arr = jnp.array([Kx, Ky, Kz], dtype=work_dtype)
    frac = (positions / box_size) * K_arr
    grid_idx = jnp.floor(frac).astype(jnp.int32)
    u = frac - grid_idx.astype(work_dtype)
    wx = _bspline4(u[:, 0])
    wy = _bspline4(u[:, 1])
    wz = _bspline4(u[:, 2])
    Q = jnp.zeros(grid_dims, dtype=work_dtype)
    q = charges * atom_mask.astype(charges.dtype)
    for dx in range(order):
        for dy in range(order):
            for dz in range(order):
                ix = (grid_idx[:, 0] + dx - 1) % Kx
                iy = (grid_idx[:, 1] + dy - 1) % Ky
                iz = (grid_idx[:, 2] + dz - 1) % Kz
                w = (wx[dx] * wy[dy] * wz[dz] * q).astype(Q.dtype)
                Q = Q.at[ix, iy, iz].add(w)
    return Q


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-fd-atoms", type=int, default=8)
    p.add_argument("--fd-eps", type=float, default=1e-4)
    p.add_argument(
        "--pallas",
        action="store_true",
        help="Compare Pallas spread/forces to XLA stencil (ADR-006 dump).",
    )
    args = p.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from prolix.api.bundle_md import physics_system_from_bundle
    from prolix.physics.pme import (
        compute_pme_grid_dims,
        spread_charges,
        spme_energy_with_forces,
        spme_reciprocal_energy,
    )
    from scripts.experiments.prof_ensemble_step import _build_1vii

    bundle = _build_1vii(True, jnp.float32)
    sys_obj = physics_system_from_bundle(bundle, bundle.positions)
    pos = jnp.asarray(sys_obj.positions)
    charges = jnp.asarray(sys_obj.charges)
    mask = jnp.asarray(sys_obj.atom_mask)
    box = jnp.asarray(sys_obj.box_size, dtype=pos.dtype)
    grid_dims = compute_pme_grid_dims(np.asarray(box), grid_spacing=1.0)

    Q_vec = spread_charges(pos, charges, mask, box, grid_dims, 4, False)
    Q_ref = _spread_loop_oracle(pos, charges, mask, box, grid_dims, 4)
    jax.block_until_ready(Q_vec)
    jax.block_until_ready(Q_ref)
    q_err = float(jnp.max(jnp.abs(Q_vec - Q_ref)))
    q_pallas_err = None
    if args.pallas:
        Q_p = spread_charges(pos, charges, mask, box, grid_dims, 4, True)
        jax.block_until_ready(Q_p)
        q_pallas_err = float(jnp.max(jnp.abs(Q_p - Q_vec)))

    e_vec = float(spme_reciprocal_energy(pos, charges, mask, box, grid_dims))
    e_prod = float(
        spme_energy_with_forces(pos, charges, mask, box, grid_dims, 0.34, 4, False)
    )

    def energy(p):
        return spme_energy_with_forces(
            p, charges, mask, box, grid_dims, 0.34, 4, bool(args.pallas)
        )

    g = jax.grad(energy)(pos)
    jax.block_until_ready(g)
    rng = np.random.default_rng(0)
    n_real = int(jnp.sum(mask))
    picks = rng.choice(n_real, size=min(args.n_fd_atoms, n_real), replace=False)
    fd_max = 0.0
    eps = args.fd_eps
    for i in picks:
        for d in range(3):
            shift = jnp.zeros_like(pos)
            shift = shift.at[int(i), d].add(eps)
            ep = energy(pos + shift)
            em = energy(pos - shift)
            fd = (ep - em) / (2.0 * eps)
            fd_max = max(fd_max, float(jnp.abs(g[int(i), d] - fd)))

    payload = {
        "n_atoms": int(pos.shape[0]),
        "n_real": n_real,
        "grid_dims": list(grid_dims),
        "q_max_abs_err": q_err,
        "energy_kcal": e_prod,
        "reciprocal_kcal": e_vec,
        "fd_max_abs_err": fd_max,
        "n_fd_atoms": int(len(picks)),
        "fd_eps": eps,
        "q_pallas_vs_xla_max_abs_err": q_pallas_err,
        "q_pass": q_err == 0.0 or q_err < 1e-5,
        "pallas_q_pass": (
            True if q_pallas_err is None else q_pallas_err < 1e-5
        ),
        "fd_pass": fd_max < 5e-2,
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "backend": jax.default_backend(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["q_pass"] or not payload["fd_pass"] or not payload["pallas_q_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
