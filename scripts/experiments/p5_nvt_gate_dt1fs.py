"""P5 NVT gate at dt=1.0 fs — extended timestep validation.

Tests whether settle_langevin can achieve stable T_rot at dt=1.0 fs
on a 895-water system with gamma=10 ps^-1 coupling strength.

Hypothesis: C3 AM conservation + C1 ambient-temperature OU (both committed)
together enable dt=1.0 fs operation with T_rot stable within ±5 K of 300 K.

Usage (via bth run):
    uv run bth run python scripts/experiments/p5_nvt_gate_dt1fs.py \\
        --out outputs/p5_nvt_gate_dt1fs.json \\
        --campaign <campaign-id>

Local L2 smoke (fast, small system):
    uv run python scripts/experiments/p5_nvt_gate_dt1fs.py --smoke --out /tmp/smoke.json

Dry-run (import check only):
    uv run python scripts/experiments/p5_nvt_gate_dt1fs.py --dry-run --out /dev/null
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _make_energy_fn(n_waters, box_vec, displacement_fn, shift_fn):
    from prolix.physics import system
    from tests.physics.test_explicit_langevin_tip3p_parity import _proxide_params_pure_water
    from tests.physics.test_p2b_nvt_216water import _make_tip3p_excl_indices
    import math

    box_edge = float(box_vec[0])
    sys_dict = {k: v for k, v in _proxide_params_pure_water(n_waters).items()
                if k != "exclusion_mask"}
    sys_dict["excl_indices"] = _make_tip3p_excl_indices(n_waters)
    grid = max(16, round(box_edge / 1.0))
    return system.make_energy_fn(
        displacement_fn, sys_dict, box=box_vec, use_pbc=True,
        implicit_solvent=False, pme_grid_points=grid, pme_alpha=0.34,
        cutoff_distance=9.0, strict_parameterization=False,
    )


def run_nvt(n_waters, positions_a, box_vec, seed, steps, burn,
            project_ou=True, remove_com=False):
    """Run NVT and return mean T + T_trans/T_rot decomposition."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from prolix.physics import pbc, settle
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    energy_fn = _make_energy_fn(n_waters, box_vec, displacement_fn, shift_fn)

    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters,
                     dtype=jnp.float64).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)
    dof = 6 * n_waters - 3
    M_water = 15.999 + 1.008 + 1.008

    # dt=1.0 fs (hardcoded for this gate)
    dt_akma = 1.0 / AKMA_TIME_UNIT_FS
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = 10.0 * AKMA_TIME_UNIT_FS * 1e-3

    init_s, apply_s = settle.settle_langevin(
        energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma,
        mass=mass, water_indices=water_indices, box=box_vec,
        remove_linear_com_momentum=remove_com,
        project_ou_momentum_rigid=project_ou,
        projection_site="post_o",
        settle_velocity_iters=10,
    )
    state = init_s(jax.random.key(seed),
                   jnp.array(positions_a, dtype=jnp.float64), mass=mass)
    apply_j = jax.jit(apply_s)
    m_flat = mass.reshape(-1)

    def step_fn(state, _):
        state = apply_j(state)
        ke_tot = jnp.sum(jnp.sum(state.momentum ** 2, axis=-1) / (2.0 * m_flat))
        # COM KE per water (translational)
        p3 = state.momentum.reshape(n_waters, 3, 3)
        v_com = p3.sum(axis=1) / M_water          # (N, 3)
        ke_trans = 0.5 * M_water * jnp.sum(v_com ** 2)
        ke_rot = ke_tot - ke_trans
        return state, (ke_tot, ke_trans, ke_rot)

    _, (kes_tot, kes_trans, kes_rot) = jax.lax.scan(
        step_fn, state, None, length=steps
    )

    prod_tot = kes_tot[burn:]
    prod_trans = kes_trans[burn:]
    prod_rot = kes_rot[burn:]
    kB = BOLTZMANN_KCAL

    dof_trans = 3 * n_waters - 3   # translational minus system COM
    dof_rot = 3 * n_waters         # rotational

    t_tot = float(2.0 * jnp.mean(prod_tot) / (dof * kB))
    t_trans = float(2.0 * jnp.mean(prod_trans) / (dof_trans * kB))
    t_rot = float(2.0 * jnp.mean(prod_rot) / (dof_rot * kB))
    return t_tot, t_trans, t_rot


def main() -> None:
    import jax
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="Fast smoke: 16-water system, 500 steps")
    p.add_argument("--dry-run", action="store_true",
                   help="Import-only check, no computation")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        import jax  # noqa: F401
        from prolix.physics import settle, system  # noqa: F401
        out.write_text(json.dumps({"dry_run": True}))
        print("dry-run ok")
        return

    import jax.numpy as jnp
    import numpy as np
    from tests.physics.test_explicit_langevin_tip3p_parity import _equil_water_positions
    from prolix.physics import pbc

    if args.smoke:
        # Small system for L2 liveness gate
        n_waters = 16
        steps = 500
        burn = 200
        seeds = [42, 43]
    else:
        # Full 895-water production gate
        n_waters = 895
        steps = 50000  # 50 ps at dt=1.0 fs
        burn = 10000   # 10 ps burn-in
        seeds = [42, 43, 44, 45, 46]

    positions_a, box_edge = _equil_water_positions(n_waters, seed=42)
    box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)

    result: dict = {
        "dt_fs": 1.0,
        "gamma_ps": 10.0,
        "n_waters": n_waters,
        "steps": steps,
        "burn": burn,
        "gate_tolerance_k": 5.0,
        "seeds": [],
        "mean_t_rot": None,
        "std_t_rot": None,
        "mean_t_trans": None,
        "std_t_trans": None,
        "mean_t_total": None,
        "std_t_total": None,
    }

    # Multi-seed stability test
    print(f"=== P5 dt=1.0 fs NVT Gate ({n_waters}w, {len(seeds)} seeds) ===")
    t_tots = []
    t_trans_list = []
    t_rot_list = []

    for seed in seeds:
        t_tot, t_trans, t_rot = run_nvt(
            n_waters, positions_a, box_vec, seed=seed,
            steps=steps, burn=burn, project_ou=True, remove_com=False,
        )
        entry = {"seed": seed, "t_total": round(t_tot, 2),
                 "t_trans": round(t_trans, 2), "t_rot": round(t_rot, 2)}
        result["seeds"].append(entry)
        t_tots.append(t_tot)
        t_trans_list.append(t_trans)
        t_rot_list.append(t_rot)
        print(f"  seed={seed}: T_tot={t_tot:.2f} K  (T_trans={t_trans:.1f}  T_rot={t_rot:.1f})")

    result["mean_t_total"] = round(float(np.mean(t_tots)), 2)
    result["std_t_total"] = round(float(np.std(t_tots)), 2)
    result["mean_t_trans"] = round(float(np.mean(t_trans_list)), 2)
    result["std_t_trans"] = round(float(np.std(t_trans_list)), 2)
    result["mean_t_rot"] = round(float(np.mean(t_rot_list)), 2)
    result["std_t_rot"] = round(float(np.std(t_rot_list)), 2)

    print(f"\n  Mean T_rot = {result['mean_t_rot']:.2f} ± {result['std_t_rot']:.2f} K")
    print(f"  Mean T_trans = {result['mean_t_trans']:.2f} ± {result['std_t_trans']:.2f} K")
    print(f"  Mean T_total = {result['mean_t_total']:.2f} ± {result['std_t_total']:.2f} K")

    # Gate pass/fail: T_rot within ±5 K of 300 K
    if args.smoke:
        gate_pass = 1  # smoke just checks the script runs end-to-end
    else:
        gate_pass = int(abs(result["mean_t_rot"] - 300.0) <= 5.0)

    result["gate_pass"] = gate_pass
    result["n_seeds"] = len(seeds)

    out.write_text(json.dumps(result, indent=2))
    print(f"\nGate: T_rot mean={result['mean_t_rot']:.2f} K  pass={gate_pass}")


if __name__ == "__main__":
    main()
