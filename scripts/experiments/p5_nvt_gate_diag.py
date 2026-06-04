"""P5 NVT gate diagnostic — multi-seed stability + OU-mode comparison.

Answers three questions in one cluster run:
  Q1  Statistical stability: is the 293.83 K result systematic or a seed fluke?
      Runs 5 seeds with the current config (constrained OU, no COM removal).
  Q2  DOF decomposition: which DOF are underpowered — T_trans or T_rot?
      One seed with T_trans/T_rot breakdown.
  Q3  Projection hypothesis: does unconstrained OU give T closer to 300 K?
      Constrained vs unconstrained OU at same seed/config.

Usage (via bth run):
    uv run bth run python scripts/experiments/p5_nvt_gate_diag.py \\
        --out outputs/p5_nvt_gate_diag.json

Local L2 smoke (fast, small system):
    uv run python scripts/experiments/p5_nvt_gate_diag.py --smoke --out /tmp/smoke.json

Dry-run (import check only):
    uv run python scripts/experiments/p5_nvt_gate_diag.py --dry-run --out /dev/null
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

    dt_akma = 0.5 / AKMA_TIME_UNIT_FS
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
                   help="Fast smoke: 8-water system, 300 steps")
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
        # Q1 only, tiny system
        n_waters = 8
        steps = 300
        burn = 100
        seeds = [42, 43]
    else:
        n_waters = 895
        steps = 3000
        burn = 1000
        seeds = [42, 43, 44, 45, 46]

    positions_a, box_edge = _equil_water_positions(n_waters, seed=42)
    box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)

    result: dict = {
        "n_waters": n_waters,
        "steps": steps,
        "burn": burn,
        "gate_tolerance_k": 5.0,
        "q1_stability": [],
        "q2_decomp": {},
        "q3_ou_comparison": {},
    }

    # Q1 — multi-seed stability (constrained OU, no COM removal)
    print(f"=== Q1: Statistical stability ({n_waters}w, {len(seeds)} seeds) ===")
    for seed in seeds:
        t_tot, t_trans, t_rot = run_nvt(
            n_waters, positions_a, box_vec, seed=seed,
            steps=steps, burn=burn, project_ou=True, remove_com=False,
        )
        entry = {"seed": seed, "T_total": round(t_tot, 2),
                 "T_trans": round(t_trans, 2), "T_rot": round(t_rot, 2)}
        result["q1_stability"].append(entry)
        print(f"  seed={seed}: T={t_tot:.2f} K  (trans={t_trans:.1f}  rot={t_rot:.1f})")

    t_vals = [e["T_total"] for e in result["q1_stability"]]
    result["q1_mean_t"] = round(float(np.mean(t_vals)), 2)
    result["q1_std_t"] = round(float(np.std(t_vals)), 2)
    print(f"  mean={result['q1_mean_t']:.2f} ± {result['q1_std_t']:.2f} K")

    # Q2 — DOF decomposition at first seed (already have it from Q1 seed 42)
    q1_seed42 = next(e for e in result["q1_stability"] if e["seed"] == 42)
    result["q2_decomp"] = {
        "seed": 42, "T_total": q1_seed42["T_total"],
        "T_trans": q1_seed42["T_trans"], "T_rot": q1_seed42["T_rot"],
    }
    print(f"\n=== Q2: DOF decomp (seed=42) ===")
    print(f"  T_total={q1_seed42['T_total']:.2f}  T_trans={q1_seed42['T_trans']:.1f}"
          f"  T_rot={q1_seed42['T_rot']:.1f}")

    # Q3 — constrained vs unconstrained OU (seed=42)
    print(f"\n=== Q3: Constrained vs unconstrained OU ===")
    t_con, _, _ = run_nvt(n_waters, positions_a, box_vec, seed=42,
                          steps=steps, burn=burn, project_ou=True, remove_com=False)
    t_unc, _, _ = run_nvt(n_waters, positions_a, box_vec, seed=42,
                          steps=steps, burn=burn, project_ou=False, remove_com=False)
    result["q3_ou_comparison"] = {
        "constrained_T": round(t_con, 2),
        "unconstrained_T": round(t_unc, 2),
        "delta_K": round(t_con - t_unc, 2),
    }
    print(f"  constrained: {t_con:.2f} K  unconstrained: {t_unc:.2f} K"
          f"  delta={t_con-t_unc:+.2f} K")

    # Gate pass/fail (only meaningful at n_waters=895; smoke uses small system)
    if args.smoke:
        gate_pass = 1  # smoke just checks the script runs end-to-end
        gate_error = 0.0
    else:
        gate_pass = int(abs(result["q1_mean_t"] - 300.0) < 5.0)
        gate_error = round(abs(result["q1_mean_t"] - 300.0), 2)
    result["gate_pass"] = gate_pass
    result["gate_error_k"] = gate_error

    out.write_text(json.dumps(result, indent=2))
    print(f"\ngate_pass={gate_pass}  mean T={result['q1_mean_t']:.2f} K"
          f"  error={gate_error:.2f} K")


if __name__ == "__main__":
    main()
