"""P5 NVT system-size sweep at dt=1.0 fs — warm-bias characterization.

Context: the dt=1.0 fs production gate (job 15870804, 895 waters, gamma=10 ps^-1)
passed with mean T_rot = 299.63 K. But pre-removal verification of the
dt=1.0 fs unit-test xfails found a large warm bias at small system size:
n=2 -> ~358-403 K, n=16 -> ~343 K (|dev| 43-103 K vs the +-15 K test tolerance),
at BOTH gamma=1 and gamma=10 ps^-1. Higher friction did not rescue small N,
implicating SYSTEM SIZE rather than friction.

This sweep maps mean T_rot vs n_waters at the gate's exact configuration
(dt=1.0 fs, gamma=10 ps^-1, settle_langevin, project_ou=True, remove_com=False)
to locate the size N* above which |mean T_rot - 300| drops below tolerance.
It reuses the gate's run_nvt() verbatim, so the measurement is identical to the
gate (raw rigid-body KE over dof=6N-3, T_trans/T_rot decomposition) — this also
disentangles real size-dependence from any metric difference vs the unit-test
helper (rigid_tip3p_box_ke_kcal) at the n=2/n=16 test sizes.

One invocation = one system size (for SLURM-array parallelism). The wrapper maps
array task id -> size.

Usage (via bth run, one size):
    uv run bth run python scripts/experiments/p5_nvt_size_sweep_dt1fs.py \\
        --out outputs/p5_size_sweep_n216.json --campaign <id> \\
        -- --n-waters 216 --seeds 42,43,44

Local L2 smoke (fast, tiny system):
    uv run python scripts/experiments/p5_nvt_size_sweep_dt1fs.py --smoke --out /tmp/smoke.json

Dry-run (import check only):
    uv run python scripts/experiments/p5_nvt_size_sweep_dt1fs.py --dry-run --out /dev/null
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reuse the gate's physics verbatim so the measurement is identical to job 15870804.
_GATE_PATH = ROOT / "scripts" / "experiments" / "p5_nvt_gate_dt1fs.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("p5_nvt_gate_dt1fs", _GATE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load gate module from {_GATE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    import jax
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-waters", type=int, default=216,
                   help="System size (single size per invocation).")
    p.add_argument("--seeds", type=str, default="42,43,44",
                   help="Comma-separated PRNG seeds.")
    p.add_argument("--steps", type=int, default=30000,
                   help="Integration steps (30000 = 30 ps at dt=1.0 fs).")
    p.add_argument("--burn", type=int, default=10000,
                   help="Burn-in steps discarded before averaging.")
    p.add_argument("--smoke", action="store_true",
                   help="Fast smoke: n=16, 500 steps, 2 seeds.")
    p.add_argument("--dry-run", action="store_true",
                   help="Import-only check, no computation.")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        from prolix.physics import settle, system  # noqa: F401
        _load_gate_module()  # verify gate import path resolves
        out.write_text(json.dumps({"dry_run": True}))
        print("dry-run ok")
        return

    import numpy as np
    import jax.numpy as jnp
    from tests.physics.test_explicit_langevin_tip3p_parity import _equil_water_positions

    gate = _load_gate_module()

    if args.smoke:
        n_waters, steps, burn, seeds = 16, 500, 200, [42, 43]
    else:
        n_waters, steps, burn = args.n_waters, args.steps, args.burn
        seeds = _parse_seeds(args.seeds)

    positions_a, box_edge = _equil_water_positions(n_waters, seed=42)
    box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)

    result: dict = {
        "dt_fs": 1.0,
        "gamma_ps": 10.0,
        "n_waters": n_waters,
        "steps": steps,
        "burn": burn,
        "seeds": [],
        "mean_t_rot": None, "std_t_rot": None,
        "mean_t_trans": None, "std_t_trans": None,
        "mean_t_total": None, "std_t_total": None,
        "dev_t_rot": None, "within_5k": None, "within_15k": None,
    }

    print(f"=== P5 size sweep dt=1.0 fs (n={n_waters}, {len(seeds)} seeds, "
          f"{steps} steps, burn {burn}) ===")
    t_tots, t_trans_list, t_rot_list = [], [], []
    for seed in seeds:
        t_tot, t_trans, t_rot = gate.run_nvt(
            n_waters, positions_a, box_vec, seed=seed,
            steps=steps, burn=burn, project_ou=True, remove_com=False,
        )
        result["seeds"].append({"seed": seed, "t_total": round(t_tot, 2),
                                "t_trans": round(t_trans, 2), "t_rot": round(t_rot, 2)})
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
    result["dev_t_rot"] = round(abs(result["mean_t_rot"] - 300.0), 2)
    result["within_5k"] = int(result["dev_t_rot"] <= 5.0)
    result["within_15k"] = int(result["dev_t_rot"] <= 15.0)

    print(f"\n  n={n_waters}: Mean T_rot = {result['mean_t_rot']:.2f} "
          f"± {result['std_t_rot']:.2f} K  (|dev|={result['dev_t_rot']:.2f}, "
          f"within_5k={result['within_5k']}, within_15k={result['within_15k']})")

    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
