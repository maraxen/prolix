#!/usr/bin/env python3
"""XR-PARITY-KUPS (#secondary): kUPS thermostat crossval + adapter probe.

Usage::

    uv run bth run python scripts/experiments/xr_parity_kups.py \\
      --tag xr-parity-kups --campaign <id> \\
      --out outputs/xr_parity_kups.json \\
      -- --probe adapter

    # Full BAOAB crossval (requires optional ``kups`` package):
    uv run bth run python scripts/experiments/xr_parity_kups.py \\
      --tag xr-parity-kups --campaign <id> \\
      --out outputs/xr_parity_kups.json \\
      -- --probe crossval --integrator BAOAB --dt-fs 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", choices=("adapter", "crossval", "all"), default="adapter")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--integrator", choices=("BAOAB", "CSVR"), default="BAOAB")
    ap.add_argument("--dt-fs", type=float, default=0.5)
    ap.add_argument("--n-particles", type=int, default=64)
    ap.add_argument("--n-equil", type=int, default=4000, help="Short default for smoke; pytest uses 40k")
    ap.add_argument("--n-sample", type=int, default=6000, help="Short default for smoke; pytest uses 60k")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _probe_adapter() -> dict:
    from prolix.physics import kups_adapter
    from prolix.simulate import AKMA_TIME_UNIT_FS

    checks = []
    checks.append(abs(AKMA_TIME_UNIT_FS - 48.88821291839) < 1e-6)
    checks.append(abs(kups_adapter.EV_TO_KCAL_MOL - 23.060549) < 1e-6)
    for dt_fs in (0.5, 1.0):
        ak = kups_adapter.dt_fs_to_akma(dt_fs)
        checks.append(abs(kups_adapter.dt_akma_to_fs(ak) - dt_fs) < 1e-12)
    for g in (1.0, 10.0):
        ak = kups_adapter.gamma_ps_to_akma(g)
        checks.append(abs(kups_adapter.gamma_akma_to_ps(ak) - g) < 1e-12)
    k_ev = 0.01
    k_kcal = kups_adapter.spring_constant_ev_per_angstrom_sq_to_kcal_mol(k_ev)
    checks.append(abs(k_kcal - k_ev * 23.060549) < 1e-12)
    ok = all(checks)
    return {
        "probe": "adapter",
        "kups_available": 0,
        "adapter_pass": int(ok),
        "gate_pass": int(ok),
        "delta_t_k": None,
        "t_prolix_k": None,
        "t_kups_k": None,
        "integrator": "",
        "dt_fs": None,
        "n_particles": 0,
        "crossval_ran": 0,
        "t_tolerance_k": None,
        "skip_reason": None,
    }


def _probe_crossval(args: argparse.Namespace) -> dict:
    try:
        import kups  # noqa: F401
    except ImportError:
        return {
            "probe": "crossval",
            "kups_available": 0,
            "adapter_pass": 1,
            "gate_pass": 0,
            "delta_t_k": None,
            "t_prolix_k": None,
            "t_kups_k": None,
            "integrator": args.integrator,
            "dt_fs": args.dt_fs,
            "n_particles": args.n_particles,
            "crossval_ran": 0,
            "t_tolerance_k": None,
            "skip_reason": "kups package not installed",
        }

    # Delegate to the pytest module helpers by importing the test file logic
    # via a minimal inline reimplementation of the BAOAB/CSVR path.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kups_crossval",
        _REPO / "tests/physics/test_kups_thermostat_crossval.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Running the full module executes pytest.importorskip at import — ok since we checked.
    spec.loader.exec_module(mod)

    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    n = args.n_particles
    dof = 3 * n
    dt_fs = args.dt_fs
    dt_kups = dt_fs * mod.FEMTO_SECOND

    if args.integrator == "BAOAB":
        state, deriv, _ = mod.create_harmonic_system(
            n_particles=n,
            k=mod.K_EV,
            m=mod.M_AMU,
            kT=mod.KT_EV,
            dt=dt_kups,
            gamma=mod.GAMMA_PS * mod.FEMTO_SECOND,
            tau=mod.TAU_PS * mod.FEMTO_SECOND,
            key=jax.random.PRNGKey(args.seed),
        )
        from kups.md.integrators import make_baoab_langevin_step, euclidean_flow

        integrator = make_baoab_langevin_step(
            particles=mod.SimpleState.particles,
            systems=mod.SimpleState.systems.get,
            derivative_computation=deriv,
            flow=euclidean_flow,
        )
        tol = 2.0
        t_prolix = float(
            mod._run_proxide_harmonic_baoab(
                n_particles=n,
                k_kcal=mod.K_KCAL,
                m_amu=mod.M_AMU,
                kT_kcal=mod.KT_KCAL,
                dt_fs=dt_fs,
                gamma_ps=mod.GAMMA_PS,
                n_equil=args.n_equil,
                n_sample=args.n_sample,
                seed=args.seed,
            )
        )
    else:
        state, deriv, _ = mod.create_harmonic_system(
            n_particles=n,
            k=mod.K_EV,
            m=mod.M_AMU,
            kT=mod.KT_EV,
            dt=dt_kups,
            tau=mod.TAU_PS * mod.FEMTO_SECOND,
            gamma=mod.GAMMA_PS * mod.FEMTO_SECOND,
            key=jax.random.PRNGKey(args.seed),
        )
        from kups.md.integrators import make_csvr_step, euclidean_flow

        integrator = make_csvr_step(
            particles=mod.SimpleState.particles,
            systems=mod.get_systems,
            derivative_computation=deriv,
            flow=euclidean_flow,
        )
        tol = 10.0 if dt_fs >= 1.0 else 2.0
        t_prolix = float(
            mod._run_proxide_harmonic_csvr(
                n_particles=n,
                k_kcal=mod.K_KCAL,
                m_amu=mod.M_AMU,
                kT_kcal=mod.KT_KCAL,
                dt_fs=dt_fs,
                tau_ps=mod.TAU_PS,
                n_equil=args.n_equil,
                n_sample=args.n_sample,
                seed=args.seed,
            )
        )

    _, temps_kups = mod.run_simulation(
        integrator,
        state,
        jax.random.PRNGKey(args.seed),
        n_equil=args.n_equil,
        n_sample=args.n_sample,
        extract_fn=lambda s: mod.compute_temperature(s, dof),
    )
    from kups.core.constants import BOLTZMANN_CONSTANT as KUPS_KB

    t_kups = float(np.mean(temps_kups)) / KUPS_KB
    delta = abs(t_prolix - t_kups)
    ok = delta <= tol and math.isfinite(t_prolix) and math.isfinite(t_kups)
    return {
        "probe": "crossval",
        "kups_available": 1,
        "adapter_pass": 1,
        "gate_pass": int(ok),
        "delta_t_k": delta,
        "t_prolix_k": t_prolix,
        "t_kups_k": t_kups,
        "integrator": args.integrator,
        "dt_fs": dt_fs,
        "n_particles": n,
        "crossval_ran": 1,
        "t_tolerance_k": tol,
    }


def _json_safe(obj: dict) -> dict:
    """Replace NaN/Inf with None for valid JSON + DuckDB outcome eval."""
    out = {}
    for k, v in obj.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def main() -> int:
    args = _parse_args()
    results: list[dict] = []
    if args.probe in ("adapter", "all"):
        results.append(_probe_adapter())
    if args.probe in ("crossval", "all"):
        results.append(_probe_crossval(args))

    primary = results[-1]
    if args.probe == "all":
        adapter = next(r for r in results if r["probe"] == "adapter")
        cross = next(r for r in results if r["probe"] == "crossval")
        if cross["crossval_ran"] == 0:
            gate = int(adapter["adapter_pass"] == 1)
        else:
            gate = int(adapter["adapter_pass"] == 1 and cross["gate_pass"] == 1)
        primary = {
            **cross,
            "probe": "all",
            "adapter_pass": adapter["adapter_pass"],
            "gate_pass": gate,
        }

    primary["campaign_slug"] = "xr-parity-kups"
    primary["seed"] = args.seed
    primary = _json_safe(primary)
    # Omit null optional fields that confuse DuckDB COALESCE on missing columns
    for key in ("t_tolerance_k", "skip_reason"):
        if key not in primary:
            primary[key] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(primary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(primary, indent=2, allow_nan=False))
    return 0 if primary.get("gate_pass", 0) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
