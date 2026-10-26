#!/usr/bin/env python3
"""Full EnsemblePlan inference step timing (Campaign E: full-md-path-matrix).

Times ``EnsemblePlan.run(..., run_mode="inference")`` via an n_steps
regression (same slope/intercept split as bench_dhfr_parity). OOM is a
valid JSON ``status=oom`` with exit 0 — do not skip the cell.

Ungated science: this sidecar only checks harness quality, not NL vs flash.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prof_ensemble_step")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _git_sha() -> str | None:
    env = os.environ.get("PROLIX_GIT_SHA", "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "resource_exhausted",
        "out of memory",
        "oom",
        "cudaerror",
        "failed to allocate",
        "insufficient memory",
    )
    return any(n in text for n in needles)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def _base_payload(args: argparse.Namespace) -> dict:
    return {
        "system": args.system,
        "nonbonded": args.nonbonded,
        "use_size_buckets": args.use_size_buckets == "on",
        "remat": args.remat == "on",
        "dtype": args.dtype,
        "n_real": None,
        "n_padded": None,
        "nl_k_max": None,
        "per_step_corrected_s": None,
        "compile_fixed_s": None,
        "r_squared": None,
        "status": "error",
        "git_sha": _git_sha(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "probe": args.probe,
        "init_neighbor_bound": None,
        "k_over_n": None,
        "rebuild_every": args.nl_update_every,
    }


def _build_1vii(use_size_buckets: bool, dtype):
    import jax.numpy as jnp
    from proxide import CoordFormat, OutputSpec, parse_structure

    from prolix.physics.solvation import solvate_protein_to_bundle
    from prolix.physics.water_models import WaterModelType

    import proxide

    ff = Path(proxide.__file__).parent / "assets" / "protein.ff19SB.xml"
    pdb = ROOT / "data" / "pdb" / "1VII.pdb"
    spec = OutputSpec(parameterize_md=True, force_field=str(ff), coord_format=CoordFormat.Full)
    protein = parse_structure(str(pdb), spec)
    bundle = solvate_protein_to_bundle(
        protein,
        padding=8.0,
        model_type=WaterModelType.TIP3P,
        ionic_strength=0.0,
        neutralize=True,
        target_box_size=jnp.array([32.0, 32.0, 32.0], dtype=dtype),
        use_size_buckets=use_size_buckets,
    )
    return bundle


def _build_dhfr(use_size_buckets: bool):
    from scripts.experiments.bench_dhfr_parity import (
        _build_dhfr_system_dict,
        _resolve_ff_path,
    )

    from prolix.physics.system import make_bundle_from_system

    sys_data = _build_dhfr_system_dict(_resolve_ff_path())
    # _build_bundle_from_sys_data always buckets; rebuild with flag.
    sys_dict, positions, box_vec, water_indices, masses, exclusion_spec, _n = sys_data
    from types import SimpleNamespace

    import jax.numpy as jnp

    work_dtype = jnp.asarray(positions).dtype
    ns = SimpleNamespace(
        positions=jnp.asarray(positions, dtype=work_dtype),
        box_size=jnp.asarray(box_vec, dtype=work_dtype),
        masses=jnp.asarray(masses, dtype=work_dtype),
        charges=jnp.asarray(sys_dict["charges"], dtype=work_dtype),
        sigmas=jnp.asarray(sys_dict["sigmas"], dtype=work_dtype),
        epsilons=jnp.asarray(sys_dict["epsilons"], dtype=work_dtype),
        bonds=sys_dict["bonds"],
        bond_params=sys_dict["bond_params"],
        angles=sys_dict["angles"],
        angle_params=sys_dict["angle_params"],
        dihedrals=sys_dict["dihedrals"],
        dihedral_params=sys_dict["dihedral_params"],
        impropers=sys_dict["impropers"],
        improper_params=sys_dict["improper_params"],
        water_indices=water_indices,
        pme_alpha=0.34,
        nonbonded_cutoff=9.0,
    )
    return make_bundle_from_system(
        ns,
        boundary_condition="periodic",
        exclusion_spec=exclusion_spec,
        use_size_buckets=use_size_buckets,
    )


def _patch_flash_remat(remat: bool) -> None:
    import prolix.physics.flash_explicit as fe

    orig = fe.flash_explicit_forces

    def _wrapped(sys, *args, **kwargs):
        kwargs.setdefault("remat", remat)
        return orig(sys, *args, **kwargs)

    fe.flash_explicit_forces = _wrapped


def _run_kwargs(nonbonded: str) -> dict:
    if nonbonded == "flash":
        return {"use_flash_forces": True, "use_neighbor_list": False}
    if nonbonded == "nl":
        return {"use_flash_forces": False, "use_neighbor_list": True}
    return {"use_flash_forces": False, "use_neighbor_list": False}


def _nl_k_max(bundle) -> int | None:
    from prolix.api.bundle_md import _host_float, displacement_fn_for_bundle, positions_with_prefix
    from prolix.physics import neighbor_list as nl_mod
    import jax.numpy as jnp

    displacement_fn, _ = displacement_fn_for_bundle(bundle)
    box_vec = jnp.diag(bundle.box)
    cutoff = _host_float(bundle.cutoff_distance, 9.0)
    neighbor_fn = nl_mod.make_neighbor_list_fn(displacement_fn, box_vec, cutoff)
    prefix = int(bundle.positions.shape[0])
    nbr = neighbor_fn.allocate(positions_with_prefix(bundle, prefix))
    return int(nbr.idx.shape[-1])


def _regress(bundle, n_steps_list, seed, dt, kT, gamma, run_kw, nl_update_every):
    import jax

    from prolix.api import EnsemblePlan

    def _block(traj) -> None:
        trajs = traj if isinstance(traj, list) else [traj]
        for t in trajs:
            jax.block_until_ready(t.positions)

    extra = {}
    if run_kw.get("use_neighbor_list"):
        extra["nl_update_every"] = nl_update_every

    results = []
    for n_steps in n_steps_list:
        plan = EnsemblePlan.from_bundle(bundle)
        t0 = time.perf_counter()
        first = plan.run(
            n_steps=1, dt=dt, kT=kT, seed=seed, gamma=gamma, run_mode="inference", **run_kw, **extra
        )
        _block(first)
        t_first = time.perf_counter() - t0
        del first
        t_ss = 0.0
        if n_steps > 1:
            t1 = time.perf_counter()
            last = plan.run(
                n_steps=n_steps - 1,
                dt=dt,
                kT=kT,
                seed=seed + 1,
                gamma=gamma,
                run_mode="inference",
                **run_kw,
                **extra,
            )
            _block(last)
            t_ss = time.perf_counter() - t1
            del last
        results.append({"n_steps": n_steps, "t_first_step": t_first, "t_steady_state": t_ss})
        log.info("n_steps=%d t_first=%.4fs t_ss=%.4fs", n_steps, t_first, t_ss)

    x_vals = [r["n_steps"] - 1 for r in results if r["n_steps"] > 1]
    y_vals = [r["t_steady_state"] for r in results if r["n_steps"] > 1]
    slope = intercept = r_squared = float("nan")
    if len(x_vals) >= 2:
        coeff = np.polyfit(x_vals, y_vals, 1)
        slope, intercept = float(coeff[0]), float(coeff[1])
        y_hat = slope * np.asarray(x_vals) + intercept
        ss_res = float(np.sum((np.asarray(y_vals) - y_hat) ** 2))
        ss_tot = float(np.sum((np.asarray(y_vals) - np.mean(y_vals)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r_squared, results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", choices=("1vii", "dhfr"), required=True)
    p.add_argument("--nonbonded", choices=("dense", "flash", "nl"), required=True)
    p.add_argument("--use-size-buckets", choices=("on", "off"), default="on")
    p.add_argument("--remat", choices=("on", "off"), default="on")
    p.add_argument("--dtype", choices=("f32", "f64"), default="f32")
    p.add_argument("--n-steps-list", default="2,4,8")
    p.add_argument("--dt-fs", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nl-update-every", type=int, default=20)
    p.add_argument(
        "--probe",
        choices=("none", "nl-stats"),
        default="none",
        help="nl-stats: also record K vs N for leftover-dense investigation.",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    payload = _base_payload(args)
    n_steps_list = sorted({int(x) for x in args.n_steps_list.split(",")})

    if args.dtype == "f64" or args.system == "dhfr":
        import jax

        jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp

    from prolix.simulate import BOLTZMANN_KCAL

    dtype = jnp.float64 if args.dtype == "f64" or args.system == "dhfr" else jnp.float32
    kT = 300.0 * BOLTZMANN_KCAL

    try:
        if args.nonbonded == "flash":
            _patch_flash_remat(args.remat == "on")
        if args.system == "1vii":
            bundle = _build_1vii(args.use_size_buckets == "on", dtype)
        else:
            bundle = _build_dhfr(args.use_size_buckets == "on")
        payload["n_real"] = int(jnp.asarray(bundle.n_atoms))
        payload["n_padded"] = int(bundle.positions.shape[0])
        if args.nonbonded == "nl" or args.probe == "nl-stats":
            k_max = _nl_k_max(bundle)
            payload["nl_k_max"] = k_max
            n_pad = payload["n_padded"] or 1
            payload["k_over_n"] = None if k_max is None else float(k_max) / float(n_pad)

        run_kw = _run_kwargs(args.nonbonded)
        slope, intercept, r_squared, _rows = _regress(
            bundle,
            n_steps_list,
            args.seed,
            args.dt_fs,
            kT,
            args.gamma,
            run_kw,
            args.nl_update_every,
        )
        payload["per_step_corrected_s"] = slope
        payload["compile_fixed_s"] = intercept
        payload["r_squared"] = r_squared
        payload["status"] = "ok"
        payload["init_neighbor_bound"] = args.nonbonded == "nl"
        _write(args.out, payload)
        return 0
    except Exception as exc:
        payload["n_real"] = payload["n_real"] if payload["n_real"] is not None else 0
        payload["n_padded"] = payload["n_padded"] if payload["n_padded"] is not None else 0
        payload["nl_k_max"] = payload["nl_k_max"] if payload["nl_k_max"] is not None else 0
        payload["k_over_n"] = payload["k_over_n"] if payload["k_over_n"] is not None else 0.0
        payload["init_neighbor_bound"] = bool(payload["init_neighbor_bound"])
        if payload["per_step_corrected_s"] is None:
            payload["per_step_corrected_s"] = float("nan")
        if payload["compile_fixed_s"] is None:
            payload["compile_fixed_s"] = float("nan")
        if payload["r_squared"] is None:
            payload["r_squared"] = float("nan")
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["status"] = "oom" if _is_oom(exc) else "error"
        log.exception("cell failed status=%s", payload["status"])
        _write(args.out, payload)
        return 0 if payload["status"] == "oom" else 1


if __name__ == "__main__":
    raise SystemExit(main())
