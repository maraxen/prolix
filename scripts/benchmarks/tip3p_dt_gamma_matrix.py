#!/usr/bin/env python3
"""Run a small (dt_fs × gamma_ps) matrix: OpenMM vs Prolix rigid mean T (tip3p_ke_compare path).

Uses the same PME box and ``settle_langevin`` / LangevinMiddle pair as ``tip3p_ke_compare.py``.
For each (dt, γ) cell, prints one row with replicate-mean rigid temperatures and |ΔT̄|.

Example (local, needs OpenMM)::

  JAX_ENABLE_X64=1 uv run --extra openmm python scripts/benchmarks/tip3p_dt_gamma_matrix.py \\
    --dt-list 1,2 --gamma-list 0.5,1,2 --n-waters 8 --replicas 3 --steps 8000 --burn 2000
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import tip3p_ke_compare as tkc

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))


def _parse_float_list(s: str) -> list[float]:
  return [float(x.strip()) for x in s.split(",") if x.strip()]


def _one_cell(
  *,
  dt_fs: float,
  gamma_ps: float,
  n_waters: int,
  steps: int,
  burn: int,
  sample_every: int,
  seed: int,
  replicas: int,
  remove_cmmotion: bool,
  openmm_integrator: str,
  timing_mode: str,
  warmup_steps: int,
  measure_steps: int,
  temperature_k: float,
  projection_site: str,
  project_ou_momentum_rigid: bool,
  settle_velocity_iters: int,
  verbose_samples: bool,
) -> dict:
  omm_reps: list[dict] = []
  plx_reps: list[dict] = []
  for rid in range(replicas):
    s = seed + rid * 100_003
    omm_reps.append(
      tkc._openmm_run_once(
        n_waters=n_waters,
        n_steps=steps,
        burn=burn,
        sample_every=sample_every,
        seed=s,
        timing_mode=timing_mode,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        dt_fs=dt_fs,
        temperature_k=temperature_k,
        gamma_ps=gamma_ps,
        remove_cmmotion=remove_cmmotion,
        openmm_integrator=openmm_integrator,
        verbose_samples=verbose_samples,
        akma_time_unit_fs=tkc._AKMA_TIME_UNIT_FS,
      )
    )
    plx_reps.append(
      tkc._prolix_run_once(
        n_waters=n_waters,
        n_steps=steps,
        burn=burn,
        sample_every=sample_every,
        seed=s,
        timing_mode=timing_mode,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        dt_fs=dt_fs,
        temperature_k=temperature_k,
        gamma_ps=gamma_ps,
        remove_linear_com_momentum=remove_cmmotion,
        verbose_samples=verbose_samples,
        akma_time_unit_fs=tkc._AKMA_TIME_UNIT_FS,
        projection_site=projection_site,
        project_ou_momentum_rigid=project_ou_momentum_rigid,
        settle_velocity_iters=settle_velocity_iters,
      )
    )
  sum_o = tkc._summarize_engine_runs(omm_reps, sample_every=sample_every, dt_fs=dt_fs)
  sum_p = tkc._summarize_engine_runs(plx_reps, sample_every=sample_every, dt_fs=dt_fs)
  t_o = float((sum_o.get("replicate_temperature") or {}).get("mean", float("nan")))
  t_p = float((sum_p.get("replicate_temperature") or {}).get("mean", float("nan")))
  sem_o = float((sum_o.get("replicate_temperature") or {}).get("sem", float("nan")))
  sem_p = float((sum_p.get("replicate_temperature") or {}).get("sem", float("nan")))
  return {
    "dt_fs": float(dt_fs),
    "gamma_ps": float(gamma_ps),
    "openmm": {
      "replicate_mean_T_K": t_o,
      "replicate_sem_T_K": sem_o,
      "n_replicas": int(sum_o.get("n_replicas", 0)),
    },
    "prolix": {
      "replicate_mean_T_K": t_p,
      "replicate_sem_T_K": sem_p,
      "n_replicas": int(sum_p.get("n_replicas", 0)),
    },
    "abs_delta_replicate_mean_T_K": float(abs(t_o - t_p)),
    "gamma_dt_check": tkc._gamma_dt_consistency(
      gamma_ps=gamma_ps, dt_fs=dt_fs, akma_time_unit_fs=tkc._AKMA_TIME_UNIT_FS
    ),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--dt-list", type=str, default="1,2", help="Comma-separated dt in fs (e.g. 1,2).")
  ap.add_argument("--gamma-list", type=str, default="0.5,1,2", help="Comma-separated γ in 1/ps.")
  ap.add_argument("--n-waters", type=int, default=8)
  ap.add_argument("--steps", type=int, default=12_000)
  ap.add_argument("--burn", type=int, default=4_000)
  ap.add_argument("--sample-every", type=int, default=20)
  ap.add_argument("--seed", type=int, default=4242)
  ap.add_argument("--replicas", type=int, default=3)
  ap.add_argument("--temperature-k", type=float, default=300.0)
  ap.add_argument(
    "--remove-cmmotion",
    choices=("true", "false"),
    default="false",
    help="OpenMM createSystem + Prolix COM removal (default false: baseline policy).",
  )
  ap.add_argument("--openmm-integrator", choices=("middle", "langevin"), default="middle")
  ap.add_argument("--timing-mode", choices=("cold", "steady", "both"), default="both")
  ap.add_argument("--warmup-steps", type=int, default=100)
  ap.add_argument("--measure-steps", type=int, default=500)
  ap.add_argument("--verbose-samples", action="store_true")
  ap.add_argument("--jax-x64", choices=("off", "on"), default="on")
  ap.add_argument("--projection-site", choices=("post_o", "post_settle_vel", "both"), default="post_settle_vel")
  ap.add_argument("--settle-velocity-iters", type=int, default=20)
  ap.add_argument("--project-ou-momentum-rigid", choices=("true", "false"), default="true")
  args = ap.parse_args()
  if args.projection_site == "both" and args.project_ou_momentum_rigid != "true":
    print("error: projection_site=both requires --project-ou-momentum-rigid true", file=sys.stderr)
    return 2
  tkc._configure_jax(args.jax_x64)
  dts = _parse_float_list(args.dt_list)
  gams = _parse_float_list(args.gamma_list)
  remove_cm = args.remove_cmmotion == "true"
  proj_ou = args.project_ou_momentum_rigid == "true"
  rows: list[dict] = []
  for dt in dts:
    for g in gams:
      rows.append(
        _one_cell(
          dt_fs=dt,
          gamma_ps=g,
          n_waters=int(args.n_waters),
          steps=int(args.steps),
          burn=int(args.burn),
          sample_every=int(args.sample_every),
          seed=int(args.seed),
          replicas=int(args.replicas),
          remove_cmmotion=remove_cm,
          openmm_integrator=str(args.openmm_integrator),
          timing_mode=str(args.timing_mode),
          warmup_steps=int(args.warmup_steps),
          measure_steps=int(args.measure_steps),
          temperature_k=float(args.temperature_k),
          projection_site=str(args.projection_site),
          project_ou_momentum_rigid=proj_ou,
          settle_velocity_iters=int(args.settle_velocity_iters),
          verbose_samples=bool(args.verbose_samples),
        )
      )
  out = {
    "schema": "tip3p_dt_gamma_matrix/v1",
    "meta": {
      "script": "tip3p_dt_gamma_matrix.py",
      "python": platform.python_version(),
      "n_waters": int(args.n_waters),
      "steps": int(args.steps),
      "burn": int(args.burn),
      "sample_every": int(args.sample_every),
      "replicas": int(args.replicas),
      "profile_id": tkc.tip3p_ke_profile.profile_id_from_remove_cmmotion(remove_cm),
      "prolix_projection_site": str(args.projection_site),
      "prolix_settle_velocity_iters": int(args.settle_velocity_iters),
      "prolix_project_ou_momentum_rigid": bool(proj_ou),
    },
    "rows": rows,
  }
  print(json.dumps(out, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
