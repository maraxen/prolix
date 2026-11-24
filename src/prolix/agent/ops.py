"""Pure operation functions for the prolix agent CLI and MCP server (spec section 4).

One function per tool. No cisternal import; heavy imports (jax, prolix.api, proxide) stay inside
bodies. Each in-process op runs on CPU only (``assert_cpu``), writes side files and then a run
record ``<tool>.json`` (spec section 5), and refuses before any write. ``bundle=`` injects a
pre-built bundle (tests and library callers); injected bundles skip the PDB checks and caches.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from prolix.agent import _runtime as rt
from prolix.agent.errors import InputError, RefusalError
from prolix.agent.system import DEFAULT_FF, count_pdb_atoms, load_system_bundle, resolve_ff_path

INSPECT_TERMS = ("bond", "angle", "dihedral", "improper", "charge", "lj")
INSPECT_LIMIT_MAX = 200
NO_H_WARNING = "no hydrogens: energies are for a heavy-atom model"
NVT_DEFAULT_TEMPERATURE_K = 300.0
NVE_INIT_TEMPERATURE_K = 300.0  # run_nve hard-codes 300 K initial velocities (A29)
_CACHE_SIZE = 2

_cache_lock = threading.Lock()
_BUNDLE_CACHE: OrderedDict[tuple[str, str, str], Any] = OrderedDict()
_CLOSURE_CACHE: OrderedDict[tuple[str, str, str], Any] = OrderedDict()


# ---------------------------------------------------------------- paths, inputs, caches


def _out_root() -> Path:
  raw = os.environ.get("PROLIX_AGENT_OUT_ROOT", "")
  return Path(raw).resolve() if raw else Path.cwd().resolve()


def _resolve_out_dir(out_dir: str | os.PathLike[str]) -> Path:
  """Resolve ``out_dir`` (symlinks followed) and require it under the allowed root."""
  target = Path(out_dir).expanduser().resolve()
  root = _out_root()
  if target != root and root not in target.parents:
    raise InputError(
      f"out_dir {target} is outside the allowed root {root} (set PROLIX_AGENT_OUT_ROOT)"
    )
  return target


def _check_out(out_dir: str | os.PathLike[str], tool: str, overwrite: bool) -> Path:
  target = _resolve_out_dir(out_dir)
  record = target / f"{tool}.json"
  if record.exists() and not overwrite:
    raise InputError(f"{record} exists; pass overwrite=true to replace it")
  return target


def _check_pdb(pdb: str | None) -> Path:
  if pdb is None:
    raise InputError("pdb is required when no bundle is injected")
  path = Path(pdb)
  if path.suffix.lower() != ".pdb":
    raise InputError(f"{pdb}: expected a .pdb file")
  if not path.is_file():
    raise InputError(f"{pdb}: no such file")
  return path


def _check_ff(ff: str) -> str:
  try:
    return resolve_ff_path(ff)
  except FileNotFoundError as exc:
    raise InputError(str(exc)) from exc


def _check_caps_atoms(n_real: int, source: str) -> None:
  if n_real > rt.MAX_ATOMS:
    raise RefusalError(
      f"{source} has {n_real} atoms; the in-process cap is {rt.MAX_ATOMS} (MAX_ATOMS). "
      "Use render_job to run a larger system as a job."
    )


def _check_steps(steps: int, cap: int, name: str) -> int:
  if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
    raise InputError(f"{name} must be a positive integer, got {steps!r}")
  if steps > cap:
    raise RefusalError(f"{name}={steps} is over the cap {cap}; use render_job for longer runs")
  return steps


def _check_dt(dt_fs: float) -> float:
  if not isinstance(dt_fs, (int, float)) or not math.isfinite(dt_fs) or dt_fs <= 0:
    raise InputError(f"dt_fs must be a positive finite number, got {dt_fs!r}")
  if dt_fs > rt.MAX_DT_FS:
    raise InputError(f"dt_fs={dt_fs} is above the cap {rt.MAX_DT_FS} fs")
  return float(dt_fs)


def _bundle_key(pdb_path: Path, ff: str, ff_path: str) -> tuple[str, str, str]:
  return (rt.sha256_file(pdb_path), ff, rt.sha256_file(ff_path))


def _validate_loaded(bundle: Any) -> None:
  """Refuse a bundle with all-zero bond parameters (the ``MolecularBundle.from_pdb`` failure mode)."""
  import jax.numpy as jnp

  mask = np.asarray(bundle.bond_mask, dtype=bool)
  params = np.asarray(bundle.bond_params)
  if mask.any() and not np.any(params[mask] != 0):
    raise InputError(
      "loaded bundle has all-zero bond parameters (the from_pdb failure mode); "
      "the loader did not apply a force field"
    )
  if int(jnp.asarray(bundle.n_atoms)) == 0:
    raise InputError("loaded bundle has no real atoms")


def _get_bundle(
  pdb: str | None, ff: str, bundle: Any | None
) -> tuple[Any, str, tuple[str, str, str] | None, str | None, int | None, list[str]]:
  """Return ``(bundle, bundle_cache, key, ff_path, n_hydrogen, warnings)``. Loads through the cache."""
  if bundle is not None:
    _check_caps_atoms(int(np.asarray(bundle.n_atoms)), "injected bundle")
    return bundle, "injected", None, None, None, []
  pdb_path = _check_pdb(pdb)
  n_records, n_h = count_pdb_atoms(pdb_path)
  _check_caps_atoms(n_records, "pdb")
  ff_path = _check_ff(ff)
  key = _bundle_key(pdb_path, ff, ff_path)
  with _cache_lock:
    if key in _BUNDLE_CACHE:
      _BUNDLE_CACHE.move_to_end(key)
      loaded, status = _BUNDLE_CACHE[key], "hit"
    else:
      loaded, status = None, "miss"
  if loaded is None:
    loaded = load_system_bundle(str(pdb_path), ff)
    _validate_loaded(loaded)
    with _cache_lock:
      _BUNDLE_CACHE[key] = loaded
      _BUNDLE_CACHE.move_to_end(key)
      while len(_BUNDLE_CACHE) > _CACHE_SIZE:
        _BUNDLE_CACHE.popitem(last=False)
  n_real = int(np.asarray(loaded.n_atoms))
  _check_caps_atoms(n_real, "loaded system")
  warnings = [NO_H_WARNING] if n_h == 0 else []
  return loaded, status, key, ff_path, n_h, warnings


def _energy_fn(bundle: Any, key: tuple[str, str, str] | None) -> Any:
  from prolix.api.bundle_md import energy_fn_from_bundle

  if key is None:
    return energy_fn_from_bundle(bundle)
  with _cache_lock:
    if key in _CLOSURE_CACHE:
      _CLOSURE_CACHE.move_to_end(key)
      return _CLOSURE_CACHE[key]
  fn = energy_fn_from_bundle(bundle)
  with _cache_lock:
    _CLOSURE_CACHE[key] = fn
    _CLOSURE_CACHE.move_to_end(key)
    while len(_CLOSURE_CACHE) > _CACHE_SIZE:
      _CLOSURE_CACHE.popitem(last=False)
  return fn


def _real_positions(bundle: Any) -> Any:
  from prolix.api.bundle_md import active_positions

  return active_positions(bundle)


def _real_masses(bundle: Any) -> Any:
  from prolix.api.bundle_md import masses_for_bundle

  return masses_for_bundle(bundle)


def _finite_energy(value: float, what: str) -> None:
  if not math.isfinite(value):
    raise InputError(f"non-finite input energy ({what}={value}); the input structure is not usable")


# ---------------------------------------------------------------- run records and side files


def _git_state() -> dict[str, Any] | None:
  repo = Path(__file__).resolve().parents[3]
  git = shutil.which("git")
  if git is None or not (repo / ".git").exists():
    return None
  try:
    rev = subprocess.run(  # noqa: S603 - fixed argv, no shell
      [git, "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=20, check=True
    ).stdout.strip()
    dirty = bool(
      subprocess.run(  # noqa: S603 - fixed argv, no shell
        [git, "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
      ).stdout.strip()
    )
  except (OSError, subprocess.SubprocessError):
    return None
  return {"rev": rev, "dirty": dirty}


def _jax_state() -> dict[str, Any]:
  import jax

  return {
    "backend": jax.default_backend(),
    "x64": bool(getattr(jax.config, "jax_enable_x64", False)),
    "devices": [str(d) for d in jax.devices()],
    "pallas": bool(os.environ.get("PROLIX_USE_PALLAS", "")),
  }


def _jax_cache_state() -> dict[str, Any]:
  try:
    from xtrax import jax_cache

    setting = jax_cache.current()
  except Exception:  # the record must still be written when xtrax is absent
    setting = None
  return setting.as_record() if setting is not None else {"configured": False}


def _sha_bytes_file(path: Path) -> dict[str, Any]:
  return {"path": str(path), "sha256": rt.sha256_file(path), "bytes": path.stat().st_size}


def _save_side_npz(folder: Path, name: str, **arrays: np.ndarray) -> Path:
  """Write ``<name>.npz`` through ``<name>.tmp.npz`` and ``os.replace`` (spec section 4)."""
  folder.mkdir(parents=True, exist_ok=True)
  tmp = folder / f"{name}.tmp.npz"
  final = folder / f"{name}.npz"
  np.savez(tmp, **arrays)
  os.replace(tmp, final)
  return final


def _write_record(
  folder: Path,
  tool: str,
  *,
  started: float,
  started_utc: str,
  args: dict[str, Any],
  inputs: list[dict[str, Any]],
  outputs: list[dict[str, Any]],
  result: dict[str, Any],
) -> Path:
  record = {
    "schema": rt.RECORD_SCHEMA,
    "tool": tool,
    "started_utc": started_utc,
    "wall_s": round(time.perf_counter() - started, 3),
    "args": args,
    "caps": {
      "MAX_ATOMS": rt.MAX_ATOMS,
      "MAX_MIN_STEPS": rt.MAX_MIN_STEPS,
      "MAX_MD_STEPS": rt.MAX_MD_STEPS,
      "MAX_DT_FS": rt.MAX_DT_FS,
    },
    "inputs": inputs,
    "outputs": outputs,
    "versions": rt.versions(),
    "jax": _jax_state(),
    "jax_cache": _jax_cache_state(),
    "git": _git_state(),
    "result": result,
  }
  target = folder / f"{tool}.json"
  rt.atomic_write_bytes(target, rt.dumps_record(record))
  return target


def _common_result(
  bundle: Any, bundle_cache: str, n_h: int | None, warnings: list[str], ff_path: str | None
) -> dict[str, Any]:
  return {
    "n_real_atoms": int(np.asarray(bundle.n_atoms)),
    "n_hydrogen": n_h,
    "warnings": list(warnings),
    "bundle_cache": bundle_cache,
    "ff_path": ff_path,
  }


def _inputs(pdb: str | None, ff_path: str | None) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  if pdb is not None and Path(pdb).is_file():
    out.append(_sha_bytes_file(Path(pdb)))
  if ff_path is not None:
    out.append(_sha_bytes_file(Path(ff_path)))
  return out


# ---------------------------------------------------------------- operations


def load_system(
  *,
  out_dir: str,
  pdb: str | None = None,
  ff: str = DEFAULT_FF,
  overwrite: bool = False,
  bundle: Any | None = None,
) -> dict[str, Any]:
  """Parameterise and validate a PDB; write ``load_system.json``."""
  rt.assert_cpu()
  started, started_utc = time.perf_counter(), _dt.datetime.now(_dt.UTC).isoformat()
  folder = _check_out(out_dir, "load_system", overwrite)
  loaded, cache, _key, ff_path, n_h, warnings = _get_bundle(pdb, ff, bundle)
  import jax.numpy as jnp

  def mask(name: str) -> np.ndarray:
    return np.asarray(getattr(loaded, name), dtype=bool)

  n_real = int(np.asarray(loaded.n_atoms))
  result = _common_result(loaded, cache, n_h, warnings, ff_path)
  result.update(
    {
      "padded_atoms": int(loaded.positions.shape[0]),
      "atom_bucket": int(loaded.positions.shape[0]),
      "term_counts": {
        "bonds": int(mask("bond_mask").sum()),
        "angles": int(mask("angle_mask").sum()),
        "dihedrals": int(mask("dihedral_mask").sum()),
        "impropers": int(mask("improper_mask").sum()),
      },
      "charge_sum": float(jnp.sum(jnp.asarray(loaded.charges)[:n_real])),
      "has_pbc": False,
      "boundary": "free",
    }
  )
  record = _write_record(
    folder,
    "load_system",
    started=started,
    started_utc=started_utc,
    args={"pdb": pdb, "ff": ff, "overwrite": overwrite},
    inputs=_inputs(pdb, ff_path),
    outputs=[],
    result=result,
  )
  return {"record": str(record), "result": result}


def inspect_params(
  *,
  out_dir: str,
  term: str,
  limit: int = 50,
  pdb: str | None = None,
  ff: str = DEFAULT_FF,
  overwrite: bool = False,
  bundle: Any | None = None,
) -> dict[str, Any]:
  """Parameter rows for one term; ``limit`` is at most 200 (``truncated`` says whether rows were cut)."""
  rt.assert_cpu()
  started, started_utc = time.perf_counter(), _dt.datetime.now(_dt.UTC).isoformat()
  if term not in INSPECT_TERMS:
    raise InputError(f"term must be one of {', '.join(INSPECT_TERMS)}, got {term!r}")
  if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= INSPECT_LIMIT_MAX:
    raise InputError(f"limit must be an integer in 1..{INSPECT_LIMIT_MAX}, got {limit!r}")
  folder = _check_out(out_dir, "inspect_params", overwrite)
  loaded, cache, _key, ff_path, n_h, warnings = _get_bundle(pdb, ff, bundle)
  n_real = int(np.asarray(loaded.n_atoms))

  rows: list[dict[str, Any]] = []
  if term in ("bond", "angle", "dihedral", "improper"):
    spec = {
      "bond": ("bond_mask", "bond_idx", "bond_params"),
      "angle": ("angle_mask", "angle_idx", "angle_params"),
      "dihedral": ("dihedral_mask", "dihedral_idx", "dihedral_params"),
      "improper": ("improper_mask", "improper_idx", "improper_params"),
    }[term]
    m = np.asarray(getattr(loaded, spec[0]), dtype=bool)
    idx = np.asarray(getattr(loaded, spec[1]))
    par = np.asarray(getattr(loaded, spec[2]))
    for i in np.flatnonzero(m):
      rows.append({"atoms": [int(a) for a in idx[i]], "params": [float(p) for p in par[i]]})
  elif term == "charge":
    q = np.asarray(loaded.charges)
    rows = [{"atom": int(i), "charge": float(q[i])} for i in range(n_real)]
  else:  # lj
    sig = np.asarray(loaded.sigmas)
    eps = np.asarray(loaded.epsilons)
    rows = [
      {"atom": int(i), "sigma": float(sig[i]), "epsilon": float(eps[i])} for i in range(n_real)
    ]

  total = len(rows)
  shown = rows[:limit]
  result = _common_result(loaded, cache, n_h, warnings, ff_path)
  result.update({"term": term, "rows": shown, "total_rows": total, "truncated": total > limit})
  record = _write_record(
    folder,
    "inspect_params",
    started=started,
    started_utc=started_utc,
    args={"term": term, "limit": limit, "pdb": pdb, "ff": ff, "overwrite": overwrite},
    inputs=_inputs(pdb, ff_path),
    outputs=[],
    result=result,
  )
  return {"record": str(record), "result": result}


def energy_breakdown(
  *,
  out_dir: str,
  pdb: str | None = None,
  ff: str = DEFAULT_FF,
  overwrite: bool = False,
  bundle: Any | None = None,
) -> dict[str, Any]:
  """Total, bonded and force statistics on real atoms, dense path only (no neighbor list).

  ``bonded`` is ``bonded_energy_fn_from_bundle``: bond + angle + dihedral + improper terms. It
  has no LJ, Coulomb or 1-4 terms, and no CMAP term (``api/bundle_md.py``, ``_bonded_energy_core``).
  ``nonbonded_by_difference`` is total - bonded, so it includes CMAP, LJ, Coulomb and 1-4.
  """
  rt.assert_cpu()
  started, started_utc = time.perf_counter(), _dt.datetime.now(_dt.UTC).isoformat()
  folder = _check_out(out_dir, "energy_breakdown", overwrite)
  loaded, cache, key, ff_path, n_h, warnings = _get_bundle(pdb, ff, bundle)
  import jax
  import jax.numpy as jnp

  from prolix.api.bundle_md import bonded_energy_fn_from_bundle

  x0 = _real_positions(loaded)
  _check_caps_atoms(int(x0.shape[0]), "system")
  total_fn = _energy_fn(loaded, key)
  bonded_fn = bonded_energy_fn_from_bundle(loaded)
  e_total = float(total_fn(x0))
  _finite_energy(e_total, "total")
  e_bonded = float(bonded_fn(x0))
  forces = -jax.grad(total_fn)(x0)
  norms = np.asarray(jnp.linalg.norm(forces, axis=-1))
  result = _common_result(loaded, cache, n_h, warnings, ff_path)
  result.update(
    {
      "total": e_total,
      "bonded": e_bonded,
      "nonbonded_by_difference": e_total - e_bonded,
      "median_force": float(np.median(norms)),
      "max_force": float(np.max(norms)),
      "force_units": "kcal/mol/A",
    }
  )
  record = _write_record(
    folder,
    "energy_breakdown",
    started=started,
    started_utc=started_utc,
    args={"pdb": pdb, "ff": ff, "overwrite": overwrite},
    inputs=_inputs(pdb, ff_path),
    outputs=[],
    result=result,
  )
  return {"record": str(record), "result": result}


def minimize(
  *,
  out_dir: str,
  sd_steps: int = 50,
  pdb: str | None = None,
  ff: str = DEFAULT_FF,
  overwrite: bool = False,
  bundle: Any | None = None,
) -> dict[str, Any]:
  """Bounded steepest descent. Refuses unless the energy after is <= the energy before.

  ``run_minimization`` ignores ``steps``, ``dt_start`` and FIRE (A4), so only ``sd_steps`` runs.
  """
  rt.assert_cpu()
  started, started_utc = time.perf_counter(), _dt.datetime.now(_dt.UTC).isoformat()
  n = _check_steps(sd_steps, rt.MAX_MIN_STEPS, "sd_steps")
  folder = _check_out(out_dir, "minimize", overwrite)
  loaded, cache, key, ff_path, n_h, warnings = _get_bundle(pdb, ff, bundle)
  from prolix.physics.simulate import run_minimization

  x0 = _real_positions(loaded)
  ef = _energy_fn(loaded, key)
  e0 = float(ef(x0))
  _finite_energy(e0, "before")
  x1 = run_minimization(ef, x0, sd_steps=n)
  x1_np = np.asarray(x1)
  e1 = float(ef(x1))
  if not (math.isfinite(e1) and e1 <= e0):
    raise RefusalError(
      f"minimization did not lower the energy (before={e0}, after={e1}); no result written"
    )
  out_path = _save_side_npz(folder, "minimized", positions=x1_np)
  result = _common_result(loaded, cache, n_h, warnings, ff_path)
  result.update({"energy_before": e0, "energy_after": e1, "sd_steps": n})
  record = _write_record(
    folder,
    "minimize",
    started=started,
    started_utc=started_utc,
    args={"sd_steps": n, "pdb": pdb, "ff": ff, "overwrite": overwrite},
    inputs=_inputs(pdb, ff_path),
    outputs=[_sha_bytes_file(out_path)],
    result=result,
  )
  return {"record": str(record), "result": result}


def md_smoke(
  *,
  out_dir: str,
  steps: int = 100,
  dt_fs: float = 1.0,
  ensemble: str = "nve",
  temperature_k: float | None = None,
  pdb: str | None = None,
  ff: str = DEFAULT_FF,
  overwrite: bool = False,
  bundle: Any | None = None,
) -> dict[str, Any]:
  """Short NVE or NVT Nose-Hoover run from the input structure, with a fixed seed 0.

  Gives a finiteness and energy check only; no conservation claim. ``temperature_k`` is NVT only:
  ``run_nve`` hard-codes 300 K velocities (A29). Masses are the bundle's (``run_nve`` defaults
  to unit masses otherwise).
  """
  rt.assert_cpu()
  started, started_utc = time.perf_counter(), _dt.datetime.now(_dt.UTC).isoformat()
  if ensemble not in ("nve", "nvt"):
    raise InputError(f"ensemble must be 'nve' or 'nvt', got {ensemble!r}")
  if ensemble == "nve" and temperature_k is not None:
    raise InputError("temperature_k applies to ensemble='nvt' only (run_nve fixes 300 K)")
  if ensemble == "nvt":
    temperature_k = NVT_DEFAULT_TEMPERATURE_K if temperature_k is None else float(temperature_k)
    if not math.isfinite(temperature_k) or temperature_k <= 0:
      raise InputError(f"temperature_k must be positive, got {temperature_k!r}")
  n = _check_steps(steps, rt.MAX_MD_STEPS, "steps")
  dt = _check_dt(dt_fs)
  folder = _check_out(out_dir, "md_smoke", overwrite)
  loaded, cache, key, ff_path, n_h, warnings = _get_bundle(pdb, ff, bundle)
  from prolix.physics.simulate import run_nve, run_nvt_nose_hoover
  from prolix.simulate import AKMA_TIME_UNIT_FS

  x0 = _real_positions(loaded)
  masses = _real_masses(loaded)
  ef = _energy_fn(loaded, key)
  e_start = float(ef(x0))
  _finite_energy(e_start, "start")
  dt_akma = dt / AKMA_TIME_UNIT_FS
  if ensemble == "nve":
    final = run_nve(ef, x0, n, dt_akma, mass=masses)
  else:
    final = run_nvt_nose_hoover(
      ef,
      x0,
      n,
      dt_akma,
      temperature=NVT_DEFAULT_TEMPERATURE_K if temperature_k is None else float(temperature_k),
      mass=masses,
    )
  final_np = np.asarray(final)
  finite = bool(np.all(np.isfinite(final_np)))
  e_end = float(ef(final)) if finite else float("nan")
  out_path = _save_side_npz(folder, "final", positions=final_np)
  result = _common_result(loaded, cache, n_h, warnings, ff_path)
  result.update(
    {
      "finite": finite,
      "energy_start": e_start,
      "energy_end": e_end,
      "steps": n,
      "dt_fs": dt,
      "dt_akma": dt_akma,
      "ensemble": ensemble,
      "seed": "fixed 0",
      "temperature_k": temperature_k if ensemble == "nvt" else None,
      "init_velocity_temperature_k": NVE_INIT_TEMPERATURE_K if ensemble == "nve" else None,
      "conservation": "not claimed",
    }
  )
  record = _write_record(
    folder,
    "md_smoke",
    started=started,
    started_utc=started_utc,
    args={
      "steps": n,
      "dt_fs": dt,
      "ensemble": ensemble,
      "temperature_k": temperature_k,
      "pdb": pdb,
      "ff": ff,
      "overwrite": overwrite,
    },
    inputs=_inputs(pdb, ff_path),
    outputs=[_sha_bytes_file(out_path)],
    result=result,
  )
  return {"record": str(record), "result": result}


__all__ = [
  "INSPECT_TERMS",
  "energy_breakdown",
  "inspect_params",
  "load_system",
  "md_smoke",
  "minimize",
]
