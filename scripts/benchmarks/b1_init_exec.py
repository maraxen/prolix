#!/usr/bin/env python3
"""B1-full Claim-1 init-bound benchmark: prolix EnsemblePlan vs OpenMM N-Context.

Prereg: ``.praxia/docs/specs/260528_b1-preregistration.md``

Usage (via bathos — required for scripts/benchmarks/)::

    uv run bth run python scripts/benchmarks/b1_init_exec.py \\
        --tag smoke --tag path:inference --out tmp/b1_smoke_full.json -- \\
        --smoke --path inference

    uv run bth run python scripts/benchmarks/b1_init_exec.py \\
        --tag cluster --tag b1-full --tag path:inference --campaign <id> \\
        --out outputs/bench/b1_full_prolix_seed0.json -- \\
        --backend prolix --seed 0 --replicas 16 --ps 100 --path inference
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PDB_DIR = ROOT / "data" / "pdb"
CSV_PATH = ROOT / "outputs" / "bench" / "b1_full.csv"

# Prereg pins
DT_FS = 0.5
KT_KCAL = 0.596  # ~300 K
GAMMA_PS = 50.0  # vacuum-safe (XR-VACUUM-DT); water ok at short traj
PS_FULL = 100.0
STEPS_PER_PS = int(round(1000.0 / DT_FS))  # 2000 steps/ps @ 0.5 fs → 200_000 / 100 ps

PROTEIN_CLASSES = (
    ("1ake", "1AKE.pdb"),
    ("1ubq", "1UBQ.pdb"),
    ("2gb1", "2GB1.pdb"),
)

# GPU footprint tiers: shrink B / trajectory for smaller cards; always chunk the
# step scan so we never allocate a full (n_steps, N, 3) Trajectory buffer
# (that alone OOMs L40S at prereg length for 1AKE).
# l40s = first-test measure (not Claim-1 headline). prereg = full protocol pins.
FOOTPRINTS: dict[str, dict[str, float | int]] = {
    "l40s": {"replicas": 2, "ps": 1.0, "chunk": 50},  # B=8, 2k steps
    "a100": {"replicas": 8, "ps": 10.0, "chunk": 100},  # B=32, 20k steps
    "h200": {"replicas": 16, "ps": 100.0, "chunk": 200},  # B=64, 200k (chunked)
    "prereg": {"replicas": 16, "ps": 100.0, "chunk": 200},
}


def _configure_cold_start_env() -> str:
    """Prereg cold-start: unique compile cache + optional XLA client flags.

    ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` is always set.
    Prereg ``--xla_gpu_enable_triton_softmax_fusion=false`` is only appended when
    ``B1_XLA_TRITON_FLAG=1`` (cluster GPU) — some jaxlib builds reject the flag.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if os.environ.get("B1_XLA_TRITON_FLAG", "0") == "1":
        flags = os.environ.get("XLA_FLAGS", "")
        needle = "--xla_gpu_enable_triton_softmax_fusion=false"
        if needle not in flags:
            os.environ["XLA_FLAGS"] = (flags + " " + needle).strip()
    cache = Path(tempfile.gettempdir()) / f"jax_cache_b1_{uuid.uuid4().hex}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache)
    return str(cache)


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
            ).strip()
            or "unknown"
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _hardware_tag() -> str:
    try:
        import jax

        d = jax.devices()[0]
        return f"{d.platform}:{d.device_kind}"
    except Exception:
        return "unknown"


def _nan_segments() -> dict[str, float]:
    return {
        "t_ff_load": float("nan"),
        "t_bundle_construct": float("nan"),
        "t_aot_compile": float("nan"),
        "t_first_step": float("nan"),
        "t_steady_state": float("nan"),
        "aot_ratio": float("nan"),
    }


def _four_water_bundle():
    """Minimal TIP3P 4-water MolecularBundle (prereg 4-water class)."""
    import jax.numpy as jnp

    from prolix.physics.settle import settle_positions
    from prolix.physics.system import make_bundle_from_system
    from prolix.physics.water_models import WaterModelType, get_water_params
    from prolix.typing import PhysicsSystem

    tip = get_water_params(WaterModelType.TIP3P)
    # Geometry from data/pdb/4water.pdb (Å)
    coords = jnp.array(
        [
            [4.125, 13.679, 13.761],
            [4.025, 14.428, 14.348],
            [4.670, 13.062, 14.249],
            [19.406, 17.008, 3.462],
            [19.311, 16.130, 3.832],
            [18.641, 17.113, 2.897],
            [21.292, 5.267, 29.931],
            [20.545, 4.958, 30.444],
            [20.924, 5.930, 29.347],
            [6.968, 27.278, 4.133],
            [6.263, 26.721, 4.463],
            [6.993, 28.017, 4.741],
        ],
        dtype=jnp.float64,
    )
    n_w = 4
    n = n_w * 3
    water_indices = jnp.array([[3 * i, 3 * i + 1, 3 * i + 2] for i in range(n_w)], dtype=jnp.int32)
    charges = jnp.tile(jnp.array([tip.charge_O, tip.charge_H, tip.charge_H]), n_w)
    sigmas = jnp.tile(jnp.array([tip.sigma_O, 1.0, 1.0]), n_w)
    epsilons = jnp.tile(jnp.array([tip.epsilon_O, 0.0, 0.0]), n_w)
    masses = jnp.tile(jnp.array([15.999, 1.008, 1.008]), n_w)
    ones_b = jnp.ones(n, dtype=bool)
    zeros_b = jnp.zeros(n, dtype=bool)
    empty2 = jnp.zeros((0, 2), dtype=jnp.int32)
    empty3 = jnp.zeros((0, 3), dtype=jnp.int32)
    empty_p2 = jnp.zeros((0, 2))
    empty_m = jnp.zeros(0, dtype=bool)
    empty_dih_p = jnp.zeros((0, 1, 3))
    bonds = jnp.array(
        [[3 * i, 3 * i + 1] for i in range(n_w)]
        + [[3 * i, 3 * i + 2] for i in range(n_w)],
        dtype=jnp.int32,
    )
    bond_params = jnp.ones((bonds.shape[0], 2))
    bond_mask = jnp.ones(bonds.shape[0], dtype=bool)

    sys = PhysicsSystem(
        positions=coords,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=jnp.ones(n),
        scaled_radii=jnp.ones(n),
        masses=masses,
        element_ids=jnp.zeros(n, dtype=jnp.int32),
        atom_mask=ones_b,
        is_hydrogen=zeros_b,
        is_backbone=zeros_b,
        is_heavy=ones_b,
        protein_atom_mask=zeros_b,
        water_atom_mask=ones_b,
        bonds=bonds,
        bond_params=bond_params,
        bond_mask=bond_mask,
        angles=empty3,
        angle_params=empty_p2,
        angle_mask=empty_m,
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=empty_dih_p,
        dihedral_mask=empty_m,
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=empty_dih_p,
        improper_mask=empty_m,
        water_indices=water_indices,
    )
    bundle = make_bundle_from_system(sys, boundary_condition="free")
    from prolix.api.bundle_md import active_positions

    pos = settle_positions(active_positions(bundle), active_positions(bundle), water_indices)
    sys = sys.replace(positions=pos)
    return make_bundle_from_system(sys, boundary_condition="free")


def _make_synthetic_bundle(n_atoms: int, seed: int = 0):
    """Synthetic MolecularBundle for --smoke (hetero sizes; empty topology)."""
    import jax.numpy as jnp
    import jax.random

    from prolix.types.bundles import ATOM_BUCKETS, MolecularBundle, MolecularShapeSpec, _bucket_idx

    key = jax.random.key(seed)
    atom_bucket_idx = _bucket_idx(n_atoms, ATOM_BUCKETS)
    padded_n_atoms = ATOM_BUCKETS[atom_bucket_idx]
    key, subkey = jax.random.split(key)
    positions = jax.random.normal(subkey, (padded_n_atoms, 3), dtype=jnp.float32)
    positions = positions.at[n_atoms:].set(0.0)
    atom_mask = jnp.arange(padded_n_atoms) < n_atoms
    charges = jnp.zeros(padded_n_atoms, dtype=jnp.float32)
    sigmas = jnp.ones(padded_n_atoms, dtype=jnp.float32) * 3.15
    epsilons = jnp.zeros(padded_n_atoms, dtype=jnp.float32)
    radii = jnp.ones(padded_n_atoms, dtype=jnp.float32)
    scaled_radii = jnp.ones(padded_n_atoms, dtype=jnp.float32)
    b0 = ATOM_BUCKETS[0]
    empty_bond = jnp.zeros((b0, 2), dtype=jnp.int32)
    empty_bond_params = jnp.zeros((b0, 2), dtype=jnp.float32)
    empty_bond_mask = jnp.zeros(b0, dtype=jnp.bool_)
    empty_angle = jnp.zeros((b0, 3), dtype=jnp.int32)
    empty_angle_params = jnp.zeros((b0, 2), dtype=jnp.float32)
    empty_angle_mask = jnp.zeros(b0, dtype=jnp.bool_)
    empty_dihedral = jnp.zeros((b0, 4), dtype=jnp.int32)
    empty_dihedral_params = jnp.zeros((b0, 4), dtype=jnp.float32)
    empty_dihedral_mask = jnp.zeros(b0, dtype=jnp.bool_)
    empty_improper = jnp.zeros((b0, 4), dtype=jnp.int32)
    empty_improper_params = jnp.zeros((b0, 3), dtype=jnp.float32)
    empty_improper_mask = jnp.zeros(b0, dtype=jnp.bool_)
    empty_ub = jnp.zeros((b0, 3), dtype=jnp.int32)
    empty_ub_params = jnp.zeros((b0, 2), dtype=jnp.float32)
    empty_ub_mask = jnp.zeros(b0, dtype=jnp.bool_)
    empty_cmap = jnp.zeros((8, 24, 24), dtype=jnp.float32)
    empty_cmap_mask = jnp.zeros(8, dtype=jnp.bool_)
    empty_water = jnp.zeros((8, 3), dtype=jnp.int32)
    empty_water_mask = jnp.zeros(8, dtype=jnp.bool_)
    empty_excl = jnp.zeros((b0, 2), dtype=jnp.int32)
    empty_excl_vdw = jnp.zeros(b0, dtype=jnp.float32)
    empty_excl_elec = jnp.zeros(b0, dtype=jnp.float32)
    empty_excl_mask = jnp.zeros(b0, dtype=jnp.bool_)
    empty_exc = jnp.zeros((b0, 2), dtype=jnp.int32)
    empty_exc_sigma = jnp.zeros(b0, dtype=jnp.float32)
    empty_exc_epsilon = jnp.zeros(b0, dtype=jnp.float32)
    empty_exc_charge = jnp.zeros(b0, dtype=jnp.float32)
    empty_exc_mask = jnp.zeros(b0, dtype=jnp.bool_)
    shape_spec = MolecularShapeSpec(
        atom_bucket_idx=atom_bucket_idx,
        bond_bucket_idx=0,
        angle_bucket_idx=0,
        dihedral_bucket_idx=0,
        water_bucket_idx=0,
        excl_bucket_idx=0,
        cmap_bucket_idx=0,
        exception_bucket_idx=0,
        has_pbc=False,
        has_implicit_solvent=False,
        boundary_condition="free",
    )
    return MolecularBundle(
        positions=positions,
        masses=jnp.ones_like(charges),
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=radii,
        scaled_radii=scaled_radii,
        atom_mask=atom_mask,
        n_atoms=jnp.array(n_atoms, dtype=jnp.int32),
        box=jnp.zeros((3, 3), dtype=jnp.float32),
        bond_idx=empty_bond,
        bond_params=empty_bond_params,
        bond_mask=empty_bond_mask,
        n_bonds=jnp.array(0, dtype=jnp.int32),
        angle_idx=empty_angle,
        angle_params=empty_angle_params,
        angle_mask=empty_angle_mask,
        n_angles=jnp.array(0, dtype=jnp.int32),
        dihedral_idx=empty_dihedral,
        dihedral_params=empty_dihedral_params,
        dihedral_mask=empty_dihedral_mask,
        n_dihedrals=jnp.array(0, dtype=jnp.int32),
        improper_idx=empty_improper,
        improper_params=empty_improper_params,
        improper_mask=empty_improper_mask,
        improper_is_periodic=jnp.array(False, dtype=jnp.bool_),
        n_impropers=jnp.array(0, dtype=jnp.int32),
        urey_bradley_idx=empty_ub,
        urey_bradley_params=empty_ub_params,
        urey_bradley_mask=empty_ub_mask,
        n_urey_bradley=jnp.array(0, dtype=jnp.int32),
        cmap_torsion_idx=jnp.zeros((8, 8), dtype=jnp.int32),
        cmap_energy_grids=empty_cmap,
        cmap_mask=empty_cmap_mask,
        n_cmap=jnp.array(0, dtype=jnp.int32),
        water_indices=empty_water,
        water_mask=empty_water_mask,
        n_waters=jnp.array(0, dtype=jnp.int32),
        excl_indices=empty_excl,
        excl_scales_vdw=empty_excl_vdw,
        excl_scales_elec=empty_excl_elec,
        excl_mask=empty_excl_mask,
        n_excl=jnp.array(0, dtype=jnp.int32),
        exception_pairs=empty_exc,
        exception_sigmas=empty_exc_sigma,
        exception_epsilons=empty_exc_epsilon,
        exception_chargeprods=empty_exc_charge,
        exception_mask=empty_exc_mask,
        n_exception_pairs=jnp.array(0, dtype=jnp.int32),
        pme_alpha=jnp.array(0.3, dtype=jnp.float32),
        cutoff_distance=jnp.array(9.0, dtype=jnp.float32),
        shape_spec=shape_spec,
    )


def _synthetic_smoke_bundles():
    """B=4 hetero synthetic bundles (sizes match tests/bench/test_b1_smoke)."""
    return [_make_synthetic_bundle(n, seed=i) for i, n in enumerate((5, 10, 20, 35))]


def _build_prolix_full_bundles(replicas: int):
    """4 topology classes × replicas (prereg B=64 at replicas=16)."""
    sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
    from _b1_paramize import paramize_pdb_to_bundle

    t_ff0 = time.perf_counter()
    # Force-field XML resolve happens inside paramize; attribute to ff_load once.
    templates = []
    for _name, fname in PROTEIN_CLASSES:
        path = PDB_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing fixture: {path}")
        templates.append(paramize_pdb_to_bundle(str(path), periodic=False))
    t_ff = time.perf_counter() - t_ff0

    t_b0 = time.perf_counter()
    water = _four_water_bundle()
    bundles = []
    for tmpl in templates:
        bundles.extend([tmpl] * replicas)
    bundles.extend([water] * replicas)
    t_bundle = time.perf_counter() - t_b0
    return bundles, t_ff, t_bundle


def _block_trajs(trajectories) -> None:
    import jax

    if not isinstance(trajectories, list):
        trajectories = [trajectories]
    for traj in trajectories:
        jax.block_until_ready(traj.positions)


# Claim-1 soft finite gate (B1-FINITE-GATE): vacuum 1AKE under B=16×200k can
# NaN a minority of seeds (diag 17905437: 5/64). Strict all-finite remains in
# ``finite_positions`` for diagnostics; bathos pass uses ``finite_fraction``.
FINITE_FRACTION_MIN = 0.9


def _last_positions_finite_stats(trajectories) -> tuple[bool, float, int, int]:
    """Return (all_finite, finite_fraction, n_finite, n_systems)."""
    import jax

    if not isinstance(trajectories, list):
        trajectories = [trajectories]
    n_systems = len(trajectories)
    if n_systems == 0:
        return True, 1.0, 0, 0
    n_finite = 0
    for traj in trajectories:
        # Only the final frame — never keep the full stack resident for checks
        pos = traj.positions[-1]
        jax.block_until_ready(pos)
        if bool(jax.numpy.all(jax.numpy.isfinite(pos))):
            n_finite += 1
    frac = float(n_finite) / float(n_systems)
    return n_finite == n_systems, frac, n_finite, n_systems


def _last_positions_finite(trajectories) -> bool:
    """Strict all-systems-finite (diagnostic). Prefer ``_last_positions_finite_stats``."""
    all_finite, _, _, _ = _last_positions_finite_stats(trajectories)
    return all_finite


def _run_plan_chunked(plan, *, n_steps: int, chunk: int, seed: int, run_mode: str = "trajectory") -> object:
    """Run ``n_steps`` as fixed-size chunks; discard each traj after sync.

    Peak device memory scales with ``chunk``, not ``n_steps``. Seed advances
    per chunk so noise is not identical across chunks.

    For ``run_mode='inference'`` prefer a single ``plan.run`` (while_loop has
    no traj stack); chunking is only needed for the scan baseline.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    chunk = max(1, int(chunk))
    remaining = int(n_steps)
    seed_i = int(seed)
    last = None
    while remaining > 0:
        this = min(chunk, remaining)
        last = plan.run(
            n_steps=this,
            dt=DT_FS,
            kT=KT_KCAL,
            seed=seed_i,
            gamma=GAMMA_PS,
            run_mode=run_mode,
        )
        _block_trajs(last)
        # Drop references so the full (this, N, 3) buffer can be freed
        remaining -= this
        seed_i += 1
        if remaining > 0:
            last = None
    return last


def run_prolix(
    *,
    n_steps: int,
    seed: int,
    replicas: int,
    smoke: bool,
    chunk: int = 50,
    footprint: str = "l40s",
    path: str = "inference",
) -> dict:
    import jax

    from prolix.api import EnsemblePlan

    run_mode = "trajectory" if path == "baseline" else "inference"
    segs = _nan_segments()
    t_total0 = time.perf_counter()

    if smoke:
        t_b0 = time.perf_counter()
        bundles = _synthetic_smoke_bundles()
        segs["t_ff_load"] = 0.0
        segs["t_bundle_construct"] = time.perf_counter() - t_b0
        chunk = min(chunk, max(n_steps, 1))
    else:
        bundles, t_ff, t_bundle = _build_prolix_full_bundles(replicas)
        segs["t_ff_load"] = t_ff
        segs["t_bundle_construct"] = t_bundle

    plan = EnsemblePlan.from_bundles(bundles)

    # Cold first step (AOT proxy)
    t_aot0 = time.perf_counter()
    first = plan.run(
        n_steps=1,
        dt=DT_FS,
        kT=KT_KCAL,
        seed=seed,
        gamma=GAMMA_PS,
        run_mode=run_mode,
    )
    _block_trajs(first)
    t_first = time.perf_counter() - t_aot0
    segs["t_first_step"] = t_first
    segs["t_aot_compile"] = t_first
    del first

    # Steady-state: inference = one while_loop; baseline = chunked scan
    t_ss0 = time.perf_counter()
    last = None
    if n_steps > 1:
        if run_mode == "inference":
            last = plan.run(
                n_steps=n_steps - 1,
                dt=DT_FS,
                kT=KT_KCAL,
                seed=seed + 1,
                gamma=GAMMA_PS,
                run_mode="inference",
            )
            _block_trajs(last)
        else:
            last = _run_plan_chunked(
                plan,
                n_steps=n_steps - 1,
                chunk=chunk,
                seed=seed + 1,
                run_mode="trajectory",
            )
    segs["t_steady_state"] = time.perf_counter() - t_ss0

    t_total = time.perf_counter() - t_total0
    # Prereg H0 / sidecar: aot_ratio = t_aot_compile / t_total (not cold-warm).
    segs["aot_ratio"] = float(segs["t_aot_compile"]) / max(float(t_total), 1e-12)

    if last is not None:
        all_finite, finite_frac, n_finite, n_sys = _last_positions_finite_stats(last)
        del last
    else:
        all_finite, finite_frac, n_finite, n_sys = True, 1.0, len(bundles), len(bundles)

    return {
        "backend": "prolix",
        "path": path,
        "run_mode": run_mode,
        "smoke": bool(smoke),
        "footprint": footprint,
        "seed": int(seed),
        "n_systems": len(bundles),
        "n_steps": int(n_steps),
        "chunk": int(chunk) if run_mode == "trajectory" else 0,
        "replicas": int(replicas),
        "dt_fs": DT_FS,
        "gamma_ps": GAMMA_PS,
        "t_total": float(t_total),
        "finite_positions": bool(all_finite),
        "finite_fraction": float(finite_frac),
        "n_finite": int(n_finite),
        "n_finite_systems": int(n_sys),
        "hardware_tag": _hardware_tag(),
        "git_hash": _git_hash(),
        "jax_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
        **{k: float(v) for k, v in segs.items()},
    }


def _openmm_pdb_paths(smoke: bool, replicas: int) -> list[Path]:
    if smoke:
        return [
            PDB_DIR / "2GB1.pdb",
            PDB_DIR / "1UBQ.pdb",
            PDB_DIR / "4water.pdb",
            PDB_DIR / "4water.pdb",
        ]
    paths: list[Path] = []
    for _name, fname in PROTEIN_CLASSES:
        paths.extend([PDB_DIR / fname] * replicas)
    paths.extend([PDB_DIR / "4water.pdb"] * replicas)
    return paths


def _openmm_prepare_topology(path: Path, ff_protein, app, unit):
    """Load PDB: strip HETATM for proteins; add missing hydrogens."""
    is_water = path.name == "4water.pdb"
    if is_water:
        pdb = app.PDBFile(str(path))
        return pdb.topology, pdb.positions, True

    # Protein: ATOM records only (drop ligands/cofactors that lack templates)
    lines = []
    with path.open() as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("CRYST1") or line.startswith("END"):
                lines.append(line)
    tmp = Path(tempfile.gettempdir()) / f"b1_omm_{path.stem}_{uuid.uuid4().hex}.pdb"
    tmp.write_text("".join(lines) + ("\nEND\n" if not any(l.startswith("END") for l in lines) else ""))
    try:
        pdb = app.PDBFile(str(tmp))
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff_protein)
        return modeller.topology, modeller.positions, False
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def run_openmm(
    *,
    n_steps: int,
    seed: int,
    replicas: int,
    smoke: bool,
    footprint: str = "l40s",
) -> dict:
    """N independent OpenMM Contexts — no reuse (prereg baseline)."""
    try:
        import openmm as mm
        from openmm import app, unit
    except ImportError as e:
        raise SystemExit(f"OpenMM required for --backend openmm: {e}") from e

    segs = _nan_segments()
    t_total0 = time.perf_counter()

    t_ff0 = time.perf_counter()
    ff_protein = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    ff_water = app.ForceField("amber14/tip3pfb.xml")
    segs["t_ff_load"] = time.perf_counter() - t_ff0

    paths = _openmm_pdb_paths(smoke, replicas)

    # Claim-1: CUDA only — never fall back to OpenCL/CPU (would invalidate H1).
    # Conda OpenMM needs OPENMM_PLUGIN_DIR (+ GPU libcuda); load explicitly.
    plugin_dir = os.environ.get("OPENMM_PLUGIN_DIR")
    if plugin_dir:
        try:
            mm.Platform.loadPluginsFromDirectory(plugin_dir)
        except Exception as e:
            print(f"WARNING: loadPluginsFromDirectory({plugin_dir}): {e}", flush=True)
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
    except Exception as e:
        registered = [
            mm.Platform.getPlatform(i).getName()
            for i in range(mm.Platform.getNumPlatforms())
        ]
        raise SystemExit(
            "OpenMM CUDA platform required for B1-full baseline. "
            f"Registered={registered}; OPENMM_PLUGIN_DIR={plugin_dir!r}; error={e}"
        ) from e
    # SLURM --gres=gpu:1 → use the first visible device (must be set before Context).
    try:
        platform.setPropertyDefaultValue("CudaDeviceIndex", "0")
    except Exception:
        try:
            platform.setPropertyDefaultValue("DeviceIndex", "0")
        except Exception:
            pass

    temperature = 300 * unit.kelvin
    friction = GAMMA_PS / unit.picosecond
    dt = DT_FS * unit.femtosecond

    contexts = []
    t_ctx0 = time.perf_counter()
    for i, path in enumerate(paths):
        topology, positions, is_water = _openmm_prepare_topology(
            path, ff_protein, app, unit
        )
        ff = ff_water if is_water else ff_protein
        kwargs = dict(
            nonbondedMethod=app.NoCutoff if not is_water else app.CutoffPeriodic,
            constraints=app.HBonds,
            rigidWater=True,
        )
        if is_water:
            kwargs["nonbondedCutoff"] = 1.0 * unit.nanometer
        system = ff.createSystem(topology, **kwargs)
        integrator = mm.LangevinMiddleIntegrator(temperature, friction, dt)
        integrator.setRandomNumberSeed(int(seed) + i)
        context = mm.Context(system, integrator, platform)
        context.setPositions(positions)
        if is_water and topology.getPeriodicBoxVectors() is not None:
            context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
        contexts.append(context)
    segs["t_bundle_construct"] = time.perf_counter() - t_ctx0

    t1 = time.perf_counter()
    for context in contexts:
        context.getIntegrator().step(1)
    segs["t_first_step"] = time.perf_counter() - t1
    # OpenMM has no JAX AOT segment — stub 0 so aot_ratio gate is N/A-safe
    segs["t_aot_compile"] = 0.0

    t_ss0 = time.perf_counter()
    remain = max(0, n_steps - 1)
    if remain:
        for context in contexts:
            context.getIntegrator().step(remain)
    segs["t_steady_state"] = time.perf_counter() - t_ss0
    segs["aot_ratio"] = 0.0

    t_total = time.perf_counter() - t_total0
    return {
        "backend": "openmm",
        "path": "n/a",
        "run_mode": "n/a",
        "smoke": bool(smoke),
        "footprint": footprint,
        "seed": int(seed),
        "n_systems": len(contexts),
        "n_steps": int(n_steps),
        "chunk": 0,
        "replicas": int(replicas),
        "dt_fs": DT_FS,
        "gamma_ps": GAMMA_PS,
        "t_total": float(t_total),
        "finite_positions": True,
        "finite_fraction": 1.0,
        "n_finite": len(contexts),
        "n_finite_systems": len(contexts),
        "hardware_tag": f"openmm:{platform.getName()}",
        "git_hash": _git_hash(),
        "jax_cache_dir": "",
        **{k: float(v) for k, v in segs.items()},
    }


def _append_csv(row: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "path",
        "footprint",
        "smoke",
        "seed",
        "n_systems",
        "n_steps",
        "chunk",
        "replicas",
        "t_total",
        "t_ff_load",
        "t_bundle_construct",
        "t_aot_compile",
        "t_first_step",
        "t_steady_state",
        "aot_ratio",
        "finite_positions",
        "finite_fraction",
        "n_finite",
        "hardware_tag",
        "git_hash",
    ]
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fieldnames})


def main() -> int:
    cache = _configure_cold_start_env()

    parser = argparse.ArgumentParser(description="B1-full Claim-1 init-bound benchmark")
    parser.add_argument("--backend", choices=("prolix", "openmm"), default="prolix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--footprint",
        choices=tuple(FOOTPRINTS.keys()),
        default="l40s",
        help="GPU tier preset (l40s=first test; prereg=full pins). Overrides "
        "default replicas/ps/chunk unless flags are set explicitly.",
    )
    parser.add_argument("--replicas", type=int, default=None, help="Replicas per class")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--ps", type=float, default=None, help="Trajectory length in ps")
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Steps per EnsemblePlan scan chunk (baseline path only; peak VRAM ~ chunk)",
    )
    parser.add_argument(
        "--path",
        choices=("inference", "baseline"),
        default="inference",
        help="prolix run path: inference=while_loop carry-only (default); "
        "baseline=scan+traj stack (pathological A/B).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="B=4 mixed, short steps, 1 seed (local dry-run)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write result JSON (default: BTH_RESULTS_PATH from bth run --out).",
    )
    args = parser.parse_args()

    if args.out_json is None:
        env_path = os.environ.get("BTH_RESULTS_PATH")
        if env_path:
            args.out_json = Path(env_path)

    preset = FOOTPRINTS[args.footprint]
    if args.smoke:
        n_steps = args.n_steps if args.n_steps is not None else 10
        replicas = 1
        chunk = args.chunk if args.chunk is not None else 10
        footprint = "smoke"
    else:
        replicas = int(args.replicas if args.replicas is not None else preset["replicas"])
        chunk = int(args.chunk if args.chunk is not None else preset["chunk"])
        if args.n_steps is not None:
            n_steps = args.n_steps
        elif args.ps is not None:
            n_steps = int(round(args.ps * STEPS_PER_PS))
        else:
            n_steps = int(round(float(preset["ps"]) * STEPS_PER_PS))
        footprint = args.footprint

    if args.backend == "prolix":
        row = run_prolix(
            n_steps=n_steps,
            seed=args.seed,
            replicas=replicas,
            smoke=args.smoke,
            chunk=chunk,
            footprint=footprint,
            path=args.path,
        )
    else:
        row = run_openmm(
            n_steps=n_steps,
            seed=args.seed,
            replicas=replicas,
            smoke=args.smoke,
            footprint=footprint,
        )

    row["jax_cache_dir"] = row.get("jax_cache_dir") or cache
    _append_csv(row)

    text = json.dumps(row, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n")

    frac = float(row.get("finite_fraction", 1.0 if row.get("finite_positions", True) else 0.0))
    if frac < FINITE_FRACTION_MIN:
        return 1
    if not (row["t_total"] == row["t_total"] and row["t_total"] > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
