"""Scope A bench: TorchMD YamlForcefield — forward+backward of bonded energy
across N mols (mixed sizes), one Adam step.

This is the TorchMD entry to the §7.1 external-baseline campaign. Each comparator
(bench_dmff.py, bench_torchmd.py, bench_espaloma.py, bench_prolix.py)
emits a JSON row with the same schema, and the campaign analysis joins on
(tool, n_mols, precision) to produce the comparison table.

Usage:
    uv run python scripts/benchmarks/external_baseline/bench_torchmd.py \\
        --n-mols 64 --n-conf-cap 100 --precision float32 \\
        --n-warmup 1 --n-trials 3 --out results.json

Emits a single-line JSON dict matching the result_schema in
bench_torchmd.bth.toml.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent
        ).decode().strip()
    except Exception:
        return "unknown"


def _device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _peak_memory_mib() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
    return -1.0


def _torchmd_version() -> str:
    try:
        import torchmd
        return getattr(torchmd, "__version__", "unknown")
    except Exception:
        return "unknown"


def _load_mol_json(json_path: Path) -> dict:
    """Load a prolix mol_*.params_init.json without importing prolix."""
    with open(json_path) as f:
        return json.load(f)


def _build_yaml_forcefield(mol_data: dict, mol_id: str, tmp_dir: str) -> str:
    """Build a TorchMD YAML forcefield from a prolix params_init JSON dict.

    TorchMD YAML bonds/angles use the same units as prolix JSON — no conversion
    needed (kcal/mol/Å² for bonds, kcal/mol/rad² for angles, Å for req,
    degrees for theta0).
    """
    n_atoms = len(mol_data["atom_types"])
    atomtypes = [f"A{i}" for i in range(n_atoms)]

    bonds_yaml: dict = {}
    for b in mol_data.get("bonds", []):
        key = f"({atomtypes[b['i']]}, {atomtypes[b['j']]})"
        bonds_yaml[key] = {"k0": float(b["k"]), "req": float(b["r0"])}

    angles_yaml: dict = {}
    for a in mol_data.get("angles", []):
        key = f"({atomtypes[a['i']]}, {atomtypes[a['j']]}, {atomtypes[a['k']]})"
        angles_yaml[key] = {
            "k0": float(a["k_theta"]),
            "theta0": float(a["theta0_deg"]),
        }

    dihedrals_yaml: dict = {}
    for t in mol_data.get("proper_torsions", []):
        key = (
            f"({atomtypes[t['i']]}, {atomtypes[t['j']]},"
            f" {atomtypes[t['k']]}, {atomtypes[t['l']]})"
        )
        terms = []
        for period, phase, k_phi in zip(
            t["periodicity"], t["phase_deg"], t["k_phi"]
        ):
            terms.append({
                "phi_k": float(k_phi),
                "phase": float(phase),
                "per": int(period),
            })
        dihedrals_yaml[key] = {"terms": terms}

    ff_dict: dict = {}
    if bonds_yaml:
        ff_dict["bonds"] = bonds_yaml
    if angles_yaml:
        ff_dict["angles"] = angles_yaml
    if dihedrals_yaml:
        ff_dict["dihedrals"] = dihedrals_yaml

    yaml_path = os.path.join(tmp_dir, f"{mol_id}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(ff_dict, f, default_flow_style=False)
    return yaml_path


def _build_torchmd_objects(
    mol_data: dict,
    mol_id: str,
    tmp_dir: str,
    dtype: torch.dtype,
    device: str,
) -> tuple:
    """Build (Forces, pos_tensor) for one molecule.

    Returns (forces_obj, pos_tensor, mol_id) or raises if TorchMD rejects the
    molecule (e.g. no supported terms).
    """
    from torchmd.forcefields.forcefield import ForceField
    from moleculekit.molecule import Molecule
    from torchmd.parameters import Parameters
    from torchmd.forces import Forces

    n_atoms = len(mol_data["atom_types"])
    atomtypes = [f"A{i}" for i in range(n_atoms)]

    _ATOMIC_MASS = {1:1.008,6:12.011,7:14.007,8:15.999,9:18.998,15:30.974,16:32.06,17:35.45,35:79.904,53:126.904}
    mol = Molecule()
    mol.empty(n_atoms)
    mol.atomtype[:] = atomtypes
    mol.masses = np.array([_ATOMIC_MASS.get(a["Z"], 12.0) for a in mol_data["atom_types"]], dtype=np.float32)

    bonds_list = mol_data.get("bonds", [])
    if bonds_list:
        mol.bonds = np.array([[b["i"], b["j"]] for b in bonds_list], dtype=np.int32)
    else:
        mol.bonds = np.zeros((0, 2), dtype=np.int32)

    angles_list = mol_data.get("angles", [])
    if angles_list:
        mol.angles = np.array(
            [[a["i"], a["j"], a["k"]] for a in angles_list], dtype=np.int32
        )
    else:
        mol.angles = np.zeros((0, 3), dtype=np.int32)

    torsions_list = mol_data.get("proper_torsions", [])
    if torsions_list:
        mol.dihedrals = np.array(
            [[t["i"], t["j"], t["k"], t["l"]] for t in torsions_list],
            dtype=np.int32,
        )
    else:
        mol.dihedrals = np.zeros((0, 4), dtype=np.int32)

    yaml_path = _build_yaml_forcefield(mol_data, mol_id, tmp_dir)
    ff = ForceField.create(mol, yaml_path)
    terms = []
    if bonds_list:
        terms.append("bonds")
    if angles_list:
        terms.append("angles")
    if torsions_list:
        terms.append("dihedrals")
    if not terms:
        raise ValueError(f"mol {mol_id}: no bonded terms — skipping")

    parameters = Parameters(ff, mol, terms=terms, precision=dtype)
    forces_obj = Forces(parameters, terms=terms)

    # Use first conformer only (n_conf_cap=1 for timing)
    # positions from params_init JSON are not stored there — use random
    # positions as placeholder (we are measuring compute cost, not accuracy)
    pos_np = np.random.default_rng(42).random((n_atoms, 3)).astype(np.float32) * 5.0
    pos_tensor = torch.tensor(pos_np, dtype=dtype, device=device, requires_grad=True)

    return forces_obj, pos_tensor, mol_id


def load_base_mols(subset_dir: Path, n_conf_cap: int | None = None) -> list[dict]:
    """Load the base mols from data/ani1x_subset/lane_a/ as raw dicts."""
    lane_a = subset_dir / "lane_a"
    mol_files = sorted(lane_a.glob("mol_*.params_init.json"))
    mols = []
    for f in mol_files:
        data = _load_mol_json(f)
        n_atoms = len(data["atom_types"])
        mols.append({
            "json_data": data,
            "mol_id": f.stem.replace(".params_init", ""),
            "n_atoms": n_atoms,
        })
    return mols


def bench_one_step(args) -> dict:
    """Run Scope A primitive: warmup + n_trials, return JSON row."""
    dtype = torch.float32 if args.precision == "float32" else torch.float64
    device = _device_str()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    base_mols = load_base_mols(Path(args.subset_dir), n_conf_cap=args.n_conf_cap)
    print(f"loaded {len(base_mols)} base mols")

    n_base = len(base_mols)
    tile_idx = [i % n_base for i in range(args.n_mols)]

    tmp_dir = tempfile.mkdtemp()
    mol_data_list = []  # list of (forces_obj, pos_tensor, mol_id)
    n_atoms_max = 0
    skipped = 0
    for i in tile_idx:
        m = base_mols[i]
        try:
            forces_obj, pos_tensor, mol_id = _build_torchmd_objects(
                m["json_data"], m["mol_id"], tmp_dir, dtype, device
            )
            mol_data_list.append((forces_obj, pos_tensor, mol_id))
            n_atoms_max = max(n_atoms_max, m["n_atoms"])
        except Exception as exc:
            print(f"  skipped {m['mol_id']}: {exc}")
            skipped += 1

    n_mols_actual = len(mol_data_list)
    print(
        f"built {n_mols_actual} TorchMD mol objects "
        f"(skipped {skipped}), n_atoms_max={n_atoms_max}"
    )
    if n_mols_actual == 0:
        raise RuntimeError("No molecules could be loaded — check subset_dir and JSON schema.")

    # Move ALL tensors in each params dict to device (floating → dtype cast,
    # integer index tensors → device only). This covers "params" (force constants)
    # AND "idx" (atom pair/triple/quad indices) that forces.compute passes to
    # index_add_ — a cpu/cuda mismatch on index tensors is the most common failure.
    all_params: list[torch.Tensor] = []
    for forces_obj, _, _ in mol_data_list:
        par = forces_obj.par
        for attr in ("bond_params", "angle_params", "dihedral_params"):
            d = getattr(par, attr)
            if d is not None:
                for key in list(d.keys()):
                    if not isinstance(d[key], torch.Tensor):
                        continue
                    if d[key].is_floating_point():
                        d[key] = d[key].to(device=device, dtype=dtype)
                    else:
                        d[key] = d[key].to(device=device)
                d["params"].requires_grad_(True)
                all_params.append(d["params"])

    optimizer = torch.optim.Adam(all_params, lr=1e-4)

    def _one_step() -> float:
        optimizer.zero_grad()
        total_loss = torch.zeros(1, dtype=dtype, device=device)
        for forces_obj, pos_tensor, _ in mol_data_list:
            n_atoms_i = pos_tensor.shape[0]
            # box: (nreplicas, 3, 3) — large vacuum box (bonded terms don't use PBC)
            box = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 1000.0
            # forces_buf: (nreplicas, natoms, 3) — output buffer for explicit forces
            forces_buf = torch.zeros(1, n_atoms_i, 3, dtype=dtype, device=device)
            # toNumpy=False → returns tensor (differentiable through FF params)
            e = forces_obj.compute(
                pos_tensor.unsqueeze(0), box, forces_buf,
                explicit_forces=True, toNumpy=False,
            )
            total_loss = total_loss + e.sum()
        total_loss.backward()
        optimizer.step()
        return total_loss.item()

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Warmup
    warmup_times = []
    for _ in range(args.n_warmup):
        _sync()
        t0 = time.perf_counter()
        _one_step()
        _sync()
        t1 = time.perf_counter()
        warmup_times.append(t1 - t0)
    compile_seconds = warmup_times[0] if warmup_times else 0.0

    # Trials
    trial_times = []
    final_loss = float("nan")
    for _ in range(args.n_trials):
        _sync()
        t0 = time.perf_counter()
        loss_val = _one_step()
        _sync()
        t1 = time.perf_counter()
        trial_times.append(t1 - t0)
        final_loss = loss_val

    trial_median = float(sorted(trial_times)[len(trial_times) // 2])
    per_mol_step = trial_median / n_mols_actual

    row = {
        "tool": "torchmd",
        "tool_version": _torchmd_version(),
        "scope": "A",
        "n_mols": args.n_mols,
        "n_conformers_per_mol": 1,
        "n_atoms_max": int(n_atoms_max),
        "precision": args.precision,
        "device": device,
        "hardware_tag": os.environ.get("HARDWARE_TAG", "rtx-pro-6000-blackwell"),
        "trial_seconds": trial_median,
        "trial_min_seconds": float(min(trial_times)),
        "trial_max_seconds": float(max(trial_times)),
        "per_mol_step_seconds": per_mol_step,
        "peak_memory_mib": _peak_memory_mib(),
        "compile_seconds": compile_seconds,
        "final_loss": final_loss,
        "git_hash": _git_hash(),
        "n_warmup": args.n_warmup,
        "n_trials": args.n_trials,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=str, default="data/ani1x_subset")
    parser.add_argument("--n-mols", type=int, default=64)
    parser.add_argument("--n-conf-cap", type=int, default=100)
    parser.add_argument(
        "--precision", type=str, choices=["float32", "float64"], default="float32"
    )
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="JSON output path; default stdout")
    args = parser.parse_args()

    row = bench_one_step(args)
    print(json.dumps(row, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(row, f, indent=2)


if __name__ == "__main__":
    main()
