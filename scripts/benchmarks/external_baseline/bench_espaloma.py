"""Scope A bench: espaloma DGL heterograph — forward+backward of bonded energy
across N mols (mixed sizes), one Adam step.

This is the espaloma entry to the §7.1 external-baseline campaign. Each comparator
(bench_dmff.py, bench_torchmd.py, bench_espaloma.py, bench_prolix.py)
emits a JSON row with the same schema, and the campaign analysis joins on
(tool, n_mols, precision) to produce the comparison table.

Usage:
    uv run python scripts/benchmarks/external_baseline/bench_espaloma.py \\
        --n-mols 64 --n-conf-cap 100 --precision float32 \\
        --n-warmup 1 --n-trials 3 --out results.json

Emits a single-line JSON dict matching the result_schema in
bench_espaloma.bth.toml.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import h5py
import torch

logger = logging.getLogger(__name__)


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
    try:
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
    except Exception:
        pass
    return -1.0


def build_graph_from_json(mol_json: dict):
    """Build a DGL heterograph from a prolix params_init JSON dict.

    Injects FF parameters from the prolix JSON into the graph node data,
    overriding espaloma's own FF assignment. Returns the dgl.DGLHeteroGraph
    or raises on failure (caller should catch and skip).
    """
    import dgl
    from rdkit import Chem
    from openff.toolkit import Molecule as OFFMolecule
    import espaloma as esp

    atom_types = mol_json["atom_types"]
    bonds = mol_json["bonds"]
    angles = mol_json["angles"]
    torsions = mol_json["proper_torsions"]

    # Build rdkit mol from atoms + bonds
    rw = Chem.RWMol()
    for at in sorted(atom_types, key=lambda x: x["idx"]):
        rw.AddAtom(Chem.Atom(at["Z"]))
    for b in bonds:
        rw.AddBond(b["i"], b["j"], Chem.BondType.SINGLE)
    mol_rd = rw.GetMol()

    # Build openff molecule from rdkit mol
    off_mol = OFFMolecule.from_rdkit(mol_rd, allow_undefined_stereo=True)

    # Build espaloma graph
    g = esp.Graph(off_mol)
    g = g.heterograph  # dgl.DGLHeteroGraph

    # Inject bond parameters (n2)
    n2 = g.number_of_nodes("n2")
    k_bonds = torch.zeros(n2)
    eq_bonds = torch.zeros(n2)
    for idx, b in enumerate(bonds):
        if idx < n2:
            k_bonds[idx] = b["k"]   # kcal/mol/Å²
            eq_bonds[idx] = b["r0"]  # Å
    g.nodes["n2"].data["k"] = k_bonds.unsqueeze(-1)
    g.nodes["n2"].data["eq"] = eq_bonds.unsqueeze(-1)

    # Inject angle parameters (n3)
    n3 = g.number_of_nodes("n3")
    k_angles = torch.zeros(n3)
    eq_angles = torch.zeros(n3)
    for idx, a in enumerate(angles):
        if idx < n3:
            k_angles[idx] = a["k_theta"]                            # kcal/mol/rad²
            eq_angles[idx] = a["theta0_deg"] * 3.14159265 / 180.0  # radians
    g.nodes["n3"].data["k"] = k_angles.unsqueeze(-1)
    g.nodes["n3"].data["eq"] = eq_angles.unsqueeze(-1)

    # Inject torsion parameters (n4) — espaloma uses 6 periodicity slots
    n4 = g.number_of_nodes("n4")
    k_tors = torch.zeros(n4, 6)
    phases = torch.zeros(n4, 6)
    periodicity = torch.zeros(n4, 6, dtype=torch.float32)
    for idx, t in enumerate(torsions):
        if idx < n4:
            for term_i, (n, ph, kp) in enumerate(
                zip(t["periodicity"], t["phase_deg"], t["k_phi"])
            ):
                if term_i < 6:
                    k_tors[idx, term_i] = kp
                    phases[idx, term_i] = ph * 3.14159265 / 180.0
                    periodicity[idx, term_i] = float(n)
    g.nodes["n4"].data["k"] = k_tors
    g.nodes["n4"].data["phases"] = phases
    g.nodes["n4"].data["periodicity"] = periodicity

    return g


def load_base_mols(subset_dir: Path, n_conf_cap: int | None = None) -> list[dict]:
    """Load base mols from data/ani1x_subset/lane_a/ without prolix library imports."""
    lane_a = subset_dir / "lane_a"
    mol_json_files = sorted(lane_a.glob("mol_*.params_init.json"))
    if not mol_json_files:
        raise FileNotFoundError(f"No mol_*.params_init.json files found in {lane_a}")

    mols = []
    for f in mol_json_files:
        with open(f) as fh:
            mol_json = json.load(fh)

        h5_path = lane_a / f.name.replace(".params_init.json", ".h5")
        with h5py.File(h5_path, "r") as h:
            n = h["positions"].shape[0]
            cap = min(n, n_conf_cap) if n_conf_cap else n
            # Use only first conformer (index 0) for espaloma Scope A
            positions = h["positions"][0]  # shape (n_atoms, 3), Å

        mols.append({
            "mol_json": mol_json,
            "positions": positions,
            "n_atoms": mol_json["n_atoms"],
        })
    return mols


def build_graphs(base_mols: list[dict], n_mols_target: int) -> tuple[list, list, int]:
    """Tile base mols to n_mols_target; build DGL graphs with error handling.

    Returns (graphs, positions_list, n_atoms_max).
    Raises RuntimeError if >50% of molecules fail graph construction.
    """
    n_base = len(base_mols)
    tile_idx = [i % n_base for i in range(n_mols_target)]

    graphs = []
    positions_list = []
    n_failed = 0
    n_atoms_max = 0

    for i in tile_idx:
        m = base_mols[i]
        try:
            g = build_graph_from_json(m["mol_json"])
            graphs.append(g)
            positions_list.append(m["positions"])
            n_atoms_max = max(n_atoms_max, m["n_atoms"])
        except Exception as exc:
            n_failed += 1
            logger.warning(
                "Skipping mol index %d (base mol %d): %s", len(graphs) + n_failed - 1, i, exc
            )

    n_total = len(tile_idx)
    if n_failed > n_total // 2:
        raise RuntimeError(
            f"Graph construction failed for {n_failed}/{n_total} molecules (>50%). "
            "Check openff-toolkit SMILES inference and rdkit bond assignment."
        )

    if n_failed > 0:
        logger.warning(
            "Graph construction failed for %d/%d molecules; proceeding with %d.",
            n_failed, n_total, len(graphs),
        )

    return graphs, positions_list, n_atoms_max


def bench_one_step(args) -> dict:
    """Run Scope A primitive: warmup + n_trials, return JSON row."""
    import dgl
    from espaloma.mm.geometry import geometry_in_graph
    from espaloma.mm.energy import energy_in_graph

    device = _device_str()
    dtype = torch.float64 if args.precision == "float64" else torch.float32

    base_mols = load_base_mols(Path(args.subset_dir), n_conf_cap=args.n_conf_cap)
    logger.info("Loaded %d base mols", len(base_mols))

    graphs, positions_list, n_atoms_max = build_graphs(base_mols, args.n_mols)
    n_mols_actual = len(graphs)
    logger.info(
        "Built %d DGL graphs (n_atoms_max=%d)", n_mols_actual, n_atoms_max
    )

    # Move graphs to device and cast parameter tensors
    for g in graphs:
        for ntype in g.ntypes:
            for key in list(g.nodes[ntype].data.keys()):
                g.nodes[ntype].data[key] = g.nodes[ntype].data[key].to(device=device, dtype=dtype)

    # Build optimizer over all parameter tensors
    param_tensors = []
    for g in graphs:
        for key in ["k", "eq"]:
            if key in g.nodes["n2"].data:
                t = g.nodes["n2"].data[key]
                t.requires_grad_(True)
                param_tensors.append(t)
            if key in g.nodes["n3"].data:
                t = g.nodes["n3"].data[key]
                t.requires_grad_(True)
                param_tensors.append(t)
        if "k" in g.nodes["n4"].data:
            t = g.nodes["n4"].data["k"]
            t.requires_grad_(True)
            param_tensors.append(t)

    optimizer = torch.optim.Adam(param_tensors, lr=1e-4)

    # Batch all graphs for a single forward pass
    batched_g = dgl.batch(graphs).to(device)

    def step_fn():
        optimizer.zero_grad()
        # Inject positions into n1.xyz for each mol in the batch
        n1_start = 0
        for g_i, pos_i in zip(graphs, positions_list):
            n1_i = g_i.number_of_nodes("n1")
            pos_tensor = torch.tensor(
                pos_i[:n1_i], dtype=dtype, device=device
            )
            batched_g.nodes["n1"].data["xyz"][n1_start : n1_start + n1_i] = pos_tensor
            n1_start += n1_i

        geometry_in_graph(batched_g)
        energy_in_graph(batched_g, terms=["n2", "n3", "n4"])
        loss = batched_g.nodes["g"].data["u"].sum()
        loss.backward()
        optimizer.step()
        return loss.item()

    # Initialise xyz node data before warmup
    if "xyz" not in batched_g.nodes["n1"].data:
        n1_total = batched_g.number_of_nodes("n1")
        batched_g.nodes["n1"].data["xyz"] = torch.zeros(
            n1_total, 3, dtype=dtype, device=device
        )

    # Warmup: compiles + discarded for timing
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    warmup_times = []
    for _ in range(args.n_warmup):
        t0 = time.perf_counter()
        step_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        warmup_times.append(t1 - t0)
    compile_seconds = warmup_times[0] if warmup_times else 0.0

    # Trials: each trial is ONE forward+backward+Adam step (Scope A primitive)
    trial_times = []
    final_loss = float("nan")
    for _ in range(args.n_trials):
        t0 = time.perf_counter()
        loss_val = step_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        trial_times.append(t1 - t0)
        final_loss = loss_val

    trial_median = float(sorted(trial_times)[len(trial_times) // 2])
    per_mol_step = trial_median / n_mols_actual

    row = {
        "tool": "espaloma",
        "tool_version": "0.3.2",
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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

    if args.seed is not None:
        torch.manual_seed(args.seed)

    row = bench_one_step(args)
    print(json.dumps(row, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(row, f, indent=2)


if __name__ == "__main__":
    main()
