"""Scope A bench: DMFF Hamiltonian — forward+backward of bonded energy
across N mols (mixed sizes), one Adam step.

This is the DMFF entry to the §7.1 external-baseline campaign. Each comparator
(bench_prolix.py, bench_torchmd.py, bench_espaloma.py) emits a JSON row with
the same schema, and the campaign analysis joins on (tool, n_mols, precision)
to produce the comparison table.

Usage:
    uv run python scripts/benchmarks/external_baseline/bench_dmff.py \\
        --n-mols 64 --precision float32 \\
        --n-warmup 1 --n-trials 3 --out results.json

Emits a single-line JSON dict matching the result_schema in
bench_dmff.bth.toml.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import optax

# Allow sibling imports (_generate_xml)
sys.path.insert(0, str(Path(__file__).parent))
from _generate_xml import generate_xml_dir


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent
        ).decode().strip()
    except Exception:
        return "unknown"


def _hardware_tag() -> str:
    try:
        d = jax.devices()[0]
        return f"{d.device_kind}"
    except Exception:
        return "cpu"


def _peak_memory_mib() -> float:
    try:
        stats = jax.devices()[0].memory_stats()
        peak = stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0))
        return float(peak) / (1024 * 1024)
    except Exception:
        return -1.0


_ELEMENT_SYMBOLS = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}


def _load_mol_data(
    json_path: Path,
    xml_path: Path,
    h5_path: Path,
) -> dict:
    """Load one molecule: parse JSON for topology info, load positions from H5.

    Returns a dict with keys:
        xml_path, n_atoms, element_symbols, bonds, pos_nm (first conformer)
    """
    import json as _json

    with open(json_path) as f:
        mol = _json.load(f)

    n_atoms = len(mol["atom_types"])
    element_symbols = [
        _ELEMENT_SYMBOLS.get(a["Z"], "C") for a in mol["atom_types"]
    ]
    bonds = [{"i": b["i"], "j": b["j"]} for b in mol["bonds"]]

    with h5py.File(h5_path, "r") as h:
        # Use first conformer only; convert Å → nm
        pos_akma = jnp.asarray(h["positions"][0])  # (n_atoms, 3) in Å
    pos_nm = pos_akma * 0.1  # Å → nm

    return {
        "xml_path": xml_path,
        "n_atoms": n_atoms,
        "element_symbols": element_symbols,
        "bonds": bonds,
        "pos_nm": pos_nm,
    }


def _build_dmff_potential(mol_data: dict):
    """Construct DMFF Hamiltonian + potential for one molecule.

    Returns (efunc, params, pos_nm, box, pairs) where:
        efunc   : callable(pos_nm, box, pairs, params) → scalar
        params  : ParamSet pytree
        pos_nm  : jnp.array (n_atoms, 3)
        box     : jnp.zeros((3,3))
        pairs   : NoCutoffNeighborList pairs array
    """
    from dmff import Hamiltonian
    from dmff.common.nblist import NoCutoffNeighborList
    from openmm import app

    xml_path_str = str(mol_data["xml_path"])
    n_atoms = mol_data["n_atoms"]
    element_symbols = mol_data["element_symbols"]
    bonds = mol_data["bonds"]
    pos_nm = mol_data["pos_nm"]

    h = Hamiltonian(xml_path_str)

    # Build minimal OpenMM topology
    topo = app.Topology()
    chain = topo.addChain()
    res = topo.addResidue("MOL", chain)
    for i in range(n_atoms):
        sym = element_symbols[i]
        try:
            elem = app.Element.getBySymbol(sym)
        except Exception:
            elem = app.Element.getBySymbol("C")
        topo.addAtom(f"A{i}", elem, res)

    atom_list = list(topo.atoms())
    for bond in bonds:
        topo.addBond(atom_list[bond["i"]], atom_list[bond["j"]])

    pot = h.createPotential(topo, nonbondedMethod=app.NoCutoff)

    box = jnp.zeros((3, 3))
    pairs = NoCutoffNeighborList(cov_map=pot.meta["cov_map"]).allocate(pos_nm)

    efunc = pot.getPotentialFunc()
    params = h.getParameters()

    return efunc, params, pos_nm, box, pairs


def bench_one_step(args) -> dict:
    """Run Scope A primitive: warmup + n_trials, return JSON row."""
    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)

    subset_dir = Path(args.subset_dir)
    lane_a = subset_dir / "lane_a"

    # Generate (or use provided) XML directory
    if args.xml_dir is not None:
        xml_dir = Path(args.xml_dir)
    else:
        _tmpdir = tempfile.mkdtemp(prefix="bench_dmff_xml_")
        xml_dir = Path(_tmpdir)
        generate_xml_dir(subset_dir, xml_dir, lane="lane_a")

    json_paths = sorted(lane_a.glob("mol_*.params_init.json"))
    h5_paths = [lane_a / p.name.replace(".params_init.json", ".h5") for p in json_paths]

    # Pair JSON → XML by molecule_id (not by sort order — XMLs are named by mol_id)
    n_base = len(json_paths)
    if n_base == 0:
        raise RuntimeError(f"No mol_*.params_init.json found in {lane_a}")

    import json as _json_mod
    mol_ids = []
    for jp in json_paths:
        with open(jp) as f:
            mol_ids.append(_json_mod.load(f)["molecule_id"])
    xml_paths = [xml_dir / f"{mid}.xml" for mid in mol_ids]
    missing = [p for p in xml_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing XMLs for {len(missing)} mols (e.g. {missing[0]}). "
                           f"Re-run without --xml-dir to regenerate.")

    print(f"found {n_base} base mols, {len(xml_paths)} XMLs")

    # Load base mol data
    base_mol_data = []
    for json_path, xml_path, h5_path in zip(json_paths, xml_paths, h5_paths):
        base_mol_data.append(_load_mol_data(json_path, xml_path, h5_path))

    # Tile to n_mols
    tile_idx = [i % n_base for i in range(args.n_mols)]
    mol_data_list = [base_mol_data[i] for i in tile_idx]

    print(f"tiled to {args.n_mols} mols")

    # Build DMFF potentials for each mol
    print("building DMFF potentials...")
    potentials = []
    for i, md in enumerate(mol_data_list):
        efunc, params, pos_nm, box, pairs = _build_dmff_potential(md)
        potentials.append((efunc, params, pos_nm, box, pairs))
    print(f"built {len(potentials)} potentials")

    # All mols share the same param pytree structure (same XML template).
    # We use the params from mol 0 as the shared differentiable parameter set.
    _, shared_params, _, _, _ = potentials[0]

    # Pre-collect (pos_nm, box, pairs) per mol for the loss loop
    mol_inputs = [(pos_nm, box, pairs) for (_, _, pos_nm, box, pairs) in potentials]
    efuncs = [efunc for (efunc, _, _, _, _) in potentials]

    n_atoms_max = max(md["n_atoms"] for md in mol_data_list)

    # Adam optimizer (Scope A: one step per trial)
    opt = optax.adam(learning_rate=1e-4)
    opt_state = opt.init(shared_params)

    def step_fn(params):
        def total_loss(p):
            loss = 0.0
            for efunc, (pos_nm, box, pairs) in zip(efuncs, mol_inputs):
                loss = loss + efunc(pos_nm, box, pairs, p)
            return loss

        val, grads = jax.value_and_grad(total_loss)(params)
        updates, _new_opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return val, new_params

    # Warmup: compiles + discarded for timing
    warmup_times = []
    current_params = shared_params
    for w in range(args.n_warmup):
        t0 = time.perf_counter()
        val, current_params = step_fn(current_params)
        # Block on a leaf of new_params
        leaves = jax.tree_util.tree_leaves(current_params)
        if leaves:
            jax.block_until_ready(leaves[0])
        t1 = time.perf_counter()
        warmup_times.append(t1 - t0)
        print(f"  warmup {w}: {t1 - t0:.3f}s  loss={float(val):.6g}")
    compile_seconds = warmup_times[0] if warmup_times else 0.0

    # Trials: each trial is ONE forward+backward+update (Scope A = primitive)
    trial_times = []
    final_loss = float("nan")
    for trial in range(args.n_trials):
        t0 = time.perf_counter()
        val, current_params = step_fn(current_params)
        leaves = jax.tree_util.tree_leaves(current_params)
        if leaves:
            jax.block_until_ready(leaves[0])
        t1 = time.perf_counter()
        trial_times.append(t1 - t0)
        final_loss = float(val)
        print(f"  trial {trial}: {t1 - t0:.3f}s  loss={final_loss:.6g}")

    trial_median = float(sorted(trial_times)[len(trial_times) // 2])
    per_mol_step = trial_median / args.n_mols

    row = {
        "tool": "dmff",
        "tool_version": "0.2.7",
        "scope": "A",
        "n_mols": args.n_mols,
        "n_conformers_per_mol": 1,
        "n_atoms_max": int(n_atoms_max),
        "precision": args.precision,
        "device": _hardware_tag(),
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
    parser.add_argument(
        "--xml-dir",
        type=str,
        default=None,
        help="Pre-generated XML directory; if None, XMLs are generated to a tempdir",
    )
    parser.add_argument("--n-mols", type=int, default=64)
    parser.add_argument(
        "--n-conf-cap",
        type=int,
        default=100,
        help="Conformer cap (unused; DMFF uses first conformer only)",
    )
    parser.add_argument(
        "--precision", type=str, choices=["float32", "float64"], default="float32"
    )
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=str, default=None, help="JSON output path; default stdout"
    )
    args = parser.parse_args()

    row = bench_one_step(args)
    print(json.dumps(row, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(row, f, indent=2)


if __name__ == "__main__":
    main()
