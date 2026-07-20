"""B1-INFER: while_loop inference path, dedup wire, XTC stream smoke."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from prolix.api.ensemble_dedup import (
    build_dedup_spec_by_shape,
    dispatch_n_mols_dedup,
)
from prolix.api.ensemble_dispatch import (
    dispatch_n_steps,
    dispatch_n_steps_inference,
)
from prolix.api.ensemble_plan import EnsemblePlan
from prolix.tiling.axes import N_MOLS
from prolix.types.bundles import MolecularBundle, MolecularShapeSpec

def _make_minimal_bundle(n_atoms=3) -> MolecularBundle:
    """Create a minimal MolecularBundle for testing (water molecule or similar).

    Args:
        n_atoms: Number of atoms (default 3 for water)

    Returns:
        MolecularBundle with minimal topology
    """
    # Create a minimal bundle directly with padded arrays
    # For simplicity: 3-atom system with no bonds, angles, dihedrals, etc.

    # Pad to smallest bucket sizes
    from prolix.types.bundles import ATOM_BUCKETS, _bucket_idx

    atom_bucket_idx = _bucket_idx(n_atoms, ATOM_BUCKETS)
    padded_n_atoms = ATOM_BUCKETS[atom_bucket_idx]

    positions = jnp.zeros((padded_n_atoms, 3), dtype=jnp.float32)
    positions = positions.at[:n_atoms].set(jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=jnp.float32))

    charges = jnp.zeros(padded_n_atoms, dtype=jnp.float32)
    charges = charges.at[:n_atoms].set(jnp.array([0.8, -0.4, -0.4], dtype=jnp.float32))

    sigmas = jnp.ones(padded_n_atoms, dtype=jnp.float32) * 3.15
    epsilons = jnp.zeros(padded_n_atoms, dtype=jnp.float32)
    radii = jnp.ones(padded_n_atoms, dtype=jnp.float32)
    scaled_radii = jnp.ones(padded_n_atoms, dtype=jnp.float32)

    atom_mask = jnp.zeros(padded_n_atoms, dtype=jnp.bool_)
    atom_mask = atom_mask.at[:n_atoms].set(True)

    # Empty topology arrays
    empty_bond = jnp.zeros((8, 2), dtype=jnp.int32)
    empty_bond_params = jnp.zeros((8, 2), dtype=jnp.float32)
    empty_bond_mask = jnp.zeros(8, dtype=jnp.bool_)

    empty_angle = jnp.zeros((32, 3), dtype=jnp.int32)
    empty_angle_params = jnp.zeros((32, 2), dtype=jnp.float32)
    empty_angle_mask = jnp.zeros(32, dtype=jnp.bool_)

    empty_dihedral = jnp.zeros((32, 4), dtype=jnp.int32)
    empty_dihedral_params = jnp.zeros((32, 4), dtype=jnp.float32)
    empty_dihedral_mask = jnp.zeros(32, dtype=jnp.bool_)

    empty_improper = jnp.zeros((32, 4), dtype=jnp.int32)
    empty_improper_params = jnp.zeros((32, 3), dtype=jnp.float32)
    empty_improper_mask = jnp.zeros(32, dtype=jnp.bool_)

    empty_ub = jnp.zeros((32, 3), dtype=jnp.int32)
    empty_ub_params = jnp.zeros((32, 2), dtype=jnp.float32)
    empty_ub_mask = jnp.zeros(32, dtype=jnp.bool_)

    empty_cmap = jnp.zeros((8, 24, 24), dtype=jnp.float32)
    empty_cmap_mask = jnp.zeros(8, dtype=jnp.bool_)

    empty_water = jnp.zeros((8, 3), dtype=jnp.int32)
    empty_water_mask = jnp.zeros(8, dtype=jnp.bool_)

    empty_excl = jnp.zeros((32, 2), dtype=jnp.int32)
    empty_excl_vdw = jnp.zeros(32, dtype=jnp.float32)
    empty_excl_elec = jnp.zeros(32, dtype=jnp.float32)
    empty_excl_mask = jnp.zeros(32, dtype=jnp.bool_)

    empty_exc = jnp.zeros((32, 2), dtype=jnp.int32)
    empty_exc_sigma = jnp.zeros(32, dtype=jnp.float32)
    empty_exc_epsilon = jnp.zeros(32, dtype=jnp.float32)
    empty_exc_charge = jnp.zeros(32, dtype=jnp.float32)
    empty_exc_mask = jnp.zeros(32, dtype=jnp.bool_)

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
        boundary_condition='free',
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



def test_dispatch_n_steps_inference_matches_scan_final():
    """while_loop final carry agrees with scan final position (atol)."""
    init = jnp.array([1.0, 2.0, 3.0])

    def scan_step(carry, _):
        new = carry + 0.5
        return new, new

    def infer_step(carry, _step_i):
        return carry + 0.5

    n = 7
    _, ys = dispatch_n_steps(scan_step, init, n)
    final = dispatch_n_steps_inference(infer_step, init, n)
    np.testing.assert_allclose(np.asarray(final), np.asarray(ys[-1]), atol=1e-6)


def test_ensemble_plan_inference_final_frame_parity():
    bundle = _make_minimal_bundle()
    plan = EnsemblePlan.from_bundle(bundle)
    n_steps = 5
    traj = plan.run(
        n_steps=n_steps,
        dt=0.5,
        kT=0.596,
        seed=0,
        gamma=10.0,
        run_mode="trajectory",
    )
    inf = plan.run(
        n_steps=n_steps,
        dt=0.5,
        kT=0.596,
        seed=0,
        gamma=10.0,
        run_mode="inference",
    )
    assert inf.positions.shape[0] == 1
    np.testing.assert_allclose(
        np.asarray(inf.positions[-1]),
        np.asarray(traj.positions[-1]),
        atol=1e-4,
        rtol=1e-4,
    )


def test_build_dedup_spec_four_identical_k1_scatter():
    bundles = [_make_minimal_bundle() for _ in range(4)]
    spec = build_dedup_spec_by_shape(bundles)
    assert spec.k == 1
    assert spec.axis_name == N_MOLS.name
    assert list(spec.index_map) == [0, 0, 0, 0]

    calls = {"n": 0}

    def body(x):
        calls["n"] += 1
        return x * 3.0

    xs = jnp.array([10.0, 10.0, 10.0, 10.0])
    got = dispatch_n_mols_dedup(spec, body, xs)
    np.testing.assert_array_equal(np.asarray(got), np.array([30.0, 30.0, 30.0, 30.0]))
    assert calls["n"] == 1


def test_inference_xtc_stream_save_every(tmp_path: Path):
    proxide = pytest.importorskip("proxide")
    bundle = _make_minimal_bundle()
    path = tmp_path / "infer.xtc"
    plan = EnsemblePlan.from_bundle(bundle)
    traj = plan.run(
        n_steps=6,
        dt=0.5,
        kT=0.596,
        seed=1,
        gamma=10.0,
        run_mode="inference",
        xtc_path=path,
        save_every=2,
    )
    assert path.exists() and path.stat().st_size > 0
    assert traj.positions.shape[0] == 1
    parsed = proxide.parse_xtc(str(path))
    coords = np.asarray(parsed["coordinates"])
    # steps 0,2,4 → 3 frames
    assert coords.shape[0] == 3
    assert coords.shape[1] == int(bundle.n_atoms)


def test_b1_init_exec_path_inference_smoke():
    """CLI smoke: --smoke --path inference exits 0 with finite metrics."""
    import json
    import subprocess
    import sys
    from pathlib import Path as P

    root = P(__file__).resolve().parents[2]
    out = root / "tmp" / "b1_infer_smoke_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(root / "scripts" / "benchmarks" / "b1_init_exec.py"),
        "--smoke",
        "--path",
        "inference",
        "--out-json",
        str(out),
    ]
    proc = subprocess.run(
        cmd, cwd=str(root), capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    row = json.loads(out.read_text())
    assert row["path"] == "inference"
    assert row["run_mode"] == "inference"
    assert row["finite_positions"] is True
    assert row.get("finite_fraction", 1.0) >= 0.9
    assert row["t_total"] > 0
    # B1-AOT-RATIO: compile share of wall, not cold-warm
    assert "aot_ratio" in row
    assert row["aot_ratio"] == pytest.approx(
        row["t_aot_compile"] / row["t_total"], rel=0, abs=1e-9
    )
