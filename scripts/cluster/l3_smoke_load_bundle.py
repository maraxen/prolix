"""L3 cluster smoke: load one HP4 curated molecule + build bundle + n_steps=0 gradient.

Validates the post-rsync pipeline on the cluster:
  - HDF5 read from data/ani1x_subset/lane_*/mol_NNN.h5
  - MolecularBundle construction via the HP3-coarsened MolecularShapeSpec
  - HP4 bucket-index assignment under the new ATOM_BUCKETS ladder
  - jax.grad(energy_fn)(bundle) returns finite forces

Usage (local, L1 + L2 — synthetic mode, no archive needed):
    uv run python scripts/cluster/l3_smoke_load_bundle.py --synthetic --n-atoms 30

Usage (cluster L3, after rsync):
    uv run python scripts/cluster/l3_smoke_load_bundle.py \\
        --data-path data/ani1x_subset/lane_a/mol_000.h5

Exit codes:
    0  success: bundle loaded, gradient finite
    1  load error (missing file, bad HDF5)
    2  gradient produced NaN/inf
    3  shape_spec assertion failed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _stub_energy(positions: jax.Array) -> jax.Array:
    """Pairwise-distance harmonic stub. O(N^2) compute, exercises gradient path.

    Used only when bonded force-field params are not yet attached to the bundle
    (HP4 fetch script ships positions + species + reference forces; the §7.1
    bonded-param toolchain runs separately). For L3 smoke, this stub validates
    that the bundle structure is consistent and JAX gradient propagates.
    """
    diff = positions[None, :, :] - positions[:, None, :]
    r2 = jnp.sum(diff * diff, axis=-1)
    return 0.5 * jnp.sum(r2)


def _make_stub_system(positions: jax.Array, species: jax.Array):
    """Duck-typed system for make_bundle_from_system (uses getattr internally).

    Sets only the attributes the factory reads; others default to None and the
    factory zero-pads. Avoids importing PhysicsSystem (which has a different
    schema across prolix versions).
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        positions=positions,
        species=species,
        masses=jnp.ones(positions.shape[0], dtype=jnp.float32) * 12.0,
        # All bonded topology empty — factory zero-pads
        bonds=None, bond_params=None,
        angles=None, angle_params=None,
        dihedrals=None, dihedral_params=None,
        impropers=None, improper_params=None,
        urey_bradley_bonds=None, urey_bradley_params=None,
        water_indices=None,
        excl_indices=None,
        box_size=None,
        charges=None, sigmas=None, epsilons=None,
        radii=None, scaled_radii=None,
    )


def _synthetic_bundle(n_atoms: int):
    """Build an in-memory minimal bundle for L1/L2 local validation."""
    from prolix.physics.system import make_bundle_from_system

    key = jax.random.PRNGKey(0)
    positions = jax.random.normal(key, (n_atoms, 3)).astype(jnp.float32) * 2.0
    species = jnp.ones(n_atoms, dtype=jnp.int8) * 6
    return make_bundle_from_system(_make_stub_system(positions, species), boundary_condition="free")


def _load_bundle_from_hdf5(path: Path):
    """Load HP4 curated molecule from per-molecule HDF5.

    Schema per spec §4.4: positions, forces, energy, species, smiles, molecule_id.
    Forces in kcal/mol/Å (unit-converted by fetch script). Bonded topology is
    NOT in the HDF5 — that's resolved by §7.1's build_system_from_smiles toolchain
    (out of scope here). L3 smoke uses positions only.
    """
    import h5py

    if not path.exists():
        log.error("HDF5 not found: %s", path)
        sys.exit(1)

    with h5py.File(path) as f:
        positions_all = f["positions"][:]
        species = jnp.asarray(f["species"][:], dtype=jnp.int8)
        bucket_idx = int(f.attrs.get("bucket_idx", -1))
        n_atoms = int(positions_all.shape[1])
        n_conf = int(positions_all.shape[0])
        log.info(
            "Loaded %s: n_atoms=%d n_conf=%d bucket_idx=%d",
            path.name, n_atoms, n_conf, bucket_idx,
        )

    # Take conformer 0 for the smoke
    positions = jnp.asarray(positions_all[0], dtype=jnp.float32)

    from prolix.physics.system import make_bundle_from_system

    bundle = make_bundle_from_system(
        _make_stub_system(positions, species), boundary_condition="free"
    )
    return bundle, bucket_idx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-path", type=Path, help="Per-molecule HDF5 from fetch script")
    src.add_argument("--synthetic", action="store_true", help="Build in-memory minimal bundle")
    p.add_argument("--n-atoms", type=int, default=30, help="(synthetic mode) atoms per bundle")
    p.add_argument("--out-json", type=Path, default=None, help="Write summary JSON here")
    args = p.parse_args()

    log.info("JAX backend: %s", jax.default_backend())
    log.info("JAX devices: %s", jax.devices())

    if args.synthetic:
        log.info("Building synthetic bundle (n_atoms=%d)", args.n_atoms)
        bundle = _synthetic_bundle(args.n_atoms)
        bucket_idx_hdf5 = -1
    else:
        bundle, bucket_idx_hdf5 = _load_bundle_from_hdf5(args.data_path)

    # Assert shape_spec consistency
    spec = bundle.shape_spec
    assert hasattr(spec, "atom_bucket_idx"), "shape_spec missing atom_bucket_idx (HP3 contract)"
    log.info("shape_spec.atom_bucket_idx=%d  (HDF5 attr=%d)", spec.atom_bucket_idx, bucket_idx_hdf5)
    if bucket_idx_hdf5 >= 0 and bucket_idx_hdf5 != spec.atom_bucket_idx:
        log.error("bucket_idx mismatch: HDF5=%d vs computed=%d", bucket_idx_hdf5, spec.atom_bucket_idx)
        return 3

    # Gradient pass (n_steps=0 forward-mode energy + grad)
    log.info("Compiling jax.grad(_stub_energy)...")
    grad_fn = jax.jit(jax.grad(_stub_energy))
    forces = jax.block_until_ready(grad_fn(bundle.positions))
    log.info("Gradient computed. shape=%s dtype=%s", forces.shape, forces.dtype)

    n_finite = int(jnp.sum(jnp.isfinite(forces)))
    n_total = int(forces.size)
    log.info("Finite force components: %d / %d", n_finite, n_total)
    if n_finite != n_total:
        log.error("NaN/inf in gradient")
        return 2

    summary = {
        "mode": "synthetic" if args.synthetic else "data-path",
        "data_path": str(args.data_path) if args.data_path else None,
        "n_atoms": int(bundle.positions.shape[0]),
        "atom_bucket_idx": int(spec.atom_bucket_idx),
        "force_norm_l2": float(jnp.linalg.norm(forces)),
        "force_max_abs": float(jnp.max(jnp.abs(forces))),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
    }
    log.info("Summary: %s", json.dumps(summary, indent=2))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2))
        log.info("Wrote %s", args.out_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
