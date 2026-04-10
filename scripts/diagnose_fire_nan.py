"""Diagnose NaN gradient — bisect energy terms (v3, skip dense GB)."""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from pathlib import Path
import dataclasses

jax.config.update("jax_enable_x64", True)

DATA_DIR = Path("prolix/data/pdb")
FF_PATH = Path("proxide/src/proxide/assets/protein.ff19SB.xml")
from proxide import parse_structure, OutputSpec, CoordFormat
from prolix.padding import bucket_proteins, collate_batch

pdb_path = DATA_DIR / "1CRN.pdb"
spec = OutputSpec(
    coord_format=CoordFormat.Full,
    parameterize_md=True,
    force_field=str(FF_PATH),
    add_hydrogens=True,
)
p1 = parse_structure(str(pdb_path), spec)
buckets = bucket_proteins([p1])
bucket_size = list(buckets.keys())[0]
sys0 = buckets[bucket_size][0]

from jax_md import space
from prolix.batched_energy import (
    _bond_energy_masked, _angle_energy_masked, _dihedral_energy_masked,
    _cmap_energy_masked, _lj_energy_masked, _coulomb_energy_masked,
    _build_dense_exclusion_scales, single_padded_energy,
)

displacement_fn, _ = space.free()
r = sys0.positions

print(f"System: {sys0.n_real_atoms} real / {sys0.n_padded_atoms} padded")
print(f"Position finite: {jnp.all(jnp.isfinite(r))}")

def test_grad(name, fn):
    e = fn(r)
    g = jax.grad(fn)(r)
    fin = jnp.all(jnp.isfinite(g))
    gmax = jnp.max(jnp.abs(jnp.where(jnp.isfinite(g), g, 0.0)))
    nan_count = int(jnp.sum(~jnp.isfinite(g)))
    status = "✓" if fin else "✗"
    print(f"  {status} {name:30s}: e={float(e):12.2f}  grad_finite={fin}  nan_count={nan_count}  grad_max={float(gmax):.2f}")

print("\n=== Individual Terms ===")
test_grad("bond", lambda r: _bond_energy_masked(r, sys0.bonds, sys0.bond_params, sys0.bond_mask, displacement_fn))
test_grad("angle", lambda r: _angle_energy_masked(r, sys0.angles, sys0.angle_params, sys0.angle_mask, displacement_fn))
test_grad("dihedral", lambda r: _dihedral_energy_masked(r, sys0.dihedrals, sys0.dihedral_params, sys0.dihedral_mask, displacement_fn))

N = len(sys0.atom_mask)
excl_vdw = jax.lax.stop_gradient(_build_dense_exclusion_scales(sys0.excl_indices, sys0.excl_scales_vdw, N))
excl_elec = jax.lax.stop_gradient(_build_dense_exclusion_scales(sys0.excl_indices, sys0.excl_scales_elec, N))

test_grad("LJ (no excl, lam=1.0)", lambda r: _lj_energy_masked(r, sys0.sigmas, sys0.epsilons, sys0.atom_mask, displacement_fn))
test_grad("LJ (no excl, lam=0.1)", lambda r: _lj_energy_masked(r, sys0.sigmas, sys0.epsilons, sys0.atom_mask, displacement_fn, soft_core_lambda=jnp.float32(0.1)))
test_grad("LJ (excl, lam=1.0)", lambda r: _lj_energy_masked(r, sys0.sigmas, sys0.epsilons, sys0.atom_mask, displacement_fn, excl_scale_vdw=excl_vdw))
test_grad("LJ (excl, lam=0.1)", lambda r: _lj_energy_masked(r, sys0.sigmas, sys0.epsilons, sys0.atom_mask, displacement_fn, soft_core_lambda=jnp.float32(0.1), excl_scale_vdw=excl_vdw))
test_grad("Coulomb (excl)", lambda r: _coulomb_energy_masked(r, sys0.charges, sys0.atom_mask, displacement_fn, excl_scale_elec=excl_elec))

# Skip dense GB — OOMs with N=4096 (would need NL path)
print("\n  ⊘ GB (skipped — dense path OOMs at N=4096)")

print("\n=== Full Energy ===")
test_grad("full (lam=1.0)", lambda r: single_padded_energy(dataclasses.replace(sys0, positions=r), displacement_fn))
test_grad("full (lam=0.1)", lambda r: single_padded_energy(dataclasses.replace(sys0, positions=r), displacement_fn, soft_core_lambda=jnp.float32(0.1)))

print("\n=== FIRE minimizer (10 steps, 1 system) ===")
p2 = parse_structure(str(pdb_path), spec)
p2_coords = [c + 0.1 for c in p2.coordinates]
p2 = dataclasses.replace(p2, coordinates=p2_coords)
buckets2 = bucket_proteins([p1, p2])
bs = list(buckets2.keys())[0]
batch = collate_batch(buckets2[bs])
from prolix.batched_simulate import batched_minimize
result = batched_minimize(batch, max_steps=10, chunk_size=1)
print(f"  Result finite: {jnp.all(jnp.isfinite(result))}")
for b in range(result.shape[0]):
    n_nan = int(jnp.sum(~jnp.isfinite(result[b])))
    print(f"    Sys {b}: {n_nan} NaN values")
