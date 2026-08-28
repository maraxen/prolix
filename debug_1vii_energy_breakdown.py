#!/usr/bin/env python3
"""Diagnostic: break down 1vii energy into components to find the physics bug."""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import types

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, "scripts/experiments")
from nonbonded_omm_parity.lj_oracle import sha256_file

# Load gold
gold_path = Path("data/oracles/openmm_8.3.1/nl_1vii.json")
rec = json.loads(gold_path.read_text())

print(f"Gold file: {gold_path}")
print(f"  - sha256: {sha256_file(gold_path)}")
print(f"  - n_atoms: {rec['n_atoms']}")
print(f"  - cutoff: {rec['cutoff_angstrom']} Å")
print(f"  - pme_grid_points: {rec['pme_grid_points']}")
print(f"  - energy_kcal (gold): {rec['energy_kcal']:.2f} kcal/mol")
print(f"  - omm_exceptions: {len(rec['omm_exceptions'])} entries")
print()

# Build bundle (identical to test)
from prolix.physics.neighbor_list import ExclusionSpec
from prolix.physics.system import make_bundle_from_system
from prolix.api.bundle_md import energy_fn_from_bundle
from prolix.physics import neighbor_list as nl
from prolix.physics.pbc import create_periodic_space

n_atoms = int(rec["n_atoms"])
omm_exc_list = rec["omm_exceptions"]

# Build ExclusionSpec
idx_12_13_list = []
exc_pairs_list = []
exc_sigmas_list = []
exc_epsilons_list = []
exc_chargeprods_list = []

for i_exc, j_exc, q_prod, sig_exc, eps_exc in omm_exc_list:
    idx_12_13_list.append([i_exc, j_exc])
    if q_prod != 0.0 or eps_exc != 0.0:
        exc_pairs_list.append([i_exc, j_exc])
        exc_sigmas_list.append(sig_exc)
        exc_epsilons_list.append(eps_exc)
        exc_chargeprods_list.append(q_prod)

idx_12_13 = jnp.asarray(idx_12_13_list, dtype=jnp.int32) if idx_12_13_list else jnp.zeros((0, 2), dtype=jnp.int32)
exc_pairs = jnp.asarray(exc_pairs_list, dtype=jnp.int32) if exc_pairs_list else jnp.zeros((0, 2), dtype=jnp.int32)
exc_sigmas = jnp.asarray(exc_sigmas_list, dtype=jnp.float32) if exc_sigmas_list else jnp.zeros((0,), dtype=jnp.float32)
exc_epsilons = jnp.asarray(exc_epsilons_list, dtype=jnp.float32) if exc_epsilons_list else jnp.zeros((0,), dtype=jnp.float32)
exc_chargeprods = jnp.asarray(exc_chargeprods_list, dtype=jnp.float32) if exc_chargeprods_list else jnp.zeros((0,), dtype=jnp.float32)
idx_14 = jnp.zeros((0, 2), dtype=jnp.int32)

exclusion_spec = ExclusionSpec(
    idx_12_13=idx_12_13,
    idx_14=idx_14,
    scale_14_elec=0.83333333,
    scale_14_vdw=0.5,
    n_atoms=n_atoms,
    exception_pairs=exc_pairs,
    exception_sigmas=exc_sigmas,
    exception_epsilons=exc_epsilons,
    exception_chargeprods=exc_chargeprods,
)

# Build proxy
positions_A = np.asarray(rec["positions_A"], dtype=np.float64)
box_A = np.asarray(rec["box_A"], dtype=np.float64)
charges = np.asarray(rec["charges"], dtype=np.float64)
sigmas_A = np.asarray(rec["sigmas_A"], dtype=np.float64)
epsilons_kcal = np.asarray(rec["epsilons_kcal"], dtype=np.float64)
pme_alpha = float(rec.get("pme_alpha_per_angstrom", 0.34))
cutoff = float(rec.get("cutoff_angstrom", 9.0))

print(f"Box: {box_A}")
print(f"Charges: min={charges.min():.4f}, max={charges.max():.4f}, sum={charges.sum():.6e}")
print()

proxy = types.SimpleNamespace(
    positions=positions_A,
    box_size=box_A,
    charges=charges,
    sigmas=sigmas_A,
    epsilons=epsilons_kcal,
    pme_alpha=pme_alpha,
    nonbonded_cutoff=cutoff,
)

# Build bundle
bundle = make_bundle_from_system(
    proxy,
    boundary_condition="periodic",
    exclusion_spec=exclusion_spec,
    use_size_buckets=False,
)

print(f"Bundle: n_atoms={bundle.n_atoms}, positions_shape={bundle.positions.shape}")
print(f"  - box: {bundle.box}")
print(f"  - cutoff_distance: {bundle.cutoff_distance}")
print(f"  - exception_pairs: {bundle.exception_pairs.shape}")
print()

# Build energy function
energy_fn = energy_fn_from_bundle(
    bundle,
    include_nonbonded=True,
    lj_switch_width=1.0,
    pme_grid_points=int(rec["pme_grid_points"]),
)

# Build NL
displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
neighbor_fn = nl.make_neighbor_list_fn(
    displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance)
)
nbr0 = neighbor_fn.allocate(bundle.positions)
nbr = neighbor_fn.update(bundle.positions, nbr0)

print(f"Neighbor list built successfully")
print(f"  - did_buffer_overflow: {nbr.did_buffer_overflow}")
print(f"  - n_neighbors: {nbr.num_neighbors.sum() if hasattr(nbr, 'num_neighbors') else 'N/A'}")
print()

# Evaluate total energy
e_prolix = float(energy_fn(bundle.positions, neighbor=nbr))
e_gold = float(rec["energy_kcal"])
delta_e = e_prolix - e_gold

print(f"Total Energy:")
print(f"  - Prolix (with NL): {e_prolix:.2f} kcal/mol")
print(f"  - Gold (OpenMM):     {e_gold:.2f} kcal/mol")
print(f"  - Delta:             {delta_e:.2f} kcal/mol ({delta_e/abs(e_gold)*100:.1f}% relative error)")
print()

# Now try to break down components
# Try evaluating WITHOUT the neighbor list to see if it's an NL-specific issue
print(f"Sanity checks:")
e_dense = float(energy_fn(bundle.positions, neighbor=None))
print(f"  - Dense (no NL): {e_dense:.2f} kcal/mol")
print(f"  - NL vs Dense delta: {e_prolix - e_dense:.2f} kcal/mol")
print()

# Check forces
forces_prolix = -jax.grad(energy_fn)(bundle.positions, neighbor=nbr)
forces_gold = np.asarray(rec["forces_kcal_mol_A"], dtype=np.float64)
force_rmse = float(np.sqrt(np.mean((np.asarray(forces_prolix) - forces_gold) ** 2)))

print(f"Forces:")
print(f"  - Force RMSE: {force_rmse:.4f} kcal/(mol*A)")
print(f"  - Max force (Prolix): {np.max(np.abs(forces_prolix)):.2f} kcal/(mol*A)")
print(f"  - Max force (Gold): {np.max(np.abs(forces_gold)):.2f} kcal/(mol*A)")
print()

# Check for unit/scaling issues
# Verify box is in correct units (should be Angstrom)
print(f"Unit checks:")
print(f"  - Box edge (should be ~43 Å for TIP3P solvated protein): {box_A[0]:.2f} Å")
print(f"  - Cutoff (should be 9.0 Å): {bundle.cutoff_distance:.2f} Å")
print(f"  - Typical sigma (should be ~3 Å): {sigmas_A[sigmas_A > 0].mean():.2f} Å")
print()

print(f"SUMMARY:")
print(f"  ✗ Energy error: {delta_e:.1f} kcal/mol (far outside 0.1 kcal/mol band)")
print(f"  ✗ Force error: {force_rmse:.1f} kcal/(mol*A) (far outside 3.0 band)")
print(f"  - Exclusion buffer correctly populated: {bundle.excl_dense_indices is not None}")
print(f"  - Exception pairs correctly threaded: {bundle.exception_pairs.shape[0]} pairs")
print(f"  - NL construction successful: no overflow detected")
print()
print(f"ROOT CAUSE INVESTIGATION NEEDED:")
print(f"  1. Is this an NL-specific bug (E_nl >> E_dense) or general?")
print(f"  2. Is dispersion tail correction the issue (~148 kcal)? ({delta_e - 148:.0f} kcal unexplained)")
print(f"  3. Which energy term is wrong? (LJ vs Coulomb vs Reciprocal)")
print(f"  4. Is there a box/unit mismatch?")
