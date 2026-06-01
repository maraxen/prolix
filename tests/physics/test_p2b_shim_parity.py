"""Test analytical vs autograd force parity for bonded terms.

Tests verify:
- bond_forces matches jax.grad of harmonic bond energy
- angle_forces_analytical matches jax.grad of harmonic angle energy
- Per-term forces agree to <1e-4 kcal/mol/Å in float64
"""

import pytest
import jax
import jax.numpy as jnp
from jax_md import space
from prolix.physics.analytical_forces import bond_forces, angle_forces_analytical
from prolix.types.bundles import (
    MolecularBundle,
    MolecularShapeSpec,
    _bucket_idx,
    ATOM_BUCKETS,
    BOND_BUCKETS,
    ANGLE_BUCKETS,
    DIHEDRAL_BUCKETS,
    WATER_BUCKETS,
    EXCL_BUCKETS,
    CMAP_BUCKETS,
    EXCEPTION_BUCKETS,
)


def _minimal_bundle(n_atoms=10, n_bonds=5, n_angles=0):
    """Create a minimal MolecularBundle for testing."""
    a = ATOM_BUCKETS[0]
    b = BOND_BUCKETS[0]

    # Build shape_spec with bucket indices (not raw counts)
    atom_bucket_idx = _bucket_idx(n_atoms, ATOM_BUCKETS)
    bond_bucket_idx = _bucket_idx(max(n_bonds, 1), BOND_BUCKETS)
    angle_bucket_idx = _bucket_idx(max(n_angles, 1), ANGLE_BUCKETS)

    spec = MolecularShapeSpec(
        atom_bucket_idx=atom_bucket_idx,
        bond_bucket_idx=bond_bucket_idx,
        angle_bucket_idx=angle_bucket_idx,
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
        positions=jnp.zeros((a, 3)),
        charges=jnp.zeros(a),
        sigmas=jnp.ones(a),
        epsilons=jnp.ones(a),
        radii=jnp.ones(a),
        scaled_radii=jnp.ones(a),
        atom_mask=jnp.concatenate(
            [jnp.ones(n_atoms, dtype=bool), jnp.zeros(a - n_atoms, dtype=bool)]
        ),
        n_atoms=jnp.array(n_atoms, dtype=jnp.int32),
        box=jnp.zeros((3, 3)),
        bond_idx=jnp.zeros((b, 2), dtype=jnp.int32),
        bond_params=jnp.zeros((b, 2)),
        bond_mask=jnp.concatenate(
            [jnp.ones(n_bonds, dtype=bool), jnp.zeros(b - n_bonds, dtype=bool)]
        ),
        n_bonds=jnp.array(n_bonds, dtype=jnp.int32),
        angle_idx=jnp.zeros((ANGLE_BUCKETS[0], 3), dtype=jnp.int32),
        angle_params=jnp.zeros((ANGLE_BUCKETS[0], 2)),
        angle_mask=jnp.zeros(ANGLE_BUCKETS[0], dtype=bool),
        n_angles=jnp.array(0, dtype=jnp.int32),
        dihedral_idx=jnp.zeros((256, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((256, 4)),
        dihedral_mask=jnp.zeros(256, dtype=bool),
        n_dihedrals=jnp.array(0, dtype=jnp.int32),
        improper_idx=jnp.zeros((256, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((256, 3)),
        improper_mask=jnp.zeros(256, dtype=bool),
        improper_is_periodic=jnp.array(False),
        n_impropers=jnp.array(0, dtype=jnp.int32),
        urey_bradley_idx=jnp.zeros((256, 3), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((256, 2)),
        urey_bradley_mask=jnp.zeros(256, dtype=bool),
        n_urey_bradley=jnp.array(0, dtype=jnp.int32),
        cmap_torsion_idx=jnp.zeros((16, 8), dtype=jnp.int32),
        cmap_energy_grids=jnp.zeros((16, 24, 24)),
        cmap_mask=jnp.zeros(16, dtype=bool),
        n_cmap=jnp.array(0, dtype=jnp.int32),
        water_indices=jnp.zeros((16, 3), dtype=jnp.int32),
        water_mask=jnp.zeros(16, dtype=bool),
        n_waters=jnp.array(0, dtype=jnp.int32),
        excl_indices=jnp.zeros((512, 2), dtype=jnp.int32),
        excl_scales_vdw=jnp.zeros(512),
        excl_scales_elec=jnp.zeros(512),
        excl_mask=jnp.zeros(512, dtype=bool),
        n_excl=jnp.array(0, dtype=jnp.int32),
        exception_pairs=jnp.zeros((512, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros(512),
        exception_epsilons=jnp.zeros(512),
        exception_chargeprods=jnp.zeros(512),
        exception_mask=jnp.zeros(512, dtype=bool),
        n_exception_pairs=jnp.array(0, dtype=jnp.int32),
        pme_alpha=jnp.array(0.0),
        cutoff_distance=jnp.array(9.0),
        shape_spec=spec,
    )


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_bond_forces_vs_ad_on_bundle(dtype):
    """Bond forces: analytical vs autograd parity on 10-atom bundle."""
    jax.config.update("jax_enable_x64", True)

    # Create bundle: 10 atoms, 5 bonds
    bundle = _minimal_bundle(n_atoms=10, n_bonds=5)

    # Set up positions: linear chain along x-axis
    # Atoms at x = 0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5 Å
    positions = jnp.arange(10, dtype=dtype)[:, None] * 1.5 * jnp.array([1.0, 0.0, 0.0], dtype=dtype)
    positions = jnp.pad(positions, ((0, ATOM_BUCKETS[0] - 10), (0, 0)), constant_values=0.0)

    # Set up bond parameters: r0 = 1.5 Å, k = 100 kcal/mol/Å²
    bond_idx = jnp.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ], dtype=jnp.int32)
    bond_params = jnp.array([
        [1.5, 100.0],
        [1.5, 100.0],
        [1.5, 100.0],
        [1.5, 100.0],
        [1.5, 100.0],
    ], dtype=dtype)
    bond_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)

    # Pad to bundle size
    bond_idx_padded = jnp.pad(bond_idx, ((0, BOND_BUCKETS[0] - 5), (0, 0)), constant_values=0)
    bond_params_padded = jnp.pad(bond_params, ((0, BOND_BUCKETS[0] - 5), (0, 0)), constant_values=0.0)
    bond_mask_padded = jnp.pad(bond_mask, (0, BOND_BUCKETS[0] - 5), constant_values=0.0)

    # Displacement function
    displacement_fn, _ = space.free()

    # Compute analytical forces
    f_analytical = bond_forces(
        positions,
        bond_idx_padded,
        bond_params_padded,
        bond_mask_padded,
        displacement_fn,
    )

    # Compute AD forces via autograd
    def harmonic_bond_energy(r):
        """Sum of harmonic bond energies."""
        energy = jnp.float32(0.0)
        for b_idx in range(5):
            i = bond_idx[b_idx, 0]
            j = bond_idx[b_idx, 1]
            r0 = bond_params[b_idx, 0]
            k = bond_params[b_idx, 1]
            r_ij = displacement_fn(r[i], r[j])
            dist = jnp.linalg.norm(r_ij)
            e = 0.5 * k * (dist - r0) ** 2
            energy = energy + e
        return energy

    grad_fn = jax.grad(harmonic_bond_energy)
    f_ad = -grad_fn(positions)

    # Compare forces on real atoms only
    f_analytical_real = f_analytical[:10]
    f_ad_real = f_ad[:10]

    assert jnp.allclose(f_analytical_real, f_ad_real, atol=1e-4), (
        f"Bond force parity failed.\n"
        f"Analytical: {f_analytical_real}\n"
        f"AD: {f_ad_real}\n"
        f"Diff: {jnp.abs(f_analytical_real - f_ad_real)}"
    )


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_angle_forces_vs_ad_on_bundle(dtype):
    """Angle forces: analytical vs autograd parity on 12-atom zigzag bundle."""
    jax.config.update("jax_enable_x64", True)

    # Create bundle: 12 atoms, 5 angles
    bundle = _minimal_bundle(n_atoms=12, n_bonds=0, n_angles=5)

    # Set up positions: zigzag chain with 109.5° (tetrahedral) angles
    # This simulates sp³ hybridization
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.25, 1.225, 0.0],
        [1.5, 1.7, 0.0],
        [0.0, 1.7, 0.0],
        [0.75, 2.925, 0.0],
        [0.0, 2.0, 0.0],
        [1.5, 3.4, 0.0],
        [2.25, 2.175, 0.0],
        [3.75, 2.175, 0.0],
        [4.5, 3.4, 0.0],
        [5.25, 2.175, 0.0],
    ], dtype=dtype)
    positions = jnp.pad(positions, ((0, ATOM_BUCKETS[0] - 12), (0, 0)), constant_values=0.0)

    # Set up angle parameters: theta0 = 1.911 rad ≈ 109.5°, k = 50 kcal/mol/rad²
    angle_idx = jnp.array([
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
    ], dtype=jnp.int32)
    angle_params = jnp.array([
        [1.911, 50.0],
        [1.911, 50.0],
        [1.911, 50.0],
        [1.911, 50.0],
        [1.911, 50.0],
    ], dtype=dtype)
    angle_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)

    # NOTE: angle_forces_analytical uses a Python loop that will iterate over ALL
    # angle entries including padded ones. To avoid NaN from (0,0,0) angles,
    # we only pass the 5 real angles and use a mask-only setup (no padding in indices).
    # This is the constraint of the analytical function's implementation.
    # For efficiency, only pass the minimal needed angles:
    angle_idx_minimal = angle_idx
    angle_params_minimal = angle_params
    angle_mask_minimal = angle_mask

    # Pad positions and create padded arrays matching bundle structure for gradient computation
    angle_idx_padded = jnp.pad(angle_idx, ((0, ANGLE_BUCKETS[0] - 5), (0, 0)), constant_values=0)
    angle_params_padded = jnp.pad(angle_params, ((0, ANGLE_BUCKETS[0] - 5), (0, 0)), constant_values=0.0)
    angle_mask_padded = jnp.pad(angle_mask, (0, ANGLE_BUCKETS[0] - 5), constant_values=0.0)

    # Displacement function
    displacement_fn, _ = space.free()

    # Compute analytical forces using only the 5 real angles
    # NOTE: angle_forces_analytical uses a Python loop that may not JIT-compile.
    # We call it directly without JIT wrapping, passing only real angles.
    f_analytical = angle_forces_analytical(
        positions,
        angle_idx_minimal,
        angle_params_minimal,
        displacement_fn,
        angle_mask_minimal,
    )

    # Compute AD forces via autograd
    def harmonic_angle_energy(r):
        """Sum of harmonic angle energies."""
        energy = jnp.float32(0.0)
        for a_idx in range(5):
            i = angle_idx[a_idx, 0]
            j = angle_idx[a_idx, 1]
            k_atom = angle_idx[a_idx, 2]
            theta0 = angle_params[a_idx, 0]
            k = angle_params[a_idx, 1]

            v_ji = displacement_fn(r[i], r[j])
            v_jk = displacement_fn(r[k_atom], r[j])

            d_ji = jnp.linalg.norm(v_ji) + jnp.float32(1e-12)
            d_jk = jnp.linalg.norm(v_jk) + jnp.float32(1e-12)

            cos_theta_a = jnp.sum(v_ji * v_jk) / (d_ji * d_jk)
            cos_theta_a = jnp.clip(cos_theta_a, -0.999999, 0.999999)
            theta_a = jnp.arccos(cos_theta_a)

            e = 0.5 * k * (theta_a - theta0) ** 2
            energy = energy + angle_mask[a_idx] * e
        return energy

    grad_fn = jax.grad(harmonic_angle_energy)
    f_ad = -grad_fn(positions)

    # Compare forces on real atoms only
    f_analytical_real = f_analytical[:12]
    f_ad_real = f_ad[:12]

    assert jnp.allclose(f_analytical_real, f_ad_real, atol=1e-4), (
        f"Angle force parity failed.\n"
        f"Analytical: {f_analytical_real}\n"
        f"AD: {f_ad_real}\n"
        f"Diff: {jnp.abs(f_analytical_real - f_ad_real)}"
    )


# TODO: dihedral force parity is deferred — dihedral_forces_analytical expects
# params (D, N_terms, 3) but MolecularBundle stores dihedral_params (D, 4).
# Format incompatibility requires param reshape; scope for follow-up task.
