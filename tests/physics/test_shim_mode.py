"""Test ShimMode: analytical force shim via custom_jvp for bonded terms.

ShimMode.ANALYTICAL registers a @jax.custom_jvp rule that computes bonded
forces analytically instead of tracing through AD. LJ/PME remain AD-traced.

Tests verify:
- ShimMode enum has correct values
- bond_forces analytical output matches AD to within 1e-4 kcal/mol/Å
- energy_with_analytical_shim wraps callable and registers custom_jvp
- Gradient of shim-wrapped energy uses analytical force path
"""

import sys
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from jax_md import space

from prolix.types.shims import ShimMode, energy_with_analytical_shim
from prolix.physics.analytical_forces import bond_forces

# Add tests directory to path for importing test utilities
sys.path.insert(0, str(Path(__file__).parent))


def test_shim_mode_enum():
    """ShimMode enum has correct string values."""
    assert ShimMode.AUTOGRAD.value == "autograd"
    assert ShimMode.ANALYTICAL.value == "analytical"
    assert ShimMode.AUTOGRAD != ShimMode.ANALYTICAL


def test_bond_forces_match_autograd():
    """Analytical bond forces must match AD within 1e-4 kcal/mol/Å."""
    disp_fn, _ = space.free()

    positions = jnp.array([
        [0., 0., 0.],
        [1.5, 0., 0.],
        [3.0, 0., 0.],
    ])

    bond_idx = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    # bond_params convention: [r0, k] — matches PhysicsSystem.bond_params and bond_forces_analytical
    bond_params = jnp.array([
        [1.0, 100.0],  # [r0=1.0 Å, k=100 kcal/mol/Å²]
        [1.0, 100.0],
    ])
    bond_mask = jnp.ones(2, dtype=bool)

    f_analytical = bond_forces(positions, bond_idx, bond_params, bond_mask, disp_fn)

    # Compute AD forces using harmonic potential U = 0.5 * k * (r - r0)^2
    def e_bonds(r):
        dvec = jax.vmap(lambda i, j: disp_fn(r[i], r[j]))(
            bond_idx[:, 0], bond_idx[:, 1]
        )
        dist = jnp.linalg.norm(dvec, axis=1)
        r0 = bond_params[:, 0]
        k = bond_params[:, 1]
        return jnp.sum(bond_mask * 0.5 * k * (dist - r0) ** 2)

    f_ad = -jax.grad(e_bonds)(positions)

    max_diff = jnp.max(jnp.abs(f_analytical - f_ad))
    assert jnp.allclose(f_analytical, f_ad, atol=1e-4), (
        f"Bond forces mismatch. Max diff: {max_diff}"
    )


def test_energy_with_analytical_shim_returns_callable():
    """energy_with_analytical_shim returns a callable that works with bundles."""
    from test_molecular_bundle import _minimal_bundle

    def base_energy(bundle):
        return jnp.array(0.0)

    def bonded_forces(bundle):
        return jnp.zeros_like(bundle.positions)

    bundle = _minimal_bundle()
    wrapped = energy_with_analytical_shim(base_energy, bonded_forces)
    result = wrapped(bundle)

    assert isinstance(result, jnp.ndarray)
    assert result.shape == ()


def test_shim_grad_uses_analytical_path():
    """Gradient of shim-wrapped energy uses analytical forces with correct sign."""
    import equinox as eqx
    from prolix.types.bundles import (
        MolecularBundle, MolecularShapeSpec, _bucket_idx,
        ATOM_BUCKETS, BOND_BUCKETS, ANGLE_BUCKETS, DIHEDRAL_BUCKETS,
        WATER_BUCKETS, EXCL_BUCKETS, CMAP_BUCKETS, EXCEPTION_BUCKETS,
    )

    n_atoms = 10
    a = ATOM_BUCKETS[0]

    def base_energy(bundle):
        return jnp.sum(bundle.positions ** 2)

    def bonded_forces(bundle):
        return -2.0 * bundle.positions  # correct: -dE/dr for E=sum(r^2)

    # Build shape_spec with bucket indices
    atom_bucket_idx = _bucket_idx(n_atoms, ATOM_BUCKETS)
    spec = MolecularShapeSpec(
        atom_bucket_idx=atom_bucket_idx, bond_bucket_idx=0, angle_bucket_idx=0,
        dihedral_bucket_idx=0, water_bucket_idx=0, excl_bucket_idx=0,
        cmap_bucket_idx=0, exception_bucket_idx=0,
        has_pbc=False, has_implicit_solvent=False, boundary_condition="free",
    )
    b = BOND_BUCKETS[0]

    # Use nonzero positions so sign errors are detectable
    pos = jnp.ones((a, 3)) * 0.5
    bundle = MolecularBundle(
        positions=pos,
        charges=jnp.zeros(a),
        sigmas=jnp.ones(a),
        epsilons=jnp.ones(a),
        radii=jnp.ones(a),
        scaled_radii=jnp.ones(a),
        atom_mask=jnp.concatenate([jnp.ones(n_atoms, bool), jnp.zeros(a - n_atoms, bool)]),
        n_atoms=jnp.array(n_atoms, dtype=jnp.int32),
        box=jnp.zeros((3, 3)),
        bond_idx=jnp.zeros((b, 2), jnp.int32),
        bond_params=jnp.zeros((b, 2)),
        bond_mask=jnp.zeros(b, bool),
        n_bonds=jnp.array(0, dtype=jnp.int32),
        angle_idx=jnp.zeros((ANGLE_BUCKETS[0], 3), jnp.int32),
        angle_params=jnp.zeros((ANGLE_BUCKETS[0], 2)),
        angle_mask=jnp.zeros(ANGLE_BUCKETS[0], bool),
        n_angles=jnp.array(0, dtype=jnp.int32),
        dihedral_idx=jnp.zeros((256, 4), jnp.int32),
        dihedral_params=jnp.zeros((256, 4)),
        dihedral_mask=jnp.zeros(256, bool),
        n_dihedrals=jnp.array(0, dtype=jnp.int32),
        improper_idx=jnp.zeros((256, 4), jnp.int32),
        improper_params=jnp.zeros((256, 3)),
        improper_mask=jnp.zeros(256, bool),
        improper_is_periodic=jnp.array(False),
        n_impropers=jnp.array(0, dtype=jnp.int32),
        urey_bradley_idx=jnp.zeros((256, 3), jnp.int32),
        urey_bradley_params=jnp.zeros((256, 2)),
        urey_bradley_mask=jnp.zeros(256, bool),
        n_urey_bradley=jnp.array(0, dtype=jnp.int32),
        cmap_torsion_idx=jnp.zeros((16, 8), jnp.int32),
        cmap_energy_grids=jnp.zeros((16, 24, 24)),
        cmap_mask=jnp.zeros(16, bool),
        n_cmap=jnp.array(0, dtype=jnp.int32),
        water_indices=jnp.zeros((16, 3), jnp.int32),
        water_mask=jnp.zeros(16, bool),
        n_waters=jnp.array(0, dtype=jnp.int32),
        excl_indices=jnp.zeros((512, 2), jnp.int32),
        excl_scales_vdw=jnp.zeros(512),
        excl_scales_elec=jnp.zeros(512),
        excl_mask=jnp.zeros(512, bool),
        n_excl=jnp.array(0, dtype=jnp.int32),
        exception_pairs=jnp.zeros((512, 2), jnp.int32),
        exception_sigmas=jnp.zeros(512),
        exception_epsilons=jnp.zeros(512),
        exception_chargeprods=jnp.zeros(512),
        exception_mask=jnp.zeros(512, bool),
        n_exception_pairs=jnp.array(0, dtype=jnp.int32),
        pme_alpha=jnp.array(0.0),
        cutoff_distance=jnp.array(9.0),
        shape_spec=spec,
    )

    wrapped = energy_with_analytical_shim(base_energy, bonded_forces)

    # Use eqx.filter_grad to differentiate only through float arrays
    grad_fn = eqx.filter_grad(wrapped)
    grad_bundle = grad_fn(bundle)

    # dE/dr for E=sum(r^2) is 2r; grad_bundle.positions should be +2r
    expected_grad = 2.0 * pos
    assert jnp.allclose(grad_bundle.positions, expected_grad, atol=1e-4), (
        f"Expected +2r gradient, got: {grad_bundle.positions[:2]}"
    )
