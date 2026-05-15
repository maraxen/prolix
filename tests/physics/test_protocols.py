"""Test EnergyFn and IntegratorFn @runtime_checkable protocols.

Tests verify:
- EnergyFn accepts MolecularBundle and returns scalar
- IntegratorFn accepts IntegratorState and returns IntegratorState
- isinstance() structural checks work for runtime type validation
"""

import pytest
import jax
import jax.numpy as jnp
from prolix.types.protocols import EnergyFn, IntegratorFn
from prolix.types.integrators import LangevinState
from prolix.types.bundles import (
    MolecularBundle,
    MolecularShapeSpec,
    ATOM_BUCKETS,
    BOND_BUCKETS,
    ANGLE_BUCKETS,
)


def _minimal_bundle(n_atoms=10, n_bonds=5):
    """Create a minimal MolecularBundle for testing."""
    a = ATOM_BUCKETS[0]
    b = BOND_BUCKETS[0]

    spec = MolecularShapeSpec(
        n_atoms=n_atoms,
        n_bonds=n_bonds,
        n_angles=0,
        n_dihedrals=0,
        n_impropers=0,
        n_urey_bradley=0,
        n_waters=0,
        n_excl=0,
        n_cmap=0,
        n_exception_pairs=0,
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
        box=jnp.zeros((3, 3)),
        bond_idx=jnp.zeros((b, 2), dtype=jnp.int32),
        bond_params=jnp.zeros((b, 2)),
        bond_mask=jnp.concatenate(
            [jnp.ones(n_bonds, dtype=bool), jnp.zeros(b - n_bonds, dtype=bool)]
        ),
        angle_idx=jnp.zeros((ANGLE_BUCKETS[0], 3), dtype=jnp.int32),
        angle_params=jnp.zeros((ANGLE_BUCKETS[0], 2)),
        angle_mask=jnp.zeros(ANGLE_BUCKETS[0], dtype=bool),
        dihedral_idx=jnp.zeros((256, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((256, 4)),
        dihedral_mask=jnp.zeros(256, dtype=bool),
        improper_idx=jnp.zeros((256, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((256, 3)),
        improper_mask=jnp.zeros(256, dtype=bool),
        improper_is_periodic=jnp.array(False),
        urey_bradley_idx=jnp.zeros((256, 3), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((256, 2)),
        urey_bradley_mask=jnp.zeros(256, dtype=bool),
        cmap_torsion_idx=jnp.zeros((16, 8), dtype=jnp.int32),
        cmap_energy_grids=jnp.zeros((16, 24, 24)),
        cmap_mask=jnp.zeros(16, dtype=bool),
        water_indices=jnp.zeros((16, 3), dtype=jnp.int32),
        water_mask=jnp.zeros(16, dtype=bool),
        excl_indices=jnp.zeros((512, 2), dtype=jnp.int32),
        excl_scales_vdw=jnp.zeros(512),
        excl_scales_elec=jnp.zeros(512),
        excl_mask=jnp.zeros(512, dtype=bool),
        exception_pairs=jnp.zeros((512, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros(512),
        exception_epsilons=jnp.zeros(512),
        exception_chargeprods=jnp.zeros(512),
        exception_mask=jnp.zeros(512, dtype=bool),
        pme_alpha=jnp.array(0.0),
        cutoff_distance=jnp.array(9.0),
        shape_spec=spec,
    )


def _make_state():
    """Create a minimal LangevinState for testing."""
    return LangevinState(
        positions=jnp.zeros((256, 3)),
        momenta=jnp.zeros((256, 3)),
        forces=jnp.zeros((256, 3)),
        key=jax.random.PRNGKey(0),
        box=jnp.zeros((3, 3)),
    )


def test_energy_fn_protocol_isinstance():
    """EnergyFn isinstance check accepts plain callable returning scalar."""
    def my_energy(bundle):
        return jnp.sum(bundle.positions)

    assert isinstance(my_energy, EnergyFn)


def test_integrator_fn_protocol_isinstance():
    """IntegratorFn isinstance check accepts plain callable returning state."""
    def my_step(state: LangevinState) -> LangevinState:
        return state

    assert isinstance(my_step, IntegratorFn)


def test_energy_fn_callable_with_bundle():
    """EnergyFn can be called with a MolecularBundle and returns scalar."""
    def my_energy(bundle):
        return jnp.sum(bundle.positions)

    bundle = _minimal_bundle()
    result = my_energy(bundle)
    assert result.shape == ()
    assert isinstance(result, jnp.ndarray)


def test_integrator_fn_callable_with_state():
    """IntegratorFn can be called with a state and returns matching state."""
    def my_step(state):
        return state

    state = _make_state()
    result = my_step(state)
    assert result.positions.shape == state.positions.shape
    assert result.momenta.shape == state.momenta.shape
    assert result.forces.shape == state.forces.shape
