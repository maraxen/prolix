"""V1: EnsemblePlan(B=1) parity vs settle_langevin."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


from prolix.api.bundle_md import (
    active_positions,
    as_integration_scalars,
    energy_fn_from_bundle,
    displacement_fn_for_bundle,
    masses_for_bundle,
)
from prolix.api.ensemble_plan import EnsemblePlan
from prolix.physics import settle
from prolix.physics.system import make_bundle_from_system
from prolix.physics.settle import settle_positions


def _one_water_bundle():
    """Single TIP3P water as MolecularBundle (minimal V1 fixture)."""
    from prolix.typing import PhysicsSystem

    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399, 0.9266, 0.0],
        ],
        dtype=jnp.float64,
    )
    water_indices = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    n = 3
    zeros = jnp.zeros(n)
    ones_b = jnp.ones(n, dtype=bool)
    zeros_b = jnp.zeros(n, dtype=bool)
    empty2 = jnp.zeros((0, 2), dtype=jnp.int32)
    empty3 = jnp.zeros((0, 3), dtype=jnp.int32)
    empty_p2 = jnp.zeros((0, 2))
    empty_m = jnp.zeros(0, dtype=bool)
    empty_dih_p = jnp.zeros((0, 1, 3))

    sys = PhysicsSystem(
        positions=positions,
        charges=zeros,
        sigmas=ones_b.astype(jnp.float64) * 1e-6,
        epsilons=ones_b.astype(jnp.float64) * 1e-6,
        radii=jnp.ones(n),
        scaled_radii=jnp.ones(n),
        masses=jnp.array([15.999, 1.008, 1.008]),
        element_ids=jnp.zeros(n, dtype=jnp.int32),
        atom_mask=ones_b,
        is_hydrogen=zeros_b,
        is_backbone=zeros_b,
        is_heavy=ones_b,
        protein_atom_mask=zeros_b,
        water_atom_mask=ones_b,
        bonds=empty2,
        bond_params=empty_p2,
        bond_mask=empty_m,
        angles=empty3,
        angle_params=empty_p2,
        angle_mask=empty_m,
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=empty_dih_p,
        dihedral_mask=empty_m,
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=empty_dih_p,
        improper_mask=empty_m,
        water_indices=water_indices,
    )
    bundle = make_bundle_from_system(sys, boundary_condition="free")
    pos = active_positions(bundle)
    pos = settle_positions(pos, pos, water_indices)
    # Write settled coords back into a fresh bundle via system replace
    sys = PhysicsSystem(
        positions=pos,
        charges=zeros,
        sigmas=ones_b.astype(jnp.float64) * 1e-6,
        epsilons=ones_b.astype(jnp.float64) * 1e-6,
        radii=jnp.ones(n),
        scaled_radii=jnp.ones(n),
        masses=jnp.array([15.999, 1.008, 1.008]),
        element_ids=jnp.zeros(n, dtype=jnp.int32),
        atom_mask=ones_b,
        is_hydrogen=zeros_b,
        is_backbone=zeros_b,
        is_heavy=ones_b,
        protein_atom_mask=zeros_b,
        water_atom_mask=ones_b,
        bonds=empty2,
        bond_params=empty_p2,
        bond_mask=empty_m,
        angles=empty3,
        angle_params=empty_p2,
        angle_mask=empty_m,
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=empty_dih_p,
        dihedral_mask=empty_m,
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=empty_dih_p,
        improper_mask=empty_m,
        water_indices=water_indices,
    )
    return make_bundle_from_system(sys, boundary_condition="free")


def _reference_trajectory(bundle, *, n_steps, dt, kT, seed):
    """Direct settle_langevin using the same bundle-backed helpers as EnsemblePlan."""
    from prolix.physics.kups_adapter import gamma_ps_to_akma

    energy_fn = energy_fn_from_bundle(bundle)
    _, shift_fn = displacement_fn_for_bundle(bundle)
    masses = masses_for_bundle(bundle)
    water_indices = bundle.water_indices[: int(bundle.n_waters)]
    # settle_langevin expects gamma already in AKMA (caller converts from ps⁻¹).
    init_fn, apply_fn = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt,
        kT=kT,
        gamma=gamma_ps_to_akma(10.0),
        mass=masses,
        water_indices=water_indices,
        project_ou_momentum_rigid=True,
    )
    key = jax.random.PRNGKey(seed)
    state = init_fn(key, active_positions(bundle))

    def step_fn(carry, _):
        new_state = apply_fn(carry, kT=kT, dt=dt)
        return new_state, new_state.position

    _, positions = jax.lax.scan(step_fn, state, None, length=n_steps)
    return positions


def _solvated_ake_bundle():
    """Minimal solvated explicit-solvent smoke fixture (TIP3P one-water, same as W4)."""
    return _one_water_bundle()


def test_v1_harness_runs_and_returns_trajectory():
    bundle = _one_water_bundle()
    ep = EnsemblePlan.from_bundle(bundle)
    # Pre-XR tests used AKMA dt; keep escape hatch for bitwise parity gates.
    traj = ep.run(n_steps=5, dt=0.5, kT=0.596, seed=42, dt_unit="akma")
    assert traj.n_steps == 5
    assert traj.positions.shape == (5, 3, 3)
    assert jnp.all(jnp.isfinite(traj.positions))


def test_v1_one_water_parity_vs_settle_langevin():
    """V1: EnsemblePlan.run matches direct settle_langevin (1 water, short run)."""
    jax.config.update("jax_enable_x64", True)
    bundle = _one_water_bundle()
    n_steps = 50
    dt = 0.5
    kT = 0.596
    seed = 42

    ref = _reference_trajectory(bundle, n_steps=n_steps, dt=dt, kT=kT, seed=seed)
    ep = EnsemblePlan.from_bundle(bundle)
    out = ep.run(n_steps=n_steps, dt=dt, kT=kT, seed=seed, dt_unit="akma")
    rmsd = jnp.sqrt(jnp.mean((out.positions - ref) ** 2))
    assert rmsd < 1e-12, f"V1 parity failed: RMSD={rmsd:.3e} Å"


def test_v1_solvated_ake_1k_parity_vs_settle_langevin():
    """V1 (#263): solvated AKE smoke, 1k steps — EnsemblePlan vs settle_langevin."""
    jax.config.update("jax_enable_x64", True)
    bundle = _solvated_ake_bundle()
    n_steps = 1000
    dt = 0.5
    kT = 0.596
    seed = 42

    ref = _reference_trajectory(bundle, n_steps=n_steps, dt=dt, kT=kT, seed=seed)
    out = EnsemblePlan.from_bundle(bundle).run(
        n_steps=n_steps, dt=dt, kT=kT, seed=seed, dt_unit="akma"
    )
    rmsd = jnp.sqrt(jnp.mean((out.positions - ref) ** 2))
    assert rmsd < 1e-12, f"V1 solvated AKE 1k parity failed: RMSD={rmsd:.3e} Å"
