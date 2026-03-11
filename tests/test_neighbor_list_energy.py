"""TDD tests for neighbor-list-based energy functions.

Tests that the O(N*K) neighbor list path produces energies matching
the O(N^2) dense path within tolerance for the same protein system.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_toy_system(n_atoms: int = 64, seed: int = 42):
    """Create a toy protein-like system with random positions and params."""
    rng = np.random.RandomState(seed)

    # Positions in a 30Å box to keep atoms within reasonable NL cutoff
    positions = rng.uniform(0, 30.0, (n_atoms, 3)).astype(np.float32)

    # LJ params (typical protein range)
    sigmas = rng.uniform(1.5, 4.0, n_atoms).astype(np.float32)
    epsilons = rng.uniform(0.01, 0.3, n_atoms).astype(np.float32)

    # Charges (mix of positive, negative, neutral)
    charges = rng.uniform(-0.8, 0.8, n_atoms).astype(np.float32)

    # Radii for GB
    radii = rng.uniform(1.0, 2.5, n_atoms).astype(np.float32)
    scaled_radii = radii * rng.uniform(0.7, 0.9, n_atoms).astype(np.float32)

    # All atoms are real (no padding)
    atom_mask = jnp.ones(n_atoms, dtype=jnp.bool_)

    return {
        "positions": jnp.array(positions),
        "sigmas": jnp.array(sigmas),
        "epsilons": jnp.array(epsilons),
        "charges": jnp.array(charges),
        "radii": jnp.array(radii),
        "scaled_radii": jnp.array(scaled_radii),
        "atom_mask": atom_mask,
    }


def _build_dense_neighbor_idx(n_atoms: int) -> jnp.ndarray:
    """Build a 'complete' neighbor list that includes ALL other atoms.

    This should reproduce the N^2 result exactly.
    Shape: (N, N-1) — every atom lists all others as neighbors.
    """
    # For each atom i, neighbors are [0, 1, ..., i-1, i+1, ..., N-1]
    idx = np.zeros((n_atoms, n_atoms - 1), dtype=np.int32)
    for i in range(n_atoms):
        neighbors = list(range(i)) + list(range(i + 1, n_atoms))
        idx[i] = neighbors
    return jnp.array(idx)


# ── LJ tests ────────────────────────────────────────────────────────────────


class TestLJNeighborList:
    """Tests that _lj_energy_neighbor_list matches _lj_energy_masked."""

    def test_lj_nl_matches_dense_exact(self):
        """With a complete neighbor list, NL LJ should exactly match N^2 LJ."""
        from prolix.batched_energy import _lj_energy_masked
        from prolix.batched_energy import _lj_energy_neighbor_list

        sys = _make_toy_system(n_atoms=32)
        displacement_fn, _ = space.free()
        neighbor_idx = _build_dense_neighbor_idx(32)

        e_dense = _lj_energy_masked(
            sys["positions"],
            sys["sigmas"],
            sys["epsilons"],
            sys["atom_mask"],
            displacement_fn,
        )

        e_nl = _lj_energy_neighbor_list(
            sys["positions"],
            sys["sigmas"],
            sys["epsilons"],
            neighbor_idx,
        )

        np.testing.assert_allclose(
            float(e_dense), float(e_nl), rtol=1e-4,
            err_msg="NL LJ energy should match N^2 LJ with complete neighbor list",
        )

    def test_lj_nl_with_cutoff_is_approximate(self):
        """With a finite cutoff, NL LJ should be close but not exact."""
        from prolix.batched_energy import _lj_energy_masked
        from prolix.batched_energy import _lj_energy_neighbor_list

        sys = _make_toy_system(n_atoms=32)
        displacement_fn, _ = space.free()

        # Build neighbor list with ~15Å cutoff (should capture most LJ interactions)
        positions = sys["positions"]
        n = 32
        # Manual cutoff-based NL
        dists = np.linalg.norm(
            np.array(positions)[:, None, :] - np.array(positions)[None, :, :],
            axis=-1,
        )
        cutoff = 15.0
        max_k = 0
        for i in range(n):
            k = int(np.sum((dists[i] < cutoff) & (dists[i] > 0)))
            max_k = max(max_k, k)

        neighbor_idx = np.full((n, max_k), n, dtype=np.int32)  # sentinel = N
        for i in range(n):
            nbrs = np.where((dists[i] < cutoff) & (dists[i] > 0))[0]
            neighbor_idx[i, : len(nbrs)] = nbrs

        e_dense = _lj_energy_masked(
            positions, sys["sigmas"], sys["epsilons"],
            sys["atom_mask"], displacement_fn,
        )
        e_nl = _lj_energy_neighbor_list(
            positions, sys["sigmas"], sys["epsilons"],
            jnp.array(neighbor_idx),
        )

        # LJ decays as 1/r^6, so 15Å cutoff should capture >99% of energy
        np.testing.assert_allclose(
            float(e_dense), float(e_nl), rtol=0.05,
            err_msg="NL LJ with 15Å cutoff should be within 5% of dense",
        )

    def test_lj_nl_is_differentiable(self):
        """Gradient of NL LJ should not produce NaN."""
        from prolix.batched_energy import _lj_energy_neighbor_list

        sys = _make_toy_system(n_atoms=16)
        neighbor_idx = _build_dense_neighbor_idx(16)

        grad_fn = jax.grad(
            lambda r: _lj_energy_neighbor_list(
                r, sys["sigmas"], sys["epsilons"], neighbor_idx,
            )
        )
        grads = grad_fn(sys["positions"])
        assert not jnp.any(jnp.isnan(grads)), "LJ NL gradient contains NaN"

    def test_lj_nl_with_padding_atoms(self):
        """Padding atoms (idx >= N) in neighbor list should be handled safely."""
        from prolix.batched_energy import _lj_energy_neighbor_list

        sys = _make_toy_system(n_atoms=16)
        # Create NL with lots of padding (sentinel = N)
        n = 16
        max_k = 32  # Over-allocated
        neighbor_idx = np.full((n, max_k), n, dtype=np.int32)  # all padding
        # Fill first few with real neighbors
        for i in range(n):
            real_nbrs = list(range(i)) + list(range(i + 1, n))
            neighbor_idx[i, : len(real_nbrs)] = real_nbrs

        e = _lj_energy_neighbor_list(
            sys["positions"], sys["sigmas"], sys["epsilons"],
            jnp.array(neighbor_idx),
        )
        assert jnp.isfinite(e), "LJ NL should handle padded neighbor lists"


# ── GB tests ────────────────────────────────────────────────────────────────


class TestGBNeighborList:
    """Tests that GB NL path matches the N^2 dense path."""

    def test_born_radii_nl_matches_dense(self):
        """With complete NL, Born radii should match N^2 calculation."""
        from prolix.physics.generalized_born import (
            compute_born_radii,
            compute_born_radii_neighbor_list,
        )

        sys = _make_toy_system(n_atoms=32)
        neighbor_idx = _build_dense_neighbor_idx(32)

        br_dense = compute_born_radii(
            sys["positions"], sys["radii"],
            mask=sys["atom_mask"],
            scaled_radii=sys["scaled_radii"],
        )
        br_nl = compute_born_radii_neighbor_list(
            sys["positions"], sys["radii"],
            neighbor_idx,
            scaled_radii=sys["scaled_radii"],
        )

        np.testing.assert_allclose(
            np.array(br_dense), np.array(br_nl), rtol=0.01,
            err_msg="NL Born radii should match dense with complete NL",
        )

    def test_gb_energy_nl_matches_dense(self):
        """With complete NL, GB energy should match N^2 calculation."""
        from prolix.physics.generalized_born import (
            compute_gb_energy,
            compute_gb_energy_neighbor_list,
        )

        sys = _make_toy_system(n_atoms=32)
        neighbor_idx = _build_dense_neighbor_idx(32)

        mask_ij = sys["atom_mask"][:, None] & sys["atom_mask"][None, :]
        n = 32
        energy_mask = mask_ij * (1.0 - jnp.eye(n))

        e_dense, _ = compute_gb_energy(
            sys["positions"], sys["charges"], sys["radii"],
            mask=sys["atom_mask"],
            energy_mask=energy_mask,
            scaled_radii=sys["scaled_radii"],
        )
        e_nl, _ = compute_gb_energy_neighbor_list(
            sys["positions"], sys["charges"], sys["radii"],
            neighbor_idx,
        )

        np.testing.assert_allclose(
            float(e_dense), float(e_nl), rtol=0.05,
            err_msg="NL GB energy should match dense with complete NL",
        )


# ── Integrated energy tests ────────────────────────────────────────────────


class TestIntegratedNLEnergy:
    """Tests the full energy function with neighbor list path."""

    def test_single_energy_nl_import_and_call(self):
        """single_padded_energy_nl should be importable and callable."""
        from prolix.batched_energy import single_padded_energy_nl

        # Verify it's callable
        assert callable(single_padded_energy_nl)

    def test_gradient_through_nl_energy(self):
        """jax.grad through NL energy path should produce finite gradients."""
        from prolix.batched_energy import _lj_energy_neighbor_list

        sys = _make_toy_system(n_atoms=32)
        neighbor_idx = _build_dense_neighbor_idx(32)

        def energy_fn(r):
            return _lj_energy_neighbor_list(
                r, sys["sigmas"], sys["epsilons"], neighbor_idx,
            )

        grad_fn = jax.grad(energy_fn)
        g = grad_fn(sys["positions"])
        assert jnp.all(jnp.isfinite(g)), "Gradients must be finite"

        # Also test second-order (hessian-vector product)
        hvp = jax.grad(lambda r: jnp.sum(grad_fn(r) ** 2))
        h = hvp(sys["positions"])
        assert jnp.all(jnp.isfinite(h)), "Second-order gradients must be finite"


# ── Simulation step tests ──────────────────────────────────────────────────


class TestNLLangevinStep:
    """Tests that make_langevin_step_nl works correctly."""

    def test_nl_step_produces_finite_state(self):
        """A single NL Langevin step should produce finite positions/momenta."""
        from prolix.batched_simulate import make_langevin_step_nl, LangevinState
        from prolix.padding import PaddedSystem

        n = 16

        # Build a minimal PaddedSystem with required fields
        sys_data = _make_toy_system(n)
        rng = np.random.RandomState(99)

        # Create a minimal PaddedSystem (just the fields needed for energy)
        padded = PaddedSystem(
            positions=sys_data["positions"],
            charges=sys_data["charges"],
            sigmas=sys_data["sigmas"],
            epsilons=sys_data["epsilons"],
            radii=sys_data["radii"],
            scaled_radii=sys_data["scaled_radii"],
            masses=jnp.ones(n) * 12.0,
            atom_mask=sys_data["atom_mask"],
            bonds=jnp.zeros((0, 2), dtype=jnp.int32),
            bond_params=jnp.zeros((0, 2)),
            bond_mask=jnp.zeros(0, dtype=jnp.bool_),
            angles=jnp.zeros((0, 3), dtype=jnp.int32),
            angle_params=jnp.zeros((0, 2)),
            angle_mask=jnp.zeros(0, dtype=jnp.bool_),
            dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
            dihedral_params=jnp.zeros((0, 3)),
            dihedral_mask=jnp.zeros(0, dtype=jnp.bool_),
            impropers=jnp.zeros((0, 4), dtype=jnp.int32),
            improper_params=jnp.zeros((0, 3)),
            improper_mask=jnp.zeros(0, dtype=jnp.bool_),
            cmap_torsions=None,
            cmap_mask=None,
            cmap_coeffs=None,
            n_real_atoms=jnp.array(n),
            n_padded_atoms=n,
            bucket_size=n,
        )

        neighbor_idx = _build_dense_neighbor_idx(n)

        # Create initial state
        key = jax.random.PRNGKey(0)
        state = LangevinState(
            positions=sys_data["positions"],
            momentum=jnp.zeros((n, 3)),
            force=jnp.zeros((n, 3)),
            mass=jnp.ones(n) * 12.0,
            key=key,
        )

        # Create step function with typical params
        dt = 0.001
        kT = 0.6  # ~300K in kcal/mol
        gamma = 1.0
        step_fn = make_langevin_step_nl(dt, kT, gamma)

        # Take one step
        new_state = step_fn(padded, state, neighbor_idx)

        assert jnp.all(jnp.isfinite(new_state.positions)), \
            "NL step positions must be finite"
        assert jnp.all(jnp.isfinite(new_state.momentum)), \
            "NL step momenta must be finite"
        assert jnp.all(jnp.isfinite(new_state.force)), \
            "NL step forces must be finite"


# ── Custom VJP tests ────────────────────────────────────────────────────────


class TestCustomVJP:
    """Tests that custom VJP LJ matches autodiff gradients."""

    def test_cvjp_gradient_matches_autodiff(self):
        """Custom VJP LJ gradient should match autodiff LJ gradient."""
        from prolix.batched_energy import (
            _lj_energy_neighbor_list,
            _make_lj_energy_nl_cvjp,
        )

        sys = _make_toy_system(n_atoms=32)
        neighbor_idx = _build_dense_neighbor_idx(32)

        # Autodiff gradient
        grad_auto = jax.grad(
            lambda r: _lj_energy_neighbor_list(
                r, sys["sigmas"], sys["epsilons"], neighbor_idx,
            )
        )(sys["positions"])

        # Custom VJP gradient
        lj_cvjp = _make_lj_energy_nl_cvjp(neighbor_idx)
        grad_cvjp = jax.grad(
            lambda r: lj_cvjp(r, sys["sigmas"], sys["epsilons"])
        )(sys["positions"])

        np.testing.assert_allclose(
            np.array(grad_auto), np.array(grad_cvjp), rtol=1e-4,
            err_msg="Custom VJP gradient must match autodiff gradient",
        )

    def test_cvjp_energy_fn_produces_finite_step(self):
        """single_padded_energy_nl_cvjp should produce a finite Langevin step."""
        from prolix.batched_energy import single_padded_energy_nl_cvjp
        from prolix.batched_simulate import make_langevin_step_nl, LangevinState
        from prolix.padding import PaddedSystem

        n = 16
        sys_data = _make_toy_system(n)

        padded = PaddedSystem(
            positions=sys_data["positions"],
            charges=sys_data["charges"],
            sigmas=sys_data["sigmas"],
            epsilons=sys_data["epsilons"],
            radii=sys_data["radii"],
            scaled_radii=sys_data["scaled_radii"],
            masses=jnp.ones(n) * 12.0,
            atom_mask=sys_data["atom_mask"],
            bonds=jnp.zeros((0, 2), dtype=jnp.int32),
            bond_params=jnp.zeros((0, 2)),
            bond_mask=jnp.zeros(0, dtype=jnp.bool_),
            angles=jnp.zeros((0, 3), dtype=jnp.int32),
            angle_params=jnp.zeros((0, 2)),
            angle_mask=jnp.zeros(0, dtype=jnp.bool_),
            dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
            dihedral_params=jnp.zeros((0, 3)),
            dihedral_mask=jnp.zeros(0, dtype=jnp.bool_),
            impropers=jnp.zeros((0, 4), dtype=jnp.int32),
            improper_params=jnp.zeros((0, 3)),
            improper_mask=jnp.zeros(0, dtype=jnp.bool_),
            cmap_torsions=None, cmap_mask=None, cmap_coeffs=None,
            n_real_atoms=jnp.array(n), n_padded_atoms=n, bucket_size=n,
        )

        neighbor_idx = _build_dense_neighbor_idx(n)
        key = jax.random.PRNGKey(0)
        state = LangevinState(
            positions=sys_data["positions"],
            momentum=jnp.zeros((n, 3)),
            force=jnp.zeros((n, 3)),
            mass=jnp.ones(n) * 12.0,
            key=key,
        )

        step_fn = make_langevin_step_nl(
            0.001, 0.6, 1.0,
            energy_fn=single_padded_energy_nl_cvjp,
        )
        new_state = step_fn(padded, state, neighbor_idx)

        assert jnp.all(jnp.isfinite(new_state.positions)), \
            "CVJP step positions must be finite"
        assert jnp.all(jnp.isfinite(new_state.force)), \
            "CVJP step forces must be finite"

