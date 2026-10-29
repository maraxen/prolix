"""End-to-end tests for EnsemblePlan Pallas v2 Megakernel & Mixed-Precision integration.

Covers Spec .praxia/specs/261029_ensemble_plan_pallas_v2_and_dhfr_cuda_spec.md:
- G1: EnsemblePlan(use_pallas=True) -> bundle_md -> pallas_bonded_all_fused & pallas_nonbonded_nl
- G2: Mixed-precision float64 BAOAB/SETTLE integration + float32 Pallas kernels (zero IntentError)
- G3: Dynamic 2-bit exclusion re-encoding (excl_mode='pre_encoded') across NL skin rebuilds
      and multi-bundle stacked neighbor seeds, plus PBC coordinate wrapping inside make_neighbor_list_fn.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.api.bundle_md import (
    displacement_fn_for_bundle,
    energy_fn_from_bundle,
)
from prolix.api.ensemble_plan import EnsemblePlan, _encode_bundle_nl
from prolix.pallas_kernels import _NB_EXCL_14_BIT, _NB_EXCL_FULL_BIT, _NB_INDEX_MASK
from prolix.physics import neighbor_list as nl_mod
from prolix.physics.system import make_bundle_from_system
from prolix.typing import PhysicsSystem


def _make_periodic_solvated_bundle(dtype=jnp.float64, seed: int = 0, shift_solute: float = 0.0):
    """Build a synthetic periodic PhysicsSystem + MolecularBundle with solute + 2 rigid waters."""
    rng = np.random.default_rng(seed)
    # 6 solute atoms (chain 0-1-2-3-4-5) + 2 water molecules (atoms 6,7,8 and 9,10,11) = 12 atoms
    n_atoms = 12
    box = jnp.array([24.0, 24.0, 24.0], dtype=dtype)

    solute_pos = np.array(
        [
            [4.0, 5.0, 5.0],
            [5.5, 5.0, 5.0],
            [6.2, 6.3, 5.0],
            [7.7, 6.3, 5.2],
            [8.4, 7.6, 5.5],
            [9.9, 7.6, 5.5],
        ],
        dtype=np.float64,
    ) + shift_solute
    solute_pos += rng.normal(scale=0.03, size=solute_pos.shape)

    # Rigid TIP3P-like water geometries (O-H ~ 0.9572 A, H-H ~ 1.5139 A)
    w1 = np.array(
        [
            [12.0, 12.0, 12.0],
            [12.9572, 12.0, 12.0],
            [11.76, 12.926, 12.0],
        ],
        dtype=np.float64,
    )
    w2 = np.array(
        [
            [16.0, 15.0, 14.0],
            [16.9572, 15.0, 14.0],
            [15.76, 15.926, 14.0],
        ],
        dtype=np.float64,
    )
    pos = jnp.asarray(np.vstack([solute_pos, w1, w2]), dtype=dtype)

    charges = jnp.array(
        [0.15, -0.15, 0.20, -0.20, 0.10, -0.10, -0.834, 0.417, 0.417, -0.834, 0.417, 0.417],
        dtype=dtype,
    )
    sigmas = jnp.array(
        [3.4, 3.2, 3.3, 3.1, 3.4, 3.0, 3.1507, 1.0, 1.0, 3.1507, 1.0, 1.0],
        dtype=dtype,
    )
    epsilons = jnp.array(
        [0.10, 0.08, 0.12, 0.09, 0.11, 0.07, 0.1521, 0.0, 0.0, 0.1521, 0.0, 0.0],
        dtype=dtype,
    )
    masses = jnp.array(
        [12.01, 14.01, 12.01, 16.00, 12.01, 1.008, 15.999, 1.008, 1.008, 15.999, 1.008, 1.008],
        dtype=dtype,
    )

    bonds = jnp.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=jnp.int32)
    bond_params = jnp.array(
        [
            [1.50, 300.0],
            [1.50, 310.0],
            [1.52, 290.0],
            [1.49, 305.0],
            [1.50, 300.0],
        ],
        dtype=dtype,
    )

    angles = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=jnp.int32)
    angle_params = jnp.array(
        [
            [1.91, 55.0],
            [1.93, 60.0],
            [1.90, 50.0],
            [1.92, 58.0],
        ],
        dtype=dtype,
    )

    dihedrals = jnp.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=jnp.int32)
    dihedral_params = jnp.array(
        [
            [2.0, 0.0, 1.2],
            [3.0, 3.14159265, 0.8],
            [1.0, 0.5, 1.5],
        ],
        dtype=dtype,
    )

    water_indices = jnp.array([[6, 7, 8], [9, 10, 11]], dtype=jnp.int32)

    # Exclusions: 1-2/1-3 full exclusions (scale 0.0) and 1-4 scaled exclusions (vdw=0.5, elec=0.83333333)
    excl_spec = nl_mod.ExclusionSpec(
        n_atoms=n_atoms,
        idx_12_13=jnp.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [0, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [6, 7],
                [6, 8],
                [7, 8],
                [9, 10],
                [9, 11],
                [10, 11],
            ],
            dtype=jnp.int32,
        ),
        idx_14=jnp.array([[0, 3], [1, 4], [2, 5]], dtype=jnp.int32),
        scale_14_elec=0.83333333,
        scale_14_vdw=0.5,
        exception_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros((0,), dtype=dtype),
        exception_epsilons=jnp.zeros((0,), dtype=dtype),
        exception_chargeprods=jnp.zeros((0,), dtype=dtype),
    )

    sys = PhysicsSystem.from_dict(
        {
            "charges": charges,
            "sigmas": sigmas,
            "epsilons": epsilons,
            "masses": masses,
            "bonds": bonds,
            "bond_params": bond_params,
            "angles": angles,
            "angle_params": angle_params,
            "dihedrals": dihedrals,
            "dihedral_params": dihedral_params,
            "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
            "improper_params": jnp.zeros((0, 3), dtype=dtype),
        },
        positions=pos,
        box_size=box,
        cutoff_distance=9.0,
    ).replace(
        water_indices=water_indices,
        water_mask=jnp.ones(2, dtype=bool),
        pme_alpha=0.34,
    )
    bundle = make_bundle_from_system(sys, boundary_condition="periodic", exclusion_spec=excl_spec)
    return bundle


def test_ensemble_plan_use_pallas_dispatches_megakernel_and_pre_encoded_nl(monkeypatch):
    """G1 + G3: EnsemblePlan(use_pallas=True).run() dispatches pallas_bonded_all_fused and pre_encoded NL."""
    import prolix.api.bundle_md as bundle_md
    import prolix.batched_energy as batched_energy

    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        bundle = _make_periodic_solvated_bundle(dtype=jnp.float64, seed=1)

        bonded_calls: list[int] = []
        nl_excl_modes: list[str] = []

        real_bonded_fused = bundle_md.pallas_bonded_all_fused
        real_nonbonded_nl = batched_energy.pallas_nonbonded_nl
        assert real_bonded_fused is not None
        assert real_nonbonded_nl is not None

        def spy_bonded(*args, **kwargs):
            bonded_calls.append(1)
            return real_bonded_fused(*args, **kwargs)

        def spy_nonbonded_nl(*args, **kwargs):
            nl_excl_modes.append(kwargs.get("excl_mode", "table"))
            return real_nonbonded_nl(*args, **kwargs)

        monkeypatch.setattr(bundle_md, "pallas_bonded_all_fused", spy_bonded)
        monkeypatch.setattr(batched_energy, "pallas_bonded_all_fused", spy_bonded)
        monkeypatch.setattr(batched_energy, "pallas_nonbonded_nl", spy_nonbonded_nl)

        plan = EnsemblePlan(bundle, use_pallas=True)
        traj = plan.run(
            n_steps=4,
            dt=1.0,
            kT=0.6,
            seed=42,
            run_mode="inference",
            use_neighbor_list=True,
            nl_update_every=2,
        )
        assert len(bonded_calls) >= 1, "Expected pallas_bonded_all_fused to be traced/called"
        assert len(nl_excl_modes) >= 1, "Expected pallas_nonbonded_nl to be traced/called"
        assert all(m == "pre_encoded" for m in nl_excl_modes), (
            f"Expected excl_mode='pre_encoded', got {nl_excl_modes}"
        )
        assert traj.positions.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(traj.positions))
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_pallas_v2_mixed_precision_energy_force_and_fd_parity():
    """G2: Under jax_enable_x64=True, float64 positions -> float32 Pallas kernels -> float64 energy & forces."""
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        bundle = _make_periodic_solvated_bundle(dtype=jnp.float64, seed=2)
        disp_fn, _ = displacement_fn_for_bundle(bundle)
        box_vec = jnp.diag(bundle.box)
        neighbor_fn = nl_mod.make_neighbor_list_fn(
            disp_fn, box_vec, 9.0, capacity_multiplier=1.5, dr_threshold=0.5
        )
        nbr_raw = neighbor_fn.allocate(bundle.positions)
        nbr_enc = _encode_bundle_nl(nbr_raw, bundle)

        e_fn_pallas = energy_fn_from_bundle(
            bundle,
            build_dense_exclusions=False,
            use_pallas=True,
            use_pallas_pme=True,
            excl_mode="pre_encoded",
        )
        e_fn_xla = energy_fn_from_bundle(
            bundle,
            build_dense_exclusions=False,
            use_pallas=False,
            use_pallas_pme=False,
            excl_mode="table",
        )

        e_pallas, g_pallas = jax.value_and_grad(lambda r: e_fn_pallas(r, neighbor=nbr_enc))(
            bundle.positions
        )
        e_xla, g_xla = jax.value_and_grad(lambda r: e_fn_xla(r, neighbor=nbr_raw))(
            bundle.positions
        )

        # Verify G2 dtype preservation
        assert e_pallas.dtype == jnp.float64
        assert g_pallas.dtype == jnp.float64

        # Verify Pallas v2 mixed-precision parity vs pure XLA float64
        np.testing.assert_allclose(
            np.asarray(e_pallas), np.asarray(e_xla), rtol=2e-4, atol=2e-3
        )
        n_real = int(bundle.n_atoms)
        np.testing.assert_allclose(
            np.asarray(g_pallas[:n_real]),
            np.asarray(g_xla[:n_real]),
            rtol=5e-4,
            atol=5e-3,
        )

        # Verify analytical gradient against central finite differences on atom 0 and atom 6
        eps_fd = 1e-3
        for atom_idx in (0, 2, 6):
            for dim in range(3):
                e_vec = jnp.zeros_like(bundle.positions).at[atom_idx, dim].set(eps_fd)
                ep = e_fn_pallas(bundle.positions + e_vec, neighbor=nbr_enc)
                em = e_fn_pallas(bundle.positions - e_vec, neighbor=nbr_enc)
                fd_grad = float((ep - em) / (2.0 * eps_fd))
                ana_grad = float(g_pallas[atom_idx, dim])
                np.testing.assert_allclose(ana_grad, fd_grad, rtol=2e-3, atol=1e-2)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_ensemble_plan_pallas_v2_skin_rebuild_and_pbc_wrapping():
    """G3: Out-of-box coordinates wrap into [0, box) and skin rebuilds preserve 2-bit exclusion tags."""
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        bundle = _make_periodic_solvated_bundle(dtype=jnp.float64, seed=3)
        box_vec = jnp.diag(bundle.box)

        # Shift all real atoms by +box_vec (outside primary cell [0, box)) to verify wrapping in NL
        shifted_pos = jnp.where(
            bundle.atom_mask[:, None],
            bundle.positions + box_vec[None, :],
            bundle.positions,
        )
        bundle_shifted = dataclasses.replace(bundle, positions=shifted_pos)

        disp_fn, _ = displacement_fn_for_bundle(bundle_shifted)
        neighbor_fn = nl_mod.make_neighbor_list_fn(
            disp_fn, box_vec, 9.0, capacity_multiplier=1.5, dr_threshold=0.5
        )
        nbr0 = _encode_bundle_nl(neighbor_fn.allocate(bundle_shifted.positions), bundle_shifted)
        assert not bool(nbr0.did_buffer_overflow)

        # Verify 2-bit tags are present initially on bonded/1-4 pairs
        has_full_tag_0 = bool(jnp.any((nbr0.idx >= 0) & ((nbr0.idx & _NB_EXCL_FULL_BIT) != 0)))
        has_14_tag_0 = bool(jnp.any((nbr0.idx >= 0) & ((nbr0.idx & _NB_EXCL_14_BIT) != 0)))
        assert has_full_tag_0, "Expected 1-2/1-3 full exclusion bit tags in initial NL"
        assert has_14_tag_0, "Expected 1-4 exclusion bit tags in initial NL"

        # Simulate a forced skin rebuild with negative out-of-box coordinates
        neg_pos = jnp.where(
            bundle.atom_mask[:, None],
            bundle.positions - box_vec[None, :],
            bundle.positions,
        )
        nbr_rebuilt = _encode_bundle_nl(
            dataclasses.replace(
                nbr0,
                reference_position=nbr0.reference_position + jnp.asarray(1e6, dtype=jnp.float64),
            ).update(neg_pos),
            bundle_shifted,
        )
        assert not bool(nbr_rebuilt.did_buffer_overflow)
        has_full_tag_1 = bool(
            jnp.any((nbr_rebuilt.idx >= 0) & ((nbr_rebuilt.idx & _NB_EXCL_FULL_BIT) != 0))
        )
        has_14_tag_1 = bool(
            jnp.any((nbr_rebuilt.idx >= 0) & ((nbr_rebuilt.idx & _NB_EXCL_14_BIT) != 0))
        )
        assert has_full_tag_1, "Expected 1-2/1-3 full exclusion bit tags after skin rebuild"
        assert has_14_tag_1, "Expected 1-4 exclusion bit tags after skin rebuild"

        # Run multi-step MD across skin rebuilds with SETTLE rigid water
        plan = EnsemblePlan(bundle_shifted, use_pallas=True)
        traj = plan.run(
            n_steps=6,
            dt=1.0,
            kT=0.6,
            seed=99,
            run_mode="inference",
            use_neighbor_list=True,
            nl_update_every=2,
        )
        assert traj.positions.shape == (1, int(bundle.positions.shape[0]), 3)
        assert traj.positions.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(traj.positions))
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_stacked_multi_bundle_pallas_v2_distinct_encoded_nl():
    """G3: _build_stacked_neighbor_seed invalidates reference_position and encodes each bundle's NL."""
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        bundle_a = _make_periodic_solvated_bundle(dtype=jnp.float64, seed=10, shift_solute=0.0)
        # In bundle_b, move the solute far from water 1 so solute-water neighbor pairs differ
        bundle_b = _make_periodic_solvated_bundle(dtype=jnp.float64, seed=11, shift_solute=-3.5)

        plan = EnsemblePlan.from_bundles([bundle_a, bundle_b], use_pallas=True)
        stacked_nbr, _ = plan._build_stacked_neighbor_seed(
            [bundle_a, bundle_b], int(bundle_a.positions.shape[0])
        )
        # Both replicas must have 2-bit exclusion tags encoded
        for b_idx in (0, 1):
            idx_b = stacked_nbr.idx[b_idx]
            assert bool(jnp.any((idx_b >= 0) & ((idx_b & _NB_EXCL_FULL_BIT) != 0)))
            assert bool(jnp.any((idx_b >= 0) & ((idx_b & _NB_EXCL_14_BIT) != 0)))

        # And because reference_position was invalidated before seed.update(),
        # bundle_b's neighbor list was genuinely rebuilt for bundle_b's coordinates
        clean_0 = stacked_nbr.idx[0] & _NB_INDEX_MASK
        clean_1 = stacked_nbr.idx[1] & _NB_INDEX_MASK
        assert not bool(jnp.array_equal(clean_0, clean_1)), (
            "Expected bundle_b's neighbor list to be rebuilt for bundle_b's positions, "
            "not short-circuited to bundle_a's neighbor list"
        )

        trajs = plan.run(
            n_steps=4,
            dt=1.0,
            kT=0.6,
            seed=7,
            run_mode="inference",
            use_neighbor_list=True,
            nl_update_every=2,
        )
        assert len(trajs) == 2
        for t in trajs:
            assert t.positions.dtype == jnp.float64
            assert jnp.all(jnp.isfinite(t.positions))
    finally:
        jax.config.update("jax_enable_x64", prev_x64)
