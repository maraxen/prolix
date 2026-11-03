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
        spread_calls: list[int] = []
        gather_calls: list[int] = []

        import prolix.physics.pme_pallas as pme_pallas

        real_bonded_fused = bundle_md.pallas_bonded_all_fused
        real_nonbonded_nl = batched_energy.pallas_nonbonded_nl
        real_spread = pme_pallas._spread_charges_pallas_single
        real_gather = pme_pallas._gather_pallas_single
        assert real_bonded_fused is not None
        assert real_nonbonded_nl is not None

        def spy_bonded(*args, **kwargs):
            bonded_calls.append(1)
            return real_bonded_fused(*args, **kwargs)

        def spy_nonbonded_nl(*args, **kwargs):
            nl_excl_modes.append(kwargs.get("excl_mode", "table"))
            return real_nonbonded_nl(*args, **kwargs)

        def spy_spread(*args, **kwargs):
            spread_calls.append(1)
            return real_spread(*args, **kwargs)

        def spy_gather(*args, **kwargs):
            gather_calls.append(1)
            return real_gather(*args, **kwargs)

        monkeypatch.setattr(bundle_md, "pallas_bonded_all_fused", spy_bonded)
        monkeypatch.setattr(batched_energy, "pallas_bonded_all_fused", spy_bonded)
        monkeypatch.setattr(batched_energy, "pallas_nonbonded_nl", spy_nonbonded_nl)
        monkeypatch.setattr(pme_pallas, "_spread_charges_pallas_single", spy_spread)
        monkeypatch.setattr(pme_pallas, "_gather_pallas_single", spy_gather)

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
        assert len(bonded_calls) >= 2, "Expected pallas_bonded_all_fused to be traced in init + while_loop"
        assert len(nl_excl_modes) >= 2, "Expected pallas_nonbonded_nl to be traced in init + while_loop"
        assert len(spread_calls) >= 2, "Expected PME spread_charges_pallas to be traced in init + while_loop"
        assert len(gather_calls) >= 2, "Expected PME gather_pallas to be traced in init + while_loop"
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
        e_pallas_jit, g_pallas_jit = jax.jit(
            jax.value_and_grad(lambda r: e_fn_pallas(r, neighbor=nbr_enc))
        )(bundle.positions)
        e_xla, g_xla = jax.jit(
            jax.value_and_grad(lambda r: e_fn_xla(r, neighbor=nbr_raw))
        )(bundle.positions)

        # Verify JIT vs eager parity (ensuring PME reciprocal is never dropped under jax.jit)
        np.testing.assert_allclose(
            np.asarray(e_pallas_jit), np.asarray(e_pallas), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(g_pallas_jit), np.asarray(g_pallas), rtol=1e-5, atol=1e-5
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


def test_pallas_v2_native_2d_grid_batching_parity(monkeypatch):
    """2D Pallas grid batching (B, n_blocks) across replicas matches single-replica energies and forces."""
    import prolix.pallas_kernels as pallas_kernels
    import prolix.physics.pallas_bonded as pallas_bonded
    import prolix.physics.pme_pallas as pme_pallas

    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        bundles = [
            _make_periodic_solvated_bundle(dtype=jnp.float64, seed=20 + b, shift_solute=-0.5 * b)
            for b in range(3)
        ]
        disp_fn, _ = displacement_fn_for_bundle(bundles[0])
        box_vec = jnp.diag(bundles[0].box)
        neighbor_fn = nl_mod.make_neighbor_list_fn(
            disp_fn, box_vec, 9.0, capacity_multiplier=1.5, dr_threshold=0.5
        )
        nbrs = [
            _encode_bundle_nl(neighbor_fn.allocate(b.positions), b)
            for b in bundles
        ]
        pos_batch = jnp.stack([b.positions for b in bundles], axis=0)
        nbr_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *nbrs)

        e_fn = energy_fn_from_bundle(
            bundles[0],
            build_dense_exclusions=False,
            use_pallas=True,
            use_pallas_pme=True,
            excl_mode="pre_encoded",
        )

        batch_sizes_seen: dict[str, list[int]] = {
            "nb": [],
            "spread": [],
            "gather": [],
            "bonded": [],
        }
        orig_nb_batch = pallas_kernels._pallas_nonbonded_nl_batch
        orig_spread_batch = pme_pallas._spread_charges_pallas_batch
        orig_gather_batch = pme_pallas._gather_pallas_batch
        orig_bonded_batch = pallas_bonded._pallas_bonded_all_fused_batch

        def spy_nb(B, *a, **kw):
            batch_sizes_seen["nb"].append(int(B))
            return orig_nb_batch(B, *a, **kw)

        def spy_spread(B, *a, **kw):
            batch_sizes_seen["spread"].append(int(B))
            return orig_spread_batch(B, *a, **kw)

        def spy_gather(B, *a, **kw):
            batch_sizes_seen["gather"].append(int(B))
            return orig_gather_batch(B, *a, **kw)

        def spy_bonded(B, *a, **kw):
            batch_sizes_seen["bonded"].append(int(B))
            return orig_bonded_batch(B, *a, **kw)

        monkeypatch.setattr(pallas_kernels, "_pallas_nonbonded_nl_batch", spy_nb)
        monkeypatch.setattr(pme_pallas, "_spread_charges_pallas_batch", spy_spread)
        monkeypatch.setattr(pme_pallas, "_gather_pallas_batch", spy_gather)
        monkeypatch.setattr(pallas_bonded, "_pallas_bonded_all_fused_batch", spy_bonded)

        vg_fn = jax.value_and_grad(lambda r, n: e_fn(r, neighbor=n))
        e_vmap, g_vmap = jax.jit(jax.vmap(vg_fn))(pos_batch, nbr_batch)

        assert 3 in batch_sizes_seen["nb"]
        assert 3 in batch_sizes_seen["spread"]
        assert 3 in batch_sizes_seen["gather"]
        assert 3 in batch_sizes_seen["bonded"]

        for b_idx in range(3):
            e_single, g_single = jax.jit(vg_fn)(pos_batch[b_idx], nbrs[b_idx])
            np.testing.assert_allclose(
                np.asarray(e_vmap[b_idx]), np.asarray(e_single), rtol=1e-5, atol=1e-5
            )
            np.testing.assert_allclose(
                np.asarray(g_vmap[b_idx]), np.asarray(g_single), rtol=1e-5, atol=1e-5
            )
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_pbc_boundary_crossing_does_not_trigger_false_skin_rebuild():
    """Minimum-image displacement in skin check prevents false rebuilds when atoms wrap across PBC."""
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        bundle = _make_periodic_solvated_bundle(dtype=jnp.float64, seed=31)
        disp_fn, shift_fn = displacement_fn_for_bundle(bundle)
        box_vec = jnp.diag(bundle.box)

        # Place atom 0 at x = 0.01 A (just inside primary cell) and move it by -0.02 A so shift_fn wraps to box_x - 0.01 A
        pos_0 = bundle.positions.at[0, 0].set(0.01)
        dpos = jnp.zeros_like(pos_0).at[0, 0].set(-0.02)
        pos_1 = shift_fn(pos_0, dpos)
        assert float(pos_1[0, 0]) > float(box_vec[0]) - 0.05

        pad_mask_3d = bundle.atom_mask[:, None]
        # Raw Euclidean diff would be ~(box_x - 0.02)^2 ~ 900 A^2; minimum-image displacement is (-0.02)^2 = 4e-4 A^2
        d_pos_min_img = jnp.where(pad_mask_3d, jax.vmap(disp_fn)(pos_1, pos_0), 0.0)
        max_dr_sq = float(jnp.max(jnp.sum(d_pos_min_img ** 2, axis=-1)))
        np.testing.assert_allclose(max_dr_sq, 0.0004, rtol=1e-6, atol=1e-8)
        assert max_dr_sq < (0.5 / 2.0) ** 2
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_settle_closed_form_cayley_hamilton_projector_matches_eigh():
    """Closed-form Cayley-Hamilton rank-1 projector matches jnp.linalg.eigh to <1e-13 on planar 4x4 Horn matrices."""
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        key = jax.random.PRNGKey(1234)
        k1, k2 = jax.random.split(key)
        B = 50
        # Reference planar water triangle in xy-plane (z = 0)
        ref = jnp.broadcast_to(
            jnp.array(
                [[0.0, 0.0, 0.0], [0.75695, 0.58588, 0.0], [-0.75695, 0.58588, 0.0]],
                dtype=jnp.float64,
            ),
            (B, 3, 3),
        )
        ref = ref - jnp.mean(ref, axis=1, keepdims=True)
        # Random 3D target water positions
        tgt = jax.random.normal(k1, (B, 3, 3), dtype=jnp.float64) + jax.random.normal(
            k2, (B, 1, 3), dtype=jnp.float64
        )
        tgt = tgt - jnp.mean(tgt, axis=1, keepdims=True)

        h = jnp.einsum("bni,bnj->bij", ref, tgt)
        sxx, sxy, sxz = h[:, 0, 0], h[:, 0, 1], h[:, 0, 2]
        syx, syy, syz = h[:, 1, 0], h[:, 1, 1], h[:, 1, 2]
        szx, szy, szz = h[:, 2, 0], h[:, 2, 1], h[:, 2, 2]
        n00 = sxx + syy + szz
        n11 = sxx - syy - szz
        n22 = -sxx + syy - szz
        n33 = -sxx - syy + szz
        n01, n02, n03 = syz - szy, szx - sxz, sxy - syx
        n12, n13, n23 = sxy + syx, sxz + szx, syz + szy
        n_mat = jnp.stack(
            [
                jnp.stack([n00, n01, n02, n03], axis=-1),
                jnp.stack([n01, n11, n12, n13], axis=-1),
                jnp.stack([n02, n12, n22, n23], axis=-1),
                jnp.stack([n03, n13, n23, n33], axis=-1),
            ],
            axis=-2,
        )

        # Reference eigh
        _, eigvecs = jnp.linalg.eigh(n_mat)
        q_eigh = eigvecs[:, :, -1]

        # Closed-form Cayley-Hamilton projector
        c2 = (
            sxx * sxx
            + sxy * sxy
            + sxz * sxz
            + syx * syx
            + syy * syy
            + syz * syz
            + szx * szx
            + szy * szy
            + szz * szz
        )
        d0 = sxx * syy - sxy * syx
        d1 = sxx * syz - sxz * syx
        d2 = sxy * syz - sxz * syy
        disc = jnp.maximum(4.0 * (d0 * d0 + d1 * d1 + d2 * d2), 0.0)
        sqrt_disc = jnp.sqrt(disc)
        lam_max_sq = jnp.maximum(c2 + sqrt_disc, 0.0)
        lam_2_sq = jnp.maximum(c2 - sqrt_disc, 0.0)
        lam_max = jnp.sqrt(lam_max_sq)

        eye4 = jnp.eye(4, dtype=jnp.float64)[None, :, :]
        n_sq = jnp.einsum("bij,bjk->bik", n_mat, n_mat)
        proj = jnp.einsum(
            "bij,bjk->bik",
            n_mat + lam_max[:, None, None] * eye4,
            n_sq - lam_2_sq[:, None, None] * eye4,
        )
        col_norms_sq = jnp.sum(proj * proj, axis=-2)
        best_col = jnp.argmax(col_norms_sq, axis=-1)
        q_raw = jnp.take_along_axis(proj, best_col[:, None, None], axis=-1)[..., 0]
        q_ch = q_raw / jnp.sqrt(jnp.maximum(jnp.sum(q_raw * q_raw, axis=-1, keepdims=True), 1e-30))

        # Align sign with q_eigh
        dots = jnp.sum(q_ch * q_eigh, axis=-1, keepdims=True)
        q_ch_aligned = jnp.where(dots >= 0.0, q_ch, -q_ch)
        np.testing.assert_allclose(np.asarray(q_ch_aligned), np.asarray(q_eigh), rtol=1e-12, atol=1e-13)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)



def test_settle_scale_free_skew_projector_near_collinear_float32():
    """Patch A: Scale-free projector P2_hat = I_4 + C(c_hat) avoids float32 cancellation when ||v|| << ||u||."""
    rng = np.random.default_rng(123)
    w = 32
    # Near-collinear water (H-O-H -> 180 deg): H-H vector u is O(1) while bisector vector v shrinks to 1e-4 * ||u||
    u = rng.normal(size=(w, 3)).astype(np.float32)
    v = (rng.normal(size=(w, 3)) * 1e-4).astype(np.float32)
    s_mat = np.zeros((w, 3, 3), dtype=np.float32)
    s_mat[:, 0, :] = u
    s_mat[:, 1, :] = v

    def _build_n_and_projectors(s_in, dt):
        s = jnp.asarray(s_in, dtype=dt)
        sxx, sxy, sxz = s[:, 0, 0], s[:, 0, 1], s[:, 0, 2]
        syx, syy, syz = s[:, 1, 0], s[:, 1, 1], s[:, 1, 2]
        szx, szy, szz = s[:, 2, 0], s[:, 2, 1], s[:, 2, 2]
        n00 = sxx + syy + szz
        n11 = sxx - syy - szz
        n22 = -sxx + syy - szz
        n33 = -sxx - syy + szz
        n01, n02, n03 = syz - szy, szx - sxz, sxy - syx
        n12, n13, n23 = sxy + syx, sxz + szx, syz + szy
        n_mat = jnp.stack(
            [
                jnp.stack([n00, n01, n02, n03], axis=-1),
                jnp.stack([n01, n11, n12, n13], axis=-1),
                jnp.stack([n02, n12, n22, n23], axis=-1),
                jnp.stack([n03, n13, n23, n33], axis=-1),
            ],
            axis=-2,
        )
        u_vec = s[:, 0, :]
        v_vec = s[:, 1, :]
        p_tr = jnp.sum(u_vec * u_vec, axis=-1) + jnp.sum(v_vec * v_vec, axis=-1)
        cross_uv = jnp.cross(u_vec, v_vec)
        q_val = jnp.linalg.norm(cross_uv, axis=-1)
        lam_max = jnp.sqrt(jnp.maximum(p_tr + 2.0 * q_val, 0.0))
        eye4 = jnp.eye(4, dtype=dt)[None, :, :]
        p1 = n_mat + lam_max[:, None, None] * eye4
        c_hat = cross_uv / jnp.maximum(q_val[:, None], jnp.asarray(1e-12, dtype=dt))
        cx, cy, cz = c_hat[:, 0], c_hat[:, 1], c_hat[:, 2]
        z = jnp.zeros_like(cx)
        c_skew = jnp.stack(
            [
                jnp.stack([cz, -cy, cx, z], axis=-1),
                jnp.stack([-cy, -cz, z, cx], axis=-1),
                jnp.stack([cx, z, -cz, cy], axis=-1),
                jnp.stack([z, cx, cy, cz], axis=-1),
            ],
            axis=-2,
        )
        p2_hat = eye4 + c_skew
        proj = jnp.matmul(p1, p2_hat)
        col_norms_sq = jnp.sum(proj * proj, axis=1)
        best_col = jnp.argmax(col_norms_sq, axis=-1)
        q_raw = jnp.take_along_axis(proj, best_col[:, None, None], axis=2)[:, :, 0]
        return q_raw / jnp.linalg.norm(q_raw, axis=-1, keepdims=True)

    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        q64 = _build_n_and_projectors(s_mat, jnp.float64)
        q32 = _build_n_and_projectors(s_mat, jnp.float32).astype(jnp.float64)
        dots = jnp.abs(jnp.sum(q64 * q32, axis=-1))
        np.testing.assert_allclose(np.asarray(dots), np.ones(w), atol=1e-5)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_pallas_nonbonded_nl_empty_and_3d_direct_dispatch():
    """Patch B: Empty N=0 or K=0 early return and direct 3D batch dispatch."""
    from prolix.pallas_kernels import _pallas_nonbonded_nl_batch, _pallas_nonbonded_nl_vmappable

    # 1. K == 0 guard
    B, N, K = 2, 8, 0
    pos = jnp.zeros((B, N, 3), dtype=jnp.float32)
    q = jnp.zeros((B, N), dtype=jnp.float32)
    sig = jnp.ones((B, N), dtype=jnp.float32)
    eps = jnp.zeros((B, N), dtype=jnp.float32)
    mask = jnp.ones((B, N), dtype=jnp.int32)
    nbr = jnp.zeros((B, N, K), dtype=jnp.int32)
    ei = jnp.full((B, N, 4), -1, dtype=jnp.int32)
    ev = jnp.ones((B, N, 4), dtype=jnp.float32)
    ee = jnp.ones((B, N, 4), dtype=jnp.float32)
    bx = jnp.broadcast_to(jnp.array([20.0, 20.0, 20.0, 0.0, 0.0, 0.0], dtype=jnp.float32), (B, 6))
    lam = jnp.ones((B, 1), dtype=jnp.float32)

    e_empty, f_empty = _pallas_nonbonded_nl_batch(
        B,
        (True,) * 11,
        pos,
        q,
        sig,
        eps,
        mask,
        nbr,
        ei,
        ev,
        ee,
        bx,
        lam,
        True,
        0.34,
        9.0,
        332.0637,
        1.0,
        0.5,
        1,
        True,
        True,
        "pre_encoded",
    )
    assert e_empty.shape == (B,)
    assert f_empty.shape == (B, N, 3)
    assert bool(jnp.all(e_empty == 0.0))
    assert bool(jnp.all(f_empty == 0.0))

    # 2. Direct 3D dispatch parity with vmap over 2D
    rng = np.random.default_rng(7)
    pos_3d = jnp.asarray(rng.uniform(1.0, 15.0, size=(2, 16, 3)).astype(np.float32))
    q_3d = jnp.asarray(rng.uniform(-0.3, 0.3, size=(2, 16)).astype(np.float32))
    sig_3d = jnp.full((2, 16), 3.0, dtype=jnp.float32)
    eps_3d = jnp.full((2, 16), 0.1, dtype=jnp.float32)
    mask_3d = jnp.ones((2, 16), dtype=jnp.int32)
    nbr_3d = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32)[None, None, :], (2, 16, 16))
    nbr_3d = jnp.where(
        nbr_3d == jnp.arange(16, dtype=jnp.int32)[None, :, None], 16, nbr_3d
    )
    ei_3d = jnp.full((2, 16, 4), -1, dtype=jnp.int32)
    ev_3d = jnp.ones((2, 16, 4), dtype=jnp.float32)
    ee_3d = jnp.ones((2, 16, 4), dtype=jnp.float32)
    bx_3d = jnp.broadcast_to(jnp.array([20.0, 20.0, 20.0, 0.0, 0.0, 0.0], dtype=jnp.float32), (2, 6))
    lam_3d = jnp.ones((2, 1), dtype=jnp.float32)

    e_direct, f_direct = _pallas_nonbonded_nl_vmappable(
        pos_3d,
        q_3d,
        sig_3d,
        eps_3d,
        mask_3d,
        nbr_3d,
        ei_3d,
        ev_3d,
        ee_3d,
        bx_3d,
        lam_3d,
        True,
        0.34,
        9.0,
        332.0637,
        1.0,
        0.5,
        1,
        True,
        True,
        "pre_encoded",
    )
    e_0, f_0 = _pallas_nonbonded_nl_vmappable(
        pos_3d[0],
        q_3d[0],
        sig_3d[0],
        eps_3d[0],
        mask_3d[0],
        nbr_3d[0],
        ei_3d[0],
        ev_3d[0],
        ee_3d[0],
        bx_3d[0],
        lam_3d[0],
        True,
        0.34,
        9.0,
        332.0637,
        1.0,
        0.5,
        1,
        True,
        True,
        "pre_encoded",
    )
    np.testing.assert_allclose(np.asarray(e_direct[0]), np.asarray(e_0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(f_direct[0]), np.asarray(f_0), rtol=1e-5, atol=1e-5)


def test_dispatch_n_steps_inference_dynamic_limit_no_retrace():
    """Patch C: Changing n_steps does not re-trace step_fn when jitted."""
    from prolix.api.ensemble_dispatch import dispatch_n_steps_inference

    trace_count = {"n": 0}

    def step_fn(state, step_i):
        trace_count["n"] += 1
        return state + step_i + jnp.int32(1)

    init = jnp.int32(0)
    out1 = dispatch_n_steps_inference(step_fn, init, 4)
    out2 = dispatch_n_steps_inference(step_fn, init, 10)
    assert int(out1) == sum(i + 1 for i in range(4))
    assert int(out2) == sum(i + 1 for i in range(10))
    assert trace_count["n"] == 1


def test_single_bundle_inference_runner_cache_no_retrace(monkeypatch):
    """Repeated plan.run(run_mode='inference') reuses cached nbr0 + step_fn with zero re-tracing."""
    from prolix.api.ensemble_dispatch import _JITTED_WHILE_CACHE
    from prolix.api.ensemble_plan import _SINGLE_INFERENCE_CACHE

    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        bundle = _make_periodic_solvated_bundle(dtype=jnp.float64, seed=11)
        alloc_calls = {"n": 0}
        orig_make_nl = nl_mod.make_neighbor_list_fn

        def spy_make_nl(*args, **kwargs):
            nf = orig_make_nl(*args, **kwargs)
            orig_alloc = nf.allocate

            def spy_alloc(*a, **kw):
                alloc_calls["n"] += 1
                return orig_alloc(*a, **kw)

            return nf._replace(allocate=spy_alloc) if hasattr(nf, "_replace") else type(nf)(spy_alloc, nf.update)

        monkeypatch.setattr(nl_mod, "make_neighbor_list_fn", spy_make_nl)

        for use_pallas_flag in (False, True):
            if use_pallas_flag:
                import sys

                sys.modules.pop("prolix.physics.pme_pallas", None)
            _SINGLE_INFERENCE_CACHE.clear()
            _JITTED_WHILE_CACHE.clear()
            alloc_before = alloc_calls["n"]

            plan = EnsemblePlan.from_bundle(bundle, use_pallas=use_pallas_flag)
            t1 = plan.run(
                n_steps=1,
                dt=1.0,
                kT=0.6,
                seed=10,
                run_mode="inference",
                use_neighbor_list=True,
                nl_update_every=2,
            )
            count_after_first = alloc_calls["n"]
            assert count_after_first > alloc_before
            assert len(_JITTED_WHILE_CACHE) == 1
            jitted_while = next(iter(_JITTED_WHILE_CACHE.values()))
            assert jitted_while._cache_size() == 1

            t2 = plan.run(
                n_steps=4,
                dt=1.0,
                kT=0.6,
                seed=20,
                run_mode="inference",
                use_neighbor_list=True,
                nl_update_every=2,
            )
            # Second call with different n_steps and seed should hit _SINGLE_INFERENCE_CACHE
            # without reallocating nbr0 and without recompiling jitted_while (_cache_size() == 1).
            assert alloc_calls["n"] == count_after_first
            assert len(_JITTED_WHILE_CACHE) == 1
            assert jitted_while._cache_size() == 1

            # Third call on a fresh EnsemblePlan.from_bundle(bundle) instance (mirroring
            # bench_dhfr_parity.py's n_steps_list loop) must also reuse jitted_while.
            plan2 = EnsemblePlan.from_bundle(bundle, use_pallas=use_pallas_flag)
            t3 = plan2.run(
                n_steps=6,
                dt=1.0,
                kT=0.6,
                seed=30,
                run_mode="inference",
                use_neighbor_list=True,
                nl_update_every=2,
            )
            assert alloc_calls["n"] == count_after_first
            assert len(_JITTED_WHILE_CACHE) == 1
            assert jitted_while._cache_size() == 1
            assert jnp.all(jnp.isfinite(t1.positions))
            assert jnp.all(jnp.isfinite(t2.positions))
            assert jnp.all(jnp.isfinite(t3.positions))

        # Also verify use_neighbor_list=False branch canonicalizes initial state via fast_reseed_fn
        _SINGLE_INFERENCE_CACHE.clear()
        _JITTED_WHILE_CACHE.clear()
        plan_no_nl = EnsemblePlan.from_bundle(bundle, use_pallas=False)
        u1 = plan_no_nl.run(
            n_steps=1,
            dt=1.0,
            kT=0.6,
            seed=10,
            run_mode="inference",
            use_neighbor_list=False,
        )
        assert len(_JITTED_WHILE_CACHE) == 1
        jitted_while_no_nl = next(iter(_JITTED_WHILE_CACHE.values()))
        assert jitted_while_no_nl._cache_size() == 1
        u2 = plan_no_nl.run(
            n_steps=3,
            dt=1.0,
            kT=0.6,
            seed=20,
            run_mode="inference",
            use_neighbor_list=False,
        )
        assert jitted_while_no_nl._cache_size() == 1
        assert jnp.all(jnp.isfinite(u1.positions))
        assert jnp.all(jnp.isfinite(u2.positions))
    finally:
        _SINGLE_INFERENCE_CACHE.clear()
        _JITTED_WHILE_CACHE.clear()
        jax.config.update("jax_enable_x64", prev_x64)


def test_pallas_v22_prune_and_encode_cell_list_cpu_and_cuda_lowering():
    """Patch E: Bit-exact parity of pallas_prune_and_encode_cell_list (B=1 & vmap B=3) + CUDA Triton MLIR lowering."""
    from jax_md import partition, space
    from prolix.physics.neighbor_list import (
        encode_nl_exclusions,
        pallas_prune_and_encode_cell_list,
    )

    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        N = 128
        box = jnp.array([30.0, 30.0, 30.0], dtype=jnp.float32)
        r_cutoff = 9.0
        dr_threshold = 0.5
        total_cutoff = r_cutoff + dr_threshold
        rng = np.random.default_rng(42)
        pos = jnp.asarray(rng.uniform(0.0, 30.0, size=(N, 3)).astype(np.float32))

        M = 16
        excl_idx = np.full((N, M), -1, dtype=np.int32)
        excl_vdw = np.ones((N, M), dtype=np.float32)
        excl_elec = np.ones((N, M), dtype=np.float32)
        for i in range(N):
            n1 = (i + 1) % N
            n2 = (i + 2) % N
            cands = sorted([n1, n2])
            excl_idx[i, 0] = cands[0]
            excl_idx[i, 1] = cands[1]
            excl_vdw[i, 0] = 0.0
            excl_elec[i, 0] = 0.0
            excl_vdw[i, 1] = 0.5
            excl_elec[i, 1] = 0.83333333
        excl_idx_j = jnp.asarray(excl_idx)
        excl_vdw_j = jnp.asarray(excl_vdw)
        excl_elec_j = jnp.asarray(excl_elec)

        disp_fn, _ = space.periodic(box)
        nl_fns = partition.neighbor_list(
            disp_fn,
            box,
            r_cutoff=r_cutoff,
            dr_threshold=dr_threshold,
            capacity_multiplier=1.25,
            mask_self=True,
        )
        nbr_ref = nl_fns.allocate(pos, extra_capacity=32)
        K = int(nbr_ref.max_occupancy)
        C = int(nbr_ref.cell_list_capacity)
        nbr_ref_enc = encode_nl_exclusions(
            nbr_ref.idx, excl_idx_j, excl_vdw_j, excl_elec_j, n_atoms=N
        )

        # 1. Single-replica CPU interpret parity
        nbr_pallas, occ_pallas = pallas_prune_and_encode_cell_list(
            pos,
            box,
            excl_idx_j,
            excl_vdw_j,
            excl_elec_j,
            cell_size=total_cutoff,
            cell_capacity=C,
            max_occupancy=K,
            r_cutoff=total_cutoff,
            interpret=True,
        )
        for i in range(N):
            s_ref = set(int(x) for x in np.asarray(nbr_ref_enc[i]) if int(x) != N)
            s_pal = set(int(x) for x in np.asarray(nbr_pallas[i]) if int(x) != N)
            assert s_ref == s_pal, f"Atom {i} mismatch: {s_ref ^ s_pal}"
        assert int(occ_pallas) > 0

        # 2. Batched vmap (B=3) parity
        B = 3
        pos_b = jnp.stack(
            [pos, jnp.mod(pos + 1.2, 30.0), jnp.mod(pos + 2.7, 30.0)], axis=0
        )
        nbr_b, occ_b = jax.vmap(
            lambda p: pallas_prune_and_encode_cell_list(
                p,
                box,
                excl_idx_j,
                excl_vdw_j,
                excl_elec_j,
                cell_size=total_cutoff,
                cell_capacity=C,
                max_occupancy=K,
                r_cutoff=total_cutoff,
                interpret=True,
            )
        )(pos_b)
        assert nbr_b.shape == (B, N, K)
        assert occ_b.shape == (B,)
        assert bool(jnp.array_equal(nbr_b[0], nbr_pallas))

        # 3. CUDA Triton lowering check (interpret=False)
        lowered = (
            jax.jit(
                lambda p: pallas_prune_and_encode_cell_list(
                    p,
                    box,
                    excl_idx_j,
                    excl_vdw_j,
                    excl_elec_j,
                    cell_size=total_cutoff,
                    cell_capacity=C,
                    max_occupancy=K,
                    r_cutoff=total_cutoff,
                    interpret=False,
                )
            )
            .trace(pos)
            .lower(lowering_platforms=("cuda",))
        )
        mlir_text = lowered.as_text()
        assert "pallas_prune_and_encode_cell_list" in mlir_text
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_antipodal_box_invalidation_and_custom_vmap_nl_rebuild():
    """Patch D: +0.5*box_vec invalidates L=32.0 A (where +1e6 aliases to 0.0 mod 32) and custom_vmap works with Pallas NL."""
    from jax_md import space
    from prolix.physics import neighbor_list as nl_mod

    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    box = jnp.array([32.0, 32.0, 32.0], dtype=jnp.float64)
    disp_fn, _ = space.periodic(box)
    r0 = jnp.array([[5.0, 5.0, 5.0], [10.0, 10.0, 10.0]], dtype=jnp.float64)

    # +1e6 aliases to 0.0 under L=32.0 because 1e6 / 32.0 == 31250.0 exactly!
    dr_alias = jax.vmap(disp_fn)(r0, r0 + 1e6)
    assert float(jnp.max(jnp.abs(dr_alias))) < 1e-9

    # Antipodal +0.5 * box_vec gives maximum minimum-image displacement (16.0 A per axis)
    dr_antipodal = jax.vmap(disp_fn)(r0, r0 + 0.5 * box)
    assert float(jnp.min(jnp.linalg.norm(dr_antipodal, axis=-1))) > 27.0

    # Verify make_neighbor_list_fn with use_pallas_nl=True works for both B=1 and vmap B=2
    N = 64
    rng = np.random.default_rng(99)
    pos_a = jnp.asarray(rng.uniform(0.0, 32.0, size=(N, 3)).astype(np.float32))
    pos_b = jnp.mod(pos_a + 2.0, 32.0)
    ei = jnp.full((N, 4), -1, dtype=jnp.int32)
    ev = jnp.ones((N, 4), dtype=jnp.float32)
    ee = jnp.ones((N, 4), dtype=jnp.float32)

    nl_fn = nl_mod.make_neighbor_list_fn(
        disp_fn,
        box.astype(jnp.float32),
        cutoff=9.0,
        dr_threshold=0.5,
        excl_indices=ei,
        excl_scales_vdw=ev,
        excl_scales_elec=ee,
        use_pallas_nl=True,
    )
    seed = nl_fn.allocate(pos_a, extra_capacity=32)
    # No displacement -> should skip rebuild and keep identical idx
    same = seed.update(pos_a)
    assert bool(jnp.array_equal(same.idx, seed.idx))

    # Batched vmap over 2 replicas where replica 1 moved > half_skin
    stacked_nbr = jax.tree.map(lambda x: jnp.stack([x, x], axis=0), seed)
    pos_batch = jnp.stack([pos_a, pos_b], axis=0)
    updated_batch = jax.vmap(lambda n, p: n.update(p))(stacked_nbr, pos_batch)
    assert updated_batch.idx.shape == (2, N, seed.idx.shape[-1])
    assert not bool(jnp.any(updated_batch.did_buffer_overflow))
    jax.config.update("jax_enable_x64", prev_x64)
