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
        assert traj.positions.shape == (1, int(bundle.n_atoms), 3)
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
        traced_nl = jax.jit(
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
        ).trace(pos)
        try:
            lowered = traced_nl.lower(lowering_platforms=("cuda",))
            mlir_text = lowered.as_text()
            assert "pallas_prune_and_encode_cell_list" in mlir_text
        except RuntimeError as exc:
            if "No supported GPU devices found" not in str(exc):
                raise
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


def test_pallas_v24_ghost_atom_filtering_and_compaction_in_nl_and_prune():
    """v2.4: Ghost atoms (~atom_mask) are excluded from cell buffers and compacted to trailing sentinels."""
    from jax_md import partition, space
    from prolix.physics.neighbor_list import (
        _NB_INDEX_MASK,
        encode_nl_exclusions,
        pallas_prune_and_encode_cell_list,
    )

    N_bucket = 64
    n_real = 20
    box = jnp.array([30.0, 30.0, 30.0], dtype=jnp.float32)
    r_cutoff = 9.0
    rng = np.random.default_rng(123)
    # Place ghost atoms (20:64) right inside the same volume as real atoms (0:20)
    pos = jnp.asarray(rng.uniform(2.0, 12.0, size=(N_bucket, 3)).astype(np.float32))
    atom_mask = jnp.arange(N_bucket) < n_real

    ei = jnp.full((N_bucket, 4), -1, dtype=jnp.int32)
    ev = jnp.ones((N_bucket, 4), dtype=jnp.float32)
    ee = jnp.ones((N_bucket, 4), dtype=jnp.float32)

    disp_fn, _ = space.periodic(box)
    nl_fns = partition.neighbor_list(
        disp_fn, box, r_cutoff=r_cutoff, dr_threshold=0.5, capacity_multiplier=1.25, mask_self=True
    )
    nbr_raw = nl_fns.allocate(pos, extra_capacity=32)
    # Raw jax_md neighbor list contains ghost indices >= n_real because ghosts sit near real atoms
    raw_real_rows = np.asarray(nbr_raw.idx[:n_real])
    assert np.any((raw_real_rows >= n_real) & (raw_real_rows < N_bucket))

    # 1. encode_nl_exclusions with atom_mask filters ghost neighbors and compacts valid entries left
    enc = np.asarray(
        encode_nl_exclusions(nbr_raw.idx, ei, ev, ee, n_atoms=N_bucket, atom_mask=atom_mask)
    )
    # Ghost rows (20:64) must be 100% sentinel N_bucket
    assert np.all(enc[n_real:] == N_bucket)
    # Real rows (0:20) must have zero ghost indices in [n_real, N_bucket) and all valid entries compacted left
    for i in range(n_real):
        row = enc[i]
        base_indices = np.where(row == N_bucket, N_bucket, row & _NB_INDEX_MASK)
        assert not np.any((base_indices >= n_real) & (base_indices < N_bucket))
        # Check compaction: once N_bucket appears, all subsequent slots must be N_bucket
        first_sent = np.where(row == N_bucket)[0]
        if len(first_sent) > 0:
            assert np.all(row[first_sent[0] :] == N_bucket)

    # 2. pallas_prune_and_encode_cell_list with atom_mask (single and batched heterogeneous masks)
    K = 64
    C = 32
    nbr_pal, occ_pal = pallas_prune_and_encode_cell_list(
        pos,
        box,
        ei,
        ev,
        ee,
        cell_size=9.5,
        cell_capacity=C,
        max_occupancy=K,
        r_cutoff=9.5,
        interpret=True,
        atom_mask=atom_mask,
    )
    nbr_pal_np = np.asarray(nbr_pal)
    assert np.all(nbr_pal_np[n_real:] == N_bucket)
    base_pal = np.where(nbr_pal_np[:n_real] == N_bucket, N_bucket, nbr_pal_np[:n_real] & _NB_INDEX_MASK)
    assert not np.any((base_pal >= n_real) & (base_pal < N_bucket))
    assert int(occ_pal) <= n_real - 1

    # Batched vmap over 2 replicas with heterogeneous atom_masks (n_real=20 and n_real=12)
    mask_b = jnp.stack([jnp.arange(N_bucket) < 20, jnp.arange(N_bucket) < 12], axis=0)
    pos_b = jnp.stack([pos, pos], axis=0)
    nbr_b, occ_b = jax.vmap(
        lambda p, m: pallas_prune_and_encode_cell_list(
            p,
            box,
            ei,
            ev,
            ee,
            cell_size=9.5,
            cell_capacity=C,
            max_occupancy=K,
            r_cutoff=9.5,
            interpret=True,
            atom_mask=m,
        )
    )(pos_b, mask_b)
    nbr_b_np = np.asarray(nbr_b)
    assert np.all(nbr_b_np[0, 20:] == N_bucket)
    assert np.all(nbr_b_np[1, 12:] == N_bucket)
    base_b1 = np.where(nbr_b_np[1, :12] == N_bucket, N_bucket, nbr_b_np[1, :12] & _NB_INDEX_MASK)
    assert not np.any((base_b1 >= 12) & (base_b1 < N_bucket))
    assert int(occ_b[1]) <= 11


def test_pallas_v24_stacked_inference_cache_and_dynamic_n_steps(monkeypatch):
    """v2.4: Repeated multi-bundle plan.run(run_mode='inference') hits _STACKED_INFERENCE_CACHE across varying n_steps."""
    from prolix.api.ensemble_plan import _STACKED_INFERENCE_CACHE

    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        _STACKED_INFERENCE_CACHE.clear()
        b1 = _make_periodic_solvated_bundle(dtype=jnp.float64, seed=21)
        b2 = _make_periodic_solvated_bundle(dtype=jnp.float64, seed=22, shift_solute=0.5)

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

        plan = EnsemblePlan.from_bundles([b1, b2], use_pallas=True)
        trajs_1 = plan.run(
            n_steps=2,
            dt=1.0,
            kT=0.6,
            seed=10,
            run_mode="inference",
            use_neighbor_list=True,
            nl_update_every=2,
        )
        first_allocs = alloc_calls["n"]
        assert first_allocs == 1, f"Expected exactly 1 host NL allocate call on first stacked run, got {first_allocs}"
        assert all(t.n_steps == 2 for t in trajs_1)

        # Second call with DIFFERENT n_steps (4) and seed (99) must reuse _STACKED_INFERENCE_CACHE
        trajs_2 = plan.run(
            n_steps=4,
            dt=1.0,
            kT=0.6,
            seed=99,
            run_mode="inference",
            use_neighbor_list=True,
            nl_update_every=2,
        )
        assert alloc_calls["n"] == first_allocs, "Expected zero additional host NL allocations on cached stacked run"
        assert all(t.n_steps == 4 for t in trajs_2)
        assert all(bool(jnp.all(jnp.isfinite(t.positions))) for t in trajs_2)
    finally:
        _STACKED_INFERENCE_CACHE.clear()
        jax.config.update("jax_enable_x64", prev_x64)


def test_pallas_v24_multi_gpu_compiler_params_resolution(monkeypatch):
    """v2.4: resolve_pallas_compiler_params adapts across GPU architectures, grid sizes, and env overrides."""
    from prolix.export import resolve_pallas_compiler_params as resolve_from_export
    from prolix.pallas_kernels import resolve_pallas_compiler_params

    assert resolve_from_export is resolve_pallas_compiler_params

    monkeypatch.delenv("PROLIX_PALLAS_NUM_WARPS", raising=False)
    monkeypatch.delenv("PROLIX_PALLAS_NUM_STAGES", raising=False)

    # Small grid (< 256 blocks) uses 1 stage for higher SM occupancy across architectures
    for gpu_name in (
        "NVIDIA B300",
        "NVIDIA RTX PRO 6000 Blackwell",
        "NVIDIA H200",
        "NVIDIA L40S",
        "NVIDIA A100",
    ):
        # Small grid (< 256 blocks) uses 1 stage for higher SM occupancy across architectures
        cp_small = resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=64, device_kind=gpu_name)
        assert cp_small.num_warps == 4
        assert cp_small.num_stages == 1

        cp_large = resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=1024, device_kind=gpu_name)
        assert cp_large.num_warps == 4
        # Blackwell (B300 / RTX PRO 6000 Blackwell) uses num_stages=1 across all grid sizes (MK-43/MK-45)
        expected_stages = 1 if ("B300" in gpu_name or "Blackwell" in gpu_name) else 2
        assert cp_large.num_stages == expected_stages

    cp_bonded = resolve_pallas_compiler_params("bonded_fused", grid_blocks=128, device_kind="NVIDIA B300")
    assert cp_bonded.num_warps == 2
    assert cp_bonded.num_stages == 1

    # Environment variable override
    monkeypatch.setenv("PROLIX_PALLAS_NUM_WARPS", "8")
    monkeypatch.setenv("PROLIX_PALLAS_NUM_STAGES", "3")
    cp_env = resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=512, device_kind="NVIDIA H200")
    assert cp_env.num_warps == 8
    assert cp_env.num_stages == 3


def test_pallas_v25_fused_bonded_and_nonbonded_nl_megakernel_parity_and_dispatch(monkeypatch):
    """MK-36: pallas_bonded_and_nonbonded_nl_fused matches separate kernels (B=1, vmap B=2, param grads, CUDA lowering) and is dispatched by single_padded_energy."""
    import prolix.pallas_kernels as pk
    from prolix.pallas_kernels import (
        pallas_bonded_all_fused,
        pallas_bonded_and_nonbonded_nl_fused,
        pallas_nonbonded_nl,
    )

    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        bundle = _make_periodic_solvated_bundle(dtype=jnp.float64, seed=55)
        disp_fn, _ = displacement_fn_for_bundle(bundle)
        box_vec = jnp.diag(bundle.box)
        neighbor_fn = nl_mod.make_neighbor_list_fn(
            disp_fn, box_vec, 9.0, capacity_multiplier=1.5, dr_threshold=0.5, atom_mask=bundle.atom_mask
        )
        nbr_enc = _encode_bundle_nl(neighbor_fn.allocate(bundle.positions), bundle)

        def _fused_eval(pos, q, sig, eps, bp, ap, dp):
            return pallas_bonded_and_nonbonded_nl_fused(
                pos,
                q,
                sig,
                eps,
                bundle.atom_mask,
                nbr_enc.idx,
                bundle.excl_dense_indices,
                bundle.excl_dense_scales_vdw,
                bundle.excl_dense_scales_elec,
                bonds=bundle.bond_idx,
                bond_params=bp,
                bond_mask=bundle.bond_mask,
                angles=bundle.angle_idx,
                angle_params=ap,
                angle_mask=bundle.angle_mask,
                dihedrals=bundle.dihedral_idx,
                dihedral_params=dp,
                dihedral_mask=bundle.dihedral_mask,
                impropers=bundle.improper_idx,
                improper_params=bundle.improper_params,
                improper_mask=bundle.improper_mask,
                exception_pairs=bundle.exception_pairs,
                exception_sigmas=bundle.exception_sigmas,
                exception_epsilons=bundle.exception_epsilons,
                exception_chargeprods=bundle.exception_chargeprods,
                exception_mask=bundle.exception_mask,
                box=box_vec,
                pme_alpha=float(bundle.pme_alpha),
                cutoff=float(bundle.cutoff_distance),
                excl_mode="pre_encoded",
                interpret=True,
            )

        def _sep_eval(pos, q, sig, eps, bp, ap, dp):
            e_b = pallas_bonded_all_fused(
                pos,
                bonds=bundle.bond_idx,
                bond_params=bp,
                bond_mask=bundle.bond_mask,
                angles=bundle.angle_idx,
                angle_params=ap,
                angle_mask=bundle.angle_mask,
                dihedrals=bundle.dihedral_idx,
                dihedral_params=dp,
                dihedral_mask=bundle.dihedral_mask,
                impropers=bundle.improper_idx,
                improper_params=bundle.improper_params,
                improper_mask=bundle.improper_mask,
                exception_pairs=bundle.exception_pairs,
                exception_sigmas=bundle.exception_sigmas,
                exception_epsilons=bundle.exception_epsilons,
                exception_chargeprods=bundle.exception_chargeprods,
                exception_mask=bundle.exception_mask,
                box=box_vec,
                interpret=True,
            )
            e_nb = pallas_nonbonded_nl(
                pos,
                q,
                sig,
                eps,
                bundle.atom_mask,
                nbr_enc.idx,
                bundle.excl_dense_indices,
                bundle.excl_dense_scales_vdw,
                bundle.excl_dense_scales_elec,
                box=box_vec,
                pme_alpha=float(bundle.pme_alpha),
                cutoff=float(bundle.cutoff_distance),
                excl_mode="pre_encoded",
                interpret=True,
            )
            return e_b + e_nb

        args = (
            bundle.positions,
            bundle.charges,
            bundle.sigmas,
            bundle.epsilons,
            bundle.bond_params,
            bundle.angle_params,
            bundle.dihedral_params,
        )
        # 1. Position-only grad parity (MD fast path)
        e_f, g_f = jax.value_and_grad(_fused_eval, argnums=0)(*args)
        e_s, g_s = jax.value_and_grad(_sep_eval, argnums=0)(*args)
        np.testing.assert_allclose(np.asarray(e_f), np.asarray(e_s), rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(np.asarray(g_f), np.asarray(g_s), rtol=1e-4, atol=1e-4)

        # 2. Full parameter gradient parity (symbolic_zeros=True active AD)
        _, grads_f = jax.value_and_grad(_fused_eval, argnums=(0, 1, 2, 3, 4, 5, 6))(*args)
        _, grads_s = jax.value_and_grad(_sep_eval, argnums=(0, 1, 2, 3, 4, 5, 6))(*args)
        for gf, gs in zip(grads_f, grads_s):
            np.testing.assert_allclose(np.asarray(gf), np.asarray(gs), rtol=1e-4, atol=1e-4)

        # 3. Batched vmap (B=2) + single_padded_energy dispatch verification
        fused_batch_calls: list[int] = []
        orig_fused_batch = pk._pallas_bonded_and_nonbonded_nl_fused_batch

        def spy_fused_batch(B, *a, **kw):
            fused_batch_calls.append(int(B))
            return orig_fused_batch(B, *a, **kw)

        monkeypatch.setattr(pk, "_pallas_bonded_and_nonbonded_nl_fused_batch", spy_fused_batch)
        e_fn = energy_fn_from_bundle(
            bundle,
            build_dense_exclusions=False,
            use_pallas=True,
            use_pallas_pme=True,
            excl_mode="pre_encoded",
        )
        pos_b = jnp.stack([bundle.positions, bundle.positions + 0.05], axis=0)
        nbr_b = jax.tree_util.tree_map(lambda x: jnp.stack([x, x], axis=0), nbr_enc)
        vg_b = jax.jit(jax.vmap(jax.value_and_grad(lambda r, n: e_fn(r, neighbor=n))))
        eb, gb = vg_b(pos_b, nbr_b)
        assert 2 in fused_batch_calls
        assert bool(jnp.all(jnp.isfinite(eb)))
        assert bool(jnp.all(jnp.isfinite(gb)))

        # 4. Offline CUDA Triton MLIR lowering verification (interpret=False)
        traced_cuda = jax.jit(
            lambda r: pallas_bonded_and_nonbonded_nl_fused(
                r,
                bundle.charges,
                bundle.sigmas,
                bundle.epsilons,
                bundle.atom_mask,
                nbr_enc.idx,
                bundle.excl_dense_indices,
                bundle.excl_dense_scales_vdw,
                bundle.excl_dense_scales_elec,
                bonds=bundle.bond_idx,
                bond_params=bundle.bond_params,
                bond_mask=bundle.bond_mask,
                angles=bundle.angle_idx,
                angle_params=bundle.angle_params,
                angle_mask=bundle.angle_mask,
                dihedrals=bundle.dihedral_idx,
                dihedral_params=bundle.dihedral_params,
                dihedral_mask=bundle.dihedral_mask,
                box=box_vec,
                pme_alpha=float(bundle.pme_alpha),
                cutoff=float(bundle.cutoff_distance),
                excl_mode="pre_encoded",
                interpret=False,
            )
        ).trace(bundle.positions)
        try:
            lowered_cuda = traced_cuda.lower(lowering_platforms=("cuda",))
            assert "pallas_bonded_and_nonbonded_nl_fused" in lowered_cuda.as_text()
        except RuntimeError as exc:
            if "No supported GPU devices found" not in str(exc):
                raise
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_pallas_v25_compacted_k_tile_early_exit_parity():
    """MK-37: while_loop K-tile early exit on compacted pre_encoded NL matches across K=32 and K=256."""
    from prolix.pallas_kernels import pallas_nonbonded_nl
    from prolix.physics.neighbor_list import encode_nl_exclusions

    N = 64
    rng = np.random.default_rng(2026)
    pos = jnp.asarray(rng.uniform(2.0, 22.0, size=(N, 3)).astype(np.float32))
    q = jnp.asarray(rng.uniform(-0.4, 0.4, size=(N,)).astype(np.float32))
    sig = jnp.full((N,), 3.1, dtype=jnp.float32)
    eps = jnp.full((N,), 0.08, dtype=jnp.float32)
    mask = jnp.arange(N) < 48
    box = jnp.array([25.0, 25.0, 25.0], dtype=jnp.float32)
    ei = jnp.full((N, 4), -1, dtype=jnp.int32)
    ev = jnp.ones((N, 4), dtype=jnp.float32)
    ee = jnp.ones((N, 4), dtype=jnp.float32)

    # Construct a sparse neighbor list with <= 12 neighbors per atom, padded to K=32 vs K=256
    nbr_raw_32 = np.full((N, 32), N, dtype=np.int32)
    for i in range(48):
        # Put valid neighbors at scattered column indices to also verify compaction when atom_mask=None
        nbr_raw_32[i, [3, 11, 19, 27]] = [(i + 1) % 48, (i + 2) % 48, (i + 3) % 48, (i + 5) % 48]
    nbr_enc_32 = encode_nl_exclusions(jnp.asarray(nbr_raw_32), ei, ev, ee, n_atoms=N, atom_mask=None)
    # Verify compaction moved the 4 valid entries to columns 0..3 even with atom_mask=None
    assert bool(jnp.all(nbr_enc_32[:48, 4:] == N))

    nbr_raw_256 = np.full((N, 256), N, dtype=np.int32)
    nbr_raw_256[:, :32] = nbr_raw_32
    nbr_enc_256 = encode_nl_exclusions(jnp.asarray(nbr_raw_256), ei, ev, ee, n_atoms=N, atom_mask=mask)
    assert bool(jnp.all(nbr_enc_256[:, 32:] == N))

    e_32, g_32 = jax.value_and_grad(
        lambda r: pallas_nonbonded_nl(
            r, q, sig, eps, mask, nbr_enc_32, ei, ev, ee,
            box=box, pme_alpha=0.34, cutoff=9.0, excl_mode="pre_encoded", interpret=True,
        )
    )(pos)
    e_256, g_256 = jax.value_and_grad(
        lambda r: pallas_nonbonded_nl(
            r, q, sig, eps, mask, nbr_enc_256, ei, ev, ee,
            box=box, pme_alpha=0.34, cutoff=9.0, excl_mode="pre_encoded", interpret=True,
        )
    )(pos)
    np.testing.assert_allclose(np.asarray(e_32), np.asarray(e_256), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(g_32), np.asarray(g_256), rtol=1e-6, atol=1e-6)


def test_pallas_v25_spme_active_ad_charge_gradient_gating(monkeypatch):
    """MK-38: spme_energy_with_forces sets compute_dq=False when charges is unperturbed and True when differentiated."""
    import prolix.physics.pme_pallas as pme_pallas
    from prolix.physics.pme import spme_energy_with_forces

    N = 32
    rng = np.random.default_rng(88)
    pos = jnp.asarray(rng.uniform(1.0, 18.0, size=(N, 3)).astype(np.float32))
    q = jnp.asarray(rng.uniform(-0.5, 0.5, size=(N,)).astype(np.float32))
    q = q - jnp.mean(q)
    mask = jnp.ones((N,), dtype=jnp.bool_)
    box = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)
    grid_dims = (16, 16, 16)

    seen_compute_dq: list[bool] = []
    orig_gather_batch = pme_pallas._gather_pallas_batch

    def spy_gather_batch(B, in_batched, positions, charges, atom_mask, box_size, theta_norm, gdims, interpret, compute_dq=True):
        seen_compute_dq.append(bool(compute_dq))
        return orig_gather_batch(
            B, in_batched, positions, charges, atom_mask, box_size, theta_norm, gdims, interpret, compute_dq=compute_dq
        )

    monkeypatch.setattr(pme_pallas, "_gather_pallas_batch", spy_gather_batch)

    # 1. Position-only gradient (MD step): compute_dq must be False
    seen_compute_dq.clear()
    e_pos, g_pos = jax.value_and_grad(
        lambda r: spme_energy_with_forces(r, q, mask, box, grid_dims, 0.34, 4, True)
    )(pos)
    assert seen_compute_dq and all(v is False for v in seen_compute_dq)
    assert bool(jnp.isfinite(e_pos)) and bool(jnp.all(jnp.isfinite(g_pos)))

    # 2. Position + charge gradient: compute_dq must be True and match XLA reference
    seen_compute_dq.clear()
    e_pq, (g_pos_pq, g_q_pal) = jax.value_and_grad(
        lambda r, c: spme_energy_with_forces(r, c, mask, box, grid_dims, 0.34, 4, True),
        argnums=(0, 1),
    )(pos, q)
    assert seen_compute_dq and all(v is True for v in seen_compute_dq)
    _, (g_pos_xla, g_q_xla) = jax.value_and_grad(
        lambda r, c: spme_energy_with_forces(r, c, mask, box, grid_dims, 0.34, 4, False),
        argnums=(0, 1),
    )(pos, q)
    np.testing.assert_allclose(np.asarray(g_pos), np.asarray(g_pos_pq), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(g_q_pal), np.asarray(g_q_xla), rtol=1e-4, atol=1e-4)


def test_pallas_v26_spme_float32_fft_and_single_megakernel_store_and_grid_points(monkeypatch):
    """MK-36, MK-41, MK-42:
    1. Under jax_enable_x64=True and use_pallas_pme=True, _spme_fwd and _spme_cached_fwd run 3D FFTs
       in float32/complex64 (theta_norm.dtype == float32) while returning float64 energy and forces,
       for both 1D (3,) and 2D batched orthorhombic (B, 3) boxes (including B=3).
    2. pallas_bonded_and_nonbonded_nl_fused with total_bonded_blocks == 0 and with mixed bonded
       protein blocks + pure-solvent blocks (blk_has_bonded direct _pl_store fast path) matches
       separate kernels to float32 precision.
    3. PROLIX_PME_GRID_POINTS (both scalar and bucket-mapped '8192:48,25000:64') overrides
       make_spme_energy_fn's FFT grid dimensions when grid_dims is None.
    """
    from prolix.pallas_kernels import (
        pallas_bonded_all_fused,
        pallas_bonded_and_nonbonded_nl_fused,
        pallas_nonbonded_nl,
    )
    from prolix.physics.neighbor_list import encode_nl_exclusions
    from prolix.physics.pme import (
        _resolve_env_pme_grid_points,
        _spme_cached_fwd,
        _spme_fwd,
        make_spme_energy_fn,
        spme_energy_with_forces,
    )

    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        N = 64
        rng = np.random.default_rng(123)
        pos_f64 = jnp.asarray(rng.uniform(1.5, 18.5, size=(N, 3)), dtype=jnp.float64)
        q_f64 = jnp.asarray(rng.uniform(-0.4, 0.4, size=(N,)), dtype=jnp.float64)
        q_f64 = q_f64 - jnp.mean(q_f64)
        mask = jnp.ones((N,), dtype=jnp.bool_)
        box_f64 = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float64)
        grid_dims = (16, 16, 16)

        # 1a. Verify _spme_fwd and _spme_cached_fwd produce float32 theta_norm when use_pallas_pme=True
        e_fwd, res_fwd = _spme_fwd(pos_f64, q_f64, mask, box_f64, grid_dims, 0.34, 4, True)
        theta_norm_fwd = res_fwd[4]
        assert theta_norm_fwd.dtype == jnp.float32
        assert e_fwd.dtype == jnp.float64

        G_dummy = jnp.ones((16, 16, 9), dtype=jnp.float64)
        e_cfwd, res_cfwd = _spme_cached_fwd(pos_f64, q_f64, mask, G_dummy, box_f64, grid_dims, 0.34, 4, True)
        theta_norm_cfwd = res_cfwd[4]
        assert theta_norm_cfwd.dtype == jnp.float32
        assert e_cfwd.dtype == jnp.float64

        # Verify end-to-end energy and force parity against XLA float64 reference
        e_pal, g_pal = jax.value_and_grad(
            lambda r: spme_energy_with_forces(r, q_f64, mask, box_f64, grid_dims, 0.34, 4, True)
        )(pos_f64)
        e_xla, g_xla = jax.value_and_grad(
            lambda r: spme_energy_with_forces(r, q_f64, mask, box_f64, grid_dims, 0.34, 4, False)
        )(pos_f64)
        assert e_pal.dtype == jnp.float64 and g_pal.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(e_pal), np.asarray(e_xla), rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(np.asarray(g_pal), np.asarray(g_xla), rtol=1e-4, atol=1e-4)

        # 1b. Verify batched orthorhombic box_size (B, 3) for B=2 and B=3 (where shape == (3, 3))
        for B_test in (2, 3):
            pos_b = jnp.stack([pos_f64 + 0.03 * b for b in range(B_test)], axis=0)
            q_b = jnp.stack([q_f64 for _ in range(B_test)], axis=0)
            mask_b = jnp.stack([mask for _ in range(B_test)], axis=0)
            box_b = jnp.stack(
                [jnp.array([20.0 + 0.5 * b, 20.0 + 0.5 * b, 20.0 + 0.5 * b], dtype=jnp.float64) for b in range(B_test)],
                axis=0,
            )
            e_b_fwd, res_b_fwd = _spme_fwd(pos_b, q_b, mask_b, box_b, grid_dims, 0.34, 4, True)
            assert res_b_fwd[4].dtype == jnp.float32
            assert e_b_fwd.shape == (B_test,) and e_b_fwd.dtype == jnp.float64

            g_b = jax.grad(
                lambda r: jnp.sum(spme_energy_with_forces(r, q_b, mask_b, box_b, grid_dims, 0.34, 4, True))
            )(pos_b)
            g_b_ref = jnp.stack(
                [
                    jax.grad(
                        lambda r1, b=b: spme_energy_with_forces(
                            r1, q_b[b], mask_b[b], box_b[b], grid_dims, 0.34, 4, False
                        )
                    )(pos_b[b])
                    for b in range(B_test)
                ],
                axis=0,
            )
            np.testing.assert_allclose(np.asarray(g_b), np.asarray(g_b_ref), rtol=1e-4, atol=1e-4)

        # 2. Verify PROLIX_PME_GRID_POINTS (scalar and bucket-mapped) in make_spme_energy_fn
        monkeypatch.setenv("PROLIX_PME_GRID_POINTS", "24")
        fn_24 = make_spme_energy_fn(box_f64, alpha=0.34, grid_spacing=1.0, use_pallas_pme=True)
        assert fn_24.G_cached.shape == (24, 24, 13)
        monkeypatch.setenv("PROLIX_PME_GRID_POINTS", "8192:48,25000:64")
        assert _resolve_env_pme_grid_points(n_atoms=4199) == 48
        assert _resolve_env_pme_grid_points(n_atoms=8192) == 48
        assert _resolve_env_pme_grid_points(n_atoms=25000) == 64
        fn_48 = make_spme_energy_fn(box_f64, alpha=0.34, grid_spacing=1.0, use_pallas_pme=True, n_atoms=8192)
        fn_64 = make_spme_energy_fn(box_f64, alpha=0.34, grid_spacing=1.0, use_pallas_pme=True, n_atoms=25000)
        assert fn_48.G_cached.shape == (48, 48, 25)
        assert fn_64.G_cached.shape == (64, 64, 33)
        monkeypatch.delenv("PROLIX_PME_GRID_POINTS", raising=False)

        # 3. Verify pallas_bonded_and_nonbonded_nl_fused with total_bonded_blocks == 0 AND
        #    with mixed bonded block 0 (atoms 0..31) + pure-water block 1 (atoms 32..63, zero bonded terms),
        #    plus MK-43 resolve_pallas_compiler_params("megakernel") num_warps=4, num_stages=1
        from prolix.export import resolve_pallas_compiler_params

        cp_mk = resolve_pallas_compiler_params("megakernel", grid_blocks=4096)
        assert cp_mk.num_warps == 4
        assert cp_mk.num_stages == 1
        sig = jnp.full((N,), 3.1, dtype=jnp.float64)
        eps = jnp.full((N,), 0.12, dtype=jnp.float64)
        ei = jnp.full((N, 4), -1, dtype=jnp.int32)
        ev = jnp.ones((N, 4), dtype=jnp.float64)
        ee = jnp.ones((N, 4), dtype=jnp.float64)
        nbr_raw = np.full((N, 16), N, dtype=np.int32)
        for i in range(N):
            nbr_raw[i, :3] = [(i + 1) % N, (i + 2) % N, (i + 3) % N]
        nbr_enc = encode_nl_exclusions(jnp.asarray(nbr_raw), ei, ev, ee, n_atoms=N, atom_mask=mask)

        e_fused_0, g_fused_0 = jax.value_and_grad(
            lambda r: pallas_bonded_and_nonbonded_nl_fused(
                r, q_f64, sig, eps, mask, nbr_enc, ei, ev, ee,
                box=box_f64, pme_alpha=0.34, cutoff=9.0, excl_mode="pre_encoded", interpret=True,
            )
        )(pos_f64)
        e_nb_only, g_nb_only = jax.value_and_grad(
            lambda r: pallas_nonbonded_nl(
                r, q_f64, sig, eps, mask, nbr_enc, ei, ev, ee,
                box=box_f64, pme_alpha=0.34, cutoff=9.0, excl_mode="pre_encoded", interpret=True,
            )
        )(pos_f64)
        np.testing.assert_allclose(np.asarray(e_fused_0), np.asarray(e_nb_only), rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(np.asarray(g_fused_0), np.asarray(g_nb_only), rtol=1e-5, atol=1e-3)

        # Mixed protein (block 0 has harmonic bonds) + pure water (block 1 has zero bonded terms)
        bonds_blk0 = jnp.array([[0, 1], [2, 3], [10, 15], [20, 31]], dtype=jnp.int32)
        bond_params_blk0 = jnp.array(
            [[1.5, 300.0], [1.4, 280.0], [1.55, 250.0], [1.48, 310.0]], dtype=jnp.float64
        )
        bond_mask_blk0 = jnp.ones((4,), dtype=jnp.float64)

        e_mixed, g_mixed = jax.value_and_grad(
            lambda r: pallas_bonded_and_nonbonded_nl_fused(
                r, q_f64, sig, eps, mask, nbr_enc, ei, ev, ee,
                bonds=bonds_blk0, bond_params=bond_params_blk0, bond_mask=bond_mask_blk0,
                box=box_f64, pme_alpha=0.34, cutoff=9.0, excl_mode="pre_encoded", interpret=True,
            )
        )(pos_f64)
        e_sep_m, g_sep_m = jax.value_and_grad(
            lambda r: (
                pallas_bonded_all_fused(
                    r, bonds=bonds_blk0, bond_params=bond_params_blk0, bond_mask=bond_mask_blk0,
                    box=box_f64, interpret=True,
                )
                + pallas_nonbonded_nl(
                    r, q_f64, sig, eps, mask, nbr_enc, ei, ev, ee,
                    box=box_f64, pme_alpha=0.34, cutoff=9.0, excl_mode="pre_encoded", interpret=True,
                )
            )
        )(pos_f64)
        np.testing.assert_allclose(np.asarray(e_mixed), np.asarray(e_sep_m), rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(np.asarray(g_mixed), np.asarray(g_sep_m), rtol=1e-5, atol=1e-3)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_pallas_v30_mk46_mk49_integrator_rebuild_nonbonded_pme():
    """Verify Pallas v3.0 Pillars 1-4 (MK-46 through MK-49) on CPU interpret and XLA reference:
    - MK-46: Fused Pallas BAOAB + analytical Miyamoto-Kollman SETTLE pre/post-force kernels
      (B=1 and B=3, rigid water geometry & velocity orthogonality, ghost atom pinning, export fallback).
    - MK-47: Per-replica independent neighbor-list rebuild masking (rebuild_mask=[False, True, False])
      preserving non-rebuilding replicas' neighbor lists in O(1) while rebuilding triggered replicas.
    - MK-48: Spatially pruned 32x32 block nonbonded kernel with O(1) ghost block skip, skin-shell load
      gating, branchless reciprocal-multiply PBC/forces, and custom_vjp parameter gradients.
    - MK-49: Separable (BLOCK_N=32, 16) YZ-plane factorized & charge-gated SPME spread/gather.
    """
    from prolix.physics.neighbor_list import (
        _pallas_prune_and_encode_cell_list_batch,
        pallas_prune_and_encode_cell_list,
    )
    from prolix.physics.pme import (
        _gather_reference,
        _spread_charges_reference,
    )
    from prolix.physics.pme_pallas import gather_pallas, spread_charges_pallas
    from prolix.physics.settle import (
        TIP3P_RHH as HH_DIST,
        TIP3P_ROH as OH_DIST,
        _langevin_settle_vel,
        pallas_baoab_post_force,
        pallas_baoab_pre_force,
    )

    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        # --- MK-46: Fused BAOAB + analytical Miyamoto-Kollman SETTLE ---
        # 4 solute atoms (0..3), 2 rigid waters (4..9), 2 ghost atoms (10..11)
        water_idx = jnp.array([[4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
        water_mask = jnp.array([True, True], dtype=jnp.bool_)
        atom_mask = jnp.array(
            [True, True, True, True, True, True, True, True, True, True, False, False],
            dtype=jnp.bool_,
        )
        masses = jnp.array(
            [12.0, 14.0, 16.0, 12.0, 15.999, 1.008, 1.008, 15.999, 1.008, 1.008, 1.0, 1.0],
            dtype=jnp.float32,
        )[:, None]
        box = jnp.array([25.0, 25.0, 25.0], dtype=jnp.float32)

        # Construct valid water geometries
        half_hh = 0.5 * HH_DIST
        h_y = float(np.sqrt(max(0.0, OH_DIST**2 - half_hh**2)))
        pos0 = np.zeros((12, 3), dtype=np.float32)
        pos0[0:4] = [[2.0, 3.0, 4.0], [3.5, 3.0, 4.0], [5.0, 3.2, 4.1], [6.2, 3.1, 4.0]]
        pos0[4] = [10.0, 10.0, 10.0]
        pos0[5] = [10.0 + half_hh, 10.0 + h_y, 10.0]
        pos0[6] = [10.0 - half_hh, 10.0 + h_y, 10.0]
        pos0[7] = [15.0, 12.0, 8.0]
        pos0[8] = [15.0 + half_hh, 12.0 + h_y, 8.0]
        pos0[9] = [15.0 - half_hh, 12.0 + h_y, 8.0]
        pos0[10:] = [[99.0, 99.0, 99.0], [88.0, 88.0, 88.0]]
        r0 = jnp.asarray(pos0)

        rng = np.random.default_rng(2026)
        p0 = jnp.asarray(rng.normal(0.0, 0.2, size=(12, 3)).astype(np.float32))
        p0 = jnp.where(atom_mask[:, None], p0, 0.0)
        f0 = jnp.asarray(rng.normal(0.0, 1.5, size=(12, 3)).astype(np.float32))
        dt = jnp.float32(0.04)
        kT = jnp.float32(0.6)
        gamma = 0.1
        key0 = jax.random.PRNGKey(42)

        r_pal, p_pal, _ = pallas_baoab_pre_force(
            r0,
            p0,
            f0,
            masses,
            key0,
            water_idx,
            water_mask,
            box,
            dt,
            kT,
            gamma,
            r_OH=OH_DIST,
            r_HH=HH_DIST,
            interpret=True,
            atom_mask=atom_mask,
        )
        # Check water geometry constraints on r_pal
        for w in range(2):
            ow, h1, h2 = r_pal[water_idx[w, 0]], r_pal[water_idx[w, 1]], r_pal[water_idx[w, 2]]
            d_oh1 = float(jnp.linalg.norm(h1 - ow))
            d_oh2 = float(jnp.linalg.norm(h2 - ow))
            d_hh = float(jnp.linalg.norm(h2 - h1))
            np.testing.assert_allclose(d_oh1, OH_DIST, atol=2e-4)
            np.testing.assert_allclose(d_oh2, OH_DIST, atol=2e-4)
            np.testing.assert_allclose(d_hh, HH_DIST, atol=2e-4)
        # Check ghost atoms pinned to original r0 and zero momentum
        np.testing.assert_allclose(np.asarray(r_pal[10:]), np.asarray(r0[10:]), atol=1e-6)
        np.testing.assert_allclose(np.asarray(p_pal[10:]), 0.0, atol=1e-6)

        # Check post-force velocity SETTLE orthogonality and parity with pure-JAX _langevin_settle_vel
        p_post = pallas_baoab_post_force(
            r_pal,
            p_pal,
            f0,
            masses,
            water_idx,
            water_mask,
            dt,
            interpret=True,
            atom_mask=atom_mask,
        )
        p_ref_kick = jnp.where(atom_mask[:, None], p_pal + jnp.float32(0.5) * dt * f0, 0.0)
        p_ref = _langevin_settle_vel(
            p_ref_kick,
            r_pal,
            r_pal,
            masses,
            water_idx,
            dt,
            15.999,
            1.008,
            water_mask=water_mask,
            atom_mask=atom_mask,
        )
        p_ref = jnp.where(atom_mask[:, None], p_ref, 0.0)
        np.testing.assert_allclose(np.asarray(p_post), np.asarray(p_ref), rtol=1e-4, atol=1e-4)

        # Verify vmap batched B=3 parity
        r_b = jnp.stack([r0, r0, r0], axis=0)
        p_b = jnp.stack([p0, p0 * 0.5, p0 * -0.5], axis=0)
        f_b = jnp.stack([f0, f0 * 0.8, f0 * 1.2], axis=0)
        keys_b = jnp.stack([key0, jax.random.PRNGKey(43), jax.random.PRNGKey(44)], axis=0)
        r_b_out, p_b_out, _ = jax.vmap(
            lambda r_i, p_i, f_i, k_i: pallas_baoab_pre_force(
                r_i,
                p_i,
                f_i,
                masses,
                k_i,
                water_idx,
                water_mask,
                box,
                dt,
                kT,
                gamma,
                interpret=True,
                atom_mask=atom_mask,
            )
        )(r_b, p_b, f_b, keys_b)
        np.testing.assert_allclose(np.asarray(r_b_out[0]), np.asarray(r_pal), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(p_b_out[0]), np.asarray(p_pal), rtol=1e-5, atol=1e-5)

        # Verify CUDA Triton tracing/lowering (interpret=False) in both float32 and float64
        for dt_check in (jnp.float32, jnp.float64):
            traced_pre = jax.jit(
                lambda r_in, p_in, f_in, k_in: pallas_baoab_pre_force(
                    r_in,
                    p_in,
                    f_in,
                    masses.astype(dt_check),
                    k_in,
                    water_idx,
                    water_mask,
                    box.astype(dt_check),
                    dt.astype(dt_check),
                    kT.astype(dt_check),
                    gamma,
                    interpret=False,
                    atom_mask=atom_mask,
                )
            ).trace(r0.astype(dt_check), p0.astype(dt_check), f0.astype(dt_check), key0)
            traced_post = jax.jit(
                lambda r_in, p_in, f_in: pallas_baoab_post_force(
                    r_in,
                    p_in,
                    f_in,
                    masses.astype(dt_check),
                    water_idx,
                    water_mask,
                    dt.astype(dt_check),
                    interpret=False,
                    atom_mask=atom_mask,
                )
            ).trace(r0.astype(dt_check), p0.astype(dt_check), f0.astype(dt_check))
            try:
                lowered_pre = traced_pre.lower(lowering_platforms=("cuda",))
                assert "pallas_baoab_pre_force" in lowered_pre.as_text()
                lowered_post = traced_post.lower(lowering_platforms=("cuda",))
                assert "pallas_baoab_post_force" in lowered_post.as_text()
            except RuntimeError as exc:
                if "No supported GPU devices found" not in str(exc):
                    raise

        # --- MK-47: Per-replica independent neighbor-list rebuild masking ---
        N_nl, K_nl = 64, 32
        pos_nl_b = jnp.asarray(rng.uniform(1.0, 23.0, size=(3, N_nl, 3)).astype(np.float32))
        box_nl = jnp.array([25.0, 25.0, 25.0], dtype=jnp.float32)
        mask_nl = jnp.ones((3, N_nl), dtype=jnp.bool_)
        ei_nl = jnp.full((3, N_nl, 4), -1, dtype=jnp.int32)
        ev_nl = jnp.ones((3, N_nl, 4), dtype=jnp.float32)
        ee_nl = jnp.ones((3, N_nl, 4), dtype=jnp.float32)
        sentinel_old = jnp.full((3, N_nl, K_nl), 777, dtype=jnp.int32)
        rebuild_mask = jnp.array([False, True, False], dtype=jnp.bool_)

        idx_masked, _ = _pallas_prune_and_encode_cell_list_batch(
            3,
            (True, False, True, True, True),
            pos_nl_b,
            box_nl,
            ei_nl,
            ev_nl,
            ee_nl,
            cell_size=10.0,
            cell_capacity=32,
            max_occupancy=K_nl,
            r_cutoff=10.0,
            encode_exclusions=True,
            interpret=True,
            atom_mask=mask_nl,
            mask_batched=True,
            rebuild_mask=rebuild_mask,
            old_idx=sentinel_old,
        )
        idx_single_1, _ = pallas_prune_and_encode_cell_list(
            pos_nl_b[1],
            box_nl,
            ei_nl[1],
            ev_nl[1],
            ee_nl[1],
            cell_size=10.0,
            cell_capacity=32,
            max_occupancy=K_nl,
            r_cutoff=10.0,
            encode_exclusions=True,
            interpret=True,
            atom_mask=mask_nl[1],
        )
        # Replicas 0 and 2 had rebuild_mask=False -> must preserve sentinel_old (777) untouched!
        assert bool(jnp.all(idx_masked[0] == 777))
        assert bool(jnp.all(idx_masked[2] == 777))
        # Replica 1 had rebuild_mask=True -> must match standalone rebuild bit-for-bit!
        np.testing.assert_array_equal(np.asarray(idx_masked[1]), np.asarray(idx_single_1))

        # --- MK-49: Separable YZ-plane factorized & charge-gated SPME spread/gather ---
        N_pme = 64
        pos_pme = jnp.asarray(rng.uniform(1.0, 19.0, size=(N_pme, 3)).astype(np.float32))
        q_pme = jnp.asarray(rng.uniform(-0.5, 0.5, size=(N_pme,)).astype(np.float32))
        # Zero out some charges and mark trailing atoms as ghosts to test active_q gating
        q_pme = q_pme.at[5:12].set(0.0)
        mask_pme = jnp.arange(N_pme) < 52
        box_pme = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)
        grid_pme = (16, 16, 16)

        mesh_pal = spread_charges_pallas(
            pos_pme, q_pme, mask_pme, box_pme, grid_pme, order=4, interpret=True
        )
        mesh_ref = _spread_charges_reference(
            pos_pme, q_pme, mask_pme, box_pme, grid_dims=grid_pme, order=4
        )
        np.testing.assert_allclose(np.asarray(mesh_pal), np.asarray(mesh_ref), rtol=1e-5, atol=1e-5)

        theta_norm = jnp.asarray(rng.normal(0.0, 1.0, size=grid_pme).astype(np.float32))
        f_pal_dq, dq_pal = gather_pallas(
            pos_pme,
            q_pme,
            mask_pme,
            box_pme,
            theta_norm,
            grid_pme,
            order=4,
            interpret=True,
            compute_dq=True,
        )
        f_pal_nodq, dq_nodq = gather_pallas(
            pos_pme,
            q_pme,
            mask_pme,
            box_pme,
            theta_norm,
            grid_pme,
            order=4,
            interpret=True,
            compute_dq=False,
        )
        f_ref, dq_ref = _gather_reference(
            pos_pme, q_pme, mask_pme, box_pme, theta_norm, grid_dims=grid_pme, order=4
        )
        np.testing.assert_allclose(np.asarray(f_pal_dq), np.asarray(f_ref), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(dq_pal), np.asarray(dq_ref), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(f_pal_dq), np.asarray(f_pal_nodq), rtol=1e-6, atol=1e-6)
        assert bool(jnp.all(dq_nodq == 0.0))
        assert bool(jnp.all(f_pal_dq[52:] == 0.0))
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_pallas_v31_mk50_mk53_pillars():
    """Verify Pallas v3.1 (MK-50..MK-53): fast init_fn, tile-pair Newton 3rd law + dynamic K-tile, O(N) cell bin, and spatial SPME."""
    from jax_md import space

    from prolix.pallas_kernels import (
        _compute_max_k_tiles,
        build_tile_pair_neighbor_list,
        nonbonded_nl_reference,
        pallas_nonbonded_tile_pairs,
    )
    from prolix.physics.neighbor_list import _build_cell_stencil_buffers_batched
    from prolix.physics.pme import _gather_reference, _spread_charges_reference
    from prolix.physics.pme_pallas import (
        compute_pme_spatial_perm,
        gather_pallas,
        spread_charges_pallas,
    )
    from prolix.physics.settle import settle_langevin

    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(20261106)

        # --- MK-50: settle_langevin precomputed_force & vectorized multi-water init_fn ---
        N_sys = 36
        # 2 rigid waters (atoms 0..5) + 30 solute atoms (atoms 6..35)
        water_idx = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        box_sys = jnp.array([25.0, 25.0, 25.0], dtype=jnp.float64)
        disp_fn, shift_fn = space.periodic(box_sys)
        pos0 = jnp.asarray(rng.uniform(2.0, 22.0, size=(N_sys, 3)).astype(np.float64))
        # Place water H atoms near O
        pos0 = pos0.at[1].set(pos0[0] + jnp.array([0.9572, 0.0, 0.0]))
        pos0 = pos0.at[2].set(pos0[0] + jnp.array([-0.24, 0.926, 0.0]))
        pos0 = pos0.at[4].set(pos0[3] + jnp.array([0.9572, 0.0, 0.0]))
        pos0 = pos0.at[5].set(pos0[3] + jnp.array([-0.24, 0.926, 0.0]))
        mass_sys = jnp.ones((N_sys,), dtype=jnp.float64) * 12.0
        mass_sys = mass_sys.at[0].set(15.999).at[1].set(1.008).at[2].set(1.008)
        mass_sys = mass_sys.at[3].set(15.999).at[4].set(1.008).at[5].set(1.008)
        atom_mask_sys = jnp.arange(N_sys) < 32

        call_counter = {"n": 0}

        def _energy_fn(r, **kwargs):
            call_counter["n"] += 1
            return 0.5 * jnp.sum((r - pos0) ** 2)

        init_fn, _ = settle_langevin(
            _energy_fn,
            shift_fn,
            dt=0.002,
            kT=0.6,
            gamma=1.0,
            mass=mass_sys,
            water_indices=water_idx,
            box=box_sys,
            atom_mask=atom_mask_sys,
        )
        state_auto = init_fn(jax.random.PRNGKey(7), pos0)
        state_pre = init_fn(jax.random.PRNGKey(7), pos0, precomputed_force=state_auto.force)
        np.testing.assert_allclose(
            np.asarray(state_auto.momentum), np.asarray(state_pre.momentum), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(state_auto.force), np.asarray(state_pre.force), rtol=1e-12, atol=1e-12
        )
        # Ghost atoms (32..35) must have strictly zero momentum
        assert bool(jnp.all(state_pre.momentum[32:] == 0.0))

        # --- MK-51: Dynamic _compute_max_k_tiles & 32x32 Tile-Pair Newton 3rd Law Nonbonded ---
        N_nb, K_nb = 64, 64
        pos_nb = jnp.asarray(rng.uniform(1.0, 19.0, size=(N_nb, 3)).astype(np.float32))
        q_nb = jnp.asarray(rng.uniform(-0.4, 0.4, size=(N_nb,)).astype(np.float32))
        sig_nb = jnp.asarray(rng.uniform(2.8, 3.4, size=(N_nb,)).astype(np.float32))
        eps_nb = jnp.asarray(rng.uniform(0.05, 0.15, size=(N_nb,)).astype(np.float32))
        mask_nb = (jnp.arange(N_nb) < 56).astype(jnp.int32)
        box_nb = jnp.array([20.0, 20.0, 20.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        lam_nb = jnp.array([1.0], dtype=jnp.float32)

        # Build dense neighbor list with exclusions (0-1 full exclusion, 2-3 1-4 scaled)
        ei_nb = jnp.full((N_nb, 4), -1, dtype=jnp.int32)
        ev_nb = jnp.ones((N_nb, 4), dtype=jnp.float32)
        ee_nb = jnp.ones((N_nb, 4), dtype=jnp.float32)
        ei_nb = ei_nb.at[0, 0].set(1).at[1, 0].set(0)
        ev_nb = ev_nb.at[0, 0].set(0.0).at[1, 0].set(0.0)
        ee_nb = ee_nb.at[0, 0].set(0.0).at[1, 0].set(0.0)
        ei_nb = ei_nb.at[2, 0].set(3).at[3, 0].set(2)
        ev_nb = ev_nb.at[2, 0].set(0.5).at[3, 0].set(0.5)
        ee_nb = ee_nb.at[2, 0].set(0.83333333).at[3, 0].set(0.83333333)
        # Ensure pair (0,1) and (2,3) are within cutoff
        pos_nb = pos_nb.at[1].set(pos_nb[0] + jnp.array([1.8, 0.1, -0.1], dtype=jnp.float32))
        pos_nb = pos_nb.at[3].set(pos_nb[2] + jnp.array([2.2, -0.2, 0.1], dtype=jnp.float32))

        # Construct padded neighbor list where only first 12 slots are populated (1 K-tile out of 2)
        nbr_rows = []
        for i in range(N_nb):
            if i >= 56:
                row = np.full((K_nb,), N_nb, dtype=np.int32)
            else:
                others = [j for j in range(56) if j != i]
                row = np.full((K_nb,), N_nb, dtype=np.int32)
                row[: len(others)] = others
            nbr_rows.append(row)
        nbr_nb = jnp.asarray(np.stack(nbr_rows, axis=0), dtype=jnp.int32)

        # Test _compute_max_k_tiles on sparse neighbor list (only first 10 slots active -> max_k_tiles == 1)
        nbr_sparse = jnp.full((1, N_nb, 64), N_nb, dtype=jnp.int32).at[:, :32, :10].set(2)
        mkt = _compute_max_k_tiles(nbr_sparse, N_nb, 64, 2, 2, False).reshape(1, 2)
        assert int(mkt[0, 0]) == 1
        assert int(mkt[0, 1]) == 0

        # Build tile-pair neighbor list and verify energy, forces, vmap, and custom_vjp gradients vs reference
        tp_nl = build_tile_pair_neighbor_list(
            nbr_nb,
            excl_indices=ei_nb,
            excl_scales_vdw=ev_nb,
            excl_scales_elec=ee_nb,
            atom_mask=mask_nb,
        )
        e_tp, f_tp = pallas_nonbonded_tile_pairs(
            pos_nb,
            q_nb,
            sig_nb,
            eps_nb,
            tp_nl,
            atom_mask=mask_nb,
            box=box_nb,
            pme_alpha=0.34,
            cutoff=8.0,
            coulomb_constant=332.0637,
            lj_switch_width=0.0,
            soft_core_lambda=lam_nb,
            softcore_alpha=0.5,
            softcore_exponent=2,
            include_pme_excl_erf=True,
            scale_14_vdw=0.5,
            scale_14_elec=0.83333333,
            return_forces=True,
            interpret=True,
        )

        def _loss_ref(p, q, s, ep):
            e_val = nonbonded_nl_reference(
                p,
                q,
                s,
                ep,
                mask_nb,
                nbr_nb,
                ei_nb,
                ev_nb,
                ee_nb,
                box_nb,
                lam_nb,
                has_box=True,
                pme_alpha=0.34,
                cutoff=8.0,
                coulomb_constant=332.0637,
                lj_switch_width=0.0,
                softcore_alpha=0.5,
                softcore_exponent=2,
                include_pme_excl_erf=True,
                scale_14_vdw=0.5,
                scale_14_elec=0.83333333,
            )
            return jnp.sum(e_val)

        e_ref, g_pos_ref = jax.value_and_grad(_loss_ref)(pos_nb, q_nb, sig_nb, eps_nb)
        f_ref = -g_pos_ref
        np.testing.assert_allclose(np.asarray(e_tp), np.asarray(e_ref), rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(np.asarray(f_tp), np.asarray(f_ref), rtol=1e-5, atol=1e-4)

        # Verify custom_vjp gradients w.r.t. (positions, charges, sigmas, epsilons)
        def _loss_tp(p, q, s, ep):
            e_val = pallas_nonbonded_tile_pairs(
                p,
                q,
                s,
                ep,
                tp_nl,
                atom_mask=mask_nb,
                box=box_nb,
                pme_alpha=0.34,
                cutoff=8.0,
                coulomb_constant=332.0637,
                lj_switch_width=0.0,
                soft_core_lambda=lam_nb,
                softcore_alpha=0.5,
                softcore_exponent=2,
                include_pme_excl_erf=True,
                scale_14_vdw=0.5,
                scale_14_elec=0.83333333,
                return_forces=False,
                interpret=True,
            )
            return jnp.sum(e_val)

        g_tp = jax.grad(_loss_tp, argnums=(0, 1, 2, 3))(pos_nb, q_nb, sig_nb, eps_nb)
        g_ref = jax.grad(_loss_ref, argnums=(0, 1, 2, 3))(pos_nb, q_nb, sig_nb, eps_nb)
        for idx_g, name_g in enumerate(["pos", "charges", "sigmas", "epsilons"]):
            np.testing.assert_allclose(
                np.asarray(g_tp[idx_g]),
                np.asarray(g_ref[idx_g]),
                rtol=2e-4,
                atol=2e-4,
                err_msg=f"Tile-pair VJP mismatch on {name_g}",
            )

        # --- MK-52: _build_cell_stencil_buffers_batched with rebuild_mask_b ---
        pos_bin = jnp.broadcast_to(pos_nb[None, :, :], (2, N_nb, 3))
        box_bin = jnp.broadcast_to(jnp.array([[20.0, 20.0, 20.0]], dtype=jnp.float32), (2, 3))
        mask_bin = jnp.broadcast_to((mask_nb > 0)[None, :], (2, N_nb))
        reb_bin = jnp.array([False, True], dtype=jnp.bool_)
        cid_b, catom_b, cx_b, cy_b, cz_b, ccnt_b, g_cells = _build_cell_stencil_buffers_batched(
            pos_bin,
            box_bin,
            cell_size=5.0,
            cell_capacity=32,
            grid_dims_static=(4, 4, 4),
            atom_mask_b=mask_bin,
            rebuild_mask_b=reb_bin,
            interpret=True,
        )
        # Replica 0 had rebuild_mask=False -> all cell counts must be 0 and all cell_id == g_cells
        assert bool(jnp.all(cid_b[0] == g_cells))
        assert bool(jnp.all(ccnt_b[0] == 0))
        # Replica 1 had rebuild_mask=True -> total binned count equals 56 active atoms
        assert int(jnp.sum(ccnt_b[1, :g_cells])) == 56

        # --- MK-53: compute_pme_spatial_perm + batched SPME with heterogeneous charges/masks ---
        grid_pme = (16, 16, 16)
        perm_pme, inv_perm_pme = compute_pme_spatial_perm(
            pos_nb, box_nb[:3], grid_pme, atom_mask=(mask_nb > 0)
        )
        # Ghost atoms (56..63) must be sorted to the very end of perm_pme
        assert set(np.asarray(perm_pme[56:]).tolist()) == set(range(56, 64))

        mesh_perm = spread_charges_pallas(
            pos_nb,
            q_nb,
            mask_nb > 0,
            box_nb[:3],
            grid_pme,
            order=4,
            atom_perm=perm_pme,
            interpret=True,
        )
        mesh_ref = _spread_charges_reference(
            pos_nb, q_nb, mask_nb > 0, box_nb[:3], grid_dims=grid_pme, order=4
        )
        np.testing.assert_allclose(np.asarray(mesh_perm), np.asarray(mesh_ref), rtol=1e-5, atol=1e-5)

        theta_pme = jnp.asarray(rng.normal(0.0, 1.0, size=grid_pme).astype(np.float32))
        f_perm, dq_perm = gather_pallas(
            pos_nb,
            q_nb,
            mask_nb > 0,
            box_nb[:3],
            theta_pme,
            grid_pme,
            order=4,
            atom_perm=perm_pme,
            atom_inv_perm=inv_perm_pme,
            interpret=True,
            compute_dq=False,
        )
        f_ref_pme, _ = _gather_reference(
            pos_nb, q_nb, mask_nb > 0, box_nb[:3], theta_pme, grid_dims=grid_pme, order=4
        )
        np.testing.assert_allclose(np.asarray(f_perm), np.asarray(f_ref_pme), rtol=1e-5, atol=1e-5)
        assert bool(jnp.all(dq_perm == 0.0))
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_pallas_v32_mk54_to_mk57_optimizations_and_parity():
    """Verify Pallas v3.2 pillars MK-54, MK-55, MK-56, and MK-57."""
    from prolix.api.ensemble_dispatch import dispatch_n_steps_inference
    from prolix.pallas_kernels import (
        _infer_first_pure_water_block,
        build_tile_pair_neighbor_list,
        pallas_bonded_and_nonbonded_nl_fused,
        pallas_nonbonded_nl,
        pallas_nonbonded_tile_pairs,
    )
    from prolix.physics.neighbor_list import (
        _max_k_tiles_from_idx,
        _pallas_prune_and_encode_cell_list_batch,
        precompute_static_nl_exclusion_tables,
    )
    from prolix.physics.pme import make_spme_energy_fn
    from prolix.physics.settle import (
        _philox2x32_box_muller_pair,
        pallas_baoab_post_force,
        pallas_baoab_pre_force,
    )

    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        # 1. MK-54: dispatch_n_steps_inference with _two_level_runner & scalar while_loop(vmap)
        init_carry = (jnp.array([1.0, 2.0], dtype=jnp.float32), jnp.array([10, 20], dtype=jnp.int32))
        def _dummy_step(c):
            return (c[0] + jnp.float32(1.0), c[1] + jnp.int32(2))
        _dummy_step._two_level_runner = lambda c, n, _it: jax.lax.fori_loop(
            0, n, lambda _, x: _dummy_step(x), c
        )
        fin_carry = dispatch_n_steps_inference(_dummy_step, init_carry, 5)
        np.testing.assert_allclose(np.asarray(fin_carry[0]), np.array([6.0, 7.0], dtype=np.float32))
        np.testing.assert_allclose(np.asarray(fin_carry[1]), np.array([20, 30], dtype=np.int32))

        # 2. MK-55: In-register Philox-4x32-10 + Box-Muller & non-multiple-of-64 BAOAB (N=50)
        offs = jnp.arange(256, dtype=jnp.int32)
        z0, z1 = _philox2x32_box_muller_pair(
            jnp.int32(123), jnp.int32(456), offs, offs + jnp.int32(0x10000001), jnp.float32
        )
        assert z0.shape == (256,) and z1.shape == (256,)
        assert bool(jnp.all(jnp.isfinite(z0)) and jnp.all(jnp.isfinite(z1)))
        assert 0.5 < float(jnp.std( jnp.concatenate([z0, z1]) )) < 1.5

        rng = np.random.default_rng(20261108)
        N_50 = 50  # N % 64 != 0 and N % 32 != 0
        _g = np.stack(np.meshgrid(np.linspace(2.5, 17.5, 4), np.linspace(2.5, 17.5, 4), np.linspace(2.5, 17.5, 4)), axis=-1).reshape(-1, 3)[:N_50]
        pos_50 = jnp.asarray((_g + rng.uniform(-0.3, 0.3, size=(N_50, 3))).astype(np.float32))
        pos_50 = pos_50.at[1].set(pos_50[0] + jnp.array([1.8, 0.1, -0.1], dtype=jnp.float32))
        pos_50 = pos_50.at[3].set(pos_50[2] + jnp.array([2.2, -0.2, 0.1], dtype=jnp.float32))
        mom_50 = jnp.asarray(rng.normal(0.0, 0.5, size=(N_50, 3)).astype(np.float32))
        f_50 = jnp.asarray(rng.normal(0.0, 0.2, size=(N_50, 3)).astype(np.float32))
        mass_50 = jnp.full((N_50, 1), 12.0, dtype=jnp.float32)
        am_50 = jnp.arange(N_50) < 44
        wi_50 = jnp.array([[38, 39, 40], [41, 42, 43]], dtype=jnp.int32)
        wm_50 = jnp.array([True, True], dtype=jnp.bool_)
        box_50 = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)
        r_pre, p_pre, k_pre = pallas_baoab_pre_force(
            pos_50,
            mom_50,
            f_50,
            mass_50,
            jax.random.PRNGKey(7),
            wi_50,
            wm_50,
            box_50,
            jnp.float32(0.04),
            jnp.float32(0.6),
            0.5,
            atom_mask=am_50,
            interpret=True,
        )
        p_post = pallas_baoab_post_force(
            r_pre, p_pre, f_50, mass_50, wi_50, wm_50, jnp.float32(0.04), atom_mask=am_50, interpret=True
        )
        assert r_pre.shape == (N_50, 3) and p_post.shape == (N_50, 3)
        assert bool(jnp.all(jnp.isfinite(r_pre)) and jnp.all(jnp.isfinite(p_post)))
        np.testing.assert_allclose(np.asarray(p_post[44:]), np.asarray(mom_50[44:]), atol=1e-7)

        # 3. MK-56: precompute_static_nl_exclusion_tables, return_max_k_tiles, _infer_first_pure_water_block, N=50 megakernel
        ei_50 = jnp.full((N_50, 4), -1, dtype=jnp.int32)
        ev_50 = jnp.ones((N_50, 4), dtype=jnp.float32)
        ee_50 = jnp.ones((N_50, 4), dtype=jnp.float32)
        ei_50 = ei_50.at[0, 0].set(1).at[1, 0].set(0).at[2, 0].set(3).at[3, 0].set(2)
        ev_50 = ev_50.at[0, 0].set(0.0).at[1, 0].set(0.0).at[2, 0].set(0.5).at[3, 0].set(0.5)
        ee_50 = ee_50.at[0, 0].set(0.0).at[1, 0].set(0.0).at[2, 0].set(0.83333333).at[3, 0].set(0.83333333)
        pre_excl = precompute_static_nl_exclusion_tables(ei_50, ev_50, ee_50, n_atoms=N_50)
        assert len(pre_excl) == 4
        idx_p, occ_p, mkt_p = _pallas_prune_and_encode_cell_list_batch(
            1,
            (False, False, False, False, False),
            pos_50,
            box_50,
            ei_50,
            ev_50,
            ee_50,
            cell_size=5.0,
            cell_capacity=32,
            max_occupancy=64,
            r_cutoff=4.8,
            encode_exclusions=True,
            interpret=True,
            grid_dims=(4, 4, 4),
            atom_mask=am_50,
            precomputed_excl=pre_excl,
            return_max_k_tiles=True,
        )
        assert mkt_p.shape == (1, 2)
        mkt_ref = _max_k_tiles_from_idx(idx_p[0])
        assert mkt_ref.shape == (2,)
        assert bool(jnp.all(mkt_p[0] >= mkt_ref))

        bonds_50 = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        b_params_50 = jnp.array([[1.5, 300.0], [1.5, 300.0]], dtype=jnp.float32)
        b_mask_50 = jnp.array([True, True], dtype=jnp.bool_)
        pw_blk = _infer_first_pure_water_block(N_50, ((bonds_50, b_mask_50, 2),))
        assert pw_blk == 1  # max bonded atom is 3 -> block 0 has bonded atoms, block 1 is pure water

        q_50 = jnp.asarray(rng.uniform(-0.4, 0.4, size=(N_50,)).astype(np.float32))
        sig_50 = jnp.full((N_50,), 3.0, dtype=jnp.float32)
        eps_50 = jnp.full((N_50,), 0.1, dtype=jnp.float32)
        e_mk56, f_mk56 = pallas_bonded_and_nonbonded_nl_fused(
            pos_50,
            q_50,
            sig_50,
            eps_50,
            am_50,
            idx_p[0],
            ei_50,
            ev_50,
            ee_50,
            bonds=bonds_50,
            bond_params=b_params_50,
            bond_mask=b_mask_50,
            box=box_50,
            pme_alpha=0.34,
            cutoff=4.8,
            include_pme_excl_erf=True,
            excl_mode="pre_encoded",
            max_k_tiles=mkt_p[0],
            first_pure_water_block=pw_blk,
            return_forces=True,
            interpret=True,
        )
        e_mk56_ref, f_mk56_ref = pallas_bonded_and_nonbonded_nl_fused(
            pos_50,
            q_50,
            sig_50,
            eps_50,
            am_50,
            idx_p[0],
            ei_50,
            ev_50,
            ee_50,
            bonds=bonds_50,
            bond_params=b_params_50,
            bond_mask=b_mask_50,
            box=box_50,
            pme_alpha=0.34,
            cutoff=4.8,
            include_pme_excl_erf=True,
            excl_mode="pre_encoded",
            max_k_tiles=None,
            first_pure_water_block=2,
            return_forces=True,
            interpret=True,
        )
        np.testing.assert_allclose(np.asarray(e_mk56), np.asarray(e_mk56_ref), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(f_mk56), np.asarray(f_mk56_ref), rtol=1e-5, atol=1e-5)
        e_mk56_no_e, f_mk56_no_e = pallas_bonded_and_nonbonded_nl_fused(
            pos_50,
            q_50,
            sig_50,
            eps_50,
            am_50,
            idx_p[0],
            ei_50,
            ev_50,
            ee_50,
            bonds=bonds_50,
            bond_params=b_params_50,
            bond_mask=b_mask_50,
            box=box_50,
            pme_alpha=0.34,
            cutoff=4.8,
            include_pme_excl_erf=True,
            excl_mode="pre_encoded",
            max_k_tiles=mkt_p[0],
            first_pure_water_block=pw_blk,
            return_forces=True,
            compute_energy=False,
            interpret=True,
        )
        assert float(e_mk56_no_e) == 0.0
        np.testing.assert_allclose(np.asarray(f_mk56_no_e), np.asarray(f_mk56), rtol=1e-6, atol=1e-6)

        # 4. MK-57 (a): Direct force-only SPME & energy_fn_from_bundle(return_forces=True) parity
        spme_e_fn = make_spme_energy_fn(box_50, alpha=0.34, grid_dims=(16, 16, 16), use_pallas_pme=True, return_forces=False)
        spme_f_fn = make_spme_energy_fn(box_50, alpha=0.34, grid_dims=(16, 16, 16), use_pallas_pme=True, return_forces=True)
        f_spme_direct = spme_f_fn(pos_50, q_50, am_50)
        f_spme_grad = -jax.grad(lambda r: spme_e_fn(r, q_50, am_50))(pos_50)
        np.testing.assert_allclose(np.asarray(f_spme_direct), np.asarray(f_spme_grad), rtol=1e-5, atol=1e-5)

        bundle = _make_periodic_solvated_bundle(dtype=jnp.float32, seed=77)
        disp_fn, _ = displacement_fn_for_bundle(bundle)
        nl_fn = nl_mod.make_neighbor_list_fn(disp_fn, jnp.diag(bundle.box), 9.0, capacity_multiplier=1.5, dr_threshold=0.5)
        nbr_enc = _encode_bundle_nl(nl_fn.allocate(bundle.positions), bundle)
        e_fn_bun = energy_fn_from_bundle(
            bundle, build_dense_exclusions=False, use_pallas=True, use_pallas_pme=True, excl_mode="pre_encoded", return_forces=False
        )
        f_fn_bun = energy_fn_from_bundle(
            bundle, build_dense_exclusions=False, use_pallas=True, use_pallas_pme=True, excl_mode="pre_encoded", return_forces=True
        )
        f_bun_direct = f_fn_bun(bundle.positions, neighbor=nbr_enc)
        f_bun_grad = -jax.grad(lambda r: e_fn_bun(r, neighbor=nbr_enc))(bundle.positions)
        np.testing.assert_allclose(np.asarray(f_bun_direct), np.asarray(f_bun_grad), rtol=1e-5, atol=1e-5)
        f_bun_jaxpr = jax.make_jaxpr(lambda r: f_fn_bun(r, neighbor=nbr_enc))(bundle.positions)
        assert sum(1 for eq in f_bun_jaxpr.jaxpr.eqns if eq.primitive.name == "reduce_sum") == 0

        # 5. MK-57 (b): Morton-sorted 32x32 Tile-Pair neighbor list & pallas_nonbonded_tile_pairs parity
        tp_unsorted = build_tile_pair_neighbor_list(
            idx_p[0], excl_indices=ei_50, excl_scales_vdw=ev_50, excl_scales_elec=ee_50, atom_mask=am_50, morton_sort=False
        )
        tp_morton = build_tile_pair_neighbor_list(
            idx_p[0],
            excl_indices=ei_50,
            excl_scales_vdw=ev_50,
            excl_scales_elec=ee_50,
            atom_mask=am_50,
            positions=pos_50,
            box=box_50,
            morton_sort=True,
        )
        assert tp_morton.atom_perm is not None and tp_morton.atom_inv_perm is not None
        e_tp_u, f_tp_u = pallas_nonbonded_tile_pairs(
            pos_50, q_50, sig_50, eps_50, tp_unsorted, atom_mask=am_50, box=box_50, pme_alpha=0.34, cutoff=4.8, include_pme_excl_erf=True, return_forces=True, interpret=True
        )
        e_tp_m, f_tp_m = pallas_nonbonded_tile_pairs(
            pos_50, q_50, sig_50, eps_50, tp_morton, atom_mask=am_50, box=box_50, pme_alpha=0.34, cutoff=4.8, include_pme_excl_erf=True, return_forces=True, interpret=True
        )
        np.testing.assert_allclose(np.asarray(e_tp_m), np.asarray(e_tp_u), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(f_tp_m), np.asarray(f_tp_u), rtol=1e-5, atol=1e-5)

        g_tp_m = jax.grad(
            lambda r, q, s, ep: pallas_nonbonded_tile_pairs(
                r, q, s, ep, tp_morton, atom_mask=am_50, box=box_50, pme_alpha=0.34, cutoff=4.8, include_pme_excl_erf=True, interpret=True
            ),
            argnums=(0, 1, 2, 3),
        )(pos_50, q_50, sig_50, eps_50)
        g_nb_ref = jax.grad(
            lambda r, q, s, ep: pallas_nonbonded_nl(
                r, q, s, ep, am_50, idx_p[0], ei_50, ev_50, ee_50, box=box_50, pme_alpha=0.34, cutoff=4.8, include_pme_excl_erf=True, excl_mode="pre_encoded", interpret=True
            ),
            argnums=(0, 1, 2, 3),
        )(pos_50, q_50, sig_50, eps_50)
        for g_m, g_r in zip(g_tp_m, g_nb_ref):
            np.testing.assert_allclose(np.asarray(g_m), np.asarray(g_r), rtol=2e-4, atol=2e-4)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_pallas_v33_mk58_to_mk61_optimizations_and_parity(monkeypatch):
    """Verify Pallas v3.3 pillars MK-58, MK-59, MK-60, and MK-61."""
    from prolix.pallas_kernels import pallas_nonbonded_nl, pallas_nonbonded_tile_pairs
    from prolix.physics.generalized_born import compute_ace_nonpolar_energy, compute_gb_energy
    from prolix.physics.neighbor_list import (
        _pallas_prune_and_encode_cell_list_batch,
        build_tile_pair_neighbor_list_gpu,
        precompute_static_nl_exclusion_tables,
    )
    from prolix.physics.pallas_generalized_born import pallas_gbsa_dense
    from prolix.physics.pme import (
        _resolve_env_pme_grid_dims,
        influence_function,
        make_spme_energy_fn,
        scale_influence_function_isotropic,
        spme_forces,
    )
    from prolix.physics.pme_pallas import gather_pallas, spread_charges_pallas
    from prolix.physics.settle import (
        _build_rattle_clusters,
        pallas_baoab_post_force,
        pallas_baoab_pre_force,
    )

    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        # 1. MK-58: Mixed-precision dual writeout (return_pos_f32=True), f32 force input,
        # reference_position/return_max_dr_sq, and in-register solute X-H RATTLE clusters
        bundle_f64 = _make_periodic_solvated_bundle(dtype=jnp.float64, seed=101)
        N = int(bundle_f64.positions.shape[0])
        box_f64 = jnp.diag(bundle_f64.box).astype(jnp.float64)
        mom_f64 = jnp.zeros_like(bundle_f64.positions)
        frc_f32 = jnp.ones((N, 3), dtype=jnp.float32) * jnp.float32(0.05)
        key0 = jax.random.PRNGKey(55)
        dt_f64 = jnp.float64(0.04)
        kT_f64 = jnp.float64(0.6)

        # Solute X-H constraint pairs on atoms (0, 1) and (0, 2)
        d01 = float(jnp.linalg.norm(bundle_f64.positions[0] - bundle_f64.positions[1]))
        d02 = float(jnp.linalg.norm(bundle_f64.positions[0] - bundle_f64.positions[2]))
        c_pairs = jnp.array([[0, 1], [0, 2], [0, 0]], dtype=jnp.int32)
        c_lens = jnp.array([d01, d02, 1.0], dtype=jnp.float64)
        c_mask = jnp.array([True, True, False], dtype=jnp.bool_)
        cX, cH0, cH1, cH2, cH3, cD0, cD1, cD2, cD3, cM, c_am = _build_rattle_clusters(
            1,
            N,
            int(c_pairs.shape[0]),
            c_pairs[None, ...],
            c_lens[None, ...],
            c_mask[None, ...].astype(jnp.int32),
            bundle_f64.masses[None, ...],
        )
        assert int(jnp.sum(cM > 0)) == 1
        assert bool(c_am[0, 0]) and bool(c_am[0, 1]) and bool(c_am[0, 2])

        pos_new, mom_mid, key_new, pos_f32_out, max_dr_sq = pallas_baoab_pre_force(
            bundle_f64.positions,
            mom_f64,
            frc_f32,
            bundle_f64.masses[:, None],
            key0,
            bundle_f64.water_indices,
            bundle_f64.water_mask,
            box_f64,
            dt_f64,
            kT_f64,
            1.0,
            atom_mask=bundle_f64.atom_mask,
            constraint_pairs=c_pairs,
            constraint_lengths=c_lens,
            constraint_mask=c_mask,
            reference_position=bundle_f64.positions,
            return_max_dr_sq=True,
            return_pos_f32=True,
            interpret=True,
        )
        assert pos_new.dtype == jnp.float64
        assert pos_f32_out.dtype == jnp.float32
        np.testing.assert_allclose(
            np.asarray(pos_f32_out),
            np.asarray(pos_new.astype(jnp.float32)),
            rtol=1e-6,
            atol=1e-6,
        )
        assert float(max_dr_sq) > 0.0
        # Verify RATTLE bond lengths (0, 1) and (0, 2) are satisfied to < 1e-5 A
        d01_after = float(jnp.linalg.norm(pos_new[0] - pos_new[1]))
        d02_after = float(jnp.linalg.norm(pos_new[0] - pos_new[2]))
        assert abs(d01_after - d01) < 1e-5
        assert abs(d02_after - d02) < 1e-5

        mom_fin = pallas_baoab_post_force(
            pos_new,
            mom_mid,
            frc_f32,
            bundle_f64.masses[:, None],
            bundle_f64.water_indices,
            bundle_f64.water_mask,
            dt_f64,
            atom_mask=bundle_f64.atom_mask,
            constraint_pairs=c_pairs,
            constraint_lengths=c_lens,
            constraint_mask=c_mask,
            interpret=True,
        )
        assert mom_fin.dtype == jnp.float64
        # Verify RATTLE velocity orthogonality on (0, 1) and (0, 2)
        v_fin = mom_fin / bundle_f64.masses[:, None]
        r01_vec = pos_new[1] - pos_new[0]
        r02_vec = pos_new[2] - pos_new[0]
        assert abs(float(jnp.dot(v_fin[1] - v_fin[0], r01_vec))) < 1e-4
        assert abs(float(jnp.dot(v_fin[2] - v_fin[0], r02_vec))) < 1e-4

        # 2. MK-59: Non-cubic (Kx, Ky, Kz) SPME grids, isotropic NPT G scaling, & 2-step RESPA PME
        monkeypatch.setenv("PROLIX_PME_GRID_POINTS", "16x16x12")
        assert _resolve_env_pme_grid_dims(64, (16, 16, 16)) == (16, 16, 12)
        monkeypatch.setenv("PROLIX_PME_GRID_POINTS", "64:16x16x12,25000:64x64x48")
        assert _resolve_env_pme_grid_dims(64, (16, 16, 16)) == (16, 16, 12)
        assert _resolve_env_pme_grid_dims(25000, (64, 64, 64)) == (64, 64, 48)
        monkeypatch.delenv("PROLIX_PME_GRID_POINTS", raising=False)

        box_ref = jnp.array([24.0, 24.0, 18.0], dtype=jnp.float32)
        box_cur = box_ref * jnp.float32(1.02)
        G_ref = influence_function((16, 16, 12), box_ref, 0.34, 4).astype(jnp.float32)
        G_cur_iso = scale_influence_function_isotropic(G_ref, box_ref, box_cur)
        pos_f32 = bundle_f64.positions.astype(jnp.float32)
        q_f32 = bundle_f64.charges.astype(jnp.float32)
        f_spme_cached = spme_forces(
            pos_f32, q_f32, bundle_f64.atom_mask, box_cur, (16, 16, 12), 0.34, 4, True, G_cur_iso
        )
        assert f_spme_cached.shape == pos_f32.shape
        assert bool(jnp.all(jnp.isfinite(f_spme_cached)))

        spme_dyn_fn = make_spme_energy_fn(
            box_ref, alpha=0.34, grid_dims=(16, 16, 12), use_pallas_pme=True, return_forces=True
        )
        f_dyn = spme_dyn_fn(pos_f32, q_f32, bundle_f64.atom_mask, box=box_cur)
        np.testing.assert_allclose(np.asarray(f_dyn), np.asarray(f_spme_cached), rtol=1e-5, atol=1e-5)

        # Test 2-step RESPA PME (PROLIX_PME_EVERY=2) and skin check (PROLIX_NL_CHECK_EVERY=10) in EnsemblePlan
        monkeypatch.setenv("PROLIX_PME_EVERY", "2")
        monkeypatch.setenv("PROLIX_NL_CHECK_EVERY", "10")
        plan_respa = EnsemblePlan.from_bundles([bundle_f64], use_pallas=True, use_pallas_pme=True)
        traj_respa = plan_respa.run(
            n_steps=6, dt=1.0, kT=0.6, seed=19, run_mode="inference",
            use_neighbor_list=True, nl_update_every=0, use_pallas=True, use_pallas_pme=True,
        )
        if isinstance(traj_respa, list):
            traj_respa = traj_respa[0]
        assert bool(jnp.all(jnp.isfinite(traj_respa.positions)))
        monkeypatch.delenv("PROLIX_PME_EVERY", raising=False)
        monkeypatch.delenv("PROLIX_NL_CHECK_EVERY", raising=False)

        # 3. MK-60: Native GPU Tile-Pair Builder (build_tile_pair_neighbor_list_gpu) parity
        jax.config.update("jax_enable_x64", False)
        rng = np.random.default_rng(330)
        N_tp = 64
        pos_tp = jnp.asarray(rng.uniform(1.0, 15.0, size=(N_tp, 3)).astype(np.float32))
        pos_tp = pos_tp.at[52:].set(0.0)
        q_tp = jnp.asarray(rng.uniform(-0.4, 0.4, size=(N_tp,)).astype(np.float32)).at[52:].set(0.0)
        sig_tp = jnp.asarray(rng.uniform(2.8, 3.4, size=(N_tp,)).astype(np.float32))
        eps_tp = jnp.asarray(rng.uniform(0.05, 0.15, size=(N_tp,)).astype(np.float32)).at[52:].set(0.0)
        am_tp = jnp.arange(N_tp) < 52
        box_tp = jnp.array([18.0, 18.0, 18.0], dtype=jnp.float32)
        ei_tp = jnp.full((N_tp, 8), -1, dtype=jnp.int32).at[0, 0].set(1).at[1, 0].set(0)
        ev_tp = jnp.ones((N_tp, 8), dtype=jnp.float32).at[0, 0].set(0.0).at[1, 0].set(0.0)
        ee_tp = jnp.ones((N_tp, 8), dtype=jnp.float32).at[0, 0].set(0.0).at[1, 0].set(0.0)

        tp_gpu = build_tile_pair_neighbor_list_gpu(
            pos_tp,
            box_tp,
            cutoff=4.8,
            skin=0.5,
            excl_indices=ei_tp,
            excl_scales_vdw=ev_tp,
            excl_scales_elec=ee_tp,
            atom_mask=am_tp,
            max_pairs=32,
        )
        assert int(tp_gpu.num_valid_pairs) > 0
        assert tp_gpu.atom_perm is not None

        pem_tp, pex_tp, pei32_tp, pet32_tp = precompute_static_nl_exclusion_tables(
            ei_tp, ev_tp, ee_tp, N_tp
        )
        idx_tp_b, _, _ = _pallas_prune_and_encode_cell_list_batch(
            1,
            (True, True, False, False, False),
            pos_tp[None, ...],
            box_tp[None, ...],
            ei_tp,
            ev_tp,
            ee_tp,
            cell_size=5.3,
            cell_capacity=64,
            max_occupancy=64,
            r_cutoff=5.3,
            encode_exclusions=True,
            interpret=True,
            grid_dims=(3, 3, 3),
            atom_mask=am_tp,
            precomputed_excl=(pem_tp, pex_tp, pei32_tp, pet32_tp),
            return_max_k_tiles=True,
        )
        e_tp_gpu, f_tp_gpu = pallas_nonbonded_tile_pairs(
            pos_tp, q_tp, sig_tp, eps_tp, tp_gpu, atom_mask=am_tp, box=box_tp,
            pme_alpha=0.34, cutoff=4.8, include_pme_excl_erf=True, return_forces=True, interpret=True,
        )
        e_nl_ref, f_nl_ref = pallas_nonbonded_nl(
            pos_tp, q_tp, sig_tp, eps_tp, am_tp, idx_tp_b[0], ei_tp, ev_tp, ee_tp,
            box=box_tp, pme_alpha=0.34, cutoff=4.8, include_pme_excl_erf=True,
            excl_mode="pre_encoded", return_forces=True, interpret=True,
        )
        np.testing.assert_allclose(np.asarray(e_tp_gpu), np.asarray(e_nl_ref), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(f_tp_gpu), np.asarray(f_nl_ref), rtol=1e-5, atol=1e-5)

        # 4. MK-61 (a): Active-atom CTA grid clamping parity (n_active_atoms=64 on N=96 padded)
        N_pad = 96
        pos_96 = jnp.pad(pos_tp, ((0, 32), (0, 0)))
        q_96 = jnp.pad(q_tp, (0, 32))
        am_96 = jnp.arange(N_pad) < 52
        mesh_full = spread_charges_pallas(
            pos_96, q_96, am_96, box_tp, (16, 16, 16), interpret=True
        )
        mesh_clamp = spread_charges_pallas(
            pos_96, q_96, am_96, box_tp, (16, 16, 16), interpret=True, n_active_atoms=64
        )
        np.testing.assert_allclose(np.asarray(mesh_clamp), np.asarray(mesh_full), rtol=1e-6, atol=1e-6)
        theta_96 = jnp.ones((16, 16, 16), dtype=jnp.float32)
        f_g_full, _ = gather_pallas(
            pos_96, q_96, am_96, box_tp, theta_96, (16, 16, 16), interpret=True, compute_dq=False
        )
        f_g_clamp, _ = gather_pallas(
            pos_96, q_96, am_96, box_tp, theta_96, (16, 16, 16), interpret=True, compute_dq=False, n_active_atoms=64
        )
        np.testing.assert_allclose(np.asarray(f_g_clamp), np.asarray(f_g_full), rtol=1e-6, atol=1e-6)

        # 5. MK-61 (b): Fused 2-pass Pallas GBSA + ACE + dense LJ/Coulomb parity & custom_vjp
        rad_tp = jnp.asarray(rng.uniform(1.2, 1.8, size=(N_tp,)).astype(np.float32))
        srad_tp = rad_tp * jnp.float32(0.85)
        e_gb_p, f_gb_p = pallas_gbsa_dense(
            pos_tp,
            q_tp,
            rad_tp,
            srad_tp,
            am_tp,
            sigmas=sig_tp,
            epsilons=eps_tp,
            excl_indices=ei_tp,
            excl_scales_vdw=ev_tp,
            excl_scales_elec=ee_tp,
            include_ace=True,
            include_lj_coulomb=False,
            return_forces=True,
            interpret=True,
        )
        def _ref_gb_ace_dense(r):
            mask_2d = (am_tp[:, None] & am_tp[None, :]).astype(jnp.float32)
            e_gb, br = compute_gb_energy(
                r, q_tp, rad_tp, scaled_radii=srad_tp, mask=mask_2d, energy_mask=mask_2d
            )
            e_ace = jnp.sum(compute_ace_nonpolar_energy(rad_tp, br) * am_tp.astype(jnp.float32))
            return e_gb + e_ace

        e_tot_ref, g_tot_ref = jax.value_and_grad(_ref_gb_ace_dense)(pos_tp)
        f_tot_ref = -g_tot_ref
        np.testing.assert_allclose(np.asarray(e_gb_p), np.asarray(e_tot_ref), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(f_gb_p), np.asarray(f_tot_ref), rtol=2e-4, atol=2e-4)

        # Vmap B=2 and jax.grad w.r.t. (r, q, radii, scaled_radii, sigmas, epsilons)
        pos_b2 = jnp.stack([pos_tp, pos_tp + jnp.float32(0.05)], axis=0)
        e_b2, f_b2 = jax.vmap(
            lambda r1: pallas_gbsa_dense(
                r1, q_tp, rad_tp, srad_tp, am_tp,
                sigmas=sig_tp, epsilons=eps_tp,
                excl_indices=ei_tp, excl_scales_vdw=ev_tp, excl_scales_elec=ee_tp,
                include_ace=True, include_lj_coulomb=True, return_forces=True, interpret=True,
            )
        )(pos_b2)
        assert e_b2.shape == (2,) and f_b2.shape == (2, N_tp, 3)
        np.testing.assert_allclose(np.asarray(e_b2[0]), np.asarray(e_b2[1]), rtol=1e-5, atol=1e-4)

        grads_gbsa = jax.grad(
            lambda r, q, rd, sr, sg, ep: pallas_gbsa_dense(
                r, q, rd, sr, am_tp,
                sigmas=sg, epsilons=ep,
                excl_indices=ei_tp, excl_scales_vdw=ev_tp, excl_scales_elec=ee_tp,
                include_ace=True, include_lj_coulomb=True, return_forces=False, interpret=True,
            ),
            argnums=(0, 1, 2, 3, 4, 5),
        )(pos_tp, q_tp, rad_tp, srad_tp, sig_tp, eps_tp)
        for g_arr in grads_gbsa:
            assert bool(jnp.all(jnp.isfinite(g_arr)))
            assert float(jnp.linalg.norm(g_arr)) > 0.0
        grads_gb_only = jax.grad(
            lambda r, q, rd, sr: pallas_gbsa_dense(
                r, q, rd, sr, am_tp,
                include_ace=True, include_lj_coulomb=False, return_forces=False, interpret=True,
            ),
            argnums=(0, 1, 2, 3),
        )(pos_tp, q_tp, rad_tp, srad_tp)
        grads_gb_ref = jax.grad(
            lambda r, q, rd, sr: (
                lambda m2d: (
                    lambda eg_br: eg_br[0] + jnp.sum(compute_ace_nonpolar_energy(rd, eg_br[1]) * am_tp.astype(jnp.float32))
                )(compute_gb_energy(r, q, rd, scaled_radii=sr, mask=m2d, energy_mask=m2d))
            )((am_tp[:, None] & am_tp[None, :]).astype(jnp.float32)),
            argnums=(0, 1, 2, 3),
        )(pos_tp, q_tp, rad_tp, srad_tp)
        for g_got, g_exp in zip(grads_gb_only, grads_gb_ref):
            np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_exp), rtol=2e-4, atol=2e-4)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)



def test_pallas_v34_cuda_guard_and_mk62_to_mk64():
    """Verify Phase 0 OpenMM CUDA anti-fallback guard and Pallas v3.4 MK-62, MK-63, MK-64."""
    import importlib.util
    import pytest
    from pathlib import Path
    from prolix.physics.pme import (
        snap_fft_dim,
        compute_openmm_pme_params,
        spme_forces,
        influence_function,
        _spme_forces_cached,
    )
    from prolix.pallas_kernels import (
        build_tile_pair_neighbor_list,
        pallas_nonbonded_tile_pairs,
        nonbonded_nl_reference,
        pallas_bonded_and_nonbonded_nl_fused,
    )

    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        # 1. Phase 0: OpenMM CUDA Anti-Fallback Guard
        guard_path = Path(__file__).resolve().parents[1] / "scripts" / "benchmarks" / "openmm_cuda_guard.py"
        spec = importlib.util.spec_from_file_location("openmm_cuda_guard", guard_path)
        assert spec is not None and spec.loader is not None
        guard_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard_mod)

        inv = guard_mod.get_openmm_platform_inventory()
        assert isinstance(inv, dict)
        assert "platforms" in inv and "has_cuda_platform" in inv

        class _FakePlatform:
            def __init__(self, name: str, props: dict[str, str] | None = None):
                self._name = name
                self._props = props or {}

            def getName(self) -> str:
                return self._name

            def getPropertyNames(self):
                return list(self._props.keys())

            def getPropertyValue(self, _ctx, key: str) -> str:
                return self._props.get(key, "")

        class _FakeContext:
            def __init__(self, plat: _FakePlatform):
                self._plat = plat

            def getPlatform(self) -> _FakePlatform:
                return self._plat

        with pytest.raises(RuntimeError, match="OpenMM Anti-Fallback Sensor tripped"):
            guard_mod.verify_openmm_cuda_context(
                _FakeContext(_FakePlatform("OpenCL")), allow_opencl_fallback=False
            )
        info_opencl = guard_mod.verify_openmm_cuda_context(
            _FakeContext(_FakePlatform("OpenCL")), allow_opencl_fallback=True
        )
        assert info_opencl["platform_name"] == "OpenCL"
        orig_find_lib = guard_mod.find_lib_openmm_cuda_path
        guard_mod.find_lib_openmm_cuda_path = lambda: "/mock/libOpenMMCUDA.so"
        try:
            info_cuda = guard_mod.verify_openmm_cuda_context(
                _FakeContext(_FakePlatform("CUDA", {"CudaPrecision": "single"})),
                allow_opencl_fallback=False,
            )
            assert info_cuda["platform_name"] == "CUDA"
        finally:
            guard_mod.find_lib_openmm_cuda_path = orig_find_lib

        # 2. MK-62 (a & b): snap_fft_dim, compute_openmm_pme_params, & fused gather force_scale
        assert snap_fft_dim(17) == 18
        assert snap_fft_dim(53) == 54
        assert snap_fft_dim(61) == 64
        for n_test in range(4, 130):
            snapped = snap_fft_dim(n_test)
            assert snapped >= max(8, n_test)
            assert snapped % 2 == 0
            rem = snapped
            for p_fac in (2, 3, 5):
                while rem % p_fac == 0:
                    rem //= p_fac
            assert rem == 1

        alpha_dhfr, grid_dhfr = compute_openmm_pme_params(
            (62.23, 62.23, 62.23), cutoff=8.0, ewald_error_tolerance=5e-4
        )
        assert 0.30 < alpha_dhfr < 0.45
        assert all(g in (54, 60, 64) for g in grid_dhfr)

        rng = np.random.default_rng(20261110)
        N_pme = 48
        box_pme = jnp.array([24.0, 24.0, 24.0], dtype=jnp.float32)
        pos_pme = jnp.asarray(rng.uniform(1.0, 23.0, size=(N_pme, 3)).astype(np.float32))
        q_pme = jnp.asarray(rng.uniform(-0.5, 0.5, size=(N_pme,)).astype(np.float32))
        q_pme = q_pme - jnp.mean(q_pme)
        am_pme = jnp.arange(N_pme) < 40

        f_pme_ref = spme_forces(
            pos_pme, q_pme, am_pme, box_pme, (18, 18, 18), 0.34,
            use_pallas_pme=False,
        )
        f_pme_pal = spme_forces(
            pos_pme, q_pme, am_pme, box_pme, (18, 18, 18), 0.34,
            use_pallas_pme=True,
        )
        np.testing.assert_allclose(
            np.asarray(f_pme_pal), np.asarray(f_pme_ref), rtol=1e-4, atol=1e-4
        )
        # Cached SPME forces path with fused force_scale
        G_cached = influence_function((18, 18, 18), box_pme, 0.34, 4).astype(jnp.float32)
        f_pme_cached = _spme_forces_cached(
            pos_pme, q_pme, am_pme, G_cached, box_pme, (18, 18, 18), 4, True,
        )
        np.testing.assert_allclose(
            np.asarray(f_pme_cached), np.asarray(f_pme_ref), rtol=1e-4, atol=1e-4
        )

        # 3. MK-62 (c) & MK-63: 32-atom tile AABB cutoff pruning + has_any_active cond + adaptive tp_group/nb_k_splits
        N_tp = 96  # 3 tiles of 32 atoms
        # Place tile 0 on a clash-free lattice in x in [1.5, 9.9], tile 1 in x in [3.2, 11.6], and tile 2 far away at x in [24, 32.4]
        grid_32 = np.stack(np.meshgrid([0.0, 2.8, 5.6, 8.4], [0.0, 2.8, 5.6, 8.4], [0.0, 2.8], indexing="ij"), axis=-1).reshape(32, 3).astype(np.float32)
        pos_t0 = grid_32 + np.array([1.5, 1.5, 1.5], dtype=np.float32) + rng.uniform(-0.08, 0.08, size=(32, 3)).astype(np.float32)
        pos_t1 = grid_32 + np.array([3.2, 3.2, 3.2], dtype=np.float32) + rng.uniform(-0.08, 0.08, size=(32, 3)).astype(np.float32)
        pos_t2 = grid_32 + np.array([24.0, 24.0, 24.0], dtype=np.float32) + rng.uniform(-0.08, 0.08, size=(32, 3)).astype(np.float32)
        pos_3t = jnp.asarray(np.concatenate([pos_t0, pos_t1, pos_t2], axis=0))
        q_3t = jnp.asarray(rng.uniform(-0.3, 0.3, size=(N_tp,)).astype(np.float32))
        sig_3t = jnp.asarray(rng.uniform(2.9, 3.3, size=(N_tp,)).astype(np.float32))
        eps_3t = jnp.asarray(rng.uniform(0.05, 0.15, size=(N_tp,)).astype(np.float32))
        am_3t = jnp.arange(N_tp) < 90
        box_3t = jnp.array([45.0, 45.0, 45.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        lam_3t = jnp.array([1.0], dtype=jnp.float32)
        ei_3t = jnp.full((N_tp, 4), -1, dtype=jnp.int32)
        ev_3t = jnp.ones((N_tp, 4), dtype=jnp.float32)
        ee_3t = jnp.ones((N_tp, 4), dtype=jnp.float32)

        # Build a neighbor list that includes a false-positive neighbor between tile 0 (atom 0) and tile 2 (atom 64, >12 A away)
        K_3t = 96
        nbr_rows_3t = []
        for i in range(N_tp):
            row = np.full((K_3t,), N_tp, dtype=np.int32)
            others = [j for j in range(90) if j != i]
            row[:len(others)] = others
            nbr_rows_3t.append(row)
        nbr_3t = jnp.asarray(np.stack(nbr_rows_3t, axis=0), dtype=jnp.int32)

        tp_unpruned = build_tile_pair_neighbor_list(
            nbr_3t, excl_indices=ei_3t, excl_scales_vdw=ev_3t, excl_scales_elec=ee_3t, atom_mask=am_3t,
        )
        tp_pruned = build_tile_pair_neighbor_list(
            nbr_3t, excl_indices=ei_3t, excl_scales_vdw=ev_3t, excl_scales_elec=ee_3t, atom_mask=am_3t,
            positions=pos_3t, box=box_3t[:3], cutoff=8.0,
        )
        # Tile (0, 2) and (1, 2) are > 12 A apart > 8.0 A cutoff, so AABB pruning strictly reduces num_valid_pairs!
        assert int(tp_pruned.num_valid_pairs) < int(tp_unpruned.num_valid_pairs)

        e_tp_p, f_tp_p = pallas_nonbonded_tile_pairs(
            pos_3t, q_3t, sig_3t, eps_3t, tp_pruned, atom_mask=am_3t, box=box_3t,
            pme_alpha=0.34, cutoff=8.0, return_forces=True, interpret=True,
        )
        def _ref_3t(p, q, s, ep):
            return jnp.sum(
                nonbonded_nl_reference(
                    p, q, s, ep, am_3t.astype(jnp.int32), nbr_3t, ei_3t, ev_3t, ee_3t, box_3t, lam_3t,
                    has_box=True, pme_alpha=0.34, cutoff=8.0,
                )
            )
        e_ref_3t, g_ref_3t = jax.value_and_grad(_ref_3t, argnums=(0, 1, 2, 3))(pos_3t, q_3t, sig_3t, eps_3t)
        np.testing.assert_allclose(np.asarray(e_tp_p), np.asarray(e_ref_3t), rtol=1e-4, atol=2e-4)
        np.testing.assert_allclose(np.asarray(f_tp_p), np.asarray(-g_ref_3t[0]), rtol=1e-4, atol=2e-4)

        g_tp_3t = jax.grad(
            lambda p, q, s, ep: jnp.sum(
                pallas_nonbonded_tile_pairs(
                    p, q, s, ep, tp_pruned, atom_mask=am_3t, box=box_3t,
                    pme_alpha=0.34, cutoff=8.0, return_forces=False, interpret=True,
                )
            ),
            argnums=(0, 1, 2, 3),
        )(pos_3t, q_3t, sig_3t, eps_3t)
        for g_got, g_exp in zip(g_tp_3t, g_ref_3t):
            np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_exp), rtol=2e-4, atol=2e-4)

        # Test B=4 vmap on tile-pairs (adaptive tp_group)
        pos_3t_b4 = jnp.stack([pos_3t + 0.01 * b for b in range(4)], axis=0)
        e_b4, f_b4 = jax.vmap(
            lambda r_b: pallas_nonbonded_tile_pairs(
                r_b, q_3t, sig_3t, eps_3t, tp_pruned, atom_mask=am_3t, box=box_3t,
                pme_alpha=0.34, cutoff=8.0, return_forces=True, interpret=True,
            )
        )(pos_3t_b4)
        assert e_b4.shape == (4,) and f_b4.shape == (4, N_tp, 3)
        np.testing.assert_allclose(np.asarray(e_b4[0]), np.asarray(e_tp_p), rtol=1e-5, atol=1e-4)

        # 4. MK-63 & MK-64: Adaptive nb_k_splits & Congeneric shared topology deduplication (B=4)
        B_cong = 4
        b_idx = jnp.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=jnp.int32)
        b_msk = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
        a_idx = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        a_msk = jnp.array([1.0, 1.0], dtype=jnp.float32)
        d_idx = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32)
        d_msk = jnp.array([1.0, 1.0], dtype=jnp.float32)
        h_idx = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        h_msk = jnp.array([1.0], dtype=jnp.float32)
        ep_idx = jnp.array([[0, 3], [4, 7]], dtype=jnp.int32)
        ep_msk = jnp.array([1.0, 1.0], dtype=jnp.float32)

        # Replicate topology indices across B=4 (to test MK-64 automatic deduplication to stride=0)
        b_idx_b4 = jnp.broadcast_to(b_idx[None, ...], (B_cong,) + b_idx.shape)
        b_msk_b4 = jnp.broadcast_to(b_msk[None, ...], (B_cong,) + b_msk.shape)
        a_idx_b4 = jnp.broadcast_to(a_idx[None, ...], (B_cong,) + a_idx.shape)
        a_msk_b4 = jnp.broadcast_to(a_msk[None, ...], (B_cong,) + a_msk.shape)
        d_idx_b4 = jnp.broadcast_to(d_idx[None, ...], (B_cong,) + d_idx.shape)
        d_msk_b4 = jnp.broadcast_to(d_msk[None, ...], (B_cong,) + d_msk.shape)
        h_idx_b4 = jnp.broadcast_to(h_idx[None, ...], (B_cong,) + h_idx.shape)
        h_msk_b4 = jnp.broadcast_to(h_msk[None, ...], (B_cong,) + h_msk.shape)
        ep_idx_b4 = jnp.broadcast_to(ep_idx[None, ...], (B_cong,) + ep_idx.shape)
        ep_msk_b4 = jnp.broadcast_to(ep_msk[None, ...], (B_cong,) + ep_msk.shape)

        # Distinct ligand parameters per replica
        q_b4 = jnp.stack([q_3t * (1.0 + 0.05 * b) for b in range(B_cong)], axis=0)
        sig_b4 = jnp.stack([sig_3t * (1.0 + 0.02 * b) for b in range(B_cong)], axis=0)
        eps_b4 = jnp.stack([eps_3t * (1.0 + 0.03 * b) for b in range(B_cong)], axis=0)
        bp_base = jnp.array([[1.5, 300.0], [1.4, 320.0], [1.5, 280.0], [1.5, 0.0]], dtype=jnp.float32)
        bp_b4 = jnp.stack([bp_base * jnp.array([1.0, 1.0 + 0.1 * b], dtype=jnp.float32) for b in range(B_cong)], axis=0)
        ap_base = jnp.array([[1.9, 50.0], [2.0, 60.0]], dtype=jnp.float32)
        ap_b4 = jnp.stack([ap_base * jnp.array([1.0, 1.0 + 0.05 * b], dtype=jnp.float32) for b in range(B_cong)], axis=0)
        dp_base = jnp.array([[2.0, 0.0, 1.5], [3.0, 3.14159, 0.8]], dtype=jnp.float32)
        dp_b4 = jnp.stack([dp_base * jnp.array([1.0, 1.0, 1.0 + 0.1 * b], dtype=jnp.float32) for b in range(B_cong)], axis=0)
        hp_base = jnp.array([[10.0, 0.0]], dtype=jnp.float32)
        hp_b4 = jnp.stack([hp_base * (1.0 + 0.05 * b) for b in range(B_cong)], axis=0)
        es_b4 = jnp.stack([jnp.array([3.0, 3.1], dtype=jnp.float32) * (1.0 + 0.01 * b) for b in range(B_cong)], axis=0)
        ee_14_b4 = jnp.stack([jnp.array([0.08, 0.09], dtype=jnp.float32) * (1.0 + 0.02 * b) for b in range(B_cong)], axis=0)
        eq_b4 = jnp.stack([jnp.array([-0.05, 0.04], dtype=jnp.float32) * (1.0 + 0.05 * b) for b in range(B_cong)], axis=0)

        e_cong, f_cong = pallas_bonded_and_nonbonded_nl_fused(
            pos_3t_b4,
            q_b4,
            sig_b4,
            eps_b4,
            jnp.broadcast_to(am_3t[None, :], (B_cong, N_tp)),
            jnp.broadcast_to(nbr_3t[None, :, :], (B_cong, N_tp, K_3t)),
            jnp.broadcast_to(ei_3t[None, :, :], (B_cong, N_tp, 4)),
            jnp.broadcast_to(ev_3t[None, :, :], (B_cong, N_tp, 4)),
            jnp.broadcast_to(ee_3t[None, :, :], (B_cong, N_tp, 4)),
            bonds=b_idx_b4, bond_params=bp_b4, bond_mask=b_msk_b4,
            angles=a_idx_b4, angle_params=ap_b4, angle_mask=a_msk_b4,
            dihedrals=d_idx_b4, dihedral_params=dp_b4, dihedral_mask=d_msk_b4,
            impropers=h_idx_b4, improper_params=hp_b4, improper_mask=h_msk_b4,
            exception_pairs=ep_idx_b4, exception_sigmas=es_b4, exception_epsilons=ee_14_b4,
            exception_chargeprods=eq_b4, exception_mask=ep_msk_b4,
            box=jnp.broadcast_to(box_3t[None, :], (B_cong, 6)),
            pme_alpha=0.34,
            cutoff=8.0,
            return_forces=True,
            interpret=True,
        )
        for b in range(B_cong):
            e_b1, f_b1 = pallas_bonded_and_nonbonded_nl_fused(
                pos_3t_b4[b],
                q_b4[b],
                sig_b4[b],
                eps_b4[b],
                am_3t,
                nbr_3t,
                ei_3t,
                ev_3t,
                ee_3t,
                bonds=b_idx, bond_params=bp_b4[b], bond_mask=b_msk,
                angles=a_idx, angle_params=ap_b4[b], angle_mask=a_msk,
                dihedrals=d_idx, dihedral_params=dp_b4[b], dihedral_mask=d_msk,
                impropers=h_idx, improper_params=hp_b4[b], improper_mask=h_msk,
                exception_pairs=ep_idx, exception_sigmas=es_b4[b], exception_epsilons=ee_14_b4[b],
                exception_chargeprods=eq_b4[b], exception_mask=ep_msk,
                box=box_3t,
                pme_alpha=0.34,
                cutoff=8.0,
                return_forces=True,
                interpret=True,
            )
            np.testing.assert_allclose(np.asarray(e_cong[b]), np.asarray(e_b1), rtol=1e-5, atol=1e-4)
            np.testing.assert_allclose(np.asarray(f_cong[b]), np.asarray(f_b1), rtol=1e-5, atol=1e-4)

        # 5. MK-64 EnsemblePlan._run_stacked_dispatch congeneric shared_top_fields execution
        bun_c0 = _make_periodic_solvated_bundle(dtype=jnp.float32, seed=40)
        bun_c1 = dataclasses.replace(
            bun_c0,
            charges=bun_c0.charges * jnp.float32(1.05),
            sigmas=bun_c0.sigmas * jnp.float32(1.01),
        )
        plan_cong = EnsemblePlan.from_bundles([bun_c0, bun_c1], use_pallas=True)
        trajs_cong = plan_cong.run(
            n_steps=2, dt=1.0, kT=0.6, seed=7, gamma=5.0,
            run_mode="inference", use_neighbor_list=True, use_pallas=True,
        )
        assert len(trajs_cong) == 2
        assert bool(jnp.all(jnp.isfinite(trajs_cong[0].positions)))
        assert bool(jnp.all(jnp.isfinite(trajs_cong[1].positions)))
    finally:
        jax.config.update("jax_enable_x64", prev_x64)
