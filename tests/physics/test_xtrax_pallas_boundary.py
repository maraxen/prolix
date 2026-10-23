"""Tests for Task 1 (#6092): xtrax ChunkedMap alignment, export_context, and pure-array kernel boundary."""

from __future__ import annotations

import inspect
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix.api import ensemble_dispatch
from prolix.export import _IN_EXPORT_CONTEXT, export_context, export_energy_fn, export_langevin_step
from prolix.padding import PaddedSystem, pad_protein
from prolix.pallas_kernels import PallasKernelConfig, _triton_gpu_available, resolve_pallas_mode
from prolix.physics import flash_explicit
from prolix.tiling import axis_dispatch, xtrax_adapter
from prolix.tiling.axes import N_ATOMS, N_MOLS, N_WATERS
from prolix.tiling.planner import AxisDecision, AxisSpec, BatchPlan


def test_chunked_map_compatibility_in_xtrax_adapter():
    """Verify ChunkedMap and SafeMap are both imported and handled in xtrax_adapter."""
    assert hasattr(xtrax_adapter, "ChunkedMap")
    assert hasattr(xtrax_adapter, "SafeMap")

    # Test _strategy_to_batch_size with both SafeMap and a synthetic ChunkedMap-like subclass
    class CustomChunked(xtrax_adapter.ChunkedMap):
        pass

    ax = N_MOLS

    class DummyDecision:
        def __init__(self, strategy, batch_size=4, reasoning="test"):
            self.strategy = strategy
            self.batch_size = batch_size
            self.reasoning = reasoning

    dec_safe = DummyDecision(xtrax_adapter.SafeMap(batch_size=4))
    res_safe = xtrax_adapter.xtrax_decision_to_prolix(dec_safe, ax)
    assert res_safe.batch_size == 4

    dec_chunked = DummyDecision(CustomChunked(batch_size=6))
    res_chunked = xtrax_adapter.xtrax_decision_to_prolix(dec_chunked, ax)
    assert res_chunked.batch_size == 6


def test_axis_dispatch_chunked_map_and_non_multiple_waters(monkeypatch):
    """Verify axis_dispatch accepts ChunkedMap and allows non-multiple n_waters when strategy is ChunkedMap."""
    assert hasattr(axis_dispatch, "ChunkedMap")
    assert hasattr(axis_dispatch, "SafeMap")

    class SafeMap(axis_dispatch.ChunkedMap):
        pass

    class ChunkedMap(axis_dispatch.SafeMap):
        pass

    plan = BatchPlan(
        decisions=[
            AxisDecision(axis=N_WATERS, batch_size=4, reasoning="test"),
        ],
        total_memory_estimate=100.0,
        axes_by_index={N_WATERS.axis_index: N_WATERS},
        budget_exceeded=False,
    )

    # With legacy SafeMap (type name == "SafeMap"), non-multiple n_waters=6 % 4 != 0 raises ValueError
    monkeypatch.setattr(axis_dispatch, "SafeMap", SafeMap)
    monkeypatch.setattr(axis_dispatch, "n_waters_strategy", lambda p, n: SafeMap(batch_size=4))
    waters = jnp.arange(6 * 3, dtype=jnp.float32).reshape(6, 3)
    with pytest.raises(ValueError, match="N_WATERS SafeMap requires n_waters % batch_size == 0"):
        axis_dispatch.dispatch_n_waters(plan, 6, lambda w: w * 2.0, waters)

    # When strategy is ChunkedMap (type name == "ChunkedMap") and iterator supports remainder chunks,
    # dispatch_n_waters does not raise the SafeMap modulo ValueError.
    monkeypatch.setattr(axis_dispatch, "ChunkedMap", ChunkedMap)
    monkeypatch.setattr(axis_dispatch, "n_waters_strategy", lambda p, n: ChunkedMap(batch_size=4))
    monkeypatch.setattr(
        axis_dispatch,
        "make_axis_dispatch",
        lambda strat, axis: (lambda fn, xs, in_axes=0: jax.vmap(fn, in_axes=in_axes)(xs)),
    )
    out = axis_dispatch.dispatch_n_waters(plan, 6, lambda w: w * 2.0, waters)
    np.testing.assert_allclose(out, waters * 2.0)


def test_ensemble_dispatch_chunked_map(monkeypatch):
    """Verify ensemble_dispatch accepts ChunkedMap in dispatch_vmap_safemap, dispatch_n_mols, dispatch_n_atoms."""
    assert hasattr(ensemble_dispatch, "ChunkedMap")
    assert hasattr(ensemble_dispatch, "SafeMap")

    class ChunkedMap(ensemble_dispatch.SafeMap):
        pass

    monkeypatch.setattr(ensemble_dispatch, "ChunkedMap", ChunkedMap)
    monkeypatch.setattr(
        ensemble_dispatch,
        "make_axis_dispatch",
        lambda strat, axis: (lambda fn, xs, in_axes=0: jax.vmap(fn, in_axes=in_axes)(xs)),
    )

    xs = jnp.arange(5, dtype=jnp.float32)
    out = ensemble_dispatch.dispatch_vmap_safemap("test_axis", ChunkedMap(batch_size=2), lambda x: x + 1.0, xs)
    np.testing.assert_allclose(out, xs + 1.0)


def test_pallas_kernel_config_registered_dataclass():
    """Verify PallasKernelConfig is a frozen JAX-registered dataclass with static fields (zero dynamic leaves)."""
    cfg = PallasKernelConfig(block_n=32, block_k=64, use_pallas=True, interpret=True)
    assert cfg.block_n == 32
    assert cfg.block_k == 64
    assert cfg.use_pallas is True
    assert cfg.interpret is True

    # Frozen check
    with pytest.raises(Exception):
        cfg.block_n = 16  # type: ignore[misc]

    # All fields are static metadata -> tree_leaves must be empty
    leaves, treedef = jax.tree_util.tree_flatten(cfg)
    assert leaves == []
    unflattened = jax.tree_util.tree_unflatten(treedef, leaves)
    assert unflattened == cfg


def test_export_context_and_resolve_pallas_mode(monkeypatch):
    """Verify export_context forces resolve_pallas_mode to return (False, False)."""
    assert _IN_EXPORT_CONTEXT.get() is False

    # Outside export_context: explicit use_pallas=True enables Pallas
    enabled, interpret = resolve_pallas_mode(use_pallas=True)
    assert enabled is True
    assert interpret == (not _triton_gpu_available())

    # Env var resolution
    monkeypatch.setenv("PROLIX_USE_PALLAS", "1")
    enabled_env, interpret_env = resolve_pallas_mode(use_pallas=None)
    assert enabled_env is True
    assert interpret_env == (not _triton_gpu_available())

    # PME env var resolution
    monkeypatch.delenv("PROLIX_USE_PALLAS", raising=False)
    monkeypatch.setenv("PROLIX_USE_PALLAS_PME", "true")
    enabled_pme, _ = resolve_pallas_mode(use_pallas=None, env_var="PROLIX_USE_PALLAS_PME")
    assert enabled_pme is True

    # Inside export_context: unconditionally (False, False) even when use_pallas=True
    with export_context():
        assert _IN_EXPORT_CONTEXT.get() is True
        assert resolve_pallas_mode(use_pallas=True) == (False, False)
        assert resolve_pallas_mode(use_pallas=None, env_var="PROLIX_USE_PALLAS_PME") == (False, False)

    assert _IN_EXPORT_CONTEXT.get() is False

    # Verify export_energy_fn enters export_context during lowering
    seen_in_export = []

    def dummy_energy(params, positions):
        seen_in_export.append(resolve_pallas_mode(use_pallas=True))
        return jnp.sum(positions ** 2) * params

    lowered = export_energy_fn(dummy_energy, jnp.array(2.0), jnp.ones((4, 3)))
    assert lowered is not None
    assert seen_in_export == [(False, False)]

    # Verify export_langevin_step enters export_context
    seen_step = []

    def dummy_step(state):
        seen_step.append(resolve_pallas_mode(use_pallas=True))
        return state

    wrapped_step = export_langevin_step(dummy_step, config=None)
    wrapped_step(jnp.ones((2, 3)))
    assert seen_step == [(False, False)]


def _make_minimal_periodic_padded_system(n_real: int = 6, n_padded: int = 8) -> PaddedSystem:
    """Construct a minimal periodic PaddedSystem with bonded exclusions and ghost atoms."""
    from prolix.physics.neighbor_list import ExclusionSpec, map_exclusions_to_dense_padded
    from prolix.utils import topology

    rng = np.random.default_rng(123)
    pos = np.zeros((n_padded, 3), dtype=np.float32)
    pos[:n_real] = rng.uniform(2.0, 12.0, size=(n_real, 3)).astype(np.float32)
    # Place ghost atoms at index 0's position or nearby to catch any unmasked or bogus excl_indices=0 bug!
    pos[n_real:] = pos[0] + np.array([0.5, 0.0, 0.0], dtype=np.float32)

    charges = np.zeros(n_padded, dtype=np.float32)
    charges[:n_real] = np.array([-0.834, 0.417, 0.417, -0.834, 0.417, 0.417][:n_real], dtype=np.float32)

    sigmas = np.ones(n_padded, dtype=np.float32) * 3.15
    epsilons = np.zeros(n_padded, dtype=np.float32)
    epsilons[:n_real] = 0.152

    atom_mask = np.arange(n_padded) < n_real

    # Bonds for two water molecules: (0,1), (0,2), (3,4), (3,5)
    bonds = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]], dtype=np.int32)
    bond_params = np.tile(np.array([[0.9572, 450.0]], dtype=np.float32), (len(bonds), 1))
    bond_mask = np.ones(len(bonds), dtype=bool)

    excl = topology.find_bonded_exclusions(bonds, n_padded)
    excl_spec = ExclusionSpec(
        idx_12_13=jnp.concatenate([excl.idx_12, excl.idx_13], axis=0),
        idx_14=excl.idx_14,
        scale_14_elec=0.83333333,
        scale_14_vdw=0.5,
        n_atoms=n_padded,
        exception_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros((0,), dtype=jnp.float32),
        exception_epsilons=jnp.zeros((0,), dtype=jnp.float32),
        exception_chargeprods=jnp.zeros((0,), dtype=jnp.float32),
    )
    excl_indices, excl_scales_vdw, excl_scales_elec = map_exclusions_to_dense_padded(
        excl_spec, max_exclusions=32
    )

    z_idx = jnp.zeros((1, 3), dtype=jnp.int32)
    z_param = jnp.zeros((1, 2), dtype=jnp.float32)
    z_mask = jnp.zeros((1,), dtype=bool)
    z_dih_idx = jnp.zeros((1, 4), dtype=jnp.int32)
    z_dih_param = jnp.zeros((1, 3), dtype=jnp.float32)

    return PaddedSystem(
        positions=jnp.asarray(pos),
        charges=jnp.asarray(charges),
        sigmas=jnp.asarray(sigmas),
        epsilons=jnp.asarray(epsilons),
        radii=jnp.ones(n_padded, dtype=jnp.float32),
        scaled_radii=jnp.ones(n_padded, dtype=jnp.float32),
        masses=jnp.ones(n_padded, dtype=jnp.float32) * 12.0,
        element_ids=jnp.ones(n_padded, dtype=jnp.int32) * 6,
        atom_mask=jnp.asarray(atom_mask),
        is_hydrogen=jnp.zeros(n_padded, dtype=bool),
        is_backbone=jnp.zeros(n_padded, dtype=bool),
        is_heavy=jnp.ones(n_padded, dtype=bool),
        protein_atom_mask=jnp.asarray(atom_mask),
        water_atom_mask=jnp.zeros(n_padded, dtype=bool),
        bonds=jnp.asarray(bonds),
        bond_params=jnp.asarray(bond_params),
        bond_mask=jnp.asarray(bond_mask),
        angles=z_idx,
        angle_params=z_param,
        angle_mask=z_mask,
        dihedrals=z_dih_idx,
        dihedral_params=z_dih_param,
        dihedral_mask=z_mask,
        impropers=z_dih_idx,
        improper_params=z_dih_param,
        improper_mask=z_mask,
        urey_bradley_bonds= jnp.zeros((1, 2), dtype=jnp.int32),
        urey_bradley_params=z_param,
        urey_bradley_mask=z_mask,
        cmap_torsions=jnp.zeros((1, 5), dtype=jnp.int32),
        cmap_indices=jnp.zeros((1,), dtype=jnp.int32),
        cmap_mask=z_mask,
        cmap_coeffs=None,
        excl_indices=excl_indices,
        excl_scales_vdw=excl_scales_vdw,
        excl_scales_elec=excl_scales_elec,
        constraint_pairs=jnp.zeros((1, 2), dtype=jnp.int32),
        constraint_lengths=jnp.zeros((1,), dtype=jnp.float32),
        constraint_mask=z_mask,
        water_indices=jnp.zeros((1, 3), dtype=jnp.int32),
        water_mask=z_mask,
        box_size=np.array([20.0, 20.0, 20.0], dtype=np.float32),
        pme_alpha=0.34,
        pme_grid_points=16,
        nonbonded_cutoff=8.0,
        n_real_atoms=jnp.array(n_real, dtype=jnp.int32),
        n_padded_atoms=n_padded,
        bucket_size=n_padded,
    )


def test_pad_to_tile_multiple_pads_exclusions_safely():
    """Verify _pad_to_tile_multiple pads excl_indices with -1 and excl_scales_* with 1.0."""
    sys = _make_minimal_periodic_padded_system(n_real=6, n_padded=8)
    (
        pos_pad,
        charges_pad,
        sigmas_pad,
        epsilons_pad,
        mask_pad,
        excl_idx_pad,
        excl_vdw_pad,
        excl_elec_pad,
        n_tiles,
    ) = flash_explicit._pad_to_tile_multiple(
        sys.positions,
        sys.charges,
        sys.sigmas,
        sys.epsilons,
        sys.atom_mask,
        sys,
        T=16,
        include_elec=True,
    )
    assert n_tiles == 1
    assert pos_pad.shape[0] == 16
    # Padded rows [8:16] must have excl_indices == -1 and excl_scales == 1.0
    np.testing.assert_array_equal(np.asarray(excl_idx_pad[8:]), -1)
    np.testing.assert_allclose(np.asarray(excl_vdw_pad[8:]), 1.0)
    np.testing.assert_allclose(np.asarray(excl_elec_pad[8:]), 1.0)


def test_flash_explicit_no_equinox_and_jit_vmap_safe(monkeypatch):
    """Verify flash_explicit does not use equinox.filter_grad or host-only find_bonded_exclusions under jit/vmap."""
    from prolix.utils import topology

    # 1. Verify equinox is not imported in flash_explicit
    src = inspect.getsource(flash_explicit)
    assert "filter_grad" not in src
    assert "import equinox" not in src

    sys = _make_minimal_periodic_padded_system(n_real=6, n_padded=8)

    # Poison topology.find_bonded_exclusions to prove it is NOT called when excl_indices is present
    def _forbidden_host_exclusions(*args, **kwargs):
        raise RuntimeError("topology.find_bonded_exclusions must not be called inside traced energy/force fn")

    monkeypatch.setattr(topology, "find_bonded_exclusions", _forbidden_host_exclusions)

    # 2. Test PME, EFA, and EFA_LEBEDEV under jax.jit
    for method in ("pme", "efa", "efa_lebedev"):
        e = jax.jit(lambda s: flash_explicit.flash_explicit_energy(s, T=8, electrostatic_method=method))(sys)
        f = jax.jit(lambda s: flash_explicit.flash_explicit_forces(s, T=8, electrostatic_method=method))(sys)
        assert jnp.isfinite(e), f"Non-finite energy for {method}"
        assert f.shape == (8, 3)
        assert jnp.all(jnp.isfinite(f)), f"Non-finite forces for {method}"
        # Ghost atoms [6:8] must have zero force
        np.testing.assert_allclose(np.asarray(f[6:]), 0.0, atol=1e-7)

    # 3. Verify forces match -jax.grad wrt positions
    grad_pos = jax.grad(
        lambda pos: flash_explicit._total_energy_from_positions(pos, sys, T=8, electrostatic_method="pme")
    )(sys.positions)
    f_pme = flash_explicit.flash_explicit_forces(sys, T=8, electrostatic_method="pme")
    np.testing.assert_allclose(np.asarray(f_pme), np.asarray(-grad_pos * sys.atom_mask[:, None]), rtol=1e-5, atol=1e-5)
