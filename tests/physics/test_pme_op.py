"""xtrax.op wrappers of the SPME charge spread and backward gather (prolix.physics.pme_op).

CPU only: the kernel runs in Pallas interpret mode, which checks the kernel's
math, not its GPU lowering.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import pytest
import xtrax
from xtrax.intent import Context
from xtrax.testing.kernel_parity import (
    KernelParityError,
    assert_kernel_parity,
    assert_kernel_ran,
    require_corruption_red,
)

from prolix.physics.pme import spme_energy_with_forces
from prolix.physics.pme_op import (
    gather_reference,
    make_gather_op,
    make_spread_op,
    spread_reference,
)
from prolix.physics.pme_pallas import gather_pallas, spread_charges_pallas

GRID = (16, 16, 16)
NON_CUBIC_GRID = (12, 16, 20)
N_ATOMS = 37
N_GHOST = 5
CPU_KERNELS = Context(kernels=True, backend="cpu")


def _system(
    seed: int = 0,
    *,
    grid: tuple[int, int, int] = GRID,
    box_edges: tuple[float, float, float] = (20.0, 20.0, 20.0),
    dtype: jnp.dtype = jnp.float32,
):
    """37 atoms (5 ghosts) in a box and a random potential mesh."""
    k_pos, k_q, k_t = jax.random.split(jax.random.PRNGKey(seed), 3)
    box = jnp.array(box_edges, dtype)
    positions = jax.random.uniform(k_pos, (N_ATOMS, 3), dtype) * box
    charges = jax.random.normal(k_q, (N_ATOMS,), dtype)
    mask = jnp.arange(N_ATOMS) < N_ATOMS - N_GHOST
    theta = jax.random.normal(k_t, grid, dtype)
    return positions, charges, mask, box, theta


def _kernel(*args):
    return gather_pallas(*args, grid_dims=GRID, interpret=True)


def _reference(*args):
    return gather_reference(*args, grid_dims=GRID)


def test_kernel_matches_reference():
    assert_kernel_parity(_kernel, _reference, *_system())


def test_non_cubic_grid_and_orthorhombic_box_parity():
    args = _system(seed=7, grid=NON_CUBIC_GRID, box_edges=(18.0, 22.0, 25.0))
    op = make_gather_op(NON_CUBIC_GRID, context=CPU_KERNELS)
    ref = functools.partial(gather_reference, grid_dims=NON_CUBIC_GRID)
    assert_kernel_ran(op, *args)
    assert_kernel_parity(op, ref, *args)

    # Also check spread op on non-cubic grid and orthorhombic box
    pos, q, mask, box, _ = args
    spread_op = make_spread_op(NON_CUBIC_GRID, context=CPU_KERNELS)
    spread_ref = functools.partial(spread_reference, grid_dims=NON_CUBIC_GRID)
    assert_kernel_ran(spread_op, pos, q, mask, box)
    assert_kernel_parity(spread_op, spread_ref, pos, q, mask, box)


def test_ghost_atoms_are_zero_in_both():
    args = _system()
    for fn in (_kernel, _reference):
        forces, potentials = fn(*args)
        assert jnp.all(forces[-N_GHOST:] == 0.0)
        assert jnp.all(potentials[-N_GHOST:] == 0.0)


def test_corrupted_kernel_fails_parity():
    def corrupt(kernel):
        def broken(*args):
            forces, potentials = kernel(*args)
            return forces.at[0, 0].add(1.0), potentials

        return broken

    require_corruption_red(_kernel, _reference, _system(), corrupt)


def test_kernels_context_runs_the_pallas_kernel():
    op = make_gather_op(GRID, context=CPU_KERNELS)
    assert op.select().primal == "kernel:cpu"
    assert_kernel_ran(op, *_system())
    assert_kernel_parity(op, _reference, *_system())


@pytest.mark.parametrize(
    "ctx",
    [None, Context(kernels=True, target="export", export_format="stablehlo")],
    ids=["default", "export"],
)
def test_default_and_export_contexts_run_the_reference(ctx):
    op = make_gather_op(GRID, context=ctx)
    assert op.select().primal == "reference"
    with pytest.raises(KernelParityError):
        assert_kernel_ran(op, *_system())


def test_gradient_through_the_kernel_comes_from_the_reference():
    """Verify xtrax.op differentiates the reference, not the kernel primal.

    ``gather_pallas`` uses ``pl.pallas_call`` without a custom JVP/VJP, so
    differentiating ``_kernel`` directly raises an ``AssertionError`` inside
    Pallas's JVP rule. Under ``xtrax.op(..., context=CPU_KERNELS)``, the
    forward pass runs ``gather_pallas`` (verified by ``assert_kernel_ran``),
    while ``jax.grad`` differentiates ``gather_reference`` and succeeds with
    exact reference gradients.
    """
    args = _system()
    positions, charges, mask, box, theta = args
    op = make_gather_op(GRID, context=CPU_KERNELS)
    assert_kernel_ran(op, *args)

    def loss(fn, pos, q, t):
        forces, potentials = fn(pos, q, mask, box, t)
        return jnp.sum(forces**2) + jnp.sum(potentials * q)

    # Direct differentiation of the raw Pallas kernel fails (pallas_call has no JVP rule):
    with pytest.raises(Exception):
        jax.grad(lambda p, q, t: loss(_kernel, p, q, t), argnums=(0, 1, 2))(
            positions, charges, theta
        )

    # Differentiation of the xtrax.op wrapper succeeds and matches the reference VJP:
    got = jax.grad(lambda p, q, t: loss(op, p, q, t), argnums=(0, 1, 2))(positions, charges, theta)
    want = jax.grad(lambda p, q, t: loss(_reference, p, q, t), argnums=(0, 1, 2))(
        positions, charges, theta
    )
    for g, w in zip(got, want, strict=True):
        assert jnp.linalg.norm(w) > 0.0
        assert jnp.allclose(g, w, rtol=1e-5, atol=1e-5)


def test_vmap_and_batched_inputs_parity():
    s0 = _system(seed=0)
    s1 = _system(seed=1)
    batched = tuple(jnp.stack([a, b], axis=0) for a, b in zip(s0, s1, strict=True))

    op = make_gather_op(GRID, context=CPU_KERNELS)
    # Direct 3D batched call
    assert_kernel_parity(op, _reference, *batched)
    # jax.vmap over 2D unbatched call
    assert_kernel_parity(jax.vmap(op), jax.vmap(_reference), *batched)


def test_float64_parity():
    jax.config.update("jax_enable_x64", True)
    try:
        args = _system(seed=3, grid=NON_CUBIC_GRID, box_edges=(18.0, 22.0, 25.0), dtype=jnp.float64)
        op = make_gather_op(NON_CUBIC_GRID, context=CPU_KERNELS)
        ref = functools.partial(gather_reference, grid_dims=NON_CUBIC_GRID)
        forces, potentials = op(*args)
        assert forces.dtype == jnp.float64
        assert potentials.dtype == jnp.float64
        assert_kernel_parity(op, ref, *args, rel=1e-12)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_spme_energy_with_forces_end_to_end_uses_xtrax_op():
    pos, q, mask, box, _ = _system(seed=5, grid=NON_CUBIC_GRID, box_edges=(18.0, 22.0, 25.0))
    e_pallas, (gpos_p, gq_p, gbox_p) = jax.value_and_grad(
        lambda p, c, b: spme_energy_with_forces(p, c, mask, b, NON_CUBIC_GRID, 0.34, 4, True),
        argnums=(0, 1, 2),
    )(pos, q, box)
    e_ref, (gpos_r, gq_r, gbox_r) = jax.value_and_grad(
        lambda p, c, b: spme_energy_with_forces(p, c, mask, b, NON_CUBIC_GRID, 0.34, 4, False),
        argnums=(0, 1, 2),
    )(pos, q, box)
    assert jnp.allclose(e_pallas, e_ref, rtol=1e-5, atol=1e-4)
    assert jnp.allclose(gpos_p, gpos_r, rtol=1e-5, atol=1e-4)
    assert jnp.allclose(gq_p, gq_r, rtol=1e-5, atol=1e-4)
    assert jnp.allclose(gbox_p, gbox_r, rtol=1e-5, atol=1e-4)
