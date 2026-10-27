"""xtrax.op wrapper of the SPME backward gather (prolix.physics.pme_op).

CPU only: the kernel runs in Pallas interpret mode, which checks the kernel's
math, not its GPU lowering.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from xtrax.intent import Context
from xtrax.testing.kernel_parity import (
    KernelParityError,
    assert_kernel_parity,
    assert_kernel_ran,
    require_corruption_red,
)

from prolix.physics.pme_op import gather_reference, make_gather_op
from prolix.physics.pme_pallas import gather_pallas

GRID = (16, 16, 16)
N_ATOMS = 37
N_GHOST = 5
CPU_KERNELS = Context(kernels=True, backend="cpu")


def _system(seed: int = 0):
    """37 atoms (5 ghosts) in a 20 A box and a random potential mesh."""
    k_pos, k_q, k_t = jax.random.split(jax.random.PRNGKey(seed), 3)
    box = jnp.array([20.0, 20.0, 20.0], jnp.float32)
    positions = jax.random.uniform(k_pos, (N_ATOMS, 3), jnp.float32) * box
    charges = jax.random.normal(k_q, (N_ATOMS,), jnp.float32)
    mask = jnp.arange(N_ATOMS) < N_ATOMS - N_GHOST
    theta = jax.random.normal(k_t, GRID, jnp.float32)
    return positions, charges, mask, box, theta


def _kernel(*args):
    return gather_pallas(*args, grid_dims=GRID, interpret=True)


def _reference(*args):
    return gather_reference(*args, grid_dims=GRID)


def test_kernel_matches_reference():
    assert_kernel_parity(_kernel, _reference, *_system())


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
    args = _system()
    positions, charges, mask, box, theta = args
    op = make_gather_op(GRID, context=CPU_KERNELS)

    def loss(fn, pos, q, t):
        forces, potentials = fn(pos, q, mask, box, t)
        return jnp.sum(forces**2) + jnp.sum(potentials * q)

    got = jax.grad(lambda p, q, t: loss(op, p, q, t), argnums=(0, 1, 2))(positions, charges, theta)
    want = jax.grad(lambda p, q, t: loss(_reference, p, q, t), argnums=(0, 1, 2))(
        positions, charges, theta
    )
    for g, w in zip(got, want, strict=True):
        assert jnp.allclose(g, w, rtol=1e-5, atol=1e-5)
