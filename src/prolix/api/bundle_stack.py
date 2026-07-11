"""Stack MolecularBundle batches for JIT vmap on N_MOLS (#2645)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from prolix.types.bundles import MolecularBundle


def can_stack_molecular_bundles(bundles: list[MolecularBundle]) -> bool:
    """True when every bundle shares shape_spec and per-field array shapes."""
    if len(bundles) < 2:
        return False
    ref = bundles[0]
    for bundle in bundles[1:]:
        if bundle.shape_spec != ref.shape_spec:
            return False
        ref_leaves, _ = jax.tree.flatten(ref)
        cur_leaves, _ = jax.tree.flatten(bundle)
        if len(ref_leaves) != len(cur_leaves):
            return False
        for a, b in zip(ref_leaves, cur_leaves, strict=True):
            if not (isinstance(a, jnp.ndarray) and isinstance(b, jnp.ndarray)):
                continue
            if a.shape != b.shape:
                return False
    return True


def _n_atoms_equal(a: MolecularBundle, b: MolecularBundle) -> bool:
    """Compare dynamic n_atoms scalars without Python int() (host-side only)."""
    return bool(
        jnp.array_equal(
            jnp.asarray(a.n_atoms, dtype=jnp.int32),
            jnp.asarray(b.n_atoms, dtype=jnp.int32),
        )
    )


def can_jit_vmap_n_mols(bundles: list[MolecularBundle]) -> bool:
    """True when stack-compatible and all bundles share the same n_atoms scalar."""
    if not can_stack_molecular_bundles(bundles):
        return False
    ref = bundles[0]
    return all(_n_atoms_equal(ref, b) for b in bundles[1:])


def stack_molecular_bundles(bundles: list[MolecularBundle]) -> MolecularBundle:
    """Stack B bundles along axis 0 on every array leaf."""
    if not bundles:
        raise ValueError("Cannot stack empty bundle list")
    if not can_stack_molecular_bundles(bundles):
        raise ValueError("Bundles are not stack-compatible (shape_spec or array shapes differ)")
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *bundles)


def unstack_trajectories(
    batched: Any,
    bundles: list[MolecularBundle],
) -> list[Any]:
    """Split a vmapped Trajectory and trim each to real atom count on the host."""
    from prolix.api.bundle_md import trim_trajectory_positions
    from prolix.api.observables import Trajectory

    if not isinstance(batched, Trajectory):
        raise TypeError(f"Expected Trajectory, got {type(batched)}")

    batch_size = len(bundles)

    def _slice_leaf(x):
        if isinstance(x, jnp.ndarray) and x.ndim > 0:
            return [x[i] for i in range(batch_size)]
        return [x] * batch_size

    positions_lists = _slice_leaf(batched.positions)
    obs_lists: dict[str, list[Any]] = {}
    for name, value in batched.observable_values.items():
        obs_lists[name] = _slice_leaf(value)

    out: list[Trajectory] = []
    for i, bundle in enumerate(bundles):
        obs_i = {name: obs_lists[name][i] for name in batched.observable_values}
        pos = trim_trajectory_positions(positions_lists[i], bundle)
        out.append(
            Trajectory(
                positions=pos,
                observable_values=obs_i,
                n_steps=batched.n_steps,
            )
        )
    return out
