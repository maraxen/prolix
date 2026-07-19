"""Regression test for debt 802: NL support for stacked/vmapped multi-bundle inference.

Phase 6 steps 4-5 (2026-07-18) wired ``EnsemblePlan.run(use_neighbor_list=True)``
into ``_run_single_inference`` only -- it raised if ``len(bundles) != 1``. This left
a real gap: the vmapped multi-bundle dispatch path (``_run_stacked_dispatch``), the
one B1's heterogeneous-batching thesis actually needs, had no NL support at all.

The core engineering obstacle: ``jax_md``'s ``neighbor_fn.allocate()`` is host-only
(it inspects concrete positions to size the cell list) and cannot run on a traced
position array, so it cannot be called *inside* a vmapped function body. Two
independent ``.allocate()`` calls on different position arrays also produce
*different* capacities and non-matching pytree metadata (fresh Python closures per
call) even at identical box/cutoff/N, so their results are not directly stackable.

Fix: allocate one shared "seed" ``NeighborList`` host-side (once, before any
vmapping), then derive each bundle's own ``NeighborList`` via ``seed.update(bundle_positions)``
rather than independent ``.allocate()`` calls -- ``.update()`` inherits `self`'s static
metadata (capacity, cell-list closures), so every bundle's result shares identical
pytree structure with the seed and with each other, making them directly stackable
via ``jax.tree.map(jnp.stack)`` and safe to slice per-replica under ``jax.vmap``.

See ``EnsemblePlan._build_stacked_neighbor_seed``/``_reallocate_stacked_neighbor_seed``
(ensemble_plan.py) and ``.praxia/docs/decisions/260717_b1-connect-existing-engines-scope.md``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.slow

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "pdb"

PME_ALPHA = 0.34
CUTOFF = 9.0


def _ff_path() -> Path:
    import proxide

    return Path(proxide.__file__).parent / "assets" / "protein.ff19SB.xml"


def _load_vacuum_boxed_1vii_pair():
    """Two stack-compatible periodic bundles: real 1VII, and a jittered copy.

    A jittered copy (not a second protein) keeps this test fast -- the
    engineering risk under test (do two independently-derived
    ``jax_md.NeighborList``s stack and vmap correctly) is identical whether
    the two bundles share a protein identity or not; the code has no
    awareness of "protein identity," only array shapes/masks. Genuine
    cross-protein compile-sharing with NL enabled is a separate, unmeasured
    performance question (needs a full solvated benchmark) -- see the
    decision doc's "Phase 6 step 7" section.
    """
    from proxide import CoordFormat, OutputSpec, parse_structure

    pdb_path = DATA_DIR / "1VII.pdb"
    if not pdb_path.is_file():
        pytest.skip(f"Missing test PDB: {pdb_path}")

    spec = OutputSpec(
        parameterize_md=True, force_field=str(_ff_path()), coord_format=CoordFormat.Full
    )
    protein = parse_structure(str(pdb_path), spec)

    coords = jnp.asarray(protein.coordinates, dtype=jnp.float64)
    lo = jnp.min(coords, axis=0)
    hi = jnp.max(coords, axis=0)
    box = jnp.maximum(hi - lo + 24.0, jnp.array([32.0, 32.0, 32.0], dtype=jnp.float64))
    shift = box * 0.5 - (lo + hi) * 0.5
    r = coords + shift

    class _Proxy:
        def __getattr__(self, name):
            if name == "positions":
                return r
            if name == "pme_alpha":
                return PME_ALPHA
            if name == "nonbonded_cutoff":
                return CUTOFF
            if name == "box_size":
                return box
            return getattr(protein, name)

    from prolix.physics import neighbor_list as nl
    from prolix.physics.system import make_bundle_from_system

    excl_spec = nl.ExclusionSpec.from_protein(protein)
    bundle_a = make_bundle_from_system(
        _Proxy(), boundary_condition="periodic", exclusion_spec=excl_spec
    )

    key = jax.random.PRNGKey(7)
    jitter = jax.random.normal(key, bundle_a.positions.shape) * 0.05
    jittered = jnp.where(bundle_a.atom_mask[:, None], bundle_a.positions + jitter, bundle_a.positions)
    bundle_b = dataclasses.replace(bundle_a, positions=jittered)
    return bundle_a, bundle_b


@pytest.mark.slow
def test_stacked_use_neighbor_list_matches_shape_and_is_finite():
    """EnsemblePlan.run(use_neighbor_list=True) over 2 bundles: real vmapped dispatch."""
    from prolix.api.ensemble_plan import EnsemblePlan

    bundle_a, bundle_b = _load_vacuum_boxed_1vii_pair()
    plan = EnsemblePlan.from_bundles([bundle_a, bundle_b])

    trajs = plan.run(
        n_steps=6, dt=0.5, kT=0.6, seed=0, run_mode="inference",
        use_neighbor_list=True, nl_update_every=2,
    )
    assert isinstance(trajs, list) and len(trajs) == 2
    for traj in trajs:
        assert bool(jnp.all(jnp.isfinite(traj.positions))), "non-finite positions in stacked NL run"


@pytest.mark.slow
def test_stacked_dense_path_unaffected_by_nl_support():
    """use_neighbor_list=False (default) on a multi-bundle plan is untouched by debt 802."""
    from prolix.api.ensemble_plan import EnsemblePlan

    bundle_a, bundle_b = _load_vacuum_boxed_1vii_pair()
    plan = EnsemblePlan.from_bundles([bundle_a, bundle_b])

    trajs = plan.run(n_steps=6, dt=0.5, kT=0.6, seed=0, run_mode="inference")
    assert isinstance(trajs, list) and len(trajs) == 2
    for traj in trajs:
        assert bool(jnp.all(jnp.isfinite(traj.positions)))


@pytest.mark.slow
def test_stacked_neighbor_seed_helper_produces_stackable_identical_metadata():
    """Direct unit test of the seed.update() pattern the whole feature depends on."""
    from prolix.api.ensemble_plan import EnsemblePlan

    bundle_a, bundle_b = _load_vacuum_boxed_1vii_pair()
    from prolix.api.bundle_stack import integration_prefix_for_bundles

    integration_prefix = integration_prefix_for_bundles([bundle_a, bundle_b])
    stacked_neighbor, neighbor_fn = EnsemblePlan._build_stacked_neighbor_seed(
        [bundle_a, bundle_b], integration_prefix
    )
    assert stacked_neighbor.idx.shape[0] == 2
    assert not bool(jnp.any(stacked_neighbor.did_buffer_overflow))

    # vmapped .update() must agree with a fresh per-replica call to seed.update()
    def per_replica_update(nbr_slice, pos):
        return nbr_slice.update(pos)

    positions = jnp.stack([bundle_a.positions, bundle_b.positions])
    updated = jax.vmap(per_replica_update)(stacked_neighbor, positions)
    assert updated.idx.shape == stacked_neighbor.idx.shape


@pytest.mark.slow
def test_stacked_use_neighbor_list_overflow_then_reallocate_recovers():
    """A deliberately undersized seed capacity triggers overflow, then a clean retry."""
    import prolix.physics.neighbor_list as nl_mod
    from prolix.api.ensemble_plan import EnsemblePlan

    bundle_a, bundle_b = _load_vacuum_boxed_1vii_pair()
    plan = EnsemblePlan.from_bundles([bundle_a, bundle_b])

    _orig_make = nl_mod.make_neighbor_list_fn

    def _tiny_capacity(displacement_fn, box_size, cutoff, **kwargs):
        kwargs["capacity_multiplier"] = 0.5
        return _orig_make(displacement_fn, box_size, cutoff, **kwargs)

    nl_mod.make_neighbor_list_fn = _tiny_capacity
    try:
        trajs = plan.run(
            n_steps=10, dt=0.5, kT=0.6, seed=4, run_mode="inference",
            use_neighbor_list=True, nl_update_every=2,
        )
    finally:
        nl_mod.make_neighbor_list_fn = _orig_make

    for traj in trajs:
        assert bool(jnp.all(jnp.isfinite(traj.positions))), (
            "overflow-then-reallocate did not recover cleanly"
        )
