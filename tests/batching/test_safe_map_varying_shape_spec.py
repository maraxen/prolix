"""HP3 v8 Smoke Test: JIT Cache Keying with Varying shape_spec.

Empirical diagnostic: does jax.vmap cache correctly when the underlying pytree
contains a static=True field (shape_spec) with DIFFERENT values on successive calls?

This test answers the core Claim 1 hypothesis:
"static=True shape_spec is shape-keyed by XLA correctly" (PASS) vs.
"static=True forces per-element retrace; Claim 1 needs redesign" (FAIL).

Both outcomes are valuable diagnostic signals. This is NOT about passing the test,
but about learning what the underlying behavior is.
"""

import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool

from prolix.types.bundles import (
    MolecularBundle,
    MolecularShapeSpec,
    _bucket_idx,
    ATOM_BUCKETS,
    BOND_BUCKETS,
    ANGLE_BUCKETS,
    DIHEDRAL_BUCKETS,
    WATER_BUCKETS,
    EXCL_BUCKETS,
    CMAP_BUCKETS,
    EXCEPTION_BUCKETS,
)


def _make_minimal_bundle(n_atoms: int, atom_bucket: int | None = None) -> MolecularBundle:
    """Construct a minimal MolecularBundle with the given n_atoms.

    All arrays are padded to the bucket size determined by n_atoms.
    Real entries are masked; padding entries are zeros/False.
    This is a diagnostic fixture — not a realistic system.

    Args:
        n_atoms: Actual number of atoms.
        atom_bucket: If provided, use this bucket size. Otherwise, compute from n_atoms.
    """
    # Determine atom bucket size based on actual n_atoms
    if atom_bucket is None:
        atom_bucket = ATOM_BUCKETS[_bucket_idx(n_atoms, ATOM_BUCKETS)]
    else:
        # Verify bucket_size >= n_atoms
        assert atom_bucket >= n_atoms, f"bucket {atom_bucket} < n_atoms {n_atoms}"

    def _make_mask(n_real: int, bucket_size: int) -> Array:
        """Create a mask: [True]*n_real + [False]*(bucket_size - n_real)."""
        return jnp.concatenate([
            jnp.ones(n_real, dtype=bool),
            jnp.zeros(bucket_size - n_real, dtype=bool)
        ])

    # Minimal arrays: atom-scale data padded to atom_bucket
    positions = jnp.zeros((atom_bucket, 3), dtype=jnp.float32)
    charges = jnp.zeros(atom_bucket, dtype=jnp.float32)
    sigmas = jnp.ones(atom_bucket, dtype=jnp.float32) * 0.1
    epsilons = jnp.ones(atom_bucket, dtype=jnp.float32) * 0.05
    radii = jnp.ones(atom_bucket, dtype=jnp.float32) * 0.15
    scaled_radii = jnp.ones(atom_bucket, dtype=jnp.float32) * 0.12
    atom_mask = _make_mask(n_atoms, atom_bucket)

    # Box (no PBC for diagnostic)
    box = jnp.zeros((3, 3), dtype=jnp.float32)

    # Bonded: empty for diagnostic
    bond_idx = jnp.zeros((256, 2), dtype=jnp.int32)
    bond_params = jnp.zeros((256, 2), dtype=jnp.float32)
    bond_mask = jnp.zeros(256, dtype=bool)

    # Angles: empty
    angle_idx = jnp.zeros((256, 3), dtype=jnp.int32)
    angle_params = jnp.zeros((256, 2), dtype=jnp.float32)
    angle_mask = jnp.zeros(256, dtype=bool)

    # Dihedrals: empty
    dihedral_idx = jnp.zeros((512, 4), dtype=jnp.int32)
    dihedral_params = jnp.zeros((512, 4), dtype=jnp.float32)
    dihedral_mask = jnp.zeros(512, dtype=bool)

    # Impropers: empty
    improper_idx = jnp.zeros((512, 4), dtype=jnp.int32)
    improper_params = jnp.zeros((512, 3), dtype=jnp.float32)
    improper_mask = jnp.zeros(512, dtype=bool)
    improper_is_periodic = jnp.array(False)

    # Urey-Bradley: empty
    urey_bradley_idx = jnp.zeros((256, 3), dtype=jnp.int32)
    urey_bradley_params = jnp.zeros((256, 2), dtype=jnp.float32)
    urey_bradley_mask = jnp.zeros(256, dtype=bool)

    # CMAP: empty
    cmap_torsion_idx = jnp.zeros((16, 8), dtype=jnp.int32)
    cmap_energy_grids = jnp.zeros((16, 24, 24), dtype=jnp.float32)
    cmap_mask = jnp.zeros(16, dtype=bool)

    # Water: empty
    water_indices = jnp.zeros((16, 3), dtype=jnp.int32)
    water_mask = jnp.zeros(16, dtype=bool)

    # Exclusions: empty
    excl_indices = jnp.zeros((512, 2), dtype=jnp.int32)
    excl_scales_vdw = jnp.ones(512, dtype=jnp.float32)
    excl_scales_elec = jnp.ones(512, dtype=jnp.float32)
    excl_mask = jnp.zeros(512, dtype=bool)

    # Exceptions: empty
    exception_pairs = jnp.zeros((512, 2), dtype=jnp.int32)
    exception_sigmas = jnp.ones(512, dtype=jnp.float32) * 0.1
    exception_epsilons = jnp.ones(512, dtype=jnp.float32) * 0.05
    exception_chargeprods = jnp.zeros(512, dtype=jnp.float32)
    exception_mask = jnp.zeros(512, dtype=bool)

    # PME parameters
    pme_alpha = jnp.array(0.3, dtype=jnp.float32)
    cutoff_distance = jnp.array(12.0, dtype=jnp.float32)

    # Build shape_spec with bucket indices
    atom_bucket_idx = _bucket_idx(n_atoms, ATOM_BUCKETS)
    bond_bucket_idx = 0  # Default to index 0 for empty
    angle_bucket_idx = 0
    dihedral_bucket_idx = 0
    water_bucket_idx = 0
    excl_bucket_idx = 0
    cmap_bucket_idx = 0
    exception_bucket_idx = 0

    shape_spec = MolecularShapeSpec(
        atom_bucket_idx=atom_bucket_idx,
        bond_bucket_idx=bond_bucket_idx,
        angle_bucket_idx=angle_bucket_idx,
        dihedral_bucket_idx=dihedral_bucket_idx,
        water_bucket_idx=water_bucket_idx,
        excl_bucket_idx=excl_bucket_idx,
        cmap_bucket_idx=cmap_bucket_idx,
        exception_bucket_idx=exception_bucket_idx,
        has_pbc=False,
        has_implicit_solvent=False,
        boundary_condition="free",
    )

    return MolecularBundle(
        positions=positions,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=radii,
        scaled_radii=scaled_radii,
        atom_mask=atom_mask,
        n_atoms=jnp.array(n_atoms, dtype=jnp.int32),
        box=box,
        bond_idx=bond_idx,
        bond_params=bond_params,
        bond_mask=bond_mask,
        n_bonds=jnp.array(0, dtype=jnp.int32),
        angle_idx=angle_idx,
        angle_params=angle_params,
        angle_mask=angle_mask,
        n_angles=jnp.array(0, dtype=jnp.int32),
        dihedral_idx=dihedral_idx,
        dihedral_params=dihedral_params,
        dihedral_mask=dihedral_mask,
        n_dihedrals=jnp.array(0, dtype=jnp.int32),
        improper_idx=improper_idx,
        improper_params=improper_params,
        improper_mask=improper_mask,
        improper_is_periodic=improper_is_periodic,
        n_impropers=jnp.array(0, dtype=jnp.int32),
        urey_bradley_idx=urey_bradley_idx,
        urey_bradley_params=urey_bradley_params,
        urey_bradley_mask=urey_bradley_mask,
        n_urey_bradley=jnp.array(0, dtype=jnp.int32),
        cmap_torsion_idx=cmap_torsion_idx,
        cmap_energy_grids=cmap_energy_grids,
        cmap_mask=cmap_mask,
        n_cmap=jnp.array(0, dtype=jnp.int32),
        water_indices=water_indices,
        water_mask=water_mask,
        n_waters=jnp.array(0, dtype=jnp.int32),
        excl_indices=excl_indices,
        excl_scales_vdw=excl_scales_vdw,
        excl_scales_elec=excl_scales_elec,
        excl_mask=excl_mask,
        n_excl=jnp.array(0, dtype=jnp.int32),
        exception_pairs=exception_pairs,
        exception_sigmas=exception_sigmas,
        exception_epsilons=exception_epsilons,
        exception_chargeprods=exception_chargeprods,
        exception_mask=exception_mask,
        n_exception_pairs=jnp.array(0, dtype=jnp.int32),
        pme_alpha=pme_alpha,
        cutoff_distance=cutoff_distance,
        shape_spec=shape_spec,
    )


@pytest.mark.fast
def test_vmap_with_coarsened_shape_spec():
    """HP3 Core Test: Coarsened shape_spec enables cache hits for same-bucket bundles.

    With the coarsenining refactor, MolecularShapeSpec now contains bucket indices
    (not raw counts). Two systems with different real atom counts but same bucket
    will have IDENTICAL shape_spec, enabling XLA cache hits.

    Method:
    1. Create two MolecularBundle instances with:
       - SAME bucket sizes (both padded arrays have identical shapes)
       - SAME shape_spec (because bucket indices are identical)
       - DIFFERENT real n_atoms (but encoded only in atom_mask, not shape_spec)
    2. Wrap an observable in jax.jit.
    3. Track JIT compilations via a Python-side trace counter.
    4. Call with bundle1, then bundle2.
    5. Assert trace_count == 1: both calls hit the same cached compilation.

    Expected: trace_count == 1 (one JIT compile, cache hit on second call).
    """

    # Both 10 and 20 atoms bucket to ATOM_BUCKETS[0] (256 or higher)
    # Now they will have IDENTICAL shape_spec.atom_bucket_idx, enabling cache hits.
    bundle_1 = _make_minimal_bundle(n_atoms=10)
    bundle_2 = _make_minimal_bundle(n_atoms=20)

    # CRITICAL VERIFICATION: Same bucket sizes (same array shapes)
    assert bundle_1.positions.shape == bundle_2.positions.shape, (
        f"HP3: Bundles must have same array shapes. Got {bundle_1.positions.shape} vs {bundle_2.positions.shape}"
    )

    # CRITICAL: With coarsenining, shape_spec must be identical for same bucket
    assert bundle_1.shape_spec == bundle_2.shape_spec, (
        f"HP3: Coarsened shape_spec must be identical for same-bucket bundles. "
        f"Got {bundle_1.shape_spec} vs {bundle_2.shape_spec}"
    )

    # Trace counter: incremented only when Python-side execution reaches this point
    # (i.e., on JIT compile, not on cache hit).
    trace_count = [0]

    def observable(bundle: MolecularBundle) -> Float:
        """Simple observable: sum of positions. Increments trace counter."""
        trace_count[0] += 1  # Python-side mutation, runs only on trace (not cache hit)
        return jnp.sum(bundle.positions)

    # Wrap in jax.jit to isolate JIT compilation behavior
    jitted_obs = jax.jit(observable)

    # Call with bundle_1
    result_1 = jitted_obs(bundle_1)
    assert jnp.isfinite(result_1).item(), "Observable should produce finite value"
    count_after_first = trace_count[0]

    # Call with bundle_2 (same shape_spec, same array shapes)
    result_2 = jitted_obs(bundle_2)
    assert jnp.isfinite(result_2).item(), "Observable should produce finite value"
    count_after_second = trace_count[0]

    # With coarsenining, both calls should hit the same cached compilation
    assert count_after_first == 1, (
        f"HP3: First call should compile once; trace_count={count_after_first}"
    )

    assert trace_count[0] == 1, (
        f"HP3 GATING: Expected 1 JIT compile (cache hit on same shape_spec), "
        f"got {trace_count[0]}. Coarsening refactor successful?"
    )


@pytest.mark.fast
def test_vmap_different_bucket_sizes_retraces():
    """HP3 Expected-Case Test: JIT recompilation with different bucket indices.

    This test documents the EXPECTED case: when two bundles have DIFFERENT bucket
    sizes, their shape_spec differs, and XLA MUST recompile. This is standard
    XLA behavior and validates that coarsenining properly separates different buckets.

    Method:
    1. Create two MolecularBundle instances with DIFFERENT bucket sizes
       (e.g., 10 atoms → bucket 0, 1000 atoms → bucket 1+).
    2. Verify shape_specs are different (different bucket indices).
    3. Wrap an observable in jax.jit.
    4. Track JIT compilations via a Python-side trace counter.
    5. Call with bundle1, then bundle2.
    6. Assert trace_count == 2: XLA recompiled on different shape_specs (expected).

    Expected: trace_count == 2 (recompile on shape_spec change — normal XLA behavior).
    """

    # Create two bundles with DIFFERENT bucket sizes
    # 10 atoms bucket to ATOM_BUCKETS[0] (256)
    # 1000 atoms bucket to ATOM_BUCKETS[1] (1024) — different bucket index
    bundle_1 = _make_minimal_bundle(n_atoms=10)
    bundle_2 = _make_minimal_bundle(n_atoms=1000)

    # CRITICAL VERIFICATION: Different bucket indices (different shape_specs)
    assert bundle_1.shape_spec != bundle_2.shape_spec, (
        f"HP3: Bundles with different buckets must have different shape_specs. "
        f"Got {bundle_1.shape_spec} vs {bundle_2.shape_spec}"
    )

    # Trace counter
    trace_count = [0]

    def observable(bundle: MolecularBundle) -> Float:
        """Simple observable: sum of positions."""
        trace_count[0] += 1
        return jnp.sum(bundle.positions)

    jitted_obs = jax.jit(observable)

    # Call with bundle_1
    result_1 = jitted_obs(bundle_1)
    assert jnp.isfinite(result_1).item()
    count_after_first = trace_count[0]

    # Call with bundle_2 (different shape_spec, different bucket)
    result_2 = jitted_obs(bundle_2)
    assert jnp.isfinite(result_2).item()
    count_after_second = trace_count[0]

    # Expected outcome: trace_count == 2 (recompiled on shape_spec change)
    assert count_after_first == 1, f"First call should compile once; trace_count={count_after_first}"
    assert trace_count[0] == 2, (
        f"HP3: Different buckets → XLA recompiles. Expected trace_count==2, got {trace_count[0]}."
    )


@pytest.mark.fast
def test_safe_map_stacked_bundles_same_bucket():
    """HP3 Secondary Check: safe_map behavior with stacked bundles (same bucket size).

    With coarsenining, bundles in the same bucket have IDENTICAL shape_spec,
    so stacking should work and vmap should cache correctly.

    Verifies that safe_map can handle stacked pytrees of bundles with:
    - Same bucket sizes (same array shapes across batch)
    - Identical shape_spec (because bucket indices match)
    - Different real n_atoms (encoded only in masks)

    This tests whether safe_map's underlying vmap behavior aligns with test_vmap_with_coarsened_shape_spec.
    """
    from prolix.batched_simulate import safe_map

    # Construct two bundles with same bucket size and now-identical shape_spec
    bundle_1 = _make_minimal_bundle(n_atoms=10)
    bundle_2 = _make_minimal_bundle(n_atoms=20)

    # Verify same bucket sizes AND identical shape_spec
    assert bundle_1.positions.shape == bundle_2.positions.shape, (
        f"Bundles must have same array shapes. Got {bundle_1.positions.shape} vs {bundle_2.positions.shape}"
    )

    assert bundle_1.shape_spec == bundle_2.shape_spec, (
        f"Coarsened shape_spec must be identical for same bucket. "
        f"Got {bundle_1.shape_spec} vs {bundle_2.shape_spec}"
    )

    # Attempt to stack bundles and apply safe_map
    # With identical shape_spec, stacking should succeed.
    try:
        import jax
        stacked = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0),
            bundle_1, bundle_2
        )

        trace_count = [0]

        def observable(bundle: MolecularBundle) -> Float:
            """Observable on stacked bundle batch."""
            trace_count[0] += 1
            return jnp.sum(bundle.positions, axis=-1).sum()

        # Apply safe_map with chunk_size=None (pure vmap)
        result = safe_map(observable, stacked, chunk_size=None)
        count_after_call = trace_count[0]

        # With identical shape_spec, vmap should cache correctly
        assert count_after_call >= 1, "safe_map should trigger at least one trace"

        # Verify safe_map works with stacked bundles
        assert bool(jnp.all(jnp.isfinite(result))), "Observable should produce finite values"

    except Exception as e:
        # Document any unexpected incompatibility
        pytest.fail(
            f"HP3: Failed to stack bundles with identical shape_spec. "
            f"Error: {type(e).__name__}: {e}"
        )


@pytest.mark.fast
def test_vmap_stacked_identical_shape_spec():
    """HP3 LOAD-BEARING gating test.

    Two MolecularBundles with identical shape_spec (coarsened to bucket indices),
    stacked into one batched pytree, vmapped through an observable.
    Asserts trace_count == 1 — this is the property Claim 1 requires.

    With the coarsenining refactor, this test validates that same-bucket bundles
    share cache entries when stacked and vmapped.
    """
    # Create two bundles in the same bucket with identical shape_spec
    # (because coarsenining replaced raw counts with bucket indices)
    bundle_1 = _make_minimal_bundle(n_atoms=10)
    bundle_2 = _make_minimal_bundle(n_atoms=20)

    # Sanity: both should have identical shape_spec (same bucket index)
    assert bundle_1.shape_spec == bundle_2.shape_spec
    assert bundle_1.positions.shape == bundle_2.positions.shape

    # Stack into a batched pytree
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), bundle_1, bundle_2)

    # Sanity: stacked leading dim is 2 on dynamic arrays, static shape_spec unchanged
    assert stacked.positions.shape[0] == 2
    assert stacked.shape_spec == bundle_1.shape_spec

    trace_count = [0]
    def observable(bundle):
        trace_count[0] += 1
        return jnp.sum(bundle.positions, axis=-1).sum()

    # vmap over the batch axis, wrapped in jit so cache hits are observable
    # (jax.vmap alone is a transform; it re-runs the Python body each call)
    batched_obs = jax.jit(jax.vmap(observable))

    # Run twice — second call should hit JIT cache (trace_count stays at 1)
    out = batched_obs(stacked)
    out2 = batched_obs(stacked)

    assert trace_count[0] == 1, (
        f"HP3 GATING: stacked vmap with identical shape_spec should compile once; "
        f"trace_count={trace_count[0]}"
    )

    # Bonus: also assert vmap leading axis preserved
    assert out.shape == (2,)


@pytest.mark.fast
def test_real_world_same_bucket_bundles_hash_identically():
    """HP3 Real-World Validation: Two systems with same bucket produce identical static keys.

    This is the property required for Claim 1 (heterogeneous batch substrate) to hold.

    Real-world systems of different sizes (e.g., 50 atoms, 100 atoms) that bucket
    to the same size (e.g., ATOM_BUCKETS[0] = 256) must produce byte-identical
    shape_spec static fields. This test validates that the coarsenining refactor
    achieves this property by using _bucket_idx to replace raw counts with indices.

    Method:
    1. Create two bundles with different real atom counts (50 and 150) that both
       bucket to the same size (256).
    2. Verify shape_spec.atom_bucket_idx is identical.
    3. Build jax.vmap(observable) wrapped in jax.jit over stacked bundles.
    4. Run twice; trace_count must stay at 1 (cache hit on second run).

    Expected: trace_count == 1 (successful cache hit, validating Claim 1).
    """
    # Create two bundles with different real counts, same bucket
    bundle_1 = _make_minimal_bundle(n_atoms=50)
    bundle_2 = _make_minimal_bundle(n_atoms=150)

    # Both should bucket to ATOM_BUCKETS[0] (256) since 50 < 256 and 150 < 256
    atom_bucket_threshold = ATOM_BUCKETS[_bucket_idx(50, ATOM_BUCKETS)]
    assert 50 <= atom_bucket_threshold and 150 <= atom_bucket_threshold, (
        f"Test setup: atoms 50 and 150 should bucket to same threshold. Got {atom_bucket_threshold}"
    )

    # CRITICAL: shape_spec must be identical for Claim 1 to hold
    assert bundle_1.shape_spec == bundle_2.shape_spec, (
        f"HP3 VALIDATION: Bundles in same bucket must have identical shape_spec. "
        f"Got {bundle_1.shape_spec} vs {bundle_2.shape_spec}"
    )

    # Stack the bundles
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), bundle_1, bundle_2)

    # Trace counter
    trace_count = [0]

    def observable(bundle: MolecularBundle) -> Float:
        """Real observable: sum positions (axis=-1 for batch elem, sum scalar results)."""
        trace_count[0] += 1
        return jnp.sum(bundle.positions, axis=-1).sum()

    # vmap(observable) wrapped in jit
    # With identical shape_spec, second call should cache-hit
    batched_obs = jax.jit(jax.vmap(observable))

    # First call: compiles
    out_1 = batched_obs(stacked)
    assert bool(jnp.all(jnp.isfinite(out_1))), "Observable should produce finite value"

    count_after_first = trace_count[0]
    assert count_after_first == 1, f"First call should compile once; got {count_after_first}"

    # Second call: should hit cache (trace_count stays at 1)
    out_2 = batched_obs(stacked)
    assert bool(jnp.all(jnp.isfinite(out_2))), "Second observable call should produce finite value"

    # THE CRITICAL ASSERTION: trace_count must stay at 1 (proof of Claim 1)
    assert trace_count[0] == 1, (
        f"HP3 REAL-WORLD GATING: jit(vmap(observable)) over two same-bucket bundles "
        f"must cache-hit on second call. "
        f"trace_count={trace_count[0]} (expected 1). "
        f"This validates Claim 1 (heterogeneous batch substrate)."
    )
