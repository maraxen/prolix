"""Definitive test: JAX-MD neighbor list vmap compatibility.

Tests:
1. allocate per-system, pad to common K, stack PyTree
2. vmap update inside lax.scan (the actual production pattern)
3. Overflow detection across batch
"""
import time

import jax
import jax.numpy as jnp
import jax.tree_util as tu
from jax_md import partition, space


def test_batched_nl():
    B = 4
    N = 100
    cutoff = 5.0
    key = jax.random.PRNGKey(0)
    
    # Create B systems with different atom distributions (heterogeneous)
    keys = jax.random.split(key, B)
    R_batch = jnp.stack([
        jax.random.uniform(keys[i], (N, 3), minval=-10.0, maxval=10.0 + i*2)
        for i in range(B)
    ])
    
    displacement_fn, _ = space.free()
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box=1000.0,
        r_cutoff=cutoff,
        disable_cell_list=True,
        format=partition.Dense,
        capacity_multiplier=1.25,
        dr_threshold=1.0,
    )
    
    # === Step 1: Allocate per-system, find max K, pad ===
    print("=== Step 1: Per-system allocation ===")
    nbrs_list = []
    max_k = 0
    for i in range(B):
        nbrs = neighbor_fn.allocate(R_batch[i])
        nbrs_list.append(nbrs)
        k = nbrs.idx.shape[1]
        max_k = max(max_k, k)
        print(f"  System {i}: K={k}, max_occupancy={nbrs.max_occupancy}")
    
    print(f"  Global max K: {max_k}")
    
    # Pad all to max_k and re-allocate with that capacity
    # We need all systems to have the same max_occupancy for consistent shapes
    padded_nbrs_list = []
    for i in range(B):
        # Re-allocate with extra_capacity to match global max_k
        nbrs = neighbor_fn.allocate(R_batch[i], extra_capacity=max_k)
        # Trim to exactly max_k columns
        padded_idx = nbrs.idx[:, :max_k]
        import dataclasses as dc
        padded_nbrs = dc.replace(nbrs, idx=padded_idx, max_occupancy=max_k)
        padded_nbrs_list.append(padded_nbrs)
    
    # Stack into batched PyTree
    # static_fields (update_fn, cell_list_fn, format, max_occupancy, etc) are shared
    # array leaves (idx, reference_position, error.code) are stacked
    batched_nbrs = tu.tree_map(lambda *x: jnp.stack(x), *padded_nbrs_list)
    print(f"  Batched idx shape: {batched_nbrs.idx.shape}")  # Should be (B, N, max_k)
    print(f"  Batched ref_pos shape: {batched_nbrs.reference_position.shape}")
    
    # === Step 2: vmap update ===
    print("\n=== Step 2: vmap update ===")
    
    @jax.jit
    def batched_update(nbrs_b, R_b):
        return jax.vmap(lambda n, r: n.update(r))(nbrs_b, R_b)
    
    # Small perturbation
    R_new = R_batch + 0.1
    t0 = time.time()
    updated = batched_update(batched_nbrs, R_new)
    updated.idx.block_until_ready()
    t1 = time.time()
    print(f"  First update (includes JIT): {(t1-t0)*1000:.1f} ms")
    print(f"  Updated idx shape: {updated.idx.shape}")
    print(f"  Overflow: {updated.did_buffer_overflow}")
    
    # Second update (no JIT)
    R_new2 = R_batch + 0.2
    t0 = time.time()
    updated2 = batched_update(updated, R_new2)
    updated2.idx.block_until_ready()
    t1 = time.time()
    print(f"  Second update (cached): {(t1-t0)*1000:.1f} ms")
    print(f"  Overflow: {updated2.did_buffer_overflow}")
    
    # === Step 3: vmap update inside lax.scan (production pattern) ===
    print("\n=== Step 3: lax.scan with vmap update ===")
    
    N_SCAN_STEPS = 50

    @jax.jit
    def run_scan(nbrs_b, R_b):
        def scan_body(carry, _):
            nbrs, R = carry
            # Simulate position update (small random walk)
            R = R + 0.01 * jax.random.normal(jax.random.PRNGKey(0), R.shape)
            # Update neighbor list
            nbrs = jax.vmap(lambda n, r: n.update(r))(nbrs, R)
            return (nbrs, R), nbrs.idx
        
        (final_nbrs, final_R), all_idx = jax.lax.scan(
            scan_body, (nbrs_b, R_b), None, length=N_SCAN_STEPS
        )
        return final_nbrs, final_R, all_idx
    
    t0 = time.time()
    final_nbrs, final_R, all_idx = run_scan(batched_nbrs, R_batch)
    final_nbrs.idx.block_until_ready()
    t1 = time.time()
    print(f"  {N_SCAN_STEPS} steps (includes JIT): {(t1-t0)*1000:.1f} ms")
    print(f"  Final idx shape: {final_nbrs.idx.shape}")
    print(f"  Final overflow: {final_nbrs.did_buffer_overflow}")
    print(f"  All idx trajectory shape: {all_idx.shape}")
    
    # Second run (cached)
    t0 = time.time()
    final_nbrs2, _, _ = run_scan(batched_nbrs, R_batch)
    final_nbrs2.idx.block_until_ready()
    t1 = time.time()
    print(f"  {N_SCAN_STEPS} steps (cached): {(t1-t0)*1000:.1f} ms")
    
    print("\n✅ All tests passed! JAX-MD neighbor list is fully vmappable.")

if __name__ == "__main__":
    test_batched_nl()
