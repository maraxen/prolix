# GPU Optimization Strategies for Explicit Solvent

> **Status**: Research Notes — Evaluate during benchmarking phase
> **Date**: 2026-03-30
> **Context**: Performance optimizations targeting the short-range cell-list
> engine, PME charge spreading, and cuFFT execution. Each strategy should be
> benchmarked for both precision and speed impact.

---

## 1. Slaying the Short-Range Bottleneck: Dense "Grid-Shift" Strategy

### Problem

Scanning over a 27-cell Eulerian stencil sequentially via `jax.lax.scan` forces
XLA to emit a serial loop inside the kernel, resulting in high warp divergence
and destroying Tensor Core / vectorization potential.

### Solution: `jnp.roll` + Newton's 3rd Law (Half-Shell)

Because our explicit solvent maps to a dense, padded grid
`(B, Nx, Ny, Nz, M, 3)`, we can compute short-range interactions **without a
single `scatter_add` or lookup array** by leveraging `jnp.roll` and Newton's
3rd Law.

```python
def compute_half_shell_sr(cell_grid, cell_mask):
    F_total = jnp.zeros_like(cell_grid)

    # 13 positive 3D offsets (e.g., dx=1, dy=0, dz=0)
    for dx, dy, dz in POSITIVE_13_SHIFTS:
        # jnp.roll is functionally FREE; XLA fuses it directly
        # into the memory load index
        nbr_grid = jnp.roll(cell_grid, shift=(-dx, -dy, -dz), axis=(1, 2, 3))
        nbr_mask = jnp.roll(cell_mask, shift=(-dx, -dy, -dz), axis=(1, 2, 3))

        # Dense vmap pairwise distance (M x M) evaluated massively in parallel
        dr = cell_grid[..., :, None, :] - nbr_grid[..., None, :, :]

        # ... apply Minimum Image Convention to dr ...

        # Force ON central cells FROM neighbors
        F_forward = compute_dense_forces(dr, mask)
        F_total += jnp.sum(F_forward, axis=-2)

        # Force ON neighbors FROM central cells (Newton's 3rd Law)
        F_backward = jnp.sum(-F_forward, axis=-3)
        # Roll neighbor force backward to correctly accumulate it in place
        F_total += jnp.roll(F_backward, shift=(dx, dy, dz), axis=(1, 2, 3))

    return F_total
```

### Why This Works

- **`jnp.roll` is free**: XLA fuses roll directly into the memory load index
  calculation. No data movement occurs — only the address computation changes.
- **13 iterations instead of 27**: Newton's 3rd Law halves the stencil. Each
  shifted pair computes both `F_forward` and `-F_forward` simultaneously.
- **Dense tensor evaluations**: Each of the 13 iterations is a fully parallel
  `(B, Nx, Ny, Nz, M, M)` pairwise computation — maximum GPU utilization.
- **No `scatter_add`**: Force accumulation uses only `+=` and `jnp.roll`,
  eliminating all atomic contention.

### Impact

Replaces 27 sequential `lax.scan` iterations with 13 dense, batched tensor
evaluations. Eliminates `scatter_add` completely from the SR pipeline.

### Evaluation Criteria

- [ ] Benchmark vs. `lax.scan` stencil: wall time per step
- [ ] Verify force parity against reference implementation (< 1e-5 relative)
- [ ] Profile XLA HLO to confirm `jnp.roll` fusion
- [ ] Memory overhead of the `(B, Nx, Ny, Nz, M, M)` intermediates

---

## 2. Hardware Traps: Ghost Atoms & `scatter_add`

### 2a. The Ghost-Atom Autodiff & Contention Trap

The naive suggestion of placing ghost/padding atoms "at infinity" is a **fatal
trap** in JAX for two reasons:

#### The Autodiff NaN Trap

If you use the Minimum Image Convention:

```
dr = ∞ - L × round(∞/L) → NaN
```

Even with a `0.0` charge multiplier, this NaN will infect the `jax.vjp`
backward tape. The gradient is computed *before* the mask multiplication in
XLA's reverse pass, so masking *after* the computation is insufficient.

#### The Atomic Lock Trap

For PME grid scaling, `∞` fractional coordinates are clipped by JAX
`scatter_add` to grid boundary `[0, 0, 0]`. Because `N_ghost` is large,
thousands of threads will attempt an `atomicAdd` of `0.0` at the exact same
memory address, serializing your SMs.

#### The Fix

1. **Place ghost atoms at the origin `(0, 0, 0)`.**
2. **Spread their pseudo-coordinates out uniformly** during the PME charge
   spreading step to load-balance the `+0.0` atomic adds across all L2 cache
   banks.
3. **Strictly use `jnp.where(is_valid_pair, dr_sq, 1.0)` BEFORE any inversion
   or norm calculations** to safeguard the autodiff tape.

> **Note**: This aligns with our proven `sigma=1.0` sanitization pattern from
> the implicit solvent pipeline (see `explicit_solvent_architecture.md`
> §11.5).

---

### 2b. Spatial Sorting: Morton Coalescing for `scatter_add`

Before computing B-splines for PME charge spreading, spatially `jnp.argsort`
atom arrays by their flattened Eulerian 3D cell index.

#### Why This Helps

When atoms are spatially clustered in memory, adjacent threads in a CUDA warp
will scatter charges to identical or adjacent PME grid voxels. Ada/Blackwell
architecture intercepts this via **Warp-Synchronous Hardware Reductions**,
coalescing the sums in registers before hitting the atomic L2 cache locks.

#### Expected Impact

Routinely yields a **3×–5× throughput spike** on `scatter_add` operations.

#### Implementation

```python
def morton_sort_atoms(positions, box_size, grid_shape):
    """Sort atoms by Morton (Z-order) code for cache-coherent scatter."""
    # Compute 3D cell indices
    cell_idx = jnp.floor(positions / box_size * jnp.array(grid_shape)).astype(int)

    # Interleave bits: x, y, z → single Morton code
    morton = interleave_bits_3d(cell_idx[:, 0], cell_idx[:, 1], cell_idx[:, 2])

    # Sort all arrays by Morton code
    sort_order = jnp.argsort(morton)
    return sort_order
```

#### Evaluation Criteria

- [ ] Benchmark `scatter_add` throughput: sorted vs unsorted atoms
- [ ] Measure L2 cache hit rate via `ncu` profiler
- [ ] Verify that `jnp.argsort` cost is amortized (run every 20 steps)
- [ ] Test on system sizes: N=10k, 40k, 100k

---

## 3. Enforce Real-to-Complex (R2C) FFTs

### Problem

Using `jnp.fft.fftn` on purely real charge grids wastes half the computation
on symmetric complex conjugate pairs.

### Fix

**Always use `jnp.fft.rfftn` and `jnp.fft.irfftn`.**

This mathematically halves the Z-dimension of the grid
($N_z \to N_z/2 + 1$), dropping:

- VRAM allocations by ~50% for the k-space grid
- cuFFT execution latency by ~50%

### Critical: Grid Dimension Factorization

Ensure PME grid dimensions strictly factorize into **2, 3, 5, and 7** to
prevent cuFFT from falling back to Bluestein's $O(N^2)$ algorithm.

Good dimensions: 48, 50, 54, 56, 60, 64, 72, 80, 84, 90, 96, 100, 108, 112, ...

Bad dimensions: 51 (=3×17), 53 (prime), 59 (prime), 61 (prime), ...

```python
def next_good_fft_size(n, max_prime=7):
    """Find smallest m >= n whose prime factors are all <= max_prime."""
    while True:
        m = n
        for p in [2, 3, 5, 7]:
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1
```

### Evaluation Criteria

- [ ] Verify `rfftn` vs `fftn` energy parity (should be exact)
- [ ] Benchmark cuFFT latency: `rfftn` vs `fftn` at grid sizes 48³–100³
- [ ] Profile VRAM savings at B=500, B=1500

---

## Strategy Summary

| Strategy | Expected Impact | Risk | Phase |
|----------|----------------|------|-------|
| Grid-shift half-shell SR | Major: eliminates serial scan | Low (pure tensor ops) | Phase 1 |
| Ghost atom origin placement | Required: prevents NaN | None (correctness fix) | Phase 1 |
| Morton sort for scatter_add | 3–5× scatter throughput | Low (amortized cost) | Phase 2/6 |
| R2C FFTs (rfftn/irfftn) | ~50% FFT speedup + memory | None (mathematically exact) | Phase 2 |
| Grid size factorization | Prevents O(N²) fallback | None (precomputation) | Phase 2 |

---

## 4. Energy Drift Prevention Strategy

> **Decision (2026-03-30)**: Cell-relative coordinates first, int64 fallback.
> Avoid general `float64` mixed precision.

### The Problem

Float32 coordinate integration suffers from catastrophic cancellation:
`50.12345 + 0.00001 = 50.12345` (trailing digit lost). Over millions of
steps, this destroys symplectic reversibility and causes energy drift.

### Strategy A (Primary): Cell-Relative Coordinates

Since we're building a cell-list architecture, use the spatial cell as a
precision anchor:

1. **`cell_index: int32`** — discrete 3D grid cell (e.g., `[5, 2, 8]`).
   Stores the "presumptive bits" without using any floating-point mantissa.
2. **`local_offset: float32`** — atom's distance from cell center, bounded
   to `[-cell_size/2, +cell_size/2]`.

With 5 Å cells, offsets are bounded to `[-2.5, 2.5]`, giving float32 vast
sub-picometer precision for velocity updates. No bits wasted on large absolute
coordinates.

```python
# Inside lax.scan integrator:
new_offset = local_offset + v * dt  # High precision: small + small

# Check cell boundary crossing (every step, cheap)
crossed = jnp.abs(new_offset) > cell_size / 2
new_cell_index = cell_index + jnp.sign(new_offset).astype(int) * crossed
new_offset = new_offset - jnp.sign(new_offset) * cell_size * crossed
```

**Synergy**: This representation feeds directly into the cell-list SR engine
without conversion. Absolute positions are only reconstructed for PME charge
spreading and trajectory output.

### Strategy B (Fallback): Int64 Accumulation

If cell-relative coords prove too complex for the initial implementation:

1. Scale positions by `2^32`, store as **`int64`**.
2. For forces: subtract ints → cast small relative distance to `float32`.
3. Compute forces in `float32` (full speed).
4. Scale `v * dt` by `2^32`, cast to `int64`, add to positions.

`int64` addition is a **1-clock-cycle operation** on ML GPUs (unlike `float64`
math which is 16-32× slower). No precision loss, no `JAX_ENABLE_X64` needed.

### What NOT to Do

- **`JAX_ENABLE_X64=True` for general mixed precision**: Halves memory
  bandwidth, doubles cache footprint, disables Tensor Cores globally.
- **Kahan summation**: XLA compiles to Kahan-like compensated summation on
  the implicit solvent path (validated). Must verify the same applies for the
  explicit solvent computation graph — if it doesn't, add manual Kahan.

---

## References

- Eastman et al. (2017). OpenMM 7: Rapid development of high performance
  algorithms for molecular dynamics. *PLOS Comp. Bio.*, 13, e1005659.
  (Mixed precision, HMR, PME implementation details)
- NVIDIA cuFFT Documentation: Batched R2C transforms, supported radix sizes.
- Morton codes / Z-order curves: well-established spatial locality technique
  used in GROMACS, LAMMPS, and Anton for cache-coherent MD.
