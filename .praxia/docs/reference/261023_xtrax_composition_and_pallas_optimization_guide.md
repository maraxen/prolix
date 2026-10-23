# Persistent Reference Guide: Post-Equinox `xtrax` Composition & Pallas Kernel Optimization in Prolix

**Date**: 2026-10-23  
**Task ID**: `261023_pallas_xtrax_optimization`  
**Branch**: `feat/pallas-xtrax-optimization-261023`  
**Related Spec**: [`docs/plans/2026-10-23-pallas-xtrax-optimization-spec.md`](file:///home/marielle/projects/prolix/docs/plans/2026-10-23-pallas-xtrax-optimization-spec.md)

---

## Part 1: Updated `xtrax` Composition Skill Guidance (Post-Equinox Migration)

### 1.1 `@jax.tree_util.register_dataclass` Replaces `eqx.Module`
In `xtrax >= 0.4.1b3` (branch `equinox-decouple-pallas-261023`), `equinox` has been completely removed from `pyproject.toml` and `src/xtrax/`. All PyTree containers use standard Python `@dataclass(frozen=True)` decorated with `@jax.tree_util.register_dataclass`:
- **Dynamic array fields** (traced by `jit`, `vmap`, `grad`, `scan`): declared as regular dataclass fields without metadata.
- **Static configuration fields** (compile-time constants, grid dimensions, block sizes, booleans, callables, strings): declared with `field(default=..., metadata={"static": True})`.
- **Forbidden Patterns**:
  - Do **not** subclass `eqx.Module` or use `eqx.field(static=True)`.
  - Do **not** use `eqx.filter_jit` or `eqx.filter_grad`. Because static fields are marked via `metadata={"static": True}` on `@jax.tree_util.register_dataclass`, standard `jax.jit` and `jax.grad(..., argnums=...)` natively partition static metadata from dynamic array leaves without runtime tree filtering overhead.

```python
from dataclasses import dataclass, field
import jax

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PallasKernelConfig:
    block_n: int = field(default=32, metadata={"static": True})
    block_k: int = field(default=32, metadata={"static": True})
    use_pallas: bool = field(default=False, metadata={"static": True})
    interpret: bool = field(default=False, metadata={"static": True})
```

### 1.2 `ChunkedMap` Replaces `SafeMap` (Remainder-Chunk Support)
In `xtrax >= 0.4.1b3`, `SafeMap` is renamed to `ChunkedMap` (with `SafeMap = ChunkedMap` exported as a backward-compatibility alias), and `ChunkedMap` supports remainder chunks when axis cardinality `N` is not divisible by `batch_size`.
- **Compatibility Pattern across `xtrax 0.4.0a6` and `0.4.1b3+`**:
  ```python
  try:
      from xtrax import ChunkedMap, SafeMap, Vmap
  except ImportError:
      from xtrax import SafeMap, Vmap
      ChunkedMap = SafeMap
  ```
- Always check `isinstance(strategy, (SafeMap, ChunkedMap))` when inspecting `xtrax` dispatch strategies.
- Only enforce `n % batch_size == 0` when running against legacy `SafeMap` (`type(strategy).__name__ == "SafeMap"`).

### 1.3 Pure-Array Kernel Boundary Between `xtrax` Dispatch and Pallas
Outer batching (`EnsemblePlan`, `dispatch_n_mols`, `dispatch_n_atoms`, `dispatch_n_waters`) belongs to `xtrax` (`Vmap` / `ChunkedMap` / `Scan`). Inner tile/stencil execution belongs to `jax.experimental.pallas`.
- **Rule**: Never pass a `PaddedSystem` or `MolecularBundle` PyTree into `pl.pallas_call`. Unpack the PyTree into flat, primitive `jax.Array` tensors (`positions`, `charges`, `sigmas`, `epsilons`, `atom_mask`, `neighbor_idx`, `excl_indices`, `excl_scales_vdw`, `excl_scales_elec`, `box`) **before** entering the `@jax.custom_vjp` / `@jax.custom_batching.custom_vmap` Pallas wrapper.
- **Export Context Guardrail (`prolix.export.export_context`)**:
  `jax.export` to StableHLO (used for IREE WASM browser deployment) cannot lower `pallas_call` custom calls onto WebAssembly/CPU targets. Wrap `export_energy_fn` and `export_langevin_step` in `with export_context():`, and have `resolve_pallas_mode()` / `resolve_use_pallas_pme()` unconditionally return `(False, False)` when `in_export_context()` is `True`.

---

## Part 2: `pallas-optimization` Skill Guidance & Compiler Rules

### 2.1 Triton Hardware & Block-Dimension Rules
1. **Compute Capability Gate (`sm_80+`)**:
   Triton GPU compilation requires NVIDIA Ampere (`sm_80`) or newer. On pre-Ampere GPUs (e.g., TITAN RTX `sm_75` on `titanix`) or CPU backends, `jax.default_backend() == "gpu"` may be `True` while Triton compilation fails. Always gate `interpret` via:
   ```python
   def _triton_gpu_available() -> bool:
       if jax.default_backend() != "gpu":
           return False
       try:
           dev = jax.devices("gpu")[0]
           cc = getattr(dev, "compute_capability", None)
           if cc is not None and float(cc) < 8.0:
               return False
       except Exception:
           return False
       return True
   ```
2. **Power-of-2 Block Dimensions (`>= 16`)**:
   Triton's `tl.dot` and vector load/store lowering require block dimensions to be powers of 2 (typically `BLOCK_N = 32`, `BLOCK_K = 32` or `64`). Never use `(1, 1)` scalar grids (`BlockSpec((1, 3), ...)`), which launch $N$ scalar threadblocks with 1 active thread per warp (3.1% SIMD utilization) and trigger `NDIndexer` / non-power-of-2 shape errors.

### 2.2 1D `Ref` Indexing Invariant (`NDIndexer` & Triton Gather/Scatter)
Both Pallas `interpret=True` (`NDIndexer.validate_indices` in `jax/_src/pallas/core.py`) and the Pallas Triton lowering enforce strict rules on array `Ref` indexing:
- **Rule 1 (No Mixed Integer-Array + Slice Indexing on 2D Refs)**:
  Indexing a 2D `Ref` of shape `(N, 3)` with a 1D integer vector `idx` and a `: ` slice (`pos_ref[idx, :]` or `pl.load(pos_ref, (idx, slice(None)))`) raises `ValueError` in `NDIndexer` or fails in Triton pointer arithmetic.
  - **Solution**: Split `(N, 3)` coordinates and forces at the Python wrapper boundary into three contiguous 1D `Ref`s (`pos_x, pos_y, pos_z` and `fx, fy, fz` of shape `(N_pad,)`), or flatten 2D tables (`neighbor_idx`, `excl_indices`, `theta`, 3D PME mesh `Kx*Ky*Kz`) into 1D `Ref`s and compute 1D linear offsets `flat_offs = (offs_i[:, None] * K_pad + offs_k[None, :]).reshape(BLOCK_N * BLOCK_K)`.
- **Rule 2 (Every `pl.load` / `pl.store` / `plt.atomic_add` Uses a 1D Power-of-2 Vector on a 1D `Ref`)**:
  When every `Ref` is 1D and every index is a 1D power-of-2 vector (`(32,)`, `(1024,)`, or `(2048,)`), followed by `.reshape(32, 32)` or `.reshape(32, 64)` in registers, the kernel is 100% compatible with both `interpret=True` on CPU/pre-Ampere GPU and Triton on `sm_80+` GPUs.

### 2.3 Full JAX Function-Transformation Composability (`jit`, `grad`, `vmap`, `scan`)
`pl.pallas_call` does **not** automatically define a JVP/VJP rule or a general `vmap` batching rule when arbitrary inputs are batched. Every Pallas kernel in `prolix` must follow the **Two-Layer Wrapper Pattern**:
1. **Inner `@jax.custom_batching.custom_vmap` Wrapper**:
   Handles arbitrary `in_batched` combinations under `jax.vmap` (and `xtrax` `Vmap` / `ChunkedMap`) by broadcasting any unbatched tensor along the leading batch dimension `B` and executing `lax.map` over the single-system Pallas kernel.
2. **Outer `@functools.partial(jax.custom_vjp, nondiff_argnums=...)` Wrapper**:
   - **Forward-Force Pattern**: For nonbonded energy kernels (`pallas_nonbonded_nl`, `pallas_flash_explicit_nonbonded`), compute BOTH the scalar energy `E` and the `(N, 3)` Cartesian force vector `F = -dE/dr` in the **same** forward Pallas pass (`out_shape=(EnergyShape, FxShape, FyShape, FzShape)`), saving `forces` in the VJP residuals.
   - In the backward pass, position cotangents are returned in $O(N)$ without launching a second $O(N \cdot K)$ or $O(N^2)$ Pallas kernel: `grad_pos = -g * forces`.
   - Parameter cotangents (`grad_charges`, `grad_sigmas`, `grad_epsilons`, `grad_excl_scales`) are computed cleanly for `DifferentiableParams` force-field optimization.

---

## Part 3: Physics & Numerical Lessons Learned

1. **Never Zero Out `custom_vjp` Position or Parameter Gradients**:
   - Stubbing `_chunked_lj_bwd` or `_chunked_coulomb_bwd` with `zeros_like(r)` silently zeroes all forces whenever `jax.grad` differentiates through the energy function.
   - Dropping `dE_gb / d(born_radii)` in `_gb_coulomb_bwd` breaks the chain rule into `compute_born_radii` in `gb_energy_split_cutoff`, losing 30–50% of the physical Generalized Born force.
   - In symmetric dense sums $E = \frac{1}{2} \sum_{i,j} e_{ij}$, differentiating with respect to $\mathbf{r}_i$ contributes from both row $i$ and column $j=i$, canceling the $\frac{1}{2}$ factor so $\frac{\partial E}{\partial \mathbf{r}_i} = \sum_j \frac{\partial e_{ij}}{\partial r_{ij}} \frac{\mathbf{r}_i - \mathbf{r}_j}{r_{ij}}$ (no extra factor of 2!).
2. **Branchless Binary Search for Sorted Exclusion Lists**:
   - Because `map_exclusions_to_dense_padded` sorts excluded atom indices in ascending order and pads unused slots with `-1` (`int32`), mapping a `-1` sentinel to `0x7FFFFFFF` produces a strictly ascending array per atom. A 5-step unrolled binary search replaces a 32-step linear scan in `get_neighbor_exclusion_scales` (6.4× fewer loop iterations).
3. **Closed-Form $3\times 3$ Gram Matrix RATTLE & Race-Free Masked Scatters in `settle.py`**:
   - A water molecule has 3 holonomic distance constraints (O–H1, O–H2, H1–H2), so its Lagrange multipliers satisfy a $3\times 3$ symmetric positive-definite linear system $G \boldsymbol{\lambda} = -\mathbf{b}$ solved in 1 pass via Cramer's rule (`_solve_3x3_batch`), replacing a 10-iteration Gauss-Seidel loop.
   - Never use `jnp.concatenate([arr, arr[:1]], axis=0)` with redirected index `N` to discard padded water rows (`water_mask=False`), because `.at[idx].set(...)` has non-deterministic write order on GPU when padded rows share index `0` before redirection or when multiple scatter sites touch Atom 0. Always use additive delta scatters `.at[idx].add(jnp.where(mask, target - arr[idx], 0.0))` with zeroed deltas for masked rows.

---

## Part 4: JAX 0.9+ Pallas Triton Lowering Rules & Offline Verification

1. **Always Pass `compiler_params=plt.CompilerParams()` When `not interpret`**:
   - In JAX 0.9+, `config.jax_pallas_use_mosaic_gpu.value` defaults to `True`. Calling `pl.pallas_call(..., compiler_params=None)` on GPU dispatches to `mosaic_gpu_backend` (`RuntimeError: Failed to infer the output layout of the iota. Please apply plgpu.layout_cast...`).
   - Always pass `compiler_params=plt.CompilerParams() if not interpret else None` (where `from jax.experimental.pallas import triton as plt`) to explicitly target the Triton backend.
2. **Forbidden Primitives in Pallas Triton Lowering (and Exact Replacements)**:
   - **`jnp.stack` with $>2$ arguments**: Raises `NotImplementedError: Only 2-argument stack is supported in Triton`. Keep 1D vectors in a Python tuple `(w0, w1, w2, w3)` and combine via arithmetic (`w0[:, None] * (offset == 0) + ...`).
   - **`jnp.round`**: Raises `NotImplementedError: Unimplemented primitive in Pallas Triton lowering: round`. Replace minimum-image wrapping `jnp.round(dx / bx)` with `jnp.floor(dx / bx + 0.5)`.
   - **`jnp.any` / `reduce_or`**: Raises `NotImplementedError: Unimplemented primitive in Pallas Triton lowering: reduce_or`. Replace `jnp.any(matches, axis=2)` with `jnp.sum(matches.astype(jnp.int32), axis=2) > 0`.
   - **`jax.scipy.special.erf` / `erfc`**: Raises `NotImplementedError: Unimplemented primitive in Pallas Triton lowering: erf / erfc`. Use Abramowitz & Stegun 7.1.26 rational polynomial approximation (`_triton_erfc` / `_triton_erf`, max error $< 1.5\times 10^{-7}$ across $x \ge 0$) built from elementary arithmetic and `jnp.exp`.
   - **Unary negation `-x` (`lax.neg_p`)**: Triggers a bug in JAX 0.9+'s `jax._src.pallas.triton.lowering._minus` (`AttributeError: 'TypedNdArray' object has no attribute 'type'`). Use `jnp.asarray(-c, dtype=dtype)` for negative constants and binary subtraction `zero - x` (`lax.sub_p`) for array negation.
3. **Offline CUDA Triton PTX Verification on CPU / Older GPUs**:
   - You can verify full Triton IR lowering and `triton.compile` PTX generation on a CPU-only or `sm_75` machine in seconds without waiting for an Ampere/Hopper GPU by mocking `jax.local_devices` to raise `RuntimeError` (which falls back to `arch_name="9.0"` inside `pallas_call_lowering`) and calling:
     ```python
     jax.jit(fn).trace(*args).lower(lowering_platforms=("cuda",))
     ```

