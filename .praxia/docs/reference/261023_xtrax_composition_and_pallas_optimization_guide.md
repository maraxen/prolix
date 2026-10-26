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


---

## Part 5: Post-Merge Verification, Benchmarking & Remote-Run Lessons (2026-10-24)

1. **Performance reality check — where Pallas pays in this repo**
   - `pallas_nonbonded_nl` (fused O(N*K) LJ + Coulomb/PME-direct): 0.67x at 512 atoms,
     1.2x at 2k, 2.1x at 8k, 3.2x at 33k, **3.8x at 131k** on an engaging GPU, float32
     round-off parity. XLA wins below ~1k atoms (launch-bound).
   - `use_pallas_pme` spread/gather: only **0.93–1.21x**, bit-identical energy. PME is FFT-
     and scatter-dominated in XLA already. Do **not** spend further effort on PME Pallas;
     target neighbor-list-side O(N*K) work. Full tables:
     `.praxia/docs/reference/261024_pallas_vs_xla_scaling_results.md`.
   - Always benchmark before expanding kernel coverage. Harness:
     `scripts/benchmarks/pallas_vs_baseline.py --kernel nonbonded|pme|nl_build`.
   - Neighbor-list rebuild (jax_md cell list) costs ~2x one fused nonbonded step at >= 8k
     atoms, but amortized over 10–50 steps a zero-cost builder saves only ~16% / <4%.
     **No-go on a Pallas NL builder**; tune skin (`dr_threshold`) / `capacity_multiplier`
     instead. Time a *real* rebuild (move atoms past `dr_threshold/2`); `update()` at
     unchanged positions is the no-op branch.
2. **Benchmark hygiene**
   - Use a physically spaced configuration (jittered lattice). Uniform-random placement makes
     overlapping atoms, ~1e13 energies, and meaningless float32 "parity".
   - `device.memory_stats()["peak_bytes_in_use"]` is a process-lifetime high-water mark; it
     cannot attribute memory to one of two paths in the same process.
   - The titanix GPUs have no Triton: Pallas there falls back to interpret mode (CPU jaxlib
     too without `--extra cuda`). Take Pallas GPU numbers only from engaging.
3. **Merge-conflict aftermath: always test the *test* side too.**
   After merging a branch that migrated `SafeMap`->`ChunkedMap` (xtrax 0.4.1b2) with one that
   made `ChunkedMap` accept ragged remainders, two assertions went stale: an error message that
   hardcoded the class name (fix: `type(strategy).__name__`), and a `rejects_remainder` test
   (replaced with `accepts_remainder`; xtrax `ChunkedMap` handles 455/32 correctly).
4. **Sync/harness traps (not code bugs)**
   - `myxcel.toml` `excludes = ["bench/", "profiling/"]` is unanchored, so it also dropped
     `tests/bench` and `tests/profiling` on remote pushes -> 11 `tests/api` collection errors.
     Root-anchor with a leading `/`.
   - Some tests read a sibling `../proxide` checkout (force-field/structure files); they error on
     titanix (no sibling repo). That is environment, not regression (21 errors + 3 dhfr-bench).
   - `JAX_PLATFORMS=cuda` alone breaks `jax.io_callback` tests (`test_b1_infer`); use
     `JAX_PLATFORMS=cuda,cpu`.
   - Engaging node `/tmp` can fill (`No space left on device` writing PTX): set
     `TMPDIR=$PWD/outputs/tmp`.
   - pytest `--timeout` default is 900 s/test; a "wide" run should deselect
     `slow/integration/openmm/kups/wasm/benchmark` and print a compact failure summary in the
     job (`-rfE --tb=line`, then `grep '^(FAILED|ERROR)'`) because `job_logs` returns only a
     500-line tail.
5. **Baseline-before-blame.** Two failures (`test_v1_solvated_ake_1k_parity_vs_settle_langevin`
   RMSD 0.27 A on CPU, `test_v4_hlo_hetero_b4_export_single_compile` 1 scan vs 4) fail
   identically on the pre-merge commit `ec34653`. Check a baseline with
   `git worktree add <path> <sha>` + `myxcel push` (auto-detects the worktree), run with the
   main venv via `PYTHONPATH=<wt>/src`, before attributing failures to a PR.
6. **Tooling.** The myxcel MCP server's `push_project`/`list_presets` can fail with
   `Built-in tier registry not found` (stale install); the `myxcel` CLI
   (`push|submit-job|cancel-job ... --apply --yes --json`) works. Never use mounts or raw ssh
   to engaging; titanix ssh is fine. Agent permission rules live in
   `~/.gemini/antigravity-cli/settings.json` (`permissions.allow` regex rules); the agent cannot
   write that file.
7. **Phase 3 Scaling Proofs & kUPS Benchmark Lessons (2026-10-26)**
   - **Per-process isolated memory measurement (`--isolate-memory`, `--kernel mem_scaling`)**:
     Spawn a fresh Python subprocess per `(path, N)` with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
     and `XLA_PYTHON_CLIENT_ALLOCATOR=platform`. Un-tiled `dense_xla` $O(N^2)$ blows up quadratically
     (`275.3 MB` at 8k $\to$ `1,100.1 MB` at 16k), whereas `pallas_flash_explicit_nonbonded` stays
     flat at `67.1 MB` up to 32k (`2.04x` lower than `flash_xla`'s `136.6 MB`), and `pallas_nonbonded_nl`
     uses `536.9 MB` (`263.1 MB` activations) vs `nl_xla`'s `555.0 MB` (`281.2 MB` activations) at 131k atoms.
   - **Dense $O(N^2)$ `pallas_flash_explicit_nonbonded` vs XLA `lax.scan` (`--kernel flash`)**:
     `pallas_flash_explicit_nonbonded` saves $2\times$ peak memory at 32k, but XLA's `lax.scan` over
     $(N, 256)$ tiles is $3.3\times\text{–}6.0\times$ faster because the $(32, 32)$ Pallas kernel executes
     $N/32$ sequential inner loop iterations per program. Always prefer $O(N \cdot K)$ `pallas_nonbonded_nl`
     ($9\times$ faster than Flash XLA and $54\times$ faster than Flash Pallas at 32k).
   - **NL skin & `capacity_multiplier` tuning (`--kernel nl_skin`)**:
     `capacity_multiplier = 1.15` + `dr_threshold = 0.5–1.0` Å minimizes net amortized step time
     (`2.37 ms` at 32k on Blackwell); larger skins (`1.5–2.0` Å) inflate $(r_c + r_{\text{skin}})^3$ and
     padded $K$ ($432 \to 656$), costing more on every step than they save in rebuild frequency.
   - **Cross-GPU generation scaling (`sm_120` RTX PRO 6000 Blackwell vs `sm_89` L40S vs `sm_90` H200)**:
     `pallas_nonbonded_nl` beats XLA across all three GPU generations for $N \ge 8,192$: up to **`4.23x`**
     on RTX PRO 6000 (`sm_120`), **`2.59x`** on L40S (`sm_89`), and **`1.39x`** on H200 (`sm_90`).
     GDDR6/GDDR7 architectures (`sm_89`, `sm_120`) see the largest relative speedup because Pallas
     eliminates global-memory intermediate traffic that H200's `4.8 TB/s` HBM3e partially absorbs.
   - **Prolix vs kUPS 3-tier parity & propagator throughput (`scripts/benchmarks/kups_benchmark.py`)**:
     - Always register custom propagator containers (`EnergyForceBundle`, `SettleLangevinPotentialStepMetrics`)
       with `@jax.tree_util.register_dataclass` so they can cross `jax.jit` and `jax.lax.scan` boundaries.
     - With exact AKMA $\leftrightarrow$ kUPS (eV, Å, amu) constants (`KUPS_FEMTOSECOND = 0.09822694788464063`),
       float64 single-step and harmonic/nonbonded parity is at machine precision (`rel_dE <= 1.96e-16`),
       10k-step BAOAB/CSVR trajectories agree to `<= 1.14e-13` K, `prolix_bundle_xla` has zero adapter overhead
       vs `kups_cached_xla` (`0.987–1.005x`), and `prolix_bundle_pallas` delivers **`4.08x`** (`82.5 ns/day`
       vs `20.2 ns/day`) on RTX PRO 6000, **`2.63x`** on L40S, and **`1.30x`** on H200.
