# Xtrax Megakernel Composition Ledger & Lessons Learned

**Purpose**: Living engineering ledger capturing architectural patterns, JAX/Pallas/Triton pitfalls, `xtrax.op` / `VjpKernel` seam mechanics, and verification invariants discovered while wrapping Prolix's Pallas v1 kernels behind `xtrax.op` (PR #40 / PR #41) and building the Pallas v2 `pallas_bonded_all_fused` megakernel + O1–O5 optimizations (PR #42).
**Target Deliverable**: Distill these entries into a general-purpose `megakernel-composition` skill for `xtrax` to ship.

---

## Ledger Schema for Subagent & Main-Agent Entries

Every entry added to this ledger must record:
1. **ID & Title**: `MK-XX: <concise rule or pattern name>`
2. **Category**: `xtrax.op Seam` | `Grid & Work-Unit Dispatch` | `Buffer Aliasing & Atomics` | `Differentiation (JVP/VJP)` | `Batching (vmap)` | `Lowering & Triton Constraints` | `Verification & Negative Controls`
3. **Symptom / Failure Mode**: Exact error, silent numerical bug, or performance cliff that occurs if violated.
4. **Root Cause**: Why JAX / Pallas / Triton / `xtrax` behaves this way.
5. **Generalizable Rule for `xtrax` Skill**: The reusable pattern or contract any `xtrax` user should follow when composing multi-term Pallas megakernels.

---

## Recorded Lessons

### MK-01: Static Shape & Configuration Closure Across `xtrax.op`
- **Category**: `xtrax.op Seam`
- **Symptom / Failure Mode**: `ConcretizationTypeError` or Pallas grid/shape errors if grid dimensions (`grid_dims`), spline order (`order`), or boolean mode flags (`has_box`, `is_harmonic`, `include_ace`) are passed as positional arguments to an `xtrax.Op`.
- **Root Cause**: `xtrax.op(reference, kernels=..., vjp_kernels=...)` traces *all* positional arguments as JAX arrays (`jax.eval_shape` during `_check_signature`, `custom_jvp` / `custom_vjp` during execution). Pallas kernels require grid dimensions, block sizes, and structural branch flags as static Python `int`/`bool`/`float` values at trace time.
- **Generalizable Rule for `xtrax` Skill**: Always expose a factory `make_<name>_op(*static_configs, context=...) -> xtrax.Op` that closes over static Python configuration (`int`, `float`, `bool`, `tuple[int, ...]`) via `functools.partial`, leaving *only* dynamic array tensors as positional arguments to the `xtrax.Op`.

### MK-02: `xtrax.op` `kernels` (`custom_jvp`) vs `vjp_kernels` (`VjpKernel` / `custom_vjp`)
- **Category**: `xtrax.op Seam` / `Differentiation (JVP/VJP)`
- **Symptom / Failure Mode**:
  1. Calling `jax.grad` directly on a raw `pl.pallas_call` raises `AssertionError: assert env.grid_context is not None` in `_pallas_call_jvp_rule` because Pallas kernels do not implement a default JVP rule.
  2. When wrapped in `xtrax.op(ref, kernels=...)` (without `vjp_kernels`), `xtrax` wraps the kernel in a `custom_jvp` whose JVP rule calls `jax.jvp(reference, primals, tangents)`. Forward passes run the Pallas kernel; reverse-mode (`jax.grad`) transposes the reference JVP!
  3. When a Pallas kernel already computes analytical forces / cotangents in the forward pass (or has a dedicated backward Pallas kernel) to avoid XLA scatter-add / autodiff overhead in reverse mode, passing it only via `kernels=` would throw away the kernel's analytical forces during `jax.grad` and differentiate the slower XLA reference instead!
- **Root Cause**: `xtrax.op` provides two distinct mechanisms:
  - `kernels={"gpu": fn, "cpu": fn}`: Uses `custom_jvp` with `jax.jvp(reference)`. Ideal when the Pallas kernel *is* the backward pass (like `pme_gather` inside `_spme_bwd`) or only needs primal acceleration while reference autodiff is cheap.
  - `vjp_kernels={"gpu": VjpKernel(fwd, bwd), "cpu": VjpKernel(fwd, bwd)}`: Uses `custom_vjp` where `fwd(*args) -> (out, residuals)` runs the forward kernel (saving analytical forces in `residuals`), `bwd(residuals, ct)` uses those analytical forces for $O(1)$ position cotangents and analytical parameter cotangents, and **second-order derivatives** (`jax.grad(jax.grad(...))` / HVP) automatically fall back to `reference` via `xtrax`'s nested `custom_jvp` on `fwd_pair` and `bwd_apply`!
- **Generalizable Rule for `xtrax` Skill**:
  - Use `kernels=` for primal-only or already-backward kernels (e.g., `pme_gather`).
  - Use `vjp_kernels={"gpu": VjpKernel(fwd, bwd), "cpu": VjpKernel(fwd, bwd)}` for energy kernels whose Pallas forward pass computes `(energy, forces)` in one launch. This gives Pallas speed on both `energy(pos)` and `jax.grad(energy)(pos)` while preserving exact higher-order AD (`jax.jvp` / Hessian-vector products) via the plain-JAX `reference`.

### MK-03: Direct Batched `(B, ...)` Calls vs `jax.vmap` on `custom_vmap` Pallas Kernels
- **Category**: `Batching (vmap)`
- **Symptom / Failure Mode**: `ValueError: jnp.pad: pad_width with nd=3 has unsupported shape (2, 2)` when calling an `xtrax.Op` directly on pre-batched 3D inputs `(B, N, 3)` instead of `jax.vmap(op)(batched_inputs)`.
- **Root Cause**: `@jax.custom_batching.custom_vmap` only intercepts `jax.vmap(fn)` transformations; it does *not* intercept direct calls where the caller passes an already-batched `(B, N, 3)` tensor. Meanwhile, `xtrax.op._check_signature` runs `jax.eval_shape(kernel, *args)` on whatever shape `args` has.
- **Generalizable Rule for `xtrax` Skill**: Every Pallas kernel registered in `xtrax.op` should both:
  1. Define `@jax.custom_batching.custom_vmap` so `jax.vmap(op)` dispatches cleanly via `jax.lax.map`, **and**
  2. Check `if positions.ndim == 3:` at the entry point and dispatch via `jax.vmap` over the leading batch axis so direct batched calls `op(batched_pos, ...)` and `jax.vmap(op)(batched_pos, ...)` behave identically.

### MK-04: Module-Level Import Topology for `xtrax.op` Factories
- **Category**: `xtrax.op Seam`
- **Symptom / Failure Mode**: Cycle-dodging lazy import failure in `test_import_architecture.py` if `physics/module.py` lazily imports `physics/module_op.py` while `physics/module_op.py` imports helpers from `physics/module.py`.
- **Root Cause**: A function-level import `A -> B` combined with a top-level import `B -> A` is a cycle-dodging lazy import.
- **Generalizable Rule for `xtrax` Skill**: Either keep the plain-JAX reference in `A` (`physics/pme.py`) and have the optional Pallas backend `B` (`physics/pme_pallas.py`) define self-contained `make_<name>_op` factories (so `A -> B` is a one-way lazy Pallas backend edge), or place the `xtrax.op` factories at the top of the Pallas module / a dedicated higher-level op module that `A` does not import.

### MK-05: Strict Output Dtype Invariant Between `xtrax.Op` Kernel and Reference (`_check_signature`)
- **Category**: `xtrax.op Seam` / `Verification & Negative Controls`
- **Symptom / Failure Mode**: `xtrax.intent.errors.IntentError: op 'pallas_cmap_torsions': kernel for backend 'cpu' returns [((), dtype('float32'))] but the reference returns [((), dtype('float64'))]` during `op._check_signature`.
- **Root Cause**: Before executing a kernel or `VjpKernel`, `xtrax.op._check_signature` runs `jax.eval_shape` on both `vjp_kernels[backend].fwd(*args)[0]` and `reference(*args)` and compares `_structs(kernel_out) == _structs(ref_out)`. If a secondary input table (e.g., `cmap_coeffs` from host NumPy spline setup) has `float64` dtype while `positions` is `float32`, a Pallas kernel that casts tables to `positions.dtype` returns `float32`, whereas a plain-JAX reference that relies on standard JAX type promotion promotes `float32 * float64 -> float64`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Define a single canonical compute/output dtype (typically `dtype = primary_arg.dtype`, e.g., `positions.dtype`) in *both* the Pallas kernel and the `reference` function.
  2. Explicitly cast all auxiliary parameter tables and the final return value in `reference` via `.astype(dtype)` so `jax.eval_shape(reference, *args)` and `jax.eval_shape(kernel, *args)` have identical PyTree structure, shapes, and dtypes under both `jax_enable_x64=False` and `jax_enable_x64=True`.

### MK-06: `assert_kernel_parity` Signature Contract (`(kernel, reference, *args, rel=...)`)
- **Category**: `Verification & Negative Controls`
- **Symptom / Failure Mode**: `TypeError: assert_kernel_parity() got an unexpected keyword argument 'rtol'` or `TypeError` when passing a single `xtrax.Op` without a separate reference callable.
- **Root Cause**: `xtrax.testing.kernel_parity.assert_kernel_parity(kernel, reference, *args, rel=1e-5)` takes *both* `kernel` and `reference` as its first two positional arguments and uses a single relative tolerance parameter `rel: float = 1e-5` (not `rtol`/`atol`).
- **Generalizable Rule for `xtrax` Skill**: When testing an `xtrax.Op` factory `make_<name>_op`:
  1. Instantiate `op_k = make_<name>_op(..., context=Context(kernels=True, backend="cpu"))` and `op_r = make_<name>_op(..., context=Context(kernels=False))`.
  2. Verify selection (`assert op_k.select().primal == "kernel:cpu"`, `assert op_r.select().primal == "reference"`), execution (`assert_kernel_ran(op_k, *args)`), primal parity (`assert_kernel_parity(op_k, op_r, *args, rel=1e-4)`), and reverse-mode VJP parity (`jax.grad(op_k)(*args)` vs `jax.grad(op_r)(*args)`).

### MK-07: Disjoint CTA `pid` Range Dispatch in Multi-Term Fused Megakernels (`bra.uni`)
- **Category**: `Grid & Work-Unit Dispatch` / `Lowering & Triton Constraints`
- **Symptom / Failure Mode**: High register pressure (spilling to local memory) and 20+ serialized `plt.atomic_add` operations per CTA if a fused bonded megakernel evaluates all term families (bonds, angles, dihedrals, impropers, 1-4 exceptions, CMAP) sequentially inside every CTA (`pid < n_blocks_group`).
- **Root Cause**: Evaluating heterogeneous N-body stencils sequentially in the same CTA forces the Triton compiler to size the CTA's register allocation to the union/maximum of live temporaries across all branches and serializes atomic scatter-adds across term families.
- **Generalizable Rule for `xtrax` Skill**: Partition a multi-term megakernel across **non-overlapping 1D CTA `pid` ranges** $[O_g, O_{g+1})$ where $O_{g+1} = O_g + N_{\text{blk},g}$ and grid size is $(N_{\text{blk,total}},) = (\sum_g N_{\text{blk},g},)$:
  1. Because `pid = pl.program_id(0)` is scalar-uniform across all threads in a CTA, dispatching `pid` via nested `jax.lax.cond(pid < O_g, ...)` compiles to a uniform branch (`bra.uni` in PTX) with zero warp divergence.
  2. Each CTA executes *only one* term family from start to finish with local block index `local_pid = pid - O_g`, writes its scalar block energy to its dedicated slot `e_blk_ref[pid]` without atomic contention, and performs at most $k$ `plt.atomic_add` calls into the shared `(fx_pad, fy_pad, fz_pad)` force buffers.

### MK-08: Active-AD Parameter Cotangent Gating Across `custom_vjp(..., symbolic_zeros=True)` and `xtrax.op` `VjpKernel`
- **Category**: `xtrax.op Seam` / `Differentiation (JVP/VJP)`
- **Symptom / Failure Mode**:
  1. Unconditional execution of expensive parameter-gradient kernels/scans (`dE/dq`, `dE/d\sigma`, `dE/d\epsilon`) during standard MD steps where only `jax.grad(E)(positions)` is requested.
  2. Naive tracer inspection (`isinstance(x, ad.JVPTracer)`) silently returns `False` under `jax.jit(jax.grad(f, argnums=1))` and `jax.grad(jax.jit(f), argnums=1)` because modern JAX uses `LinearizeTracer` and `@jax.jit` wraps all arguments in `DynamicJaxprTracer` before linearization.
  3. **Fatal `xtrax.op` `VjpKernel` incompatibility if `_fwd` / `_bwd` naively assume `symbolic_zeros=True`**:
     - Inside `xtrax/kernel_seam.py`, `Op._build_vjp_form` registers `fn.defvjp(fn_fwd, fn_bwd)` with `symbolic_zeros=False` and wraps both `vk.fwd` (`fwd_pair`) and `vk.bwd` (`bwd_apply`) in `@jax.custom_jvp` (and `Op._check_signature` calls `jax.eval_shape(vk.fwd, *args)`).
     - Consequently, when `_fwd` is called via `VjpKernel`, its arguments are plain JAX arrays/tracers (no `.perturbed` attribute), and if `vk.bwd` returns `None` for an unperturbed float argument `a` (`_is_float(a) == True`), `_normalize_cotangents` passes `None` through to `@jax.custom_jvp def bwd_apply`, which crashes with `TypeError` because a `@jax.custom_jvp` function cannot return `None`!
- **Root Cause**:
  - Direct `custom_vjp(..., symbolic_zeros=True)` passes `CustomVJPPrimal(value, perturbed)` to `fwd` and accepts `None` or `SymbolicZero` in `bwd`.
  - `xtrax.op`'s `VjpKernel` wraps `fwd` and `bwd` in `@jax.custom_jvp` functions (`fwd_pair` and `bwd_apply`) to route second-order derivatives (`jax.jvp` of `bwd`) to `reference`, which requires all inputs to `fwd` and all float outputs of `bwd` to be valid JAX arrays.
- **Generalizable Rule for `xtrax` Skill**:
  1. Unwrap arguments in `_fwd` via `val = x.value if hasattr(x, "perturbed") else x` and detect perturbation via `is_pert = x.perturbed if hasattr(x, "perturbed") else bool(compute_param_grads)`.
  2. In `make_<name>_op(..., compute_param_grads: bool = True)`, expose `compute_param_grads` as a static config (default `False` for MD force-only ops, `True` for parameter-learning ops).
  3. When parameter gradients are gated off (`not any_param_perturbed`), store only static `jax.ShapeDtypeStruct(val.shape, val.dtype)` metadata in the residuals (0 device buffers saved on the AD tape!) and return `jnp.zeros(meta.shape, dtype=meta.dtype)` (never `None`) for float parameter cotangents in `_bwd`. This skips the parameter-gradient scan at trace time while remaining 100% compatible with `xtrax.op`'s `@jax.custom_jvp` `bwd_apply` wrapper.

### MK-09: Lossless Bit-Tagging for Shared Neighbor Lists Across Heterogeneous Kernels
- **Category**: `Buffer Aliasing & Atomics` / `Lowering & Triton Constraints`
- **Symptom / Failure Mode**:
  1. Materializing a 3D `(BLOCK_N, BLOCK_K, MAX_EXCL) = (32, 32, 32)` tensor inside a Pallas loop to match neighbor indices against per-atom exclusion lists consumes 256 registers/thread and causes severe register spilling.
  2. Overwriting excluded neighbor indices in `neighbor_idx` with the padding sentinel `N` corrupts downstream kernels (`pallas_gb_ace_nl`, `pallas_sasa_nl`, PME exclusion erf correction) that share the same `neighbor_idx` tensor and require covalently bonded 1-2/1-3/1-4 pairs to remain present!
- **Root Cause**: Different physics terms sharing a single `(N, K)` neighbor list have different exclusion semantics (LJ/Coulomb excludes 1-2/1-3 and scales 1-4; PME direct space evaluates $-\text{erf}(\alpha r)/r$ on excluded pairs; OBC-II GB and SASA include all covalent neighbors).
- **Generalizable Rule for `xtrax` Skill**: Pre-encode per-pair exclusion states into the upper bits of the 32-bit neighbor index at neighbor-list build time (`_NB_EXCL_FULL_BIT = 1 << 30`, `_NB_EXCL_14_BIT = 1 << 29`, `_NB_INDEX_MASK = (1 << 29) - 1`) while preserving the true atom index `j = raw_j & _NB_INDEX_MASK` in the lower 29 bits. Nonbonded kernels decode exclusion weights in 3 bitwise ops with 0 memory loads, while GB/SASA kernels strip the upper 2 bits (`raw_j & _NB_INDEX_MASK`) to recover the un-excluded neighbor index.

### MK-10: Triton-Legal Masked Anchor Reduction for Block Bounding Boxes
- **Category**: `Lowering & Triton Constraints`
- **Symptom / Failure Mode**:
  1. Dynamic register indexing (`px_i[first_valid_idx]`) fails Triton lowering because Triton register tensors cannot be indexed by a runtime scalar tensor.
  2. Computing `jnp.min(px_i)` / `jnp.max(px_i)` over a 32-atom block containing padded ghost atoms (`mask_i == False` at `(0, 0, 0)`) corrupts the block bounding box and defeats interior-block periodic wrap bypass.
- **Root Cause**: Triton requires vector-to-scalar extraction to be expressed as a masked reduction (`jnp.sum(jnp.where(lane_id == first_idx, px_i, 0.0))`).
- **Generalizable Rule for `xtrax` Skill**: To compute a tight bounding box `[r_min, r_max]` over a masked 1D Pallas block `px_i` of width `B`:
  ```python
  lane_id = jnp.arange(B, dtype=jnp.int32)
  first_idx = jnp.argmax(valid_i.astype(jnp.int32))
  ax = jnp.sum(jnp.where(lane_id == first_idx, px_i, zero))
  px_v = jnp.where(valid_i, px_i, ax)
  r_min_x, r_max_x = jnp.min(px_v), jnp.max(px_v)
  ```

### MK-11: Plane-Normal (Bekker) Force Projection & Regularization Parity Guardrails
- **Category**: `Differentiation (JVP/VJP)` / `Verification & Negative Controls`
- **Symptom / Failure Mode**:
  1. Differentiating 4-body dihedral angles via normalized-vector projections $\mathbf{v}, \mathbf{w}$ requires 5 cross products and ~85 scalar ops/torsion, inflating register pressure inside a fused bonded megakernel.
  2. Transcribing OpenMM's Bekker plane-normal formula without matching the codebase's $\mathbf{b}_0$ sign convention (`b0_is_ji=True` vs `b0_is_ji=False`) or legacy angle cosine denominator regularization (`cos_theta = dot / (r_ji * r_jk + 1e-8)`) causes either $\pi$ phase inversions on dihedral energies or $\sim 2 \times 10^{-6}$ unit-test parity drift on harmonic angles.
- **Root Cause**:
  - Shifting $\phi$ by $-\pi$ when $\mathbf{b}_0$ is flipped changes `atan2(y, x)` to `atan2(-y, -x) - pi`, while leaving the analytical Cartesian derivatives $\partial \phi / \partial \mathbf{r}$ completely invariant!
  - Adding `denom_eps = 1e-8` to `r_ji * r_jk` perturbs $\theta$ by $\Delta\theta \approx \frac{5 \times 10^{-9}}{\sin\theta}$.
- **Generalizable Rule for `xtrax` Skill**:
  1. Use plane-normal cross-product projection (`n1 = b0_ij x b1`, `n2 = b1 x b2`) for 4-body torsions and 5-body CMAP: both `b0_is_ji=False` and `b0_is_ji=True` share the identical 2-cross-product force projection (`f_i + f_j + f_k + f_l == 0` identically in IEEE-754 float32), differing only in the scalar `atan2` sign/offset.
  2. When replacing an `arccos` quotient-rule derivative with a cross-product direction $\mathbf{v}_{ji} \times (\mathbf{v}_{ji} \times \mathbf{v}_{jk})$, preserve the reference's regularized denominator $r_{\text{denom}}\sqrt{1 - \cos^2\theta_{\text{clipped}}}$ in parity mode so the kernel achieves `< 1e-6` parity with the existing plain-JAX reference.

### MK-12: Horner 4-Tuple Evaluation for 2D Bicubic Spline Stencils (CMAP)
- **Category**: `Lowering & Triton Constraints`
- **Symptom / Failure Mode**: Materializing power basis vectors `[1, u, u^2, u^3]` and `[1, v, v^2, v^3]` and looping 16 times over ungrouped products `c_{p,q} * u^p * v^q` executes 48 multiplications + 48 additions and keeps 8 basis registers live per thread.
- **Root Cause**: A 2D bicubic polynomial $\sum_{p=0}^3 \sum_{q=0}^3 c_{p,q} u^p v^q$ factors into 4 inner 1D Horner cubics $A_p(v) = ((c_{p,3} v + c_{p,2}) v + c_{p,1}) v + c_{p,0}$ and $A'_p(v) = (3 c_{p,3} v + 2 c_{p,2}) v + c_{p,1}$ followed by outer 1D Horner cubics in $u$.
- **Generalizable Rule for `xtrax` Skill**: Evaluate separable 2D/3D polynomial spline tables inside Pallas kernels via nested Horner 4-tuples (`21` hardware `fma.rn.f32` instructions total for value + both partials $\partial E/\partial u, \partial E/\partial v$, zero basis arrays).

### MK-13: Sentinel- & Tag-Preserving Spatial Morton Permutation Across Shared Neighbor Lists
- **Category**: `Grid & Work-Unit Dispatch` / `Buffer Aliasing & Atomics`
- **Symptom / Failure Mode**:
  1. Permuting `neighbor_idx` `(N, K)` via `inv_perm[neighbor_idx[perm]]` indexes out of bounds on padding sentinels (`N` or `-1`) or strips upper exclusion tag bits (`_NB_EXCL_FULL_BIT`, `_NB_EXCL_14_BIT`).
  2. Permuting per-atom exclusion lists `excl_indices` through non-monotonic `inv_perm` destroys row-wise ascending sort order, breaking binary-search or `[excl_min, excl_max]` bounding-interval checks.
- **Root Cause**: Spatial Morton order `perm` is non-monotonic with respect to topological atom indices, and `neighbor_idx` multiplexes valid atom indices `[0, N-1]`, padding sentinels `N`, and bit-tags in a single `int32` tensor.
- **Generalizable Rule for `xtrax` Skill**: When applying a spatial curve permutation `(perm, inv_perm)` at neighbor-list rebuild time:
  1. Force ghost/masked atoms (`~atom_mask`) to `0x7FFFFFFF` in the Morton key so valid atoms stay in `[0, N_real - 1]`.
  2. Strip tag bits (`tag = raw & ~MASK`, `j = raw & MASK`), guard `valid = (j >= 0) & (j < N)`, map `j_new = jnp.where(valid, inv_perm[jnp.clip(j, 0, N - 1)], N)`, and re-attach `j_new | tag`.
  3. Re-sort per-atom exclusion rows ascending after mapping through `inv_perm`.

### MK-14: Compile-Time Dead-Group Pruning in Uniform-CTA Megakernels
- **Category**: `Grid & Work-Unit Dispatch` / `Lowering & Triton Constraints`
- **Symptom / Failure Mode**: If a megakernel unconditionally pads absent force-field groups (e.g., CMAP on AMBER ff19SB without CMAP, or Urey-Bradley on a non-CHARMM system) to `1` dummy CTA block (`max(M, 1)`) and includes all 6 term branches in `jax.lax.cond`, the Triton compiler still compiles the heaviest branch (e.g., 5-body CMAP with 16-term bicubic spline loads) into the kernel binary, inflating register pressure for *every* CTA in the megakernel even when `M_cmap == 0`!
- **Root Cause**: Triton's register allocator sizes the kernel to the maximum register demand across all reachable CFG branches in the compiled module. A branch guarded by `jax.lax.cond(pid < O_g, ...)` where `O_g == O_{g-1}` is still traced and lowered by JAX/Pallas into the Triton IR unless pruned in Python at trace time.
- **Generalizable Rule for `xtrax` Skill**:
  1. Inspect static array counts `count_g = table_g.shape[0]` in Python at trace time and set `n_blk_g = (count_g + B - 1) // B if count_g > 0 else 0`.
  2. Build the `active_groups: list[tuple[int, Callable[[], None]]]` list containing *only* groups with `n_blk_g > 0`, and construct the right-nested `jax.lax.cond(pid < o_end, fn_g, next_branch)` chain exclusively over `active_groups`.
  3. Absent groups contribute `0` CTAs and `0` instructions to the lowered Triton IR, so a system without CMAP or Urey-Bradley never pays register pressure for unused stencil families.

### MK-15: Direct Forward-Force Bypass (`return_forces=True`) for Non-Differentiable Force Evaluation
- **Category**: `Differentiation (JVP/VJP)` / `xtrax.op Seam`
- **Symptom / Failure Mode**: In force-only MD integrators (e.g., `single_padded_force` or velocity-Verlet step functions that call `force_fn(r)` directly without needing an AD tape), computing bonded forces via `-jax.grad(bonded_energy)(r)` traces the forward pass, builds a `custom_vjp` residual tuple, and runs the backward rule just to negate `-fx_pad` that the Pallas forward kernel already computed!
- **Root Cause**: A Pallas energy+force kernel computes `(energy, forces)` in a single GPU launch during the forward pass. Wrapping it exclusively behind a scalar-returning `energy_fn` forces callers that want `forces` to go through `jax.grad`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Expose `return_forces: bool = False` on the public kernel entry point (`pallas_bonded_all_fused(..., return_forces=True) -> (energy, forces)`).
  2. When `return_forces=True`, call the `@custom_vmap` vmappable kernel directly and return `(energy, forces)` without constructing a `custom_vjp` residual frame or running a VJP backward pass.

### MK-16: Zero-Leaf `@jax.tree_util.register_static` Shape Metadata in `custom_vjp` Residuals Across `jax.grad(jax.jit(f))`
- **Category**: `Differentiation (JVP/VJP)` / `xtrax.op Seam`
- **Symptom / Failure Mode**: `TypeError: A ShapeDtypeStruct does not have a value and cannot be used as a constant in a JAX function` when differentiating across a JIT boundary (`jax.grad(jax.jit(fn))(pos)`) if `custom_vjp` forward residuals store `jax.ShapeDtypeStruct(t.shape, t.dtype)` instances to avoid saving unperturbed parameter buffers.
- **Root Cause**: `jax.ShapeDtypeStruct` is registered in JAX's PyTree registry as a **leaf** (not a static auxiliary node). When `jax.grad(jax.jit(fn))` linearizes across the `pjit` boundary, JAX returns all PyTree leaves of the `custom_vjp` residuals as runtime output values of the compiled `pjit` computation, which fails because `ShapeDtypeStruct` has no backing buffer.
- **Generalizable Rule for `xtrax` Skill**: Wrap static `(shape, dtype)` metadata in a `@jax.tree_util.register_static` frozen dataclass (`_StaticShapesMeta`) before returning it in `custom_vjp` / `VjpKernel` residuals. Because `@jax.tree_util.register_static` registers the container with **0 PyTree leaves** (`tree_flatten` returns `leaves = []`), `pjit` passes it purely via static PyTree AuxData with zero device buffers and zero `ShapeDtypeStruct` leaf errors under both `jax.jit(jax.grad(fn))` and `jax.grad(jax.jit(fn))`.

### MK-17: Pallas Triton Lowering Primitive Traps (`reduce_or` from `jnp.any` and `int64` `argmax` under `x64`)
- **Category**: `Lowering & Triton Constraints` / `Verification & Negative Controls`
- **Symptom / Failure Mode**:
  1. `NotImplementedError: Unimplemented primitive in Pallas Triton lowering: reduce_or` when using `jnp.any(mask)` inside a Pallas kernel (e.g., as a `jax.lax.cond` predicate). Note that `interpret=True` on CPU succeeds, hiding this failure unless CUDA Triton MLIR lowering is explicitly tested!
  2. `ValueError: index_type must be i32` in `_argreduce_lowering` when calling `jnp.argmax(...)` inside a Pallas kernel while `jax_enable_x64=True`.
- **Root Cause**:
  1. `jax._src.pallas.triton.lowering` registers `reduce_sum`, `reduce_max`, `reduce_min`, and `reduce_xor`, but does **not** register a lowering rule for `reduce_or` or `reduce_and`.
  2. Under `jax_enable_x64=True`, `jnp.argmax` defaults to `index_dtype=jnp.int64`, which Pallas Triton's `_argreduce_lowering` rejects.
- **Generalizable Rule for `xtrax` Skill**:
  1. Never use `jnp.any(b)` or `jnp.all(b)` inside a Pallas Triton kernel; always write `jnp.sum(b.astype(jnp.int32)) > 0` (for `any`) or `jnp.sum(b.astype(jnp.int32)) == b.size` (for `all`).
  2. Avoid `jnp.argmax` for masked bounding-box reductions; instead use sentinel values `r_min = jnp.min(jnp.where(mask, x, 1e30))` and `r_max = jnp.max(jnp.where(mask, x, -1e30))`.
  3. Always include a CPU-executable `.trace(...).lower(lowering_platforms=("cuda",))` test with `interpret=False` in CI so unimplemented Triton primitives are caught even on CPU-only test runners.

### MK-18: Mixed-Precision Boundary (`float64` BAOAB/SETTLE Integrator + `float32` Pallas Triton Kernels)
- **Category**: `xtrax.op Seam` / `Lowering & Triton Constraints`
- **Symptom / Failure Mode**:
  1. Production explicit-solvent MD (`EnsemblePlan`, BAOAB Langevin + analytical SETTLE water constraints) requires `jax_enable_x64=True` and `float64` positions/velocities/box so rigid-water holonomic projection and coordinate wrapping do not suffer `float32` precision drift.
  2. Passing `float64` positions directly into Pallas Triton kernels (`pallas_bonded_all_fused`, `pallas_nonbonded_nl`, `spread_charges_pallas`, `gather_forces_pallas`) either fails `float64` atomic scatter-add (`tl.atomic_add` on `fp64` vectors) or executes `fp64` math at 1/32nd hardware throughput on consumer/datacenter GPUs and forces `complex128` 3D FFTs in SPME.
  3. Casting only the kernel output without matching `orig_dtype = positions.dtype` at the `_single` kernel boundary triggers `xtrax.intent.errors.IntentError` in `xtrax.op._check_signature` when `reference(positions)` returns `float64` and `kernel(positions)` returns `float32`.
- **Root Cause**: OpenMM's `"mixed"` precision mode achieves both integration stability and GPU Tensor/FP32 throughput by computing forces in `float32` and accumulating/integrating coordinates and constraints in `float64`. In `xtrax.op`, both `kernel` and `reference` must agree on output `ShapeDtypeStruct`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Inside every `_pallas_*_single` entry point (and Pallas PME path), record `orig_dtype = positions.dtype`, cast all floating-point kernel inputs to `jnp.float32` before `pl.pallas_call` (and run 3D SPME `rfftn`/`irfftn` in `float32`/`complex64`), and cast the returned `(energy, forces)` back to `orig_dtype` via `.astype(orig_dtype)`.
  2. Cast `single_padded_energy` / `single_padded_energy_nl` outputs to `r.dtype` so both `xtrax.op._check_signature` and `jax.grad` see exact `float64` primal and cotangent dtypes while 100% of GPU force/FFT compute executes in `float32`.

### MK-19: Dynamic Neighbor-List Bit-Tag Re-Encoding & `update_fn` PyTree Invariance Across `jax.lax.cond`
- **Category**: `Buffer Aliasing & Atomics` / `Grid & Work-Unit Dispatch`
- **Symptom / Failure Mode**:
  1. If `encode_nl_exclusions` is only applied at initial neighbor-list allocation (`nbr0`), the first `jax.lax.cond(should_update, ...)` skin rebuild inside `EnsemblePlan._run_single_inference` overwrites `nbr.idx` with raw untagged `jax_md` cell-list indices, silently dropping all 1-2/1-3/1-4 exclusions on step $k > 0$ when `excl_mode="pre_encoded"`!
  2. Calling `_encode_bundle_nl(n.update(pos), bundle)` inside the `True` branch of `jax.lax.cond` *without* invalidating `n.reference_position` is unsafe because `jax_md`'s `neighbor_fn.update` has its own internal `max_dr < threshold` short-circuit (`lax.cond`) that returns the already-encoded `n.idx` on one branch and a raw `idx` on the rebuild branch.
  3. Wrapping `allocate` / `update` in `make_neighbor_list_fn` without re-binding `dataclasses.replace(nbr, update_fn=update)` causes `n.update(pos)` inside `EnsemblePlan` to bypass periodic coordinate wrapping (`jnp.mod(pos, box)`) and can produce a `TypeError` from mismatched static `update_fn` PyTree treedefs across `lax.cond` branches.
- **Root Cause**: `jax_md.partition.NeighborList` stores `update_fn` as a static metadata field (`static_fields`) initialized to the inner `neighbor_list_fn` closure, and `n.update(pos)` delegates to `self.update_fn(pos, self)`.
- **Generalizable Rule for `xtrax` Skill**:
  1. In any custom `make_neighbor_list_fn` wrapper, always return `dataclasses.replace(nbr, update_fn=update)` from both `allocate` and `update` so `nbr.update(pos)` preserves both coordinate wrapping and static PyTree identity across `lax.cond` / `while_loop` steps.
  2. Make `encode_nl_exclusions` idempotent (`clean_nbr = jnp.where(neighbor_idx >= 0, neighbor_idx & _NB_INDEX_MASK, neighbor_idx)` before matching) and, when forcing a rebuild inside an outer `should_update` branch, invalidate `reference_position` (`+ 1e6`) before calling `.update(pos)` and re-encoding.

### MK-20: Pre-Encoded Neighbor-List Bit-Tag Stripping for XLA Fallback & GB/SASA Kernels
- **Category**: `Buffer Aliasing & Atomics` / `Verification & Negative Controls`
- **Symptom / Failure Mode**:
  1. When `EnsemblePlan` pre-encodes `nbr.idx` with `_NB_EXCL_14_BIT` (`1 << 29`) and `_NB_EXCL_FULL_BIT` (`1 << 30`) and passes the encoded `nb_idx` to XLA fallback neighbor-list kernels (`chunked_lj_energy_nl`, `chunked_coulomb_energy_nl`, `get_neighbor_exclusion_scales`, `compute_gb_energy_neighbor_list`), the XLA kernels index `r[nb_idx]` directly without stripping bits 29/30, treating 1-2/1-3/1-4 pairs (`>= 2^29 > N`) as out-of-bounds padding slots (`valid_pair = nb_idx < N == False`) and silently dropping the 1-4 scaled LJ/Coulomb interaction and the PME direct-space $-\text{erf}(\alpha r)/r$ exclusion term!
- **Root Cause**: `pallas_nonbonded_nl(..., excl_mode="pre_encoded")` decodes bits 29 and 30 in registers, whereas XLA reference neighbor-list kernels and GB/SASA kernels expect clean `[0, N]` atom indices and use `nb_idx < N` as the valid-neighbor predicate.
- **Generalizable Rule for `xtrax` Skill**:
  1. Strip upper tag bits at the energy dispatcher entry point (`nb_clean = jnp.where(nb_raw == -1, N, nb_raw & ((1 << 29) - 1))`).
  2. Maintain two views at the energy dispatcher boundary: `nb_idx` (retaining bits 29 & 30 for `pallas_nonbonded_nl(..., excl_mode="pre_encoded")`) and `nb_clean` (stripped `[0, N]` indices for XLA reference kernels and GB/SASA neighbor-list consumers).

### MK-21: Static `MolecularShapeSpec` Feature Gating vs Unconditional Zero-Padded Array Allocation ($O(N^2)$ Memory Cliff)
- **Category**: `Grid & Work-Unit Dispatch` / `Verification & Negative Controls`
- **Symptom / Failure Mode**: `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 12.40GiB (13319504896 bytes)` on 23,558-atom solvated DHFR inside `init_fn` (`single_padded_energy`) even when `use_neighbor_list=True` and `build_dense_exclusions=False`.
- **Root Cause**: `MolecularBundle` pre-allocates concrete zero-padded arrays for all optional per-atom features (such as `cosolvent_group_id = jnp.zeros(N, dtype=jnp.int32)`) so heterogeneous bundles can be stacked in a uniform PyTree. Forwarding `bundle.cosolvent_group_id` unconditionally to `PhysicsSystem.cosolvent_group_id` caused `if sys.cosolvent_group_id is not None:` in `single_padded_energy` to evaluate `cosolvent_repulsive_energy` (`dr = space.map_product(displacement_fn)(r, r)`), materializing a dense `(23558, 23558, 3)` `float64` tensor (`12.40 GiB`) + `(23558, 23558)` distance matrix (`4.13 GiB`) to compute `0.0` energy on a system with zero cosolvents!
- **Generalizable Rule for `xtrax` Skill**:
  1. Never gate an $O(N^2)$ or optional physics term on `tensor is not None` when upstream bundle builders zero-pad optional tensors for PyTree uniformity.
  2. Always record a host-static boolean capability flag (`has_real_cosolvents: bool = False`, `has_real_water: bool = False`, `has_real_constraints: bool = False`) on the static `ShapeSpec` at bundle construction time (`bool(np.any(np.asarray(gid) > 0))`) and gate the corresponding `PhysicsSystem` field / kernel dispatch on that static flag so zero-count features compile to zero ops and zero memory.
