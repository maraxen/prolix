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

### MK-22: Scale-Free Unit Cross-Product Skew Projector in Analytical 4x4 Horn Quaternion (`settle.py`)
- **Category**: `Verification & Negative Controls` / `Mixed-Precision Stability`
- **Symptom / Failure Mode**: In `_settle_water_batch`, computing the rank-1 Cayley-Hamilton projector M = (N + lam_max I_4)(N^2 - lam_2^2 I_4) via explicit 4x4 matrix multiplication `n_sq = jnp.matmul(n_mat, n_mat); p2 = n_sq - lam2_sq * eye4` incurs O(||u||^4) scaling and suffers catastrophic `float32` cancellation when target water geometry is near-collinear (||u x v|| << ||u||^2 + ||v||^2).
- **Root Cause**: Because S has a zero 3rd row, N^2 - p_tr I_4 = 2 C(u x v) identically, where C(c) is the symmetric zero-diagonal 4x4 cross-product skew matrix. Thus N^2 - lam_2^2 I_4 = 2 q (I_4 + C(c_hat)) where c_hat = (u x v) / ||u x v||.
- **Generalizable Rule for `xtrax` Skill**:
  1. Replace explicit N^2 - lam_2^2 I_4 matrix squaring with the scale-free unit projector P2_hat = I_4 + C(c_hat), which is O(1)-bounded in `float32`/`float64` and saves a 4x4 batched matmul per water molecule per step.

### MK-23: Empty-Dimension (`N == 0` or `K == 0`) Early Return and Direct 3D Batch Dispatch in `pallas_nonbonded_nl`
- **Category**: `Grid & Work-Unit Dispatch` / `Buffer Aliasing & Atomics`
- **Symptom / Failure Mode**:
  1. Calling `_pallas_nonbonded_nl_batch` with `N == 0` or `K == 0` allocates a 0-length `nbr_flat` buffer while launching `grid=(B, 1)`, causing out-of-bounds DMA reads at offset 0.
  2. Calling `_pallas_nonbonded_nl_vmappable` with an explicit 3D `positions` tensor `(B, N, 3)` previously routed through `jax.vmap(lambda b: _vmappable(positions[b], ...))(jnp.arange(B))`, adding unnecessary dynamic gather slice ops before entering `def_vmap`.
- **Root Cause**: `max(1, (N + BLOCK_N - 1) // BLOCK_N)` clamps grid dimensions to >= 1 even when `N == 0` or `K == 0`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Always guard `_pallas_*_batch` entry points with an early `if N == 0 or K == 0: return zeros` before computing `max(1, ...)` grid dimensions.
  2. When a vmappable entry point receives already-batched 3D inputs `(B, N, 3)`, dispatch directly to `_pallas_*_batch(B, in_batched, ...)`.

### MK-24: Dynamic Scalar Loop Limit in `dispatch_n_steps_inference` (`WhileCarry`)
- **Category**: `Grid & Work-Unit Dispatch` / `Compilation Cache Sharing`
- **Symptom / Failure Mode**: Calling `EnsemblePlan.run(..., n_steps=N1)` followed by `EnsemblePlan.run(..., n_steps=N2)` with different `n_steps` triggered a full ~30-second XLA recompilation of the entire BAOAB + PME + Pallas step graph.
- **Root Cause**: `dispatch_n_steps_inference` captured Python `n = int(n_steps)` as a compile-time constant inside the `cond` closure of `jax.jit(_while)(init_state)`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Pass `n_limit = jnp.asarray(int(n_steps), dtype=jnp.int32)` as a traced dynamic scalar argument to `jax.jit(_while)(init_state, n_limit)` so varying `n_steps` (e.g. warmup vs production) reuses the same compiled XLA executable with zero recompilation.

### MK-25: Box-Aware Antipodal Reference-Position Invalidation (`+ 0.5 * box_vec`) Under Minimum-Image PBC
- **Category**: `Verification & Negative Controls` / `Periodic Boundary Conditions`
- **Symptom / Failure Mode**: Forcing a neighbor-list rebuild by shifting `reference_position + 1e6` silently fails to trigger a rebuild whenever any periodic box side length L divides 10^6 (such as L = 32.0 A where 10^6 / 32 = 31,250 exactly, giving minimum-image displacement 0.0!).
- **Root Cause**: Under periodic minimum-image convention, displacement is evaluated modulo L, so any fixed constant shift + C aliases to 0 (mod L) when C / L is an integer.
- **Generalizable Rule for `xtrax` Skill**:
  1. Never invalidate a periodic reference position by adding a magic constant like `+ 1e6`. Always shift by the antipodal half-box vector `+ 0.5 * box_vec`, which maximizes minimum-image displacement for every valid periodic box.

### MK-26: Prefix-Sliced `pad_mask_3d` When `integration_prefix < N_padded`
- **Category**: `Grid & Work-Unit Dispatch`
- **Symptom / Failure Mode**: When `integration_prefix < bundle.positions.shape[0]`, `pos = _integrator_positions(langevin_state)` has shape `(integration_prefix, 3)` whereas `bundle.atom_mask[:, None]` has shape `(N_padded, 1)`, raising a broadcast shape mismatch in `jnp.where(pad_mask_3d, ...)`.
- **Root Cause**: `pad_mask_3d` was constructed from unsliced `bundle.atom_mask[:, None]` instead of slicing to `n_nl_rows = int(ghost_target_positions.shape[0])`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Always slice per-atom masks to the active integrator prefix `bundle.atom_mask[:n_nl_rows, None]`.

### MK-27: `@jax.custom_batching.custom_vmap` Scalar `any_update` Gate for Conditional Neighbor-List Rebuilds Under `jax.vmap`
- **Category**: `Grid & Work-Unit Dispatch` / `Lowering & Triton Constraints`
- **Symptom / Failure Mode**: Under `jax.vmap` across B > 1 replicas (e.g. B = 8 DHFR or B = 32 ragged buckets), `jax.lax.cond(should_update, rebuild_fn, noop_fn, neighbor)` inside `_nl_step_fn` has a batched boolean predicate `should_update` of shape `(B,)`. JAX lowers `vmap(lax.cond)` with a batched predicate into `lax.select_n` after unconditionally executing both branches on every MD step (~15x more rebuild work!).
- **Root Cause**: Standard `jax.vmap` cannot branch on a vector predicate without evaluating both branches unless a `custom_vmap` rule reduces the vector predicate to a scalar `any_update = jnp.any(should_update_b)`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Wrap conditional neighbor-list rebuilds in `@jax.custom_batching.custom_vmap` and, inside `def_vmap`, reduce `any_upd = jnp.any(su_b)` to a scalar boolean before calling `jax.lax.cond(any_upd, _do_batch, _keep_batch, ...)`.
  2. When `any_upd` is `True`, rebuild all B replicas simultaneously in a single 2D-grid `(N, B)` Pallas launch and reset `last_pos = pos_b` across all B replicas so their Verlet skin clocks stay synchronized.

### MK-28: Cell-Level Stencil + Pallas 2D-Grid `(N, B)` Neighbor-List Pruning & In-Register Exclusion Tagging (v2.2 / v2.3)
- **Category**: `Grid & Work-Unit Dispatch` / `Buffer Aliasing & Atomics`
- **Symptom / Failure Mode**: On 23,558-atom solvated DHFR, Pallas v2.1 single-step force evaluation took only `0.537 ms/step` (`0.427 ms` direct-space nonbonded + `0.110 ms` fused bonded + `0.163 ms` SPME) + `0.227 ms` SETTLE, but `jax_md` neighbor-list rebuild + exclusion encoding consumed `2.999 ms` (`2.923 ms` device time) due to scattering 27 cells per atom into an `(N, 27 * C, 3)` candidate buffer (~6.4 GB HBM traffic) and `encode_nl_exclusions` materializing `(N, K)` float32 scale arrays (144 MB HBM traffic).
- **Root Cause**: Building the 27-cell stencil at the atom level `(N, 27 * C)` duplicates every cell candidate list across all atoms in that cell, and post-hoc exclusion encoding re-reads and re-writes the entire `(B, N, K)` neighbor list in HBM.
- **Generalizable Rule for `xtrax` Skill**:
  1. Build the 27-cell stencil at the cell level `(B, G_cells, 27, C)` via 26 periodic `jnp.roll` shifts of the binned `(nx, ny, nz, C)` cell grid (~1.2 MB for DHFR instead of 6.4 GB).
  2. Launch a single 2D-grid `(N, B)` Pallas kernel (`pallas_prune_and_encode_cell_list`) where each program `(i_atom, b_id)` streams its cell `27 * C` candidates in `BLOCK_C = 64` tiles, filters r^2 < (r_cut + r_skin)^2 under minimum-image PBC, matches valid hits against the atom `(M=32,)` exclusion row in registers, bitwise-ORs `_NB_EXCL_FULL_BIT` / `_NB_EXCL_14_BIT` before storing to `nbr_out_ref`, and updates `max_occupancy` via `pl.atomic_max`.
  3. **Verified Engaging GPU Impact**:
     - On **NVIDIA H100 80GB HBM3 (`node2900`, Job `27692626`)**: `pallas_prune_and_encode_cell_list` reduces full DHFR neighbor-list rebuild + 2-bit exclusion encoding from `2.999 ms` to `1.178 ms` (`1.066 ms` device time, **`2.55x` speedup**), bringing steady-state 23,558-atom solvated DHFR rollout to **`1.063 ms/step`** (**`81.28 ns/day`** at `dt = 1.0 fs`, **`17.1x` faster** than Pallas v2.0's `18.202 ms/step` / `4.75 ns/day` and **`60.2x` faster** than pure XLA's `64.007 ms/step` / `1.35 ns/day`). Across $B=8$ multi-replica DHFR (`2D` grid batching), per-replica force evaluation drops from `0.573 ms` ($B=1$) to **`0.377 ms/replica`** (**`229.3 ns/day`** aggregate throughput), and on $N=1,024, B=32$ ragged buckets 2D-grid batching reaches `0.271 ms` (`0.0085 ms/replica`, **`6.43x` faster** than `lax.map`).
     - On **NVIDIA RTX PRO 6000 Blackwell Server Edition (`node5204`, Job `27727006`)**: With `MK-30` applied, the full 4-point `bench_dhfr_parity.py` regression (`n_steps = [100, 500, 2000, 5000]`) achieves **`1.187 ms/step`** (**`72.78 ns/day`**, $R^2 = 0.9981$, `15.33x` speedup over Pallas v2.0's `18.202 ms/step` on the exact same Blackwell SKU) with zero Call-2 recompilation (`t_steady_state = 0.399 s` on `n_steps=100`, `0.616 s` on `500`, `2.445 s` on `2000`, `6.107 s` on `5000`).

### MK-29: Six Pallas Triton / CPU-Interpret Traps in Dynamic Compaction Kernels (`pallas_prune_and_encode_cell_list`)
- **Category**: `Lowering & Triton Constraints` / `Buffer Aliasing & Atomics`
- **Symptom / Failure Mode**:
  1. `TypeError: _kernel() takes 15 positional arguments but 16 were given` when using `input_output_aliases={13: 0}` with 2 outputs `(nbr_out, occ_out)`.
  2. `TypeError: cond branches must have equal output types (int64[] vs int32[])` when accumulating `count_acc + jnp.sum(valid_i32)` inside `jax.lax.cond` under `jax_enable_x64=True`.
  3. `ValueError: The kernel function in the pallas_call _kernel captures constants [i32[32], i32[64]]` when `jnp.arange(64)` is hoisted outside `_kernel`.
  4. `ValueError: Mosaic GPU kernel exceeds available shared memory` when `compiler_params=plt.CompilerParams()` is omitted on `pl.pallas_call`.
  5. `NotImplementedError: Unimplemented primitive in Pallas Triton lowering: round` when using `jnp.rint(dx * inv_Lx)` for minimum-image PBC inside `_kernel`.
  6. `ValueError` when inspecting `box` inside `@jax.custom_batching.custom_vmap` because `custom_vmap` traces all positional arguments as `JitTracer`s.
- **Root Cause**:
  1. Every input argument in `pl.pallas_call` (including aliased inputs in `input_output_aliases`) receives its own input `Ref` parameter in `_kernel` in addition to the output `Ref` parameters.
  2. Under `jax_enable_x64=True`, `jnp.sum(int32_arr)` and `jnp.cumsum(int32_arr)` promote their accumulator dtype to `int64` unless `dtype=jnp.int32` is passed explicitly.
  3. Pallas forbids closed-over constant arrays; `jnp.arange` index vectors must be constructed inside `_kernel`.
  4. Modern JAX defaults `pallas_call` on CUDA to Mosaic GPU unless `plt.CompilerParams()` is passed.
  5. Pallas Triton does not lower `lax.round` (`jnp.rint`), whereas `jnp.floor(x + jnp.float32(0.5))` lowers to `tl.floor`.
  6. Static grid dimensions `(nx, ny, nz)` must be computed before entering `custom_vmap` (or passed from host `self.bundles[0].box`).
- **Generalizable Rule for `xtrax` Skill**:
  1. Always pass `dtype=jnp.int32` explicitly to `jnp.sum(..., dtype=jnp.int32)` and `jnp.cumsum(..., dtype=jnp.int32)` inside Pallas kernels.
  2. Construct `jnp.arange` lane vectors inside `_kernel`, pass `compiler_params=plt.CompilerParams() if not interpret else None`, use `jnp.floor(x + jnp.float32(0.5))` for nearest-integer PBC wrapping, precompute static `grid_dims=(nx, ny, nz)` before `custom_vmap`, and route masked-out `_pl_store` lanes to a valid dummy sentinel slot (`B * N * K`) so CPU `interpret=True` never overwrites slot 0.

### MK-30: Canonicalizing Initial Carry Through `@jax.jit` Reseeder & Recomputing Cache Keys After Lazy Imports (`_cache_size() == 1`)
- **Category**: `Grid & Work-Unit Dispatch` / `Compilation Cache Sharing`
- **Symptom / Failure Mode**: When caching a single-bundle inference runner (`_SINGLE_INFERENCE_CACHE` + `_JITTED_WHILE_CACHE`), Call 1 (`cached_entry is None`, e.g. `n_steps=1` warmup) compiled `jitted_while`, and Call 2 (`cached_entry is not None`, e.g. `n_steps=99`) triggered a second `23.66 s` compilation of `jitted_while`, whereas Calls 3+ hit Call 2's cache in `0.0028 s`.
- **Root Cause**: Two distinct mechanisms can cause a Call-2 cache miss:
  1. **Leaf commitment mismatch**: On Call 1 (`cached_entry is None`), `state` came from eager/unjitted `self._setup_integrator(...)` while the warmup output of `@jax.jit`-compiled `fast_reseed_fn` was discarded (`_ = fast_reseed_fn(seed, kT_s)`). On Call 2 (`cached_entry is not None`), `state` came from `cached_entry["fast_reseed_fn"](seed, kT_s)`. Because JAX's C++ `pjit` cache key includes per-leaf device commitment (`committed`) and weak-type metadata, passing unjitted `init_fn` leaves on Call 1 vs `@jax.jit` `fast_reseed_fn` leaves on Call 2 caused a single `pjit` cache miss on Call 2.
  2. **Lazy-import `sys.modules` drift in `_SINGLE_INFERENCE_CACHE` key**: On Call 1 of a fresh Python process with `use_pallas=True`, `cache_key` was computed *before* `self._setup_integrator(...)` lazily imported `prolix.physics.pme_pallas` into `sys.modules` (so `sys.modules.get("prolix.physics.pme_pallas")` returned `None` when constructing `cache_key`). When `cached_entry` was stored under that pre-import `cache_key`, Call 2 saw `prolix.physics.pme_pallas` populated in `sys.modules`, computed a different `cache_key` containing the actual `_spread_charges_pallas_single` / `_gather_pallas_single` function objects, missed `_SINGLE_INFERENCE_CACHE`, created a brand-new `_nl_step_fn` closure, and compiled a second `jitted_while` in `_JITTED_WHILE_CACHE`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Whenever a cached runner uses a `@jax.jit` reseeder (`fast_reseed_fn`) on warm calls (`cached_entry is not None`), assign `state = fast_reseed_fn(seed, kT_s)` on the initial cache-populating call (`cached_entry is None`) as well rather than discarding the warmup output. This guarantees identical PyTree leaf metadata (`aval`, `weak_type`, `sharding`, `committed`) across Call 1 and all subsequent calls (`jitted_while._cache_size() == 1`).
  2. When a runner cache key inspects `sys.modules` for monkeypatchable kernel references that may be lazily imported during setup, gate optional module attributes on whether that backend is active and recompute `cache_key = _make_cache_key()` *after* setup right before inserting into the cache (`len(_JITTED_WHILE_CACHE) == 1`).

### MK-31: Ragged-Bucket Ghost-Atom Sentinel Cell Routing (`c_i = G_cells`) & Row-Level Valid-Neighbor Compaction
- **Category**: `Grid & Work-Unit Dispatch` / `Verification & Negative Controls`
- **Symptom / Failure Mode**: When a smaller solvated system (e.g., 6,658 real atoms) is placed into a larger size bucket (`25,000` padded atoms) for multi-system batching, un-filtered 3D cell-list builders place the 18,342 ghost atoms (`~atom_mask`) into spatial cells inside the periodic box. Real atoms' neighbor rows accumulate **1,899,431 ghost neighbor entries**, wasting >50% of cell-list pruning and nonbonded tile iterations, and scattered sentinel entries (`idx == N`) prevent `pallas_nonbonded_nl`'s `row_max_k` early exit from halting early.
- **Root Cause**: `_build_cell_stencil_buffers_batched` binned all `N` padded positions into `[0, G_cells)` regardless of `atom_mask`, and `encode_nl_exclusions` left invalid/ghost slots scattered across the `K` columns of each row.
- **Generalizable Rule for `xtrax` Skill**:
  1. Route every ghost atom (`~atom_mask`) to an out-of-bounds sentinel cell index `c_i = G_cells` (`G_cells = nx * ny * nz`), size the 27-neighbor stencil LUT to `((G_cells + 1) * 27,)`, and zero out the sentinel cell's occupancy (`cell_counts = cell_counts.at[:, G_cells].set(0)`).
  2. In `encode_nl_exclusions`, replace any neighbor where `~atom_mask[i]` or `~atom_mask[j]` with sentinel `N` and compact valid (`< N`) neighbors to the front of each row via `order = jnp.argsort((out == N).astype(jnp.int32), axis=-1); out = jnp.take_along_axis(out, order, axis=-1)` so `row_max_k` halts immediately after the last real neighbor tile.
  3. **Verified Multi-GPU Impact (`261102_{h200,l40s,preempt}_pallas_v24_e2e_ensemble_buckets.json`)**:
     - Reduces ghost neighbors in real rows from **`1,899,431` (v2.3) $\to$ `0` (v2.4)** with exact ragged-bucket padding invariance (`8,192` vs `25,000` bucket: $\Delta E \le 0.00098\text{ kcal/mol}$, $\max |\Delta F| \le 4.56\times 10^{-5}\text{ kcal/(mol}\cdot\text{\AA)}$).
     - Speeds up 25k ragged-bucket cell-list pruning by **`1.95x` on Hopper H200 (`0.740 ms -> 0.380 ms`)**, **`1.94x` on Blackwell RTX PRO 6000 (`0.665 ms -> 0.343 ms`)**, **`1.61x` on Ada L4 (`2.553 ms -> 1.581 ms`)**, and **`1.51x` on Ada L40S (`0.900 ms -> 0.597 ms`)**.
     - Speeds up 25k ragged-bucket force evaluation by **`1.24x–1.36x`** across H200 (`0.274 ms -> 0.221 ms`), Blackwell (`0.252 ms -> 0.203 ms`), L40S (`0.245 ms -> 0.197 ms`), and L4 (`0.397 ms -> 0.293 ms`), matching the compact 8k bucket latency within ~7–10%.

### MK-32: CTA-Level $O(1)$ Empty-Tile Early Exit & Triton `scf.yield` MLIR `ir.Value` Zero-Carry Rule
- **Category**: `Lowering & Triton Constraints` / `Grid & Work-Unit Dispatch`
- **Symptom / Failure Mode**:
  1. In ragged size buckets, trailing atom tiles (`pid * BLOCK_N` where all `mask_i == 0`) and trailing bonded-term tiles (`b_msk == 0`, `a_msk == 0`, `d_msk == 0`, etc.) still execute full transcendental math (`erf`, `erfc`, `cos`, `sin`, `atan2`, 64-point B-spline stencils) and atomic/global memory traffic.
  2. Wrapping a Pallas Triton tile body in `jax.lax.cond(has_any, _body, lambda _: jnp.int32(0), None)` or returning `jnp.zeros_like(x)` from the false branch crashes Triton MLIR lowering with `ValueError: Operand 0 of operation "scf.yield" must be a Value (was TypedNdArray)`.
- **Root Cause**: Pallas Triton lowers `jax.lax.cond` to an MLIR `scf.if` op whose `scf.yield` operands must be dynamic MLIR `ir.Value`s rather than host-constant `TypedNdArray` literals.
- **Generalizable Rule for `xtrax` Skill**:
  1. Guard every Pallas tile group (`pallas_nonbonded_nl`, `pallas_prune_and_encode_cell_list`, `pallas_bonded_all_fused`, `pme_charge_spread`, `pme_bwd_gather`) with a CTA-level `jax.lax.cond(jnp.max(valid_mask.astype(jnp.int32)) > 0, _active_branch, _empty_branch, zero_carry)`.
  2. Always construct `zero_carry` via algebraic cancellation of an existing in-kernel MLIR `ir.Value` (e.g., `zero_hits = c_i - c_i` for scalar `int32` or `zero_vec = (offs - offs).astype(dtype)` for vector `float32`) and pass it as the carry operand into `jax.lax.cond`.

### MK-33: Single-Pass SETTLE Water COM/Inertia Reduction & Direct Tuple Impulse Projection (`settle.py`)
- **Category**: `Grid & Work-Unit Dispatch` / `Buffer Aliasing & Atomics`
- **Symptom / Failure Mode**: In explicit-solvent BAOAB dynamics (`_langevin_step_o_constrained`), calling `settle_velocities` followed by `_remove_angular_momentum_from_impulse` recomputed water center-of-mass positions/velocities and the regularized $3\times 3$ inertia tensor a second time across all $W$ waters, and scattered `(dv_O, dv_H1, dv_H2)` into a full `(N, 3)` atom array only to immediately re-gather `dv[water_indices]` on the next line.
- **Root Cause**: `settle_velocities` and `_remove_angular_momentum_from_impulse` were authored as independent full-array stages rather than sharing the water-indexed `(W, 3)` impulse tuple.
- **Generalizable Rule for `xtrax` Skill**:
  1. Fuse water center-of-mass, relative coordinates, and regularized $3\times 3$ inertia tensor construction into a single pass per step.
  2. Pass `dv_tuple = (dv_O, dv_H1, dv_H2)` directly into `_remove_angular_momentum_from_impulse` before performing a single final `.at[idx_O/H1/H2].set(...)` scatter onto the `(N, 3)` velocity array.

### MK-34: `BatchTracer` Host-Fallback Box-Size Trap, Densest-Bundle Union-Mask NL Seed, & `_STACKED_INFERENCE_CACHE` (`B >= 2`)
- **Category**: `Grid & Work-Unit Dispatch` / `Compilation Cache Sharing`
- **Symptom / Failure Mode**:
  1. Under `jax.vmap` (`B >= 2`), `bundle.box` is a `BatchTracer`, causing `_host_box_size(bundle, default=(30.0, 30.0, 30.0))` to catch `TracerArrayConversionError` and silently substitute a `30.0 A` box on a `61.645 A` solvated DHFR system—wrapping coordinates across a `30 A` boundary inside Pallas kernels and exploding to `NaN` on Step 0.
  2. In heterogeneous same-bucket batches (e.g., `[sys_4199, ubq_6656]` in an `8,192` bucket), allocating the shared `NeighborList` seed from `bundles[0]` under-sizes `max_occupancy` and `cell_list_capacity` for the denser `bundles[1]`, whereas calling `neighbor_fn.allocate` per bundle produces incompatible `cell_list_capacity` shapes that cannot be stacked.
  3. Repeated multi-bundle `EnsemblePlan.run(run_mode="inference")` calls re-stacked bundles, re-allocated neighbor lists on the host, and re-traced `_batched` whenever `n_steps` changed.
- **Root Cause**:
  1. Silent exception fallback in `_host_box_size` masked `BatchTracer` box inspection inside `vmap`.
  2. `neighbor_fn.allocate` sizes `cell_list_capacity` and `max_occupancy` from the active density of the single input configuration.
  3. Multi-bundle inference lacked an analog of `_SINGLE_INFERENCE_CACHE` with a dynamic `n_steps` scalar parameter.
- **Generalizable Rule for `xtrax` Skill**:
  1. Extract concrete `host_box_vec` on the host *before* entering `@jax.jit` / `jax.vmap` and attach it as a concrete `np.ndarray` on the energy-construction bundle so `vmap` never falls back to a default box size.
  2. In `_build_stacked_neighbor_seed`, compute `union_mask = jnp.any(jnp.stack([b.atom_mask for b in bundles]), axis=0)` and call `neighbor_fn.allocate` **once** on `densest_bundle` with `atom_mask=union_mask`, pad columns in $O(1)$, and initialize the remaining bundles via `seed_invalidated.update(b_pos)`.
  3. Cache `(stacked, stacked_neighbor, neighbor_fn, _batched)` in `_STACKED_INFERENCE_CACHE` with `n_steps` passed as a dynamic `jnp.int32` scalar into `_batched(stacked, seeds, stacked_neighbor, n_steps_arr)`.

### MK-35: Capability- and Grid-Size-Aware `resolve_pallas_compiler_params` Across GPU Architectures (`prolix.export`)
- **Category**: `Lowering & Triton Constraints` / `Grid & Work-Unit Dispatch`
- **Symptom / Failure Mode**: Hardcoding `plt.CompilerParams()` (or over-tuning `num_warps`/`num_stages` for a single GPU SKU or a single large problem size) either wastes shared memory / registers on small-grid buckets (`grid_blocks < 256` on 132-SM Hopper / 140+-SM Blackwell) or under-utilizes pipeline stages on large multi-replica 2D grids (`B * n_blocks >= 256`), and placing `resolve_pallas_compiler_params` in `prolix.pallas_kernels` prevented `prolix.physics.{neighbor_list,pallas_bonded,pme_pallas}` from importing it at module top level without violating `tests/test_import_architecture.py`.
- **Root Cause**: Different Pallas kernel families (`bonded_fused` 1D vs `nonbonded_nl`, `cell_prune`, `pme` 2D tiles) and different grid sizes (`grid_blocks < 256` vs `>= 256`) have distinct register and shared-memory footprints across `sm_80` (Ampere), `sm_89` (Ada L4/L40S), `sm_90` (Hopper H100/H200), and `sm_100`/`sm_120` (Blackwell B200/B300/RTX PRO 6000).
- **Generalizable Rule for `xtrax` Skill**:
  1. Place `resolve_pallas_compiler_params(kernel_kind, grid_blocks, ...)` in the shared leaf `prolix.export` (imported at top level by `neighbor_list.py`, `pallas_bonded.py`, `pme_pallas.py`, and `pallas_kernels.py`) and wire it into every `pl.pallas_call`.
  2. Use `num_warps=2, num_stages=1` for 1D register-light `bonded_fused` kernels; for 2D-tiled `nonbonded_nl`, `cell_prune`, and `pme` kernels, use `num_warps=4` with `num_stages=1` on small grids (`0 < grid_blocks < 256`) to maximize multi-CTA SM occupancy and `num_stages=2` on large grids (`grid_blocks >= 256`), with `PROLIX_PALLAS_NUM_WARPS` and `PROLIX_PALLAS_NUM_STAGES` environment overrides.
  3. **Verified Multi-GPU End-to-End `EnsemblePlan.run()` Throughput**:
     - **NVIDIA H200 (`sm_90`, `261102_h200_pallas_v24_e2e_ensemble_buckets.json`)**:
       - Solvated DHFR (`23,558` real atoms in `25,000` bucket): $B=1 \to 1.897\text{ ms/step}$ (`91.08 ns/day` at $\Delta t=2\text{ fs}$), $B=2 \to 1.173\text{ ms/step/replica}$ (`147.27 ns/day` agg), $B=4 \to 1.063\text{ ms/step/replica}$ (`162.52 ns/day` agg), $B=8 \to \mathbf{1.001\text{ ms/step/replica}}$ (**`172.56 ns/day` agg**).
       - Medium solvated protein (`6,658` real atoms in `8,192` bucket): $B=1 \to 0.524\text{ ms/step}$ (`329.75 ns/day`), $B=4 \to 0.295\text{ ms/step/replica}$ (`585.13 ns/day` agg), $B=8 \to \mathbf{0.241\text{ ms/step/replica}}$ (**`716.44 ns/day` agg**).
       - Heterogeneous same-bucket (`4,199` + `6,656` atoms in `8,192` bucket): **`0.384 ms/step/system`** (`0.767 ms/step` batch).
       - Solvated DHFR WT + Mutant (`2 × 23,558` atoms with perturbed charges/$\sigma$/$\varepsilon$ in `25,000` bucket): **`1.177 ms/step/variant`** (`2.354 ms/step` batch, `5.305 Å` RMSD divergence, all finite).
       - Heterogeneous multi-bucket (`[4199, 23558, 6656, 23558]` across `8,192` + `25,000` buckets): **`0.780 ms/step/system`** (`3.120 ms/step` total for 4 systems).
     - **NVIDIA RTX PRO 6000 Blackwell (`sm_120`, Job `27817876`)**:
       - Solvated DHFR (`23,558` real atoms in `25,000` bucket): $B=1 \to 1.884\text{ ms/step}$ (`91.74 ns/day`), $B=2 \to 1.387\text{ ms/step/replica}$ (`124.55 ns/day`), $B=4 \to 1.273\text{ ms/step/replica}$ (`135.72 ns/day`), $B=8 \to \mathbf{1.184\text{ ms/step/replica}}$ (**`145.94 ns/day` agg**).
       - Medium solvated protein (`6,658` real atoms in `8,192` bucket): $B=1 \to 0.601\text{ ms/step}$ (`287.57 ns/day`), $B=4 \to 0.360\text{ ms/step/replica}$ (`479.54 ns/day`), $B=8 \to \mathbf{0.320\text{ ms/step/replica}}$ (**`539.74 ns/day` agg**).
     - **NVIDIA L40S (`sm_89`, `261102_l40s_pallas_v24_e2e_ensemble_buckets.json`)**:
       - Solvated DHFR (`23,558` real atoms in `25,000` bucket): $B=1 \to 2.168\text{ ms/step}$ (`79.69 ns/day`), $B=8 \to \mathbf{1.558\text{ ms/step/replica}}$ (**`110.93 ns/day` agg**); DHFR WT + Mutant: **`1.697 ms/step/variant`**; Medium `8,192` bucket $B=8$: **`0.372 ms/step/replica`** (**`464.45 ns/day` agg**); Heterogeneous multi-bucket (4 systems): **`1.066 ms/step/system`**.
     - **NVIDIA L4 (`sm_89`, `261102_preempt_pallas_v24_e2e_ensemble_buckets.json`)**:
       - Solvated DHFR (`25,000` bucket): $B=1 \to 5.438\text{ ms/step}$, $B=8 \to 5.002\text{ ms/step/replica}$ (`34.55 ns/day` agg); Medium `8,192` bucket: $B=1 \to 1.024\text{ ms/step}$ (`168.68 ns/day`); Heterogeneous same-bucket: **`1.000 ms/step/system`**.


