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

### MK-36: True Single-Launch Fused Bonded + Direct-Space Nonbonded Megakernel (`pallas_bonded_and_nonbonded_nl_fused`) Across All Bucket Sizes
- **Category**: `Grid & Work-Unit Dispatch` / `Buffer Aliasing & Atomics`
- **Symptom / Failure Mode**:
  1. Launching `pallas_bonded_all_fused` and `pallas_nonbonded_nl` as two separate Pallas kernels (or falling back to two separate launches when `B * N > 16384`) incurs back-to-back GPU kernel launch latency, redundant zero-initialization of two `(B, N)` force buffers per coordinate axis, and an extra host/XLA elementwise buffer addition (`g_pos = f_bonded + f_nb`) on every BAOAB step.
  2. Inside `_megakernel`'s `_do_nonbonded_nl` branch, returning four `(32,)` vectors `(e_blk, fx_blk, fy_blk, fz_blk)` from `jax.lax.cond(has_any_valid, _run_k_tiles, _skip_k_tiles, None)` forces Triton's `scf.if` / `scf.yield` to allocate 128 `float32` registers across the branch and execute 96 `atomic_add` operations even on trailing ghost-atom tiles (`jnp.max(mask_i) == 0`).
- **Root Cause**:
  1. Both `pallas_bonded_all_fused` and `pallas_nonbonded_nl` scatter/accumulate forces into the same `(B, N)` force buffers `(fx_out_ref, fy_out_ref, fz_out_ref)` and reduce scalar block energies into `e_blk_out_ref`.
  2. Performing the force writeback outside `jax.lax.cond(has_any_valid, ...)` requires `scf.yield` to carry the 3D force accumulator vectors out of the conditional and executes global memory atomics unconditionally.
- **Generalizable Rule for `xtrax` Skill**:
  1. Fuse all bonded CTA groups (`0 <= pid < total_bonded_blocks`) and direct-space nonbonded NL CTA blocks (`total_bonded_blocks <= pid < total_blocks`) into a single 2D-grid `(B, total_blocks)` Pallas megakernel (`pallas_bonded_and_nonbonded_nl_fused`) across all bucket sizes (`8,192`, `25,000`, etc.) with `compiler_params=resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=B * total_blocks)`.
  2. Move the force writeback inside `_run_k_tiles` so ghost-atom blocks (`jnp.max(mask_i) == 0`) skip global memory writes completely and `jax.lax.cond` yields only a single scalar `jnp.sum(e_blk)` instead of 128 vector registers.
  3. When `total_bonded_blocks == 0`, write `fx_blk, fy_blk, fz_blk` directly via `_pl_store` instead of `_pl_atomic_add_1d`. Even when `total_bonded_blocks > 0`, precompute a loop-invariant per-32-atom-block `blk_has_bonded` mask (`(B_hb * n_atom_blocks,)` `int32`, hoisted out of `lax.while_loop` by XLA LICM) from active bonded indices/masks so pure-solvent water/padding blocks (~85–90% of blocks in solvated protein systems, since rigid TIP3P water is handled by SETTLE) execute direct `_pl_store` (`_store_write`) instead of `_pl_atomic_add_1d` (`_atomic_write`).

### MK-37: Avoiding `jax.lax.while_loop` (`scf.while`) C++ Triton Layout Pass Segfaults in Pallas K-Tile Loops
- **Category**: `Grid & Work-Unit Dispatch` / `Lowering & Triton Constraints`
- **Symptom / Failure Mode**:
  1. Using `jax.lax.while_loop` (`scf.while`) inside `pallas_nonbonded_nl` or `pallas_bonded_and_nonbonded_nl_fused` to carry 1D `(32,)` vector accumulators `(acc_e, acc_fx, acc_fy, acc_fz)` alongside scalar `int32` loop-continuation state (`k_tile, keep_going`) crashes the C++ Triton compiler (`SIGSEGV` in `jax/_src/compiler.py:353 in backend_compile_and_load` during `TritonGPURemoveLayoutConversionsPass`) on Hopper (`H100`/`H200`) and Ada (`L40S`).
  2. Even after `encode_nl_exclusions` compacts all valid neighbors (`< N`) to the front of each neighbor-list row (`MK-31`), scanning trailing sentinel tiles without skipping empty tiles wastes global memory loads on padded sentinel columns (`K` up to `544–864`).
- **Root Cause**: Triton's MLIR `TritonGPURemoveLayoutConversionsPass` segfaults when reconciling `#ttg.blocked` vector layouts and scalar `i32` condition carries across the `scf.condition` / `scf.yield` regions of an `scf.while` op inside a multi-branch Pallas kernel. By contrast, `jax.lax.fori_loop` lowers to `scf.for` with a flat homogeneous `(acc_e, acc_fx, acc_fy, acc_fz)` vector carry.
- **Generalizable Rule for `xtrax` Skill**:
  1. Never use `jax.lax.while_loop` (`scf.while`) to carry 1D/2D vector accumulators inside a Pallas Triton kernel.
  2. Always use `jax.lax.fori_loop(0, k_blocks, _k_tile_body, c0)` with a flat homogeneous `(acc_e, acc_fx, acc_fy, acc_fz)` carry and guard the inner tile evaluation via `jax.lax.cond(has_any_valid, _eval_nonempty_tile, lambda c_in: c_in, carry)` so empty sentinel tiles skip all global memory loads and math while avoiding the C++ `scf.while` layout-conversion compiler segfault.

### MK-38: Active-AD Charge-Gradient Gating (`compute_dq=False`) in Pallas SPME Backward Gather
- **Category**: `Differentiation (JVP/VJP)` / `Buffer Aliasing & Atomics`
- **Symptom / Failure Mode**: During standard MD trajectory steps (`jax.grad(energy_fn)(positions)`), `charges` is unperturbed, yet `_spme_bwd` unconditionally computed `dE/dq` (`potentials` output of `gather_pallas`), allocating a `(B, N)` output buffer and executing 64-point B-spline potential summation and global memory stores per atom.
- **Root Cause**: `_spme_fwd` did not record `charges_perturbed = _is_perturbed(charges)` in its `custom_vjp` residuals, so `_spme_bwd` always requested both `forces` and `potentials` from `gather_pallas`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Record `charges_perturbed = _is_perturbed(charges)` in `_spme_fwd` / `_spme_cached_fwd` residuals.
  2. Pass `compute_dq=bool(charges_perturbed)` to `gather_pallas` so position-only MD steps skip the `potentials` accumulation and global store (`1.07x–1.11x` faster PME gather on H100 and L40S), while parameter-learning / charge-perturbation gradients (`argnums=(0, 1)`) still compute exact `dE/dq`.

### MK-39: Eliminating `float32` Energy Summation Quantization Before `.astype(orig_dtype)` & Caching Host Influence Function (`G_cached_np`)
- **Category**: `Mixed-Precision Stability` / `Compilation Cache Sharing` / `Grid & Work-Unit Dispatch`
- **Symptom / Failure Mode**:
  1. In mixed-precision mode (`orig_dtype = float64`, internal Pallas kernel `dtype = float32`), summing per-atom nonbonded energies inside the Pallas CTA (`jnp.sum(e_blk)`) or summing the output buffer across blocks in `float32` before casting `.astype(orig_dtype)` accumulates ~`0.05–0.11 kcal/mol` rounding error across thousands of atoms (`N = 8,192–25,000`), breaking tight (`< 1e-4`) mixed-precision energy parity and ragged-bucket padding invariance.
  2. Recomputing the 3D reciprocal-space influence function $G(\mathbf{m})$ inside the traced energy function on every call or re-evaluating it on the device when `box_size` is a static orthorhombic box adds unnecessary FFT frequency grid setup ops.
- **Root Cause**:
  1. `float32` has only 24 bits of mantissa (~7 decimal digits); reducing thousands of `float32` terms of magnitude $10^1\text{–}10^2\text{ kcal/mol}$ in `float32` loses 3–4 bits of precision compared to upcasting the per-atom/per-block array to `orig_dtype` (`float64`) before `jnp.sum`.
  2. When `box_size` is concrete at `make_spme_energy_fn` construction time, $G(\mathbf{m})$ depends only on static `(grid_dims, box_size, alpha, order)`.
- **Generalizable Rule for `xtrax` Skill**:
  1. In `_pallas_bonded_and_nonbonded_nl_fused_batch`, size `e_out_ref` to `e_out_dim = max(1, n_nb_blocks * _NB_BLOCK_N + total_bonded_blocks)`, store per-atom nonbonded energies `jnp.where(mask_i, e_blk, zero)` at `b_id * e_out_dim + nb_pid * _NB_BLOCK_N + offs_i` (and zeros in `_skip_k_tiles`) and per-block bonded energies at `b_id * e_out_dim + e_atom_slots + b_pid`, and always cast `.astype(orig_dtype)` **before** `jnp.sum(..., axis=-1)` on the host in `_pallas_nonbonded_nl_batch`, `_pallas_bonded_and_nonbonded_nl_fused_batch`, and `_pallas_bonded_all_fused_batch`.
  2. Precompute and LRU-cache `G_cached_np` (and its `float32` cast `G_cached_f32_np`) on the host via `_cached_host_influence_function_np` and pass it into `_spme_energy_with_forces_cached`.

### MK-40: Multi-GPU Pallas v2.5 Verification Across Hopper (`H100`) and Ada (`L40S`)
- **Category**: `Verification & Negative Controls`
- **Symptom / Failure Mode**: Verifying that `MK-36`..`MK-39` preserve numerical parity (`< 1e-4` vs XLA, exact ragged-bucket invariance between `8,192` and `25,000` buckets) and pass all 19 GPU tests in `tests/test_ensemble_plan_pallas_v2.py`.
- **Root Cause**: Multi-architecture Triton codegen (`sm_89` vs `sm_90`) can exhibit different register allocation and atomic coalescing behavior.
- **Generalizable Rule for `xtrax` Skill**:
  1. Always validate fused megakernel changes across at least two distinct GPU SM architectures (`sm_89` Ada and `sm_90` Hopper / `sm_120` Blackwell) with full energy/force parity and ragged-bucket padding invariance checks.

### MK-41: Eliminating the `float64` / `complex128` 3D FFT Trap in Mixed-Precision Pallas SPME (`src/prolix/physics/pme.py`)
- **Category**: `Mixed-Precision Stability` / `Lowering & Triton Constraints`
- **Symptom / Failure Mode**:
  1. Under `jax_enable_x64=True` (required for `float64` coordinate/velocity accumulation in `mixed` precision mode), even when `spread_charges_pallas` and `gather_pallas` run their B-spline stencil kernels in `float32`, `spread_charges_pallas` casts its output `charge_grid` back to `orig_dtype` (`float64` at `pme_pallas.py:206`), and `make_spme_energy_fn` passed `G_cached` in `positions.dtype` (`float64`).
  2. Consequently, `jnp.fft.rfftn(Q)` and `jnp.fft.irfftn(Q_hat * G)` executed in **double precision (`float64` / `complex128`)** on every MD step (`(B, 64, 64, 64)` cuFFT double-precision plans), consuming `2x` memory bandwidth and up to `4x–32x` lower FFT throughput on GPUs (`L40S`, `Blackwell`, `H100`), whereas OpenMM's `mixed` precision mode always executes PME 3D FFTs (`cufftExecR2C` / `cufftExecC2R`) and force evaluations in `float32`!
  3. Checking `jnp.asarray(box_size).ndim == 1` instead of `_is_ortho_box(positions, box_size)` silently disabled Pallas SPME and `float32` 3D FFT whenever `box_size` was a 2D batched orthorhombic array `(B, 3)` (including `B = 3` where `box_size.shape == (3, 3)`).
- **Root Cause**: `Q` and `G` were allowed to inherit `positions.dtype` (`float64`) before calling `jnp.fft.rfftn` / `jnp.fft.irfftn` inside `_spme_fwd`, `_spme_cached_fwd`, and `spme_reciprocal_energy`, and orthorhombic detection checked `ndim == 1` rather than distinguishing `(3,)` / `(B, 3)` orthorhombic vectors from `(3, 3)` lower-triangular triclinic matrices.
- **Generalizable Rule for `xtrax` Skill**:
  1. Whenever `use_pallas_pme=True` (outside `export_context`) and `_is_ortho_box(positions, box_size)` is `True` (covering both 1D `(3,)` and 2D batched `(B, 3)` orthorhombic boxes), explicitly keep `Q`, `G`, `G_cached`, `theta`, and `theta_norm` in `jnp.float32` (so `rfftn` produces `complex64` and `irfftn` produces `float32`), pass the `float32` `theta_norm` directly to `gather_pallas`, and cast **only** the final scalar reciprocal energy `e_recip` and `(N, 3)` atomic forces back to `orig_dtype = positions.dtype`.
  2. Pre-cast `G_cached_f32_np = G_cached_np.astype(np.float32)` on the host in `make_spme_energy_fn` so zero per-step `float64 -> float32` conversion of the influence grid occurs on the device.

### MK-42: System-Scaled PME FFT Grids (`PROLIX_PME_GRID_POINTS`), Single vs Mixed Precision Modes, and OpenMM `CUDA` Head-to-Head Parity (`bench_e2e_ensemble_buckets.py`)
- **Category**: `Verification & Negative Controls` / `Grid & Work-Unit Dispatch`
- **Symptom / Failure Mode**:
  1. Truncating a 23,558-atom DHFR bundle (`64^3` PME grid) to `small_4200` (`4,199` atoms) and `medium_6658` (`6,658` atoms) without scaling the PME FFT grid forced a 4,199-atom system to execute a `64^3 = 262,144`-point 3D FFT at every step instead of the `32^3 = 32,768`-point (`8x` smaller) grid appropriate for a ~34.8 Å solvated box (`48^3` for 6,658 atoms).
  2. Spawning an external OpenMM Python environment (`b1-openmm/bin/python`) via `subprocess.run` without populating `LD_LIBRARY_PATH` with `b1-openmm/lib`, `nvidia/*/lib`, and `/orcd/software/core/001/pkg/cuda/*/lib64` caused `dlopen("libOpenMMCUDA.so")` to fail on missing `libcufft.so` / `libnvrtc.so` and silently fall back to `OpenCL`.
  3. Unpacking `for (ai, aj), (k_b, r0_b), valid_b in zip(d['bond_idx'], d['bond_params'], d['bond_mask'])` in `build_sys_from_npz` swapped `r0` (`bond_params[:, 0]`) and `k` (`bond_params[:, 1]`), and wrapping truncated water coordinates via atom-wise `jnp.mod(pos_real, box_vec)` split boundary water molecules across opposite periodic faces, causing OpenMM SETTLE to explode (`Particle coordinate is NaN`).
- **Root Cause**:
  1. `MolecularBundle.bond_params` stores `(r0, k)` and `angle_params` stores `(theta0, k)`.
  2. Rigid water molecules must be re-imaged as whole molecules (`H1, H2` wrapped to `O`'s nearest periodic image) before SETTLE initialization.
  3. Dynamic library search paths (`LD_LIBRARY_PATH`) must include the CUDA runtime/cuFFT/NVRTC directories before `openmm.Platform.loadPluginsFromDirectory` runs.
- **Generalizable Rule for `xtrax` Skill**:
  1. Support both scalar (`"32"`, `"48"`, `"64"`) and bucket-mapped (`"8192:48,25000:64"`) `PROLIX_PME_GRID_POINTS` in `make_spme_energy_fn` (and include `PROLIX_PME_GRID_POINTS` in `EnsemblePlan`'s `_SINGLE_INFERENCE_CACHE` / `_STACKED_INFERENCE_CACHE` keys) so benchmarks and heterogeneous multi-bucket runners match PME FFT grids (`32^3` for 4.2k atoms, `48^3` for 6.7k/8k bucket, `64^3` for 23.6k/25k bucket) identically across Prolix and OpenMM (`nb.setPMEParameters(alpha, g_pts, g_pts, g_pts)`).
  2. Benchmark both `single` (`float32` bundle, where BAOAB + SETTLE + Pallas megakernel + SPME 3D FFT all execute in `float32`) and `mixed` (`float64` positions/velocities + `float32` Pallas megakernel + `float32` SPME 3D FFT) precision modes across batch sizes $B \in [1, 2, 4, 8, 16, 32]$.
  3. Always re-image rigid water hydrogens onto their oxygen's periodic image after box wrapping, unpack `(r0, k)` / `(theta0, k)` in canonical `MolecularBundle` column order, configure `LD_LIBRARY_PATH` + `OPENMM_PLUGIN_DIR` for `libOpenMMCUDA.so`, and log `openmm.Platform.getPluginLoadFailures()` in the benchmark output JSON.

### MK-43: Eliminating `num_stages=2` Software-Pipelining Register Spill and `sm_120` Per-Thread Stack Overflow in the 58-Argument Multi-Branch Megakernel (`src/prolix/export.py`, `src/prolix/pallas_kernels.py`)
- **Category**: `Lowering & Triton Constraints` / `Multi-Architecture Tuning`
- **Symptom / Failure Mode**:
  1. On `NVIDIA RTX PRO 6000 Blackwell` (`sm_120`), changing `_pallas_bonded_and_nonbonded_nl_fused_batch` (`59` buffer arguments, `7` top-level `pid` branches) to use `resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=B * total_blocks)` (`num_stages=2` for `grid_blocks >= 256`) and adding a nested runtime `jax.lax.cond(has_b_blk, _atomic_write, _store_write)` inside `_run_k_tiles` caused `mk36_fused_direct_and_bonded_megakernel_8k_ms` to regress `5.5x` (`0.138 ms -> 0.766 ms`) and triggered `CUDA_ERROR_ILLEGAL_ADDRESS` (per-thread local-memory stack overflow) at `DHFR [mixed] B=4` (`grid=(4, 965)`), even though `NVIDIA L40S` (`sm_89`) compiled and ran at `0.1168 ms` (`1.21x` faster than separate launches) and `NVIDIA H200` (`sm_90`) ran at `0.1342 ms` (`1.10x` faster).
- **Root Cause**:
  1. While standalone `_pallas_nonbonded_nl_batch` on Hopper/Ada benefits from `num_stages=2`, `_megakernel` has 58 buffer refs and 7 top-level `pid` branches. On `sm_120` (Blackwell, `cc >= 10.0`), `num_stages=2` software pipelining under `PTX 8.8` combined with the 59th `blk_has_bonded_ref` pointer and nested `jax.lax.cond` around vector force stores inside `_run_k_tiles` spilled tile registers to per-thread local memory (`.local`), exceeding the default CUDA per-thread stack size at `B=4, N=25000`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Dedicated `kernel_kind == "megakernel"` in `resolve_pallas_compiler_params` MUST return `num_warps=4, num_stages=1` across all grid sizes and GPU architectures (`sm_80`, `sm_89`, `sm_90`, `sm_120`), and all Pallas kernels (`megakernel`, `nonbonded_nl`, `cell_prune`, `pme`) on Blackwell (`cc >= 10.0` / `"b300"` / `"blackwell"`) should use `num_stages=1` unless overridden by `PROLIX_PALLAS_NUM_STAGES`.
  2. Keep `_run_k_tiles` writeback branchless at runtime: use compile-time `if total_bonded_blocks == 0: _pl_store(...) else: _pl_atomic_add_1d(...)` with 58 input refs (`input_output_aliases={55: 1, 56: 2, 57: 3}`), never a runtime `jax.lax.cond` around vector force stores inside the megakernel.

### MK-44: Vectorized `float32` Threefry Thermal Noise in Constrained & Unconstrained Langevin BAOAB O-Steps (`src/prolix/physics/settle.py`, `src/prolix/batched_simulate.py`)
- **Category**: `Mixed-Precision Stability` / `Host-Boundary & Scan Overhead`
- **Symptom / Failure Mode**:
  1. Under `jax_enable_x64=True` (`mixed` precision mode), `EnsemblePlan.run` calls `_langevin_step_o_constrained` (`src/prolix/physics/settle.py:2514-2572`), which on **every MD step** executed `jax.random.split(key, n_w + 1)` (`8,193` Threefry splits per replica!) plus `jax.random.normal(split, momentum.shape, dtype=float64)` and `2 * n_w` per-water `jax.random.normal(..., (3,), dtype=float64)` calls inside `step_one_water` (`16,384` separate `float64` Threefry + Box-Muller calls per replica per step!).
  2. Similarly, unconstrained BAOAB integrators in `src/prolix/batched_simulate.py` called `jax.random.normal(subkey, p.shape)` which defaulted to `float64` when `jax_enable_x64=True`.
- **Root Cause**:
  1. `jax.random.normal` without `dtype=jnp.float32` defaults to `float64` under `jax_enable_x64=True`, running twice as many Threefry2x32 rounds and `float64` Box-Muller `log`/`sqrt`/`sin`/`cos` instructions even though thermal Langevin noise is scaled by $\sqrt{m k_B T (1 - e^{-2\gamma \Delta t})}$ and has no physical accuracy beyond `float32`.
  2. Splitting PRNG keys per water molecule (`jax.random.split(key, n_w + 1)` + `jax.random.split(key_w, 3)`) per step executed $3 \times N_{\text{waters}}$ sequential Threefry rounds instead of a single contiguous `(N_waters, 6)` draw.
- **Generalizable Rule for `xtrax` Skill**:
  1. In all Langevin BAOAB O-steps (`_langevin_step_o_constrained`, `_langevin_step_o`, and `batched_simulate.py`), always sample Gaussian thermal deviates in `dtype=jnp.float32` and cast `.astype(momentum.dtype)` before multiplying by `c2`.
  2. In rigid-water constrained O-steps (`_langevin_step_o_constrained`), sample a single contiguous `z6_all = jax.random.normal(split_w, (n_w, 6), dtype=jnp.float32).astype(r_water.dtype)` tensor once per step outside `step_one_water` and slice `z_trans = z6_w[:3]`, `z_rot = z6_w[3:]`, eliminating $3 \times N_{\text{waters}}$ per-step PRNG splits and $2 \times N_{\text{waters}}$ `float64` Box-Muller calls.

### MK-45: Uniform Floating-Point Bundle Precision in Mixed-Precision `EnsemblePlan` Runs & `lax.cond` `reference_position.dtype` Normalization (`src/prolix/api/ensemble_plan.py`, `src/prolix/physics/neighbor_list.py`)
- **Category**: `Mixed-Precision Stability` / `Buffer Aliasing & Atomics`
- **Symptom / Failure Mode**:
  1. When `MolecularBundle` instances are constructed from serialized `.npz` fixtures while `jax_enable_x64=True` (e.g., `B_dhfr._build_bundle_from_sys_data(sys_data)`), `positions` can be promoted to `float64` while `masses`, `charges`, `sigmas`, and `epsilons` remain `float32` (as stored in the `.npz`).
  2. Passing a half-`float64` (`positions`), half-`float32` (`masses`) bundle into `EnsemblePlan.run` causes `settle_positions` (`pos_oxygen_c` in `float64`) and `_langevin_step_o_constrained` (`r_water` in `float64` combined with `m_water` in `float32`) to attempt `scatter` updates of `float64` values into `float32` target arrays (`FutureWarning: scatter inputs have incompatible types: cannot safely cast value from dtype=float64 to dtype=float32 with jax_numpy_dtype_promotion=standard`), and causes `neighbor.reference_position` (`float32` if allocated before promotion) vs `positions` (`float64`) to mismatch inside `_maybe_rebuild_nl`'s `jax.lax.cond`.
- **Root Cause**:
  1. JAX's `scatter` primitive requires update values to be safely castable to the target buffer's dtype; mixing `float64` `positions` with `float32` `masses` inside `settle.py` produces `float64` impulses/velocities scattered into a `float32`-derived momentum array.
  2. In `_maybe_rebuild_nl` (`ensemble_plan.py`) and `_cond_rebuild` (`neighbor_list.py`), the `True` branch returned `p_w` (with `positions.dtype`) while the `False` branch returned `r_o` (with `neighbor.reference_position.dtype`).
- **Generalizable Rule for `xtrax` Skill**:
  1. In `_maybe_rebuild_nl` and `_cond_rebuild`, always cast the rebuilt reference position `p_w.astype(r_o.dtype)` so both branches of `jax.lax.cond(needs_rebuild, ...)` return identical dtypes regardless of how `neighbor` was initially allocated.
  2. Before running `mixed` precision (`jax_enable_x64=True`, `float64` integration state + `float32` Pallas force/PME kernels) vs `single` precision (`float32` integration state + `float32` Pallas force/PME kernels), normalize all floating-point fields of the `MolecularBundle` (`positions`, `masses`, `charges`, `sigmas`, `epsilons`, `bond_params`, `angle_params`, `dihedral_params`, `improper_params`, `cmap_coeffs`) to a uniform floating-point dtype (`float64` for `mixed`, `float32` for `single`) via `_cast_bundle_precision`.
  3. **Verified Pallas v2.6 / v2.7 Multi-GPU End-to-End `EnsemblePlan.run()` Throughput (`NVIDIA H200` `sm_90` Job `27989419`, `NVIDIA L40S` `sm_89` Job `28016116`, `NVIDIA A100-SXM4-80GB` `sm_80` Job `28016109`, & `NVIDIA RTX PRO 6000 Blackwell` `sm_120`)**:
     - **All 7 GPU Unit Tests in `tests/test_ensemble_plan_pallas_v2.py` (`test_pallas_v24`, `test_pallas_v25`, `test_pallas_v26`) PASSED (`7 passed`) Across All 4 GPU Architectures (`sm_80` A100, `sm_89` L40S, `sm_90` H200, `sm_120` Blackwell)** with zero `scatter` dtype warnings (`MK-45`).
     - **Single-Launch Fused Megakernel (`MK-36`, `8,192` Bucket)**:
       - `NVIDIA L40S` (`sm_89`, v2.7 Job `28016116`): Separate `0.0944 ms` $\to$ Fused **`0.0802 ms` (`1.18x` speedup)**; `cell_prune_25k_ragged_speedup`: **`1.56x`** (`0.858 ms` $\to$ `0.551 ms`); `force_eval_25k_ragged_speedup`: **`1.29x`** (`0.183 ms` $\to$ `0.142 ms`); `mk38_pme_gather_compute_dq_false_vs_true`: **`1.09x`** (`0.0392 ms` $\to$ `0.0359 ms`).
       - `NVIDIA A100-SXM4-80GB` (`sm_80`, v2.7 Job `28016109`): Separate `0.1464 ms` $\to$ Fused **`0.1188 ms` (`1.23x` speedup)**; `cell_prune_25k_ragged_speedup`: **`2.22x`** (`1.334 ms` $\to$ `0.601 ms`); `force_eval_25k_ragged_speedup`: **`1.23x`** (`0.353 ms` $\to$ `0.286 ms`).
       - `NVIDIA H200` (`sm_90`, Job `27989419`): Separate `0.1481 ms` $\to$ Fused **`0.1342 ms` (`1.10x` speedup)**; `cell_prune_25k_ragged_speedup`: **`1.99x`** (`0.745 ms` $\to$ `0.374 ms`).
     - **Solvated DHFR (`23,558` real atoms in `25,000` bucket, `64^3` PME grid)**:
       - `NVIDIA H200` (`sm_90`):
         - `single`: $B=1 \to 1.798\text{ ms/step}$ (`96.08 ns/day`), $B=2 \to 1.197\text{ ms/step/rep}$ (`144.42 ns/day`), $B=4 \to 1.088\text{ ms/step/rep}$ (`158.78 ns/day`), $B=8 \to 1.033\text{ ms/step/rep}$ (`167.33 ns/day`), $B=16 \to \mathbf{1.019\text{ ms/step/rep}}$ (**`169.60 ns/day` agg**).
         - `mixed`: $B=1 \to \mathbf{1.479\text{ ms/step}}$ (**`116.85 ns/day`**, **`1.28x` faster** than v2.4's `1.897 ms/step`), $B=2 \to 1.237\text{ ms/step/rep}$ (`139.68 ns/day`), $B=4 \to 1.118\text{ ms/step/rep}$ (`154.51 ns/day`), $B=8 \to 1.065\text{ ms/step/rep}$ (`162.23 ns/day`), $B=16 \to \mathbf{1.050\text{ ms/step/rep}}$ (**`164.61 ns/day` agg**).
       - `NVIDIA RTX PRO 6000 Blackwell` (`sm_120`, v2.7 `MK-43`/`MK-44`):
         - `single`: $B=1 \to 1.711\text{ ms/step}$ (`101.00 ns/day`) $\to B=8 \to 1.196\text{ ms/step/rep}$ (`144.43 ns/day`) $\to B=16 \to \mathbf{1.158\text{ ms/step/rep}}$ (**`149.17 ns/day` agg**).
         - `mixed`: $B=1 \to \mathbf{1.551\text{ ms/step}}$ (**`111.42 ns/day`**, **`1.21x` faster** than v2.4's `1.884 ms/step`).
       - `NVIDIA L40S` (`sm_89`, v2.7 Job `28016116`):
         - `single`: $B=1 \to 1.955\text{ ms/step}$ (`88.37 ns/day`) $\to B=16 \to \mathbf{1.375\text{ ms/step/rep}}$ (**`125.72 ns/day` agg**).
         - `mixed`: $B=1 \to \mathbf{1.844\text{ ms/step}}$ (**`93.70 ns/day`**, **`1.18x` faster** than v2.4's `2.168 ms/step`) $\to B=16 \to \mathbf{1.545\text{ ms/step/rep}}$ (**`111.84 ns/day` agg**).
       - `NVIDIA A100-SXM4-80GB` (`sm_80`, v2.7 Job `28016109`):
         - `single`: $B=1 \to 2.992\text{ ms/step}$ (`57.76 ns/day`) $\to B=8 \to \mathbf{1.783\text{ ms/step/rep}}$ (**`96.92 ns/day` agg**), $B=16 \to 1.790\text{ ms/step/rep}$ (`96.54 ns/day` agg).
         - `mixed`: $B=1 \to \mathbf{2.401\text{ ms/step}}$ (**`71.98 ns/day`**) $\to B=8 \to \mathbf{1.828\text{ ms/step/rep}}$ (**`94.52 ns/day` agg**), $B=16 \to 1.836\text{ ms/step/rep}$ (`94.10 ns/day` agg).
     - **Medium Solvated Protein (`6,658` real atoms in `8,192` bucket, `48^3` PME grid)**:
       - `NVIDIA H200` (`sm_90`): `single` $B=32 \to \mathbf{0.204\text{ ms/step/rep}}$ (**`847.38 ns/day` agg**); `mixed` $B=32 \to \mathbf{0.212\text{ ms/step/rep}}$ (**`813.48 ns/day` agg**).
       - `NVIDIA L40S` (`sm_89`, v2.7 Job `28016116`): `single` $B=32 \to \mathbf{0.286\text{ ms/step/rep}}$ (**`604.79 ns/day` agg**); `mixed` $B=1 \to 0.711\text{ ms/step}$ (`243.10 ns/day`) $\to B=32 \to \mathbf{0.333\text{ ms/step/rep}}$ (**`518.18 ns/day` agg**).
       - `NVIDIA A100-SXM4-80GB` (`sm_80`, v2.7 Job `28016109`): `single` $B=32 \to \mathbf{0.357\text{ ms/step/rep}}$ (**`483.65 ns/day` agg**); `mixed` $B=32 \to \mathbf{0.368\text{ ms/step/rep}}$ (**`469.20 ns/day` agg**).
     - **Small Solvated Protein (`4,199` real atoms in `8,192` bucket, `32^3` PME grid)**:
       - `NVIDIA H200` (`sm_90`): `single` $B=16 \to 0.169\text{ ms/step/rep}$ (`1,024.51 ns/day` agg), $B=32 \to \mathbf{0.167\text{ ms/step/rep}}$ (**`1,036.49 ns/day` agg**); `mixed` $B=32 \to \mathbf{0.175\text{ ms/step/rep}}$ (**`984.83 ns/day` agg**).
       - `NVIDIA L40S` (`sm_89`, v2.7 Job `28016116`): `single` $B=32 \to \mathbf{0.249\text{ ms/step/rep}}$ (**`693.18 ns/day` agg**); `mixed` $B=32 \to \mathbf{0.298\text{ ms/step/rep}}$ (**`579.41 ns/day` agg**).
       - `NVIDIA A100-SXM4-80GB` (`sm_80`, v2.7 Job `28016109`): `single` $B=32 \to \mathbf{0.277\text{ ms/step/rep}}$ (**`623.71 ns/day` agg**); `mixed` $B=32 \to \mathbf{0.289\text{ ms/step/rep}}$ (**`597.14 ns/day` agg**).

---

### MK-46: Fused Pallas BAOAB + Analytical Miyamoto–Kollman / Horn Quaternion SETTLE + RATTLE Integrator Kernels & Triton `lax.neg_p` `TypedNdArray` Rule (`src/prolix/physics/settle.py`, `src/prolix/export.py`, `src/prolix/api/ensemble_plan.py`)
- **Category**: `Megakernel Fusion` / `Pallas Triton Lowering`
- **Symptom / Failure Mode**:
  1. In Pallas v2.7, while bonded, direct-space nonbonded, and SPME electrostatics were fused into Pallas kernels (`~0.131 ms/step` on `8,192` bucket), the Langevin BAOAB + rigid-water analytical SETTLE + RATTLE velocity projection integrator in `src/prolix/physics/settle.py` still launched ~18 separate per-step XLA scatter/gather/elementwise kernels (`_langevin_step_b`, `_langevin_step_a`, `settle_positions`, `_langevin_step_r`, `_remove_angular_momentum_from_impulse`, `_langevin_step_o_constrained`, `_langevin_step_a`, `settle_positions`, `_langevin_step_r`, `_langevin_step_b`, `settle_velocities`, and ghost-atom `jnp.where` pinning in `ensemble_plan.py`), costing `0.1199 ms/step` at `B=1` on `NVIDIA L40S`.
  2. During initial GPU compilation of the fused Pallas BAOAB+SETTLE kernel on `L40S` (`JAX 0.10.2`), Triton lowering crashed with `AttributeError: 'TypedNdArray' object has no attribute 'type'` inside `jax._src.pallas.triton.lowering._minus` (`lax.neg_p`).
- **Root Cause**:
  1. Un-fused BAOAB+SETTLE repeatedly gathers `(B, N_waters, 3, 3)` water triads from `(B, N_atoms, 3)` state arrays and scatters updated positions/momenta back to global memory 6 times per step.
  2. In JAX's Pallas Triton lowering (`_minus`), `lax.neg_p` checks `_is_triton_pointer_type(x.type)` before negating `x`. When unary `-` is applied inside a Pallas kernel to a scalar constant created via `jnp.asarray(val, dtype=dtype)` (which Pallas represents as a `TypedNdArray` literal rather than a `triton.language.tensor`), `x.type` does not exist and raises `AttributeError`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Fuse the entire pre-force BAOAB half-step (`B1 -> A1 -> PBC wrap -> SETTLE1 -> R1 + AM -> 6D rigid-body OU / isotropic float32 O-step -> A2 -> SETTLE2 -> R2 + AM -> ghost pinning`) into a single 2D-grid `(B, cdiv(N_atoms, 64) + cdiv(N_waters, 32))` Pallas kernel (`pallas_baoab_pre_force`), and fuse the post-force half-step (`B2 -> RATTLE 3x3 Gram velocity projection + AM -> ghost pinning`) into a second 2D-grid Pallas kernel (`pallas_baoab_post_force`).
  2. Precompute the loop-invariant `(B, N_atoms)` water/ghost exclusion mask (`_iwg_precomputed`) once outside the `while_loop` (`settle_langevin` setup) and mark `apply_fn._pallas_settle_pins_ghosts = True` so `ensemble_plan.py` skips redundant post-step `jnp.where(pad_mask_3d, ...)` scatters.
  3. **Triton `lax.neg_p` Constant Rule**: Never apply unary `-` to a `jnp.asarray(c, dtype=dtype)` constant inside a Pallas Triton kernel; negate the Python float before wrapping (`jnp.asarray(-c, dtype=dtype)`) or write `zero - expr`.
  4. Always fall back automatically to pure-JAX/StableHLO SETTLE when `in_export_context()` is active, when JVP tracers are present (`_is_ad`), or when `settle_velocity_iters == 0`.
  5. **Measured Speedup (`NVIDIA L40S`, Job `28062261`)**: Reduces BAOAB+SETTLE integrator time on the `8,192` bucket from `0.1199 ms` (XLA) to **`0.0530 ms` (Pallas)** — a **`2.26x` speedup**.

### MK-47: Per-Replica Independent Neighbor-List Rebuild Masking (`rebuild_mask: bool[B]`) & `old_idx` In-Place Aliasing (`src/prolix/physics/neighbor_list.py`, `src/prolix/api/ensemble_plan.py`)
- **Category**: `Multi-Replica Scaling` / `Buffer Aliasing & Atomics`
- **Symptom / Failure Mode**:
  - In multi-replica `EnsemblePlan` batches (`B = 2..32`), `_maybe_rebuild_nl_vmap` previously triggered a full batch neighbor-list rebuild across all `B` replicas whenever `jnp.any(should_update)` was `True`, even if only a single replica crossed the Verlet skin threshold (`dr_threshold / 2`).
- **Root Cause**:
  - `_build_cell_stencil_buffers_batched` and `_make_prune_and_encode_kernel` lacked per-replica `rebuild_mask: bool[B]` awareness and allocated a fresh sentinel-filled `(B * N * K + 1,)` output buffer instead of aliasing the existing neighbor list `old_idx`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Thread `rebuild_mask: bool[B]` and `old_idx: int32[B, N, K]` into `_build_cell_stencil_buffers_batched` and `_pallas_prune_and_encode_cell_list_batch`.
  2. For any replica `b` with `rebuild_mask[b] == False`, assign sentinel cell ID `G_cells` to all atoms in replica `b` and alias `old_idx` to `nbr_out_ref` via `input_output_aliases={15: 0}`. Inside `_make_prune_and_encode_kernel`, CTAs with `c_i == G_cells` exit in $O(1)$ at the top of the kernel without touching `nbr_out_ref`, preserving `old_idx[b]` in-place while only updating `reference_position[b]` and `last_pos[b]` where `rebuild_mask[b]` is `True`.

### MK-48: Spatially Pruned `32x32` Block Nonbonded Megakernel (`O(1)` Ghost Block Skip, Skin-Shell Load Gating, & Branchless Reciprocal-Multiply PBC/Forces) (`src/prolix/pallas_kernels.py`)
- **Category**: `Ragged-Bucket Compaction` / `Megakernel Fusion`
- **Symptom / Failure Mode**:
  - Inside `_nb_nl_kernel` and `_megakernel` (`_do_nonbonded_nl`), trailing ghost 32-atom blocks (`~atom_mask`) and skin-shell neighbor tiles (`min_r_sq >= cutoff_sq`) still executed redundant per-tile global parameter loads (`charges`, `sigmas`, `epsilons`) and `jnp.round(dx / Lx)` floating-point divisions inside the inner K-tile loop.
- **Root Cause**:
  - Unconditional `q_j`, `sig_j`, `eps_j` gathers before checking whether any pair in the 32-atom neighbor tile lies within `cutoff_sq`, plus per-tile `dx / Lx` divisions.
- **Generalizable Rule for `xtrax` Skill**:
  1. Gate the entire 32-atom row block at the CTA level on `jnp.max(mask_i_b) > 0` so pure-ghost padding blocks exit in $O(1)$ with zero neighbor-list reads or atomic writes.
  2. Precompute `inv_Lx, inv_Ly, inv_Lz` once per CTA and use `dx - Lx * jnp.floor(dx * inv_Lx + 0.5)` for minimum-image PBC wrapping.
  3. Compute `r_sq` from `px_j, py_j, pz_j` first and gate the `q_j, sig_j, eps_j` global loads and LJ/PME force evaluation on `jnp.max((r_sq < cutoff_sq).astype(jnp.int32)) > 0` via `jax.lax.cond`, skipping parameter loads for pure skin-shell tiles.
  4. **Measured Speedup (`NVIDIA L40S`, Job `28062261`)**: Reduces `8,192`-bucket force evaluation from `0.1387 ms` (v2.7) to **`0.1313 ms` (v3.0)** and `25,000`-bucket ragged force evaluation to **`0.1366 ms`** (`1.28x` faster than unfiltered).

### MK-49: Separable `(BLOCK_N=32, 16)` YZ-Plane Factorized & Charge-Gated Pallas SPME Spread/Gather + Pallas v3.0 `NVIDIA L40S` Benchmark Summary (`src/prolix/physics/pme_pallas.py`)
- **Category**: `Buffer Aliasing & Atomics` / `End-to-End Throughput`
- **Symptom / Failure Mode**:
  - In `src/prolix/physics/pme_pallas.py`, `_spread_kernel` and `_gather_kernel` iterated over `u` (`0..3`) in a `lax.fori_loop` and flattened all `16` `(v, w)` grid points into a `(32, 16)` tensor per `u`, recomputing `v = k_idx // 4` and `w = k_idx % 4` and executing `atomic_add` even for uncharged/ghost atoms (`q_i == 0`).
- **Root Cause**:
  - Un-factorized 3D B-spline weight evaluation and un-gated atomic updates for zero-charge ghost atoms.
- **Generalizable Rule for `xtrax` Skill**:
  1. Factorize the 64-point $(4 \times 4 \times 4)$ B-spline stencil into a precomputed `(BLOCK_N=32, 16)` YZ-plane outer product (`wyz = wy[:, :, None] * wz[:, None, :]` and its derivatives `dwyz_dy`, `dwyz_dz`) outside the 4-iteration X-axis loop (`u = 0..3`), and gate global `atomic_add` writes in `_spread_kernel` on `(mask_n > 0) & (jnp.abs(q) > 0)` with early CTA exit when `jnp.max(active_i) == 0`.
  2. **Verified Pallas v3.0 End-to-End `EnsemblePlan.run()` Performance on `NVIDIA L40S` (`sm_89`, Job `28062261`, [`outputs/bench/261105_l40s_pallas_v30_e2e_ensemble_buckets.json`](file:///home/marielle/projects/prolix/outputs/bench/261105_l40s_pallas_v30_e2e_ensemble_buckets.json))**:
     - **All 8 GPU Unit Tests in `tests/test_ensemble_plan_pallas_v2.py` (`test_pallas_v24`, `test_pallas_v25`, `test_pallas_v26`, `test_pallas_v30`) PASSED (`8 passed in 145.98s`)**:
       - `energy_diff_pallas_vs_xla_kcal_mol`: **`0.01395 kcal/mol`** (`< 0.02 kcal/mol`)
       - `force_max_diff_pallas_vs_xla_kcal_mol_A`: **`2.67e-4 kcal/(mol*A)`** (`< 5e-4 kcal/(mol*A)`)
       - `ragged_bucket_invariance_energy_diff_8k_vs_25k`: **`1.56e-5 kcal/mol`**
       - `ghost_neighbors_in_real_rows_v24`: **`0`**
     - **Kernel Microbenchmarks (`NVIDIA L40S` `sm_89`)**:
       - `mk46_pallas_fused_baoab_settle_8k_ms`: **`0.0530 ms`** vs `0.1199 ms` XLA (**`2.26x` speedup**)
       - `force_eval_8k_compact_bucket_v25_ms`: **`0.1313 ms`** (improved from `0.1387 ms` in v2.7)
       - `force_eval_25k_ragged_v24_filtered_ms`: **`0.1366 ms`** vs `0.1748 ms` unfiltered (**`1.28x` speedup**)
       - `cell_prune_25k_ragged_v24_filtered_ms`: **`0.5749 ms`** vs `0.9685 ms` unfiltered (**`1.68x` speedup**)
       - `mk38_pme_gather_compute_dq_false_ms`: **`0.0342 ms`** vs `0.0376 ms` (**`1.10x` speedup**)
     - **End-to-End `EnsemblePlan.run()` Throughput on `NVIDIA L40S` (`sm_89`, v3.0 vs v2.7)**:
       - **Small (`4,199` atoms in `8,192` bucket, `32^3` PME)**:
         - `single`: $B=1 \to \mathbf{0.734\text{ ms/step}}$ (**`235.57 ns/day`**, **`+19.7%`** vs v2.7), $B=2 \to \mathbf{0.488\text{ ms/step}}$ batch / **`0.244 ms/step/rep`** (**`708.54 ns/day` agg**, **`+24.1%`** vs v2.7, **`53.3%` of OpenMM mixed** / `53.0%` of OpenMM single), $B=32 \to 0.274\text{ ms/step/rep}$ (`630.90 ns/day`).
         - `mixed`: $B=1 \to \mathbf{0.628\text{ ms/step}}$ (**`275.13 ns/day`**, **`+10.5%`** vs v2.7), $B=2 \to \mathbf{0.337\text{ ms/step/rep}}$ (**`512.26 ns/day`**, **`+26.1%`** vs v2.7), $B=32 \to \mathbf{0.301\text{ ms/step/rep}}$ (`574.72 ns/day`).
       - **Medium (`6,658` atoms in `8,192` bucket, `48^3` PME)**:
         - `single`: $B=1 \to \mathbf{0.821\text{ ms/step}}$ (**`210.56 ns/day`**, **`+9.6%`** vs v2.7), $B=2 \to \mathbf{0.567\text{ ms/step}}$ batch / **`0.284 ms/step/rep`** (**`609.27 ns/day` agg**, **`+17.4%`** vs v2.7, **`49.0%` of OpenMM mixed**), $B=32 \to 0.307\text{ ms/step/rep}$ (`562.20 ns/day`).
         - `mixed`: $B=1 \to \mathbf{0.619\text{ ms/step}}$ (**`279.09 ns/day`**, **`+14.8%`** vs v2.7), $B=32 \to 0.348\text{ ms/step/rep}$ (`496.70 ns/day`).
       - **Solvated DHFR (`23,558` atoms in `25,000` bucket, `64^3` PME)**:
         - `single`: $B=1 \to \mathbf{1.751\text{ ms/step}}$ (**`98.67 ns/day`**, **`+11.7%`** vs v2.7's `88.37 ns/day`), $B=2 \to 1.486\text{ ms/step/rep}$ (`116.26 ns/day`), $B=4 \to \mathbf{1.475\text{ ms/step/rep}}$ (**`117.17 ns/day`**), $B=16 \to 1.486\text{ ms/step/rep}$ (`116.27 ns/day`).
         - `mixed`: $B=1 \to \mathbf{1.757\text{ ms/step}}$ (**`98.38 ns/day`**, **`+5.0%`** vs v2.7's `93.70 ns/day`), $B=16 \to 1.666\text{ ms/step/rep}$ (`103.73 ns/day`).
       - **Heterogeneous Same-Bucket (`4,199 + 6,656` in `8,192` bucket)**: **`0.521 ms/step` batch (`0.260 ms/step/system`)**, **`+21.0%` faster** than v2.7 (`0.630 ms/step`).
       - **Heterogeneous Multi-Bucket (`[8192, 25000, 8192, 25000]`)**: **`3.474 ms/step` total for 4 systems (`0.869 ms/step/system`)**, **`+9.2%` faster** than v2.7 (`3.795 ms/step`).

### MK-50: Zero-Overhead `init_fn` / `fast_reseed_fn`, Precomputed Step-0 Forces, & Single-JIT Fused `B=1` / `B>=2` Inference Loop (`src/prolix/physics/settle.py`, `src/prolix/api/ensemble_plan.py`)
- **Category**: `End-to-End Throughput` / `Compilation Cache Sharing`
- **Symptom / Failure Mode**:
  1. Inside `settle_langevin` (`src/prolix/physics/settle.py`), `init_fn` previously executed `jnp.isin(all_indices, real_idx_flat, assume_unique=False)` ($O(N \times 3N_w) = 25000 \times 24576 = 614.4\text{M}$ integer comparisons on DHFR) and `jax.random.split(rng, n_w + 1)` (`8,193` splits) + an `8,192`-row `vmap` over `_sample_water_momentum_single`, **and** recomputed `force_fn(position)` at step 0 on every `EnsemblePlan.run` invocation, even though `_iwg_precomputed` was already built right below `init_fn` and initial positions `r0` are invariant across cached runs!
  2. In `EnsemblePlan._run_single_inference` (`B=1`), cached neighbor-list runs previously executed 3 separate dispatches (`fast_reseed_fn`, `while_loop`, and `trim_trajectory_positions` with a synchronous `int(jnp.asarray(bundle.n_atoms))` device-to-host transfer) plus a synchronous `bool(final_carry.did_overflow)` check on every run. Furthermore, building `_fused_nl_run` after Call 1 without warming it left `_fused_nl_run` uncompiled on Call 2 (`26.2 s` XLA compile inside the timed warm window!).
  3. In `_build_stacked_neighbor_seed` and `_run_stacked_dispatch` (`B >= 2` heterogeneous mutant screens), when `excl_indices` varied across mutants, neighbor seeds were built via a sequential Python `for b in range(B):` loop instead of a single batched 2D-grid `_pallas_prune_and_encode_cell_list_batch` launch, and every distinct `bundles` object identity re-allocated `NeighborList` (`max_occupancy` could vary slightly across calls, changing `PyTreeDef`) and re-traced `_batched` across benchmark sections.
- **Root Cause**:
  - `init_fn` was defined before `_iwg_precomputed`, re-evaluated `force_fn(r0)` inside `fast_reseed_fn` on every run, `_fused_nl_run` was not warmed on Call 1, and `_run_stacked_dispatch` keyed its cache only on `id(bundles)` rather than sharing a process-wide `_STACKED_NL_SEED_CACHE[(B, N_pad)]` and `_STACKED_JIT_FN_CACHE`.
- **Generalizable Rule for `xtrax` Skill**:
  1. Precompute `_iwg_precomputed` (`int32[N]` with `1` for water/ghost atoms and `0` for non-water solute atoms) once at integrator construction time, derive `is_water_or_ghost = (_iwg_precomputed != 0)` in $O(N)$, vectorize rigid-water Maxwell-Boltzmann sampling via a single `jax.random.normal(split_w, (n_w, 6))` draw, and accept `precomputed_force` in `init_fn`.
  2. Fuse `B=1` cached neighbor-list execution (`fast_reseed_fn` + `while_loop` + real-atom trajectory slicing) into a single `@jax.jit` (`fused_nl_run_fn`), **warm `_fused_nl_run(rng)` on Call 1** (or when `verified_no_overflow` is set), and share `_STACKED_NL_SEED_CACHE[(B, N_pad)]` + `_STACKED_JIT_FN_CACHE` across homogeneous and heterogeneous multi-replica batches of the same `(B, bucket_size)`.
  3. **Measured Impact on `NVIDIA L40S` (`sm_89`, `23,558`-atom DHFR in `25,000` bucket)**:
     - `fast_reseed_fn` latency drops from `0.24 ms` (`B=1`) / `3.71 ms` (`B=16` legacy `jnp.isin`) to **`0.052 ms` (`B=1`) / `0.097 ms` (`B=16`)** (**`4.6x–38.2x` speedup**).
     - Warm `B=1` `EnsemblePlan.run(n_steps=200)` wall time drops from `131.028 ms/step` (`26.2 s` un-warmed `_fused_nl_run`) to **`0.588 ms/step` (homogeneous) / `0.566 ms/step` (heterogeneous)** (**`293.9 / 305.3 ns/day`**, matching pure `lax.while_loop` `0.581 ms/step` within `0.007 ms/step`).

### MK-51: Dynamic Per-Block `max_k_tiles[b, i_block]` K-Tile Early Exit & Spatially Sorted $32 \times 32$ Tile-Pair Newton's 3rd Law Nonbonded Kernel (`src/prolix/pallas_kernels.py`)
- **Category**: `Ragged-Bucket Compaction` / `Megakernel Fusion` / `Differentiation (JVP/VJP)`
- **Symptom / Failure Mode**:
  1. In `_nb_nl_kernel` and `_megakernel` (`_do_nonbonded_nl`), the inner K-tile loop previously ran `jax.lax.fori_loop(0, k_blocks, _k_tile_body, ...)` over all `k_blocks = 24` tiles (`K = 768`) even when a 32-atom row block's actual neighbor count was ~360 (`12` tiles), issuing `12` redundant `32x32` global memory gathers of `neighbor_idx` per CTA.
  2. In `_compute_max_k_tiles`, padding `nbr_b` (`(B, 25000, 768)`) along `N` (`25000 -> 25024`) **before** slicing `::_NB_BLOCK_K` copied the entire `76.8 MB * B` neighbor list (`1.23 GB` at `B=16`) on every single force evaluation step!
  3. Atom-by-atom `(N, K)` neighbor lists evaluate every interacting pair $(i, j)$ twice ($(i, j)$ in row $i$ and $(j, i)$ in row $j$) and perform scattered `32x32` global gathers for `j` coordinates and parameters.
- **Root Cause**:
  - Static `k_blocks` loop bound, pre-slice `jnp.pad` on `(B, N, K)`, and asymmetric `(N, K)` atom-row neighbor representation.
- **Generalizable Rule for `xtrax` Skill**:
  1. Compute per-block `max_k_tiles[b, i_block]` (`int32[B, n_blocks]`) via `_compute_max_k_tiles` by **slicing `first_cols = nbr_b[:, :, ::_NB_BLOCK_K]` (`(B, N, 24)`) FIRST** and only padding the small `bool[B, N, 24]` validity mask (`pad_n = 24`), eliminating the `76.8 MB * B` copy per force step, then pass `max_k_tiles_ref` into both `_nb_nl_kernel` and `_megakernel` so `jax.lax.fori_loop(0, max_k_blk, _k_tile_body, c0)` exits immediately after each 32-atom block's last valid neighbor tile.
  2. Provide `TilePairNeighborList`, `build_tile_pair_neighbor_list`, and `pallas_nonbonded_tile_pairs` (`32x32` spatially sorted tile-pair Newton's 3rd Law kernel with 32-bit topological exclusion bitmasks `excl_mask_full` and `excl_mask_14`): each interacting tile pair `(tile_I, tile_J)` loads two contiguous 32-atom blocks (`O(32)` coalesced loads instead of `O(1024)` scattered loads), evaluates all $32 \times 32$ interactions once (`0.5x` pair evaluations for `tile_I < tile_J`), accumulates $\mathbf{F}_I \mathrel{+}= \mathbf{f}_{ij}$ across columns and $\mathbf{F}_J \mathrel{-}= \mathbf{f}_{ij}$ down rows before issuing two vector `plt.atomic_add` writes, and preserves exact `@jax.custom_vjp` coordinate and parameter gradients (`positions, charges, sigmas, epsilons`).
  3. **Measured Impact on `NVIDIA L40S` (`sm_89`, `23,558`-atom DHFR)**:
     - `pallas_bonded_and_nonbonded_nl_fused` (`MK-48` + `MK-51` megakernel) runs in **`0.222 ms` (`B=1` homog) / `0.224 ms` (`B=1` hetero)** and **`0.200–0.202 ms/rep` (`B=8..16` homog) / `0.192–0.193 ms/rep` (`B=8..16` hetero)**.

### MK-52: $O(N)$ Pallas Atomic 3D Cell-List Binning with Per-Replica `rebuild_mask[b]` $O(1)$ Early Exit & Zero-Copy `old_idx` Aliasing (`src/prolix/physics/neighbor_list.py`)
- **Category**: `Multi-Replica Scaling` / `Grid & Work-Unit Dispatch` / `Buffer Aliasing & Atomics`
- **Symptom / Failure Mode**:
  1. In `_build_cell_stencil_buffers_batched`, binning `(B, 25000)` atoms via `jax.vmap(jnp.argsort)` executed a radix sort over all `25,000` atoms per replica (`0.106 ms/rep` at $B=1$, rising to `0.208 ms/rep` = `3.33 ms` at $B=16$).
  2. In `_pallas_prune_and_encode_cell_list_batch`, unconditionally executing `jnp.pad(old_idx.reshape(B * N * K), (0, 1))` and slicing `nbr_flat[:-1]` around `pallas_call(..., input_output_aliases={15: 0})` (originally added as a CPU `interpret=True` out-of-bounds write scratch slot) **forced XLA to allocate and copy a fresh `(B * N * K + 1)` buffer (`76.8 MB/rep` = `1.23 GB` at `B=16`) on every single MD step** (even when `do_rebuild == False`!) and made `input_output_aliases={15: 0}` alias a temporary copy rather than `old_idx`.
- **Root Cause**:
  - Full-array `jnp.argsort(c_id)` radix sort inside `jax.vmap(_bin_single)` and unconditional `jnp.pad(..., (0, 1))` around an in-place aliased `pallas_call` buffer.
- **Generalizable Rule for `xtrax` Skill**:
  1. Replace `jax.vmap(jnp.argsort)` on GPU (`not interpret`) with a 2D-grid `(B, cdiv(N, 128))` Pallas atomic cell-binning kernel (`_make_pallas_bin_cells_kernel`) that checks `reb_ref[b_id]` first (exiting in $O(1)$ when `rebuild_mask[b] == False`), claims a unique cell slot in $O(1)$ via `slot = plt.atomic_add(counts_out_ref, (b_id * (G_cells + 1) + safe_c,), ones, mask=valid_cell)`, and scatters `(offs, px, py, pz)` directly into the L2-resident `(B, G_cells, C)` cell buffers.
  2. **Never wrap an `input_output_aliases` buffer in `jnp.pad` or slice its output on GPU**: gate `use_scratch_slot = bool(interpret)` so GPU (`not interpret`) passes `old_idx.reshape(B * N * K)` directly into `pallas_call(..., input_output_aliases={15: 0})` with `mask=in_b & (k_slot < K_cap)` on the store, preserving true zero-copy in-place aliasing of the `76.8 MB/rep` neighbor table.
  3. **Measured Impact on `NVIDIA L40S` (`sm_89`, `23,558`-atom DHFR)**:
     - `MK-52` Pallas atomic cell binning runs in **`0.054 ms/rep` (`B=1`) to `0.062–0.069 ms/rep` (`B=16`)** vs `0.106 ms/rep` (`B=1`) to `0.208 ms/rep` (`B=16`) for `vmap(argsort)` (**`1.96x–3.35x` faster cell binning**).
     - Eliminating the `jnp.pad` scratch slot on `old_idx` cuts full NL rebuild time at `B=16` from `6.88 ms -> 4.24 ms` (all 16 replicas) and `5.54 ms -> 2.98 ms` (1-replica rebuild, **`1.86x` faster**), while cutting pure 200-step `lax.while_loop` step time at `B=16` from `0.858 ms/step/rep` to **`0.640 ms/step/rep`** (**`+34.1%` end-to-end throughput gain**, saving `0.218 ms/step/rep`).

### MK-53: Active-Charge CTA Gating & Spatial Sub-Grid Permutation for `64^3` Pallas SPME Spread/Gather + Verified `NVIDIA L40S` Multi-Bucket & Heterogeneous DHFR Mutant Screen (`src/prolix/physics/pme_pallas.py`, `src/prolix/physics/pme_op.py`, `docs/benchmarks/261106_l40s_pallas_v31_e2e_ensemble_buckets.json`)
- **Category**: `Buffer Aliasing & Atomics` / `Ragged-Bucket Compaction` / `Multi-Replica Scaling`
- **Symptom / Failure Mode**:
  1. In `_spread_charges_pallas_batch` and `_gather_pallas_batch`, CTAs containing only uncharged or ghost atoms (`active_q == False`) still entered `_do_spread` / `_do_gather` if `is_real` had any true lane, and `_gather_kernel` with `compute_dq=False` (the MD force path) still issued global `theta_flat_ref` gather loads for zero-charge atoms (`qi == 0`).
  2. Unsorted biomolecular atom order scatters adjacent threads across distant 3D PME mesh planes while placing bonded/water atoms in the same warp onto identical grid cells.
- **Root Cause**:
  - CTA entry was gated on `is_real` instead of `active_q` (`is_real & (jnp.abs(qi) > 0)`), and `pme_op.make_gather_op` lacked the `compute_dq: bool = True` parameter.
- **Generalizable Rule for `xtrax` Skill**:
  1. Gate `_spread_kernel` CTAs on `active_q = is_real & (jnp.abs(qi) > 0)` and gate `_gather_kernel` CTAs on `active_lane = is_real if compute_dq else (is_real & (jnp.abs(qi) > 0))`, using 1D-indexed scalar box loads `_pl_load(box_ref, (box_base + jnp.zeros((1,), dtype=jnp.int32),))[0]` (avoiding `TypedInt` scalar index `AttributeError` when `b_box=False` under `interpret=True`) and precomputed `iyz_16 = mesh_base + iy_16 * Kz + iz_16`.
  2. Provide `compute_pme_spatial_perm(positions, box_size, grid_dims, atom_mask)` (and `PROLIX_PME_SPATIAL_SORT=1` auto-dispatch in `spread_charges_pallas` / `gather_pallas`) to sort active atoms by $4 \times 4 \times 4$ PME sub-grid tiles and push ghost padding atoms (`~atom_mask`) to trailing CTAs where they exit in $O(1)$.
  3. Wire `excl_mode="tile_pairs"` (and `PROLIX_USE_TILE_PAIRS=1` / `isinstance(neighbor_idx, TilePairNeighborList)`) into both `pallas_nonbonded_nl` and `pallas_bonded_and_nonbonded_nl_fused` with symmetric `i <-> j` exclusion bitmask extraction, and ensure `_should_use_pallas_settle_default` defaults `PROLIX_USE_PALLAS_SETTLE` to `""` so CPU runs default to pure-JAX float64 (`< 1e-12 Å` 100-step parity vs `settle_langevin`) while GPU runs default to fused Pallas BAOAB+SETTLE (`MK-46`).
  4. **Verified `NVIDIA L40S` (`sm_89`, `node4102`, Jobs `28156459` & `28156471`) Benchmark Record**:
     - **DHFR Sub-Step Profile (`23,558` real atoms, `25,000` bucket, `64^3` PME)**:
       - `B=1`: `reseed=0.052ms` | `BAOAB=0.101ms/rep` | `MK48(hom/het)=0.222/0.224ms/rep` | `SPME=0.132ms/rep` (`spread=0.040`, `fft=0.054`, `gather=0.038`) | `NL_reb(all/1rep)=0.262/0.262ms` (`bin_argsort=0.106`, `bin_mk52=0.054/0.056ms`) | `while_loop=0.581ms/step/rep` | `plan.run(hom/het)=0.588/0.566ms/step/rep` (**`293.9 / 305.3 ns/day`**, **`3.09x` faster than v3.0's `98.67 ns/day`**).
       - `B=2`: `BAOAB=0.051ms/rep` | `MK48(hom/het)=0.211/0.204ms/rep` | `SPME=0.071ms/rep` | `NL_reb(all/1rep)=0.541/0.453ms` (`bin_mk52=0.057ms/rep`) | `while_loop=0.660ms/step/rep` | `plan.run(hom/het)=0.685/0.683ms/step/rep` (**`253.9 ns/day`**).
       - `B=4`: `BAOAB=0.027ms/rep` | `MK48(hom/het)=0.204/0.196ms/rep` | `SPME=0.047ms/rep` | `NL_reb(all/1rep)=1.073/0.821ms` (`bin_mk52=0.059ms/rep`) | `while_loop=0.653ms/step/rep` | `plan.run(hom/het)=0.677/0.676ms/step/rep` (**`258.4 ns/day`**).
       - `B=8`: `BAOAB=0.017ms/rep` | `MK48(hom/het)=0.200/0.192ms/rep` | `SPME=0.036ms/rep` | `NL_reb(all/1rep)=2.135/1.539ms` (`bin_mk52=0.060ms/rep`) | `while_loop=0.636ms/step/rep` | `plan.run(hom/het)=0.660/0.660ms/step/rep` (**`262.8 ns/day`**).
       - `B=16`: `BAOAB=0.011ms/rep` | `MK48(hom/het)=0.202/0.193ms/rep` | `SPME=0.029ms/rep` (`spread=0.013`, `fft=0.009`, `gather=0.007`) | `NL_reb(all/1rep)=4.242/2.978ms` (`bin_argsort=0.208`, `bin_mk52=0.069/0.062ms`) | `while_loop=0.640ms/step/rep` | `plan.run(hom/het)=0.666/0.666ms/step/rep` (**`259.5 ns/day`**, **`2.21x` faster than v3.0's `117.17 ns/day`**).
     - **Multi-Bucket Homogeneous & Heterogeneous Mutant Screen Throughput (`NVIDIA L40S`)**:
       - `Small (4,199 real / 8k bucket, 32^3 PME)`: **`0.134 ms/step/rep` (`1,291.06 ns/day` aggregate at `B=2`, `97.2%` of OpenMM CUDA `mixed` `1,328.60 ns/day`)**; `1,068.15 ns/day` at `B=32`.
       - `Medium (6,658 real / 8k bucket, 48^3 PME)`: **`0.143 ms/step/rep` (`1,206.31 ns/day` aggregate at `B=2`, `97.0%` of OpenMM CUDA `mixed` `1,243.65 ns/day`)**; `1,013.16 ns/day` at `B=32`.
       - `DHFR Heterogeneous Mutant Screen (23,501–23,558 real atoms / 25k bucket, 64^3 PME, point + insertion/deletion length variants)`: **`305.29 ns/day` (`0.566 ms/step/var` at `B=1`, `70.1%` of OpenMM CUDA `mixed` `435.58 ns/day`)**, **`253.89 ns/day` (`B=2`)**, **`258.39 ns/day` (`B=4`)**, **`262.78 ns/day` (`B=8`)**, **`259.47 ns/day` (`B=16`)**, and **`232.22 ns/day` (`0.744 ms/step/var` at `B=32`, `6,349 MB` peak GPU memory)** with zero recompilation across variants.

