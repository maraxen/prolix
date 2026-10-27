# Technical Specification: Full-Potential Pallas Kernel Suite & End-to-End Integration

**Date**: 2026-10-26  
**Status**: Frozen (`Verdict: ACCEPT` — 5-Round Adversarial Spec Review, 29/29 Findings Resolved)  
**Target Branch**: `feat/phase3-evidence-and-kups-261026` (base `b1/full`)  
**Scope**: Convert all remaining molecular mechanics energy/force kernels in Prolix to Pallas-optimized kernels with forward-force `@jax.custom_vjp` and `@jax.custom_batching.custom_vmap`, wire them into all full-system energy, force, and integrator factories (`make_energy_fn`, `make_energy_fn_pure`, `single_padded_energy`, `single_padded_energy_nl`, `single_padded_energy_nl_cvjp`, `single_padded_force`, `flash_explicit_forces`, `flash_explicit_total_energy`, `energy_fn_from_bundle`, `force_fn_from_bundle`, `bonded_energy_fn_from_bundle`, `make_langevin_step`, `make_langevin_step_nl`), fix batched PME rank-safety in `pme.py`, and benchmark the complete potential (bonded + 1-4 exceptions + direct NL nonbonded + PME reciprocal + dispersion tail + GBSA/SASA + SETTLE propagator).

---

## 1. Motivation & Problem Statement

A code review of PR #37 and the full Prolix energy stack (`src/prolix/physics/system.py`, `src/prolix/batched_energy.py`, `src/prolix/physics/flash_explicit.py`, `src/prolix/api/bundle_md.py`) identified two major gaps:

1. **Unwired Nonbonded Pallas Kernel in Full-System Factories**:
   Although `pallas_nonbonded_nl` and `pallas_flash_explicit_nonbonded` exist in `src/prolix/pallas_kernels.py`, none of the full-potential neighbor-list entry points (`make_energy_fn`, `make_energy_fn_pure`, `single_padded_energy`, `single_padded_energy_nl`, `single_padded_energy_nl_cvjp`, `energy_fn_from_bundle`) dispatch to `pallas_nonbonded_nl`, and `flash_explicit.py:162` silently bypasses `pallas_flash_explicit_nonbonded` under `jax.jit` because `soft_core_lambda` is a traced JAX scalar array.
2. **Remaining Non-Pallas Energy Kernels Across the Full Potential**:
   In a complete explicit- or implicit-solvent simulation step, total potential energy is:
   $$E_{\text{total}}(\mathbf{r}) = E_{\text{bond}} + E_{\text{UB}} + E_{\text{angle}} + E_{\text{dih}} + E_{\text{imp}} + E_{\text{CMAP}} + E_{\text{exc,1-4}} + E_{\text{vdW,dir}} + E_{\text{elec,dir}} + E_{\text{PME,recip}} + E_{\text{PME,bg}} + E_{\text{PME,excl}} + E_{\text{LJ,tail}} + E_{\text{GB}} + E_{\text{ACE/SASA}}$$
   Currently:
   - **Bonded terms** (`bonds`, `urey_bradley`, `angles`, `dihedrals`, `impropers`, `cmap`, `exception_pairs`, `pme_exclusion_correction`) run as 6–8 separate XLA gather graphs differentiated via reverse-mode AD (`jax.grad`), materializing intermediate displacement/cross-product/spline tensors in HBM and launching 6–8 separate scatter-add passes in the backward pass.
   - **Implicit-solvent neighbor-list terms** (`compute_born_radii_neighbor_list`, `compute_gb_energy_neighbor_list`, `compute_sasa_neighbor_list` in `src/prolix/physics/generalized_born.py`) materialize full `(N, K, 3)` neighbor coordinate tensors and `(N, K)` pair integral matrices in HBM.
   - **Parameter VJP in `_nonbonded_nl_param_grads`** (`src/prolix/pallas_kernels.py:1048`) uses an un-tiled `positions[safe_j]` gather of shape `(N, K, 3)` when differentiating with respect to `DifferentiableParams`, `_pallas_nonbonded_nl_single` (`line 715`) pads `excl_indices` along axis 0 by `N_pad - N` instead of `N_pad - excl_indices.shape[0]`, and `pallas_nonbonded_nl` / `pallas_flash_explicit_nonbonded` call `float(soft_core_lambda)` on traced JAX scalars.

---

## 2. Complete Energy Kernel Inventory & Pallas Optimization Matrix

| ID | Kernel Family | Existing XLA Implementation | Pallas Kernel & Strategy | Complexity & Memory |
| :--- | :--- | :--- | :--- | :--- |
| **R1** | **2-Body Bonded & Pair Corrections** (Bonds, Urey-Bradley, 1-4 Exceptions, PME Exclusion Pairs) | `_bond_energy_masked`, `_exception_energy_masked`, `_pme_exclusion_correction_from_pairs`, `make_bond_energy_fn`, `make_exception_pair_energy_fn` | `pallas_bonds_2body`, `pallas_exceptions_14`, `pallas_pme_exclusion_pairs`: 1D-tiled (`BLOCK_B = 64`), analytical pair forces accumulated into `(fx, fy, fz)` via `_pl_atomic_add_1d`, forward-force `custom_vjp` + `custom_vmap` | $O(N_{\text{pairs}})$ time, $O(N)$ HBM (0 AD intermediates) |
| **R2** | **3-Body Harmonic Angles** | `_angle_energy_masked`, `make_angle_energy_fn` | `pallas_angles_harmonic`: 1D-tiled (`BLOCK_A = 64`), 3-tuple vector algebra, analytical 3-atom forces on $(i, j, k)$ via `_pl_atomic_add_1d`, forward-force `custom_vjp` + `custom_vmap` | $O(N_{\text{angles}})$ time, $O(N)$ HBM |
| **R3** | **4-Body Torsions** (Proper Dihedrals, Periodic & Harmonic Impropers) | `_dihedral_energy_masked`, `make_dihedral_energy_fn`, `make_harmonic_improper_energy_fn` | `pallas_torsions_4body`: 1D-tiled (`BLOCK_D = 64`), 3-tuple `_cross3`/`_dot3` algebra, singularity-free analytical 4-atom torsion forces on $(i, j, k, l)$ via `_pl_atomic_add_1d`, supports both $\mathbf{b}_0$ conventions and periodic/harmonic impropers | $O(N_{\text{torsions}})$ time, $O(N)$ HBM |
| **R4** | **5-Body CMAP 2D Bicubic Torsions** | `_cmap_energy_masked`, `compute_cmap_energies` (`src/prolix/physics/cmap.py`) | `pallas_cmap_torsions`: 1D-tiled (`BLOCK_C = 32`), evaluates $(\phi, \psi)$ via 3-tuple algebra, supports `first_is_psi: bool`, loads 16 bicubic spline coefficients from 1D `coeffs_flat_ref`, scatters 5-atom forces via `_pl_atomic_add_1d` | $O(N_{\text{CMAP}})$ time, $O(N)$ HBM |
| **R5** | **Direct-Space Nonbonded $O(N\cdot K)$ & Flash $O(N^2)$** | `chunked_lj_energy_nl` + `chunked_coulomb_energy_nl`, `_lj_energy_neighbor_list` + `_coulomb_energy_neighbor_list`, `chunked_explicit_nonbonded_energy` | `pallas_nonbonded_nl` & `pallas_flash_explicit_nonbonded` (harden `excl_indices` row-padding, zero-box guard, traced `soft_core_lambda` `Ref`, undamped $r \ge 10^{-4}$ guard, mixed-axis `vmap` `_bwd`, `lax.scan`-chunked 6-tuple `_nonbonded_nl_param_grads`) | $O(N \cdot K)$ time, $O(N \cdot K)$ NL index only (0 $(N, K, 3)$ buffers) |
| **R6** | **Implicit Solvent GB/OBC-II + ACE Nonpolar + SASA ($O(N\cdot K)$)** | `compute_born_radii_neighbor_list`, `compute_gb_energy_neighbor_list`, `compute_sasa_neighbor_list` | `pallas_gb_ace_nl` (3-pass atomic-free `(32, 32)` kernel) + `pallas_sasa_nl` (2-pass atomic-free `(32, 32)` kernel) with symmetric-NL dual evaluation and forward-force `custom_vjp` + `custom_vmap` | $O(N \cdot K)$ time, $O(N)$ activation HBM |
| **R7** | **Reciprocal PME & Isotropic Tail** | `spme_energy_with_forces`, `spme_background_energy`, `lj_dispersion_tail_energy` | `spread_pallas` + 3D cuFFT + `gather_pallas` (wire `use_pallas_pme` through `_pme_reciprocal_and_corrections`, `single_padded_energy`, `energy_fn_from_bundle`) | $O(N + M^3 \log M^3)$ |
| **R8** | **Unified Full-Potential Wiring & End-to-End Propagator Benchmark** | Separate unwired paths in `system.py`, `batched_energy.py`, `flash_explicit.py`, `api/bundle_md.py` | Wire `use_pallas` across all factories; add `--kernel full_potential` and kUPS Tier 4 full solvated benchmark | Full potential + SETTLE step in 1 `value_and_grad` |

---

## 3. Detailed Kernel Specifications

### 3.1 Compiler, Safety, and Pallas Invariants (Mandatory Across R1–R6)

Every Pallas kernel MUST obey all rules documented in `.praxia/docs/reference/261023_xtrax_composition_and_pallas_optimization_guide.md` plus the following 7 compiler and safety invariants:

1. **1D `Ref` Indexing Only**: Split `(N, 3)` coordinates and forces into 1D `px_ref, py_ref, pz_ref` and `fx_out_ref, fy_out_ref, fz_out_ref` of shape `(N_atoms_pad,)`. Split multi-column index/parameter arrays (`(M, 2)`, `(M, 3)`, `(M, 4)`, `(M, 5)`) into 1D `Ref`s of shape `(M_pad,)`. Every `_pl_load`, `_pl_store`, and `_pl_atomic_add_1d` uses a 1D power-of-2 vector index (`BLOCK = 32` or `64`).
2. **Pure 3-Tuple 1D Vector Algebra Inside Kernel Bodies (resolves `R3-B1`)**:
   In Pallas Triton lowering, `jnp.cross` lowers to a 3-argument `jnp.stack`, which crashes with `NotImplementedError: Only 2-argument stack is supported in Triton`. Therefore, **no `(BLOCK, 3)` arrays, `jnp.cross`, or `jnp.stack` calls are permitted inside any Pallas kernel body**. All 3D vectors inside R1–R4 kernel bodies MUST be represented as Python 3-tuples `Vec3 = tuple[jax.Array, jax.Array, jax.Array]` of 1D `(BLOCK,)` arrays and manipulated via:
   ```python
   Vec3 = tuple[jax.Array, jax.Array, jax.Array]

   def _sub3(a: Vec3, b: Vec3) -> Vec3:
       return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

   def _add3(a: Vec3, b: Vec3) -> Vec3:
       return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

   def _scale3(s: jax.Array, a: Vec3) -> Vec3:
       return (s * a[0], s * a[1], s * a[2])

   def _dot3(a: Vec3, b: Vec3) -> jax.Array:
       return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

   def _cross3(a: Vec3, b: Vec3) -> Vec3:
       return (
           a[1] * b[2] - a[2] * b[1],
           a[2] * b[0] - a[0] * b[2],
           a[0] * b[1] - a[1] * b[0],
       )
   ```
3. **Duplicate-Index-Safe Atomic Accumulation (`_pl_atomic_add_1d`, resolves B1)**:
   Under `interpret=True` (CPU fallback), vectorized `plt.atomic_add(ref, (idx_1d,), val_1d, mask=mask_1d)` clobbers duplicate indices within the same block (cf. `src/prolix/physics/pme_pallas.py:149-156`). All R1–R4 bonded kernels MUST accumulate forces via:
   ```python
   def _pl_atomic_add_1d(
       ref,
       idx_1d: jax.Array,
       val_1d: jax.Array,
       mask_1d: jax.Array,
       interpret: bool,
   ) -> None:
       """1D atomic add that serializes duplicate-index updates under interpret=True."""
       zero = jnp.asarray(0.0, dtype=val_1d.dtype)
       val_masked = jnp.where(mask_1d, val_1d, zero)
       if interpret:
           block_size = idx_1d.shape[0]
           def _serial_body(k, _):
               plt.atomic_add(ref, (idx_1d[k],), val_masked[k])
               return None
           jax.lax.fori_loop(0, block_size, _serial_body, None)
       else:
           plt.atomic_add(ref, (idx_1d,), val_masked, mask=mask_1d)
   ```
4. **Empty-Table Guard & Sentinel-Safe Index Clamping (resolves B2, `R3-B4`)**:
   - **Batched-aware empty-table guard**: At the Layer 1 Python wrapper boundary of every R1–R4 kernel, if `indices is None or indices.shape[-2] == 0`, return `jnp.zeros(positions.shape[:-2], dtype=positions.dtype)` (shape `()` when unbatched `(N, 3)`, shape `(B,)` when batched `(B, N, 3)`) without launching `pallas_call`.
   - **Sentinel-safe clamping**: Padding rows in `MolecularBundle` and `PhysicsSystem` may contain sentinel `N_real` (e.g., `excl_pair_indices` in `bundle_md.py:484-486`), `-1`, or `0`. Inside every k-body kernel with raw atom indices $(a_0, \dots, a_{k-1})$:
     ```python
     in_bounds = (offs < M_real) & (_pl_load(mask_ref, (offs,)) > 0)
     for a_m in (a_0, ..., a_{k-1}):
         in_bounds = in_bounds & (a_m >= 0) & (a_m < N_real)
     safe_a_m = jnp.where(in_bounds, jnp.clip(a_m, 0, max(N_real - 1, 0)), 0)
     ```
     Every `_pl_load` and `_pl_atomic_add_1d` MUST use `safe_a_m` and `mask_1d=in_bounds`.
5. **Zero-Box-Safe Minimum-Image PBC Helper (resolves B7)**:
   `DifferentiableParams.from_system` (`src/prolix/typing.py:482`) sets `box_size = jnp.zeros((3,))` when `system.box_size is None`, and `MolecularBundle.box` is `jnp.zeros((3, 3))` when `has_pbc=False`. Every kernel applying PBC minimum-image wrapping MUST guard each axis against $\le 0$:
   ```python
   def _min_image_1d(d: jax.Array, box_len: jax.Array, has_box: bool) -> jax.Array:
       if not has_box:
           return d
       dtype = d.dtype
       half = jnp.asarray(0.5, dtype=dtype)
       one = jnp.asarray(1.0, dtype=dtype)
       b_safe = jnp.where(box_len > 0.0, box_len, one)
       return jnp.where(box_len > 0.0, d - box_len * jnp.floor(d / b_safe + half), d)

   def _min_image3(d: Vec3, box_xyz: Vec3, has_box: bool) -> Vec3:
       return (
           _min_image_1d(d[0], box_xyz[0], has_box),
           _min_image_1d(d[1], box_xyz[1], has_box),
           _min_image_1d(d[2], box_xyz[2], has_box),
       )
   ```
6. **Triton-Forbidden Primitives & `float64` Precision (resolves M1, `R3-M1`)**:
   - No unary negation `-x` on traced arrays inside the kernel body; use `zero - x`.
   - No `jnp.round`; use `_min_image_1d`.
   - No `jnp.sign`; use `jnp.where(x >= zero, one, neg_one)`.
   - Use `_triton_erf` / `_triton_erfc` on Triton GPU (`not interpret`), and exact `jax.scipy.special.erf` / `erfc` when `interpret=True and dtype == jnp.float64` (preserving `float64` OpenMM parity on CPU).
   - Pass `compiler_params=plt.CompilerParams() if not interpret else None`.
7. **3-Layer `public_fn -> @custom_vjp -> @custom_vmap` Architecture & Mixed-Axis `vmap` Support (resolves `R3-B3`, `R3-B4`)**:
   Every Pallas kernel (R1–R6, including `pallas_nonbonded_nl` and `pallas_flash_explicit_nonbonded`) MUST follow the 3-layer boundary pattern:
   - **Layer 1 (`public_fn`)**:
     - Checks `in_export_context()`; if `True`, delegates to the portable XLA reference function.
     - Checks `M_real == 0` (for R1–R4) and returns `jnp.zeros(positions.shape[:-2], dtype=positions.dtype)`.
     - Normalizes all optional inputs (`mask=None`, `atom_mask=None`, `box=None`, `scaled_radii=None`, `map_indices=None`) and traced scalars (`soft_core_lambda`) into concrete, non-`None` `jax.Array`s, and extracts `has_box: bool = (box is not None)`.
   - **Layer 2 (`@functools.partial(jax.custom_vjp, nondiff_argnums=...) def _diff`)**:
     - Receives ONLY non-`None` `jax.Array`s and static Python scalars (`has_box`, `interpret`, floats/bools).
     - In `_bwd`, supports both unbatched (`positions.ndim == 2`) and **arbitrary mixed-axis batched** (`positions.ndim == 3`, batch size $B$) execution:
       For each array argument $A_k$ in `res` with unbatched rank $r_k$, let $\text{batched}_k = (A_k.\text{ndim} > r_k)$. Inside `jax.lax.map(_step, jnp.arange(B))`, slice $A_k[b]$ if $\text{batched}_k$ else pass $A_k$. For each resulting cotangent $\text{grad\_}A_k$ of shape $(B, \dots)$, return $\text{grad\_}A_k$ if $\text{batched}_k$ else `jnp.sum(grad_A_k, axis=0)`.
   - **Layer 3 (`@jax.custom_batching.custom_vmap def _vmappable`)**:
     - Calls `_single(...)` when unbatched, and maps `_vmappable` over `jnp.arange(axis_size)` via `jax.lax.map` inspecting `in_batched` when batched.

---

### 3.2 R1: 2-Body Bonded & Pair-Correction Kernels (`pallas_bonds_2body`, `pallas_exceptions_14`, `pallas_pme_exclusion_pairs`)

#### 3.2.1 `pallas_bonds_2body` (Harmonic Bonds & Urey-Bradley, resolves `R5-M1`)
- **Signature**:
  `pallas_bonds_2body(positions, indices, params, mask=None, box=None, *, dist_eps: float = 1e-10, interpret=None) -> jax.Array`
  - Slices `idx_i = indices[..., :, 0].astype(jnp.int32)` and `idx_j = indices[..., :, 1].astype(jnp.int32)` to support both `(B, 2)` and `MolecularBundle`'s 3-column `(B, 3)` `urey_bradley_idx` (`src/prolix/physics/system.py:866`).
  - `dist_eps` (`R5-M1`): Defaults to `1e-10` for `bonded_masked.py` callers (`_bond_energy_masked` in `bonded_masked.py:42`); passed as `dist_eps=0.0` from `system.py` (`make_energy_fn` and `make_energy_fn_pure`) to match `jax_md.space.distance` in `bonded.py:39` to `< 1e-12` in `float64`.
- **Mathematical Formulation**:
  For each bond $b \in [0, B_{\text{real}})$ with valid mask $m_b \in \{0, 1\}$, equilibrium length $r_{0,b} = \text{params}[b, 0]$, force constant $k_b = \text{params}[b, 1]$, and $d_{\text{sq},b} = \text{dot3}(\mathbf{d}_b, \mathbf{d}_b)$:
  $$\mathbf{d}_b = \text{min\_image3}(\mathbf{r}_{i_b} - \mathbf{r}_{j_b}, \mathbf{L}), \quad r_b = \begin{cases} \sqrt{d_{\text{sq},b} + \text{dist\_eps}} & \text{if } \text{dist\_eps} > 0 \\ \text{where}\big(d_{\text{sq},b} > 0, \sqrt{\text{where}(d_{\text{sq},b} > 0, d_{\text{sq},b}, 1.0)}, 0.0\big) & \text{if } \text{dist\_eps} == 0 \end{cases}$$
  $$E_b = \frac{1}{2} m_b k_b (r_b - r_{0,b})^2, \quad r_{\text{denom},b} = \text{where}(r_b > 0, r_b, 1.0)$$
  $$\mathbf{f}_{i_b} = \text{where}\left(r_b > 0, \text{scale3}\left(\frac{\text{zero} - m_b k_b (r_b - r_{0,b})}{r_{\text{denom},b}}, \mathbf{d}_b\right), \mathbf{0}\right), \quad \mathbf{f}_{j_b} = \text{scale3}(-1, \mathbf{f}_{i_b})$$
- **Kernel Grid & Memory Layout**:
  - `BLOCK_B = 64`, `grid = (B_pad // 64,)`.
  - Inputs (1D `Ref`s): `px_ref, py_ref, pz_ref` `(N_pad,)`, `idx_i_ref, idx_j_ref` `(B_pad,)`, `r0_ref, k_ref, mask_ref` `(B_pad,)`, `box_ref` `(3,)`.
  - Outputs: `e_out_ref` `(n_blocks,)` (summed after `pallas_call`), `fx_out_ref, fy_out_ref, fz_out_ref` `(N_pad,)` (zero-initialized via `input_output_aliases` and updated via `_pl_atomic_add_1d` using `safe_i, safe_j` and `valid_b`).
- **Custom VJP & Parameter Gradients**:
  - Forward returns `energy` and saves `(forces, positions, indices, params, mask, box)` in residuals.
  - Backward returns:
    - `grad_positions = -g * forces` in $O(N)$ without launching a second kernel.
    - `grad_params`: analytical $\frac{\partial E_b}{\partial r_{0,b}} = -m_b k_b (r_b - r_{0,b})$, $\frac{\partial E_b}{\partial k_b} = \frac{1}{2} m_b (r_b - r_{0,b})^2$ (stacked on host as `(B, 2)`), supporting `jax.grad` wrt `DifferentiableParams.bond_params` and `urey_bradley_params`.

#### 3.2.2 `pallas_exceptions_14` (Explicit AMBER 1-4 Exception Pairs)
- **Signature**:
  `pallas_exceptions_14(positions, exception_pairs, exception_sigmas, exception_epsilons, exception_chargeprods, mask=None, box=None, coulomb_constant=332.0637, *, interpret=None) -> jax.Array`
- **Mathematical Formulation**:
  For each 1-4 exception pair $e \in [0, E_{\text{real}})$ with indices $(i_e, j_e)$, pre-scaled $\sigma_e, \epsilon_e, q_{\text{prod},e}$, and valid mask $m_e$:
  $$r_e = \sqrt{\text{dot3}(\mathbf{d}_e, \mathbf{d}_e) + 10^{-12}}, \quad u_e = (\sigma_e / r_e)^6$$
  $$E_e = m_e \left[ 4\epsilon_e (u_e^2 - u_e) + \frac{C q_{\text{prod},e}}{r_e} \right]$$
  $$\frac{1}{r_e}\frac{\partial E_e}{\partial r_e} = \frac{m_e}{r_e^2} \left[ 4\epsilon_e (-12 u_e^2 + 6 u_e) - \frac{C q_{\text{prod},e}}{r_e} \right], \quad \mathbf{f}_{i_e} = \text{scale3}\left(\text{zero} - \frac{1}{r_e}\frac{\partial E_e}{\partial r_e}, \mathbf{d}_e\right), \quad \mathbf{f}_{j_e} = \text{scale3}(-1, \mathbf{f}_{i_e})$$
- Uses `BLOCK_B = 64`, `_pl_atomic_add_1d`, forward-force `custom_vjp` + `custom_vmap`, and analytical parameter gradients wrt `(sigmas, epsilons, chargeprods)`.

#### 3.2.3 `pallas_pme_exclusion_pairs` (PME Reciprocal-Space Bonded Exclusion Correction)
- **Mathematical Formulation**:
  Matches `_pme_exclusion_correction_from_pairs` (`src/prolix/batched_energy.py:285-344`) and `_excl_correction_from_sys` (`src/prolix/physics/flash_explicit.py:402-425`):
  For each excluded pair $p \in [0, P_{\text{real}})$ with $(i_p, j_p)$, electrostatic scale $s_p \in [0, 1]$, valid mask $m_p = (\text{offs} < P_{\text{real}}) \land (i_p \ge 0) \land (i_p < N_{\text{real}}) \land (j_p \ge 0) \land (j_p < N_{\text{real}}) \land (i_p \neq j_p)$:
  $$r_{\text{raw},p} = \sqrt{\text{dot3}(\mathbf{d}_p, \mathbf{d}_p)}, \quad r_p = r_{\text{raw},p} + 10^{-6}$$
  $$E_p = -m_p (1 - s_p) C q_{i_p} q_{j_p} \frac{\text{erf}(\alpha r_p)}{r_p}$$
  $$\frac{\partial E_p}{\partial r_p} = -m_p (1 - s_p) C q_{i_p} q_{j_p} \frac{\frac{2\alpha}{\sqrt{\pi}} r_p e^{-\alpha^2 r_p^2} - \text{erf}(\alpha r_p)}{r_p^2}$$
  $$\mathbf{f}_{i_p} = \text{scale3}\left(\frac{\text{zero} - \frac{\partial E_p}{\partial r_p}}{\max(r_{\text{raw},p}, 10^{-12})}, \mathbf{d}_p\right), \quad \mathbf{f}_{j_p} = \text{scale3}(-1, \mathbf{f}_{i_p})$$
- **Call-Site Guard (resolves M2, `R4-B3`)**:
  Called in `_pme_reciprocal_and_corrections` (`batched_energy.py:407-420`) when `not direct_space_includes_exclusion_correction and sys.excl_pair_indices is not None and sys.excl_pair_indices.shape[0] > 0`, and in `_excl_correction_from_sys` (`flash_explicit.py:402-425`) when `excl_pair_indices is not None`.

---

### 3.3 R2: 3-Body Harmonic Angle Kernel (`pallas_angles_harmonic`, resolves `R5-M1`)

- **Signature**:
  `pallas_angles_harmonic(positions, indices, params, mask=None, box=None, *, dist_eps: float = 1e-10, interpret=None) -> jax.Array`
  - `dist_eps` (`R5-M1`): Defaults to `1e-10` for `bonded_masked.py` callers (`_angle_energy_masked` in `bonded_masked.py:72-73`); passed as `dist_eps=0.0` from `system.py` (`make_energy_fn` and `make_energy_fn_pure`) to match `bonded.py:72-73` in `float64`.
- **Mathematical Formulation (Pure 3-Tuple Algebra)**:
  For each angle $a \in [0, A_{\text{real}})$ with atoms $(i, j, k)$ ($j$ is the vertex atom), equilibrium angle $\theta_{0,a} = \text{params}[a, 0]$, force constant $k_a = \text{params}[a, 1]$, and valid mask $m_a$:
  $$\mathbf{v}_{ji} = \text{min\_image3}(\mathbf{r}_i - \mathbf{r}_j, \mathbf{L}), \quad \mathbf{v}_{jk} = \text{min\_image3}(\mathbf{r}_k - \mathbf{r}_j, \mathbf{L})$$
  $$d_{ji} = \begin{cases} \sqrt{\text{dot3}(\mathbf{v}_{ji}, \mathbf{v}_{ji}) + \text{dist\_eps}} & \text{if } \text{dist\_eps} > 0 \\ \text{where}(d_{\text{sq},ji} > 0, \sqrt{\text{where}(d_{\text{sq},ji} > 0, d_{\text{sq},ji}, 1.0)}, 0.0) & \text{if } \text{dist\_eps} == 0 \end{cases}, \quad d_{jk} = \text{analogous}, \quad \text{denom} = d_{ji} d_{jk} + 10^{-8}$$
  $$\text{dot}_{ik} = \text{dot3}(\mathbf{v}_{ji}, \mathbf{v}_{jk}), \quad c_{\text{raw}} = \frac{\text{dot}_{ik}}{\text{denom}}, \quad c = \text{clip}\left(c_{\text{raw}}, -0.999999, 0.999999\right), \quad \theta = \arccos(c)$$
  $$E_a = \frac{1}{2} m_a k_a (\theta - \theta_{0,a})^2$$
  $$\frac{\partial E_a}{\partial c} = \begin{cases} -m_a k_a (\theta - \theta_{0,a}) / \sqrt{1 - c^2} & \text{if } |c_{\text{raw}}| < 0.999999 \\ 0 & \text{otherwise} \end{cases}$$
  $$d_{ji,\text{safe}} = \text{where}(d_{ji} > 0, d_{ji}, 1.0), \quad d_{jk,\text{safe}} = \text{where}(d_{jk} > 0, d_{jk}, 1.0)$$
  $$\frac{\partial c}{\partial \mathbf{v}_{ji}} = \text{sub3}\left(\text{scale3}\left(\frac{1}{\text{denom}}, \mathbf{v}_{jk}\right), \text{scale3}\left(\frac{\text{dot}_{ik} d_{jk}}{\text{denom}^2 d_{ji,\text{safe}}}, \mathbf{v}_{ji}\right)\right)$$
  $$\frac{\partial c}{\partial \mathbf{v}_{jk}} = \text{sub3}\left(\text{scale3}\left(\frac{1}{\text{denom}}, \mathbf{v}_{ji}\right), \text{scale3}\left(\frac{\text{dot}_{ik} d_{ji}}{\text{denom}^2 d_{jk,\text{safe}}}, \mathbf{v}_{jk}\right)\right)$$
  $$\mathbf{f}_i = \text{scale3}\left(\text{zero} - \frac{\partial E_a}{\partial c}, \frac{\partial c}{\partial \mathbf{v}_{ji}}\right), \quad \mathbf{f}_k = \text{scale3}\left(\text{zero} - \frac{\partial E_a}{\partial c}, \frac{\partial c}{\partial \mathbf{v}_{jk}}\right), \quad \mathbf{f}_j = \text{scale3}(-1, \text{add3}(\mathbf{f}_i, \mathbf{f}_k))$$
- Scatters $\mathbf{f}_i, \mathbf{f}_j, \mathbf{f}_k$ into `fx_out_ref, fy_out_ref, fz_out_ref` via `_pl_atomic_add_1d`.

---

### 3.4 R3: 4-Body Torsion Kernel (`pallas_torsions_4body`, resolves B4, `R3-B1`)

- **Signature**:
  `pallas_torsions_4body(positions, indices, params, mask=None, box=None, *, b0_is_ji: bool = True, is_harmonic: bool = False, interpret=None) -> jax.Array`
- **Multi-Term `(N_dih, T, P)` Handling**:
  - If `params.ndim == 3` of shape `(N_dih, T, P)`:
    Record `orig_shape = params.shape`, flatten `indices_flat = jnp.repeat(indices, T, axis=0)`, `params_flat = params.reshape(N_dih * T, P)`, and `mask_flat = jnp.repeat(mask, T, axis=0)` at Layer 1, and reshape `grad_params_flat.reshape(orig_shape)` on return.
- **Exact Dual-Convention Geometry Using Pure 3-Tuple `_cross3` / `_dot3` Algebra**:
  For each torsion $d \in [0, D_{\text{real}})$ with atoms $(i, j, k, l)$ and valid mask $m_d$:
  1. **Bond vectors (as 3-tuples of `(BLOCK_D,)` arrays)**:
     - If `b0_is_ji=True` (matches `bonded_masked.py:_dihedral_energy_masked`):
       $$\mathbf{b}_0 = \text{min\_image3}(\mathbf{r}_j - \mathbf{r}_i, \mathbf{L}), \quad \mathbf{b}_1 = \text{min\_image3}(\mathbf{r}_k - \mathbf{r}_j, \mathbf{L}), \quad \mathbf{b}_2 = \text{min\_image3}(\mathbf{r}_l - \mathbf{r}_k, \mathbf{L})$$
       $$r_{b_1} = \sqrt{\text{dot3}(\mathbf{b}_1, \mathbf{b}_1) + 10^{-12}}, \quad \text{denom}_{b_1} = r_{b_1}, \quad \hat{\mathbf{b}}_1 = \text{scale3}(1 / \text{denom}_{b_1}, \mathbf{b}_1)$$
     - If `b0_is_ji=False` (matches `bonded.py:compute_dihedral_angles`, resolves `R5-M1`):
       $$\mathbf{b}_0 = \text{min\_image3}(\mathbf{r}_i - \mathbf{r}_j, \mathbf{L}), \quad \mathbf{b}_1 = \text{min\_image3}(\mathbf{r}_k - \mathbf{r}_j, \mathbf{L}), \quad \mathbf{b}_2 = \text{min\_image3}(\mathbf{r}_l - \mathbf{r}_k, \mathbf{L})$$
       $$b_{1,\text{sq}} = \text{dot3}(\mathbf{b}_1, \mathbf{b}_1), \quad r_{b_1} = \text{where}\big(b_{1,\text{sq}} > 0, \sqrt{\text{where}(b_{1,\text{sq}} > 0, b_{1,\text{sq}}, 1.0)}, 0.0\big), \quad r_{b_1,\text{safe}} = \text{where}(r_{b_1} > 0, r_{b_1}, 1.0)$$
       $$\text{denom}_{b_1} = r_{b_1} + 10^{-8}, \quad \hat{\mathbf{b}}_1 = \text{scale3}(1 / \text{denom}_{b_1}, \mathbf{b}_1)$$
  2. **Plane projections and angle**:
     $$\mathbf{v} = \text{sub3}(\mathbf{b}_0, \text{scale3}(\text{dot3}(\mathbf{b}_0, \hat{\mathbf{b}}_1), \hat{\mathbf{b}}_1)), \quad \mathbf{w} = \text{sub3}(\mathbf{b}_2, \text{scale3}(\text{dot3}(\mathbf{b}_2, \hat{\mathbf{b}}_1), \hat{\mathbf{b}}_1))$$
     $$x = \text{dot3}(\mathbf{v}, \mathbf{w}), \quad y = \text{dot3}(\text{cross3}(\hat{\mathbf{b}}_1, \mathbf{v}), \mathbf{w})$$
     $$\text{overlap} = (\neg m_d) \lor ((x == 0.0) \land (y == 0.0)), \quad x_s = \text{where}(\text{overlap}, 1.0, x), \quad y_s = \text{where}(\text{overlap}, 0.0, y)$$
     $$\phi = \begin{cases} \text{atan2}(y_s, x_s) - \pi & \text{if } \text{b0\_is\_ji} \\ \text{atan2}(y_s, x_s) & \text{if } \neg\text{b0\_is\_ji} \end{cases}$$
  3. **Periodic vs Harmonic Energy & $\frac{\partial E_d}{\partial \phi}$**:
     - If `not is_harmonic` (`params` columns `[n, gamma, k]`):
       $$E_d = m_d k_d (1 + \cos(n_d \phi - \gamma_d)), \quad g_\phi = \frac{\partial E_d}{\partial \phi} = -m_d k_d n_d \sin(n_d \phi - \gamma_d)$$
     - If `is_harmonic` (`params` columns `[k, phi0]`, matching `make_harmonic_improper_energy_fn` in `bonded.py:176-184`):
       $$\Delta\phi = (\phi - \phi_{0,d} + \pi) - 2\pi \left\lfloor \frac{\phi - \phi_{0,d} + \pi}{2\pi} \right\rfloor - \pi, \quad E_d = m_d k_d (\Delta\phi)^2, \quad g_\phi = \frac{\partial E_d}{\partial \phi} = 2 m_d k_d \Delta\phi$$
  4. **Exact Singularity-Free Chain Rule $(x, y) \to (\mathbf{b}_0, \mathbf{b}_1, \mathbf{b}_2) \to (\mathbf{f}_i, \mathbf{f}_j, \mathbf{f}_k, \mathbf{f}_l)$**:
     $$r_{xy}^2 = x_s^2 + y_s^2, \quad \frac{\partial \phi}{\partial x} = \text{where}\left(\text{overlap}, 0.0, \frac{\text{zero} - y_s}{r_{xy}^2}\right), \quad \frac{\partial \phi}{\partial y} = \text{where}\left(\text{overlap}, 0.0, \frac{x_s}{r_{xy}^2}\right)$$
     $$\mathbf{g}_{\mathbf{v}} = \text{add3}\left(\text{scale3}\left(\frac{\partial \phi}{\partial x}, \mathbf{w}\right), \text{scale3}\left(\frac{\partial \phi}{\partial y}, \text{cross3}(\mathbf{w}, \hat{\mathbf{b}}_1)\right)\right)$$
     $$\mathbf{g}_{\mathbf{w}} = \text{add3}\left(\text{scale3}\left(\frac{\partial \phi}{\partial x}, \mathbf{v}\right), \text{scale3}\left(\frac{\partial \phi}{\partial y}, \text{cross3}(\hat{\mathbf{b}}_1, \mathbf{v})\right)\right)$$
     $$\mathbf{g}_{\mathbf{b}_0} = \text{sub3}(\mathbf{g}_{\mathbf{v}}, \text{scale3}(\text{dot3}(\mathbf{g}_{\mathbf{v}}, \hat{\mathbf{b}}_1), \hat{\mathbf{b}}_1)), \quad \mathbf{g}_{\mathbf{b}_2} = \text{sub3}(\mathbf{g}_{\mathbf{w}}, \text{scale3}(\text{dot3}(\mathbf{g}_{\mathbf{w}}, \hat{\mathbf{b}}_1), \hat{\mathbf{b}}_1))$$
     $$\mathbf{g}_{\hat{\mathbf{b}}_1} = \text{add3}\Big(\text{scale3}\left(\frac{\partial \phi}{\partial y}, \text{cross3}(\mathbf{v}, \mathbf{w})\right), \text{sub3}\big(\text{sub3}(\mathbf{0}, \text{add3}(\text{scale3}(\text{dot3}(\mathbf{b}_0, \hat{\mathbf{b}}_1), \mathbf{g}_{\mathbf{v}}), \text{scale3}(\text{dot3}(\mathbf{g}_{\mathbf{v}}, \hat{\mathbf{b}}_1), \mathbf{b}_0))), \text{add3}(\text{scale3}(\text{dot3}(\mathbf{b}_2, \hat{\mathbf{b}}_1), \mathbf{g}_{\mathbf{w}}), \text{scale3}(\text{dot3}(\mathbf{g}_{\mathbf{w}}, \hat{\mathbf{b}}_1), \mathbf{b}_2))\big)\Big)$$
     $$\mathbf{g}_{\mathbf{b}_1} = \begin{cases} \text{scale3}(1 / r_{b_1}, \text{sub3}(\mathbf{g}_{\hat{\mathbf{b}}_1}, \text{scale3}(\text{dot3}(\mathbf{g}_{\hat{\mathbf{b}}_1}, \hat{\mathbf{b}}_1), \hat{\mathbf{b}}_1))) & \text{if } \text{b0\_is\_ji} \\ \text{sub3}\left(\text{scale3}(1 / \text{denom}_{b_1}, \mathbf{g}_{\hat{\mathbf{b}}_1}), \text{scale3}\left(\frac{\text{dot3}(\mathbf{g}_{\hat{\mathbf{b}}_1}, \mathbf{b}_1)}{r_{b_1,\text{safe}} \text{denom}_{b_1}^2}, \mathbf{b}_1\right)\right) & \text{if } \neg\text{b0\_is\_ji} \end{cases}$$
     Let $\mathbf{F}_{\mathbf{b}_m} = \text{scale3}(\text{zero} - g_\phi, \mathbf{g}_{\mathbf{b}_m})$ ($m \in \{0, 1, 2\}$):
     - When `b0_is_ji=True` ($\mathbf{b}_0 = \mathbf{r}_j - \mathbf{r}_i$):
       $$\mathbf{f}_i = \text{scale3}(-1, \mathbf{F}_{\mathbf{b}_0}), \quad \mathbf{f}_j = \text{sub3}(\mathbf{F}_{\mathbf{b}_0}, \mathbf{F}_{\mathbf{b}_1}), \quad \mathbf{f}_k = \text{sub3}(\mathbf{F}_{\mathbf{b}_1}, \mathbf{F}_{\mathbf{b}_2}), \quad \mathbf{f}_l = \mathbf{F}_{\mathbf{b}_2}$$
     - When `b0_is_ji=False` ($\mathbf{b}_0 = \mathbf{r}_i - \mathbf{r}_j$):
       $$\mathbf{f}_i = \mathbf{F}_{\mathbf{b}_0}, \quad \mathbf{f}_j = \text{sub3}(\text{scale3}(-1, \mathbf{F}_{\mathbf{b}_0}), \mathbf{F}_{\mathbf{b}_1}), \quad \mathbf{f}_k = \text{sub3}(\mathbf{F}_{\mathbf{b}_1}, \mathbf{F}_{\mathbf{b}_2}), \quad \mathbf{f}_l = \mathbf{F}_{\mathbf{b}_2}$$
     Scatters $\mathbf{f}_i, \mathbf{f}_j, \mathbf{f}_k, \mathbf{f}_l$ into `fx_out_ref, fy_out_ref, fz_out_ref` via `_pl_atomic_add_1d`.

---

### 3.5 R4: 5-Body CMAP Bicubic Spline Kernel (`pallas_cmap_torsions`, resolves B3, `R3-B1`)

- **Signature**:
  `pallas_cmap_torsions(positions, cmap_torsions, map_indices, cmap_coeffs, mask=None, box=None, *, first_is_psi: bool = False, interpret=None) -> jax.Array`
- **Mathematical Formulation**:
  Each CMAP torsion $c \in [0, C_{\text{real}})$ has 5 atoms $(a_0, a_1, a_2, a_3, a_4) = \text{cmap\_torsions}[c, :5]$ where $\phi$ is the dihedral $(a_0, a_1, a_2, a_3)$ and $\psi$ is the dihedral $(a_1, a_2, a_3, a_4)$ (both computed with the `b0_is_ji=False` 3-tuple helper from Section 3.4).
  1. If `map_indices is None`, default to `jnp.zeros((C_real,), dtype=jnp.int32)`.
  2. If `cmap_coeffs.ndim == 3` (`(N_maps, G, G)` raw grids), precompute `(N_maps, G, G, 16)` bicubic coefficients via `cmap.precompute_cmap_coefficients(cmap_coeffs)` outside `pallas_call`.
  3. **Axis Ordering (`first_is_psi`)**:
     - When `first_is_psi=False` (`make_energy_fn` in `system.py:109`): $(\theta_1, \theta_2) = (\phi, \psi)$.
     - When `first_is_psi=True` (`_cmap_energy_masked` in `bonded_masked.py:131`): $(\theta_1, \theta_2) = (\psi, \phi)$.
  4. In `pallas_cmap_torsions` (`BLOCK_C = 32`):
     - Compute $\phi \in [-\pi, \pi]$ and its 4-atom 3-tuple gradients $\frac{\partial \phi}{\partial \mathbf{r}_{a_0 \dots a_3}}$, and $\psi \in [-\pi, \pi]$ and its 4-atom 3-tuple gradients $\frac{\partial \psi}{\partial \mathbf{r}_{a_1 \dots a_4}}$.
     - Normalize $\theta_{1,\text{norm}} = (\theta_1 + 2\pi) - 2\pi \lfloor (\theta_1 + 2\pi)/(2\pi) \rfloor$, $\theta_{2,\text{norm}} = (\theta_2 + 2\pi) - 2\pi \lfloor (\theta_2 + 2\pi)/(2\pi) \rfloor$, grid spacing $\Delta = 2\pi / G$.
     - Grid cell indices: $i_1 = \lfloor \theta_{1,\text{norm}} / \Delta \rfloor \bmod G$, $i_2 = \lfloor \theta_{2,\text{norm}} / \Delta \rfloor \bmod G$; local coords $u = \theta_{1,\text{norm}} / \Delta - \lfloor \theta_{1,\text{norm}} / \Delta \rfloor$, $v = \theta_{2,\text{norm}} / \Delta - \lfloor \theta_{2,\text{norm}} / \Delta \rfloor$.
     - Clamp $m_{\text{safe}} = \text{where}(m_{\text{valid},c}, \text{clip}(m_c, 0, N_{\text{maps}} - 1), 0)$ and load the 16 bicubic coefficients $c_{p,q} = \text{coeffs}[m_{\text{safe}}, i_1, i_2, 4p + q]$ ($p, q \in \{0,1,2,3\}$) from 1D `coeffs_flat_ref` at base offset $((m_{\text{safe}} \cdot G + i_1) \cdot G + i_2) \cdot 16$:
       $$E_c = m_{\text{valid},c} \sum_{p=0}^3 \sum_{q=0}^3 c_{p,q} u^p v^q$$
       $$\frac{\partial E_c}{\partial \theta_1} = \frac{m_{\text{valid},c}}{\Delta} \sum_{p=1}^3 \sum_{q=0}^3 p \, c_{p,q} u^{p-1} v^q, \quad \frac{\partial E_c}{\partial \theta_2} = \frac{m_{\text{valid},c}}{\Delta} \sum_{p=0}^3 \sum_{q=1}^3 q \, c_{p,q} u^p v^{q-1}$$
     - Map $(\frac{\partial E_c}{\partial \theta_1}, \frac{\partial E_c}{\partial \theta_2})$ back to $(\frac{\partial E_c}{\partial \phi}, \frac{\partial E_c}{\partial \psi})$ according to `first_is_psi`, and scatter $-\left(\frac{\partial E_c}{\partial \phi} \frac{\partial \phi}{\partial \mathbf{r}} + \frac{\partial E_c}{\partial \psi} \frac{\partial \psi}{\partial \mathbf{r}}\right)$ onto atoms $(a_0, a_1, a_2, a_3, a_4)$ via `_pl_atomic_add_1d`.

---

### 3.6 R5: Hardening `pallas_nonbonded_nl` & `pallas_flash_explicit_nonbonded` (resolves B7, M5, `R3-B2`, `R3-B3`, `R3-B4`, `R4-B3`)

1. **Row-Padding Fix in `_pallas_nonbonded_nl_single` and `_pallas_flash_explicit_single`**:
   In `src/prolix/pallas_kernels.py:714-717, 1402-1405`, compute `pad_n_excl = N_pad - excl_indices.shape[0]` instead of `pad_n = N_pad - N` so systems with empty `(0, 0)` or unpadded `excl_indices` pad cleanly to `(N_pad, _NB_MAX_EXCL)`.
2. **Zero-Box Guard & Undamped $r \ge 10^{-4}$ Guard (`R3-B2`, B7)**:
   - Replace raw `dx / bx` with `_min_image_1d(dx, bx, has_box)` in both `_pallas_nonbonded_nl_single` and `_pallas_flash_explicit_single`.
   - When `pme_alpha == 0.0`, mask `e_coul` and `de_coul_dr` with `dist >= jnp.asarray(1e-4, dtype=dtype)` to match `_coulomb_energy_neighbor_list` (`batched_energy.py:274`) and `_make_coulomb_energy_nl_cvjp` (`batched_energy.py:883`).
3. **Traced `soft_core_lambda` `Ref` & Exact Beutler Soft-Core Formula (`R3-B3`, `R4-B3`)**:
   - Remove `soft_core_lambda` from `nondiff_argnums` in `_pallas_nonbonded_nl_diff` and `_pallas_flash_explicit_diff`; pass `lam_arr` `(1,)` as a 1D `Ref` into the kernel.
   - Compute `soft_alpha_lj = jnp.asarray(0.5, dtype=dtype) * jnp.where(lam < 1.0, jnp.maximum(1.0 - lam, 1e-8), 0.0)`, matching both `lam == 1.0` standard LJ and `lam < 1.0` soft-core Beutler LJ (`batched_energy.py:231-240, 811-817`).
   - Remove `and not isinstance(soft_core_lambda, jax.core.Tracer)` from `src/prolix/physics/flash_explicit.py:162` so `pallas_flash_explicit_nonbonded` executes under `jax.jit(single_padded_force)` and `force_fn_from_bundle` (`R4-B3`).
4. **Chunked Parameter Gradients & Mixed-Axis `vmap` in `_pallas_nonbonded_nl_bwd` (M5, `R3-B4`)**:
   - Replace the un-tiled `(N, K, 3)` gather in `_nonbonded_nl_param_grads` with a `jax.lax.scan` over atom chunks of size `chunk_size = min(N_pad, 256)`, returning `(grad_q, grad_sig, grad_eps, grad_excl_vdw, grad_excl_elec, grad_lam)` in $O(\text{chunk\_size} \cdot K)$ memory.
   - In `_pallas_nonbonded_nl_bwd`, replace the `jax.vmap` call with the mixed-axis `jax.lax.map` reduction pattern from Invariant 7 (`R3-B4`).

---

### 3.7 R6: Implicit Solvent GB/OBC-II + ACE Nonpolar + SASA Neighbor-List Pallas Kernels (`pallas_gb_ace_nl`, `pallas_sasa_nl`, resolves B5, M4, `R3-M1`, `R3-M2`, `R5-B2`, `R5-M2`)

#### 3.7.1 `pallas_gb_ace_nl` (OBC-II Generalized Born + ACE Nonpolar over Symmetric Neighbor List)
For implicit-solvent simulations with a symmetric neighbor list `neighbor_idx` of shape `(N, K)`:
- **`K == 0` Safe Neighbor-List Padding (`R5-M2`)**:
  In both `_pallas_gb_ace_nl_single` and `_pallas_sasa_nl_single`, pad `neighbor_idx` with sentinel `N_real` to `K_pad = max(1, (K + 31) // 32) * 32` (`k_blocks = K_pad // 32 >= 1`, `pad_n = N_pad - neighbor_idx.shape[0]`, `pad_k = K_pad - K`, `constant_values=N_real`) so `K == 0` neighbor lists never pass a 0-sized dimension or `k_blocks == 0` to `pallas_call`.
- **Export Context Fallback (`R5-B2`)**:
  When `in_export_context()` is `True`, mask `charges_masked = jnp.where(mask_bool, charges, 0.0)` and `ace_mask = mask_bool.astype(positions.dtype)` on `compute_ace_nonpolar_energy(radii, born_radii, dielectric_offset=dielectric_offset, mask=ace_mask)` so ghost atoms with default `radii = 1.5` Å contribute zero GB and ACE energy.
- **Active Neighbor Mask (`R3-M2`)**:
  Across all three passes of `pallas_gb_ace_nl` (and both passes of `pallas_sasa_nl`), for central block indices `offs_i` `(32,)` and neighbor indices `j_idx` `(32, 32)`:
  $$m_{ik} = (\text{offs}_i[:, \text{None}] < N_{\text{real}}) \land (j_{\text{idx}} \ge 0) \land (j_{\text{idx}} < N_{\text{real}}) \land (j_{\text{idx}} \neq \text{offs}_i[:, \text{None}]) \land (\text{mask}_i[:, \text{None}] > 0) \land (\text{mask}_j > 0)$$
- **Pass 1 (`_pallas_born_radii_and_dBdI_nl`)**:
  Grid `(N_pad // 32,)`, inner loop over `k_blocks = K_pad // 32` (`BLOCK_N = 32, BLOCK_K = 32`).
  With $\text{rad}_i = \max(\text{radii}_i, 10^{-6})$, $\text{or}_i = \max(\text{rad}_i - \text{offset}, 10^{-4})$, and $\text{sr}_j = (\text{scaled\_radii} \text{ if not None else } \text{rad})[j]$, accumulates $I_i = \sum_{k} m_{ik} I_{i, \text{nbr}(i,k)}$ in registers, then computes:
  $$\Psi_i = \text{or}_i \cdot I_i, \quad \text{arg}_i = \alpha \Psi_i - \beta \Psi_i^2 + \gamma \Psi_i^3$$
  $$r_{\text{safe},i} = \max(\text{rad}_i, 10^{-4}), \quad B_{\text{inv},i} = \text{or}_i^{-1} - r_{\text{safe},i}^{-1} \tanh(\text{arg}_i), \quad B_i = \frac{1}{\max(B_{\text{inv},i}, 10^{-4})}$$
  $$\frac{\partial B_i}{\partial I_i} = \mathbb{I}(B_{\text{inv},i} > 10^{-4}) B_i^2 \frac{\text{or}_i}{r_{\text{safe},i}} (1 - \tanh^2(\text{arg}_i)) (\alpha - 2\beta \Psi_i + 3\gamma \Psi_i^2)$$
  Stores `born_radii` `(N_pad,)` and `dB_dI` `(N_pad,)` to 1D output `Ref`s.
- **Pass 2 (`_pallas_gb_direct_and_dEdB_nl`)**:
  With $\tau = \epsilon_{\text{solute}}^{-1} - \epsilon_{\text{solvent}}^{-1}$ and $\text{prefactor} = -\frac{1}{2} C_{\text{Coulomb}} \tau$:
  - **Self GB + ACE per-atom energy and $\frac{\partial E}{\partial B_i}$ on host/XLA in $O(N)$ (`R5-B2`)**:
    With $\text{radius\_nm}_i = (\text{radii}_i - \text{dielectric\_offset}) \cdot 0.1 + 0.009$ (matching `compute_ace_nonpolar_energy` in `generalized_born.py:632-637` for arbitrary `dielectric_offset`):
    $$E_{\text{GB,self}} = \text{prefactor} \sum_{i=0}^{N-1} m_i \frac{q_i^2}{B_i}, \quad \frac{\partial E_{\text{GB,self}}}{\partial B_i} = -\text{prefactor} \cdot m_i \frac{q_i^2}{B_i^2}$$
    $$E_{\text{ACE},i} = m_i \cdot 28.3919551 \cdot 0.239006 \cdot (\text{radius\_nm}_i + 0.14)^2 \left(\frac{\text{radius\_nm}_i}{B_i / 10}\right)^6, \quad \frac{\partial E_{\text{ACE},i}}{\partial B_i} = -6 \frac{E_{\text{ACE},i}}{B_i}$$
  - **Neighbor-pair GB energy, direct force, $\frac{\partial E_{\text{GB,pair}}}{\partial B_i}$, and $\frac{\partial E_{\text{GB}}}{\partial q_i}$ in a `(32, 32)` Pallas kernel**:
    For each active neighbor $j = \text{nbr}(i, k)$ with $r_{ij} = \sqrt{\|\mathbf{r}_i - \mathbf{r}_j\|^2 + 10^{-12}}$, $P_{ij} = B_i B_j$, $u_{ij} = \frac{r_{ij}^2}{4 P_{ij}}$, $e_{ij} = \exp(-u_{ij})$, $f_{\text{GB}} = \sqrt{r_{ij}^2 + P_{ij} e_{ij}}$:
    $$E_{\text{GB,pair}} = \text{prefactor} \sum_{i, k} m_{ik} \frac{q_i q_j}{f_{\text{GB}}}$$
    $$\mathbf{f}_{i,\text{direct}} = 2 \cdot \text{prefactor} \sum_{k=0}^{K-1} m_{ik} q_i q_j \frac{1 - \frac{1}{4} e_{ij}}{f_{\text{GB}}^3} (\mathbf{r}_i - \mathbf{r}_j)$$
    $$\frac{\partial E_{\text{GB,pair}}}{\partial B_i} = -\text{prefactor} \sum_{k=0}^{K-1} m_{ik} q_i q_j \frac{B_j e_{ij} (1 + u_{ij})}{f_{\text{GB}}^3}$$
    $$\frac{\partial E_{\text{GB}}}{\partial q_i} = 2 \cdot \text{prefactor} \cdot m_i \left(\frac{q_i}{B_i} + \sum_{k=0}^{K-1} m_{ik} \frac{q_j}{f_{\text{GB}}}\right)$$
- **Pass 3 (`_pallas_gb_born_chain_forces_nl` — 3-Regime Piecewise Derivative & Atomic-Free Symmetric-NL Accumulation)**:
  Given total $w_i = \left(\frac{\partial E_{\text{GB,self}}}{\partial B_i} + \frac{\partial E_{\text{ACE},i}}{\partial B_i} + \frac{\partial E_{\text{GB,pair}}}{\partial B_i}\right) \frac{\partial B_i}{\partial I_i}$:
  Define `_pair_integral_deriv(r, or_a, sr_b)` differentiating `compute_pair_integral(r, or_a, sr_b)` (`generalized_born.py:238-263`) wrt $r$:
  $$D = |r - \text{sr}_b|, \quad L = \max(\text{or}_a, D), \quad L_s = \max(L, 10^{-4}), \quad U = r + \text{sr}_b, \quad U_s = \max(U, 10^{-4}), \quad r_s = \max(r, 10^{-4})$$
  $$\frac{\partial \text{total}}{\partial L_s} = -\frac{1}{2} L_s^{-2} + \frac{1}{4 r_s} L_s^{-1} + \frac{1}{4}\left(r - \frac{\text{sr}_b^2}{r_s}\right) L_s^{-3}, \quad \frac{\partial \text{total}}{\partial U_s} = \frac{1}{2} U_s^{-2} - \frac{1}{4 r_s} U_s^{-1} - \frac{1}{4}\left(r - \frac{\text{sr}_b^2}{r_s}\right) U_s^{-3}$$
  $$\frac{\partial \text{total}}{\partial r_s} = -\frac{1}{4}\frac{\ln(L_s / U_s)}{r_s^2} + \frac{1}{8}\frac{\text{sr}_b^2}{r_s^2}(U_s^{-2} - L_s^{-2}), \quad \frac{\partial \text{total}}{\partial r_{\text{exp}}} = \frac{1}{8}(U_s^{-2} - L_s^{-2})$$
  $$\frac{\partial L_s}{\partial r} = \mathbb{I}(D > \text{or}_a) \mathbb{I}(L > 10^{-4}) \, \text{where}(r \ge \text{sr}_b, 1.0, -1.0), \quad \frac{\partial U_s}{\partial r} = \mathbb{I}(U > 10^{-4}), \quad \frac{\partial r_s}{\partial r} = \mathbb{I}(r > 10^{-4})$$
  $$\frac{\partial I(r, \text{or}_a, \text{sr}_b)}{\partial r} = \mathbb{I}(r + \text{sr}_b > \text{or}_a)\left(\frac{\partial \text{total}}{\partial L_s}\frac{\partial L_s}{\partial r} + \frac{\partial \text{total}}{\partial U_s}\frac{\partial U_s}{\partial r} + \frac{\partial \text{total}}{\partial r_s}\frac{\partial r_s}{\partial r} + \frac{\partial \text{total}}{\partial r_{\text{exp}}}\right)$$
  Because `neighbor_idx` is symmetric ($j \in \text{nbr}(i) \iff i \in \text{nbr}(j)$), row $i$ loads both $(w_i, \text{or}_i, \text{sr}_i)$ and $(w_j, \text{or}_j, \text{sr}_j)$ and accumulates both $(i, j)$ and $(j, i)$ chain-rule forces in registers with **zero `atomic_add`**:
  $$\mathbf{f}_{i,\text{born}} = -\sum_{k=0}^{K-1} m_{ik} \left( w_i \frac{\partial I(r_{ij}, \text{or}_i, \text{sr}_j)}{\partial r_{ij}} + w_j \frac{\partial I(r_{ij}, \text{or}_j, \text{sr}_i)}{\partial r_{ij}} \right) \frac{\mathbf{r}_i - \mathbf{r}_j}{r_{ij}}$$
  Total force is $\mathbf{f}_i = \mathbf{f}_{i,\text{direct}} + \mathbf{f}_{i,\text{born}}$, saved with `grad_q` in `@jax.custom_vjp` residuals. `_pallas_gb_ace_nl_bwd` returns `(-g * forces, g * grad_q, jnp.zeros_like(radii), jnp.zeros_like(scaled_radii), jnp.zeros_like(atom_mask), jnp.zeros_like(neighbor_idx))`.

#### 3.7.2 `pallas_sasa_nl` (Neighbor-List SASA Pallas Kernel, resolves M4, `R3-M2`, `R5-B2`, `R5-M2`)
Matches `compute_sasa_neighbor_list` and `compute_nonpolar_energy` (`src/prolix/physics/generalized_born.py:498-598`):
- **Ghost-Atom Masking in `compute_sasa_neighbor_list` (`R5-B2`)**:
  Add optional `atom_mask: Array | None = None` to `compute_sasa_neighbor_list` in `src/prolix/physics/generalized_born.py:498`: when `atom_mask is not None`, replace any neighbor entry where `~(mask_bool[:, None] & mask_bool[safe_j])` with sentinel `N` before computing $b_{ij}$, and zero out `A_i = jnp.where(mask_bool, A_i, 0.0)` so padded ghost atoms with `radii = 1.5` Å never contribute $+105.68\text{ \AA}^2$. `pallas_sasa_nl` passes `atom_mask=atom_mask` to `compute_sasa_neighbor_list` in its `in_export_context()` fallback.
- Let $R_i = r_i + r_{\text{probe}}$ and $S_i = 4\pi R_i^2$.
- For atom $i$ and neighbor $j = \text{nbr}(i, k)$ with active mask $m_{ik}$, $d_{ij} = \sqrt{\|\mathbf{r}_i - \mathbf{r}_j\|^2 + 10^{-12}}$, and $d_{\text{safe}} = d_{ij} + 10^{-6}$:
  - Overlap regimes:
    $$\text{no\_overlap} = d_{ij} \ge R_i + R_j, \quad \text{i\_in\_j} = R_j \ge R_i + d_{ij}, \quad \text{j\_in\_i} = R_i \ge R_j + d_{ij}$$
    $$\text{smooth}_{ij} = m_{ik} \land (\neg\text{no\_overlap}) \land (\neg\text{i\_in\_j}) \land (\neg\text{j\_in\_i})$$
    $$b_{ij} = \begin{cases} \frac{\pi R_i}{d_{\text{safe}}}\left(R_j^2 - (d_{ij} - R_i)^2\right) & \text{if } \text{smooth}_{ij} \\ S_i & \text{if } m_{ik} \land \text{i\_in\_j} \land (\neg\text{j\_in\_i}) \\ 0 & \text{otherwise} \end{cases}$$
    $$\frac{\partial b_{ij}}{\partial d_{ij}} = \begin{cases} \pi R_i \left[ -\frac{R_j^2 - (d_{ij} - R_i)^2}{d_{\text{safe}}^2} - \frac{2(d_{ij} - R_i)}{d_{\text{safe}}} \right] & \text{if } \text{smooth}_{ij} \\ 0 & \text{otherwise} \end{cases}$$
    $$f_{ij} = \max\left(0, 1 - \frac{b_{ij}}{S_i}\right)$$
- **Pass 1 (`_pallas_sasa_area_nl`)**:
  In a `(32, 32)` Pallas kernel, accumulates `log_sum_i = sum_k where(f_ij <= 1e-12, 0.0, log(max(f_ij, 1e-12)))` and `zero_count_i = sum_k (f_ij <= 1e-12)` across `k_blocks`, storing $A_i = \text{where}((\text{offs}_i < N_{\text{real}}) \land (\text{mask}_i > 0) \land (\text{zero\_count}_i == 0), S_i \exp(\text{log\_sum}_i), 0.0)$.
- **Pass 2 (`_pallas_sasa_forces_nl`)**:
  In a `(32, 32)` Pallas kernel over the symmetric neighbor list, computes for each active pair $(i, j)$:
  $$\frac{\partial A_i}{\partial d_{ij}} = \text{where}\left((\text{zero\_count}_i == 0) \land (f_{ij} > 10^{-12}), -\frac{A_i}{S_i f_{ij}} \frac{\partial b_{ij}(d_{ij}, R_i, R_j)}{\partial d_{ij}}, 0.0\right)$$
  $$\frac{\partial A_j}{\partial d_{ij}} = \text{where}\left((\text{zero\_count}_j == 0) \land (f_{ji} > 10^{-12}), -\frac{A_j}{S_j f_{ji}} \frac{\partial b_{ji}(d_{ij}, R_j, R_i)}{\partial d_{ij}}, 0.0\right)$$
  and accumulates $\mathbf{f}_{i,\text{SASA}} = -\gamma_{\text{SASA}} \sum_{k=0}^{K-1} m_{ik} \left(\frac{\partial A_i}{\partial d_{ij}} + \frac{\partial A_j}{\partial d_{ij}}\right) \frac{\mathbf{r}_i - \mathbf{r}_j}{d_{ij}}$ purely in registers with **zero `atomic_add`**. `_pallas_sasa_nl_bwd` returns `(-g * forces, jnp.zeros_like(radii), jnp.zeros_like(atom_mask), jnp.zeros_like(neighbor_idx))`.

---

### 3.8 R7: Unified Full-Potential Wiring (`system.py`, `batched_energy.py`, `flash_explicit.py`, `api/bundle_md.py`, `pme.py`, `batched_simulate.py`, resolves B6, B7, M1–M3, m1, `R3-B2`, `R3-B5`, `R4-B1` to `R4-B3`, `R4-M1`, `R5-B1`, `R5-B3`, `R5-M1`)

1. **`src/prolix/physics/pme.py` (Batched `vmap` Rank & Axis Hardening, resolves `R5-B1`)**:
   - In `spme_reciprocal_energy` (`pme.py:247-262`), `_spme_fwd` (`pme.py:365-371`), `_spme_energy_with_forces_cached` (`pme.py:440-460`), and `_spme_cached_fwd` (`pme.py:463-477`):
     - Pass explicit `axes=(-3, -2, -1)` to `jnp.fft.rfftn(charge_grid, s=(Gx, Gy, Gz), axes=(-3, -2, -1))` and `jnp.fft.irfftn(phi_k, s=(Gx, Gy, Gz), axes=(-3, -2, -1))` so batched `(B, Gx, Gy, Gz)` grids never transform along the batch axis `0`.
     - Replace scalar `jnp.sum(...)` with `e_recip = 0.5 * COULOMB_CONSTANT * jnp.sum(charge_grid * phi_grid, axis=(-3, -2, -1))` (shape `()` when unbatched, `(B,)` when batched).
   - In `spme_self_energy` (`pme.py:297`) and `spme_background_energy` (`pme.py:310`):
     - Sum over the atom axis `axis=-1` (`jnp.sum(charges**2, axis=-1)` and `jnp.sum(charges, axis=-1)`) so batched `(B, N)` charges return `(B,)` energies matching `e_recip`.
   - In `_spme_bwd` (`pme.py:409-431`) and `_spme_cached_bwd`:
     - Broadcast batched cotangents `g` of shape `(B,)` cleanly against `(B, N, 3)` forces and `(B, N)` charge gradients via `g_pos = jnp.expand_dims(g, axis=(-2, -1)) if g.ndim > 0 else g` and `g_q = jnp.expand_dims(g, axis=-1) if g.ndim > 0 else g`.
     - Reduce unbatched `charges` / `box_size` gradients over leading batch axes (`while dE_dq.ndim > charges.ndim: dE_dq = jnp.sum(dE_dq, axis=0)` and `while dE_dbox.ndim > box_size.ndim: dE_dbox = jnp.sum(dE_dbox, axis=0)`) so `jax.vmap(jax.grad(...), in_axes=(0, None, None))` never raises a shape mismatch.
2. **`src/prolix/physics/system.py` (`make_energy_fn` and `make_energy_fn_pure`)**:
   - Accept `use_pallas: bool | None = None` and `gb_use_neighbor_list: bool = False`.
   - **Hardening `make_energy_fn_pure` (`R4-B1`, `R3-B5`)**:
     - Set `has_pbc = (physics_system.box_size is not None) and bool(kwargs.get("use_pbc", physics_system.box_size is not None))`. Only compute `grid_dims` when `has_pbc` is `True`.
     - Normalize `excl_idx, excl_sv, excl_se` at factory setup time identically to `make_energy_fn` (`system.py:161-171`), supporting `exclusion_spec` and defaulting `None` to `jnp.zeros((n_atoms, 0), dtype=jnp.int32)`.
     - Fix the `use_pallas=False` improper branch (`system.py:315-318`) so that when `physics_system.improper_params.shape[-1] == 2`, it dispatches to `bonded.make_harmonic_improper_energy_fn(displacement_fn, imp_idx)` (`R3-B5`).
     - Inside `total_energy_pure_impl`, guard `e_recip` (`spme_energy_with_forces + spme_background_energy`) behind `if has_pbc and not implicit_solvent and float(pme_alpha) > 0.0:` and `e_tail` (`lj_dispersion_tail_energy`) behind `if has_pbc and not implicit_solvent:`.
   - **Pallas Dispatch in `make_energy_fn` and `make_energy_fn_pure` (`R4-B2`, `R5-M1`)**:
     - Define `pallas_box = jnp.asarray(box) if (use_pbc and box is not None) else None` (in `make_energy_fn`) and `pallas_box = box_arr if has_pbc else None` (in `make_energy_fn_pure`).
     - When `resolve_pallas_mode(use_pallas)[0]` is `True`:
       - Bonded component callables (`bond`, `urey_bradley`, `angle`, `dihedral`, `improper`, `cmap`, `exception`) dispatch to `pallas_bonds_2body(..., dist_eps=0.0)`, `pallas_angles_harmonic(..., dist_eps=0.0)`, `pallas_torsions_4body(..., b0_is_ji=False, is_harmonic=(imp_prm.shape[-1] == 2))`, `pallas_cmap_torsions(..., first_is_psi=False)`, and `pallas_exceptions_14` (preserving `return_decomposed=True` component parity and `float64` `< 1e-12` parity).
       - When `neighbor is not None`, `pallas_nonbonded_nl` replaces **only** the direct-space pair sums (`chunked_lj_energy_nl` + `chunked_coulomb_energy_nl`), while still adding `lj_dispersion_tail_energy` (when `use_pbc and box is not None`), `_spme + spme_background_energy` (when `not implicit_solvent and _spme is not None`), and `e_gb + e_ace` (when `implicit_solvent=True`, using `pallas_gb_ace_nl` when `gb_use_neighbor_list=True`).
3. **`src/prolix/batched_energy.py`, `src/prolix/physics/flash_explicit.py`, & `src/prolix/batched_simulate.py` (`R4-B3`, `R4-M1`, `R5-B3`)**:
   - Define `_box_for_displacement_fn(sys_box, displacement_fn)` (`R4-M1`) returning `jnp.asarray(sys_box)` iff `sys_box is not None and getattr(displacement_fn, "__closure__", None) is not None`, else `None`.
   - Add `use_pallas: bool | None = None` and `use_pallas_pme: bool | None = None` to `single_padded_energy`, `single_padded_energy_nl_cvjp`, `single_padded_energy_nl`, `single_padded_force`, and `_pme_reciprocal_and_corrections`.
   - In `src/prolix/batched_simulate.py` (`make_langevin_step` at line 58 and `make_langevin_step_nl` at line 277), accept `use_pallas: bool | None = None` and forward it to `single_padded_energy`, `single_padded_energy_nl_cvjp`, and `single_padded_energy_nl` (`R5-B3`).
   - In `flash_explicit.py`, remove `and not isinstance(soft_core_lambda, jax.core.Tracer)` from line 162, dispatch `_excl_correction_from_sys` (`lines 402-425`) to `pallas_pme_exclusion_pairs` when `pallas_enabled`, and dispatch bonded terms in `flash_explicit_total_energy` (`lines 743-761`) to R1–R4 Pallas kernels when `pallas_enabled`.
   - When `resolve_pallas_mode(use_pallas)[0]` is `True`:
     - Bonded terms in `single_padded_energy`, `single_padded_energy_nl_cvjp`, `single_padded_energy_nl`, and `single_padded_force` dispatch to `pallas_bonds_2body(..., dist_eps=1e-10)`, `pallas_angles_harmonic(..., dist_eps=1e-10)`, `pallas_torsions_4body(..., b0_is_ji=True, is_harmonic=False)`, and `pallas_cmap_torsions(..., first_is_psi=True)` with `box=_box_for_displacement_fn(sys.box_size, displacement_fn)`.
     - In `single_padded_energy` (when `neighbor is not None`), direct-space LJ + Coulomb dispatches to `pallas_nonbonded_nl(..., box=_box_for_displacement_fn(sys.box_size, displacement_fn), pme_alpha=float(sys.pme_alpha), cutoff=float(sys.nonbonded_cutoff), lj_switch_width=float(lj_switch_width), soft_core_lambda=soft_core_lambda, include_pme_excl_erf=True)`.
     - In `single_padded_energy_nl` and `single_padded_energy_nl_cvjp`, direct-space LJ + Coulomb dispatches to `pallas_nonbonded_nl(..., box=None, pme_alpha=0.0, cutoff=1e9, lj_switch_width=0.0, soft_core_lambda=soft_core_lambda, include_pme_excl_erf=False)` (`R3-B2`), and GB + ACE dispatches to `pallas_gb_ace_nl` when `implicit_solvent=True`.
4. **`src/prolix/api/bundle_md.py` (`physics_system_from_bundle`, `energy_fn_from_bundle`, `force_fn_from_bundle`, `bonded_energy_fn_from_bundle`, resolves `R4-B3`, `R4-M1`)**:
   - In `physics_system_from_bundle` (`bundle_md.py:449, 524`), check `is_periodic = (bundle.shape_spec.boundary_condition == "periodic" and bundle.shape_spec.has_pbc)` (`R4-M1`).
   - Thread `use_pallas: bool | None = None` and `use_pallas_pme: bool | None = None` into `energy_fn_from_bundle`, `force_fn_from_bundle`, and `bonded_energy_fn_from_bundle` (`_bonded_energy_core`), dispatching `_exception_energy_masked` to `pallas_exceptions_14` when `pallas_enabled`.

---

### 3.9 R8: Full-Potential Verification & Benchmark Suite

1. **Unit & Parity Test Suite (`tests/physics/test_pallas_full_potential_parity.py`)**:
   - Individual kernel parity (`interpret=True` on CPU and Triton on GPU) for:
     - `pallas_bonds_2body`, `pallas_exceptions_14`, `pallas_pme_exclusion_pairs` vs XLA (`rel_dE < 1e-6`, `max_abs_dF < 1e-5`, parameter gradients, duplicate atom indices within a single block, and 3-column `urey_bradley_idx`).
     - `pallas_angles_harmonic` vs `_angle_energy_masked` (including near-collinear angles $|c| \to 1$).
     - `pallas_torsions_4body` vs `_dihedral_energy_masked` (`b0_is_ji=True`), `make_dihedral_energy_fn` (`b0_is_ji=False`), and `make_harmonic_improper_energy_fn` (`is_harmonic=True`), including 3D `(N_dih, T, P)` parameter gradients and collinear overlap guard `x = y = 0`.
     - `pallas_cmap_torsions` vs `_cmap_energy_masked` (`first_is_psi=True`) and `make_energy_fn` (`first_is_psi=False`) on 24x24 bicubic spline grids.
     - `pallas_gb_ace_nl` and `pallas_sasa_nl` vs `compute_gb_energy_neighbor_list` + `compute_ace_nonpolar_energy` and `compute_sasa_neighbor_list` across all 3 descreening/overlap regimes, including `jax.grad` wrt `charges`.
     - Mixed-axis `jax.vmap(jax.grad(...), in_axes=(0, None, ...))` and traced `soft_core_lambda` (`lam = jnp.array(0.5)`) tests on `pallas_nonbonded_nl`, `pallas_flash_explicit_nonbonded`, and bonded kernels.
   - Full-system parity (`make_energy_fn` including `return_decomposed=True`, `make_energy_fn_pure`, `single_padded_energy`, `single_padded_energy_nl`, `single_padded_energy_nl_cvjp`, `single_padded_force`, `energy_fn_from_bundle`, `force_fn_from_bundle`, `bonded_energy_fn_from_bundle` with `use_pallas=False` vs `use_pallas=True`) on both periodic explicit-solvent and non-periodic (`box_size=None`) implicit-solvent systems.
   - StableHLO export guardrail test verifying `make_energy_fn_pure` and `single_padded_energy` with `use_pallas=True` lower cleanly under `export_context()` with zero `pallas_call`/`triton_call` ops.
2. **Full-Potential & Full-Propagator Benchmark (`scripts/benchmarks/pallas_vs_baseline.py --kernel full_potential` and `scripts/benchmarks/kups_benchmark.py --tier 4`, resolves `R5-B3`)**:
   - In `scripts/benchmarks/pallas_vs_baseline.py`: add `"full_potential"` to `--kernel` choices, benchmarking both component-by-component (`bonded_all`, `nonbonded_nl`, `pme_recip`, `gb_ace_nl`, `full_explicit_energy_grad`, `full_implicit_energy_grad`) and 20-step `settle_langevin_potential_cached_step` `lax.scan` propagator.
   - In `scripts/benchmarks/kups_benchmark.py`: add `"4"` to `--tier` choices (`choices=("1", "2", "3", "4", "all")`) and implement `run_tier4_full_potential_pallas(n_Targets, n_Timing_iters)` comparing 20-step BAOAB + SETTLE `jax.lax.scan` propagator (`settle_langevin_potential_cached_step` with `value_energy_and_forces`) between `make_energy_fn(..., use_pallas=False)` and `make_energy_fn(..., use_pallas=True, use_pallas_pme=True)` across `N_Targets`.
