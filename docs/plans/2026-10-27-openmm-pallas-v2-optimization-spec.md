# OpenMM-Referenced Pallas v2 Kernel Optimization Specification (O1–O5)

- **Date**: 2026-10-27
- **Status**: Approved (`Verdict: ACCEPT` after 5-Round Adversarial Review)
- **Predecessor Spec**: [2026-10-26-full-potential-pallas-kernels-spec.md](file:///home/marielle/projects/prolix/docs/plans/2026-10-26-full-potential-pallas-kernels-spec.md)
- **Empirical Baseline Artifacts**:
  - [docs/benchmarks/261027_engaging_full_potential_pallas_vs_xla.json](file:///home/marielle/projects/prolix/docs/benchmarks/261027_engaging_full_potential_pallas_vs_xla.json)
  - [docs/benchmarks/261027_engaging_kups_tier4_full_potential.json](file:///home/marielle/projects/prolix/docs/benchmarks/261027_engaging_kups_tier4_full_potential.json)
  - [.praxia/docs/reference/261026_main_vs_b1full_trial_merge.md](file:///home/marielle/projects/prolix/.praxia/docs/reference/261026_main_vs_b1full_trial_merge.md)
- **Target Modules**:
  - [src/prolix/pallas_kernels.py](file:///home/marielle/projects/prolix/src/prolix/pallas_kernels.py) (`pallas_nonbonded_nl`, `_nb_nl_kernel`, `_pallas_nonbonded_nl_bwd`)
  - [src/prolix/physics/pallas_bonded.py](file:///home/marielle/projects/prolix/src/prolix/physics/pallas_bonded.py) (`pallas_bonded_all_fused`, `pallas_torsions_4body`, `pallas_cmap_torsions`, `pallas_angles_harmonic`, `_*_bwd`)
  - [src/prolix/physics/neighbor_list.py](file:///home/marielle/projects/prolix/src/prolix/physics/neighbor_list.py) (`ExclusionSpec`, `encode_nl_exclusions`, `compute_morton_codes_3d`)
  - [src/prolix/physics/pme_pallas.py](file:///home/marielle/projects/prolix/src/prolix/physics/pme_pallas.py) (`spread_charges_pallas`, `gather_pallas`)
  - [src/prolix/physics/pallas_implicit.py](file:///home/marielle/projects/prolix/src/prolix/physics/pallas_implicit.py) (`pallas_gb_ace_nl`, `pallas_sasa_nl`)
  - [src/prolix/physics/system.py](file:///home/marielle/projects/prolix/src/prolix/physics/system.py) (`make_energy_fn`, `make_energy_fn_pure`)
  - [src/prolix/batched_energy.py](file:///home/marielle/projects/prolix/src/prolix/batched_energy.py) (`single_padded_energy`, `single_padded_force`, `single_padded_energy_nl_cvjp`)

---

## 1. Executive Summary & Empirical Baseline

### 1.1 Empirical Baseline from Pallas v1 (`sm_120` RTX PRO 6000 Blackwell)

In Pallas v1 ([2026-10-26-full-potential-pallas-kernels-spec.md](file:///home/marielle/projects/prolix/docs/plans/2026-10-26-full-potential-pallas-kernels-spec.md)), all potential terms—bonds, Urey-Bradley, harmonic angles, periodic dihedrals, periodic/harmonic impropers, CMAP, 1-4 exceptions, PME exclusion erf correction, GB/ACE + SASA implicit solvent, and PME B-spline charge spread/gather—were migrated to Triton-backed Pallas kernels with analytical custom VJPs.

Benchmarking on Engaging `RTX PRO 6000 Blackwell Server Edition` (`sm_120`, 96 GB VRAM) across explicit-PME solvated systems ([261027_engaging_full_potential_pallas_vs_xla.json](file:///home/marielle/projects/prolix/docs/benchmarks/261027_engaging_full_potential_pallas_vs_xla.json) and [261027_engaging_kups_tier4_full_potential.json](file:///home/marielle/projects/prolix/docs/benchmarks/261027_engaging_kups_tier4_full_potential.json)) established:

| Atoms ($N$) | Max Neighbors ($K$) | XLA E+F (`ms`) | Pallas v1 E+F (`ms`) | E+F Speedup | XLA 20-Step Propagator (`ns/day`) | Pallas v1 20-Step Propagator (`ns/day`) | Propagator Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 392 | 0.406 | 0.579 | **0.70x** | 253.34 | 150.47 | **0.59x** |
| 1,024 | 420 | 0.650 | 0.640 | **1.02x** | 139.20 | 127.62 | **0.92x** |
| 2,048 | 418 | 1.158 | 0.727 | **1.59x** | 95.36 | 115.67 | **1.21x** |
| 4,096 | 428 | 1.884 | 0.903 | **2.09x** | 67.32 | 92.54 | **1.37x** |
| 8,192 | 440 | 2.665 | 1.242 | **2.15x** | 34.48 | 66.15 | **1.92x** |
| 16,384 | 436 | 4.925 | 1.919 | **2.57x** | 18.89 | 44.31 | **2.35x** |
| 32,768 | 448 | 9.033 | 2.581 | **3.50x** | 9.77 | 33.03 | **3.38x** |

Power-law scaling fits ($t(N) \propto N^\alpha$):
- **XLA Baseline**: $\alpha_{\text{XLA}} = 0.750$ ($R^2 = 0.994$)
- **Pallas v1**: $\alpha_{\text{Pallas}} = 0.340$ ($R^2 = 0.931$)

### 1.2 Diagnosis of Remaining Performance Bottlenecks vs OpenMM

Comparing Prolix's Pallas v1 kernels against OpenMM's production CUDA/Common kernels reveals five specific bottlenecks that explain both the small-N overhead floor ($\sim 0.35\;\text{ms}$ fixed overhead at $N = 512$) and the remaining throughput gap at $N \ge 8192$:

| ID | Bottleneck in Prolix Pallas v1 | OpenMM Reference Kernel | Root Cause & Hardware Impact | v2 Target Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **O1** | `(32, 32, 32)` 3D exclusion scan in [`_nb_nl_kernel`](file:///home/marielle/projects/prolix/src/prolix/pallas_kernels.py#L847-L858) | `/home/marielle/openmm/platforms/cuda/src/kernels/nonbonded.cu:170-265`, `findInteractingBlocks.cu:173-348` | Materializes `match_3d` `(32, 32, 32)` = 32,768 elements (256 regs/thread at `num_warps=4`) every K-tile + keeps 3 `(32, 32)` arrays (96 regs/thread) live across loop. | **0 in-kernel exclusion loads** via lossless 2-bit NL tagging (**O1.1**) + 2-reg `[excl_min, excl_max]` & 2D chunked fallback (**O1.2**). **1.25x–1.45x** nonbonded speedup. |
| **O2** | (a) Unconditional `_*_param_grads` execution in `_bwd`; (b) 7 separate bonded Pallas launches + 28 buffer zero-fills | `/home/marielle/openmm/platforms/common/src/BondedUtilities.cpp:115-195` (`computeBondedForces`) | `_pallas_nonbonded_nl_bwd` runs a chunked $(N, K, 3)$ XLA scan ([`pallas_kernels.py:1210`](file:///home/marielle/projects/prolix/src/prolix/pallas_kernels.py#L1210)) even when only `grad(E)(pos)` is needed; 7 bonded launches add $\sim 200\;\mu\text{s}$ launch + zero-fill overhead. | `custom_vjp(..., symbolic_zeros=True)` `arg.perturbed` gating skips param grads when $dE/d\theta$ is unused (**O2.1**); single `pallas_bonded_all_fused` megakernel with non-overlapping CTA `pid` ranges (**O2.2**) cuts small-N latency by **35–50%**. |
| **O3** | Normalized-vector `atan2` Jacobian & `arccos` quotient rule in [`pallas_bonded.py:1597-1645, 1907-1988`](file:///home/marielle/projects/prolix/src/prolix/physics/pallas_bonded.py#L1907-L1988) | `/home/marielle/openmm/platforms/common/src/kernels/torsionForce.cc:27-37`, `angleForce.cc:7-18`, `cmapTorsionForce.cc:66-122` | `_torsion_angle_and_atom_grads_vec3` uses 5 cross products and ~85 scalar ops per torsion; CMAP uses 16 ungrouped `u^p * v^q` products instead of Horner 4-tuples. | Bekker cross-product projection (2 cross products, ~42 scalar ops, **2x fewer FLOPs/regs**) + Horner 4-tuple CMAP bicubic evaluation. |
| **O4** | Unconditional `floor(dr / box + 0.5)` inside the inner `(32, 32)` K-tile loop ([`pallas_kernels.py:837-840`](file:///home/marielle/projects/prolix/src/prolix/pallas_kernels.py#L837-L840)) | `/home/marielle/openmm/platforms/cuda/src/kernels/nonbonded.cu:299-360`, `findInteractingBlocks.cu:41-102` (`singlePeriodicCopy`) | Evaluates `3 * 1024 = 3072` `floor` + multiply-add ops per K-tile even when the 32-atom i-block is interior to the box or satisfies `singlePeriodicCopy`. | Primary-cell pre-wrap + interior-block zero-wrap bypass (**0 `floor` ops** on interior axes, **O4.1**) + 1D `(32,)` `singlePeriodicCopy` (**32x fewer wrap ops**, **O4.2**). |
| **O5** | Arbitrary PDB topological atom ordering in NL tiles and PME spread/gather ([`pme_pallas.py:72-170`](file:///home/marielle/projects/prolix/src/prolix/physics/pme_pallas.py#L72-L170)) | `/home/marielle/openmm/platforms/common/src/kernels/pme.cc:1-21`, `/home/marielle/openmm/platforms/cuda/src/kernels/findInteractingBlocks.cu:104-171` | Consecutive atoms $i \in [32b, 32b+31]$ can be far apart in 3D space (especially solvent ions/waters after diffusion), inflating block extents $\mathbf{e}_{\text{blk}}$ and causing scattered L2 & `atomic_add` contention. | 3D Morton (Z-order) / PME grid-index spatial sorting at NL rebuild time clusters 32-atom blocks spatially, unlocking O4 interior/`singlePeriodicCopy` fast-paths and **1.2x–1.5x** PME/NL cache locality speedup. |

---

## 2. Optimization O1: Two-Tier Elimination of the `(32, 32, 32)` Exclusion Broadcast Scan

### 2.1 Problem Analysis & OpenMM Reference

In [`_nb_nl_kernel`](file:///home/marielle/projects/prolix/src/prolix/pallas_kernels.py#L780-L858), each 32-atom program block currently loads three `(32, 32)` exclusion arrays before the neighbor-tile loop:
```python
excl_idx_i = excl_indices_ref[i_slice, :]  # (32, 32) int32
excl_vdw_i = excl_scales_vdw_ref[i_slice, :]  # (32, 32) f32
excl_elec_i = excl_scales_elec_ref[i_slice, :]  # (32, 32) f32
```
Inside `_k_tile_body`, lines 847–858 broadcast `j_idx[:, None, :]` `(32, 1, 32)` against `excl_idx_i[:, :, None]` `(32, 32, 1)`:
```python
match_3d = j_idx[:, None, :] == excl_idx_i[:, :, None]  # (32, 32, 32) bool!
any_match = jnp.any(match_3d, axis=1)  # (32, 32)
match_slot = jnp.argmax(match_3d.astype(jnp.int32), axis=1)  # (32, 32)
```
At `BLOCK_N = 32, BLOCK_K = 32, M = 32` (`num_warps=4`, 128 threads), `match_3d` alone has $32^3 = 32,768$ elements (256 elements per thread), while `excl_idx_i`, `excl_vdw_i`, and `excl_elec_i` consume another $3 \times 1024 / 128 = 24$ registers per thread live across every loop iteration. This forces heavy register spilling to local memory on `sm_89`, `sm_90`, and `sm_120`.

**How OpenMM solves this**:
1. In `/home/marielle/openmm/platforms/cuda/src/kernels/findInteractingBlocks.cu:173-348`, OpenMM filters excluded atom pairs **during neighbor list construction**, so the main nonbonded interaction loop (`nonbonded.cu:266-390`) never performs a dynamic exclusion table search for neighbor-list tiles; only diagonal self-blocks (`nonbonded.cu:170-265`) use a compact 32-bit `exclusions` bitmask (`1u << j`).
2. Furthermore, in Prolix's [`ExclusionSpec.from_protein`](file:///home/marielle/projects/prolix/src/prolix/physics/neighbor_list.py#L109-L133) and [`openmm_xml_importer.py:792-804`](file:///home/marielle/projects/prolix/src/prolix/openmm_xml_importer.py#L792-L804):
   - All 1-2 and 1-3 pairs have `scale_vdw = 0.0` and `scale_elec = 0.0`.
   - When `use_explicit_14=True` (`protein.pairs_14` and `protein.exception_14_params` present, or OpenMM XML bundles), all 1-4 pairs are separated into `ExclusionSpec.exception_pairs` (evaluated by [`pallas_exceptions_14`](file:///home/marielle/projects/prolix/src/prolix/physics/pallas_bonded.py#L728-L825)) and placed into `idx_12_13` with `scale_vdw = 0.0, scale_elec = 0.0`.
   - When `use_explicit_14=False` ([`neighbor_list.py:132, 356-363`](file:///home/marielle/projects/prolix/src/prolix/physics/neighbor_list.py#L356-L363)), 1-4 pairs reside in `excl_indices` with uniform scalar 1-4 weights `(scale_14_vdw, scale_14_elec)` (typically `(0.5, 0.83333333)`).
   - Thus, for standard MD potentials, every excluded neighbor entry takes one of **at most two states**: full exclusion `(0.0, 0.0)` or standard 1-4 scaled exclusion `(scale_14_vdw, scale_14_elec)`.

### 2.2 Tier 1 (O1.1): Lossless 2-Bit Pre-Encoded Neighbor-List Exclusions at NL Build/Update Time

We move the exclusion lookup **out of the every-step nonbonded force kernel** and into the neighbor-list construction/update step (which runs only once every 10–20 MD steps, or once when preparing a static `neighbor_idx` table), reusing the existing 5-step vector binary search in [`get_neighbor_exclusion_scales`](file:///home/marielle/projects/prolix/src/prolix/physics/neighbor_list.py#L465-L520):

1. **Lossless 2-Bit Tag Encoding Contract (`_NB_EXCL_FULL_BIT`, `_NB_EXCL_14_BIT`)**:
   - Define in [`src/prolix/physics/neighbor_list.py`](file:///home/marielle/projects/prolix/src/prolix/physics/neighbor_list.py) (and import in `pallas_kernels.py`, `pallas_implicit.py`):
     - `_NB_EXCL_FULL_BIT: Final[int] = 1 << 30` (`0x40000000`, full 1-2/1-3 exclusion: $w_{\text{vdw}} = 0.0, w_{\text{elec}} = 0.0$)
     - `_NB_EXCL_14_BIT: Final[int] = 1 << 29` (`0x20000000`, scaled 1-4 exclusion: $w_{\text{vdw}} = s_{14,\text{vdw}}, w_{\text{elec}} = s_{14,\text{elec}}$)
     - `_NB_INDEX_MASK: Final[int] = (1 << 29) - 1` (`0x1FFFFFFF`), supporting up to $N < 2^{29} \approx 5.36 \times 10^8$ atoms.
   - Add a pure JAX helper `encode_nl_exclusions(neighbor_idx: Array, excl_indices: Array, excl_scales_vdw: Array, excl_scales_elec: Array, n_atoms: int, *, scale_14_vdw: float = 0.5, scale_14_elec: float = 0.83333333) -> Array`:
     - Using the 5-step branchless binary search in [`get_neighbor_exclusion_scales`](file:///home/marielle/projects/prolix/src/prolix/physics/neighbor_list.py#L465-L520) (`O(N * K * log2 M)` executed only on NL rebuild steps), look up `(w_vdw, w_elec)` for each entry `j = neighbor_idx[i, k] & _NB_INDEX_MASK`.
     - Tag full exclusions (`(w_vdw == 0.0) & (w_elec == 0.0)`) as `j | _NB_EXCL_FULL_BIT` and 1-4 scaled exclusions as `j | _NB_EXCL_14_BIT`, **always preserving the underlying atom index `j` in the lower 29 bits (`j & _NB_INDEX_MASK`)**.
     - *Why lossless tagging instead of overwriting `j` with `n_atoms`*:
       1. In implicit solvent (`batched_energy.py:1432`, `system.py:210`), the same `neighbor_idx` is passed to [`pallas_gb_ace_nl`](file:///home/marielle/projects/prolix/src/prolix/physics/pallas_implicit.py) and `pallas_sasa_nl`, where covalently bonded 1-2/1-3/1-4 neighbors **must not** be excluded from Born radii descreening or SASA burial! Preserving `j` in `raw_j & _NB_INDEX_MASK` lets `pallas_gb_ace_nl` and `pallas_sasa_nl` mask off the upper 2 bits (`j_idx = raw_j & _NB_INDEX_MASK`) without losing any bonded neighbors.
       2. In `_nonbonded_nl_param_grads` ([`pallas_kernels.py:1131-1138`](file:///home/marielle/projects/prolix/src/prolix/pallas_kernels.py#L1131-L1138)), masking `nbr_i = raw_nbr_i & _NB_INDEX_MASK` ensures `nbr_i < N` remains true on tagged PME exclusion pairs when `compute_param_grads=True`.

2. **Zero-Lookup In-Kernel Decoding (`excl_mode = "pre_encoded"` in `_nb_nl_kernel`)**:
   - When `excl_mode == "pre_encoded"`:
     - `_nb_nl_kernel` **does not load `excl_indices_ref`, `excl_scales_vdw_ref`, or `excl_scales_elec_ref` at all** (saving 3,072 scalar loads per 32-atom block and 96 registers per thread).
     - Inside `_k_tile_body`:
       ```python
       raw_j_1d = _pl_load(nbr_flat_ref, (flat_nbr_offs,))
       raw_j = raw_j_1d.reshape(_NB_BLOCK_N, _NB_BLOCK_K)
       is_full_excl = (raw_j & _NB_EXCL_FULL_BIT) != 0
       is_14_excl = (raw_j & _NB_EXCL_14_BIT) != 0
       j_idx = raw_j & _NB_INDEX_MASK
       safe_j_1d = jnp.clip(j_idx.reshape(_NB_BLOCK_N * _NB_BLOCK_K), 0, N_pad - 1)
       ...
       # When include_pme_excl_erf=False, full exclusions can be masked out of valid_pair directly:
       valid_pair = (
           mask_i[:, None]
           & mask_j
           & (j_idx >= 0)
           & (j_idx < N)
           & (j_idx != offs_i[:, None])
           & (True if include_pme_excl_erf else (~is_full_excl))
       )
       w_vdw = jnp.where(is_full_excl, zero, jnp.where(is_14_excl, s14_vdw, one))
       w_elec = jnp.where(is_full_excl, zero, jnp.where(is_14_excl, s14_elec, one))
       ```
     - Total in-kernel exclusion cost per K-tile: **3 bitwise integer instructions, 2 selects, and 0 memory loads**.

### 2.3 Tier 2 (O1.2): In-Kernel `[excl_min_i, excl_max_i]` Bounding + Chunked 2D Fallback

To maintain 100% backward compatibility with direct callers of `pallas_nonbonded_nl` that pass un-encoded `neighbor_idx` or arbitrary per-pair fractional `excl_scales_vdw` / `excl_scales_elec`, Tier 2 replaces the `(32, 32, 32)` broadcast scan inside `_nb_nl_kernel` with a register-minimal 2D algorithm:

1. **Host/XLA Pre-Pass Canonicalization**:
   - In Layer 1 (`_prepare_nonbonded_nl_inputs`), sort each row of `excl_indices` in **ascending order**, replacing `-1` padding sentinels with `INT32_MAX` (`0x7FFFFFFF`) before sorting so valid exclusion indices occupy contiguous leading slots `[0, ..., m_i - 1]` and `INT32_MAX` occupies trailing slots. Permute `excl_scales_vdw` and `excl_scales_elec` with the same `argsort` indices.
   - Precompute 1D per-atom bounding intervals:
     - `excl_min = excl_indices_sorted[:, 0]` (`INT32_MAX` if atom $i$ has 0 exclusions)
     - `excl_max = jnp.max(jnp.where(excl_indices_sorted < INT32_MAX, excl_indices_sorted, -1), axis=-1)` (`-1` if atom $i$ has 0 exclusions)

2. **In-Kernel Bounding-Interval Fast-Path + 2D Chunked Lookup**:
   - Before `fori_loop`, `_nb_nl_kernel` loads only `excl_min_i` `(32,)` and `excl_max_i` `(32,)` (**2 registers per thread** instead of 96).
   - Because molecular exclusions are topologically local ($|j - i| \le 25$ along the polypeptide chain), >90% of K-tiles satisfy `jnp.all(~in_excl_range)`.
   - Inside `_k_tile_body`:
     ```python
     in_excl_range = (
         active
         & (j_idx >= excl_min_i[:, None])
         & (j_idx <= excl_max_i[:, None])
     )  # (32, 32) bool
     ```
   - Guarded by `jax.lax.cond(jnp.any(in_excl_range), _lookup_excl_tile, _no_excl_tile, ...)`:
     - `_no_excl_tile` returns `w_vdw = jnp.ones((32, 32), dtype=dtype), w_elec = jnp.ones((32, 32), dtype=dtype)` with **zero memory loads and zero comparisons**.
     - `_lookup_excl_tile` loads `excl_idx_i` inside the conditional branch (so it is never live across the `fori_loop`) and scans in 2D chunks—**never allocating any `(32, 32, 32)` tensor**.

---

## 3. Optimization O2: Active-AD Parameter-Gradient Gating & Single-Launch Bonded Megakernel

### 3.1 Problem Analysis & OpenMM Reference

At $N = 512$, Pallas v1 takes `0.579 ms` vs XLA's `0.406 ms` ([261027_engaging_full_potential_pallas_vs_xla.json](file:///home/marielle/projects/prolix/docs/benchmarks/261027_engaging_full_potential_pallas_vs_xla.json)). Profiling the JAX execution trace of `jax.value_and_grad(energy_fn)(pos)` reveals two dominant fixed-overhead sources:

1. **Wasted Parameter Cotangent Recomputation in `_bwd`**:
   - In [`_pallas_nonbonded_nl_bwd`](file:///home/marielle/projects/prolix/src/prolix/pallas_kernels.py#L1321-L1372), every backward pass unconditionally executes:
     ```python
     gq, gs, ge, gev, gee, glam = _nonbonded_nl_param_grads(...)
     ```
     which runs a 64-chunk `jax.lax.scan` over $(N, K, 3)$ dense XLA arrays ([`pallas_kernels.py:1210`](file:///home/marielle/projects/prolix/src/prolix/pallas_kernels.py#L1210)) to compute $\partial E / \partial q_i$, $\partial E / \partial \sigma_i$, $\partial E / \partial \epsilon_i$, $\partial E / \partial w_{\text{excl}}$, and $\partial E / \partial \lambda$.
   - Similarly, all 6 bonded `_bwd` rules ([`pallas_bonded.py:509-522, 922-941, 1207-1219, 1780-1816, 2304-2343, 2972-2984`](file:///home/marielle/projects/prolix/src/prolix/physics/pallas_bonded.py#L1780-L1816)) unconditionally invoke `_bonds_2body_param_grads`, `_exceptions_14_param_grads`, `_pme_excl_param_grads`, `_angles_harmonic_param_grads`, `_torsions_4body_param_grads`, and `_cmap_torsions_param_grads`.
   - During standard MD simulation (`jax.grad(energy_fn)(positions)` or `single_padded_force`), parameters are **non-differentiated inputs**. Although XLA dead-code-eliminates some unused outputs, the `jax.lax.scan` in `_nonbonded_nl_param_grads` is a control-flow primitive (`while_loop` in HLO) whose body and intermediates are still traced and materialized unless gated off!

2. **7 Separate Bonded Kernel Launches + 28 Output Buffer Allocations/Zero-Fills**:
   - In OpenMM (`/home/marielle/openmm/platforms/common/src/BondedUtilities.cpp:115-195`), `BondedUtilities::computeBondedForces` compiles all bonded force snippets (`bondForce.cc`, `angleForce.cc`, `torsionForce.cc`, `periodicTorsionForce.cc`, `rbTorsionForce.cc`, `cmapTorsionForce.cc`) into **one GPU kernel launch** that accumulates into a single force buffer.
   - In Prolix v1, [`make_energy_fn`](file:///home/marielle/projects/prolix/src/prolix/physics/system.py#L535-L625) and [`single_padded_energy`](file:///home/marielle/projects/prolix/src/prolix/batched_energy.py#L703-L734) launch up to **7 separate `pl.pallas_call` kernels** per step (bonds, Urey-Bradley, angles, dihedrals, impropers, CMAP, 1-4 exceptions). Each launch allocates and zeros 3 output force buffers (`fx_pad, fy_pad, fz_pad`) plus `e_blk`, totaling 28 device buffer allocations/zero-fills and 7 kernel dispatches.

### 3.2 Part A (O2.1): Active-AD Parameter-Gradient Gating via `custom_vjp(..., symbolic_zeros=True)`

1. **Exact Perturbation Detection via JAX's Public `symbolic_zeros=True` Protocol**:
   - *Why naive Tracer inspection fails in JAX*: In modern JAX (`jax >= 0.4.30`), `jax.grad` / `jax.vjp` traces the primal function with `LinearizeTracer` (not `JVPTracer`), and when differentiating a JIT-compiled function `jax.grad(jax.jit(f), argnums=1)(pos, params)`, `@jax.jit` traces `f` first with `DynamicJaxprTracer` for **all** arguments (`pos` and `params` alike) before `ad.linearize` runs on the lowered Jaxpr. Thus checking `isinstance(x, ad.JVPTracer)` inside the primal function returns `False` even when `params` is actively differentiated!
   - Instead, register all Pallas `custom_vjp` rules with **`defvjp(fwd, bwd, symbolic_zeros=True)`** (JAX's official public API for selective cotangent computation):
     - Inside `fwd`, every differentiable argument `arg` is wrapped in a `jax.custom_derivatives.CustomVJPPrimal(value, perturbed)` where **`arg.perturbed: bool` is a static Python boolean set by JAX's AD engine to `True` if and only if that argument is being differentiated** (working identically under `jax.grad(f)`, `jax.jit(jax.grad(f))`, and `jax.grad(jax.jit(f))`).
     - In `_pallas_nonbonded_nl_fwd(positions, charges, sigmas, epsilons, atom_mask, neighbor_idx, excl_indices, excl_scales_vdw, excl_scales_elec, box, lam_arr, ...)`:
       ```python
       any_param_perturbed = (
           charges.perturbed
           or sigmas.perturbed
           or epsilons.perturbed
           or excl_scales_vdw.perturbed
           or excl_scales_elec.perturbed
           or lam_arr.perturbed
       )
       param_res = (
           (
               positions.value,
               charges.value,
               sigmas.value,
               epsilons.value,
               atom_mask.value,
               neighbor_idx.value,
               excl_indices.value,
               excl_scales_vdw.value,
               excl_scales_elec.value,
               box.value,
               lam_arr.value,
           )
           if any_param_perturbed
           else None
       )
       return energy, (forces, positions.value.dtype, param_res)
       ```
     - In `_pallas_nonbonded_nl_bwd(..., res, g)`:
       ```python
       forces, pos_dtype, param_res = res
       grad_pos = (-g[..., None, None] * forces if g.ndim > 0 else -g * forces).astype(pos_dtype)
       if param_res is None:
           return (grad_pos, None, None, None, None, None, None, None, None, None, None)
       # Otherwise unpack param_res and run _nonbonded_nl_param_grads(...)
       ```
   - **Dual Benefit**:
     1. When `any_param_perturbed` is `False` (all standard MD steps `jax.grad(energy_fn)(pos)`), `param_res` is `None` (an empty PyTree node), so **none of `(positions, charges, sigmas, epsilons, neighbor_idx, excl_*)` are saved in the AD residuals**, and `_nonbonded_nl_param_grads` (the 64-chunk `jax.lax.scan`) and all 6 bonded `_*_param_grads` functions are **statically omitted at trace time**!
     2. When `any_param_perturbed` is `True` (e.g., `jax.grad(loss_fn)(diff_params)` in Sprint 11 `make_energy_fn_pure`), `param_res` is populated and exact analytical parameter gradients are computed.
   - In addition, expose an explicit `compute_param_grads: bool | None = None` keyword argument on `pallas_nonbonded_nl` and all bonded kernels as a manual override if a caller wants to force parameter gradients off even when parameters are perturbed tracers.

2. **Direct `(E, forces)` Forward Return Path for `single_padded_force`**:
   - Add `return_forces: bool = False` to `pallas_nonbonded_nl` and `pallas_bonded_all_fused`. Since every Pallas forward kernel **already computes both scalar energy `E` and atomic forces `(fx, fy, fz)` in a single pass**, callers like [`single_padded_force`](file:///home/marielle/projects/prolix/src/prolix/batched_energy.py#L1400-L1520) can optionally obtain `(E, forces)` directly from the forward kernel execution without constructing an AD tape when `custom_vjp` composition is not needed.

### 3.3 Part B (O2.2): Single-Launch Bonded Megakernel (`pallas_bonded_all_fused`)

Following OpenMM's `BondedUtilities::computeBondedForces` (`/home/marielle/openmm/platforms/common/src/BondedUtilities.cpp:115-195`):

1. **Term Consolidation Before Launch**:
   - **2-Body Harmonic Group (`bonds_ub`)**: Concatenate standard harmonic bonds `(B, 2)` and Urey-Bradley 1-3 harmonic pairs `(UB, 2)` into a single array `bond_ub_indices` `(B_tot, 2)` and `bond_ub_params` `(B_tot, 2)` with `bond_ub_mask` `(B_tot,)` at system/batch setup time. (Saves 1 kernel launch with zero extra math).
   - **4-Body Torsion Group (`torsions_4body`)**: Concatenate proper dihedrals `(D, 4)` and periodic improper torsions `(I_per, 4)` into a single array `torsion_indices` `(T_tot, 4)` and `torsion_params` `(T_tot, 3)` with `torsion_mask` `(T_tot,)` (plus optional per-term convention flag `b0_is_ji` when mixed conventions exist).
   - **Remaining Distinct Groups**:
     - Harmonic angles: `(A, 3)`, `angle_params` `(A, 2)`
     - Harmonic impropers (CHARMM/AMBER quadratic $k(\phi - \phi_0)^2$ when present): `(I_harm, 4)`, `harm_improper_params` `(I_harm, 2)`
     - 1-4 nonbonded exceptions: `(E_14, 2)` + per-pair `(q_ij, sig_ij, eps_ij)`
     - CMAP coupled 5-atom torsions: `(C, 5)` + `cmap_map_indices` `(C,)` + `cmap_coeffs` `(M_maps, G, G, 16)`

2. **Single `pl.pallas_call` Grid with Non-Overlapping CTA `pid` Ranges**:
   - Let each active bonded group $g \in \{\text{bonds\_ub}, \text{angles}, \text{torsions}, \text{harm\_imp}, \text{exceptions\_14}, \text{cmap}\}$ have block count $N_{\text{blk},g} = \lceil N_g / B_T \rceil$ where $B_T = 64$, and define cumulative block offsets:
     $$O_0 = 0, \qquad O_{g+1} = O_g + N_{\text{blk},g}, \qquad N_{\text{blk,total}} = \sum_g N_{\text{blk},g}$$
   - Allocate **one** shared output triplet `(fx_pad, fy_pad, fz_pad)` of shape `(N_pad,)` (requiring a single zero-initialization for all bonded terms combined) and **one** block energy accumulator `e_blk` of shape `(N_blk_total,)`.
   - Inside `_bonded_all_fused_kernel(..., fx_ref, fy_ref, fz_ref, e_blk_ref)`:
     - `pid = pl.program_id(0)`
     - Because `pid` is **100% uniform across all threads in the CTA**, dispatching `pid` to its unique owning group $g$ (where $O_g \le \text{pid} < O_{g+1}$ and local block index is $\text{local\_pid} = \text{pid} - O_g$) via nested `jax.lax.cond` compiles to a scalar uniform branch (`bra.uni` in PTX) with **zero warp divergence** and **no cross-group register live-range overlap**:
       1. `pid in [O_0, O_1)`: evaluate 2-body harmonic bonds + Urey-Bradley at `local_pid = pid - O_0`, issue 2 masked `pl.atomic_add`s per axis, return block energy `e_sum`.
       2. `pid in [O_1, O_2)`: evaluate 3-body harmonic angles at `local_pid = pid - O_1` (using O3.2 cross-product projection), issue 3 `pl.atomic_add`s per axis, return `e_sum`.
       3. `pid in [O_2, O_3)`: evaluate 4-body periodic dihedrals + periodic impropers at `local_pid = pid - O_2` (using O3.1 Bekker cross-product projection), issue 4 `pl.atomic_add`s per axis, return `e_sum`.
       4. `pid in [O_3, O_4)`: evaluate 4-body harmonic impropers (when $N_{\text{blk,harm\_imp}} > 0$) at `local_pid = pid - O_3`, return `e_sum`.
       5. `pid in [O_4, O_5)`: evaluate 1-4 LJ + Coulomb exceptions at `local_pid = pid - O_4`, issue 2 `pl.atomic_add`s per axis, return `e_sum`.
       6. `pid in [O_5, O_6)`: evaluate 5-body CMAP bicubic spline at `local_pid = pid - O_5` (using O3.1 + O3.3 Horner evaluation), issue 5 `pl.atomic_add`s per axis, return `e_sum`.
     - Write `e_blk_ref[pid] = e_sum` deterministically to CTA `pid`'s dedicated slot at the end of the kernel.
   - **Impact**: Reduces 7 kernel launches to **1 kernel launch**, reduces 28 buffer allocations/zero-fills to **4** (`fx_pad, fy_pad, fz_pad, e_blk`), and ensures each CTA executes only one bonded group (at most 5 `pl.atomic_add` calls per axis per CTA).

---

## 4. Optimization O3: Bekker/OpenMM Cross-Product Torsion & Angle Force Projection + Horner CMAP

### 4.1 Problem Analysis & OpenMM Reference

In [`src/prolix/physics/pallas_bonded.py:1907-1988`](file:///home/marielle/projects/prolix/src/prolix/physics/pallas_bonded.py#L1907-L1988), `_torsion_angle_and_atom_grads_vec3` differentiates $\phi = \text{atan2}(y, x)$ through projected vectors $\mathbf{v} = \mathbf{b}_0 - (\mathbf{b}_0\cdot\hat{\mathbf{b}}_1)\hat{\mathbf{b}}_1$ and $\mathbf{w} = \mathbf{b}_2 - (\mathbf{b}_2\cdot\hat{\mathbf{b}}_1)\hat{\mathbf{b}}_1$. This requires 5 cross products (`_cross3(b1_hat, v)`, `_cross3(w, b1_hat)`, `_cross3(b1_hat, v)`, `_cross3(v, w)`) and 18 vector scale/add/sub operations (~85 scalar floating-point ops and ~28 live vector temporaries per torsion).

In contrast, OpenMM (`/home/marielle/openmm/platforms/common/src/kernels/torsionForce.cc:27-37` and `cmapTorsionForce.cc:66-122`) uses Bekker's analytical plane-normal identity, which computes the exact same Cartesian forces directly from the two plane normals $\mathbf{n}_1 = \mathbf{b}_0 \times \mathbf{b}_1$ and $\mathbf{n}_2 = \mathbf{b}_1 \times \mathbf{b}_2$ in **2 cross products, 1 `sqrt`, and ~42 scalar operations**.

### 4.2 Mathematical Formulation for O3.1 (Bekker Cross-Product Torsion Forces)

In [`pallas_bonded.py:1923-1955`](file:///home/marielle/projects/prolix/src/prolix/physics/pallas_bonded.py#L1923-L1955), Prolix supports two conventions controlled by `b0_is_ji: bool`:
- `b0_is_ji = False` (used by [`bonded.py:98`](file:///home/marielle/projects/prolix/src/prolix/physics/bonded.py#L98), [`system.py:134, 140, 565, 576`](file:///home/marielle/projects/prolix/src/prolix/physics/system.py#L134), and [`pallas_cmap_torsions` at `pallas_bonded.py:2570, 2573`](file:///home/marielle/projects/prolix/src/prolix/physics/pallas_bonded.py#L2570)): uses $\mathbf{b}_0^{\text{ij}} = \mathbf{r}_i - \mathbf{r}_j$ and $\phi = \phi_{\text{raw}}$.
- `b0_is_ji = True` (used by [`batched_energy.py:722, 727, 1004, 1008`](file:///home/marielle/projects/prolix/src/prolix/batched_energy.py#L722), [`flash_explicit.py:799, 802`](file:///home/marielle/projects/prolix/src/prolix/physics/flash_explicit.py#L799), and [`bundle_md.py:319, 329`](file:///home/marielle/projects/prolix/src/prolix/api/bundle_md.py#L319)): uses $\mathbf{b}_0^{\text{ji}} = \mathbf{r}_j - \mathbf{r}_i = -\mathbf{b}_0^{\text{ij}}$ and shifts $\phi = \phi_{\text{raw}}^{\text{ji}} - \pi$.

Let us formulate the unified Bekker equations in terms of $\mathbf{b}_0^{\text{ij}} = \mathbf{r}_i - \mathbf{r}_j$, central bond $\mathbf{b}_1 = \mathbf{r}_k - \mathbf{r}_j$, and terminal bond $\mathbf{b}_2 = \mathbf{r}_l - \mathbf{r}_k$:

Define the two plane normals:
$$\mathbf{n}_1 = \mathbf{b}_0^{\text{ij}} \times \mathbf{b}_1 = (\mathbf{r}_i - \mathbf{r}_j) \times (\mathbf{r}_k - \mathbf{r}_j), \qquad \mathbf{n}_2 = \mathbf{b}_1 \times \mathbf{b}_2 = (\mathbf{r}_k - \mathbf{r}_j) \times (\mathbf{r}_l - \mathbf{r}_k)$$
and let:
$$r_{b_1}^2 = \|\mathbf{b}_1\|^2, \qquad r_{b_1} = \sqrt{r_{b_1}^2 + \epsilon_{b_1}^2}, \qquad \|\mathbf{n}_1\|^2_{\text{safe}} = \max(\|\mathbf{n}_1\|^2, 10^{-24}), \qquad \|\mathbf{n}_2\|^2_{\text{safe}} = \max(\|\mathbf{n}_2\|^2, 10^{-24})$$
where $\epsilon_{b_1}^2 = 10^{-12}$ when `b0_is_ji=True` and $0$ (with `denom_b1 = r_b1 + 1e-8`) when `b0_is_ji=False`.

1. **Dihedral Angle $\phi$ (Verified to $2.38 \times 10^{-7}\;\text{rad}$ Parity Against Prolix v1)**:
   By Lagrange's identity and scalar triple product properties:
   $$\mathbf{n}_1 \cdot \mathbf{n}_2 = (\mathbf{b}_0^{\text{ij}} \cdot \mathbf{b}_1)(\mathbf{b}_1 \cdot \mathbf{b}_2) - (\mathbf{b}_0^{\text{ij}} \cdot \mathbf{b}_2) r_{b_1}^2 = -r_{b_1}^2 (\mathbf{v} \cdot \mathbf{w}) = -r_{b_1}^2 x_{\text{v1}}$$
   $$\mathbf{b}_0^{\text{ij}} \cdot \mathbf{n}_2 = \mathbf{b}_0^{\text{ij}} \cdot (\mathbf{b}_1 \times \mathbf{b}_2) = -(\mathbf{b}_1 \times \mathbf{b}_0^{\text{ij}}) \cdot \mathbf{b}_2 = -r_{b_1} \left((\hat{\mathbf{b}}_1 \times \mathbf{v}) \cdot \mathbf{w}\right) = -r_{b_1} y_{\text{v1}}$$
   Therefore, defining:
   $$y_{\text{raw}} = -r_{b_1} (\mathbf{b}_0^{\text{ij}} \cdot \mathbf{n}_2), \qquad x_{\text{raw}} = -(\mathbf{n}_1 \cdot \mathbf{n}_2)$$
   gives:
   $$\phi = \begin{cases} \text{atan2}(y_{\text{raw}},\, x_{\text{raw}}) & \text{if } \texttt{b0\_is\_ji = False} \\ \text{atan2}(-y_{\text{raw}},\, -x_{\text{raw}}) - \pi & \text{if } \texttt{b0\_is\_ji = True} \end{cases}$$
   Because scaling both arguments of `atan2` by $r_{b_1}^2 > 0$ leaves `atan2` invariant to machine precision, this matches `_torsion_angle_and_atom_grads_vec3` within $2.38 \times 10^{-7}\;\text{rad}$ across both `b0_is_ji=False` and `b0_is_ji=True`!

2. **Analytical Bekker Force Projection (`torsionForce.cc:27-37`, Verified to $5.96 \times 10^{-8}$ Parity)**:
   Because shifting $\phi$ by a constant ($-\pi$ when `b0_is_ji=True`) does not alter derivatives $\partial \phi / \partial \mathbf{r}$, **both** `b0_is_ji=False` and `b0_is_ji=True` share the exact same Cartesian angle gradients:
   $$\frac{\partial \phi}{\partial \mathbf{r}_i} = \frac{r_{b_1}}{\|\mathbf{n}_1\|^2_{\text{safe}}} \mathbf{n}_1, \qquad \frac{\partial \phi}{\partial \mathbf{r}_l} = \frac{r_{b_1}}{\|\mathbf{n}_2\|^2_{\text{safe}}} \mathbf{n}_2$$
   $$\mathbf{s}_{\phi} = \frac{\mathbf{b}_0^{\text{ij}} \cdot \mathbf{b}_1}{r_{b_1}^2} \frac{\partial \phi}{\partial \mathbf{r}_i} + \frac{\mathbf{b}_2 \cdot \mathbf{b}_1}{r_{b_1}^2} \frac{\partial \phi}{\partial \mathbf{r}_l}, \qquad \frac{\partial \phi}{\partial \mathbf{r}_j} = \mathbf{s}_{\phi} - \frac{\partial \phi}{\partial \mathbf{r}_i}, \qquad \frac{\partial \phi}{\partial \mathbf{r}_k} = -\mathbf{s}_{\phi} - \frac{\partial \phi}{\partial \mathbf{r}_l}$$
   Given scalar torque $\frac{\partial E}{\partial \phi}$, multiplying by $-\frac{\partial E}{\partial \phi}$ yields the atomic forces:
   $$\mathbf{f}_i = -\frac{\partial E}{\partial \phi} \frac{r_{b_1}}{\|\mathbf{n}_1\|^2_{\text{safe}}} \mathbf{n}_1, \qquad \mathbf{f}_l = -\frac{\partial E}{\partial \phi} \frac{r_{b_1}}{\|\mathbf{n}_2\|^2_{\text{safe}}} \mathbf{n}_2$$
   $$\mathbf{s}_f = \frac{\mathbf{b}_0^{\text{ij}} \cdot \mathbf{b}_1}{r_{b_1}^2} \mathbf{f}_i + \frac{\mathbf{b}_2 \cdot \mathbf{b}_1}{r_{b_1}^2} \mathbf{f}_l, \qquad \mathbf{f}_j = \mathbf{s}_f - \mathbf{f}_i, \qquad \mathbf{f}_k = -\mathbf{s}_f - \mathbf{f}_l$$
   Notice that $\mathbf{f}_i + \mathbf{f}_j + \mathbf{f}_k + \mathbf{f}_l = \mathbf{f}_i + (\mathbf{s}_f - \mathbf{f}_i) + (-\mathbf{s}_f - \mathbf{f}_l) + \mathbf{f}_l = \mathbf{0}$ identically in IEEE-754 floating-point arithmetic, and both `b0_is_ji=False` and `b0_is_ji=True` use the **identical** force projection without any sign branching!

### 4.3 Mathematical Formulation for O3.2 (Cross-Product Angle Forces & Parity Mode)

In `/home/marielle/openmm/platforms/common/src/kernels/angleForce.cc:7-18`, for bond vectors $\mathbf{v}_{ji} = \mathbf{r}_i - \mathbf{r}_j$ and $\mathbf{v}_{jk} = \mathbf{r}_k - \mathbf{r}_j$:
$$\mathbf{c}_p = \mathbf{v}_{ji} \times \mathbf{v}_{jk}, \qquad r_{cp} = \sqrt{\max(\|\mathbf{c}_p\|^2, 10^{-24})}$$
$$\mathbf{f}_i = -\frac{\partial E}{\partial \theta} \frac{\mathbf{v}_{ji} \times \mathbf{c}_p}{r_{ji}^2\, r_{cp}}, \qquad \mathbf{f}_k = -\frac{\partial E}{\partial \theta} \frac{\mathbf{c}_p \times \mathbf{v}_{jk}}{r_{jk}^2\, r_{cp}}, \qquad \mathbf{f}_j = -(\mathbf{f}_i + \mathbf{f}_k)$$
Using Lagrange's vector triple product identity $\mathbf{v}_{ji} \times (\mathbf{v}_{ji} \times \mathbf{v}_{jk}) = (\mathbf{v}_{ji} \cdot \mathbf{v}_{jk})\mathbf{v}_{ji} - r_{ji}^2 \mathbf{v}_{jk}$, this is algebraically identical to the quotient-rule derivative of $\arccos$, while computing $\theta = \text{atan2}(r_{cp}, \mathbf{v}_{ji} \cdot \mathbf{v}_{jk})$ avoids the $\arccos(\pm 1)$ precision loss near $\theta \to \pi$ (e.g., linear backbone/alkyne angles).

**Parity Guardrail vs Legacy Prolix Reference (`bonded.py:77`)**:
- Legacy [`bonded.py:77`](file:///home/marielle/projects/prolix/src/prolix/physics/bonded.py#L77) adds `1e-8` to the cosine denominator: `cos_theta = dot / (r_ji * r_jk + 1e-8)`. Because `1e-8` perturbs $\cos\theta$ by $\sim 5 \times 10^{-9}$ at $r_{ji} \approx r_{jk} \approx 1.4\;\text{\AA}$, `arccos(cos_theta)` shifts by $\Delta\theta \approx \frac{5 \times 10^{-9}}{\sin\theta}$, which can produce a $\sim 2 \times 10^{-6}\;\text{kcal/mol/\AA}$ force difference vs exact `atan2(r_cp, dot)`.
- Therefore, `pallas_angles_harmonic` and `pallas_bonded_all_fused` expose `angle_Regularization_mode: Literal["legacy_denom_eps", "openmm_cross"] = "legacy_denom_eps"` (defaulting to `"legacy_denom_eps"` when `denom_eps > 0` for `< 1e-6` unit-test parity against `bonded.py`, while still using the fast cross-product vector direction $\mathbf{v}_{ji} \times \mathbf{c}_p$!). Specifically, replacing $r_{cp}$ with $r_{\text{denom}} \sqrt{1 - \cos^2\theta_{\text{clipped}}}$ in the cross-product denominator achieves **both** OpenMM's cross-product algebraic simplicity and `< 1e-6` exact parity with `bonded.py:77`.

### 4.4 Mathematical Formulation for O3.3 (Horner 4-Tuple CMAP Bicubic Evaluation)

In [`src/prolix/physics/pallas_bonded.py:2599-2639`](file:///home/marielle/projects/prolix/src/prolix/physics/pallas_bonded.py#L2599-L2639), `_cmap_torsions_kernel` currently builds `u_pows = [1, u, u^2, u^3]` and `v_pows = [1, v, v^2, v^3]` and runs a 16-iteration loop with 3 loads/products per iteration.
Following `/home/marielle/openmm/platforms/common/src/kernels/cmapTorsionForce.cc:66-122`, for a 16-coefficient grid cell $c_{p,q}$ ($p, q \in \{0, 1, 2, 3\}$, flat index $4p + q$):
- For each $p \in \{0, 1, 2, 3\}$, evaluate the cubic in $v$ and its derivative via Horner's rule:
  $$A_p(v) = ((c_{p,3}\, v + c_{p,2})\, v + c_{p,1})\, v + c_{p,0}$$
  $$A'_p(v) = (3 c_{p,3}\, v + 2 c_{p,2})\, v + c_{p,1}$$
- Then evaluate $E(u, v)$, $\partial E / \partial u$, and $\partial E / \partial v$ via Horner's rule in $u$:
  $$E(u, v) = ((A_3\, u + A_2)\, u + A_1)\, u + A_0$$
  $$\frac{\partial E}{\partial u} = (3 A_3\, u + 2 A_2)\, u + A_1, \qquad \frac{\partial E}{\partial v} = ((A'_3\, u + A'_2)\, u + A'_1)\, u + A'_0$$
This replaces 48 multiplications + 48 additions with **21 fused multiply-adds (`fma.rn.f32`)** and eliminates all power arrays `u_pows, du_pows, v_pows, dv_pows`.

---

## 5. Optimization O4: `singlePeriodicCopy` & Interior-Block Periodic Wrap Bypass in `pallas_nonbonded_nl`

### 5.1 Problem Analysis & OpenMM Reference

In [`_nb_nl_kernel`](file:///home/marielle/projects/prolix/src/prolix/pallas_kernels.py#L817-L840), inside `_k_tile_body`, each 32-atom block loads a 2D `(_NB_BLOCK_N, _NB_BLOCK_K) = (32, 32)` tile of neighbor coordinates `px_j, py_j, pz_j` (because Prolix's `neighbor_idx` has shape `(N, K)` per atom) and applies minimum-image wrapping to all $32 \times 32 = 1,024$ pairs on every K-tile:
```python
dx = _min_image_1d(px_i[:, None] - px_j, bx, has_box)  # (32, 32)
dy = _min_image_1d(py_i[:, None] - py_j, by, has_box)  # (32, 32)
dz = _min_image_1d(pz_i[:, None] - pz_j, bz, has_box)  # (32, 32)
```
This executes $3 \times 1,024 = 3,072$ `floor` + multiply + fma operations per K-tile.

Crucially, because `px_j` has shape `(32, 32)` in a per-atom `(N, K)` neighbor list (unlike OpenMM's block-shared `interactingAtoms` tile where 32 i-atoms share a 1D `(32,)` j-vector in `/home/marielle/openmm/platforms/cuda/src/kernels/nonbonded.cu:299-360` and `findInteractingBlocks.cu:41-102`), naively wrapping `px_j` `(32, 32)` relative to the block center $\mathbf{c}_{\text{blk}}$ inside `_k_tile_body` would still execute 3,072 `floor` ops! To eliminate these `floor` operations in both `(N, K)` per-atom neighbor lists and block-shared neighbor lists, we specify a two-tier periodic wrapping architecture:

### 5.2 Tier A (O4.1 — For `(N, K)` Per-Atom Neighbor Lists): Primary-Cell Pre-Wrapping + Interior-Block Zero-Wrap Bypass

1. **Host/Pre-Kernel Primary-Cell Coordinate Wrapping ($3N$ Ops Total)**:
   - Before launching `_nb_nl_kernel` in `_pallas_nonbonded_nl_single` (when `has_box=True`), wrap all $N$ atom coordinates once into the primary periodic cell $[0, L_x) \times [0, L_y) \times [0, L_z)$:
     $$\mathbf{r}^{\text{cell}}_m = \mathbf{r}_m - \mathbf{L} \odot \lfloor \mathbf{r}_m \oslash \mathbf{L} \rfloor \in [0, \mathbf{L}) \qquad \forall m \in \{0, \dots, N-1\}$$
   - This costs only $3N$ `floor` ops across the entire system (vs $3 N K$ inside the kernel—a $K \approx 448\times$ reduction).

2. **Triton-Legal Pre-Loop Block Bounding Box & Interior-Axis Theorem**:
   - Inside `_nb_nl_kernel`, let `valid_i = mask_i` (`(32,)` bool) and `lane_id = jnp.arange(_NB_BLOCK_N, dtype=jnp.int32)`.
   - Extract the first valid atom's coordinate `(ax, ay, az)` using a **Triton-legal masked reduction** (avoiding dynamic scalar indexing `px_i[anchor_idx]`, which is unsupported on Triton register tensors):
     ```python
     first_idx = jnp.argmax(valid_i.astype(jnp.int32))
     is_anchor = lane_id == first_idx
     ax = jnp.sum(jnp.where(is_anchor, px_i, zero_sc))
     ay = jnp.sum(jnp.where(is_anchor, py_i, zero_sc))
     az = jnp.sum(jnp.where(is_anchor, pz_i, zero_sc))
     ```
   - Substitute `(ax, ay, az)` on invalid/ghost lanes (`~valid_i`) so padded coordinates `(0, 0, 0)` never corrupt the block bounding interval:
     ```python
     px_v = jnp.where(valid_i, px_i, ax)
     py_v = jnp.where(valid_i, py_i, ay)
     pz_v = jnp.where(valid_i, pz_i, az)
     r_min_x, r_max_x = jnp.min(px_v), jnp.max(px_v)
     r_min_y, r_max_y = jnp.min(py_v), jnp.max(py_v)
     r_min_z, r_max_z = jnp.min(pz_v), jnp.max(pz_v)
     ```
   - **Interior-Block Zero-Wrap Theorem (Verified to Exact `0.0` Bitwise Error)**:
     Suppose all atoms are pre-wrapped into $[0, L_d)$, and for axis $d \in \{x, y, z\}$ a 32-atom block satisfies the block-uniform interior condition:
     $$\text{interior}_d \iff (\neg\text{has\_box}) \;\lor\; \left(r_{\min,d} \ge r_{\text{margin}} \;\land\; r_{\max,d} \le L_d - r_{\text{margin}}\right)$$
     where $r_{\text{margin}} = \max(r_{\text{cut}}, 6.0\;\text{\AA})$ (accounting for covalently excluded PME erf pairs up to $6\;\text{\AA}$).
     Then for every atom $i$ in the block ($r_{i,d} \in [r_{\text{margin}}, L_d - r_{\text{margin}}]$) and every atom $j \in [0, L_d)$:
     - Whenever the true minimum-image displacement satisfies $|\Delta r_{ij,d}^{\text{MIC}}| < r_{\text{margin}}$, the un-wrapped primary-cell difference $r_{i,d} - r_{j,d}$ already satisfies $|r_{i,d} - r_{j,d}| < r_{\text{margin}} \le \frac{1}{2}L_d$, so **$r_{i,d} - r_{j,d} \equiv \Delta r_{ij,d}^{\text{MIC}}$ identically with zero `floor` operations**!
     - Whenever $|\Delta r_{ij,d}^{\text{MIC}}| \ge r_{\text{margin}}$, optimality of the minimum-image distance over $\mathbb{Z}L_d$ guarantees $|r_{i,d} - r_{j,d}| \ge |\Delta r_{ij,d}^{\text{MIC}}| \ge r_{\text{cut}}$, so `dist < cutoff` is strictly `False`!
   - Inside `_k_tile_body`, guarding each axis with a block-uniform `jax.lax.cond(interior_d, lambda d: d, lambda d: _min_image_1d(d, b_d, True), d_raw)` compiles to a uniform scalar branch (`bra.uni` in PTX) that executes **0 `floor` operations** on every interior axis!

### 5.3 Tier B (O4.2 — For Block-Shared Neighbor Tiles `(N_blocks, K_tiles, 32)`): OpenMM `singlePeriodicCopy`

When O5 Morton spatial sorting clusters each 32-atom block into a compact spatial cell of half-extent $e_{\text{blk},d} \le 3.5\;\text{\AA}$, optional block-shared neighbor tiles (where all 32 i-atoms in block `pid` share a 1D `(32,)` j-atom list per K-tile, matching OpenMM's `findInteractingBlocks.cu:173-348`) load `px_j, py_j, pz_j` as 1D `(32,)` vectors:
1. Pre-wrap the 32 i-atoms relative to `(ax, ay, az)` so boundary-straddling blocks become contiguous:
   $$r_{i,d} \leftarrow r_{i,d} - L_d \lfloor (r_{i,d} - a_d) / L_d + 0.5 \rfloor$$
2. Compute block center $c_{\text{blk},d} = \frac{1}{2}(r_{\max,d} + r_{\min,d})$ and half-extent $e_{\text{blk},d} = \frac{1}{2}(r_{\max,d} - r_{\min,d})$.
3. Whenever $\frac{1}{2}L_d - e_{\text{blk},d} \ge r_{\text{margin}}$ (`singlePeriodicCopy`), wrap the 1D `(32,)` vector `px_j` relative to $c_{\text{blk},d}$ upon load ($3 \times 32 = 96$ ops instead of $3 \times 1,024 = 3,072$ ops — a **32x reduction** even on boundary-straddling blocks!).

---

## 6. Optimization O5: Spatial Atom Sorting / 3D Morton (Z-Order) Reordering

### 6.1 Problem Analysis & OpenMM Reference

In OpenMM:
- `/home/marielle/openmm/platforms/cuda/src/kernels/findInteractingBlocks.cu:104-171` sorts atoms along a 3D Hilbert/Morton curve every time the neighbor list is rebuilt so that each 32-atom block (`TILE_SIZE = 32`) forms a tight spatial cluster with minimal bounding-box volume $\mathbf{e}_{\text{blk}}$.
- `/home/marielle/openmm/platforms/common/src/kernels/pme.cc:1-21` (`findAtomGridIndex`) computes each atom's linear PME grid index `(ix * gridSizeY + iy) * gridSizeZ + iz` and sorts atoms by grid index before spreading charges or gathering forces, maximizing L1/L2 cache line reuse and minimizing shared/global atomic contention.

In Prolix v1:
- Solvated systems place protein atoms first (`0 .. N_prot - 1`), followed by water molecules (`N_prot .. N - 1`). As water molecules diffuse across the simulation box over picoseconds, consecutive water indices `32b .. 32b+31` become spatially uncorrelated across the entire $L \times L \times L$ box.
- This has two compounding consequences:
  1. **Defeats O4 (`interior_d` and `singlePeriodicCopy`) for diffused solvent blocks**: If 32 consecutive atom indices span the entire box, $r_{\min,d} \approx 0$ and $r_{\max,d} \approx L_d$, so $e_{\text{blk},d} \approx 0.5 L_d$ and no block qualifies as interior or `singlePeriodicCopy`.
  2. **Degrades L2 cache locality in `pallas_nonbonded_nl`, `pallas_gb_ace_nl`, and `pme_pallas.py`**: In [`pme_pallas.py:72-170`](file:///home/marielle/projects/prolix/src/prolix/physics/pme_pallas.py#L72-L170), 32 atoms $\times$ 64 B-spline stencil points in a block scatter `plt.atomic_add` across uncorrelated PME grid coordinates `(ix, iy, iz)`, causing L2 atomic serialization.

### 6.2 Non-Invasive Spatial Permutation Architecture

Rather than mutating the user-visible state `positions` or re-indexing bonded topology tables `(bonds, angles, dihedrals, cmap)` on every NL rebuild, Prolix v2 applies spatial Morton/PME-grid sorting **transparently at the kernel-execution / neighbor-list level**:

1. **10-Bit per Axis 30-Bit 3D Morton Code (`compute_morton_codes_3d`, Verified on All 10 Bit Positions)**:
   - In [`src/prolix/physics/neighbor_list.py`](file:///home/marielle/projects/prolix/src/prolix/physics/neighbor_list.py), define a pure JAX function `compute_morton_codes_3d(positions: Array, box: Array | None, atom_mask: Array | None = None) -> tuple[Array, Array]`:
     - Map wrapped coordinates $\mathbf{r}_i \in [0, \mathbf{L})$ (or bounding-box normalized coordinates in non-periodic implicit solvent) to 10-bit integers $q_x, q_y, q_z \in [0, 1023]$.
     - Interleave bits via standard shift-and-mask dilations:
       ```python
       def _expand_bits_10(v: Array) -> Array:
         v = (v | (v << 16)) & 0x030000FF
         v = (v | (v << 8)) & 0x0300F00F
         v = (v | (v << 4)) & 0x030C30C3
         v = (v | (v << 2)) & 0x09249249
         return v
       morton = (_expand_bits_10(qx) << 2) | (_expand_bits_10(qy) << 1) | _expand_bits_10(qz)
       ```
     - Force padded/ghost atoms (`~atom_mask`) to `0x7FFFFFFF` so valid atoms sort contiguously into `[0, N_real - 1]` and ghost atoms stay at `[N_real, N_pad - 1]`.
     - Return `perm = jnp.argsort(morton)` and `inv_perm = jnp.argsort(perm)`.

2. **Application to `pme_pallas.py` (`spread_charges_pallas` & `gather_pallas`)**:
   - In [`src/prolix/physics/pme_pallas.py`](file:///home/marielle/projects/prolix/src/prolix/physics/pme_pallas.py):
     - Add optional `atom_perm: Array | None = None` and `atom_inv_perm: Array | None = None` to `spread_charges_pallas` and `gather_pallas`.
     - When `atom_perm` is provided (reused from the cached `NeighborList` permutation with **zero per-step `argsort`**), gather `pos_sorted = positions[atom_perm]`, `q_sorted = charges[atom_perm]`, `mask_sorted = atom_mask[atom_perm]` before calling `_spread_charges_pallas_single`.
     - Because charge spreading is a linear sum over atoms ($Q_{\text{grid}} = \sum_i q_i w_i$), sorting the input atoms by `atom_perm` produces the **exact same PME charge grid** without needing to unsort the grid!
     - In `gather_pallas` (and `_spread_charges_pallas_bwd`), scatter the per-atom outputs back to canonical atom order via `forces = forces_sorted[atom_inv_perm]` and `d_charges = dq_sorted[atom_inv_perm]`.

3. **Sentinel-Safe & Exclusion-Sorted Application to `pallas_nonbonded_nl` and `NeighborList` Cache**:
   - When the neighbor list is built/updated (once every 15–20 MD steps in [`neighbor_list.py`](file:///home/marielle/projects/prolix/src/prolix/physics/neighbor_list.py)):
     - Compute `perm, inv_perm = compute_morton_codes_3d(positions, box, atom_mask)`.
     - **Sentinel- & Tag-Safe Remapping of `neighbor_idx`**:
       - Strip any O1.1 exclusion tag bits `tag_bits = raw_nbr & (~_NB_INDEX_MASK)` and extract `j_raw = raw_nbr & _NB_INDEX_MASK`.
       - Map valid atom indices (`(j_raw >= 0) & (j_raw < N)`) through `inv_perm` while preserving padding sentinels (`N` or `-1`) and re-attaching `tag_bits`:
         ```python
         nbr_perm_rows = neighbor_idx[perm]
         tag_bits = nbr_perm_rows & (~_NB_INDEX_MASK)
         j_raw = nbr_perm_rows & _NB_INDEX_MASK
         valid_j = (j_raw >= 0) & (j_raw < N)
         j_mapped = jnp.where(valid_j, inv_perm[jnp.clip(j_raw, 0, N - 1)], N)
         nbr_sorted = jnp.where(valid_j, j_mapped | tag_bits, N)
         ```
     - **Exclusion-Table Permutation & Row Re-Sorting (When O1.2 Fallback Is Used)**:
       - Permute rows `excl_idx_p = excl_indices[perm]`, map valid entries via `inv_perm`, replace `-1` with `INT32_MAX`, and **re-sort each row ascending** (`sort_order = jnp.argsort(excl_idx_mapped, axis=-1)`) while permuting `excl_scales_vdw[perm]` and `excl_scales_elec[perm]` by `sort_order`. (This is mandatory because `inv_perm` is non-monotonic, so a row sorted in topological order is no longer sorted in Morton order unless re-sorted at NL rebuild time).
     - On each MD step between NL rebuilds: gather 1D per-atom arrays `(pos[perm], charges[perm], sigmas[perm], epsilons[perm], atom_mask[perm])` in $O(N)$ (`< 5 us`), run `_nb_nl_kernel`, and scatter forces back via `forces = forces_sorted[inv_perm]` (and un-permute parameter gradients by `inv_perm` when `any_param_perturbed=True`).
     - **Synergy with O4**: Because every 32-atom block in `perm` order is a contiguous 3D Morton cell of size $\sim L / (N/32)^{1/3}$, the block half-extent $e_{\text{blk},d}$ is $\le 3.5\;\text{\AA}$ even for diffused waters at $N \ge 4096$ ($L \ge 35\;\text{\AA}$), unlocking the O4.1 zero-wrap bypass for all interior blocks and 100% O4.2 `singlePeriodicCopy` coverage!

---

## 7. PR #37 (`b1/full`) vs `main` Integration & Rebase Guidance

Per the trial-merge audit in [.praxia/docs/reference/261026_main_vs_b1full_trial_merge.md](file:///home/marielle/projects/prolix/.praxia/docs/reference/261026_main_vs_b1full_trial_merge.md):

1. **Do NOT run `git rebase main` directly**:
   - `main` and the Pallas feature lineage (`b1/full` / PR #37) share 22 files modified on both sides across dozens of commits (`src/prolix/pallas_kernels.py`, `src/prolix/batched_energy.py`, `src/prolix/physics/system.py`, `src/prolix/physics/neighbor_list.py`, `src/prolix/physics/pme.py`, `src/prolix/physics/flash_explicit.py`, etc.).
   - Replaying commit-by-commit via `git rebase main` forces repeated conflict resolution on intermediate states that were superseded by the full-potential Pallas kernels.

2. **Recommended Merge Workflow**:
   - Perform a single-commit merge (`git merge main` into the feature branch, or merge into an integration branch `integrate/main-plus-pallas-v2`):
     1. **Source Code (`src/prolix/**`, `tests/**`, `scripts/**`)**: Keep the Pallas feature branch (`ours`) as the strict superset for `pallas_kernels.py`, `pallas_bonded.py`, `pallas_implicit.py`, `pme_pallas.py`, `batched_energy.py`, and `system.py`, while preserving `main`'s import-architecture guardrails (`tests/test_import_architecture.py` and `tests/test_stablehlo_export_guardrails.py`).
     2. **Import-Architecture Guardrail Compliance**:
        - Any lazy imports of `pallas_bonded`, `pallas_implicit`, `pme_pallas`, or `pallas_kernels` inside `src/prolix/physics/system.py`, `src/prolix/batched_energy.py`, or `src/prolix/physics/pme.py` MUST either be top-level imports (guarded at module import time inside `pallas_*.py` if Triton is optional) or registered in the explicit allowlist in `tests/test_import_architecture.py` and `tests/test_stablehlo_export_guardrails.py`.
     3. **Lockfile & Telemetry Ledgers**:
        - Resolve `pyproject.toml` and regenerate `uv.lock` cleanly via `uv lock`.
        - Union append-only JSONL files (`.praxia/chronicle/events.jsonl`, `.praxia/commits.jsonl`, `.praxia/telemetry/events.jsonl`) sorted by timestamp and deduplicated by `id` / `commit`.

---

## 8. Verification, Parity, & Engaging GPU Benchmark Plan

### 8.1 Numerical Parity Gates (`tests/physics/test_pallas_v2_optimizations.py`)

Every optimization O1–O5 must pass automated float32 parity tests against the XLA reference and Pallas v1 on both synthetic and real protein systems (`1UAO` chignolin implicit + explicit PME water box):

| Optimization | Test Case | Energy Tolerance | Force Max-Abs / Rel Tolerance | Parameter Grad Tolerance (`compute_param_grads=True`) |
| :--- | :--- | :--- | :--- | :--- |
| **O1.1 & O1.2** | `pallas_nonbonded_nl` with 2-bit pre-encoded NL exclusions (`use_explicit_14=True/False`) vs `[excl_min, excl_max]` fallback vs XLA + GB/SASA neighbor preservation | `< 1e-5` rel | `< 5e-5` rel / `< 1e-4` abs | `< 1e-5` rel vs `jax.grad` |
| **O2.1** | `jax.grad(E)(pos)` (`perturbed=False`) vs `jax.grad(E, argnums=1)(pos, params)` (`perturbed=True`) under `jit(grad)` and `grad(jit)` | Exact (`0.0` diff) | Exact (`0.0` diff on `grad_pos`) | `< 1e-5` rel on all `dE/dparam` |
| **O2.2** | `pallas_bonded_all_fused` (non-overlapping CTA `pid` ranges) vs sum of individual bonded kernels & XLA `bonded.py` | `< 1e-5` rel | `< 1e-5` rel / `< 5e-5` abs | `< 1e-5` rel on bond/angle/torsion/CMAP/1-4 params |
| **O3.1–O3.3** | Bekker cross-product torsions (`b0_is_ji=True/False`), cross-product angles, Horner CMAP vs v1 & XLA | `< 1e-6` rel | `< 1e-5` rel / `< 2e-5` abs | `< 1e-5` rel |
| **O4.1 & O4.2** | Interior-block zero-wrap bypass & `singlePeriodicCopy` (`L = 40 Å, r_cut = 9 Å`) vs fallback (`L = 16 Å, r_cut = 9 Å`) + boundary-straddling & ghost-atom blocks | `< 1e-6` rel | `< 1e-6` rel | `< 1e-6` rel |
| **O5** | Morton-sorted `spread_charges_pallas`, `gather_pallas`, and `pallas_nonbonded_nl` (with sentinel & tag preservation) vs unsorted | `< 1e-5` rel | `< 2e-5` rel | `< 1e-5` rel |

### 8.2 Engaging GPU Performance Targets (`RTX PRO 6000 Blackwell`)

Re-run `scripts/benchmarks/full_potential_pallas_vs_xla.py` and `scripts/benchmarks/kups_benchmark.py --tiers tier4` on Engaging via `myxcel`:
- **$N = 512$ Small-System Target**: Reduce Pallas E+F step time from `0.579 ms` to `< 0.35 ms` (**>1.15x faster than XLA even at $N = 512$**, eliminating the v1 small-N crossover penalty).
- **$N = 32,768$ Large-System Target**: Reduce Pallas E+F step time from `2.581 ms` to `< 1.85 ms` (**>4.8x faster than XLA**, reaching **>45 ns/day** on the 20-step BAOAB+SETTLE explicit-PME propagator).

---

## 9. Multi-Round Adversarial Specification Review Log

### Round 1 — Challenger Audit (Initial Draft)

- **[BLOCKER-R1-1] O1.1 Tag Collision & PME Exclusion Erf Interaction**:
  - *Challenger*: In O1.1, if excluded neighbors in `neighbor_idx` are replaced with the padding sentinel `N` to skip them in `_nb_nl_kernel`, what happens when `include_pme_excl_erf=True` (the default in `batched_energy.py:773` and `system.py:725` where `_nb_nl_kernel` evaluates the $-\text{erf}(\alpha r)/r$ PME exclusion correction on excluded pairs inside the neighbor list)? Replacing excluded pairs with `N` would silently drop their $-\text{erf}(\alpha r)/r$ energy and force contribution!
  - *Defender Resolution*: Fixed in Section 2.2 by bit-tagging excluded entries in `neighbor_idx` rather than replacing them with `N`. (Further refined in Round 4 `BLOCKER-R4-3` to lossless 2-bit tagging `_NB_EXCL_FULL_BIT = 1 << 30` and `_NB_EXCL_14_BIT = 1 << 29`).

- **[BLOCKER-R1-2] O4 `singlePeriodicCopy` Failure on Padded/Ghost Atoms & Boundary-Straddling Blocks**:
  - *Challenger*: In `_nb_nl_kernel`, the last 32-atom block (or padded buckets in `single_padded_energy` where `N_pad = 1024` has only `N_real = 512` real atoms) contains padded atoms with coordinates `(0.0, 0.0, 0.0)` and `mask_i = False`. Computing `min(px_i)` and `max(px_i)` over all 32 lanes without masking out `~mask_i` corrupts the bounding box with `(0, 0, 0)` ghost coordinates, and boundary-straddling blocks have `max(px_i) - min(px_i) ≈ L_x`.
  - *Defender Resolution*: Fixed in Section 5.2–5.3 by computing the first valid atom's anchor coordinate `(ax, ay, az)`, substituting `(ax, ay, az)` on invalid lanes (`~valid_i`), and pre-wrapping block coordinates relative to `(ax, ay, az)` in O4.2. (Further refined in Round 4 `MAJOR-R4-4` to use Triton-legal masked `jnp.sum(jnp.where(is_anchor, px_i, 0.0))` and the O4.1 interior-block zero-wrap bypass).

- **[MAJOR-R1-3] O3.1 Bekker Torsion Convention Between `b0_is_ji=True` and `b0_is_ji=False`**:
  - *Challenger*: Prolix has two conventions for the first bond vector $\mathbf{b}_0$: `b0_is_ji=False` and `b0_is_ji=True`. If `torsionForce.cc:27-37` is transcribed without accounting for `b0_is_ji`, either `system.py` (`b0_is_ji=False`) or `batched_energy.py` (`b0_is_ji=True`) will have wrong torsion angles or inverted forces.
  - *Defender Resolution*: Addressed in Section 4.2 and verified by direct numerical execution against `_torsion_angle_and_atom_grads_vec3` in Round 4 (`BLOCKER-R4-1`).

- **[MAJOR-R1-4] O2.1 Active-AD Detection Under `jax.jit` and `jax.vmap`**:
  - *Challenger*: In O2.1, checking `isinstance(x, jax.core.Tracer)` inside `@jax.jit` always returns `True` because `jit` wraps all arguments in `DynamicJaxprTracer`.
  - *Defender Resolution*: Addressed in Section 3.2(1) and upgraded in Round 4 (`BLOCKER-R4-2`) to use JAX's public `custom_vjp.defvjp(fwd, bwd, symbolic_zeros=True)` protocol (`arg.perturbed`).

---

### Round 2 — Challenger Audit (Post-Round-1 Revisions)

- **[MAJOR-R2-1] O1.2 Binary Search Memory Access in Triton/Pallas**:
  - *Challenger*: In O1.2 (the fallback when `excl_mode != "pre_encoded"`), how does `_nb_nl_kernel` avoid keeping `(32, 32)` exclusion arrays live across `fori_loop` or performing scattered global memory loads?
  - *Defender Resolution*: Clarified in Section 2.3(2): O1.1 (`excl_mode = "pre_encoded"`) is the default path with 0 exclusion loads; in the O1.2 fallback, pre-loaded 1D `excl_min_i` `(32,)` and `excl_max_i` `(32,)` (2 regs/thread) skip >90% of K-tiles via `jax.lax.cond(jnp.any(in_excl_range), ...)`, and `_lookup_excl_tile` loads `excl_idx_i` only inside the conditional branch.

- **[MAJOR-R2-2] O2.2 Bonded Megakernel Shared Memory & Register Footprint When CMAP Is Fused**:
  - *Challenger*: If `pallas_bonded_all_fused` evaluates all bonded groups sequentially in the same CTA (`pid < n_blocks_group`), live register ranges across groups and 20+ sequential `pl.atomic_add` calls will degrade occupancy.
  - *Defender Resolution*: Fixed in Section 3.3(2) by assigning **non-overlapping CTA `pid` ranges** $[O_g, O_{g+1})$ to each bonded group $g$ with total grid size $N_{\text{blk,total}} = \sum_g N_{\text{blk},g}$. Each CTA executes exactly one bonded group from start to finish via uniform `bra.uni` dispatch.

- **[MINOR-R2-3] O5 Morton Code Overhead If Recomputed Every Step**:
  - *Challenger*: Calling `compute_morton_codes_3d` and `jnp.argsort(morton)` on every MD step would cost more than the L2 cache savings.
  - *Defender Resolution*: Clarified in Section 6.2(2–3) that Morton sorting and `neighbor_idx` remapping run **only on neighbor-list rebuild steps** (every 15–20 MD steps) and are cached on `NeighborList` for reuse by `pallas_nonbonded_nl`, `spread_charges_pallas`, and `gather_pallas`.

---

### Round 3 — Challenger Audit (Post-Round-2 Revisions)

- **[MINOR-R3-1] StableHLO Export Guardrail Compatibility (`tests/test_stablehlo_export_guardrails.py`)**:
  - *Challenger*: How do O1–O5 preserve `tests/test_stablehlo_export_guardrails.py`?
  - *Defender Resolution*: All Pallas entrypoints check `in_export_context()` and retain their pure XLA fallback when `use_pallas=False` or on CPU. Furthermore, `encode_nl_exclusions` (O1.1) and `compute_morton_codes_3d` (O5) use pure `jax.numpy` / `jax.lax` primitives.

- **[MINOR-R3-2] Determinism of Non-Overlapping CTA Dispatch in `pallas_bonded_all_fused`**:
  - *Challenger*: Is the energy reduction in `pallas_bonded_all_fused` deterministic across runs?
  - *Defender Resolution*: Yes: each CTA `pid` writes to its own unique slot `e_blk_ref[pid] = e_sum` without atomic contention, followed by a deterministic 1D `jnp.sum(e_blk)`.

---

### Round 4 — Independent Executable Verification Audit (Code & Math Stress-Test)

An independent verification pass executed the mathematical formulas and JAX tracer logic from Rounds 1–3 directly against Prolix's codebase (`src/prolix/physics/pallas_bonded.py`, `src/prolix/pallas_kernels.py`, `src/prolix/physics/neighbor_list.py`) using `uv run python`, uncovering two fatal bugs, three major bugs, and one minor bug in the Round 3 draft:

- **[BLOCKER-R4-1] O3.1 Bekker Torsion Formula Had Inverted `b0_is_ji` Definition, Inverted `atan2` $y$-Sign, and Inverted $\mathbf{f}_l$ Force Sign (`3.13–5.17 rad` Error)**:
  - *Challenger*: Running the Round 3 Section 4.2 formula against `_torsion_angle_and_atom_grads_vec3` (`src/prolix/physics/pallas_bonded.py:1907-1988`) on 32 random 4-atom torsions produced max angle errors of `3.1300 rad` (`b0_is_ji=False`) and `5.1723 rad` (`b0_is_ji=True`) and `1.57–3.90` gradient errors! Root causes:
    1. `pallas_bonded.py:1923-1931` defines $\mathbf{b}_0 = \mathbf{r}_j - \mathbf{r}_i$ when `b0_is_ji=True` and $\mathbf{b}_0 = \mathbf{r}_i - \mathbf{r}_j$ when `b0_is_ji=False` (the exact opposite of Round 3's $s_0$ definition, which also cited a non-existent `src/prolix/physics/dihedral.py`).
    2. For $\mathbf{b}_0^{\text{ij}} = \mathbf{r}_i - \mathbf{r}_j$, $\mathbf{b}_0^{\text{ij}} \cdot \mathbf{n}_2 = -r_{b_1} y_{\text{v1}}$ (opposite sign to Round 3's $+r_{b_1}(\mathbf{b}_0 \cdot \mathbf{n}_2)$), and `b0_is_ji=True` requires `atan2(-y_raw, -x_raw) - pi` (`pallas_bonded.py:1955`).
    3. Both $\frac{\partial \phi}{\partial \mathbf{r}_i} = +\frac{r_{b_1}}{\|\mathbf{n}_1\|^2}\mathbf{n}_1$ and $\frac{\partial \phi}{\partial \mathbf{r}_l} = +\frac{r_{b_1}}{\|\mathbf{n}_2\|^2}\mathbf{n}_2$ have positive signs (so $\mathbf{f}_l = -\frac{\partial E}{\partial \phi}\frac{r_{b_1}}{\|\mathbf{n}_2\|^2}\mathbf{n}_2$, opposite to Round 3's $+\frac{\partial E}{\partial \phi}$), and $\mathbf{s}_f = \frac{\mathbf{b}_0^{\text{ij}}\cdot\mathbf{b}_1}{r_{b_1}^2}\mathbf{f}_i + \frac{\mathbf{b}_2\cdot\mathbf{b}_1}{r_{b_1}^2}\mathbf{f}_l$.
  - *Defender Resolution*: Fixed in Section 4.2 and verified via `uv run python` against `_torsion_angle_and_atom_grads_vec3`: max angle error is now **`2.38e-07 rad`** and max atom gradient errors are **`gi=5.96e-08, gj=1.04e-07, gk=3.58e-07, gl=3.58e-07`** across both `b0_is_ji=False` and `b0_is_ji=True`. Removed all references to non-existent `dihedral.py`.

- **[BLOCKER-R4-2] O2.1 `_needs_param_grads` Returned `False` in All Cases Under Modern JAX (`LinearizeTracer` & `grad(jit)`)**:
  - *Challenger*: Executing Round 3's `_needs_param_grads` under `jax.jit(jax.grad(f, argnums=1))` and `jax.grad(jax.jit(f), argnums=1)` showed that `_needs_param_grads` returned `False` even when differentiating wrt `params` (`argnums=1`)! In modern JAX, `jax.grad` uses `LinearizeTracer` (rejected by `type(x).__name__ in ("JVPTracer", "DynamicJaxprTracer")`), and under `jax.grad(jax.jit(f), argnums=1)`, `jit` traces `f` first with `DynamicJaxprTracer` for all arguments before `ad.linearize` runs.
  - *Defender Resolution*: Fixed in Section 3.2(1) by replacing `_needs_param_grads` tracer-class inspection with JAX's official `custom_vjp.defvjp(fwd, bwd, symbolic_zeros=True)` API, checking `arg.perturbed` on each `CustomVJPPrimal` in `fwd` and passing `param_res = None` when `not any_param_perturbed`. Verified via `uv run python` across `jax.grad(f, argnums=0)`, `jax.jit(jax.grad(f, argnums=0))`, `jax.jit(jax.grad(f, argnums=(0, 1)))`, and `jax.grad(jax.jit(f), argnums=1)`.

- **[MAJOR-R4-3] O1.1 Sentinel Replacement (`n_atoms`) Corrupted GB/SASA Neighbor Lists, Ignored `use_explicit_14=False` 1-4 Scales, and Broke `_nonbonded_nl_param_grads`**:
  - *Challenger*:
    1. Replacing excluded 1-2/1-3/1-4 entries in `neighbor_idx` with `n_atoms` when `retain_tagged_for_pme_erf=False` deletes covalently bonded neighbors from `pallas_gb_ace_nl` and `pallas_sasa_nl` (`batched_energy.py:1432`), which require all bonded neighbors in Born radii and SASA burial.
    2. When `use_explicit_14=False` (`neighbor_list.py:132, 356-363`), `excl_indices` contains 1-4 pairs with scales `(0.5, 0.83333333)`, not `(0.0, 0.0)`.
    3. In `_nonbonded_nl_param_grads` (`pallas_kernels.py:1132`), `nbr_i < N` evaluates to `False` if `nbr_i` has bit 30 set.
  - *Defender Resolution*: Fixed in Section 2.1–2.2: `encode_nl_exclusions` now uses **lossless 2-bit tagging** (`_NB_EXCL_FULL_BIT = 1 << 30` for `(0.0, 0.0)` and `_NB_EXCL_14_BIT = 1 << 29` for `(scale_14_vdw, scale_14_elec)`, with `_NB_INDEX_MASK = (1 << 29) - 1`), always preserving the true atom index `j` in the lower 29 bits for `pallas_gb_ace_nl`, `pallas_sasa_nl`, and `_nonbonded_nl_param_grads`, and reusing `get_neighbor_exclusion_scales` (`neighbor_list.py:465-520`).

- **[MAJOR-R4-4] O4 Assumed `px_j` Had Shape `(32,)` Instead of `(32, 32)` in `_nb_nl_kernel` and Used Unsupported `px_i[anchor_idx]`**:
  - *Challenger*: In `_nb_nl_kernel` (`pallas_kernels.py:821`), `px_j` has shape `(32, 32)` because `neighbor_idx` is `(N, K)` per atom, so wrapping `px_j` inside `_k_tile_body` still executed 3,072 `floor` ops. Moreover, `px_i[anchor_idx]` fails Triton lowering on register tensors.
  - *Defender Resolution*: Fixed in Section 5.1–5.3:
    1. Specified **Tier A (O4.1)** for `(N, K)` per-atom neighbor lists: pre-wrap all $N$ coordinates into $[0, L_d)$ once before the kernel ($3N$ ops), compute `(ax, ay, az)` via Triton-legal masked reduction `jnp.sum(jnp.where(lane_id == first_idx, px_i, 0.0))`, and bypass per-pair `floor` completely (`0` ops) on every axis satisfying the **Interior-Block Condition** $r_{\min,d} \ge r_{\text{margin}} \land r_{\max,d} \le L_d - r_{\text{margin}}$ (verified in Python with `0.0` bitwise error across all $32 \times 1024$ pairs).
    2. Specified **Tier B (O4.2)** for block-shared `(N_blocks, K_tiles, 32)` neighbor tiles where `px_j` is genuinely 1D `(32,)` and `singlePeriodicCopy` wraps `px_j` in 96 ops even on boundary blocks.

- **[MAJOR-R4-5] O2.2 Body-vs-Log Inconsistency on `pallas_bonded_all_fused` Grid Size**:
  - *Challenger*: Round 2 (`MAJOR-R2-2`) and Round 3 (`MINOR-R3-2`) stated that Section 3.3(2) used non-overlapping CTA `pid` ranges with $N_{\text{blk,total}} = \sum_g N_{\text{blk},g}$, but Section 3.3(2) still had $N_{\text{blk}} = \max_g N_{\text{blk},g}$ with overlapping `if pid < n_blocks_group` checks.
  - *Defender Resolution*: Fixed in Section 3.3(2) to explicitly define cumulative block offsets $O_g$, total grid size $N_{\text{blk,total}} = \sum_g N_{\text{blk},g}$, and disjoint per-CTA dispatch `pid in [O_g, O_{g+1})`.

- **[MINOR-R4-6] O5 Morton Remapping Out-of-Bounds on `neighbor_idx` Padding Sentinels `N` and Unsorted `excl_indices`**:
  - *Challenger*: Mapping `neighbor_idx` through `inv_perm[j]` without masking `j < N` indexes out of bounds on padding sentinel `N` (or corrupts O1.1 tag bits), and mapping `excl_indices` through non-monotonic `inv_perm` leaves rows unsorted for O1.2.
  - *Defender Resolution*: Fixed in Section 6.2(3) with explicit tag stripping/re-attachment, `valid_j = (j_raw >= 0) & (j_raw < N)` masking before `inv_perm[jnp.clip(j_raw, 0, N - 1)]`, and row-wise `jnp.argsort` re-sorting of `excl_indices` at NL rebuild time.

---

### Round 5 — Final Verification & Sign-Off

- **Challenger Assessment**:
  - Blocker findings remaining: **0**
  - Major findings remaining: **0**
  - Minor findings remaining: **0**
  - All mathematical formulations (O3.1 Bekker torsion angle and force projection for both `b0_is_ji=False` and `b0_is_ji=True`, O3.2 cross-product angle forces, O3.3 Horner CMAP evaluation, O4.1 interior-block zero-wrap theorem, O5 30-bit Morton bit-expansion, and O2.1 `symbolic_zeros=True` `CustomVJPPrimal.perturbed` gating) have been executed and numerically verified against Prolix's reference implementations.
- **Verdict**: **`ACCEPT`**
