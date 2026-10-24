# Prolix Pallas Kernel Optimization & `xtrax` Post-Equinox Alignment Specification (v2)

**Date**: 2026-10-23
**Task ID**: `261023_pallas_xtrax_optimization`
**Status**: v2 (Incorporating Adversarial Critique Round 1 Resolutions)

---

## 1. Executive Summary & Architectural Thesis

`prolix` has established $O(N^2)$ tiled `lax.scan` ("FlashMD") and $O(N \cdot K)$ neighbor-list paths, alongside an initial Pallas SPME prototype (`src/prolix/physics/pme_pallas.py`, ADR-006) and an unwired Born-radii prototype (`src/prolix/pallas_kernels.py`). A deep hardware, compiler, and codebase audit reveals five critical optimization and correctness axes:

1. **Silent Correctness Bugs in Existing Custom VJPs & $O(N^2)$ NL Leakage**:
   - `src/prolix/physics/optimization.py:182-196, 346-359`: `_chunked_lj_nl_bwd` and `_chunked_coulomb_nl_bwd` are stubs returning `-g * jnp.zeros_like(r)` (zero position gradients), and both dense and NL custom VJPs (`_chunked_lj_bwd` L125, `_chunked_coulomb_bwd` L299) return zero parameter cotangents (`grad_sigma`, `grad_eps`, `grad_q`) and omit the switching-function derivative $S'(r)$.
   - `src/prolix/pallas_kernels.py:130, 172, 244-265`: `_gb_coulomb_forces_chunked` has a factor-of-2 bug (multiplying row-$i$-only derivative sums by `prefactor = -0.5 * k_e * tau` instead of `2 * prefactor`), and `_gb_coulomb_bwd` returns `(grad_positions, None, None, None)` — omitting `dE/d(born_radii)` and `dE/d(charges)` and breaking the $\sum_k \frac{\partial E_{\text{GB}}}{\partial B_k}\frac{\partial B_k}{\partial \mathbf{r}_i}$ chain rule in `gb_energy_split_cutoff`.
   - `src/prolix/batched_energy.py:991-1011, 1072-1092`: `single_padded_energy_nl_cvjp` and `single_padded_energy_nl` compute `_excl_scales_elec_nl` via `get_neighbor_exclusion_scales` and then discard it, unconditionally allocating a dense $(N, N)$ exclusion matrix and running $O(N^2)$ dense `_coulomb_energy_masked`!

2. **SIMT Underutilization & Unrolled Loops in Existing Pallas Kernels**:
   - `src/prolix/physics/pme_pallas.py:39-208`: `spread_charges_pallas` and `gather_pallas` launch `grid=(n_atoms,)` where each program instance (`pl.program_id(0)`) processes **1 scalar atom**. On the Triton GPU backend, each program instance is a CUDA Thread Block (CTA, $\ge 32$ threads), leaving 31/32 warp lanes masked off (3.1% SIMT utilization) and issuing 64 unrolled scalar global atomics/loads per CTA.
   - `src/prolix/pallas_kernels.py:313-408`: `pallas_born_radii` uses nested Python `for i in range(128): for k in range(K_inner):` loops over scalar array elements, unrolling into a massive single-lane Triton IR block with no `@jax.custom_vjp`, no `custom_vmap`, and no CPU `interpret=True` fallback.

3. **HBM Bandwidth & Intermediate Tensor Materialization in Nonbonded & PME Paths**:
   - In `src/prolix/physics/flash_explicit.py`, each `lax.scan` tile step materializes an $(N, M=32, T=256)$ exclusion comparison tensor ($>1\text{ GB/tile}$, $137\text{ GB/step}$ at $N=32,768$), pads `excl_indices` with `0` and `excl_scales_vdw` with `0.0` in `_pad_to_tile_multiple` (`L90-93`), calls host-only `topology.find_bonded_exclusions` inside traced energy evaluation (`L409, 483, 574`), and uses `@eqx.filter_grad` + `@jax.checkpoint` (recomputing the forward scan in backward and differentiating wrt all floating-point fields of `PaddedSystem`).
   - In `src/prolix/physics/neighbor_list.py:396-462`, `get_neighbor_exclusion_scales` runs a 32-iteration sequential `lax.fori_loop` over $(N, K)$ on every call even though `excl_indices` rows are already sorted ascending (`L302, 386`).
   - In `src/prolix/physics/pme.py:184-206, 286-305, 396-407`, the 3D reciprocal Green's function grid `G` is duplicated inline in `_spme_fwd` and recomputed on every MD step even when `box_size` and `grid_dims` are fixed in `make_spme_energy_fn`.

4. **SETTLE / Water RATTLE / Integrator Hot-Path Bottlenecks (`src/prolix/physics/settle.py`)**:
   - `settle_velocities` (`L928-942`) runs 10 global gather/scatter iterations of diagonal Jacobi updates for 3-site water RATTLE, whereas rigid-triangle water velocity constraints form a coupled $3 \times 3$ linear system with constant positive diagonal $(\mu_O+\mu_H, \mu_O+\mu_H, 2\mu_H)$ solvable to machine precision in **1 exact pass** in registers via Cramer's rule (`_solve_3x3_batch`).
   - `settle.py` executes 16+ full-array `jnp.concatenate([arr, arr[:1]], axis=0)` copies per step (`L210, 279, 520, 568, 726, 865, 1244, 2299`) to avoid duplicate-index `.set()` collisions on Atom 0 from `[0, 0, 0]` padded water rows. Converting overwrite sites to race-free additive deltas (`arr.at[idx].add(jnp.where(mask, new_val - arr[idx], 0.0))`) and additive sites to masked `.add(jnp.where(mask, dv, 0.0))` eliminates all 16 full-array copies while remaining 100% commutative, associative, and race-free on Atom 0.
   - `_ou_noise_one_water_rigid` (`L2200-2203`) and `_project_one_water_momentum_rigid` (`L226-254`) invoke `jnp.linalg.cholesky(3x3)` + `solve(L.T, ...)` and `jnp.linalg.solve(6x6)` inside `vmap` every step. Because `rrel` is COM-centered, the $6\times 6$ rigid-body projection decouples into $\mathbf{v}_{\text{COM}} = \frac{1}{M_{\text{tot}}}\sum \mathbf{p}_i$ plus a $3\times 3$ solve on $I_{\text{reg}}$, and $L_I^T \boldsymbol{\omega} = \sqrt{k_B T}\,\mathbf{z}_{\text{rot}}$ is solved via a 20-op closed-form $3\times 3$ Cholesky + upper-triangular backward substitution (`_cholesky_upper_solve_3x3`).

5. **Alignment with `xtrax` (`v0.4.1b3`, `equinox-decouple-pallas-261023`)**:
   - `xtrax` is decoupling from `equinox` (`eqx.Module`, `eqx.filter_jit`, `eqx.filter_grad`) toward `@dataclass(frozen=True)` + `@jax.tree_util.register_dataclass` (with `dataclasses.field(metadata={"static": True})`), standard `jax.jit` / `jax.grad` / `jax.jvp`, and `ChunkedMap` (replacing `SafeMap`).
   - **Core Boundary Rule**: Every Pallas kernel and `@jax.custom_vjp` primitive in `prolix` MUST have a **100% `equinox`-free signature**: dynamic `jax.Array` positional arguments first, followed by hashable host primitives (`tuple[int, ...]`, `float`, `int`, `bool`) marked in `nondiff_argnums`. New kernel configuration containers MUST use `@jax.tree_util.register_dataclass` + `@dataclass(frozen=True)`.

---

## 2. Requirements & Technical Specifications

### R1: Post-Equinox `xtrax` Alignment, Export Context & Kernel Boundary Contract

#### R1.1 `SafeMap` $\to$ `ChunkedMap` Compatibility (`src/prolix/tiling/xtrax_adapter.py`, `src/prolix/tiling/axis_dispatch.py`, `src/prolix/api/ensemble_dispatch.py`)
- Import `ChunkedMap` from `xtrax.tiling.strategy` with a fallback alias to `SafeMap`:
  ```python
  try:
      from xtrax.tiling.strategy import Bucket, ChunkedMap, DedupGather, SafeMap, Scan, Vmap, WhileCarry
  except ImportError:
      from xtrax.tiling.strategy import Bucket, ChunkedMap, DedupGather, Scan, Vmap, WhileCarry
      SafeMap = ChunkedMap
  ```
- Accept `(SafeMap, ChunkedMap)` in `isinstance` checks across `xtrax_adapter.py:153`, `axis_dispatch.py:59`, and `ensemble_dispatch.py:120, 164`.
- In `src/prolix/tiling/axis_dispatch.py:59-68`: only enforce `n_waters % strategy.batch_size != 0` when running against legacy `xtrax < 0.4.0a11` where `SafeMap` is not `ChunkedMap`; additionally guard `m_stack` division in `src/prolix/physics/settle.py:240` (`jnp.maximum(m_stack[:, None], 1e-30)`) and `msum` in `settle.py:2182, 2189` (`jnp.maximum(msum, 1e-30)`) so zero-padded remainder water slots never produce `0 / 0 = NaN`.

#### R1.2 Eliminate `equinox` from Physics Force & Kernel Boundaries (`src/prolix/physics/flash_explicit.py`, `src/prolix/physics/system.py`)
- In `src/prolix/physics/flash_explicit.py:17, 683-716`: remove `import equinox as eqx` and replace `@eqx.filter_grad` over `PaddedSystem` with `jax.grad` wrt `positions: Array` as argument 0 (`_total_energy_from_positions(positions, sys, ...)`), so JAX only differentiates wrt `positions` (`(N, 3)`) instead of all floating-point leaves of `PaddedSystem`.
- Also replace traced calls to `topology.find_bonded_exclusions(sys.bonds, n_atoms)` inside `flash_explicit.py:409, 483, 574` with precomputed `sys.excl_pair_indices` / `sys.excl_pair_scales_elec` (or `sys.excl_indices` / `sys.excl_scales_elec`) so `jax.jit` and `jax.vmap` never hit `TracerArrayConversionError`.
- Define `PallasKernelConfig` in `src/prolix/pallas_kernels.py` using `@jax.tree_util.register_dataclass` + `@dataclass(frozen=True)` with `dataclasses.field(metadata={"static": True})` for all static fields (`block_n`, `block_k`, `use_pallas`, `interpret`).

#### R1.3 Export Context & Backend Resolution (`src/prolix/export.py`, `src/prolix/pallas_kernels.py`, `src/prolix/physics/pme.py`)
- In `src/prolix/export.py`: define `_IN_EXPORT_CONTEXT: contextvars.ContextVar[bool] = contextvars.ContextVar("_IN_EXPORT_CONTEXT", default=False)` and `@contextlib.contextmanager def export_context():`. Enter `with export_context():` around `jax.export.export(...)` inside both `export_energy_fn` (`export.py:61`) and `export_langevin_step` (`export.py:97`).
- Define `resolve_pallas_mode(use_pallas: bool | None = None, env_var: str = "PROLIX_USE_PALLAS") -> tuple[bool, bool]` returning `(enabled, interpret)`:
  - If `_IN_EXPORT_CONTEXT.get()` is `True`, return `(False, False)` unconditionally so `jax.export` and `iree-compile` (`wasm32-unknown-unknown`) always trace pure-JAX HLO without `triton_kernel_call`.
  - Otherwise resolve `enabled` from explicit `use_pallas` (if not `None`) else `os.environ.get(env_var)` / `os.environ.get("PROLIX_USE_PALLAS_PME")`.
  - When `enabled` is `True`, set `interpret = (jax.default_backend() != "gpu")`.

---

### R2: Fix Silent Custom-VJP Correctness Bugs & $O(N^2)$ NL Leakage

#### R2.1 Exact Analytical Position & Parameter VJPs in `src/prolix/physics/optimization.py`
Fix both the neighbor-list custom VJPs (`_chunked_lj_nl_bwd` at `L182-198`, `_chunked_coulomb_nl_bwd` at `L346-361`) and the dense tiled custom VJPs (`_chunked_lj_bwd` at `L76-127`, `_chunked_coulomb_bwd` at `L254-301`) to compute exact position gradients, parameter gradients (`grad_sigma`, `grad_eps`, `grad_q`, `grad_excl_scales`), and the switching-function derivative $S'(r)$:

1. **LJ Energy & Cotangents (`chunked_lj_energy_nl` & `chunked_lj_energy`)**:
   - Given $s_{ij} = \frac{1}{2}(\sigma_i + \sigma_j)$, $\epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j + 10^{-30}}$, $u_{ij} = (s_{ij} / r_{ij})^6$, $V_{\text{unsw}}(r_{ij}) = 4\epsilon_{ij}(u_{ij}^2 - u_{ij})$, switching weight $S(r_{ij}) = \text{lj\_switch\_weight}(r_{ij}, r_{\text{cut}}, w)$, and switching derivative for $r_{\text{sw}} = r_{\text{cut}} - w < r_{ij} < r_{\text{cut}}$ with $z = \text{clip}((r_{ij} - r_{\text{sw}})/w, 0, 1)$:
     $$S'(r_{ij}) = \begin{cases} -\frac{30 z^2 (1 - z)^2}{w} & \text{if } w > 0 \text{ and } r_{\text{sw}} < r_{ij} < r_{\text{cut}} \\ 0 & \text{otherwise} \end{cases}$$
   - Because the energy sum includes $\frac{1}{2}$ over symmetric pairs ($(i, j)$ and $(j, i)$ both present in `jax_md.partition.Dense` neighbor lists and dense matrices), summing over $j$ in row $i$ without the $\frac{1}{2}$ factor gives the exact total derivatives for atom $i$:
     $$\begin{aligned}
     \frac{\partial e_{ij}}{\partial r_{ij}} &= m_{ij} w_{\text{excl}, ij} \left[ 4\epsilon_{ij} \frac{-12 u_{ij}^2 + 6 u_{ij}}{r_{ij}} S(r_{ij}) + V_{\text{unsw}}(r_{ij}) S'(r_{ij}) \right] \\
     \nabla_{\mathbf{r}_i} E_{\text{LJ}} &= \sum_k \frac{\partial e_{ij}}{\partial r_{ij}} \frac{\mathbf{r}_i - \mathbf{r}_j}{r_{ij}} \\
     \frac{\partial E_{\text{LJ}}}{\partial \sigma_i} &= \sum_k m_{ij} w_{\text{excl}, ij} S(r_{ij}) \cdot 4\epsilon_{ij}(12 u_{ij}^2 - 6 u_{ij}) \frac{1}{2 s_{ij}} \\
     \frac{\partial E_{\text{LJ}}}{\partial \epsilon_i} &= \sum_k m_{ij} w_{\text{excl}, ij} S(r_{ij}) \cdot 2(u_{ij}^2 - u_{ij}) \frac{\epsilon_{ij}}{\max(\epsilon_i, 10^{-30})} \\
     \frac{\partial E_{\text{LJ}}}{\partial w_{\text{excl}, ij}} &= \frac{1}{2} m_{ij} V_{\text{unsw}}(r_{ij}) S(r_{ij}) \quad (\text{accumulated onto matching } (i, m) \text{ exclusion slots})
     \end{aligned}$$
   - Return `(g * grad_r, g * grad_sigma, g * grad_eps, jnp.zeros_like(excl_indices), g * grad_excl_scales, jnp.zeros_like(nb_idx))` in `_chunked_lj_nl_bwd` (and without `nb_idx` in `_chunked_lj_bwd`).

2. **Coulomb / Direct-Space PME Energy & Cotangents (`chunked_coulomb_energy_nl` & `chunked_coulomb_energy`)**:
   - For $E_{\text{coul}} = \frac{1}{2} \sum_{i, j} m_{ij} C \frac{q_i q_j}{r_{ij}} \left(w_{\text{excl}, ij} - \operatorname{erf}(\alpha r_{ij})\right)$:
     $$\begin{aligned}
     \nabla_{\mathbf{r}_i} E_{\text{coul}} &= \sum_k m_{ij} C q_i q_j \left[ -\frac{w_{\text{excl}, ij} - \operatorname{erf}(\alpha r_{ij})}{r_{ij}^2} - \frac{2\alpha}{\sqrt{\pi}} \frac{e^{-\alpha^2 r_{ij}^2}}{r_{ij}} \right] \frac{\mathbf{r}_i - \mathbf{r}_j}{r_{ij}} \\
     \frac{\partial E_{\text{coul}}}{\partial q_i} &= \sum_k m_{ij} C \frac{q_j}{r_{ij}} \left( w_{\text{excl}, ij} - \operatorname{erf}(\alpha r_{ij}) \right) \\
     \frac{\partial E_{\text{coul}}}{\partial w_{\text{excl}, ij}} &= \frac{1}{2} m_{ij} C \frac{q_i q_j}{r_{ij}}
     \end{aligned}$$
   - Return `(g * grad_r, g * grad_q, jnp.zeros_like(excl_indices), g * grad_excl_scales, jnp.zeros_like(nb_idx))` in `_chunked_coulomb_nl_bwd` (and without `nb_idx` in `_chunked_coulomb_bwd`).

#### R2.2 Fix Factor-of-2 Bug & Missing `born_radii` / `charges` Cotangents in `src/prolix/pallas_kernels.py`
- In `_gb_coulomb_forces_chunked` (`src/prolix/pallas_kernels.py:106-186`) and `gb_coulomb_energy_dense` (`L193-270`):
  - With $\text{prefactor} = -\frac{1}{2} C_{\text{coul}} (\epsilon_{\text{in}}^{-1} - \epsilon_{\text{out}}^{-1})$, $P_{ij} = B_i B_j$, $E_{ij} = \exp(-r_{ij}^2 / (4 P_{ij}))$, and $f_{\text{GB}, ij} = \sqrt{r_{ij}^2 + P_{ij} E_{ij} + \epsilon_{\text{sqrt}}}$:
    - Because $E_{\text{GB}} = \text{prefactor} \sum_{i, j} m_{ij} \frac{q_i q_j}{f_{\text{GB}, ij}}$ is symmetric in $(i, j)$, differentiating wrt atom $i$ gets equal contributions from row $i$ ($(i, j)$) and column $i$ ($(j, i)$):
      $$\begin{aligned}
      \nabla_{\mathbf{r}_i} E_{\text{GB}}\Big|_{B} &= -2 \cdot \text{prefactor} \sum_{j=1}^N m_{ij} \frac{q_i q_j \left(1 - \frac{1}{4} E_{ij}\right)}{f_{\text{GB}, ij}^3} (\mathbf{r}_i - \mathbf{r}_j) \\
      \frac{\partial E_{\text{GB}}}{\partial B_i} &= -\text{prefactor} \sum_{j=1}^N m_{ij} \frac{q_i q_j B_j E_{ij} \left(1 + \frac{r_{ij}^2}{4 B_i B_j}\right)}{f_{\text{GB}, ij}^3} \\
      \frac{\partial E_{\text{GB}}}{\partial q_i} &= 2 \cdot \text{prefactor} \sum_{j=1}^N m_{ij} \frac{q_j}{f_{\text{GB}, ij}}
      \end{aligned}$$
    - Note that for $i = j$ ($\mathbf{r}_i - \mathbf{r}_i = \mathbf{0}$), the self-term has $r_{ii} = 0, E_{ii} = 1, f_{\text{GB}, ii} = B_i$, so $\frac{\partial}{\partial B_i}\left(\text{prefactor} \frac{q_i^2}{B_i}\right) = -\text{prefactor} \frac{q_i^2}{B_i^2}$, which matches the $j = i$ term $-\text{prefactor} \frac{q_i^2 B_i (1)(1 + 0)}{B_i^3} = -\text{prefactor}\frac{q_i^2}{B_i^2}$!
  - `_gb_coulomb_bwd` returns `(g * grad_positions, g * grad_charges, g * grad_born_radii, None)`, restoring full chain-rule differentiability to `gb_energy_split_cutoff` (`L276-306`).

#### R2.3 Fix Unconditional Dense $O(N^2)$ Coulomb in `single_padded_energy_nl_cvjp` and `single_padded_energy_nl` (`src/prolix/batched_energy.py:991-1011, 1072-1092`)
- In both `single_padded_energy_nl_cvjp` (`L991-1011`) and `single_padded_energy_nl` (`L1072-1092`):
  - Mask ghost atoms in `neighbor_idx` before evaluation:
    ```python
    N = len(sys.atom_mask)
    nb_idx = jnp.where(
        sys.atom_mask[:, None] & sys.atom_mask[jnp.minimum(neighbor_idx, N - 1)],
        neighbor_idx,
        N,
    )
    ```
  - Use the already-computed `excl_scales_elec_nl` from `get_neighbor_exclusion_scales(sys.excl_indices, sys.excl_scales_vdw, sys.excl_scales_elec, nb_idx, N)` to evaluate $O(N \cdot K)$ neighbor-list Coulomb (`_make_coulomb_energy_nl_cvjp` in `single_padded_energy_nl_cvjp`, and `_coulomb_energy_neighbor_list` in `single_padded_energy_nl`), removing `_build_dense_exclusion_scales` and `_coulomb_energy_masked`.

---

### R3: 5-Step Branchless Binary Search & Register Exclusion Matching (`src/prolix/physics/neighbor_list.py`)
1. **5-Step Branchless Binary Search in `get_neighbor_exclusion_scales` (`src/prolix/physics/neighbor_list.py:396-462`)**:
   - Because `map_exclusions_to_dense_padded` (`L302`) and `pair_list_to_dense_padded` (`L386`) sort valid exclusion indices in ascending order per atom and pad unused slots with `-1` (`max_exclusions = 32`), mapping `< 0` entries to `INT_MAX = jnp.int32(2**30)` produces a monotonically non-decreasing array `excl_sorted` of shape `(N, 32)`.
   - Replace the 32-iteration `lax.fori_loop` with a 5-step ($\log_2 M$) unrolled binary search over `excl_sorted` (for general $M$, $\lceil \log_2 M \rceil$ steps; for $M=32$, steps `(16, 8, 4, 2, 1)`), cutting loop iterations by $6.4\times$ without mutating `neighbor.idx` or breaking OBC-II Born radii descreening.

---

### R4: Vectorized Pallas SPME Charge Spread, Force Gather & Green's Function Caching (`src/prolix/physics/pme_pallas.py` & `pme.py`)

#### R4.1 Vectorized `(BLOCK_N=32, 64)` Pallas SPME Kernels (`src/prolix/physics/pme_pallas.py`)
1. **Grid, Padding & `BlockSpec` Layout**:
   - Set `BLOCK_N = 32` (power of 2; $32 \times 64 = 2048$ stencil elements per CTA).
   - At the Python wrapper boundary of `spread_charges_pallas` and `gather_pallas`:
     - Pad `positions` `(N, 3)`, `charges` `(N,)`, and `atom_mask` `(N,)` to `N_pad = ((N + BLOCK_N - 1) // BLOCK_N) * BLOCK_N` (with `atom_mask = False` and `charges = 0.0` on padded rows).
     - Flatten the 3D mesh `(Kx, Ky, Kz)` to 1D `(Kx * Ky * Kz,)` at the `pallas_call` boundary and reshape back to `(Kx, Ky, Kz)` (or slice `[:N]` on forces/potentials) on return.
2. **Power-of-2 Scalar-Column Loads & 1D Flattened Stencil Indexing (Valid in both `interpret=True` and Triton)**:
   - Inside `_spread_kernel` and `_gather_kernel`, compute `offs = pl.program_id(0) * 32 + jnp.arange(32)` and load coordinates via 1D power-of-2 column slices:
     ```python
     px = pl.load(pos_ref, (offs, 0))  # (32,)
     py = pl.load(pos_ref, (offs, 1))  # (32,)
     pz = pl.load(pos_ref, (offs, 2))  # (32,)
     qi = pl.load(charges_ref, (offs,))  # (32,)
     is_real = pl.load(mask_ref, (offs,)) > 0  # (32,) bool
     ```
   - Compute 1D order-4 B-spline weights `wx, wy, wz` and derivatives `dwx, dwy, dwz` of shape `(32, 4)` in registers.
   - Expand across the $4^3 = 64$ stencil offsets `s_idx = jnp.arange(64, dtype=jnp.int32)` (`ox = s_idx // 16`, `oy = (s_idx // 4) % 4`, `oz = s_idx % 4`) to form `(32, 64)` flattened mesh indices `flat_idx = (ix * Ky + iy) * Kz + iz` and weights `w_2d`.
   - Reshape `flat_idx`, `w_2d`, and `mask_2d` to 1D `(2048,)` (`32 * 64`) before `plt.atomic_add(mesh_out_ref, (flat_idx_1d,), w_1d, mask=mask_1d)` (in spread) and `pl.load(theta_flat_ref, (flat_idx_1d,), mask=mask_1d, other=0.0).reshape(32, 64)` (in gather). This satisfies both Pallas `NDIndexer` (1D index on 1D Ref in `interpret=True`) and Triton 2048-wide vectorized execution!
   - In `gather_pallas`, store forces via three 1D column stores `pl.store(dpos_out_ref, (offs, 0), fx)`, `pl.store(dpos_out_ref, (offs, 1), fy)`, `pl.store(dpos_out_ref, (offs, 2), fz)` and `pl.store(dq_out_ref, (offs,), dq)`.
3. **`custom_vmap` & Named Scopes**:
   - Wrap `spread_charges_pallas` and `gather_pallas` with `@jax.custom_batching.custom_vmap` (using `jax.lax.map` over the leading batch dimension) so `jax.vmap` over molecules/replicas works seamlessly.
   - Pass `name="pme_charge_spread"` and `name="pme_bwd_gather"` to `pl.pallas_call` and wrap `spread_charges_pallas` in `with jax.named_scope("pme_charge_spread"):` in `src/prolix/physics/pme.py:159`.

#### R4.2 Green's Function Precomputation & Deduplication (`src/prolix/physics/pme.py:184-206, 286-305, 396-407`)
- Remove the duplicate inline Green's function math inside `_spme_fwd` (`pme.py:286-305`) and call `influence_function(grid_dims, box, alpha, order)` (or use a precomputed `G` when provided).
- In `make_spme_energy_fn` (`pme.py:396-407`), precompute `G_cached = influence_function(grid_dims, jnp.asarray(box_size), alpha, order)` once at closure construction time when `box_size` is static/concrete, eliminating per-step `pme_greens_setup` recomputation in NVT simulations.

#### R4.3 Expose `use_pallas` on `EnsemblePlan` (`src/prolix/api/ensemble_plan.py`)
- Add `use_pallas: bool | None = None` to `EnsemblePlan` and thread it into `flash_explicit` / `single_padded_energy` / `pme`.

---

### R5: Fused Pallas Nonbonded Kernels ($O(N \cdot K)$ NL, 2D-Tiled Flash Explicit, & Vectorized OBC-II GB)

#### R5.1 Fused $O(N \cdot K)$ Nonbonded Kernel (`pallas_nonbonded_nl` in `src/prolix/pallas_kernels.py`)
- **Capabilities**: Computes damped-erfc PME direct-space electrostatics (or standard Coulomb when `pme_alpha == 0.0`) + switched/softcore Lennard-Jones over a symmetric `(N, K)` neighbor list in a single pass.
- **Tiling & Register Layout**:
  - `BLOCK_N = 32`, `BLOCK_K = 32`, `MAX_EXCL = 32` (all powers of 2).
  - Pad `N` to a multiple of `BLOCK_N = 32` and `K` to a multiple of `BLOCK_K = 32` at the Python wrapper boundary.
  - Preload atom-$i$ coordinates `px_i, py_i, pz_i` `(32,)`, parameters `q_i, sig_i, eps_i` `(32,)`, `mask_i` `(32,)`, and exclusion tables `excl_idx_i, excl_vdw_i, excl_elec_i` `(32, 32)` once before the `lax.fori_loop(0, K_pad // 32, ...)` loop.
  - Inside each `k_tile` step:
    - Load `j_idx` of shape `(32, 32)` (via flattened 1D index `offs_i[:, None] * K_pad + (k_tile * 32 + jnp.arange(32))[None, :]` reshaped to `(1024,)` on `nbr_flat_ref`).
    - Gather `px_j, py_j, pz_j, q_j, sig_j, eps_j` of shape `(32, 32)` from `pl.ANY` refs using `safe_j = jnp.clip(j_idx, 0, N_pad - 1)`.
    - Apply minimum-image convention (when `box` is provided), compute $r_{ij}^2$, and match `j_idx` against `excl_idx_i` `(32, 32)` in registers using the `[excl_min_i, excl_max_i]` bounding check.
    - Compute pairwise LJ + Coulomb energy $e_{ij}$ (using the exact PME exclusion convention matching the caller: either `(scale_elec - erf(alpha * r)) / r` for `optimization.py` or `scale_elec * erfc(alpha * r) / r` for `flash_explicit.py` via a static `include_pme_excl_erf: bool` flag) and radial force factor $-\frac{1}{r_{ij}}\frac{\partial e_{ij}}{\partial r_{ij}}$.
    - Accumulate `acc_fx, acc_fy, acc_fz` `(32,)` and `acc_e` `(32,)` in registers and store once after the loop.
- **Custom VJP & `custom_vmap`**:
  - Wrap with `@jax.custom_vjp`: forward pass executes `pallas_nonbonded_nl` once to obtain `(total_energy, forces)` and saves `(forces, positions, charges, sigmas, epsilons, ...)` in residuals; backward pass returns `-g * forces` for `grad_positions` and computes parameter cotangents `(grad_q, grad_sigma, grad_eps)` when needed.
  - Support `jax.vmap` via `@jax.custom_batching.custom_vmap`.

#### R5.2 2D-Tiled Flash Explicit Nonbonded Kernel (`pallas_flash_explicit_nonbonded` in `src/prolix/physics/flash_explicit.py` / `src/prolix/pallas_kernels.py`)
- Fix `_pad_to_tile_multiple` in `src/prolix/physics/flash_explicit.py:90-93` so `excl_indices` pads with **`-1`** (not `0`) and `excl_scales_vdw` / `excl_scales_elec` pad with **`1.0`** (not `0.0`).
- Provide a forward-force `@jax.custom_vjp` direct-space nonbonded primitive (`BLOCK_I = 32`, `BLOCK_J = 32`) that computes both direct-space energy and analytical forces in a single pass over $j$-tiles (with Pallas execution when `resolve_pallas_mode` enables Pallas and a fused analytical-force `lax.scan` fallback otherwise), eliminating the 2x AD rematerialization pass in `flash_explicit_forces` while keeping PME reciprocal (`spme_energy_with_forces`), `_pme_exclusion_correction_from_pairs`, `spme_background_energy`, and `lj_dispersion_tail_energy` modular outside the direct-space kernel.

#### R5.3 Vectorized Pallas OBC-II Generalized Born Kernel (`pallas_born_radii` in `src/prolix/pallas_kernels.py`)
- Replace the scalar-unrolled double `for` loop in `pallas_born_radii` (`src/prolix/pallas_kernels.py:313-408`) with vectorized `(BLOCK_N = 32, BLOCK_K = 32)` tile operations inside `lax.fori_loop(0, K_pad // 32, ...)`.
- Add `interpret` mode resolution (`interpret = jax.default_backend() != "gpu"` when not explicitly passed), `@jax.custom_batching.custom_vmap`, and an analytical `@jax.custom_vjp` backward pass computing `(grad_positions, grad_radii, None, None, grad_scaled_radii)` so `pallas_born_radii` is fully differentiable under `jax.grad` and `gb_energy_split_cutoff`.

---

### R6: SETTLE / Water RATTLE / Integrator Hot-Path Optimizations (`src/prolix/physics/settle.py`)

#### R6.1 Closed-Form 1-Pass $3 \times 3$ Gram Matrix Water RATTLE in `settle_velocities` (`src/prolix/physics/settle.py:872-979`)
- Replace the 10-iteration `jax.lax.fori_loop` over `_apply_rattle_velocity_correction` in `settle_velocities` with a single-pass exact $3 \times 3$ Gram matrix solve per water molecule:
  - Gather constrained positions `pos_O, pos_H1, pos_H2` and unconstrained velocities `vel_O, vel_H1, vel_H2` **once**.
  - Compute unit bond vectors $\mathbf{n}_1 = \hat{\mathbf{r}}_{OH_1}$, $\mathbf{n}_2 = \hat{\mathbf{r}}_{OH_2}$, $\mathbf{n}_3 = \hat{\mathbf{r}}_{H_1 H_2}$ (zero-masked where `~water_mask`).
  - Construct the $3 \times 3$ Gram matrix $G$ with constant positive diagonal $(\mu_O + \mu_H, \mu_O + \mu_H, 2\mu_H)$ and off-diagonal couplings $G_{01} = G_{10} = \mu_O(\mathbf{n}_1 \cdot \mathbf{n}_2)$, $G_{02} = G_{20} = -\mu_H(\mathbf{n}_1 \cdot \mathbf{n}_3)$, $G_{12} = G_{21} = \mu_H(\mathbf{n}_2 \cdot \mathbf{n}_3)$, and right-hand side $\mathbf{b} = [(\mathbf{v}_{H_1} - \mathbf{v}_O)\cdot\mathbf{n}_1, (\mathbf{v}_{H_2} - \mathbf{v}_O)\cdot\mathbf{n}_2, (\mathbf{v}_{H_2} - \mathbf{v}_{H_1})\cdot\mathbf{n}_3]^T$.
  - Solve $G \boldsymbol{\lambda} = -\mathbf{b}$ in 1 pass via `_solve_3x3_batch(G, -b)` (which has $\det(G) = 2\mu_H(\mu_O + \mu_H)^2 > 0$ even on zero-vector padded water rows!) and scatter the masked velocity corrections `dv_O, dv_H1, dv_H2` to `vel_unconstrained` **once** using `.at[indices.*].add(jnp.where(water_mask[:, None], dv_*, 0.0))`.
  - Retain `_apply_rattle_velocity_correction`, `_apply_rattle_velocity_correction_with_residual`, and `_remove_angular_momentum_from_impulse` as standalone helper functions in `settle.py` for existing unit tests that import them directly.

#### R6.2 Replace All `jnp.concatenate([arr, arr[:1]], axis=0)` Scatters with Race-Free Additive Delta Scatters (`src/prolix/physics/settle.py:210, 279, 520, 568, 726, 865, 1244, 2299`)
- At all overwrite scatter sites (`L210, 279, 1244, 2299`), replace the `(N_atoms + 1, 3)` scratch-row concatenation and `.set()` with race-free masked additive deltas:
  ```python
  delta = jnp.where(mask, new_val - arr[idx], 0.0)
  arr = arr.at[idx].add(delta)
  ```
- At all additive scatter sites (`L520, 568, 726, 865`), replace the `(N_atoms + 1, 3)` concatenation with direct masked `.add(jnp.where(water_mask[:, None], dv, 0.0))`, and guard `_inv` in Cramer's rule (`L690, 840`) against `_det == 0` on padded rows so `NaN` is never produced before masking.

#### R6.3 Closed-Form $3 \times 3$ Upper-Triangular Cholesky & Decoupled $3 \times 3$ Rigid Momentum Projection (`src/prolix/physics/settle.py:226-254, 2152-2301`)
- Replace `L_I = jnp.linalg.cholesky(I_reg); omega = jnp.linalg.solve(L_I.T, ...)` in `_ou_noise_one_water_rigid` (`L2200-2203`) with closed-form `_cholesky_upper_solve_3x3(I_reg, jnp.sqrt(kT) * z_rot)` solving $L_I^T \boldsymbol{\omega} = \sqrt{k_B T}\,\mathbf{z}_{\text{rot}}$ via 3-step backward substitution.
- Replace the $6 \times 6$ `jnp.linalg.solve(g_reg, rhs)` in `_project_one_water_momentum_rigid` (`L226-254`) with decoupled COM velocity $\mathbf{v}_{\text{COM}} = \frac{\sum_i \mathbf{p}_i}{\max(\sum_i m_i, 10^{-30})}$ and $3 \times 3$ Cramer solve $\boldsymbol{\omega} = \text{\_solve\_3x3}(I_{\text{reg}}, \sum_i \mathbf{r}_{\text{rel}, i} \times \mathbf{p}_i)$.

---

## 3. Verification & Cluster Execution Strategy
1. **Local & `titanix` Verification (`interpret=True` & Pure-JAX Parity)**:
   - Run fast parity and gradient checks (`jax.test_util.check_grads` / finite differences, `interpret=True` vs pure-JAX, `jax.vmap` batching, and `export_energy_fn` / `export_langevin_step` StableHLO export) locally and on `titanix` (via `myxcel` when offloading compute-intensive non-Triton suites).
2. **`engaging` GPU Triton Verification (via `myxcel` MCP Tools Only — No Raw SSH)**:
   - Sync code and submit GPU Triton verification and benchmark jobs to `engaging` (`gpu` L40S or `gpu-h200` / `pi-so3-gpu` presets) exclusively through `myxcel_myxcel` MCP tools (`push_project`, `rsync_wait`, `submit_job`, `job_status`, `job_wait`, `tail_job_log`, `pull_project`).
