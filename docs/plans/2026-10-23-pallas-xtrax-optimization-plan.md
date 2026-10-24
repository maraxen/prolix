# Pallas Kernel Optimization & Post-Equinox `xtrax` Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate silent zero-force custom VJP bugs, replace SIMT-underutilized 1-atom-per-CTA Pallas kernels with vectorized `(32, 64)` / `(32, 32)` Pallas kernels and forward-force `@jax.custom_vjp` primitives, replace 10-iteration water RATTLE and 16 full-array copies in `settle.py` with 1-pass closed-form $3\times 3$ solvers and race-free additive delta scatters, and align `prolix` with `xtrax` v0.4.1b3's post-Equinox architecture (`ChunkedMap`, `@jax.tree_util.register_dataclass`, pure-array kernel boundaries, and `export_context()`).

**Architecture:** All Pallas and `@jax.custom_vjp` kernel boundaries take pure `jax.Array` positional inputs followed by hashable static primitives (`nondiff_argnums`), configured via a `@jax.tree_util.register_dataclass` `PallasKernelConfig`. `resolve_pallas_mode()` automatically selects pure-JAX HLO inside `export_context()` (for `jax.export` / IREE WASM), `interpret=True` on CPU/TPU (`titanix`), and Triton GPU compilation on `engaging`.

**Tech Stack:** JAX (`jax.experimental.pallas`, `jax.experimental.pallas.triton`, `jax.custom_vjp`, `jax.custom_batching.custom_vmap`, `jax.tree_util.register_dataclass`), `xtrax` (`ChunkedMap` / `SafeMap`), `pytest`, `myxcel` (`titanix` and `engaging`).

---

## Praxia Backlog DAG (`261023_pallas_xtrax_optimization`)

```
#6092 [T1: Post-Equinox xtrax ChunkedMap alignment, export_context, & pure-array kernel boundary]
  ├──► #6093 [T2: Fix custom_vjp zero-force bugs, GB chain rule, O(N^2) NL Coulomb leak, & 5-step binary-search exclusions]
  │      └──► #6096 [T5: Fused Pallas nonbonded kernels (O(N*K) NL, 2D-tiled Flash Explicit, & vectorized OBC-II Born radii)]
  ├──► #6094 [T3: Vectorized (32, 64) Pallas SPME charge spread & force gather + Green's function caching]
  └──► #6095 [T4: Closed-form 1-pass water RATTLE, race-free additive delta scatters, & 3x3 Cholesky in settle.py]
         │
         ▼ (all 6093, 6094, 6095, 6096 converge)
       #6097 [T6: Cluster verification (titanix/engaging via myxcel), persistent xtrax+Pallas skill record, code review & PR]
```

---

### Task 1 (Backlog #6092): Post-Equinox `xtrax` Alignment, `export_context()`, and Pure-Array Kernel Boundary

**Files:**
- Modify: `src/prolix/tiling/xtrax_adapter.py:15-25, 146-180`
- Modify: `src/prolix/tiling/axis_dispatch.py:10-20, 50-69`
- Modify: `src/prolix/api/ensemble_dispatch.py:20-35, 110-170`
- Modify: `src/prolix/export.py:1-110`
- Modify: `src/prolix/pallas_kernels.py:1-40`
- Modify: `src/prolix/physics/flash_explicit.py:15-25, 85-98, 405-415, 480-490, 570-595, 683-716`
- Test: `tests/physics/test_xtrax_pallas_boundary.py`

**Step 1: Write the failing tests in `tests/physics/test_xtrax_pallas_boundary.py`**
- Test `ChunkedMap` / `SafeMap` dispatch in `xtrax_adapter`, `axis_dispatch` (including non-multiple `n_waters % batch_size != 0` when `ChunkedMap` is active), and `ensemble_dispatch`.
- Test `PallasKernelConfig` as a `@jax.tree_util.register_dataclass` `@dataclass(frozen=True)` with `jax.tree_util.tree_leaves(cfg) == []`.
- Test `export_context()` in `src/prolix/export.py` ensuring `resolve_pallas_mode(use_pallas=True)` returns `(False, False)` inside `export_context()` and `(True, interpret)` outside.
- Test `flash_explicit_forces` runs without `@eqx.filter_grad` and without calling host-only `topology.find_bonded_exclusions` under `jax.jit` and `jax.vmap`.

**Step 2: Run test to verify failure**
- Run: `uv run pytest tests/physics/test_xtrax_pallas_boundary.py -v`

**Step 3: Implement R1 changes**
- Add `ChunkedMap` / `SafeMap` compatibility imports and `isinstance(..., (SafeMap, ChunkedMap))` across `xtrax_adapter.py`, `axis_dispatch.py`, and `ensemble_dispatch.py`.
- Add `_IN_EXPORT_CONTEXT` and `export_context()` to `src/prolix/export.py` and wrap `export_energy_fn` and `export_langevin_step`.
- Add `PallasKernelConfig` and `resolve_pallas_mode` to `src/prolix/pallas_kernels.py`.
- In `src/prolix/physics/flash_explicit.py`: fix `_pad_to_tile_multiple` (`excl_indices` padded with `-1`, `excl_scales_*` padded with `1.0`), replace `topology.find_bonded_exclusions` with precomputed `sys.excl_pair_indices` / `sys.excl_indices`, and replace `@eqx.filter_grad` over `PaddedSystem` with `jax.grad` wrt `positions` (arg 0).

**Step 4: Run tests to verify pass**
- Run: `uv run pytest tests/physics/test_xtrax_pallas_boundary.py tests/test_export.py -v`

---

### Task 2 (Backlog #6093): Fix Custom-VJP Zero-Force Bugs, GB Chain Rule, $O(N^2)$ NL Coulomb Leak, and 5-Step Binary-Search Exclusions

**Files:**
- Modify: `src/prolix/physics/optimization.py:76-127, 182-198, 254-301, 346-361`
- Modify: `src/prolix/pallas_kernels.py:106-306`
- Modify: `src/prolix/batched_energy.py:991-1011, 1072-1092`
- Modify: `src/prolix/physics/neighbor_list.py:396-462`
- Test: `tests/physics/test_custom_vjp_gradients_parity.py`

**Step 1: Write failing tests in `tests/physics/test_custom_vjp_gradients_parity.py`**
- Test `jax.grad(chunked_lj_energy_nl, argnums=(0, 1, 2, 4))` and `jax.grad(chunked_coulomb_energy_nl, argnums=(0, 1, 3))` against exact autodiff of un-custom-VJP'd reference LJ (including switching window $r_{\text{switch}} < r < r_{\text{cut}}$) and damped-erfc Coulomb: verify position gradients are non-zero and match autodiff to $<10^{-5}$ relative error!
- Test `jax.grad(chunked_lj_energy, argnums=(0, 1, 2, 4))` and `jax.grad(chunked_coulomb_energy, argnums=(0, 1, 3))` parameter gradients (`sigma`, `epsilon`, `charge`, `excl_scales`).
- Test `jax.grad(gb_coulomb_energy_dense, argnums=(0, 1, 2))` and `jax.grad(gb_energy_split_cutoff, argnums=(0, 1, 2))` against exact autodiff of `_gb_coulomb_energy_chunked` and full GB energy (catching both the factor-of-2 bug and the missing `dE/dB` chain-rule term).
- Test `get_neighbor_exclusion_scales` 5-step binary search parity against the 32-step reference across random and protein topologies.
- Test `single_padded_energy_nl_cvjp` and `single_padded_energy_nl` use $O(N \cdot K)$ neighbor-list Coulomb and match `jax.grad` parity.

**Step 2: Run test to verify failure**
- Run: `uv run pytest tests/physics/test_custom_vjp_gradients_parity.py -v`

**Step 3: Implement R2 & R3 fixes**
- Implement exact analytical `_chunked_lj_nl_bwd`, `_chunked_lj_bwd`, `_chunked_coulomb_nl_bwd`, and `_chunked_coulomb_bwd` in `src/prolix/physics/optimization.py`.
- Fix factor-of-2 and implement `grad_charges` and `grad_born_radii` in `_gb_coulomb_forces_chunked` and `_gb_coulomb_bwd` in `src/prolix/pallas_kernels.py`.
- Replace dense `_build_dense_exclusion_scales` + `_coulomb_energy_masked` in `single_padded_energy_nl_cvjp` and `single_padded_energy_nl` (`src/prolix/batched_energy.py`) with $O(N \cdot K)$ neighbor-list Coulomb and ghost-atom `nb_idx` masking.
- Replace the 32-iteration `lax.fori_loop` in `get_neighbor_exclusion_scales` (`src/prolix/physics/neighbor_list.py`) with the branchless $\lceil \log_2 M \rceil$-step binary search over `excl_sorted`.

**Step 4: Run tests to verify pass**
- Run: `uv run pytest tests/physics/test_custom_vjp_gradients_parity.py -v`

---

### Task 3 (Backlog #6094): Vectorized `(32, 64)` Pallas SPME Charge Spread & Force Gather + Green's Function Caching

**Files:**
- Modify: `src/prolix/physics/pme_pallas.py:1-209`
- Modify: `src/prolix/physics/pme.py:130-408`
- Modify: `src/prolix/api/ensemble_plan.py`
- Test: `tests/physics/test_pme_pallas_parity.py`

**Step 1: Expand tests in `tests/physics/test_pme_pallas_parity.py`**
- Test `spread_charges_pallas` and `gather_pallas` with non-power-of-2 atom counts (e.g. `N = 37`, `N = 5000`), masked ghost atoms, `jax.vmap` across a batch of molecules (`B = 3`), `spme_energy_with_forces` forward + backward `jax.grad` wrt `(positions, charges)`, and `make_spme_energy_fn` cached Green's function `G_cached` parity.

**Step 2: Run test to verify failure on `vmap` / new assertions**
- Run: `uv run pytest tests/physics/test_pme_pallas_parity.py -v`

**Step 3: Implement R4 changes**
- Rewrite `spread_charges_pallas` and `gather_pallas` in `src/prolix/physics/pme_pallas.py` to use `BLOCK_N = 32`, host padding to `N_pad = cdiv(N, 32) * 32`, 1D flattened mesh `(Kx * Ky * Kz,)`, 1D scalar-column coordinate loads/stores `(offs, 0..2)`, vectorized `(32, 64)` B-spline outer products reshaped to 1D `(2048,)` for `plt.atomic_add` and `pl.load`, `@jax.custom_batching.custom_vmap`, and `name="pme_charge_spread"` / `name="pme_bwd_gather"`.
- Deduplicate `influence_function` in `_spme_fwd` and cache `G_cached` in `make_spme_energy_fn` (`src/prolix/physics/pme.py`).
- Expose `use_pallas: bool | None = None` on `EnsemblePlan` (`src/prolix/api/ensemble_plan.py`).

**Step 4: Run tests to verify pass**
- Run: `uv run pytest tests/physics/test_pme_pallas_parity.py -v`

---

### Task 4 (Backlog #6095): Closed-Form 1-Pass Water RATTLE, Race-Free Additive Delta Scatters, and $3\times 3$ Cholesky in `settle.py`

**Files:**
- Modify: `src/prolix/physics/settle.py:206-285, 462-730, 849-995, 1225-1250, 2150-2305`
- Test: `tests/physics/test_settle_optimizations.py`

**Step 1: Write tests in `tests/physics/test_settle_optimizations.py`**
- Test `settle_velocities` 1-pass $3\times 3$ Gram matrix solver with padded water rows (`water_indices` containing `[0, 0, 0]` and `water_mask=False` while Atom 0 is a real water oxygen): verify holonomic velocity constraints $(\mathbf{v}_b - \mathbf{v}_a)\cdot\hat{\mathbf{n}}_{ab} = 0$ to $<10^{-6}$ (FP32) / $<10^{-12}$ (FP64), zero corruption of Atom 0, and zero `NaN`s in forward and `jax.grad`.
- Test `_cholesky_upper_solve_3x3(I_reg, rhs)` against `jnp.linalg.solve(jnp.linalg.cholesky(I_reg).T, rhs)` and verify `_ou_noise_one_water_rigid` rotational covariance $\mathbb{E}[\boldsymbol{\omega}\boldsymbol{\omega}^T] = k_B T I_{\text{reg}}^{-1}$.
- Test `_project_one_water_momentum_rigid` decoupled COM + $3\times 3$ Cramer solve against the $6\times 6$ linear solve to machine precision.

**Step 2: Run tests**
- Run: `uv run pytest tests/physics/test_settle_optimizations.py -v`

**Step 3: Implement R6 in `src/prolix/physics/settle.py`**
- Implement 1-pass $3\times 3$ Gram matrix RATTLE in `settle_velocities` with constant positive diagonal $(\mu_O+\mu_H, \mu_O+\mu_H, 2\mu_H)$ and `_solve_3x3_batch`.
- Replace all 8 `jnp.concatenate([arr, arr[:1]], axis=0)` scratch-row scatters (`L210, 279, 520, 568, 726, 865, 1244, 2299`) with race-free additive delta scatters `arr.at[idx].add(jnp.where(mask, new_val - arr[idx], 0.0))` (at overwrite sites) and masked `.add(jnp.where(water_mask[:, None], dv, 0.0))` (at additive sites), plus `_det == 0` guards in Cramer's rule (`L690, 840`) and zero-mass guards (`L240, 2182, 2189`).
- Implement `_cholesky_upper_solve_3x3` in `_ou_noise_one_water_rigid` and decoupled COM + $3\times 3$ Cramer solve in `_project_one_water_momentum_rigid`.

**Step 4: Run tests to verify pass**
- Run: `uv run pytest tests/physics/test_settle_optimizations.py tests/physics/test_settle*.py -v`

---

### Task 5 (Backlog #6096): Fused Pallas Nonbonded Kernels ($O(N \cdot K)$ NL, 2D-Tiled Flash Explicit, and Vectorized OBC-II Born Radii)

**Files:**
- Modify: `src/prolix/pallas_kernels.py`
- Modify: `src/prolix/physics/flash_explicit.py`
- Modify: `src/prolix/physics/system.py`
- Modify: `src/prolix/batched_energy.py`
- Test: `tests/physics/test_pallas_nonbonded_parity.py`

**Step 1: Write tests in `tests/physics/test_pallas_nonbonded_parity.py`**
- Test `pallas_nonbonded_nl` (`BLOCK_N=32, BLOCK_K=32`) energy and `jax.grad` forces + parameter cotangents (`sigma, epsilon, charges`) against pure-JAX reference for both standard Coulomb + LJ and PME direct-space damped-erfc + switched LJ, including `jax.vmap` over a batch of systems and non-multiple `N, K`.
- Test `pallas_flash_explicit_nonbonded` (`BLOCK_I=32, BLOCK_J=32`) energy and `jax.grad` forces against `chunked_explicit_nonbonded_energy` and `flash_explicit_forces`.
- Test vectorized `pallas_born_radii` (`BLOCK_N=32, BLOCK_K=32`) forward values and `jax.grad` (`custom_vjp`) against `compute_born_radii_neighbor_list`, plus end-to-end `gb_energy_split_cutoff` gradient parity.

**Step 2: Run test to verify failure**
- Run: `uv run pytest tests/physics/test_pallas_nonbonded_parity.py -v`

**Step 3: Implement R5 kernels**
- Implement `pallas_nonbonded_nl` (with `@jax.custom_vjp` and `@jax.custom_batching.custom_vmap`) in `src/prolix/pallas_kernels.py`.
- Implement `pallas_flash_explicit_nonbonded` (with forward-computed forces in `@jax.custom_vjp` and `@jax.custom_batching.custom_vmap`) in `src/prolix/pallas_kernels.py` and wire into `src/prolix/physics/flash_explicit.py` when `use_pallas` is enabled, plus fused analytical-force `lax.scan` fallback so `flash_explicit_forces` avoids 2x AD rematerialization even on CPU/non-Pallas backends.
- Rewrite `pallas_born_radii` in `src/prolix/pallas_kernels.py` using vectorized `(32, 32)` tiles inside `lax.fori_loop`, with `interpret` auto-resolution, `@jax.custom_batching.custom_vmap`, and analytical `@jax.custom_vjp`.

**Step 4: Run tests to verify pass**
- Run: `uv run pytest tests/physics/test_pallas_nonbonded_parity.py -v`

---

### Task 6 (Backlog #6097): Cluster Verification (`titanix` & `engaging` via `myxcel`), Persistent Skill/Lesson Record, Code Review & PR

**Files:**
- Create: `.praxia/docs/reference/261023_xtrax_composition_and_pallas_optimization_guide.md`
- Log: Praxia `lesson` and `knowledge` entries

**Step 1: Execute compute-intensive verification on `titanix` (interpret/XLA) and `engaging` (Triton GPU) via `myxcel`**
- Push branch to `titanix` / `engaging` via `myxcel_myxcel` (`push_project` with explicit `worktree` slug if in a worktree, or branch on checkout) and submit GPU verification (`preset="gpu"` or `"pi-so3-gpu"` on `engaging`) to verify real Triton compilation and numerical parity on GPU hardware.

**Step 2: Persist lessons learned and skill reference guide**
- Record lessons in Praxia (`lesson(action="add", ...)` and `knowledge(action="ingest_markdown", ...)`) and write `.praxia/docs/reference/261023_xtrax_composition_and_pallas_optimization_guide.md` covering all patterns for the updated `xtrax` composition (post-Equinox) and `pallas-optimization` skills.

**Step 3: Independent Subagent Code Review & Remediation**
- Dispatch `reviewer` subagent to audit the full branch diff against the spec, remediate any findings, re-verify tests, commit, and create the PR.
