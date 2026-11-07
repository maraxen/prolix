# Spec: End-to-End `EnsemblePlan` Pallas v2 Megakernel Plumbing, Mixed-Precision Boundary, Dynamic NL Exclusion Tagging, and Native OpenMM `CUDA` / Solvated DHFR Benchmarks

**Date**: 2026-10-29
**Status**: Approved (0 P0/P1 defects remaining after Challenger/Defender review)
**Branch**: `feat/pallas-v2-megakernel-o1-o5-261028` (PR #42)

---

## 1. Problem Statement & Architectural Gaps

While PR #42 delivered the single-launch bonded megakernel (`pallas_bonded_all_fused`, `O2.2`) and optimizations `O1`–`O5` (`encode_nl_exclusions`, `excl_mode="pre_encoded"`, fused PME exclusion `erf`, primary-cell pre-wrapping, active-AD parameter gradient gating), four architectural gaps prevent end-to-end production trajectories (`EnsemblePlan.run()`) and the 23,558-atom solvated DHFR benchmark (`scripts/experiments/bench_dhfr_parity.py`) from realizing Pallas v2 speedups against native OpenMM `CUDA`:

1. **G1 (`EnsemblePlan` -> `bundle_md` Plumbing Gap)**:
   - `EnsemblePlan.__init__`, `from_bundle`, `from_bundles`, and `run()` accept `use_pallas: bool | None = None`, and `self.use_pallas` is stored on `self`, **but `EnsemblePlan.run()` never reads `use_pallas` or `self.use_pallas`**, dropping it before `_run_inference` (`ensemble_plan.py:364`) and `_run_trajectory` (`ensemble_plan.py:380`).
   - `use_pallas_pme: bool | None = None` is absent from `EnsemblePlan.__init__`, `from_bundle`, `from_bundles`, and `run()`.
   - `_setup_integrator` (`ensemble_plan.py:853-868`) and `_run_single_inference` (`ensemble_plan.py:1244-1267`) call `energy_fn_from_bundle` and `force_fn_from_bundle` without `use_pallas` or `use_pallas_pme`.
   - `force_fn_from_bundle` (`bundle_md.py:854-888`) accepts `use_pallas` but not `use_pallas_pme`, and `_bonded_energy_core` (`bundle_md.py:273-355`), `single_padded_energy_nl`, and `single_padded_energy_nl_cvjp` (`batched_energy.py`) still dispatch 6–7 separate v1 Pallas bonded kernels instead of `pallas_bonded_all_fused`.

2. **G2 (Mixed-Precision Boundary: `float64` BAOAB/SETTLE + `float32` Pallas Kernels)**:
   - Solvated explicit-solvent MD trajectories with rigid water (e.g., `bench_dhfr_parity.py`) run with `jax_enable_x64=True` and `float64` `positions`, `velocities`, `masses`, and `box` so BAOAB (`settle_langevin`) and analytical SETTLE constraints maintain `< 1e-6` bond-length holonomic accuracy at `dt = 1.0–2.0 fs`.
   - However, when `positions.dtype == jnp.float64` is passed into Pallas kernels:
     - `pallas_bonded_all_fused` and `pallas_nonbonded_nl` currently set internal kernel `dtype = positions.dtype` (`float64`), doubling register pressure, halving shared-memory capacity, and hitting FP64 execution units (32x–64x slower than FP32 on NVIDIA GPUs).
     - `pme_pallas.py` (`_spread_charges_pallas_single` and `_gather_pallas_single`) hardcodes `out_shape = ShapeDtypeStruct(..., jnp.float32)` while `_spread_charges_reference` and `_gather_reference` return `positions.dtype` (`float64`), causing `xtrax.op._check_signature` to crash with `IntentError` when `positions` is `float64`!
     - `pme.py` (`_spme_fwd` / `_spme_cached_fwd`) runs `jnp.fft.rfftn` / `irfftn` in `complex128` (`float64`) when `Q` is `float64` instead of running PME spread + 3D FFT + PME gather in `float32` like OpenMM `CUDA` `"mixed"` precision.

3. **G3 (Dynamic `encode_nl_exclusions` on Skin Rebuilds & `excl_mode="pre_encoded"`)**:
   - `single_padded_energy` (`batched_energy.py:838,879`) currently calls `pallas_nonbonded_nl` with default `excl_mode="table"`, paying per-tile exclusion-range checks instead of `O1.1` 2-bit tag decoding (`excl_mode="pre_encoded"`).
   - `EnsemblePlan` allocates `nbr0` (`_run_single`, `_run_single_inference`, `_build_stacked_neighbor_seed`, `_reallocate_stacked_neighbor_seed`) and rebuilds the neighbor list inside `_nl_step_fn` (`jax.lax.cond(should_update, ...)`) without calling `encode_nl_exclusions`.
   - Additionally, in `_nl_step_fn` (`ensemble_plan.py:1151-1156`), `n.update(pos)` calls `jax_md`'s raw `fns.update` on unwrapped `pos` without `make_neighbor_list_fn`'s `_wrap(pos, box)` into `[0, box)`, and if `by_stride` triggers `should_update` before `verlet` exceeds `dr_threshold / 2`, `jax_md`'s internal displacement check may skip rebuilding while `new_last` is reset to `pos`.

4. **G4 (Native OpenMM `CUDA` & Solvated DHFR Benchmark Harness)**:
   - Prior synthetic OpenMM comparisons (`261028_prolix_vs_openmm.json`) fell back to `OpenCL` because the Prolix `.venv` pip wheel lacked `libOpenMMCUDA.so`.
   - Conda-forge OpenMM 8.4.0 with native `libOpenMMCUDA.so` is installed on Engaging at `/orcd/pool/007/maarxaru/local/share/mamba/envs/b1-openmm` (`OPENMM_PLUGIN_DIR=/orcd/pool/007/maarxaru/local/share/mamba/envs/b1-openmm/lib/plugins` + `module load cuda/12.9.1`).
   - Both `scripts/benchmarks/pallas_vs_baseline.py` (`--kernel openmm_comparison`) and `scripts/experiments/bench_dhfr_parity.py` (23,558-atom solvated DHFR JAC benchmark) must execute OpenMM on native `CUDA` (`Precision="single"` and `"mixed"`) and compare against Prolix `XLA`, `Pallas v1`, and `Pallas v2` on the same Engaging B300 GPU node.

---

## 2. Detailed Design (G1–G4)

### 2.1 G1: `EnsemblePlan` -> `bundle_md` -> `batched_energy` Plumbing

1. **`EnsemblePlan` API (`src/prolix/api/ensemble_plan.py`)**:
   - Add `use_pallas_pme: bool | None = None` to `EnsemblePlan.__init__`, `EnsemblePlan.from_bundle`, `EnsemblePlan.from_bundles`, and `EnsemblePlan.run`, storing `self.use_pallas_pme = use_pallas_pme`.
   - In `EnsemblePlan.run(..., use_pallas: bool | None = None, use_pallas_pme: bool | None = None)`:
     Resolve:
     ```python
     eff_use_pallas = use_pallas if use_pallas is not None else self.use_pallas
     eff_use_pallas_pme = (
         use_pallas_pme if use_pallas_pme is not None else self.use_pallas_pme
     )
     ```
   - Thread `use_pallas=eff_use_pallas` and `use_pallas_pme=eff_use_pallas_pme` through:
     - `_run_inference` -> `_run_single_inference`
     - `_run_trajectory` -> `_run_grouped` -> `_run_stacked_dispatch` & `_run_single`
     - `_setup_integrator(..., use_pallas=use_pallas, use_pallas_pme=use_pallas_pme, excl_mode=excl_mode)`
   - Inside `_setup_integrator`:
     Pass `use_pallas=use_pallas, use_pallas_pme=use_pallas_pme, excl_mode=excl_mode` to `force_fn_from_bundle` and `energy_fn_from_bundle`.
   - Inside `_run_single_inference`:
     Pass `use_pallas=use_pallas, use_pallas_pme=use_pallas_pme, excl_mode=excl_mode` to `energy_fn_from_bundle` and `force_fn_from_bundle`.

2. **`bundle_md` API (`src/prolix/api/bundle_md.py`)**:
   - Update `energy_fn_from_bundle` and `force_fn_from_bundle` signatures to accept `use_pallas: bool | None = None`, `use_pallas_pme: bool | None = None`, and `excl_mode: str = "table"`.
   - Note that for explicit solvent (`not has_implicit_solvent`) or periodic systems, `force_fn_from_bundle` already delegates to `-jax.grad(energy_fn)(positions)` (which calls `single_padded_energy` with `neighbor`, `use_pallas`, `use_pallas_pme`, `excl_mode`). For non-PBC implicit solvent without neighbor list, `single_padded_force` receives `use_pallas=use_pallas`.
   - Update `_bonded_energy_core` (`bundle_md.py:273-355`) so `bonded_energy_fn_from_bundle(..., use_pallas=True)` uses `pallas_bonded_all_fused` (with fallback to individual kernels only if `pallas_bonded_all_fused is None`).

### 2.2 G2: OpenMM-Style Mixed-Precision Boundary (`float64` Integrator + `float32` Pallas Kernels)

1. **Contract**:
   - Callers (including `EnsemblePlan` under `jax_enable_x64=True` and `bench_dhfr_parity.py`) may pass `positions` (and `charges`, `sigmas`, `epsilons`, `box`, bonded params) in either `jnp.float32` or `jnp.float64`.
   - Every Pallas kernel (`pallas_bonded_all_fused`, `pallas_bonds_2body`, `pallas_angles_harmonic`, `pallas_torsions_4body`, `pallas_cmap_torsions`, `pallas_exceptions_14`, `pallas_pme_exclusion_pairs`, `pallas_nonbonded_nl`, `pallas_flash_explicit_nonbonded`, `spread_charges_pallas`, `gather_pallas`):
     - Records `orig_dtype = positions.dtype`.
     - Executes the internal `pl.pallas_call` in `jnp.float32` (casting floating-point kernel inputs to `jnp.float32`).
     - Casts the returned `energy` and `forces` (and `mesh` / `potentials` in PME spread/gather) back to `orig_dtype` (`.astype(orig_dtype)`).
   - This guarantees:
     - **Zero `xtrax.op._check_signature` `IntentError` (`MK-05`)**: Both the Pallas kernel and the plain-JAX `reference` return outputs of `positions.dtype` (`float32` when `positions` is `float32`, `float64` when `positions` is `float64`).
     - **Full FP32 GPU throughput & 1x register footprint**: All Triton / Pallas kernels and `plt.atomic_add` operations execute on `float32` buffers/registers even when `jax_enable_x64=True`.
     - **Full FP64 integrator & SETTLE accuracy**: `_bwd` returns `grad_pos` in `float64` (`(-g * forces).astype(orig_dtype)`), so `force_fn(pos)` returns `float64` forces `(N, 3)` into BAOAB (`settle_langevin`) and SETTLE (`settle_positions` / `settle_velocities`).
   - **SPME Mixed Precision (`src/prolix/physics/pme.py` & `src/prolix/physics/pme_pallas.py`)**:
     - In `_spread_charges_pallas_single` (`pme_pallas.py:218`): return `mesh_flat.reshape((Kx, Ky, Kz)).astype(positions.dtype)`.
     - In `_gather_pallas_single` (`pme_pallas.py:359`): return `forces.astype(positions.dtype), pot_pad[:N].astype(positions.dtype)`.
     - In `spme_energy_with_forces` and `_spme_energy_with_forces_cached` (`pme.py`):
       When `use_pallas_pme=True`, cast `positions, charges, box_size, G_cached` to `jnp.float32` inside `_spme_fwd` / `_spme_cached_fwd` / `_spme_bwd` so charge spreading, 3D `rfftn`/`irfftn` (`complex64`), and force gathering all execute in `float32`, and cast the returned scalar energy and backward cotangents (`dE_dpos`, `dE_dq`, `dE_dbox`) back to `positions.dtype`, `charges.dtype`, and `box_size.dtype`.

### 2.3 G3: Dynamic `encode_nl_exclusions` on Skin Rebuilds & `excl_mode="pre_encoded"`

1. **Helper `_encode_bundle_nl(nbr, bundle)` in `src/prolix/api/ensemble_plan.py`**:
   ```python
   def _encode_bundle_nl(nbr: Any, bundle: MolecularBundle) -> Any:
       if nbr is None or bundle.excl_dense_indices is None:
           return nbr
       encoded_idx = encode_nl_exclusions(
           nbr.idx,
           bundle.excl_dense_indices,
           bundle.excl_dense_scales_vdw,
           bundle.excl_dense_scales_elec,
           n_atoms=int(bundle.positions.shape[0]),
       )
       return dataclasses.replace(nbr, idx=encoded_idx)
   ```
   - Because `encode_nl_exclusions` strips any existing upper tag bits (`j_clean = nbr_i32 & _NB_INDEX_MASK`) before re-tagging bit 30 (`_NB_EXCL_FULL_BIT`) and bit 29 (`_NB_EXCL_14_BIT`), `_encode_bundle_nl` is **idempotent**, preserves shape `(N, K)` and dtype `int32`, and is traceable under `jax.jit`, `jax.vmap`, and `jax.lax.cond`.
   - For batched/stacked bundles (`_build_stacked_neighbor_seed` and `_reallocate_stacked_neighbor_seed`), `_encode_bundle_nl` is applied per-bundle at seed allocation/reallocation time and via `jax.vmap` when `stacked_bundle.excl_dense_indices` has a leading batch axis.

2. **Wrapping & Force-Rebuild Invariant in `make_neighbor_list_fn` and `_nl_step_fn`**:
   - In `make_neighbor_list_fn` (`src/prolix/physics/neighbor_list.py:232-242`):
     Ensure `allocate` attaches the wrapping `update` function (`dataclasses.replace(nbr, update_fn=update)`) so `n.update(pos)` always wraps `pos` into `[0, box)` before binning into the cell list.
   - In `_nl_step_fn` (`src/prolix/api/ensemble_plan.py:1143-1158`):
     When `should_update` (`verlet | by_stride`) is `True`:
     - Force `n.update(pos)` to rebuild the candidate list by invalidating `n.reference_position` when `by_stride` triggers (`n_in = dataclasses.replace(n, reference_position=n.reference_position + jnp.asarray(1e6, dtype=n.reference_position.dtype))`), call `n_upd = nl_fn.update(pos, n_in)`, and apply `_encode_bundle_nl(n_upd, bundle)`.
     - When `should_update` is `False`, the false branch `lambda n: n` returns `neighbor` unchanged (preserving the pre-encoded `.idx` with 0 ops).
     - Both branches of `jax.lax.cond(should_update, ...)` return identical PyTree structure, shapes `(N, K)`, and `int32` dtype.

3. **`excl_mode="pre_encoded"` in `single_padded_energy`**:
   - Add `excl_mode: str = "table"` parameter to `single_padded_energy` (`batched_energy.py:644`), `energy_fn_from_bundle`, and `force_fn_from_bundle`, and forward `excl_mode=excl_mode` into `pallas_nonbonded_nl(..., excl_mode=excl_mode)`.
   - In `EnsemblePlan` (`_run_single`, `_run_single_inference`, `_run_stacked_dispatch`), whenever `bundle.excl_dense_indices is not None`, `EnsemblePlan` pre-encodes `nbr` via `_encode_bundle_nl` and passes `excl_mode="pre_encoded"` to `_setup_integrator` / `energy_fn_from_bundle` / `force_fn_from_bundle`.
   - Direct callers of `single_padded_energy` that do not pre-encode `neighbor.idx` default to `excl_mode="table"`, which also honors any pre-encoded tag bits (`has_tag = is_full_excl | is_14_excl` at `pallas_kernels.py:1121`) while falling back to the `[excl_min, excl_max]` interval table for untagged indices.

### 2.4 G4: Native OpenMM `CUDA` & Solvated DHFR Benchmark Harness

1. **Synthetic Scaling against Native OpenMM `CUDA` (`scripts/benchmarks/pallas_vs_baseline.py`)**:
   - Add `--openmm-python` CLI option (defaulting to `os.environ.get("OPENMM_PYTHON", "")`, e.g., `/orcd/pool/007/maarxaru/local/share/mamba/envs/b1-openmm/bin/python`) and `--openmm-precision` (`"single"` or `"mixed"`, default `"mixed"`).
   - When `--openmm-python` is provided (or when current Python lacks `CUDA` platform and `/orcd/pool/007/maarxaru/local/share/mamba/envs/b1-openmm/bin/python` exists), `bench_openmm_comparison` invokes a self-contained helper subprocess in `--openmm-python` with `OPENMM_PLUGIN_DIR` set to measure OpenMM `CUDA` (`Precision="single"` and `"mixed"`) on the exact same serialized system topology & coordinates, while the main process measures Prolix `XLA`, `Pallas v1`, and `Pallas v2` on the same GPU.

2. **23,558-Atom Solvated DHFR JAC Benchmark (`scripts/experiments/bench_dhfr_parity.py` & `scripts/slurm/bench_dhfr_parity_split.slurm`)**:
   - Add `--prolix-backend` choice `("xla", "pallas_v2", "both")` (default `"both"`) and `--openmm-precision` `("single", "mixed", "both")` (default `"mixed"`) to `scripts/experiments/bench_dhfr_parity.py`.
   - In `run_prolix_phase` / `run_combined`:
     - Run Prolix on the 23,558-atom solvated DHFR `MolecularBundle` (`use_size_buckets=False`, `nl_skin=1.5`, `dt=1.0 fs`) for both `use_pallas=False` (`XLA`) and `use_pallas=True` (`Pallas v2`: `pallas_bonded_all_fused` + `pallas_nonbonded_nl(excl_mode="pre_encoded")` + `pme_pallas` in mixed precision) across step counts $N_{\text{steps}} \in \{100, 500, 2000, 5000\}$.
     - Separate JIT compile time from steady-state compute throughput (`ns/day` at 1 fs and 2 fs) via linear regression (`scipy.stats.linregress(step_counts, wall_times)`) AND direct warm second-call timing at each $N_{\text{steps}}$.
     - Verify single-point energy & force parity (`Pallas v2` vs `XLA` and vs OpenMM `CUDA`) and multi-step ensemble observable parity (temperature, kinetic energy, potential energy, water geometry).

---

## 3. Adversarial Spec Review (Challenger & Defender Loop)

### Round 1 — Spec Challenger Findings
- **[P0-1] AMBER 1-4 Exceptions vs `encode_nl_exclusions` (`_NB_EXCL_14_BIT`)**:
  - *Challenge*: In `ExclusionSpec.from_protein` (`neighbor_list.py:109-133`), AMBER proteins with explicit `exception_14_params` place 1-4 pairs into `exception_pairs` (evaluated by `pallas_bonded_all_fused`) and into `idx_12_13` with `scale_vdw=0.0, scale_elec=0.0`. If `encode_nl_exclusions` tags them with `_NB_EXCL_FULL_BIT`, will `pallas_nonbonded_nl(..., excl_mode="pre_encoded", include_pme_excl_erf=True)` still compute the required PME reciprocal exclusion correction $-\text{erf}(\alpha r)/r$ for those 1-4 pairs without double-counting 1-4 LJ/Coulomb?
  - *Defense / Resolution*: Verified in `pallas_kernels.py:1092, 1117-1118, 1200-1207`:
    When `is_full_excl` is `True` and `include_pme_excl_erf` is `True`, `valid_pair` remains `True`, `w_vdw = 0.0`, `w_elec = 0.0`, and `eff_w = w_elec - erf(alpha * r) = -erf(alpha * r)`. Thus `e_lj = 0.0` and `e_coul = -q_i q_j erf(alpha * r) / r`, while `pallas_bonded_all_fused` evaluates the 1-4 LJ and bare Coulomb $q_i q_j / (1.2 r)$. Total 1-4 electrostatic energy is $\frac{q_i q_j}{1.2 r} - \frac{q_i q_j \text{erf}(\alpha r)}{r}$, which is exact!

- **[P1-1] `jax_md` Internal Displacement Check When `by_stride` Triggers in `_nl_step_fn`**:
  - *Challenge*: In `_nl_step_fn` (`ensemble_plan.py:1151`), `should_update = verlet | by_stride`. If `by_stride` is `True` while `max_dr_sq <= thresh_sq`, `new_last` is reset to `pos`, but `jax_md`'s internal `fns.update` checks `|pos - n.reference_position| > dr_threshold / 2` and may skip rebuilding `n.idx`, leaving `n.reference_position` stale while `last_nl_pos` was reset!
  - *Defense / Resolution*: Fixed in Section 2.3.2: inside `_nl_step_fn`'s `should_update` true-branch, we invalidate `n.reference_position` (`n.reference_position + 1e6`) before calling `nl_fn.update(pos, n_in)` so that whenever `EnsemblePlan`'s `atom_mask`-aware `should_update` predicate fires, `jax_md` unconditionally rebuilds at `_wrap(pos, box)` and `_encode_bundle_nl` tags the fresh `idx`.

- **[P1-2] `test_ensemble_carry_scan.py` Source Inspection Guard**:
  - *Challenge*: `tests/api/test_ensemble_carry_scan.py:52` asserts `"jax.lax.scan" not in inspect.getsource(EnsemblePlan._run_single)`.
  - *Defense / Resolution*: Recorded as a hard constraint — `_run_single` must not contain the literal substring `"jax.lax.scan"` in code or comments.

- **[P1-3] Stacked Dispatch (`_run_stacked_dispatch`) Neighbor Encoding**:
  - *Challenge*: When multiple bundles in the same shape bucket run via `_run_stacked_dispatch` (`jax.vmap` over replicas), `stacked_bundle.excl_dense_indices` has shape `(B, N, 32)` while inside the `vmap`ped `_single_replica_body(b, nbr_seed)` each replica `b` has `b.excl_dense_indices` of shape `(N, 32)`.
  - *Defense / Resolution*: `_encode_bundle_nl(nbr, b)` inside `_nl_step_fn` operates per-replica on `(N, K)` and `(N, 32)` (which `jax.vmap` batches automatically), while `_build_stacked_neighbor_seed` and `_reallocate_stacked_neighbor_seed` call `_encode_bundle_nl(nbr_i, b)` on each individual `b` in `group` before stacking `stacked_idx`!

### Round 2 — Final Verdict
- **P0 Defects**: 0
- **P1 Defects**: 0
- **Status**: **APPROVED FOR PHASE 2 IMPLEMENTATION**.
