---
title: 'B1 connect-existing-engines scope: NL+PME and analytical/flash forces into EnsemblePlan'
description: Phase 5 scoping of debt 760 (NL+PME) and 761 (analytical/flash forces) connection work for EnsemblePlan, with concrete technical designs and recommended sequencing
status: accepted
task_id: 260715_b1_physics_parity
date: '260717'
supersedes: ''
backlog_ids: ''
---
# B1 connect-existing-engines scope: NL+PME and analytical/flash forces into EnsemblePlan

## Context

Phase 4 of the B1 physics-parity investigation (task `260715_b1_physics_parity`) redirected mid-flight: the original water/PME step-scaling investigation found 1AKE (vacuum/GB), not water, dominates B1's aggregate per-step cost — orthogonal to prolix's actual thesis (heterogeneous batching + portability, not raw throughput). The user redirected toward testing that thesis directly for solvated (PME+SETTLE) systems, which was empirically **confirmed**: `EnsemblePlan` shares one compiled program across two genuinely different real solvated proteins (1VII+2GB1, `compile_fixed_s` 3.258s/3.356s/combined 3.285s, R²>0.9999) once their shape buckets align. Memory: `project_b1_heterogeneous_batching_confirmed_260717`.

That work also surfaced an incorrect claim: periodic+PME+neighbor-list support and analytical/custom_vjp forces were initially claimed to need building from scratch for `EnsemblePlan` (debt 758). The user caught this ("i remember intensively developing that") — both already exist in a separate `Protein`/`PhysicsSystem`/`batched_simulate` type-system track, never wired to `EnsemblePlan`. Debt 758 was resolved/corrected and replaced by 760 (NL+PME) and 761 (analytical forces). This document is the scoping deliverable for both, produced via two parallel read-only research passes over the relevant source files, per explicit user instruction to scope the connection work "carefully and thoroughly" before resuming profiling.

## Debt 760 — connect `simulate.py`'s existing NL+PME engine

**File correction**: the engine is `src/prolix/simulate.py` (top-level, 1228 lines) — a different, older, 509-line minimization+RATTLE runner lives at `src/prolix/physics/simulate.py`; don't conflate them.

### Findings

- jax_md's `NeighborList` (`Dense` format) is a plain pytree with a **fixed static capacity** baked in at `.allocate()`. `.update()` is a pure, jittable, vmap-safe positions→NeighborList function. `run_simulation`'s own docstring (`simulate.py:238-243`) documents `jax.vmap(run_simulation, ...)` as supported usage — direct prior art for vmapping over replicas exactly as `EnsemblePlan` does.
- `make_energy_fn` (`src/prolix/physics/system.py:29-262`) already has a working NL-consuming branch (`system.py:175-218`, dispatching to `chunked_lj_energy_nl`/`chunked_coulomb_energy_nl`).
- `dispatch_n_steps_inference` (`src/prolix/api/ensemble_dispatch.py:162-226`, `lax.while_loop` at line 218) has a fully generic `(step_i, state)` carry — no changes needed there for a bigger carry.
- `EnsemblePlan._run_stacked_dispatch` (`src/prolix/api/ensemble_plan.py:349-433`) wraps one `@jax.jit` around `vmap(replicas) × dispatch_n_steps_inference`; `_setup_integrator` (`ensemble_plan.py:463-505`) builds `settle_langevin`'s `init_fn`/`apply_fn` from `energy_fn_from_bundle(bundle)`; `step_fn` (`ensemble_plan.py:610-611`) is the carry-extension point.
- `energy_fn_from_bundle`/`single_padded_energy` (`bundle_md.py:446`, `batched_energy.py:497`) has no `neighbor` parameter today — confirms it's unconditionally dense.
- **What does NOT port**: `simulate.py`'s neighbor-list overflow recovery (`simulate.py:1034-1039`) is a host-Python branch *between* two separate compiled `lax.scan` calls (`simulate.py:1015-1046`) — incompatible with a single-compiled `lax.while_loop`, which has no host round-trip mid-loop.
- **Confirmed still-present, independently relevant bug**: `cutoff_distance` is a named parameter (`system.py:32`) that never reaches `**kwargs`, so the internal tapering read at `system.py:125` (`kwargs.get("cutoff_distance", kwargs.get("cutoff", 9.0))`) silently stays at 9.0 regardless of what's passed. `simulate.py`'s own call (`simulate.py:498-512`) is affected — the neighbor list itself is sized correctly, but the energy function's cutoff-tapering value is not.

### Recommended design

1. Allocate the neighbor list once at bundle-construction time (host-side, generous static capacity via `neighbor_fn.allocate()`).
2. Thread the resulting `NeighborList` as an explicit field of the while_loop's carry `state`.
3. Add an optional `neighbor=` kwarg to `single_padded_energy`/`energy_fn_from_bundle` that dispatches to `chunked_lj_energy_nl`/`chunked_coulomb_energy_nl` — mirrors `system.py:175-218` exactly, no new physics.
4. Wrap `_setup_integrator`'s state in `{langevin_state, neighbor, overflow_flag}`; `step_fn` unpacks, calls `apply_fn(..., neighbor=nbrs, ...)`, periodically (every K steps, mirroring `simulate.py:924-936`'s convention) calls `nbrs.update()`, OR-accumulates `did_buffer_overflow` into the flag instead of branching on it, repacks.
5. After `dispatch_n_steps_inference` returns (a host sync point that already exists), check the accumulated flag once on host — assert/raise on overflow, do not attempt a runtime resize.
6. Fix the `cutoff_distance` kwarg-absorption bug simultaneously — it becomes load-bearing once NL cutoff must physically match tapering.

**No changes needed to `dispatch_n_steps_inference` itself.**

### Open design question (decide before implementation starts)

Neighbor-list capacity depends on local atom density — a variable orthogonal to the atom-count bucket that made 1VII/2GB1 bucket-match this session. Two options:

- **(a) Shared conservative capacity + overflow-check** — simpler, preserves the just-confirmed compile-sharing property, costs some wasted compute. **Recommended default.**
- **(b) New bucket axis keyed on required capacity** — more precise, but risks fragmenting the compile-sharing this session just spent real effort confirming.

Default to (a) unless evidence emerges that it's too lossy in practice — don't undercut the confirmed heterogeneous-batching thesis without cause.

**Effort: 3-5 days. Biggest risk: the capacity-vs-bucket decision above.**

## Debt 761 — connect `single_padded_force`/`flash_explicit_forces`

### Correction to the debt's original premise

`flash_explicit_forces` is **not** closed-form analytical — it computes forces via `eqx.filter_grad` (still autodiff, just static-field-safe, the same fix class as this session's tracer-leak bug). True closed-form analytical gradients (`src/prolix/physics/analytical_forces.py`) exist only in `single_padded_force`'s `use_flash=False` **legacy** path, which needs the same dense (N,N) exclusion matrices as today's `single_padded_energy` — no memory win there, and it's not what "flash" defaults to. The PME custom_vjp benefit (`make_spme_energy_fn`) is **already present** in today's `single_padded_energy`+`jax.grad` path — it is not unique to flash.

What flash *actually* changes: `chunked_explicit_nonbonded_energy` (`src/prolix/physics/flash_explicit.py:51-83`) tiles the direct-space LJ+Coulomb pass into 256×256 blocks under `jax.checkpoint` — a genuine memory-locality optimization, but **an unmeasured hypothesis, not a proven win**.

### Findings

- **Wiring mechanics are small and low-risk**: `settle_langevin` (`src/prolix/physics/settle.py:1097`) already auto-detects energy-vs-force via `jax.eval_shape`-probing (`canonicalize_force`/`make_force_fn_like_canonicalize`) — the same pattern already used for this session's `pme_alpha`/`box_size` vmap-safe fallbacks. **No changes needed in `settle.py`.** Just add `force_fn_from_bundle(bundle)` mirroring `energy_fn_from_bundle` (`bundle_md.py:446-475`), and swap the call site in `_setup_integrator` (`ensemble_plan.py:463`).
- **Term coverage**: no missing nonbonded term found for the PME+SETTLE explicit-solvent case (LJ, damped direct-space Coulomb, PME reciprocal, PME exclusion correction, LJ dispersion tail all present in `flash_explicit.py:459-491`).
- **New risk found, filed as debt 763, independent of 761's fate**: `chunked_explicit_nonbonded_energy` computes `n_tiles = N // T` (integer division, `flash_explicit.py:83`, `T=256`) — if a bundle's padded atom count is not an exact multiple of 256, remainder atoms are silently dropped from all nonbonded interactions. Same bug class as the already-documented `optimization.py::inner_tile_size` tiling bug in project `CLAUDE.md`.
- **Test coverage is essentially nonexistent**: `tests/test_analytical_forces.py`'s only reference to `single_padded_force` is an importability check (`assert callable(single_padded_force)`); no numeric test exercises `use_flash=True`/`explicit_solvent=True`, let alone a bundle-derived `PhysicsSystem`. A parity test (flash forces vs. the already-validated `single_padded_energy`+`jax.grad`, on the existing 1VII/2GB1 bundles) is required, not optional, before trusting this in production — budget for finding at least one more real bug here given this session's 4-for-4 track record of verify-before-trust catching real issues.

### Recommendation: measure before committing

Run a cheap, bathos-tracked benchmark (`scripts/experiments/profile_b1_flash_vs_autodiff_forces.py`, campaign `32d6574e`) comparing `single_padded_energy`+`jax.grad` vs. `single_padded_force(use_flash=True)` wall-clock on the existing 1VII/2GB1 bundles *before* deciding whether to wire this into `EnsemblePlan` at all. If the benchmark shows no material win, close 761 as measured-not-worth-pursuing and keep only debt 763 (tile-size bug) open independently.

**Effort: wiring itself ~1-2 hours; full scope (parity test + tile-size check/fix) 1-2 days if the benchmark warrants it.**

### Post-scoping runtime finding (2026-07-17): debt 761 is BLOCKED, not just unmeasured

Running the benchmark (L2 CPU, then a bathos-tracked run `cb191665-a6c4-4552-95d8-892f1956f7b4`, outcome `fail`) surfaced a decisive result *before* any speed number was even reachable: `single_padded_force(sys, disp_fn, implicit_solvent=False, explicit_solvent=True, use_flash=True)` **crashes** on the real 1VII bundle-derived `PhysicsSystem` — both under `jax.jit` (`ConcretizationTypeError` from `flash_explicit.py:439`'s `float(mean_l)`) and unjitted/eager (`IndexError: tuple index out of range` at `flash_explicit.py:114`, `jax.lax.dynamic_slice(sys.excl_scales_vdw, (start_idx, 0), (T, sys.excl_scales_vdw.shape[1]))` — `.shape[1]` on a 1D array).

**Root cause (filed as debt 765, P1 — supersedes this section's earlier "measure benefit" framing as the primary blocker)**: `chunked_explicit_nonbonded_energy`/`_compute_tile_inner` (`flash_explicit.py:89-123`) expects `sys.excl_indices`/`excl_scales_vdw`/`excl_scales_elec` in a **per-atom-row** layout — shape `(N, max_excl_per_atom)`, sliced per-tile via `dynamic_slice`. `physics_system_from_bundle` (`bundle_md.py:440-442`) instead populates these fields in the **pair-list** layout — `excl_indices` shape `(n_pairs, 2)`, `excl_scales_vdw`/`excl_scales_elec` shape `(n_pairs,)` — the same layout `single_padded_energy`'s dense path (`_build_dense_exclusion_scales`) and the PME exclusion correction (`_pme_exclusion_correction_from_pairs`) both correctly consume. This is a genuine, structural incompatibility between two different exclusion-array conventions in the codebase, not a small bug — `flash_explicit_forces` was evidently built and tested against a different, older `PhysicsSystem` construction path that populated the per-atom-row layout, and was never reconciled with the bundle-derived path.

**Consequence**: debt 761 cannot proceed to wiring, or even to a completed speed measurement, until debt 765 is fixed (either build a per-atom-row exclusion array from the bundle's pair-list at construction time — a new `max_excl_per_atom` bucket-relevant constant — or rewrite `chunked_explicit_nonbonded_energy`'s exclusion handling to consume the pair-list format directly). This raises 761's real effort beyond the "1-2 days" estimated during read-only scoping — that estimate did not (and, being code-reading rather than execution, could not) catch this runtime-only shape mismatch. Baseline context captured regardless: on CPU, 1VII (1963 real / 5000 padded atoms), `single_padded_energy` forward-only = 285ms, +`jax.grad` (current EnsemblePlan path) = 1259ms.

**Updated recommendation**: do not invest further time in 761 (wiring or re-benchmarking) until debt 765 is resolved as its own piece of work. Debt 763 (tile-size-256-divisibility) remains a separate, independent, still-unverified risk on top of this — both must be addressed before flash forces can be trusted on bundle-derived systems.

## Recommended sequencing (updated post-runtime-finding)

1. ~~Build+run the flash-vs-autodiff benchmark~~ — done. Result: debt 761 is blocked by debt 765 (exclusion-layout mismatch), not just unmeasured — see the runtime-finding section above.
2. File debt for the tile-size divisibility bug — done, debt 763. File debt for the exclusion-layout mismatch — done, debt 765 (P1, primary blocker for 761).
3. Decide the NL-capacity-vs-bucket design for 760 (recommend option (a)).
4. **Implement 760 (3-5 days) — proceed first.** The connection work with the clearer, larger expected payoff (dense O(N²) → O(N·K) is a first-order algorithmic win at ~2000-atom solvated-protein scale), no unresolved blocking unknowns, and no dependency on 761/763/765.
5. **761 is now deprioritized behind 760.** Before any further 761 work: fix debt 765 (exclusion-layout reconciliation) as its own scoped task, then debt 763 (tile-size), then re-run this benchmark to see if there's actually a speed win worth wiring in.
6. After 760 lands: re-run the 1VII/2GB1 compile-sharing test to confirm no regression, then resume the deferred profiling backlog (tasks #26/#27/#28/#30/#32, debts 750/756).
