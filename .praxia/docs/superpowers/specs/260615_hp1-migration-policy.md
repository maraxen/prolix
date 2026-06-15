---
name: hp1-migration-policy
description: HP1 legacy entry-points deprecation and replacement policy for prolix.api migration
metadata:
  type: project
---

# HP1 Migration Policy for Prolix Legacy Entry-Points

**File:** `.praxia/docs/superpowers/specs/260615_hp1-migration-policy.md`
**Date:** 2026-06-15
**Task:** 260615_sprint39 (#327)
**Status:** approved

---

## 1. Legacy Entry-Points Inventory

| Symbol | Current Location | Category |
|--------|-----------------|----------|
| `batched_produce` | `prolix.batched_simulate` | deprecated |
| `batched_equilibrate` | `prolix.batched_simulate` | deprecated (unsafe: zero-force NaN) |
| `LangevinState` (re-export) | `prolix.batched_simulate` | deprecated |
| `pad_protein` | `prolix.padding` / `prolix` top-level | deprecated |
| `collate_batch` | `prolix.padding` / `prolix` top-level | deprecated (fast-track: removal v1.2) |
| `PaddedSystem` | `prolix` top-level / `prolix.padding` | deprecated alias |
| `bucket_proteins` | `prolix.padding` / `prolix` top-level | gap: missing DeprecationWarning |
| `run_simulation` | `prolix.simulate` | legacy (retained, system_params= kwarg deprecated) |
| `simulate_frames` | `prolix.simulate` | legacy (retained) |
| `batched_simulate_frames` | `prolix.simulate` | legacy (retained) |

---

## 2. Replacement Mapping

- `batched_produce(batch, state, n_saves, steps_per_save)` → `EnsemblePlan([bundle]).run(n_steps, dt, kT)` — available now (commit 721dd83)
- `batched_equilibrate(...)` → removed; use cold-start with real forces: compute initial forces via `value_energy_and_grad_energy`, initialize `LangevinState` with real `force` field (see CLAUDE.md cold-start recipe)
- `LangevinState` from `prolix.batched_simulate` → `from prolix.types.integrators import IntegratorState` (internal); external callers use `EnsemblePlan.run()` and do not hold `LangevinState` directly
- `pad_protein(protein, ...)` → `MolecularBundle.from_pdb(path, forcefield='amber14')` — available now (commit 8219595); `MolecularBundle.from_protein()` planned v1.2
- `collate_batch(systems)` → `EnsemblePlan.from_bundles(bundles)` — planned v1.2 (#283); interim: construct `EnsemblePlan([bundle])` per-bundle and stack trajectories
- `PaddedSystem` → `MolecularBundle` (`prolix.types.bundles`) for new code; internal low-level code may continue using `prolix.typing.PhysicsSystem` (not deprecated)
- `bucket_proteins(proteins)` → no direct equivalent; construct per-protein `MolecularBundle.from_protein()` (v1.2); `EnsemblePlan.from_bundles()` handles heterogeneous bucketing internally (#283)
- `run_simulation(system, ...)` → `EnsemblePlan([bundle]).run(n_steps, dt, kT)` for new ensemble code; no removal planned before v2.0
- `system_params=` kwarg in `run_simulation` → pass `AtomicSystem` or `dict` as first positional arg
- `simulate_frames / batched_simulate_frames` → `EnsemblePlan.run()` for production; no removal planned before v2.0

---

## 3. Timeline

| Phase | Version | Action |
|-------|---------|--------|
| Now (current) | v1.0/v1.1 | Legacy symbols emit `DeprecationWarning` on use (batched_produce, batched_equilibrate, LangevinState re-export, pad_protein, collate_batch, PaddedSystem) |
| v1.1 | current | Warning text updated to reference `EnsemblePlan` where applicable |
| v1.2 | next minor | `collate_batch` removed; `EnsemblePlan.from_bundles()` available (#283); `MolecularBundle.from_protein()` available; `bucket_proteins` gets DeprecationWarning |
| v2.0 | major | `batched_produce`, `batched_equilibrate`, `LangevinState` re-export, `pad_protein`, `PaddedSystem`, `bucket_proteins` all removed |

---

## 4. CHANGELOG Migration Table

| Legacy | Replacement | Deprecated Since | Removed |
|--------|-------------|-----------------|---------|
| `batched_produce` | `EnsemblePlan.run()` | v1.1 | v2.0 |
| `batched_equilibrate` | cold-start with real forces | v1.1 | v2.0 |
| `LangevinState` (batched_simulate re-export) | `prolix.types.integrators.IntegratorState` | v1.1 | v2.0 |
| `pad_protein` | `MolecularBundle.from_pdb()` / `from_protein()` | v1.1 | v2.0 |
| `collate_batch` | `EnsemblePlan.from_bundles()` | v1.1 | **v1.2** |
| `PaddedSystem` (top-level) | `MolecularBundle` | v1.1 | v2.0 |
| `bucket_proteins` | `MolecularBundle` per-bundle + `EnsemblePlan` | v1.2 (pending) | v2.0 |

---

## 5. Rationale

The migration from `batched_produce`/`pad_protein`/`collate_batch` to `EnsemblePlan`/`MolecularBundle` serves three coupled goals.

**Paper-facing API alignment:** The §7.1 external comparator figure (#259) must demonstrate a clean `EnsemblePlan.from_bundles(...).run(...)` call path that is legible to reviewers comparing against DMFF/TorchMD/espaloma; the padded-system ad-hoc API obscures the heterogeneous-batch claim.

**Batch-planner integration:** `EnsemblePlan` is the insertion point for the xtrax.tiling backend (#1842), which decides `vmap` vs `safe_map` per axis at plan time rather than hard-coding dispatch in calling code; this abstraction cannot be retrofitted onto `batched_produce` without a new API boundary.

**MolecularBundle correctness by design:** `MolecularBundle` enforces the bucketed-JIT invariant (shape_spec is the only static=True field) and eliminates the class of bugs where callers construct heterogeneous `PaddedSystem` batches that silently trigger O(n) XLA recompilations; `collate_batch` in particular asserts homogeneity at runtime rather than by design, which is why it is on the v1.2 fast-track.

---

## 6. Cross-Surface Notes

- `collate_batch` has an accelerated removal target (v1.2) while all other deprecated symbols target v2.0 — this asymmetry exists because `collate_batch` only handles homogeneous topologies and its replacement (`EnsemblePlan.from_bundles`) directly unblocks #283.
- `bucket_proteins` is the only top-level export that lacks a `DeprecationWarning` despite internally calling deprecated `pad_protein` — it will chain-warn via `pad_protein` but has no top-level warning of its own.
- `PaddedSystem` triggers a warning only when imported from `prolix` top-level (via `__getattr__`); importing from `prolix.padding` or `prolix.typing` does NOT warn.
- `LangevinState` is deprecated as a re-export from `prolix.batched_simulate` but the underlying `prolix.typing.IntegratorState` is NOT deprecated.
- `run_simulation` / `simulate_frames` / `batched_simulate_frames` are legacy but have no deprecation warnings — retained because `EnsemblePlan` does not yet cover explicit solvent / NPT barostat / FIRE minimization.
- `batched_equilibrate_nl` and `batched_equilibrate_nl_dynamic` are NOT deprecated — they are the current NL equilibration paths.

---

## 7. Open Gaps Requiring Action Before v1.2

1. `bucket_proteins` is missing a `DeprecationWarning` — add warning at call site in `padding.py:572` pointing to `MolecularBundle.from_protein()`.
2. `EnsemblePlan.from_bundles()` not yet implemented — this is backlog item #283; blocks HP1 completion gate.
3. `MolecularBundle.from_protein()` (from a `Protein` object, not a PDB path) not yet implemented — needed to replace `pad_protein` in existing batched-minimize pipelines.
4. `run_simulation` / `simulate_frames` / `batched_simulate_frames` have no deprecation warnings and no replacement in `EnsemblePlan` yet for their full feature set (explicit solvent, PME, NPT barostat) — scope these for v2.0 deprecation only after `EnsemblePlan` reaches feature parity.

**Open questions:**
- Should `bucket_proteins` get a DeprecationWarning now (v1.1) or wait until `MolecularBundle.from_protein()` is available (v1.2)? Fast-tracking the warning without the replacement creates a dead-end for callers.
- `EnsemblePlan.from_bundles()` is the replacement for `collate_batch` but is not yet implemented (#283) — does `collate_batch` removal in v1.2 need to be gated on #283 landing first?
- `MolecularBundle.from_protein()` needs a design decision: does it share the same `from_pdb` factory or is it a separate class method?
- At what version does `run_simulation` get deprecated, and what is the `EnsemblePlan` equivalent for NPT/explicit-solvent simulations?
