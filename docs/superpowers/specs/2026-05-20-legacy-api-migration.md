# HP1 — Legacy API Migration Policy

**Date:** 2026-05-20
**Status:** Approved sub-spec
**Task ID:** 260520_hp1_migration
**Roadmap anchor:** `docs/superpowers/specs/2026-05-19-prolix-long-horizon-roadmap.md#2.7`
**Triggered by:** AR1 mitigation — *"Migration breaks downstream user code"* (roadmap §2.7 hard prerequisite)
**Scope:** Decision document only. No source edits. Deprecation warning implementation is a downstream fixer dispatch.

---

## §1 Motivation

Prolix v1.2 introduces a two-layer substrate API: `MolecularBundle` (typed, bucketed data container) and `EnsemblePlan` (execution plan built by `BatchPlanner`). This substrate is the implementation vehicle for Claim 1 (heterogeneous-batch SIMD dispatch) and the public-facing surface for Claim 3 (declarative ensemble API).

Five legacy entry-points predate this design and create AR1 risk: if they remain as-is when the new substrate lands, users cannot reliably know which path to take, internal code accumulates technical debt, and the paper's reproducibility package ships ambiguous APIs. The roadmap §2.7 oracle escalation locked HP1 from "open question" to "hard prerequisite": migration policy must be decided and documented before any Claim 1 implementation begins.

The five entry-points span two categories:

- **Active legacy infrastructure** (`batched_produce`, `LangevinState`) — used in current tests and single-system code paths; removal carries migration cost.
- **Data-layer predecessors** (`PaddedSystem`, `pad_protein`, `collate_batch`) — conceptually superseded by `MolecularBundle`; the two data types will coexist during the transition window.

A key implementation nuance: `prolix.batched_simulate` re-exports `LangevinState` as an alias for `prolix.typing.IntegratorState`. The canonical typed subclass lives in `prolix.types.integrators.LangevinState`. Tests import from both paths; a fixer must add deprecation warnings to both.

---

## §2 Decision Table

Callsite counts are grep-verified against `src/`, `tests/`, and `docs/` trees as of 2026-05-20. Internal counts include all occurrences in `src/` (definitions and usages). Test counts are per-test-file occurrences summed. Docs counts include all `.md` and `.json` files under `docs/`.

| entry_point | callsite_count_internal | callsite_count_test | callsite_count_docs | decision | replacement | deprecation_warning_text |
|---|---|---|---|---|---|---|
| `batched_produce` | 6 (1 file: definition) | 23 (4 files) | 2 (1 file: roadmap) | **co-exist** (soft deprecation) | `EnsemblePlan.from_bundles(...).run(...)` | `"batched_produce is deprecated as of v1.1 and will be removed in v2.0. Use EnsemblePlan.from_bundles(...).run(...) instead. See docs/superpowers/specs/2026-05-20-legacy-api-migration.md."` |
| `LangevinState` (both paths) | 75 (4 files) | 72 (10 files) | 2 (1 file: roadmap) | **co-exist** (soft deprecation on `batched_simulate` path only) | `prolix.types.integrators.LangevinState` (canonical); `IntegratorState` (base class) | `"Importing LangevinState from prolix.batched_simulate is deprecated as of v1.1. Import from prolix.types.integrators instead."` |
| `pad_protein` | 4 (2 files: def + `__init__`) | 10 (3 files) | 15 (6 files) | **co-exist** (soft deprecation) | `MolecularBundle.from_protein(protein, shape_spec)` (factory, to be specified in Claim 3 sub-spec) | `"pad_protein is deprecated as of v1.1 and will be removed in v2.0. Use MolecularBundle.from_protein() once available (v1.2). See migration guide."` |
| `PaddedSystem` | 76 (10 files) | 24 (8 files) | 35 (9 files) | **co-exist** (type alias, no-op deprecation note in docstring only until v1.3) | `MolecularBundle` | `"PaddedSystem is a deprecated alias for PhysicsSystem. Migrate to MolecularBundle (prolix.types.bundles) for new code. Removal planned for v2.0."` |
| `collate_batch` | 3 (2 files: def + `__init__`) | 9 (4 files) | 4 (3 files) | **hard-deprecate** (v1.1 warning → remove v1.2) | `EnsemblePlan.from_bundles(bundles)` | `"collate_batch is deprecated as of v1.1 and will be removed in v1.2. It only handles homogeneous topologies (same-bucket same-bond-count stacking). Use EnsemblePlan.from_bundles() for heterogeneous support."` |

---

## §3 Per-Entry-Point Rationale

### `batched_produce`

`batched_produce` is the current workhorse for batched production runs and is called in four test files (23 occurrences total), including `test_batched_produce_stability.py` and `test_batched_workflow_cold_start.py`. The function is defined only in `batched_simulate.py` (not in `__init__.py`) so it was never a top-level public export. Its replacement — `EnsemblePlan.from_bundles(...).run(...)` — does not exist yet; removing `batched_produce` before the replacement lands would break CI and user code simultaneously. Decision: co-exist with a `DeprecationWarning` in v1.1, targeting v2.0 removal. This gives a full release cycle (v1.1 → v1.2 → v2.0) for callers to migrate, and the warning fires immediately when users run their existing code against v1.1.

### `LangevinState`

`LangevinState` has the highest callsite count of all five entry-points (75 internal, 72 test). There are two distinct identities: (1) `prolix.types.integrators.LangevinState`, the canonical typed subclass used in `export.py` and imported explicitly in `test_export_langevin_step.py`; and (2) `prolix.batched_simulate.LangevinState`, which is `prolix.typing.IntegratorState` re-exported under that name. The `export.py` usage and `prolix.types.integrators` path are already correct and require no deprecation. Only the `batched_simulate` re-export path should carry a warning, directing users to import from `prolix.types.integrators`. Because `LangevinState` is the integrator state type for single-system non-batched code paths too (e.g., `export_langevin_step`), removal would break physics code unrelated to batching — co-exist is the only viable decision here.

### `pad_protein`

`pad_protein` appears in 15 doc locations (spread across 6 files, mostly architecture and implementation plan docs for explicit solvent). It is exported in `__init__.py` and used in test padding tests. Its replacement, `MolecularBundle.from_protein()`, is specified conceptually but has not been implemented as of the Claim 3 sub-spec writing. Until that factory exists, removing `pad_protein` would eliminate the only protein-ingestion path. Decision: co-exist with a `DeprecationWarning` in v1.1, targeting v2.0 removal. The warning message explicitly names the replacement factory and notes it is coming in v1.2 so users know when to migrate.

### `PaddedSystem`

`PaddedSystem` is the highest-volume entry-point by combined callsite count (76 internal, 24 test, 35 docs). It is a pure type alias (`PaddedSystem = PhysicsSystem`) defined in two places: `src/prolix/typing.py` and `src/prolix/padding.py`. The alias is already semantically inert — it adds no behavior — and is woven throughout energy functions, neighbor-list code, sharding, and explicit solvent infrastructure. Attempting to hard-deprecate it now would generate hundreds of noisy warnings across production code that is not yet migrated to `MolecularBundle`. The correct path is a docstring-level note marking it as a deprecated alias, followed by a single `DeprecationWarning` added to `__init__.py`'s re-export, not to every internal usage site. Removal is targeted for v2.0 when the full `MolecularBundle` migration is complete.

### `collate_batch`

`collate_batch` is the only candidate for hard-deprecation. The function's implementation contains an explicit assertion that all batched systems must share identical bucket size, bond count, angle count, dihedral count, and constraint count — making it structurally incompatible with heterogeneous batching. It is precisely what `EnsemblePlan.from_bundles()` is designed to replace. Callsite count is low (3 internal, 9 test, 4 docs), making migration tractable. All test callers are in `test_batched_simulate.py`, `test_batched_energy.py`, `test_cross_topology_integration.py`, and `test_padding.py` — all of which test infrastructure that is being superseded by the new substrate. Decision: `DeprecationWarning` in v1.1 with v1.2 removal (one release cycle). The warning text calls out the homogeneity limitation explicitly so users understand why the replacement is necessary.

---

## §4 CHANGELOG Migration Table

Add the following block to `CHANGELOG.md` under a new `## v1.1.x` section (or insert into the existing v1.1 section if one exists):

```markdown
### Deprecated

- **`batched_produce`** (`src/prolix/batched_simulate.py`): Issues `DeprecationWarning` on call.
  Replacement: `EnsemblePlan.from_bundles(...).run(...)` (available v1.2).
  Removal target: v2.0.

- **`LangevinState` re-export from `prolix.batched_simulate`**: Issues `DeprecationWarning` on import.
  Replacement: `from prolix.types.integrators import LangevinState`.
  Removal target: v2.0.

- **`pad_protein`** (`src/prolix/padding.py`): Issues `DeprecationWarning` on call.
  Replacement: `MolecularBundle.from_protein()` (available v1.2).
  Removal target: v2.0.

- **`PaddedSystem`** (`src/prolix/__init__.py`, `src/prolix/typing.py`): Docstring-level deprecation note + single `DeprecationWarning` in `__init__.py` re-export.
  Replacement: `MolecularBundle` (`prolix.types.bundles`).
  Removal target: v2.0.

- **`collate_batch`** (`src/prolix/padding.py`): Issues `DeprecationWarning` on call.
  Replacement: `EnsemblePlan.from_bundles(bundles)` (available v1.2).
  **Removal target: v1.2** (hard-deprecate, one cycle only).
```

| entry_point | deprecated_in | removal_version | replacement |
|---|---|---|---|
| `batched_produce` | v1.1 | v2.0 | `EnsemblePlan.from_bundles(...).run(...)` |
| `LangevinState` (batched_simulate path) | v1.1 | v2.0 | `from prolix.types.integrators import LangevinState` |
| `pad_protein` | v1.1 | v2.0 | `MolecularBundle.from_protein()` |
| `PaddedSystem` | v1.1 | v2.0 | `MolecularBundle` |
| `collate_batch` | v1.1 | **v1.2** | `EnsemblePlan.from_bundles(bundles)` |

---

## §5 Implementation Notes

A fixer agent applying this spec should make the following and no other changes:

### `src/prolix/batched_simulate.py`

- Wrap `batched_produce` body with `warnings.warn(...)` as the first executable statement (before any JAX computation). Use `stacklevel=2` so the warning points to the caller's frame, not the function definition.
- Add a module-level `__getattr__` (or convert the `LangevinState` alias to a lazy import that emits a warning) so that `from prolix.batched_simulate import LangevinState` issues a `DeprecationWarning`. The alias at line 25 (`from prolix.typing import PaddedSystem, IntegratorState as LangevinState`) must remain functional for all existing `batched_simulate.py` internal usages — only the external import path needs the warning.

### `src/prolix/padding.py`

- Wrap `pad_protein` body (first statement) with `warnings.warn(...)`, `stacklevel=2`.
- Wrap `collate_batch` body (first statement) with `warnings.warn(...)`, `stacklevel=2`. This is the hard-deprecate; the warning should state v1.2 removal.

### `src/prolix/__init__.py`

- Add a `DeprecationWarning` to the `PaddedSystem` re-export. Because `PaddedSystem` is a type alias (not callable), the warning cannot fire on construction. Instead, add it via a module-level `__getattr__` that intercepts the attribute access at `prolix.PaddedSystem` and emits the warning before returning the alias. This is the standard pattern for deprecating type-alias exports in Python 3.10+.
- Do not add warnings inside `src/prolix/typing.py` for `PaddedSystem` — that file is an internal typing module and the alias is used by energy functions that are not being deprecated.

### Warning category

All warnings use the standard library `DeprecationWarning` category (not `FutureWarning`). This matches the existing `batched_equilibrate` deprecation pattern already in `batched_simulate.py` (line ~966).

### Test updates (downstream fixer scope)

The fixer should add `pytest.warns(DeprecationWarning)` context managers to the existing test callsites that exercise these deprecated paths, or mark those tests as expected-warning tests. This keeps CI clean without suppressing the warnings globally.

---

## §6 Exit Criteria

> Sub-spec merged; CHANGELOG migration table complete; deprecation warnings added to legacy entry-points.

Specifically, this task (HP1) is complete when:

1. This file (`docs/superpowers/specs/2026-05-20-legacy-api-migration.md`) is committed to `main`.
2. `CHANGELOG.md` contains the migration table from §4.
3. The following five `DeprecationWarning` calls are live in the codebase:
   - `batched_produce` in `src/prolix/batched_simulate.py`
   - `LangevinState` import path in `src/prolix/batched_simulate.py` (module-level `__getattr__`)
   - `pad_protein` in `src/prolix/padding.py`
   - `collate_batch` in `src/prolix/padding.py`
   - `PaddedSystem` re-export in `src/prolix/__init__.py` (module-level `__getattr__`)
4. All existing test callsites of deprecated functions either pass with `pytest.warns(DeprecationWarning)` or are updated to use the canonical replacement path.
5. No test regressions: `uv run pytest -m "not slow"` exits 0.

HP1 does not require that `EnsemblePlan` or `MolecularBundle.from_protein()` exist — those are Claim 1 and Claim 3 deliverables. The deprecation warnings may reference replacements that are "available v1.2" without those replacements being present at the time of warning insertion.
