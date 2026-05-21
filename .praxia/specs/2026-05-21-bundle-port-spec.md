# Bundle Composability Stack: Port Specification

**prolix v1.2 §7.1 — Phase B2 composability gate**
**Spec date:** 2026-05-21 | **EOW checkpoint:** 2026-05-29
**Status:** recon-validated 2026-05-21 (3 corrections applied) | oracle-reviewed 2026-05-21 (7 fixes applied)

---

## §0. Scope

§7.1 claims composability of the **fitting loop substrate** only. Conformer scheduling (`sample_one_conformer`) is a host-side concern in v1.2 and joins the substrate in v1.3 as `ConformerScheduleBundle`. The paper draft and figure caption MUST NOT reference "scheduled conformer sampling" as part of the v1.2 substrate.

**Fallback trigger:** If Phase 4 is not green by **Tue 2026-05-27 EOD**, drop to flattened narrow `BondedFittingBundle` (no per-mol/batched split, no FittingPlan composition) and defer full composability to v1.3. Pre-committed at spec time to avoid in-flight scope debate.

---

## Validation log

- 2026-05-21: recon agent verified all tev_design citations (§A pass), all prolix frozen-symbol contracts (§B pass with `--lr`→`--learning-rate` fix), base types `BondedParams`/`BondedTopology` exist (§C pass), no `"n_systems"` string in any config file (§F pass = rename safe), Phase 7 fixture must be built during Phase 3-5 (§D acknowledged, no spec change).
- 2026-05-21: axis_index clarified as REORDER not extension (recon §E).

---

## §1. File Layout

| Path | Status | Purpose |
|---|---|---|
| `src/prolix/fitting/bundles.py` | NEW | Per-mol `ConformerBundle`, `FittingBundle`; `eqx.Module` types only, no logic |
| `src/prolix/fitting/bundle_builder.py` | NEW | Host-side `build_fitting_bundle()` factory; resolves Optional fields to zero-filled concrete arrays before JIT boundary |
| `src/prolix/fitting/config.py` | NEW | `FittingConfig` and `FittingPlan` dataclasses; `make_fitting_plan()` factory |
| `src/prolix/run/spec.py` | NEW | `BatchingConfig` dataclass; `FittingAxisNames` string constants |
| `src/prolix/tiling/axes.py` | MODIFY | Add `N_BONDS`, `N_ANGLES`, `N_TORSIONS`, `N_CONFORMERS`, `N_MOLS`; keep `N_ATOMS`, alias `N_SYSTEMS → N_MOLS` |
| `src/prolix/fitting/__init__.py` | MODIFY | Export new public symbols; no removals |
| `src/prolix/fitting/batched.py` | NO CHANGE | `BondedParamsBundle`, `BondedTopologyBundle`, `stack_molecules` untouched through Phase 6 |
| `src/prolix/fitting/train.py` | NO CHANGE | Existing `train_loop_batched` untouched through Phase 6 |
| `scripts/experiments/fit_bonded_hp4.py` | NO CHANGE until Phase 6 | Frozen through numerical-equivalence gate |
| `tests/fitting/test_bundles.py` | NEW | Bundle invariant tests |
| `tests/fitting/test_bundle_builder.py` | NEW | Builder round-trip + zero-fill tests |
| `tests/fitting/test_fitting_plan.py` | NEW | End-to-end FittingPlan `.step()` / `.evaluate()` |

---

## §2. Type Signatures

### Design decision: per-mol + batched Bundle (two-level)

`ConformerBundle` and `FittingBundle` wrap **one molecule** (no `B` axis). A separate `BatchedFittingBundle` is the type `train_loop_batched` consumes — produced via a typed constructor `BatchedFittingBundle.stack(bundles: list[FittingBundle])`, not a side ritual on the host.

Rationale (oracle Q1): a per-mol-only design with host-side `stack_fitting_bundles()` would force callers to know the "secret handshake" between per-mol Bundle and `train_loop_batched`, undermining the composability claim. Two typed bundle levels (per-mol AND batched), both implementing `.step/.evaluate`, lets callers compose at the natural level. Paper claim then reads: "heterogeneous fitting is a `BatchedFittingBundle` produced by stacking N `FittingBundle`s," which is the cleanest possible composability story.

`BatchedFittingBundle.step` delegates to `train_loop_batched` internals; the existing `BondedParamsBundle` / `BondedTopologyBundle` in `batched.py` are wrapped, not replaced. R2 (in-flight sweep) still holds.

### `ConformerBundle(eqx.Module)`

```python
class ConformerBundle(eqx.Module):
    positions: Float[Array, "N_conf n_atoms 3"]
    forces_ref: Float[Array, "N_conf n_atoms 3"]
    energies_ref: Float[Array, "N_conf"]
    atom_mask: Bool[Array, "n_atoms"]
    n_conf: int = eqx.field(static=True)
    n_atoms: int = eqx.field(static=True)
```

### `FittingBundle(eqx.Module)`

```python
class FittingBundle(eqx.Module):
    conformers: ConformerBundle
    params: BondedParams              # trainable
    topology: BondedTopology          # constant connectivity
    box: Float[Array, "3 3"]          # always concrete; vacuum = jnp.zeros((3,3))
```

Oracle Q2 (box Optional resolution): the original `box: Float[Array, "3 3"] | None` was a real recompile hazard. `has_box: bool = eqx.field(static=True)` would be the worst of both worlds (static booleans flipping cause retracing). Resolution: `box` is always a concrete (3,3) array; vacuum systems carry `jnp.zeros((3,3))` as sentinel. Builder enforces this.

### `BatchedFittingBundle(eqx.Module)`

```python
class BatchedFittingBundle(eqx.Module):
    conformers_batched: "BatchedConformerBundle"   # padded across mols
    params_batched: BondedParamsBundle             # existing from batched.py
    topology_batched: BondedTopologyBundle         # existing from batched.py
    box_batched: Float[Array, "B 3 3"]
    n_mols_real: int = eqx.field(static=True)      # real B (≤ padded B)

    @staticmethod
    def stack(bundles: list[FittingBundle]) -> "BatchedFittingBundle":
        """Typed constructor — calls existing stack_molecules() internally."""
        ...

    def step(self, state: TrainState, conformer_idx: int) -> tuple[TrainState, TrainMetrics]: ...
    def evaluate(self, state: TrainState) -> TrainMetrics: ...
```

`BatchedConformerBundle` mirrors `BondedParamsBundle`'s padded-across-mols shape: `Float[Array, "B N_conf n_atoms 3"]` etc.

### `FittingConfig`

```python
@dataclasses.dataclass(frozen=True)
class FittingConfig:
    lr: float
    n_steps: int
    alpha: float = 0.25
    w_reg: float = 0.01
    grad_clip_norm: float | None = None
```

### `TrainState(eqx.Module)`

Oracle Q2 (state lifecycle + R6 RNG plumbing). `TrainState` bundles everything carried across `.step()` calls — no bare `opt_state`, no `key` as side argument.

```python
class TrainState(eqx.Module):
    params: BondedParams
    opt_state: optax.OptState
    key: PRNGKeyArray
    step_count: Int[Array, ""]
```

### `FittingPlan`

```python
@dataclasses.dataclass(frozen=True)
class FittingPlan:
    optimizer: optax.GradientTransformation
    loss_fn: Callable
    config: FittingConfig

    @eqx.filter_jit
    def step(self, bundle, state, conformer_idx) -> tuple[TrainState, TrainMetrics]: ...

    @eqx.filter_jit
    def evaluate(self, bundle, state) -> TrainMetrics: ...
```

Oracle Q3 resolution: **internal `@eqx.filter_jit`** is correct — callers don't need to know about JIT (composability claim demands this). Static signature: `(n_mols_real, max_atoms_bucket, n_steps, lr)`. All non-array Bundle fields must be `eqx.field(static=True)` and hashable. Phase 7 gate test asserts JIT compile count == 1 on second call.

### `build_fitting_bundle()`

```python
def build_fitting_bundle(
    positions_all, forces_all, energies_all,
    params, topology,
    *,
    atom_mask=None, box=None, n_conf_real=None,
) -> FittingBundle: ...
```

### `AxisSpec` reorder + extensions (`tiling/axes.py`)

**Current state** (verified by recon 2026-05-21):
- `N_SYSTEMS @ axis_index=0`
- `N_ATOMS @ axis_index=1`

Phase 2 is a **REORDER + EXTEND**, not pure additive. Complete new ordering (innermost → outermost):

```
axis_index 0: N_ATOMS       (MOVED from 1; cardinality + granularity unchanged)
axis_index 1: N_BONDS       (NEW, cardinality=512, tile_granularity=64, heterogeneous=False)
axis_index 2: N_ANGLES      (NEW, cardinality=512, tile_granularity=64, heterogeneous=False)
axis_index 3: N_TORSIONS    (NEW, cardinality=512, tile_granularity=64, heterogeneous=False)
axis_index 4: N_CONFORMERS  (NEW, cardinality=2048, tile_granularity=1, heterogeneous=True)
axis_index 5: N_MOLS        (RENAMED from N_SYSTEMS @0→5; keep `N_SYSTEMS = N_MOLS` alias for one release)
```

Recon confirmed no pre-reg configs reference `"n_systems"` as a string key — only `axes.py:15` (definition) and `tests/physics/test_tiling_axes.py:15,37` (assertions). Safe to rename; tests need updating in Phase 2 gate.

---

## §3. Phased Plan

Phase 1 (DONE, commit 5b2e1ce): rename to `BondedParamsBundle` / `BondedTopologyBundle`.

| # | Title | Files | Gate | Depends | Rollback |
|---|---|---|---|---|---|
| 2 | AxisSpec reorder + extensions (REORDER, see §2) | `tiling/axes.py`, `tests/physics/test_tiling_axes.py` | `tests/physics/test_tiling_axes.py` passes with new ordering | 1 | revert axes.py + test_tiling_axes.py |
| 3 | ConformerBundle + FittingBundle types | `fitting/bundles.py` | `test_bundle_pytree_round_trip` | 2 | delete bundles.py |
| 4 | `build_fitting_bundle` | `fitting/bundle_builder.py` | `test_optional_fields_resolve_to_zeros` | 3 | delete builder |
| 5 | `FittingConfig` + `FittingPlan` + `make_fitting_plan` | `fitting/config.py` | `test_step_runs_without_nan` | 4 | delete config.py |
| 6 | `BatchingConfig` + `FittingAxisNames` + `make_fitting_planner` | `run/spec.py` | `test_make_fitting_planner_returns_valid_plan` | 5 | delete run/spec.py |
| 7 | Numerical equivalence gate | read-only script | manual: loss curves match `train_loop_batched` to 1e-5 on 3-mol fixture | 6 | abort Phase 8 |
| 8 | Cutover `fit_bonded_hp4.py` + `__init__.py` | exports + script | full fitting suite + smoke CLI | 7 passed | revert script |

---

## §4. Risk Table

| ID | Risk | Sev | Like | Mitigation | Detection |
|---|---|---|---|---|---|
| R1 | Domain mismatch (tev_design backbone vs prolix variable atoms) | Med | Low | Per-mol bundle adopts prolix's (N_conf, n_atoms, 3) natively; mask invariant unchanged | Phase 3 gate test |
| R2 | In-flight 24-cell sweep broken by mid-port refactor | High | Med | `batched.py`, `train.py`, `fit_bonded_hp4.py` read-only through Phase 6. `BatchedFittingBundle` wraps existing types | `git diff --name-only HEAD` pre-flight every phase |
| R3 | Test coverage gaps (16 happy-path tests) | Med | High | §5 property tests at Phase 3–5 gates; equivalence test at Phase 7 | Phase 7 gate hard-fails on divergence |
| R4 | jaxtyping runtime overhead | Low | Low | Type-annotation only, no decorators (mirrors tev_design) | Compile time benchmark if needed |
| R5 | Pre-reg blocked if port slips past 2026-05-29 | High | Med | **Pre-committed fallback trigger Tue 2026-05-27 EOD.** If Phase 4 not green by then → flattened narrow `BondedFittingBundle`, no FittingPlan composition | Phase 4 status check Tue 2026-05-27 17:00 |
| **R6** | RNG plumbing leak: `key` as side argument vs bundled in `TrainState` | Med | Med | `TrainState` is eqx.Module bundling `(params, opt_state, key, step_count)`; no bare keys cross the API | `test_state_purity_no_bare_keys` at Phase 5 gate |
| **R7** | Save/load round-trip asymmetry (`FittingConfig` frozen dataclass vs `FittingBundle` eqx.Module) | Med | Med | All non-array Bundle fields `eqx.field(static=True)` and hashable; explicit round-trip test required at Phase 7 | `test_bundle_save_load_round_trip` at Phase 7 |
| **R8** | `FittingPlan.step` recompiles on static-field changes across calls | Med | Med | Documented static signature `(n_mols_real, max_atoms_bucket, n_steps, lr)`; Phase 7 compile-count test | `test_step_compile_count_one` asserts JIT cache hit on second call |

---

## §5. Test Strategy

**Bundle invariants:**
- `test_conformer_bundle_pytree_leaves_count` — 4 leaves (positions, forces_ref, energies_ref, atom_mask)
- `test_conformer_bundle_static_fields_not_leaves` — n_conf, n_atoms absent
- `test_fitting_bundle_jit_passthrough` — round-trips through `jax.jit(lambda b: b)`
- `test_fitting_bundle_filter_diff` — `eqx.filter(bundle, eqx.is_array)` isolates params

**Builder round-trip:**
- `test_optional_fields_resolve_to_zeros` — atom_mask=None → all-ones
- `test_box_none_resolves_to_zero_matrix` — `.box == jnp.zeros((3,3))`
- `test_n_conf_real_stored_as_static`

**FittingPlan end-to-end:**
- `test_step_runs_without_nan`
- `test_evaluate_deterministic`
- `test_step_decreases_loss_over_10_steps`

**Numerical equivalence (Phase 7):** new `BatchedFittingBundle.step` path vs `train_loop_batched` direct path on 3-mol, 50-step fixture:
- Per-step loss: `jnp.allclose(new, old, atol=1e-6, rtol=1e-5)`
- Per-step gradients: `jnp.allclose(new, old, atol=1e-6, rtol=1e-5)`
- Final params: `jnp.allclose(new, old, atol=1e-5, rtol=1e-4)`

Tolerances chosen to detect non-trivial divergence while permitting floating-point reordering.

**Compile-count gate (Phase 7):** `test_step_compile_count_one` — call `.step()` twice with identical static signature, assert JIT cache hit (no re-trace).

**Save/load round-trip (Phase 7):** `test_bundle_save_load_round_trip` — `eqx.tree_serialise_leaves` → reload → `eqx.tree_equal`. Tests `FittingBundle`, `BatchedFittingBundle`, plus `FittingConfig` via `dataclasses.asdict` + JSON round-trip.

**State purity (Phase 5):** `test_state_purity_no_bare_keys` — `.step()` signature contains no `key` parameter (all RNG bundled in `TrainState`).

---

## §6. Interfaces to Keep Stable

| Symbol | Stability through |
|---|---|
| `train_loop_looped_baseline(...)` | Phase 8 |
| `train_loop_batched(...)` | Phase 8 |
| `stack_molecules(...)` return type | Phase 8 |
| `load_params_init_json(...)` return type | Phase 8 |
| `fit_bonded_hp4.py` CLI flags (`--mode`, `--n-steps`, `--learning-rate`, `--alpha`, `--w-reg`, `--seed`, `--n-mols-target`, `--n-conf-cap`) | Phase 8 |
| `fitting/__init__.py` exports | additive only |

---

## §7. Open Questions (resolved by oracle 2026-05-21)

1. ✅ **JIT boundary**: internal `@eqx.filter_jit` on `FittingPlan.step` + documented static signature `(n_mols_real, max_atoms_bucket, n_steps, lr)` + Phase 7 compile-count test.
2. ✅ **`N_SYSTEMS` alias**: recon confirmed no string-key references in pre-reg configs. Safe to rename; only `axes.py:15` and `tests/physics/test_tiling_axes.py:15,37` need updating in Phase 2.
3. ✅ **`stack_fitting_bundles()` signature**: replaced by typed constructor `BatchedFittingBundle.stack(bundles: list[FittingBundle])`. The dict-schema concern for `train_loop_batched.batched_data` is absorbed inside `BatchedFittingBundle.step()` — caller never sees the dict.
4. ✅ **Conformer scheduler placement**: deferred to v1.3. §0 scope sentence pins paper language to "fitting loop substrate" only; conformer scheduling is host-side concern in v1.2.
5. ✅ **`box` Optional handling**: resolved as always-concrete `Float[Array, "3 3"]` with vacuum sentinel `jnp.zeros((3,3))`. No `has_box` flag (worst-of-both-worlds rejected).
