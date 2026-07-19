---
title: XR-WHILE-CARRY
description: Scope a lax.while_loop-backed WhileCarry strategy for xtrax (upstream), motivated by debt 760's neighbor-list carry in prolix's EnsemblePlan inference dispatch
status: draft
task_id: 260715_b1_physics_parity
date: '260718'
backlog_ids: ''
adversarial_review: ''
---
# XR-WHILE-CARRY

## Goal

Scope (not implement — this belongs in the `xtrax` repo, not prolix) a first-class `lax.while_loop`-backed strategy for `xtrax.tiling`, closing an abstraction gap debt 760 (prolix's NL+PME `EnsemblePlan` connection work) worked around with a hand-rolled carry instead of an xtrax primitive.

## Motivation

`EnsemblePlan.run(run_mode="inference")` dispatches its step loop via `dispatch_n_steps_inference` (`prolix/api/ensemble_dispatch.py`), which is a raw `lax.while_loop` — chosen specifically because `lax.scan` unrolls per-step IR and was measured to cost 300-400x more compile time at production scale (see `using-jax` skill, and `B1-INFER`, `.praxia/docs/specs/260713_b1-infer.md`, which locked this decision). Debt 760 needed to extend that loop's carry with a neighbor list that must periodically update in-loop (`neighbor.update()` gated on `step_i % K == 0` via `jax.lax.cond`) and an OR-accumulated overflow flag, host-checked once after the loop returns.

xtrax is genuinely wired into most of `EnsemblePlan` already: shape-bucket planning (`EnsembleMDPlanner` → `plan_axes_with_xtrax` → xtrax `BatchPlanner`/`MemoryBudget`), `N_MOLS` dispatch (`Vmap`/`SafeMap` via `make_axis_dispatch`), duplicate-topology dedup (`DedupGather`), and the **trajectory** (scan) step-loop (`dispatch_n_steps` → `make_axis_dispatch(Scan(), ...)`). But `dispatch_n_steps_inference` — the one loop debt 760's carry extends — is not xtrax-wired, and this was confirmed to be a **deliberate, locked** decision, not an oversight:

- `XR-CARRY` (`.praxia/docs/specs/260709_xr-carry.md:25`) scoped its execution wiring to `_run_single`/scan only — while_loop/inference didn't exist yet on that date.
- `B1-INFER` locked `lax.while_loop` (carry-only, not reverse-mode AD safe) for inference, with no xtrax `CarrySpec` mention for that path.
- Structurally, xtrax's `CarrySpec`/`Scan` strategy maps to `lax.scan`'s `(carry, x) -> (carry, y)` contract, not `lax.while_loop`'s `carry -> carry`. There is no "carry-only unstacked while_loop" strategy among xtrax's five named `AxisStrategy` variants (`Vmap | SafeMap | Scan | DedupGather | Bucket`) as used anywhere in prolix. `plan_n_steps_with_carry` (xtrax `step_carry.py:24-53`) does build a real `CarrySpec`, but it is planning-only, dead in the execute path (no caller outside its own module + its own test).

None of debt 760's actual new code is a natural fit for xtrax's axis-dispatch abstractions either way: the damped-Coulomb-over-neighbors term is a plain per-step JAX kernel (not a batch-axis problem), NL allocation/rebuild is host-side-once + periodic in-loop `.update()` (not an axis-dispatch problem), and the carry extension is exactly the kind of "add a field to an existing loop's state" change `XR-CARRY`/`B1-INFER` already anticipated without xtrax.

**Decision (prolix side, via user AskUserQuestion during debt 760 scoping)**: neither "keep the carry hand-rolled forever" nor "switch `dispatch_n_steps_inference` to `Scan` and eat the compile-time regression" — make `lax.while_loop` a first-class, upstream-supported xtrax strategy, closing the gap at the library level. Debt 760 itself proceeded on the hand-rolled carry (implemented and verified 2026-07-18, `prolix/api/ensemble_plan.py`'s `_NLDispatchCarry`/`_nl_step_fn`) — that implementation is now the validated reference instance this spec's design should generalize from, not a blind guess.

## Current xtrax architecture (as of this scoping, `/home/marielle/projects/xtrax/src/xtrax/tiling/`)

- `AxisStrategy` (`strategy.py:109`) is a sealed union: `Vmap | SafeMap | Scan | DedupGather | Bucket`.
- `Scan` (`strategy.py:56-63`) carries a `ScanTransition` Protocol — `(carry, x) -> (new_carry, y)` (`strategy.py:12-15`), matching `jax.lax.scan` exactly, including per-step output collection (`y`).
- `make_axis_dispatch` (`dispatch.py:31-102`) maps `Scan` → `JaxScanIterator` (`iterator.py:153`), `__call__(fn, init, xs) -> (final_carry, stacked_outputs)`.
- `CarrySpec` (`carry.py:22-47`) is the planner-facing declaration that pre-selects `Scan` for a named axis in `BatchPlanner.plan()`'s Phase 0.
- **No `lax.while_loop`-backed strategy exists anywhere in xtrax today** (`grep -rn "while_loop"` across `src/` and `.praxia/` returns zero hits — this has never been discussed or backlogged upstream before this scoping).

## The gap

`Scan`'s `(carry, x) -> (carry, y)` contract assumes (a) a known input sequence `x` to iterate over and (b) that per-step outputs `y` are wanted (stacked into a trajectory). `dispatch_n_steps_inference` needs neither: `step_fn: (state, step_i) -> state`, no external per-step input beyond the loop's own counter, no per-step output collection — just "run N times, keep only the final carry." That is exactly `lax.while_loop`'s shape, and exactly why `B1-INFER` chose it over `lax.scan` in the first place.

## Proposed design (for xtrax's own team/session to refine — not locked here)

- New sealed-union member `WhileCarry` (`strategy.py`), analogous to `Scan` but carry-only:

  ```python
  @dataclass(frozen=True)
  class WhileCarry:
      """Carry-only strategy: compiles to lax.while_loop, no per-step output
      collection. For inference-only loops where only the final carry matters.
      Not reverse-mode AD safe (lax.while_loop limitation, same as Scan's
      heterogeneous-axis restriction)."""
      body: WhileBodyFn | None = None   # (carry) -> new_carry -- no x, no y
      cond: WhileCondFn | None = None   # (carry) -> bool; a simple step-counter
                                          #   default should cover the common
                                          #   "run exactly N steps" case
      init: Any | None = None
  ```

- New `WhileBodyFn`/`WhileCondFn` Protocols (`strategy.py`), narrower than `ScanTransition` (no `x`, no `y`).
- `make_axis_dispatch` (`dispatch.py`): new branch, `isinstance(strategy, WhileCarry)` → new `WhileLoopIterator` (`iterator.py`), `__call__(fn, init) -> Any` (final carry only — a genuinely different return arity than `JaxScanIterator`, not a flag on the same iterator).
- `BatchPlanner`/`CarrySpec` (`plan.py`, `carry.py`): needs a way to select `WhileCarry` over `Scan` for a given axis — likely a `collect_outputs: bool = True` field on `CarrySpec` (default preserves today's `Scan` behavior; `collect_outputs=False` routes to `WhileCarry`), rather than a wholly separate spec type, to keep the declarative surface small.
- Same heterogeneous-axis rejection as `Scan` applies (`dispatch.py:84-92`) — `lax.while_loop` has the identical static-carry-shape constraint across all axis elements.

## Reference implementation to generalize from

`prolix/api/ensemble_plan.py`'s `_NLDispatchCarry` (a `NamedTuple {langevin_state, neighbor, did_overflow}`) and `_run_single_inference`'s `_nl_step_fn`, landed 2026-07-18 (commit `7367f93`), is a real, verified, production instance of exactly the pattern `WhileCarry` should generalize:

- Periodic in-loop work gated by the loop's own step counter via `jax.lax.cond` (here: `neighbor.update()` every `nl_update_every` steps).
- A field OR-accumulated across iterations without ever branching on it in-loop (`did_overflow`), host-checked once after the loop returns.
- Masked post-processing applied every iteration outside the "core" computation (here: ghost-position/momentum pinning via `eqx.tree_at`).
- `dispatch_n_steps_inference`'s `step_fn` signature is `(state, step_i) -> state`, reusing the while_loop's own already-tracked iteration counter rather than threading a redundant counter through the carry itself — worth carrying into `WhileCarry`'s own `body` signature convention.

Verified in production use (1VII, real periodic PME): zero ghost-position drift over 20 steps with NL updates every 3 steps; overflow-then-reallocate confirmed both to recover when the capacity bump is sufficient and to correctly fail loud (not silently corrupt) when it isn't.

## Non-goals of this spec

- Does not implement `WhileCarry` — that is xtrax-repo work, in its own session/worktree, reviewed under xtrax's own `.praxia/docs/specs/` convention.
- Does not change `dispatch_n_steps_inference`'s current hand-rolled implementation — it stays as-is until/unless a working `WhileCarry` lands upstream and a follow-up prolix-side migration is separately scoped and justified.
- Does not claim `WhileCarry` is needed for debt 760 to function — debt 760 is done and shipped without it (Phase 6 steps 4-7, commits `7367f93`/`83899a2`/`3000024`).

## Status

Scoping only, drafted 2026-07-18 alongside debt 760's Phase 6 implementation. No xtrax-side backlog item filed yet — the concrete next step is writing this design up as a proper xtrax-side spec (matching that project's own conventions) and getting it reviewed there before any code lands.
