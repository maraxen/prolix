---
title: "XR-SHADOW — fitting↔MD plan equality"
backlog_id: XR-SHADOW
praxia_id: 3276
epic: 260709_xtrax_rewire
depends_on: [XR-BUDGET, XR-DISPATCH]
priority: P1
difficulty: standard
status: near_ready
challenge_verdict: near_ready
challenge_summary: "Option 1 locked: both sides via shared xtrax path in test only; SHADOW_AXES named; strategy-type asserts required."
---

# XR-SHADOW

## Goal
Pytest-tier fixture proving fitting and MD planners agree on strategy objects and that runtime dispatch matches the plan (prevents dual-authority drift).

## Locked decisions (option 1)

| Topic | Lock |
|-------|------|
| Fitting path in test | `BatchPlanner(...).plan_with_xtrax()` (same adapter as MD) |
| MD path in test | `plan_axes_with_xtrax(...)` (same as `EnsembleMDPlanner`) |
| Production fitting | Unchanged greedy `.plan()` until XR-FIT-FLIP |
| Shared fixture name | `SHADOW_AXES` in `tests/tiling/test_xr_shadow_equality.py` |
| Axes | Homogeneous `N_ATOMS` + homogeneous `N_MOLS` (fixed cardinalities); compare both |
| Strategy recovery | Map prolix decision → `Vmap()` / `SafeMap(batch_size=...)` (same rules as `n_mols_strategy`); assert `isinstance` / `type` |
| Dispatch check | For `N_MOLS`: `make_axis_dispatch(n_mols_strategy(plan, n))` → `VmapIterator` or `SafeMapIterator` |
| Kill / AC4 | Fail if test only compares ints: require strategy-type asserts |
| Cases | (1) large budget → Vmap retain; (2) tight budget → SafeMap demotion |

## Scope
- New test: `tests/tiling/test_xr_shadow_equality.py`
- Compare strategy **types** (Vmap/SafeMap) and `batch_size`, not ints alone
- Assert runtime MD dispatch (`n_mols_strategy` + `make_axis_dispatch`) agrees with planned strategy

## Non-goals
Bathos claim-tier (XR-PARITY-OMM-WATER); flipping production `make_fitting_planner` (XR-FIT-FLIP); fitting runtime → `make_axis_dispatch`; deleting greedy planner.

## Acceptance Criteria
1. Given `SHADOW_AXES` planned via `plan_with_xtrax` (fitting side) and `plan_axes_with_xtrax` (MD side), when shadow test runs, then each axis decision matches on `type(strategy)` and `batch_size`.
2. Given the same fixture, when `n_mols_strategy` + `make_axis_dispatch` run, then the iterator class agrees with the planned Vmap/SafeMap strategy.
3. Given the test is pytest-tier, when listed in CI fast suite, then it runs without bathos/`bth run`.
4. Given prolix still collapses strategies to ints, when shadow runs, then the test asserts strategy object types (not ints alone).

## Rollback
Delete the test file.

## References
- Epic AC3; adversarial critique on int-only shadow
- XR-BUDGET / XR-DISPATCH (prerequisites)
