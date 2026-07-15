---
title: "B1-SETTLE-STACK"
backlog_id: B1-SETTLE-STACK
epic: 260528_b1-full
depends_on: [B1-XTRAX-WIRE]
priority: P1
status: completed
challenge_verdict: pass
challenge_summary: "Un-park SETTLE for the stacked path: mask-guard padded water rows, wire bundle.water_indices through instead of None."
completed_2026_07_15: true
---

# B1-SETTLE-STACK

## Goal

Make rigid-water SETTLE constraints (and, by the same fallback logic in `settle_langevin`, H-bond-equivalent constrained dynamics) work in `EnsemblePlan`'s stacked/vmapped dispatch path. Today `_setup_integrator` forces `water_indices=None` whenever `len(group)>1`, so **every** B1 shape class runs fully unconstrained flexible dynamics — a real physics gap vs. OpenMM's `constraints=HBonds, rigidWater=True` baseline, not a deliberate scope decision. Un-parks a P2 backlog item that was deprioritized 2026-07-14 after "water finite but pathological drift" — root cause identified this session (see below).

## Root cause of the original park

Padded `water_indices` rows are `[0,0,0]` (`src/prolix/padding.py:47-52`). Passing the padded array unmasked makes every pad "water" reference atom 0 three times — a degenerate zero-length bond vector feeds the Horn-quaternion `jnp.linalg.eigh` (`settle.py:335`), producing an arbitrary rotation scattered onto **real** atom 0 every step. `MolecularBundle.water_mask` (`bundles.py:174`) exists but was never consumed in `settle.py` — this is almost certainly the actual cause of the previously-observed drift, not a fundamental instability.

## Locked decisions

| Topic | Lock |
|-------|------|
| `water_indices` source | `bundle.water_indices` (already bucket-sized via `water_bucket_size`, `bundle_md.py:75-77`) passed directly — **no** `int(jnp.asarray(...))` host call anywhere in the stacked path |
| Masking | `MolecularBundle.water_mask` threaded through every SETTLE/RATTLE function that scatters onto per-water indices; padded rows contribute exactly zero force/rotation/momentum |
| `apply_fn` signature | Unchanged (`apply_fn(state, kT, dt) -> state`) — composes directly with the existing `step_fn` passed to `dispatch_n_steps`/`dispatch_n_steps_inference`; no new xtrax abstraction (`Fuse`/`Tap`/`Sink` do not apply to a physics substep — see research note below) |
| Single-system path | Unchanged — already correct, not touched by this leaf |

## Acceptance Criteria

1. A full-bucket water group (real `n_waters` == bucket size, no padding) reproduces the single-system `settle_langevin` trajectory exactly (bitwise or tight atol).
2. A sub-bucket-full water group: padded atom positions stay at a fixed sentinel across all steps (provably inert); real-atom energy and angular-momentum conservation match the single-system path within existing SETTLE tolerances.
3. Compile smoke test at the largest water bucket (8000) — the new nested (systems × waters) `vmap` of `jnp.linalg.eigh` must not regress the previously-fixed XLA hang (`settle.py:511-519`, Cramer's-rule fix, job 15374423).
4. `_setup_integrator`'s `integration_prefix is not None` branch no longer sets `water_indices=None`; stacked B1 water class runs with SETTLE active end to end (`b1_init_exec.py --smoke` green).

## Implementation

- [`src/prolix/physics/settle.py`](../../src/prolix/physics/settle.py) — mask-guarded scatter in `settle_positions`/`_settle_water_batch`, `settle_velocities`, `_apply_rattle_velocity_correction`, `_r_step_conserve_angular_momentum`, momentum-init masking (~L1024-1026)
- [`src/prolix/api/ensemble_plan.py`](../../src/prolix/api/ensemble_plan.py) — `_setup_integrator`, `integration_prefix is not None` branch (~L477-483)
- [`tests/physics/test_settle_batched.py`](../../tests/physics/test_settle_batched.py) — extend with masked-padding cases

## Result

Completed 2026-07-15. All 4 acceptance criteria met — verified via `tests/physics/test_settle_batched.py` (3 new tests: padded-vs-unpadded equivalence at gamma=0, heavier-padding finiteness at bucket=8, nested-vmap compile smoke) plus no regressions in the existing SETTLE/EnsemblePlan test suites, including a real `b1_init_exec.py --smoke --path inference` end-to-end run with SETTLE now active on the stacked water class.

One additional, previously-latent bug was found and fixed while wiring this through: `settle_langevin`'s `init_fn` had a `non_water_mask` branch using `if jnp.any(non_water_mask):` (Python control flow on a traced value) and boolean-mask indexing (dynamic shape) — both unsafe under `vmap`. This was never exercised before because `water_indices=None` always skipped it for the stacked path; wiring real water indices through made it live. Fixed with a fixed-shape `jnp.where` select instead.

## Explicit non-goals

Fixing the FF-parameter-source mismatch (`ff19SB` vs `amber14-all`); periodic/PME nonbonded (tracked separately, `B1-NONBONDED-PARITY`); re-running B1-full (tracked separately, gated on both leaves).

## Research note (xtrax composability)

`xtrax.stages`' `Fuse`/`Tap`/`Sink`/`AxisBoundary` are an I/O-pipeline-boundary abstraction for batched data/training pipelines — no call site in xtrax or prolix invokes them inside a physics step, and there is no precedent anywhere in `src/prolix/physics/*.py` for routing constraint math through them. SETTLE stays a plain composable function targeting the `xtrax.tiling` layer (`AxisSpec`/`BatchPlanner`/`make_axis_dispatch`) that `ensemble_dispatch.py`/`ensemble_dedup.py` already use — fixing the `water_indices=None` gap is sufficient; no new abstraction is needed.
