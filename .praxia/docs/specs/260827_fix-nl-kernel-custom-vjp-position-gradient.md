---
title: "NL-KERNEL-VJP-FIX — chunked_lj_energy_nl / chunked_coulomb_energy_nl position-gradient fix"
backlog_id: "4623"
praxia_id: 4623
epic: 260824_sprint_nb_parity
depends_on: []
blocks: ["4383"]
priority: P1
difficulty: extended
status: draft
challenge_verdict: pending
lesson_id: 428
---

# NL-KERNEL-VJP-FIX

## Goal
Fix `chunked_lj_energy_nl` and `chunked_coulomb_energy_nl`'s (`src/prolix/physics/optimization.py`)
`custom_vjp` backward passes, which currently return `jnp.zeros_like(r)` for the position
gradient unconditionally — silently breaking every `jax.grad`/autodiff force computation
that goes through the NL branch. Not a bathos campaign, not #4383's comparison wiring
(that's already done and blocked on this).

## Root cause (see lesson id 428 for full narrative)
`_chunked_lj_nl_bwd` (line ~182-196) and `_chunked_coulomb_nl_bwd` (line ~346-359) both
return `-g * jnp.zeros_like(r)` for the position argument. The justifying comment ("gradient
w.r.t. an integer index array is always zero") is true and correct for `excl_indices`/
`neighbor_idx` (the two genuinely non-differentiable integer arguments in the same tuple)
but does not apply to `r` — a real, differentiable float array that should carry the actual
analytical gradient. This looks like an incomplete stub: the **dense** siblings
(`chunked_lj_energy`, `chunked_coulomb_energy`, same file) already have correct analytical
backward passes for `r` (via `f_tile_grad`/`pair_vals`, computing real per-pair
`dE/dd * dr/d` forces) — the NL/tiled variants never received the equivalent treatment.

Confirmed mechanistically during #4383: finite-difference of `energy_fn_from_bundle`'s NL
branch on a real 7507-atom solvated 1VII system matches OpenMM gold forces almost exactly
(atom 6092: fd=`[-2032.14, 2310.29, -764.16]` vs gold=`[-2032.13, 2310.35, -764.23]`), while
`jax.grad` on the identical function returns `[3.76, 1.46, -13.84]` for the same atom — proof
the physics is right and only the custom_vjp is broken.

Undetected until now because the only prior `jax.grad`-through-NL test
(`compare_nl_two_particle` in `nonbonded_omm_parity/lj_oracle.py`) uses a two-particle
geometry placed inside the LJ switch window where the true force is coincidentally near
zero — "gradient always zero" produced RMSE=0.002 against a near-zero gold force, which
looked like a pass. Production dynamics uses FlashMD analytical forces
(`force_fn_from_bundle`/`single_padded_force`), not `jax.grad` through this path, so the bug
had no other surface.

## Locked decisions

| Topic | Lock |
|-------|------|
| Scope boundary | Fix **only** the position (`r`) gradient in both NL custom_vjp backward passes. Leave `charges`/`sigmas`/`epsilons`/`excl_scales` gradients zeroed — this matches the **already-established, unchanged convention** the dense siblings (`chunked_lj_energy`, `chunked_coulomb_energy`) use for the same arguments (both dense backward passes also zero these). Not a regression to introduce force-field-parameter-fitting gradients through the NL path in this fix — that is separate future scope if a real use case (e.g. NL-path parameter fitting) ever needs it, not implied or required by #4383. |
| Method | Reuse the exact analytical per-pair force formulas already implemented in the dense backward passes — `_chunked_lj_bwd`'s `pair_vals` (LJ: `dE_ds`-based, `optimization.py:100-107`) and `_chunked_coulomb_bwd`'s `pair_vals` (Coulomb: `dE_dd`-based including the `erf`/`derf` PME-damping terms, `optimization.py:277-285`) — adapted to `tile_reduction_nl`'s neighbor-list tile-callback interface (`(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j)`, `tiling.py:165-182`) instead of the dense variant's slice-based `(pos_i, pos_j, mask_i, mask_j, start_i, start_j)` with separate `idx_j = start_j + arange(...)` computation. The NL tile callback already receives `nb_idx_tile` directly (gathered neighbor indices) — no need to reconstruct `idx_j` from a start offset the way the dense case does. |
| Sign convention — **do not assume symmetry between the two kernels** | The two dense backward passes use **opposite** signs: `_chunked_lj_bwd` returns `-g * f_res[:N]`; `_chunked_coulomb_bwd` returns `g * f_res[:N]` (no negation). This is not necessarily a bug in either — each kernel's own `pair_vals` formula likely already encodes a complementary internal sign such that the composed result is correct — but it means the NL fix must NOT copy one kernel's exact backward structure onto the other by pattern-matching. Each of `_chunked_lj_nl_bwd` and `_chunked_coulomb_nl_bwd` must be independently verified against finite-difference (see Acceptance Criteria) — a wrong-sign gradient is a subtler, more dangerous bug than an all-zero one (it looks structurally correct and can pass shape/smoke checks while pointing every force backward). |
| Double-counting / pairing convention | Forward energy in both NL kernels uses `0.5 * jnp.sum(...)` over the tile iteration (since this codebase's neighbor lists are apparently mutual/symmetric — atom `a` in `b`'s list and `b` in `a`'s list — so each unique pair is visited from both sides, requiring the 0.5 correction for energy). The backward force-per-atom-`i` computation should **not** apply an extra 0.5 factor — matching the dense backward's own established convention (no 0.5 in `f_tile_grad`) — because `f_on_i = sum_j(force from neighbor j on i)` already gives the full (not halved) net force on atom `i` when accumulated across its own neighbor row, independent of the energy-sum's double-counting concern. |
| Validation gate 1 — targeted finite-difference | Before touching #4383's own test, validate the fix in isolation: extend/promote `scripts/explore/debug_4383_finite_diff.py` (currently an ephemeral diagnostic, not tracked per `ephemeral_scripts.md` convention since it produced a real finding — persist it) into a permanent regression test asserting `jax.grad` matches central finite-difference (`eps=1e-4` Å, as already used) within a tight relative tolerance on a real multi-atom system — not just the coincidental two-particle probe. Cover both kernels (LJ via a system with genuine LJ-dominant interactions, Coulomb via the same 1vii solvated system already available as vendored gold). |
| Validation gate 2 — #4383 integration test | Once gate 1 passes, re-run `tests/physics/test_1vii_nl_nonbonded_parity.py::test_force_parity` (AC3) unchanged — it should now pass under the existing `force_rmse_kcal_mol_A < 3.0` band without any changes to the test itself, since the bug was entirely in the kernel, not in #4383's comparison wiring. |
| Validation gate 3 — no regression | Re-run the existing two-particle water/flash probes (`nl_omm_parity.py --probe water`, `flash_omm_parity.py --probe water`) after the fix — they must continue to pass. Given their true force is near-zero, a passing result here is necessary but not sufficient (see lesson id 428's action item) — gates 1 and 2 are the real correctness evidence, this gate is only a non-regression check. |
| Known open issue — explicitly NOT fixed by this item | #4383's `test_nonbonded_energy_parity` (AC2) showed a residual `-156.65` kcal/mol energy delta even *after* the position-wrap fix, still outside the `±0.1` kcal/mol band. Energy is computed via the **forward** pass only and does not go through the custom_vjp backward at all — this fix does not, by construction, touch that computation path. Do not assume closing this backlog item also resolves AC2; if it doesn't, that residual needs its own follow-up investigation (separate from the gradient bug this item scopes). |
| Compute / locality | This is CPU-only JAX work (no OpenMM needed — gold is already vendored). Given this box's current memory pressure (confirmed OOM-kill during #4383's own local attempts), run all validation on Engaging (`cpu` or `pi_so3`/`mit_preemptable` preset, matching what #4383 used), not locally. |

## Scope
- Fix `_chunked_lj_nl_bwd` and `_chunked_coulomb_nl_bwd` (`src/prolix/physics/optimization.py`) to compute a real analytical position gradient, each independently derived and finite-difference-verified per the Locked decisions above.
- Promote `scripts/explore/debug_4383_finite_diff.py` to a permanent, tracked regression test (e.g. `tests/physics/test_nl_kernel_gradient_correctness.py`) asserting `jax.grad` vs finite-difference agreement on a real multi-atom system for both kernels.
- Re-run `tests/physics/test_1vii_nl_nonbonded_parity.py` (from #4383) and the existing water/flash probes as validation gates 2 and 3.
- Do **not** touch `charges`/`sigmas`/`epsilons`/`excl_scales` gradient handling (stays zeroed, matching the dense siblings' existing convention).
- Do **not** attempt to resolve the separate, still-open `-156.65` kcal/mol energy residual (AC2) — out of scope, flag as a follow-up if it doesn't resolve as a side effect.

## Acceptance Criteria
1. Given the fixed `_chunked_lj_nl_bwd`/`_chunked_coulomb_nl_bwd`, when `jax.grad` is evaluated on `energy_fn_from_bundle`'s NL branch at the same atoms already diagnosed (6092, 2147, 6521, 7397, and a broader sample), then it matches central finite-difference (`eps=1e-4` Å) within a tight relative tolerance (target: relative error `< 1e-3`, consistent with the project's `jax_enable_x64` precision lock).
2. Given the fix, when `tests/physics/test_1vii_nl_nonbonded_parity.py::test_force_parity` is re-run unchanged, then it passes (`force_rmse_kcal_mol_A < 3.0`) without any modification to the test itself.
3. Given the fix, when the existing water/flash two-particle probes are re-run, then they continue to pass (no regression).
4. Given a new permanent finite-difference-vs-`jax.grad` regression test on a real (non-two-particle) system, when added to the test suite, then it exercises genuinely non-trivial (non-near-zero) true forces for both LJ and Coulomb NL kernels.
5. Given scope review, when changes to `charges`/`sigmas`/`epsilons`/`excl_scales` gradient handling, the dense kernel variants, or an attempt to resolve the separate AC2 energy residual appear in the diff, then they are rejected as out of scope for this item.

## Non-goals
- Force-field-parameter-fitting gradients through the NL path (differentiating w.r.t. charges/sigmas/epsilons/excl_scales) — separate future scope.
- The dense kernel variants (`chunked_lj_energy`, `chunked_coulomb_energy`) — already correct, not touched.
- #4383's `-156.65` kcal/mol energy residual (AC2) — a separate, still-open issue this fix does not address.
- Any bathos campaign machinery — this is a kernel bugfix with regression tests, not a confirmatory campaign (that's #4384, downstream of both this fix and #4383).

## Rollback
Additive/isolated: revert the two backward-pass functions to their prior (broken)
zero-gradient form, drop the new permanent regression test. No other code depends on the
corrected gradient existing yet (nothing currently calls `jax.grad` through this path in
production — that's the whole reason the bug was latent) — reverting is safe and
regression-free, it just re-blocks #4383's AC3.

## References
- Lesson: id 428 (full root-cause narrative, mechanistic evidence)
- Backlog: #4623 (this item), blocks #4383
- Buggy code: `src/prolix/physics/optimization.py` — `_chunked_lj_nl_bwd` (~182-196), `_chunked_coulomb_nl_bwd` (~346-359)
- Correct dense templates: `src/prolix/physics/optimization.py` — `_chunked_lj_bwd` (~76-127), `_chunked_coulomb_bwd` (~254-299)
- Tile-reduction interface: `src/prolix/physics/tiling.py` — `tile_reduction` (~101), `tile_reduction_nl` (~165-182)
- Diagnostic to promote: `scripts/explore/debug_4383_finite_diff.py`
- Blocked consumer: `.praxia/docs/specs/260827_nb-parity-1vii-nl-switch-energy-compare.md` (#4383), `tests/physics/test_1vii_nl_nonbonded_parity.py`
- Coincidental-pass probe (do not trust as gradient-correctness evidence alone): `scripts/experiments/nonbonded_omm_parity/lj_oracle.py::compare_nl_two_particle`
