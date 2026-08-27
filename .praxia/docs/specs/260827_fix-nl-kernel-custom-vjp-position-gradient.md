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
challenge_verdict: round-2-revision-applied-pending-round-3
challenge_summary: "Round 1: another revision round needed, 2 blocking findings. (1) The dense-kernel 'Method' template is analytically incomplete for chunked_lj_energy_nl -- the dense chunked_lj_energy has no switch_width handling at all, so a literal port omits the LJ switch function's product-rule chain-rule term; #4383's own comparison always uses lj_switch_width=1.0, so this would produce a still-wrong (not zero, but incomplete) gradient that could coincidentally pass an under-specified regression test. Fixed: Method row now specifies the exact analytical formula (dS_dd, E_lj_unswitched, full switched per-pair force) and Validation gate 1 now requires explicit switch-window coverage in the promoted test, not just 'worst atoms by pre-fix deviation.' (2) Root Cause/Rollback's claim that 'nothing calls jax.grad through this path in production' is false -- src/prolix/simulate.py::run_simulation calls jax.grad through these exact kernels today when use_neighbor_list=True (FIRE dt_start sizing, simulate.py:647-653 -- the only jax.grad call site in the file), a real, documented, RECOMMENDED path (runbook, notebook, script). Fixed: corrected the narrative, added Validation gate 4 + AC6 requiring a new/extended test_robust_minimization.py case, corrected Rollback's safety claim.

**Round 2 correction (this revision): round 1's own revision fabricated two citations, both removed here.** (a) A 'steepest-descent preconditioner (simulate.py:177)' does not exist -- line 177 is `TrajectoryWriter.write`; grepping the file confirms `simulate.py:647-653` (FIRE dt_start sizing) is the *only* jax.grad/value_and_grad call site in `simulate.py`. (b) `test_robust_minimization.py` does **not** already contrast `use_neighbor_list=True` vs `False` -- every fixture in that file sets `use_pbc=False` and never sets `box`, and the only `use_neighbor_list=` occurrence in the file is `False` (line 133). Since the NL branch is gated on `spec.use_neighbor_list and spec.box is not None` (`simulate.py:517`), no existing test in that file can execute the NL+jax.grad path regardless of which boolean is passed -- Validation gate 4 and AC6 now require adding a genuinely new case (`use_neighbor_list=True` with a real `box` set) rather than citing a contrast that doesn't exist. Also fixed (significant): the NL tile-reduction interface's mask_j is already 2D unlike the dense template's 1D mask_j -- a verbatim port would silently broadcast wrong rather than error; now called out explicitly in Method. Dense-kernel math (LJ=force, Coulomb=dE/dr, explaining their sign difference), the no-extra-0.5-factor reasoning, and the sign-inconsistency finding were all independently re-derived/verified and confirmed accurate as originally written -- no change needed there."
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

Undetected until now because the only prior *test* exercising `jax.grad`-through-NL
(`compare_nl_two_particle` in `nonbonded_omm_parity/lj_oracle.py`) uses a two-particle
geometry placed inside the LJ switch window where the true force is coincidentally near
zero — "gradient always zero" produced RMSE=0.002 against a near-zero gold force, which
looked like a pass.

**Correction (round-1 adversarial review, finding 6): this is NOT dead/unused in production
the way originally assumed.** `src/prolix/simulate.py::run_simulation` (the top-level
`PhysicsSpec` entry point — distinct from `prolix.physics.simulate.run_simulation`) builds
its energy function via `physics_system.make_energy_fn(...)` (`system.py:191-236`, which
routes `neighbor is not None` directly into these exact kernels, `system.py:194,221,229`),
and calls `jax.grad` through it when `spec.use_neighbor_list=True` to size FIRE's `dt_start`
(`simulate.py:647-653` — the only `jax.grad`/`value_and_grad` call site anywhere in
`simulate.py`, confirmed by grep). This is a documented, **recommended** path — the
explicit-solvent runbook (`.praxia/docs/reference/260415_explicit-solvent-runbook.md:25`)
recommends `use_neighbor_list=True`, and `notebooks/explicit_solvent_colab.ipynb:181` and
`scripts/simulate_explicit_rust.py:278` both use it. In other words: this bug has likely been
silently degrading real minimization runs on that path today (FIRE's `dt_start` is sized off
a near-zero `initial_grads` from the broken NL custom_vjp, pinning it near its ceiling
regardless of the true initial force magnitude) rather than having "no other surface." See
the added Validation gate 4 below.

## Locked decisions

| Topic | Lock |
|-------|------|
| Scope boundary | Fix **only** the position (`r`) gradient in both NL custom_vjp backward passes. Leave `charges`/`sigmas`/`epsilons`/`excl_scales` gradients zeroed — this matches the **already-established, unchanged convention** the dense siblings (`chunked_lj_energy`, `chunked_coulomb_energy`) use for the same arguments (both dense backward passes also zero these). Not a regression to introduce force-field-parameter-fitting gradients through the NL path in this fix — that is separate future scope if a real use case (e.g. NL-path parameter fitting) ever needs it, not implied or required by #4383. |
| Method | Reuse the exact analytical per-pair force formulas already implemented in the dense backward passes — `_chunked_lj_bwd`'s `pair_vals` (LJ, `optimization.py:100-107`) and `_chunked_coulomb_bwd`'s `pair_vals` (Coulomb: `dE_dd`-based including the `erf`/`derf` PME-damping terms, `optimization.py:277-285`) — adapted to `tile_reduction_nl`'s neighbor-list tile-callback interface (`(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j)`, `tiling.py:165-182`) instead of the dense variant's slice-based `(pos_i, pos_j, mask_i, mask_j, start_i, start_j)` with separate `idx_j = start_j + arange(...)` computation. The NL tile callback already receives `nb_idx_tile` directly (gathered neighbor indices) — no need to reconstruct `idx_j` from a start offset the way the dense case does. Round-1 adversarial review (see References) confirmed both `pair_vals` formulas are mathematically correct derivatives (LJ's returns true force; Coulomb's returns `∂E/∂r` directly — different quantities, which is *why* their outer sign differs, see the Sign convention row) — verified by hand-derivation, not just "assumed correct because not the buggy kernel." **Two adaptations are required beyond a mechanical port, both confirmed by that review:** |
| Method — mask shape trap (**new, round-1 finding 2, significant**) | `tile_reduction_nl`'s `mask_j` is already **2D** (`(inner_tile_size, tile_size)`, `tiling.py:242`) by the time it reaches `f_tile`/`f_tile_grad` — unlike the dense template's 1D `mask_j`, which the dense backward broadcasts via `mask_j[None, :] & mask_i[:, None]` (`optimization.py:116`). Porting that broadcast pattern verbatim into the NL backward will **not raise a shape error** — it will silently produce a wrong-shaped mask (`(1, I, T)` instead of `(I, T)`) that JAX broadcasts without complaint. The NL backward's mask construction must use `mask_j` as-is (already `(inner_tile_size, tile_size)`), matching how the existing NL *forward* kernels already correctly consume it (`optimization.py:170`, `optimization.py:335`) — do not reintroduce a `[None, :]` expansion. |
| Method — LJ switch-function chain rule (**new, round-1 finding 7b, blocking**) | `chunked_lj_energy_nl`'s forward multiplies the LJ energy by `switch = lj_switch_weight(ds, cutoff, sw)` (`optimization.py:172-174`) — a smooth, genuinely `d`-dependent factor (`lj_switching.py`: `S(z) = 1 − 10z³ + 15z⁴ − 6z⁵`, `z = (r − r_switch)/width`, nonzero `dS/dz` inside the switch window). **The dense `chunked_lj_energy` has no `switch_width` parameter at all** — its `pair_vals` template has zero chain-rule handling for this factor. A literal port produces an analytically **incomplete** gradient (missing a product-rule term) whenever `switch_width > 0` — which #4383's own comparison always uses (`lj_switch_width=1.0`), since OpenMM's Reference platform gold applies the same switching function by construction. This is not optional to handle: the fixed `_chunked_lj_nl_bwd` must compute, per pair inside the switch window (`r_switch < d < r_cut`; 0 outside it): <br>`dS_dd = (-30·z² + 60·z³ - 30·z⁴) / width` (the derivative of `S(z)` above, chain-ruled through `z = (d - r_switch)/width`)<br>`E_lj_unswitched = 4·eps·(inv_r6² − inv_r6)` (the un-switched LJ pair energy — an intermediate `pair_vals` must compute internally but does not currently return)<br>Full switched per-pair force: `F_pair = S(d)·F_pair_unswitched − E_lj_unswitched·dS_dd·(dr/d)`, where `F_pair_unswitched = f_mag·dr` is exactly what the existing dense `pair_vals` already computes. An incomplete fix could pass the promoted regression test **by coincidence** if no sampled pair happens to sit inside the switch window — the same "coincidental pass" failure mode this spec already flags for the two-particle probe (see Validation gate 1/promoted test requirement below, tightened accordingly). |
| Sign convention — **do not assume symmetry between the two kernels** | The two dense backward passes use **opposite** signs: `_chunked_lj_bwd` returns `-g * f_res[:N]`; `_chunked_coulomb_bwd` returns `g * f_res[:N]` (no negation). This is not necessarily a bug in either — each kernel's own `pair_vals` formula likely already encodes a complementary internal sign such that the composed result is correct — but it means the NL fix must NOT copy one kernel's exact backward structure onto the other by pattern-matching. Each of `_chunked_lj_nl_bwd` and `_chunked_coulomb_nl_bwd` must be independently verified against finite-difference (see Acceptance Criteria) — a wrong-sign gradient is a subtler, more dangerous bug than an all-zero one (it looks structurally correct and can pass shape/smoke checks while pointing every force backward). |
| Double-counting / pairing convention | Forward energy in both NL kernels uses `0.5 * jnp.sum(...)` over the tile iteration (since this codebase's neighbor lists are apparently mutual/symmetric — atom `a` in `b`'s list and `b` in `a`'s list — so each unique pair is visited from both sides, requiring the 0.5 correction for energy). The backward force-per-atom-`i` computation should **not** apply an extra 0.5 factor — matching the dense backward's own established convention (no 0.5 in `f_tile_grad`) — because `f_on_i = sum_j(force from neighbor j on i)` already gives the full (not halved) net force on atom `i` when accumulated across its own neighbor row, independent of the energy-sum's double-counting concern. |
| Validation gate 1 — targeted finite-difference (**tightened, round-1 finding 5**) | Before touching #4383's own test, validate the fix in isolation: extend/promote `scripts/explore/debug_4383_finite_diff.py` (currently an ephemeral diagnostic, not tracked per `ephemeral_scripts.md` convention since it produced a real finding — persist it) into a permanent regression test asserting `jax.grad` matches central finite-difference (`eps=1e-4` Å, as already used) within a tight relative tolerance (`< 1e-3`) on a real multi-atom system — not just the coincidental two-particle probe. **Must explicitly guarantee coverage of the LJ switch window**, not just pick "worst atoms by current (buggy) deviation" the way the ephemeral script currently does (round-1 review's finding 5: that selection criterion has no tie to switch-window coverage at all) — select or verify at least one tested pair whose separation genuinely falls inside `[cutoff − switch_width, cutoff]` (9.0 − 1.0 = 8.0 to 9.0 Å for #4383's 1vii system) before trusting a pass; a fix that's still missing the switch-function chain-rule term (see Method row above) could otherwise pass this gate by the same "coincidental pass" mechanism the two-particle probe already demonstrated. The existing 1vii gold/bundle is sufficient for both LJ and Coulomb coverage — no separate system needed, provided switch-window coverage is explicitly checked, not assumed. |
| Validation gate 2 — #4383 integration test | Once gate 1 passes, re-run `tests/physics/test_1vii_nl_nonbonded_parity.py::test_force_parity` (AC3) unchanged — it should now pass under the existing `force_rmse_kcal_mol_A < 3.0` band without any changes to the test itself, since the bug was entirely in the kernel, not in #4383's comparison wiring. |
| Validation gate 3 — no regression (two-particle probes) | Re-run the existing two-particle water/flash probes (`nl_omm_parity.py --probe water`, `flash_omm_parity.py --probe water`) after the fix — they must continue to pass. Given their true force is near-zero, a passing result here is necessary but not sufficient (see lesson id 428's action item) — gates 1 and 2 are the real correctness evidence, this gate is only a non-regression check. |
| Validation gate 4 — `run_simulation`/`use_neighbor_list=True` (**new, round-1 finding 6; corrected round 2**) | `src/prolix/simulate.py::run_simulation` calls `jax.grad` through these exact kernels today when `spec.use_neighbor_list=True` (FIRE `dt_start` sizing, `simulate.py:647-653` — the only `jax.grad`/`value_and_grad` call site in the file) — a real, documented, **recommended** path (explicit-solvent runbook, a notebook, and a script all use it), not dead code. **`tests/physics/test_robust_minimization.py` does not currently exercise this path at all**: every existing fixture sets `use_pbc=False` and never sets `box`, and the file's one `use_neighbor_list=` occurrence is `False` (line 133); since the NL branch only builds when `spec.use_neighbor_list and spec.box is not None` (`simulate.py:517`), no existing case in that file reaches the NL+jax.grad code path regardless of the flag. This gate therefore requires **adding a new test case** — a solvated (or otherwise periodic) system with `use_pbc=True`, a real `box`, and `use_neighbor_list=True` — run before and after the fix. Before the fix, expect `dt_start` pinned near its ceiling (symptom of the broken near-zero `initial_grads`); after the fix, confirm a genuine improvement (energy actually decreases during minimization, `dt_start` reflects the true initial force magnitude rather than sitting at its ceiling) rather than a new failure mode. |
| Known open issue — explicitly NOT fixed by this item | #4383's `test_nonbonded_energy_parity` (AC2) showed a residual `-156.65` kcal/mol energy delta even *after* the position-wrap fix, still outside the `±0.1` kcal/mol band. Energy is computed via the **forward** pass only and does not go through the custom_vjp backward at all — this fix does not, by construction, touch that computation path. Do not assume closing this backlog item also resolves AC2; if it doesn't, that residual needs its own follow-up investigation (separate from the gradient bug this item scopes). |
| Compute / locality | This is CPU-only JAX work (no OpenMM needed — gold is already vendored). Given this box's current memory pressure (confirmed OOM-kill during #4383's own local attempts), run all validation on Engaging (`cpu` or `pi_so3`/`mit_preemptable` preset, matching what #4383 used), not locally. |

## Scope
- Fix `_chunked_lj_nl_bwd` and `_chunked_coulomb_nl_bwd` (`src/prolix/physics/optimization.py`) to compute a real analytical position gradient, each independently derived and finite-difference-verified per the Locked decisions above.
- Promote `scripts/explore/debug_4383_finite_diff.py` to a permanent, tracked regression test (e.g. `tests/physics/test_nl_kernel_gradient_correctness.py`) asserting `jax.grad` vs finite-difference agreement on a real multi-atom system for both kernels.
- Re-run `tests/physics/test_1vii_nl_nonbonded_parity.py` (from #4383) and the existing water/flash probes as validation gates 2 and 3; add a new `use_neighbor_list=True` + real-`box` case to `tests/physics/test_robust_minimization.py` (no existing case reaches the NL+jax.grad path) and run it before/after as validation gate 4.
- Do **not** touch `charges`/`sigmas`/`epsilons`/`excl_scales` gradient handling (stays zeroed, matching the dense siblings' existing convention).
- Do **not** attempt to resolve the separate, still-open `-156.65` kcal/mol energy residual (AC2) — out of scope, flag as a follow-up if it doesn't resolve as a side effect.

## Acceptance Criteria
1. Given the fixed `_chunked_lj_nl_bwd`/`_chunked_coulomb_nl_bwd`, when `jax.grad` is evaluated on `energy_fn_from_bundle`'s NL branch at the same atoms already diagnosed (6092, 2147, 6521, 7397, and a broader sample), then it matches central finite-difference (`eps=1e-4` Å) within a tight relative tolerance (target: relative error `< 1e-3`, consistent with the project's `jax_enable_x64` precision lock).
2. Given the fix, when `tests/physics/test_1vii_nl_nonbonded_parity.py::test_force_parity` is re-run unchanged, then it passes (`force_rmse_kcal_mol_A < 3.0`) without any modification to the test itself.
3. Given the fix, when the existing water/flash two-particle probes are re-run, then they continue to pass (no regression).
4. Given a new permanent finite-difference-vs-`jax.grad` regression test on a real (non-two-particle) system, when added to the test suite, then it exercises genuinely non-trivial (non-near-zero) true forces for both LJ and Coulomb NL kernels.
5. Given scope review, when changes to `charges`/`sigmas`/`epsilons`/`excl_scales` gradient handling, the dense kernel variants, or an attempt to resolve the separate AC2 energy residual appear in the diff, then they are rejected as out of scope for this item.
6. Given a new `use_neighbor_list=True` + real-`box` case added to `tests/physics/test_robust_minimization.py` (no existing case in that file reaches the NL+jax.grad path — see Validation gate 4), when it is run before and after the fix, then post-fix minimization genuinely progresses (energy decreases, `dt_start` reflects the true initial force magnitude rather than sitting pinned at its ceiling) versus the pre-fix near-zero-force near-no-op.
7. Given the promoted finite-difference regression test (Validation gate 1), when its sampled atoms/pairs are inspected, then at least one genuinely falls inside the LJ switch window (`cutoff − switch_width` to `cutoff`) — not merely "worst atoms by pre-fix deviation."

## Non-goals
- Force-field-parameter-fitting gradients through the NL path (differentiating w.r.t. charges/sigmas/epsilons/excl_scales) — separate future scope.
- The dense kernel variants (`chunked_lj_energy`, `chunked_coulomb_energy`) — already correct, not touched.
- #4383's `-156.65` kcal/mol energy residual (AC2) — a separate, still-open issue this fix does not address.
- Any bathos campaign machinery — this is a kernel bugfix with regression tests, not a confirmatory campaign (that's #4384, downstream of both this fix and #4383).

## Rollback
Additive/isolated at the code level: revert the two backward-pass functions to their prior
(broken) zero-gradient form, drop the new permanent regression test. **Correction (round-1
review, finding 6): this is not consequence-free the way originally claimed** —
`src/prolix/simulate.py::run_simulation` with `use_neighbor_list=True` genuinely depends on
`jax.grad` through this path today (see Validation gate 4); reverting re-introduces that
path's `dt_start` sizing off a broken near-zero `initial_grads`, not just re-blocking
#4383's AC3. Reverting is still low-risk (returns to the exact status quo this session found
the codebase in) but is not "regression-free" in the sense of "nothing was relying on the
fix" — something already was, silently.

## References
- Lesson: id 428 (full root-cause narrative, mechanistic evidence)
- Backlog: #4623 (this item), blocks #4383
- Buggy code: `src/prolix/physics/optimization.py` — `_chunked_lj_nl_bwd` (~182-196), `_chunked_coulomb_nl_bwd` (~346-359)
- Correct dense templates (math independently hand-verified round 1): `src/prolix/physics/optimization.py` — `_chunked_lj_bwd` (~76-127), `_chunked_coulomb_bwd` (~254-299)
- LJ switch function (missing chain-rule term, round-1 finding 7b): `src/prolix/physics/lj_switching.py::lj_switch_weight`
- Tile-reduction interface (2D `mask_j` in NL path, round-1 finding 2): `src/prolix/physics/tiling.py` — `tile_reduction` (~101), `tile_reduction_nl` (~165-244)
- Production consumer requiring Validation gate 4 (round-1 finding 6; citation corrected round 2): `src/prolix/simulate.py::run_simulation` (`jax.grad` call site at ~647-653 — confirmed the only one in the file), `src/prolix/physics/system.py::make_energy_fn` (~191-236), `tests/physics/test_robust_minimization.py` (currently has no `use_neighbor_list=True`/`box`-set case — new case required, not an existing contrast)
- Documented recommendation of the now-relevant path: `.praxia/docs/reference/260415_explicit-solvent-runbook.md`, `notebooks/explicit_solvent_colab.ipynb`, `scripts/simulate_explicit_rust.py`
- Diagnostic to promote: `scripts/explore/debug_4383_finite_diff.py`
- Blocked consumer: `.praxia/docs/specs/260827_nb-parity-1vii-nl-switch-energy-compare.md` (#4383), `tests/physics/test_1vii_nl_nonbonded_parity.py`
- Coincidental-pass probe (do not trust as gradient-correctness evidence alone): `scripts/experiments/nonbonded_omm_parity/lj_oracle.py::compare_nl_two_particle`
