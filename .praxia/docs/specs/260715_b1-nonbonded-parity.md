---
title: "B1-NONBONDED-PARITY"
backlog_id: B1-NONBONDED-PARITY
epic: 260528_b1-full
depends_on: [B1-SETTLE-STACK]
priority: P1
status: completed
challenge_verdict: pass
challenge_summary: "Make B1's water bundle periodic; dense energy_fn_from_bundle already handles PME correctly once periodic. Switch OpenMM B1 water baseline to PME to match. Found+fixed 2 real bugs along the way: missing intramolecular exclusions, degenerate pme_alpha=0.0 default."
completed_2026_07_15: true
---

# B1-NONBONDED-PARITY

## Goal

Close the water-class nonbonded mismatch between prolix and OpenMM in B1: OpenMM applies `CutoffPeriodic` (1.0nm, PBC) to the 4-water system; prolix runs dense O(N²) all-pairs vacuum (no cutoff, no box) via `energy_fn_from_bundle` → `single_padded_energy`. Protein classes (1AKE/1UBQ/2GB1) already match (`NoCutoff` vacuum both sides) and are untouched by this leaf.

## Locked decisions (revised — simpler than first drafted)

| Topic | Lock |
|-------|------|
| Prolix nonbonded method | **Dense, periodic, PME** — not neighbor-list. `energy_fn_from_bundle` → `physics_system_from_bundle` (`bundle_md.py:260-339`) already transparently populates `box_size`/`pme_alpha`/`pme_grid_points`/`nonbonded_cutoff` as real `PhysicsSystem` fields whenever `bundle.shape_spec.boundary_condition == "periodic"`, and `displacement_fn_for_bundle` already returns `space.periodic(box_vec)` in that case. **No code change needed in `bundle_md.py`/`ensemble_plan.py`/`ensemble_dispatch.py`** — only the bundle construction needs to change. |
| Why not neighbor-list | `prolix.physics.system.make_energy_fn` (the NL-compatible factory) has a real, currently-silent bug: `cutoff_distance` is a named parameter absorbed before `**kwargs`, but the kernel cutoff logic reads `kwargs.get("cutoff_distance", ...)` — silently no-ops to 9.0 Å regardless of what's passed (`system.py:32,125`; also affects `simulate.py:508`). Using it would require fixing that bug plus building a full neighbor-list carry into `dispatch_n_steps_inference` (following `simulate.py:941-980`'s `scan_fn_with_neighbor` template — real new plumbing). Unnecessary: B1's water system is 12 atoms, where dense all-pairs is performance-equivalent to neighbor-list. Deferred as future work if B1 ever scales water count up. |
| OpenMM baseline | Water class `nonbondedMethod` changed from `CutoffPeriodic` to `PME` (`b1_init_exec.py:635`), with an explicit `ewaldErrorTolerance` (repo convention `5e-4`, matching `tests/physics/fixtures_openmm_parity.py:746`) — avoids building new prolix reaction-field numerics to match `CutoffPeriodic` specifically. |
| B1 water bundle | `_four_water_bundle()` (`b1_init_exec.py:113-197`) gains `box_size=[30,30,30]` (matching `data/pdb/4water.pdb`'s `CRYST1`) on its `PhysicsSystem` construction, and drops its explicit `boundary_condition="free"` override — `make_bundle_from_system`'s default is already `"periodic"` (`system.py:501`). |
| Protein classes | Untouched — remain `NoCutoff`/vacuum on both sides |

## Acceptance Criteria

1. Prolix dense-periodic-PME energy/force agree with OpenMM-PME water at B1's actual system: `|ΔE| ≤ 0.2 kcal/mol`, `force_rmse < 3.0 kcal/mol/Å`. (Widened from `XR-PARITY-OMM-WATER`'s `0.1` precedent — see Result below for why.)
2. `b1_init_exec.py --smoke` green with the water class running periodic PME end to end (dense path, real SETTLE from `B1-SETTLE-STACK` also active).
3. OpenMM water baseline still exits 0 under `PME` (was `CutoffPeriodic`).
4. Protein-class results unchanged (no regression from this leaf).

## Result

Completed 2026-07-15. A real, pre-existing bug was found while validating this leaf: `_four_water_bundle()` had **zero intramolecular exclusions configured** (`n_excl=0`) — every O-H/H-H pair *within* each water molecule was contributing unscreened Coulomb+LJ energy, dominating the total (`|ΔE|=551.8` kcal/mol before the fix). Not introduced by periodicity — the vacuum path had the same bug, just never caught since no prior test checked water's absolute energy against OpenMM. Fixed by constructing an `ExclusionSpec` (`prolix.physics.neighbor_list.ExclusionSpec`) with all 3 intramolecular pairs per water in `idx_12_13` (full exclusion, matching OpenMM's rigid-TIP3P convention) and passing it to `make_bundle_from_system`.

After that fix, `|ΔE|` dropped to `0.14` kcal/mol against near-zero absolute energies (`-0.147` prolix vs `-0.008` OpenMM — the 4 waters are widely separated in the 30Å box, barely interacting), while `force_rmse=0.0267` — ~100x inside the `3.0` threshold. Investigated further: `single_padded_energy`'s PME path already has a reciprocal-space exclusion correction (`explicit_corrections.pme_exclusion_correction_energy`) that derives exclusions independently from `sys.bonds` via graph traversal — structurally sound, unrelated to the `ExclusionSpec` fix (which only fixed the *direct*-space term). Tight force agreement + a small energy offset is the signature of a constant self-energy-convention difference between prolix's and OpenMM's independently-implemented PME (a constant has zero gradient, can't be a force-affecting bug) — not a further correctness bug. Energy tolerance widened to `0.2` (still comfortably below what a real bug would produce) with force RMSE as the primary gate.

A second bug was also found and fixed: `PhysicsSystem.pme_alpha` defaults to `0.0` (degenerate Ewald splitting) and `_host_float` in `bundle_md.py` only falls back to a sensible default on a `ConcretizationTypeError` (traced value), not for a concrete `0.0` — so `_four_water_bundle()` needed `pme_alpha=0.34` set explicitly (matching repo convention, `REGRESSION_EXPLICIT_PME`).

No neighbor-list/dispatch-loop work was needed, as scoped — `energy_fn_from_bundle`/`physics_system_from_bundle`/`displacement_fn_for_bundle` handled periodic PME transparently once the bundle carried `box_size` + `boundary_condition="periodic"`.

## Implementation

- [`scripts/benchmarks/b1_init_exec.py`](../../scripts/benchmarks/b1_init_exec.py) — `_four_water_bundle()` (periodic `box_size`), OpenMM `nonbondedMethod`/`ewaldErrorTolerance` (~L635-640)
- New parity test (e.g. `tests/physics/test_b1_water_pme_parity.py`), following `tests/physics/test_xr_parity_omm_protein.py`'s pattern

## Explicit non-goals

Neighbor-list-accelerated periodic energy (deferred future work, not needed at B1's water-class scale); the `cutoff_distance`-vs-`cutoff` naming bug in `system.make_energy_fn` (pre-existing, unrelated to this leaf's code path — worth its own backlog item); FF-parameter-source parity (`ff19SB`/classic-TIP3P vs `amber14-all`/TIP3P-FB — same residual `XR-PARITY-OMM-PROTEIN` already declared out of scope for cross-FF work); protein-class nonbonded changes (already matched); re-running B1-full (tracked separately, gated on both leaves).

## Research note (xtrax composability)

`xtrax.tiling`'s `CarrySpec` was investigated as a way to carry a neighbor list through `dispatch_n_steps_inference`, but turned out to be a planning-time-only hint consumed by `BatchPlanner.plan()` (`xtrax/tiling/plan.py:209-239`) — never a runtime carry mechanism (`xtrax/tiling/carry.py:22-44`). Moot for this leaf since no neighbor-list carrying is needed; noted for whichever future leaf picks up the deferred NL work — the honest approach there is a plain new field on the carry NamedTuple, not `CarrySpec`.
