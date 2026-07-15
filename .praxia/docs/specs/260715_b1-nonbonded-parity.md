---
title: "B1-NONBONDED-PARITY"
backlog_id: B1-NONBONDED-PARITY
epic: 260528_b1-full
depends_on: [B1-SETTLE-STACK]
priority: P1
status: in_progress
challenge_verdict: pending
challenge_summary: "Periodic PME+NL nonbonded for B1 water class, replacing dense all-pairs vacuum; switch OpenMM B1 water baseline to PME to match instead of building a new reaction-field kernel."
---

# B1-NONBONDED-PARITY

## Goal

Close the water-class nonbonded mismatch between prolix and OpenMM in B1: OpenMM applies `CutoffPeriodic` (1.0nm, PBC) to the 4-water system; prolix runs dense O(N²) all-pairs vacuum (no cutoff, no box) via `energy_fn_from_bundle` → `single_padded_energy`. Protein classes (1AKE/1UBQ/2GB1) already match (`NoCutoff` vacuum both sides) and are untouched by this leaf.

## Locked decisions

| Topic | Lock |
|-------|------|
| Prolix nonbonded method | **PME**, not reaction-field — reuse the already-validated periodic path `prolix.physics.system.make_energy_fn(..., use_pbc=True, cutoff_distance=...)` + `neighbor_list.make_neighbor_list_fn` (validated in `tests/physics/test_protein_nl_explicit_parity.py`). The existing "fast" NL path (`single_padded_energy_nl_cvjp`/`fused_energy_and_forces_nl`) is **not used** — it is structurally vacuum-only (raw Euclidean `dr`, no `displacement_fn`/PBC honored in-kernel) and would need new reaction-field numerics to match OpenMM exactly. |
| OpenMM baseline | Water class `nonbondedMethod` changed from `CutoffPeriodic` to `PME` (`b1_init_exec.py:635`) — one-line, well-supported OpenMM change, avoids building new prolix numerics to match reaction-field specifically |
| Neighbor-list lifecycle | `.allocate()` is host-side, called once before the `while_loop`; `.update()` is jit-safe and carried inside `dispatch_n_steps_inference`'s loop state, cadence expressed as a `CarrySpec` (mirroring `step_carry.py:24-53`) — not rebuilt from scratch every step |
| B1 water bundle | `_four_water_bundle()` gains real box vectors matching OpenMM's `data/pdb/4water.pdb` `CRYST1` geometry; `boundary_condition` changes from `"free"` to periodic for this class only |
| Protein classes | Untouched — remain `NoCutoff`/vacuum on both sides |

## Acceptance Criteria

1. Prolix PME+NL energy/force agree with OpenMM-PME water at B1's actual system, within XR-PARITY-style tolerance (target: `|ΔE| ≤ 0.1 kcal/mol`, `force_rmse < 3.0 kcal/mol/Å`, matching `XR-PARITY-OMM-WATER`'s precedent tolerance).
2. `dispatch_n_steps_inference` with NL carry completes a smoke run (few hundred steps) without host-side stalls or `.allocate()` calls inside the compiled loop.
3. `b1_init_exec.py --smoke` green with the water class running periodic PME+NL end to end.
4. OpenMM water baseline still exits 0 under `PME` (was `CutoffPeriodic`).

## Implementation

- [`src/prolix/api/bundle_md.py`](../../src/prolix/api/bundle_md.py) — `energy_fn_from_bundle` periodic branch
- [`src/prolix/api/ensemble_dispatch.py`](../../src/prolix/api/ensemble_dispatch.py) — `dispatch_n_steps_inference` NL-carry restructure
- [`scripts/benchmarks/b1_init_exec.py`](../../scripts/benchmarks/b1_init_exec.py) — periodic water bundle (`_four_water_bundle`), OpenMM `nonbondedMethod` switch (~L635)
- [`tests/physics/test_protein_nl_explicit_parity.py`](../../tests/physics/test_protein_nl_explicit_parity.py) — extend with B1 water-system parity case

## Explicit non-goals

Reaction-field kernel matching OpenMM's exact `CutoffPeriodic` numerics; FF-parameter-source parity (`ff19SB`/classic-TIP3P vs `amber14-all`/TIP3P-FB — same residual `XR-PARITY-OMM-PROTEIN` already declared out of scope for cross-FF work); protein-class nonbonded changes (already matched); re-running B1-full (tracked separately, gated on both leaves).

## Research note (xtrax composability)

Same as `B1-SETTLE-STACK`: target `xtrax.tiling`'s `CarrySpec`/`AxisSpec` layer (already used by `step_carry.py`, `ensemble_dedup.py`), not `xtrax.stages`' `Fuse`/`Tap`/`Sink`/`AxisBoundary` (I/O-pipeline-boundary abstraction, no fit for an in-loop neighbor-list rebuild).
