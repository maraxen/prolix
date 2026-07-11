---
session_id: e7f1c1ef
topic: Prolix × xtrax 0.4 rewire epic: collapse dual tiling fork, adopt joint MemoryBudget / make_axis_dispatch / host bucketize, pin dependency policy, and bathos confirmatory parity vs OpenMM (kUPS/TorchMD where applicable). Paper Claim-1 benchmarking is sequenced AFTER this epic. Draft at .praxia/docs/specs/260709_xtrax-rewire-epic-DRAFT.md.
task_type: architectural
winner: Hybrid pin-first + water-parity-gated cut, strengthened: XR-PIN → XR-FIT-FLIP → XR-BUDGET (MemoryBudget under adapter) → XR-SHADOW (fitting↔MD decision equality) → XR-PARITY-OMM-WATER → XR-KILL-FORK; XR-DISPATCH on critical path before or with OMM-WATER; Sprint A A2/A3 parallel sibling gating only protein parity. StageBundle/Zarr/Carry/Dedup deferred.
created_at: 2026-07-09T18:51:06.742811+00:00
---

# Brainstorm: Prolix × xtrax 0.4 rewire epic: collapse dual tiling fork, adopt joint MemoryBudget / make_axis_dispatch / host bucketize, pin dependency policy, and bathos confirmatory parity vs OpenMM (kUPS/TorchMD where applicable). Paper Claim-1 benchmarking is sequenced AFTER this epic. Draft at .praxia/docs/specs/260709_xtrax-rewire-epic-DRAFT.md.

## Problem Frame
Frame (fixed vs negotiable):

FIXED:
- xtrax is the single tiling authority after this epic; prolix must not keep a competing BatchPlanner forever
- Paper Claim-1 / B1-full / HP4 §7.1 are OUT OF SCOPE (follow-on)
- Rewire must be falsifiable via bathos confirmatory parity vs OpenMM on at least water/TIP3P (protein parity may wait on Sprint A)
- Sprint A A2/A3 are physics, not tiling — must not be conflated, but protein OpenMM parity depends on them
- Dependency pin policy is required (no silent tracking of moving alpha without a contract)

NEGOTIABLE:
- One-cut kill-fork vs strangler (flip callers → delete fork last)
- Whether joint MemoryBudget is mandatory for all MD paths or only multi-axis/fitting
- Whether MolecularBundle becomes StageBundle (stretch)
- How much of Bucket/Carry/Dedup/Zarr lands in this epic vs later
- Exact parity matrix breadth (kUPS/TorchMD mandatory vs secondary)

Confirm this frame.

## Idea Pool
- [ai] Kill-fork-now (one-cut): Delete prolix BatchPlanner.plan greedy loop and adapter secondary demotion in one cut; re-export xtrax AxisSpec/BatchPlanner; point EnsembleMDPlanner + run/spec.py at same call. Mandatory: MemoryBudget joint, make_axis_dispatch all EnsemblePlan axes, pin xtrax>=0.4.0a5, host Bucket for variable-N. Optional: StageBundle, Zarr, Carry/Dedup. Sprint A parallel; hard gate only for protein OpenMM parity. Critical path: XR-PIN → XR-KILL-FORK → XR-BUDGET → XR-DISPATCH → XR-PARITY-OMM-WATER.
- [ai] Strangler-fig: Flip callers behind existing adapter first (fitting → plan_with_xtrax); adopt MemoryBudget inside adapter and delete secondary demotion only when estimates match; widen make_axis_dispatch; thin re-exports; delete prolix fork LAST after V-gates + OpenMM parity green. Same mandatory/optional split as kill-fork but XR-KILL-FORK is terminal not early. Watch ConcretizationTypeError on stacked paths. Sprint A parallel.
- [ai] Hybrid pin-first + water-parity-gated cut: XR-PIN + XR-FIT-FLIP + XR-BUDGET under adapter (strangler start), then hard cut XR-KILL-FORK only after XR-PARITY-OMM-WATER passes; protein parity + A2/A3 remain sibling. Differs from pure strangler by making water OpenMM parity the kill-date for the fork rather than V-gate green alone.
- [user] Competing approaches (no evaluation yet):
- [user] A) Kill-fork-now — one-cut delete of prolix greedy BatchPlanner + adapter secondary demotion; re-export xtrax types; critical path PIN→KILL→BUDGET→DISPATCH→OMM-WATER.
- [user] B) Strangler-fig — flip fitting to adapter, adopt MemoryBudget inside shim, widen dispatch, delete fork last after V-gates + OpenMM green; KILL is terminal.
- [user] C) Hybrid pin-first + water-parity-gated cut — strangler start (PIN/FIT-FLIP/BUDGET under adapter) but fork deletion is gated specifically on XR-PARITY-OMM-WATER pass, not merely V-gate green.
- [user] Probe: For approach A, what is the blast radius if EnsembleMDPlanner and fitting share one xtrax BatchPlanner call on day one — which existing tests/gates are most likely to break first? For B, what concrete kill-date metric ends the shim (not just "when green")? For C, is water OpenMM parity sufficient to prove the fork is dead, or can fitting still silently diverge after water passes?
- [ai] Shadow-mode variant: call both prolix greedy and xtrax planners in CI; assert decision equality on fixed fixtures before deleting greedy (complements hybrid).
- [critic] Forever-wrapper variant: keep prolix AxisSpec/budget metadata wrapper forever; only re-export xtrax strategies — rejects fixed constraint of single authority unless wrapper is identity-thin.
- [ai] StageBundle-first variant: promote MolecularBundle to xtrax StageBundle before planner collapse — higher blast radius, orthogonal to tiling authority.
- [user] Probe answers (still no evaluation):
- [user] A blast radius: first breakers are greedy-contract tests — test_batch_planner_v7 demotion order, test_xtrax_adapter dual-path, test_spec/test_batchplan_dispatch fitting .plan() — not V3–V5 trajectory gates (EnsembleMDPlanner already on adapter).
- [user] B kill-date metric: CI has zero hits for 'prolix budget demotion|greedy' under src/prolix/tiling AND fixture asserts make_fitting_planner decisions == EnsembleMDPlanner plan decisions (same axis names + batch_size).
- [user] C hybrid risk: YES — water OpenMM parity only checks MD observables; fitting can still use greedy while MD uses xtrax, so N_MOLS/N_CONFORMERS tiles diverge silently. Therefore any hybrid must also require XR-FIT-FLIP (or decision-equality shadow) before XR-KILL-FORK.
- [user] Additional variants recorded: shadow-mode CI equality; forever-wrapper (conflicts with fixed single-authority unless identity-thin); StageBundle-first (orthogonal, defer).
- [user] Both A and B/C approaches are steelmanned enough on migration order. Ready to converge — recommend advancing to Phase 3.

## Decision Log
- [ACCEPT] Hybrid strengthened vs Kill-fork-now vs pure Strangler: Hybrid wins: strangler start preserves reversibility through PIN/FIT-FLIP/BUDGET/SHADOW; water OpenMM + decision-equality + FIT-FLIP form an explicit kill-date that pure Strangler lacks; one-cut Kill-fork-now unnecessarily breaks V7/adapter/fitting tests on day one while EnsembleMDPlanner is already on the adapter.
- [DEFER] MolecularBundle as StageBundle in this epic: Orthogonal blast radius; StageBundle is Python-side optional slots, not tiling authority. Defer to follow-on.
- [DEFER] Zarr sinks / Carry / Dedup in this epic: No MD caller yet; traj integrity and Carry/Dedup are optional speedups after single authority lands.
- [REJECT] FIT-FLIP before BUDGET (original hybrid order): Adversarial critique: flipping fitting onto dual-demotion adapter before MemoryBudget bakes the fork into the second consumer. BUDGET must land first.
- [ACCEPT] Revised DAG: PIN→BUDGET→SHADOW→FIT-FLIP→WATER→KILL; DISPATCH after PIN into SHADOW/WATER; A2/A3→PROTEIN sibling: Fixes inverted strangler, makes DISPATCH a hard edge, keeps protein off fork-kill critical path.

## Assumptions

## TBDs

## Pre-mortem Record
**User:** Pre-mortem (6 months later, rewire "done" but failed):

We declared XR-KILL-FORK after OMM-WATER + shadow equality passed, but shadow only compared batch_size integers — not xtrax strategy objects — so fitting and MD both looked "equal" while still routing through a thin prolix wrapper that reintroduced secondary demotion under a renamed helper. Joint MemoryBudget was adopted with a hand-rolled estimate_memory that disagreed with lowered_memory_estimate; under hetero N_MOLS the plan silently demoted differently on cluster GPUs vs CI CPU, and OpenMM water gates (physics-invariant to tile size) never fired. Meanwhile Sprint A A2 never landed, so protein parity stayed "blocked" forever and the paper epic started on a half-rewired stack. V7 tests were deleted rather than rewritten to joint-budget semantics, so demotion regressions had no oracle. Failure mode: dual authority returned under new names; bathos claim concluded pass without strategy-object equality or XLA memory_analysis-backed estimates.
**AI:** _not recorded_

## Acceptance Criteria
**Given** Frame (fixed vs negotiable):

FIXED:
- xtrax is the single tiling authority after this epic; prolix must not keep a competing BatchPlanner forever
- Paper Claim-1 / B1-full / HP4 §7.1 are OUT OF SCOPE (follow-on)
- Rewire must be falsifiable via bathos confirmatory parity vs OpenMM on at least water/TIP3P (protein parity may wait on Sprint A)
- Sprint A A2/A3 are physics, not tiling — must not be conflated, but protein OpenMM parity depends on them
- Dependency pin policy is required (no silent tracking of moving alpha without a contract)

NEGOTIABLE:
- One-cut kill-fork vs strangler (flip callers → delete fork last)
- Whether joint MemoryBudget is mandatory for all MD paths or only multi-axis/fitting
- Whether MolecularBundle becomes StageBundle (stretch)
- How much of Bucket/Carry/Dedup/Zarr lands in this epic vs later
- Exact parity matrix breadth (kUPS/TorchMD mandatory vs secondary)

Confirm this frame.
**When** implementing Hybrid pin-first + water-parity-gated cut, strengthened: XR-PIN → XR-FIT-FLIP → XR-BUDGET (MemoryBudget under adapter) → XR-SHADOW (fitting↔MD decision equality) → XR-PARITY-OMM-WATER → XR-KILL-FORK; XR-DISPATCH on critical path before or with OMM-WATER; Sprint A A2/A3 parallel sibling gating only protein parity. StageBundle/Zarr/Carry/Dedup deferred.
**Then**
  - [ ] _add specific measurable criteria_
