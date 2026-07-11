---
title: "DRAFT — Prolix × xtrax 0.4 rewire epic"
task_id: 260709_xtrax_rewire
date: 2026-07-09
status: superseded
brainstorm_session: e7f1c1ef
supersedes: 260614_xtrax-tiling-integration.md
superseded_by: 260709_xtrax-rewire-epic.md
---

# DRAFT Epic: Prolix × xtrax 0.4 Rewire + Parity Gates

> **SUPERSEDED** by `.praxia/docs/specs/260709_xtrax-rewire-epic.md` after
> contemplex session `e7f1c1ef` + adversarial critique. Kept for provenance.

## Problem (working)

Prolix's 2026-06-14 integration plan assumed xtrax had no joint budget loop and
deferred behavioral delegation. Reality overtook the plan: MD multi-bundle
already delegates via `xtrax_adapter`, but prolix still ships a full parallel
tiling fork (`AxisSpec`/`BatchPlanner`/`safe_map`), dual budget demotion, and
ignores xtrax 0.4 capabilities (joint `MemoryBudget`, `StageBundle`, Bucket/
Carry/Dedup, Zarr sinks). Main path-deps live `xtrax@0.4.0a5`; the cutover
worktree pins PyPI `0.3.0`. Claim-1 protein MD is still blocked on Sprint A
A2/A3 (exclusions + settle dt) — orthogonal but must not be conflated with
tiling rewire.

## Goals

1. **Single tiling authority**: xtrax owns strategy selection + joint budget;
   prolix keeps physics-domain axis registry and MD dispatch only.
2. **Pin & contract**: explicit xtrax version floor (path→release policy) so
   prolix does not silently track a moving alpha.
3. **Parity evidence (bathos)**: confirmatory campaigns proving prolix energy/
   force/trajectory observables match OpenMM (and where applicable kUPS /
   TorchMD) on fixed fixtures — *rewire correctness*, not paper throughput.
4. **Leave paper lane alone**: HP4/#259 and B1-full Claim-1 headline stay a
   follow-on epic after this DAG closes.

## Non-goals (this epic)

- §7.1 fitting figure / HP4 ANI-1x curation
- B1-full Claim-1 throughput headline vs OpenMM
- NPT long-traj / Phase 6–7 thermostat work
- Adopting xtrax `Trainer`/`Engine` for MD (MD is not supervised training)
- Full `StageBundle` migration of `MolecularBundle` (optional stretch only)

## Recon facts (load-bearing)

| Fact | Source |
|------|--------|
| xtrax `0.4.0a5` adds joint-budget `MemoryBudget`, Zarr sinks, StageBundle validator fixes | `~/projects/xtrax` CHANGELOG |
| Prolix adapter still uses per-axis `memory_estimator` + secondary greedy demotion | `src/prolix/tiling/xtrax_adapter.py` |
| EnsembleMDPlanner already calls xtrax; fitting `run/spec.py` on main still uses greedy `.plan()` | recon |
| Local fork: prolix `AxisSpec` has `axis_index`/`doc`; xtrax has `role`/`bucket_boundaries`/`dedup_eligible` | dual planners |
| OpenMM/TorchMD/DMFF bathos sidecars exist; kUPS has tests but no sidecar | `scripts/benchmarks/` |
| Sprint A A2/A3 still open on worktree — protein MD path | handoff 260706 |
| Spec `260614_…` status=draft and obsolete on "no budget loop" | docs |

## Candidate workstreams (pre-brainstorm — not ordered)

| ID | Theme | Notes |
|----|-------|-------|
| XR-PIN | Dependency policy: path vs PyPI floor; CI pin | Unblocks reproducible campaigns |
| XR-KILL-FORK | Collapse prolix BatchPlanner → thin re-exports / delete greedy path | Highest blast radius |
| XR-BUDGET | Adopt `MemoryBudget` + `lowered_memory_estimate`; delete secondary demotion | Replaces obsolete gap |
| XR-DISPATCH | `make_axis_dispatch` for all EnsemblePlan axes; retire prolix `safe_map` fork | |
| XR-BUCKET | Host `bucketize` for variable-N; tie to MolecularShapeSpec | Related to #746 |
| XR-SINK | Optional ZarrStagingSink for traj dumps + integrity digests | Nice-to-have |
| XR-PARITY-OMM | Bathos confirmatory: energy/force/T vs OpenMM on ala-dip + TIP3P | Claim-tier |
| XR-PARITY-KUPS | Optional thermostat crossval sidecar | |
| XR-PARITY-TORCH | Re-home external_baseline as rewire regression not paper gate | |
| XR-A2A3 | Finish Sprint A exclusions + settle-dt | **Prerequisite for protein parity**, may be sibling epic |

## Open design questions (for brainstorm)

1. Kill the fork in one cut vs strangler (adapter stays, greedy deleted last)?
2. Is joint-budget mandatory for MD, or only for fitting/hetero ensembles?
3. Should MolecularBundle become a `StageBundle` subclass, or stay domain-local?
4. Where does Sprint A (A2/A3) sit in the DAG — hard gate before parity, or parallel?
5. What is the minimum parity matrix that *falsifies* a bad rewire (kill condition)?

## Sequenced after this epic

- Paper benchmarking / Claim-1 B1-full comparison epic
- HP4 → §7.1 figure
