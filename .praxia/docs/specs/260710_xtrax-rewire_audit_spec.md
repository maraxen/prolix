---
title: "Epic audit — XR-EPIC / 260709_xtrax_rewire closeout"
task_id: 260710_epic-audit_xtrax-rewire
date: 2026-07-10
status: confirmed
brainstorm_session: f153ab15
brainstorm_artifact: .praxia/docs/specs/260710_hmw-audit-epic-260709-xtrax-rewire-xr-ep.md
research_memo: .praxia/docs/research/260710_xtrax-rewire_audit.md
closed_epic: XR-EPIC / #3269 / 260709_xtrax_rewire
next_epic: TBD (paper / B1-full / HP4 triage after VERIFY PASS)
autonomy: L2
---

# Spec: Inter-epic audit of Prolix × xtrax 0.4 rewire

## Winner

**4-track audit DAG** with hygiene as P0:

| Track | Work | Parallelism |
|-------|------|-------------|
| **a** | Mechanical: `uv run pytest -m "not slow"` + yellow-gate / unexpected skip inventory on XR tests | ∥ **b** |
| **b** | Research: `bth sync` (or catalog pull) for OMM-WATER; refresh ROLLUP header; cite local outcome | ∥ **a** |
| **c** | Drift: `EnsemblePlan.run(dt=)` / `gamma=` call-site audit; freeze VACUUM-DT + `exception_*` as next-epic invariants | after a∥b |
| **d** | Closeout: lesson/debt; bootstrap `.praxia/loop_priorities.toml` `[invariants]`; write closeout memo; route loop → TRIAGE | after c |

**Hygiene P0:** reviewable ship state = commits on a branch **or** open PR/worktree with the XR delta. Uncommitted dirty `main` alone → **REQUEST_CHANGES** even if CI is green.

**Verdict rule:** REQUEST_CHANGES until (CI green ∧ hygiene ∧ dt call-site audit clean). AC5 sync failure → mark **UNVERIFIED**, do not hard-block. NL assert asymmetry → **debt**, not blocker. TorchMD stays out of CI.

## Acceptance Criteria (GWT)

| ID | Given | When | Then |
|----|-------|------|------|
| AC1 | Working tree with XR changes | `uv run pytest -m "not slow"` | exit 0; no unexpected skip/xfail on new XR always-on tests |
| AC2 | Bathos remotes configured | sync + SQL for `xr-parity-omm-water` | local row shows `outcome=pass` **or** memo marks AC5 **UNVERIFIED** with reason |
| AC3 | Dirty XR tree | audit hygiene check | commits on branch **or** open PR exists; else REQUEST_CHANGES |
| AC4 | `EnsemblePlan.run` / vacuum callers | call-site audit (`rg` + contract tests) | no silent AKMA `dt` assuming old units without `dt_unit="akma"`; gamma documented as ps⁻¹ |
| AC5 | Audit sprint complete | closeout memo written | `.praxia/docs/research/260710_xtrax-rewire-epic-closeout-audit.md` exists with cited PASS/REQUEST_CHANGES |
| AC6 | Process debt | track d | `.praxia/loop_priorities.toml` has `[invariants]` (`default_ci`, `no_autonomous_push_or_merge_to_main`) and epic-audit templates remain under `.praxia/templates/epic-audit/` |
| AC7 | NL vs dense tiling | track d debt | debt item filed; XR-BUCKET **not** reopened unless silent-drop repro |
| AC8 | Next epic | VERIFY PASS | paper/B1/HP4 checklist may start only if AC1∧AC3∧AC5; AC2 UNVERIFIED must be called out on checklist |

## Decision Log

| Option | Verdict | Rationale |
|--------|---------|-----------|
| VERIFY PASS from leaf rollup alone | reject | Working-tree completion ≠ inter-epic handoff |
| Reopen TorchMD in default CI | reject | Scope A planner regression is the always-on gate |
| NL assert asymmetry as hard blocker | defer | No repro; debt only |
| Hygiene = main commits only | reject | Stalls triage; PR/worktree is enough for reviewability |
| Hygiene = commits **or** open PR | accept | Cognitive-forcing fix |
| Sync-first sequential hard-block | reject | Unreachable Titanix falsely stalls audit |
| a∥b then c then d | accept | Parallel evidence + ordered drift/closeout |
| Commit/PR inside audit sprint vs follow-up | negotiable | Prefer in-sprint for AC3; human may defer land to main |

## Assumptions

| ID | Assumption | Risk if false |
|----|------------|---------------|
| A1 | `pytest -m "not slow"` is the project default CI | Wrong marker → false green; pin in `[invariants].default_ci` |
| A2 | Open PR without merge is acceptable hygiene for VERIFY PASS | Pre-mortem: PR never lands → paper cites vapor; mitigate via AC8 checklist note |
| A3 | OMM-WATER Titanix pass is recoverable via sync | If lost, AC5 stays UNVERIFIED forever → paper must not cite numeric gates |
| A4 | No silent-drop at production water counts after XR-BUCKET | Deferred NL assert bites later → debt + optional scale job |

## TBDs

| ID | Item | Owner |
|----|------|-------|
| T1 | Exact commit strategy (atomic series vs one PR) | human at sprint_approved |
| T2 | Whether to run `openmm`/`kups` markers in audit or leave advisory | track a planner |
| T3 | Next epic slug after TRIAGE | human |

## Pre-mortem

Audit declares VERIFY PASS after a green subset and an open PR that never merges; paper cites Titanix water while local catalog stays `unknown`; a caller still passes `dt` in AKMA and runs ~20× too hot; NL silent-drop returns at 895 waters; `loop_priorities.toml` invariants never written so the next loop skips inter-epic audit.

**Mitigations baked into ACs:** AC1 full default marker; AC2 UNVERIFIED honesty; AC3 hygiene; AC4 call-site audit; AC6 invariants; AC7 debt; AC8 checklist gates paper start.

## Out of scope

- Re-implementing closed XR leaves without hard regression
- Push/merge to `main` by the agent
- Paper / B1-full / HP4 implementation
- Promoting epic-audit to Praxia PCW (debt only)

## L2 gates

1. **`spec_confirmed`** — **confirmed 2026-07-10** by human.
2. **`sprint_approved`** — **waived 2026-07-10** (human: no sprint TOML; execute via XA-* backlog leaves).

## References

- Research: `.praxia/docs/research/260710_xtrax-rewire_audit.md`
- Epic: `.praxia/docs/specs/260709_xtrax-rewire-epic.md`
- Rollup: `.praxia/docs/audits/xr_rewire_challenges/ROLLUP.md`
- Brainstorm raw: `.praxia/docs/specs/260710_hmw-audit-epic-260709-xtrax-rewire-xr-ep.md`
