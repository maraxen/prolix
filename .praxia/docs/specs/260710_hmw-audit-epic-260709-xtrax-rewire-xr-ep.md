---
session_id: f153ab15
topic: HMW: audit epic 260709_xtrax_rewire (XR-EPIC #3269) before next epic (paper/B1-full/HP4 triage) — closeout verification of uncommitted rewire + parity gates
task_type: constrained-technical
winner: 4-track audit DAG (a mechanical CI ∥ b bathos/ROLLUP sync → c drift: dt call-sites + yellow-gates → d lesson/debt + invariants.toml + closeout memo); hygiene P0 = reviewable commits OR open PR; REQUEST_CHANGES until CI green + hygiene + dt audit; AC5 sync failure → UNVERIFIED not hard-block; NL assert → debt; TorchMD stays out of CI
created_at: 2026-07-10T13:12:59.997850+00:00
---

# Brainstorm: HMW: audit epic 260709_xtrax_rewire (XR-EPIC #3269) before next epic (paper/B1-full/HP4 triage) — closeout verification of uncommitted rewire + parity gates

## Problem Frame
Frame: Closeout-audit the completed XR-EPIC before next-epic triage. Produce a GWT audit spec + sprint tracks that decide PASS vs REQUEST_CHANGES on evidence (CI, bathos sync, API drift, commit hygiene). Fixed: cited claims only; no push/merge to main; do not reopen XR leaves without hard regression; L2 stop before execute. Negotiable: commit/PR in-sprint vs follow-up; pytest depth (not slow vs full). Confirm.

## Idea Pool
- [ai] Track A first: run full default CI (pytest -m \"not slow\") as the hard gate; only then decide PASS vs REQUEST_CHANGES.
- [ai] Track B first: bth sync OMM-WATER + refresh ROLLUP before any CI spend — close the local evidence gap that undermines AC5 citation.
- [ai] Commit-hygiene as P0 audit child: atomic conventional commits or one PR branch before next epic; treat uncommitted main as REQUEST_CHANGES blocker regardless of green tests.
- [ai] Call-site audit for EnsemblePlan.run(dt=) / gamma unit change: rg + contract tests; changelog note; fail audit if silent AKMA callers remain.
- [ai] NL vs dense tiling assert parity as debt-not-blocker: document asymmetry; do not reopen XR-BUCKET unless silent-drop repro found.
- [ai] Yellow-gate / skip-xfail inventory on new XR tests: zero unexpected skip in default suite.
- [ai] Paper/B1/HP4 readiness checklist as audit output artifact (not implementation): AC8 lifted only if kill-fork + water claim sync + commit hygiene pass.
- [ai] Adversarial mini-pass on VACUUM-DT + exception_* energy path as frozen invariants for next epic.
- [ai] Split audit epic into 4 backlog children matching tracks a–d with depends_on: a∥b → c → d.
- [ai] Closeout memo path: .praxia/docs/research/260710_xtrax-rewire-epic-closeout-audit.md as VERIFY artifact.
- [ai] Defer full openmm/kups markers to advisory; keep always-on adapters in default CI.
- [ai] Add loop_priorities.toml [invariants] bootstrap (default_ci, no_autonomous_push) as audit track d deliverable.
- [ai] Staging: promote epic-audit templates already copied; debt for PCW workflow promotion.
- [ai] Reject \"declare VERIFY PASS on leaf rollup alone\" — working-tree-only completion is insufficient for inter-epic handoff.
- [ai] Reject reopening XR-PARITY-TORCH TorchMD in CI — keep out-of-CI; Scope A planner regression only.
- [user] PEGS map:
- [user] Processes: default CI, bathos sync, structural kill grep, call-site unit audit, commit/PR hygiene, ROLLUP refresh, lesson/debt filing, loop → TRIAGE.
- [user] Events: epic leaves completed; uncommitted dirty tree; OMM-WATER local outcome=unknown; VACUUM-DT unit change; PROT exception_* fix; bucket assert.
- [user] Goals: GWT audit spec; PASS vs REQUEST_CHANGES; sprint tracks a–d; blockers for paper/B1; L2 human gate.
- [user] States: HEAD 6b9e588 clean baseline; WT dirty; XR-EPIC completed in backlog; templates bootstrapped.
- [user] Ideas already recorded (15). More angles: (1) CI-first hard gate, (2) sync-first evidence, (3) commit as P0 blocker, (4) dt call-site audit, (5) NL assert as debt, (6) yellow-gates, (7) paper readiness checklist, (8) freeze VACUUM-DT+exception invariants, (9) 4-child DAG, (10) closeout memo path, (11) openmm/kups advisory, (12) invariants.toml bootstrap, (13) template/PCW debt, (14) reject leaf-rollup-only PASS, (15) keep TorchMD out of CI.
- [user] Novelty: also consider worktree/PR branch for review without committing to main yet — still satisfies hygiene if PR exists.
- [user] converge

## Decision Log
- [REJECT] Declare VERIFY PASS from leaf rollup alone without CI/commit/sync: Working-tree-only completion is insufficient for inter-epic handoff; AC5 citation and ship hygiene remain open.
- [REJECT] Reopen TorchMD in default CI: XR-PARITY-TORCH intentionally out of CI; Scope A planner regression is the always-on gate.
- [DEFER] NL tiling assert asymmetry as hard audit blocker: No silent-drop repro in audit research; document as debt; reopen XR-BUCKET only if repro found.
- [ACCEPT] Hygiene = commits on main OR open PR/worktree with reviewable delta; CI a∥ sync b then drift c then closeout d: Addresses cognitive-forcing risk: don't stall paper triage on main-only commits; PR satisfies reviewability. Parallel a∥b avoids false stall if sync unreachable (document UNVERIFIED).

## Assumptions

## TBDs

## Pre-mortem Record
**User:** Pre-mortem (6 mo failure): Audit declared VERIFY PASS after a green \"not slow\" subset and an open PR that never merged; paper epic cited Titanix water numbers while local catalog stayed unknown; a caller still passed dt in AKMA and silently ran 20× too hot; NL silent-drop resurfaced at 895 waters; loop_priorities invariants.toml was never written so the next autonomous loop skipped inter-epic audit. Failure modes: hygiene theater (PR without land), AC5 citation theater, unit-drift miss, deferred bucket bite, process debt unpaid.
**AI:** _not recorded_

## Acceptance Criteria
**Given** Frame: Closeout-audit the completed XR-EPIC before next-epic triage. Produce a GWT audit spec + sprint tracks that decide PASS vs REQUEST_CHANGES on evidence (CI, bathos sync, API drift, commit hygiene). Fixed: cited claims only; no push/merge to main; do not reopen XR leaves without hard regression; L2 stop before execute. Negotiable: commit/PR in-sprint vs follow-up; pytest depth (not slow vs full). Confirm.
**When** implementing 4-track audit DAG (a mechanical CI ∥ b bathos/ROLLUP sync → c drift: dt call-sites + yellow-gates → d lesson/debt + invariants.toml + closeout memo); hygiene P0 = reviewable commits OR open PR; REQUEST_CHANGES until CI green + hygiene + dt audit; AC5 sync failure → UNVERIFIED not hard-block; NL assert → debt; TorchMD stays out of CI
**Then**
  - [ ] _add specific measurable criteria_
