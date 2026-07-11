# Adversarial review — XR-EPIC closeout audit

**task_id:** `260710_epic-audit_xtrax-rewire`  
**date:** 2026-07-10  
**mode:** lightweight (human `spec_confirmed`; full TIER3 dual-agent duel deferred)  
**verdict:** **ACCEPT**

## Challenger objections

| # | Severity | Objection | Confidence |
|---|----------|-----------|------------|
| C1 | high | Hygiene = open PR without merge → “hygiene theater”; paper cites vapor | 0.9 |
| C2 | high | AC5 UNVERIFIED escape lets audit PASS without local water evidence | 0.85 |
| C3 | med | `not slow` may miss openmm/kups regressions that matter for paper | 0.7 |
| C4 | med | Skipping sprint TOML loses PCW emit/verify discipline | 0.75 |
| C5 | low | NL assert debt underweights #746 history (10^62 K blowup) | 0.65 |

## Defender rebuttals

| # | Rebuttal | Evidence |
|---|----------|----------|
| C1 | AC8 requires checklist callout; VERIFY PASS ≠ paper start without AC1∧AC3∧AC5; pre-mortem named this | audit spec AC8 + pre-mortem |
| C2 | Intentional: unreachable Titanix must not false-stall; UNVERIFIED is honest, not pass | Decision Log: sync hard-block rejected |
| C3 | Always-on adapters stay in default CI; openmm/kups remain advisory (TBD T2) | XR-PARITY-TORCH/KUPS leaf policy |
| C4 | Human explicitly skipped sprint TOML; backlog DAG + design doc retain executable queue | user 2026-07-10 |
| C5 | No silent-drop repro in Phase 0; debt + optional scale job; reopen only on repro | research memo risk 4 |

## Verdict rubric

- Blocking objections resolved or mitigated in ACs → **ACCEPT**
- Confidence ≥ 0.85 on residual risks with mitigations documented
- Do **not** ESCALATE; human already confirmed spec

## Residual watch items (for XA-CLOSEOUT)

1. If PR opened for hygiene, track land-to-main separately before paper numeric claims.
2. Prefer `bth sync` success; treat UNVERIFIED as paper citation ban for water gates.
3. File debt: praxia `insert_backlog` / `insert_staging` DB failure (blocks daemon queue).
