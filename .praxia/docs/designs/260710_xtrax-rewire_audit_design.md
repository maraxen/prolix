# Staff design — XR-EPIC closeout audit

**task_id:** `260710_epic-audit_xtrax-rewire`  
**date:** 2026-07-10  
**spec:** `.praxia/docs/specs/260710_xtrax-rewire_audit_spec.md`  
**research:** `.praxia/docs/research/260710_xtrax-rewire_audit.md`  
**sprint TOML:** skipped (human request 2026-07-10)

## Audit scope

| In | Out |
|----|-----|
| Default CI (`pytest -m "not slow"`) | Re-implementing closed XR leaves |
| Bathos OMM-WATER sync / UNVERIFIED honesty | Push/merge to `main` by agent |
| Ship hygiene (commits **or** open PR) | Paper / B1-full / HP4 implementation |
| `dt`/`gamma` call-site drift | TorchMD back into CI |
| Closeout memo + `[invariants]` bootstrap | Sprint TOML / `dw_emit_sprint` |
| NL tiling assert → debt only | Full TIER3 multi-round agent duel (lightweight ACCEPT) |

## Subagent routing (when executing leaves — not emitted as sprint)

| Track | Backlog | Agent | Success metric |
|-------|---------|-------|----------------|
| a | XA-CI `#3289` | reviewer | `pytest -m "not slow"` exit 0; no unexpected XR skips |
| b | XA-SYNC `#3290` | librarian | local bathos `outcome=pass` **or** memo UNVERIFIED |
| P0 | XA-HYGIENE `#3291` | orchestrator | branch commits or open PR URL recorded |
| c | XA-DRIFT `#3292` | fixer (surgical) | call-site audit clean; invariants listed |
| d | XA-CLOSEOUT `#3293` | orchestrator | closeout memo + debt/lesson + loop → TRIAGE |

## Worktree safety

- Prefer execute on current dirty tree or a dedicated audit branch; do **not** force-push `main`.
- Hygiene leaf may create branch/PR; landing to `main` remains human-gated.
- No autonomous merge (`no_autonomous_push_or_merge_to_main`).

## Success metrics

1. AC1–AC8 from audit spec observable.
2. Verdict **VERIFY PASS** only if CI ∧ hygiene ∧ dt audit; else **REQUEST_CHANGES**.
3. Closeout at `.praxia/docs/research/260710_xtrax-rewire-epic-closeout-audit.md`.

## DAG

See `.praxia/docs/audits/260710_xtrax-rewire_audit_dag.md`.
