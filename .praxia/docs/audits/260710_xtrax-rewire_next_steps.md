# Next steps — after TRIAGE (B1-full locked)

**date:** 2026-07-10  
**task_id:** `260710_epic-audit_xtrax-rewire`  
**branch:** `audit/xtrax-rewire-xa`  
**closeout memo:** `.praxia/docs/research/260710_xtrax-rewire-epic-closeout-audit.md`  
**triage:** `.praxia/docs/audits/260710_xtrax-rewire_triage.md`  
**invariants:** `.praxia/loop_priorities.toml`  
**next epic:** `260528_b1-full`

## Where we are

| Leaf | Status | Gate |
|------|--------|------|
| XA-* audit leaves | **completed** | VERIFY PASS |
| XA-REHOME | **completed** | 591 passed |
| XA-NL-DEBT | **ready** | not on B1 critical path |
| **TRIAGE** | **completed** | next = B1-full |
| B1-LAND | **ready** | P0 push/PR (human) |
| B1-SMOKE / B1-FULL | **ready** | prereg cadence |

```mermaid
flowchart LR
  triageDone[TRIAGE done]
  land[B1-LAND]
  smoke[B1-SMOKE]
  full[B1-FULL]
  paper[Paper later]
  triageDone --> land
  land --> smoke
  smoke --> full
  full --> paper
```

## Immediate

1. **B1-LAND** — push/PR `audit/xtrax-rewire-xa` when human requests (no autonomous push/merge to main).
2. **B1-SMOKE** — B=4 mixed EnsemblePlan regression (prereg).
3. **B1-FULL** — B=64 Claim-1 campaign via `bth run` (prereg); do not use pytest for cluster runs.
4. Paper / HP4 remain deferred until B1-full numbers exist and branch is landed.

## Frozen invariants (pinned)

See `.praxia/loop_priorities.toml`: `default_ci`, no autonomous push/merge, EnsemblePlan dt=fs / gamma=ps⁻¹, vacuum γ policy, `exception_*` on bundle energy path.
