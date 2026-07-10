# Epic closeout audit — XR-EPIC / xtrax 0.4 rewire

**task_id:** `260710_epic-audit_xtrax-rewire`  
**closed_epic:** `#3269` / `XR-EPIC` — Prolix × xtrax 0.4 rewire (`260709_xtrax_rewire`)  
**audit_epic:** `XA-EPIC` (`260710_epic-audit_xtrax-rewire`)  
**next_epic:** TRIAGE (paper / B1-full / HP4 — human picks slug)  
**date:** 2026-07-10  
**branch:** `audit/xtrax-rewire-xa`  
**status:** VERIFY PASS

## Shipped vs claimed (inter-epic audit ACs)

| Acceptance criterion | Status | Evidence |
|---------------------|--------|----------|
| AC1 Default CI green | **PASS** | `uv run pytest -m "not (slow or integration or dynamics)"` → **570 passed**, 0 fail; commit `bf8e5f1`; junit `tmp/xa_ci_junit_final.xml` |
| AC2 OMM-WATER bathos | **PASS** (catalog `outcome=unknown` quirk) | Titanix run `dfa001bf`, campaign `944fef0b`, `gate_pass=1`, \|ΔE\|=0.0399, force_rmse=0.0108, mean_T=303.6 K; JSON `outputs/xr_parity_omm_tip3p_xa_sync.json` (XA-SYNC). Do not treat bathos `outcome` column alone as authority. |
| AC3 Hygiene | **PASS** | Commits on `audit/xtrax-rewire-xa` (atomic series through XA-DRIFT `586cf87`). Push/PR deferred by human — not a REQUEST_CHANGES under AC3 (commits-or-PR). |
| AC4 dt/gamma call-site | **PASS** | XA-DRIFT: EnsemblePlan.run sites fs-ok or intentional `dt_unit="akma"`; smoke diagnostics fs/γ conversion fixed (`586cf87`); freeze note in `.praxia/docs/audits/260710_xtrax-rewire_next_steps.md` |
| AC5 Closeout memo | **PASS** | This file |
| AC6 `loop_priorities.toml` | **PASS** | `.praxia/loop_priorities.toml` `[invariants]` + epic pointers; templates under `.praxia/templates/epic-audit/` |
| AC7 NL tiling debt | **PASS** (debt filed) | Backlog `XA-NL-DEBT`; XR-BUCKET **not** reopened |
| AC8 Next epic gate | **PASS** | AC1∧AC3∧AC5 hold → paper/B1/HP4 checklist may start; call out AC2 quirk + no merge to `main` yet |

## Regression status

```
uv run pytest -m "not (slow or integration or dynamics)"
→ 570 passed, 0 failures (2026-07-10, XA-CI / bf8e5f1)
```

Pinned in `.praxia/loop_priorities.toml` as `[invariants].default_ci`.

## Open risks for next epic

1. **Branch not merged to `main`** — paper must not cite vapor; land via PR before production claims.
2. **Bathos `outcome=unknown`** on gate_pass=1 runs — cite JSON / gate fields, not `outcome` alone.
3. **VACUUM-DT / exception_*** — frozen in `loop_priorities.toml`; do not invent new unit conventions without updating invariants + `tests/api/test_xr_vacuum_dt.py`.
4. **NL tiling assert asymmetry** — debt `XA-NL-DEBT`; reopen XR-BUCKET only on silent-drop repro.
5. **XA-CI deselections** — many modules still `@pytest.mark.slow` (heavy/OpenMM/long MD). Cheap API-drift re-admit tracked as `XA-REHOME` (not a VERIFY blocker).

## Debt / lessons

| Kind | Item |
|------|------|
| debt | `XA-NL-DEBT` — NL vs dense tile-size assert asymmetry |
| debt | `XA-REHOME` — re-admit cheap API-drift tests (replica-exchange, cell_list) |
| debt | Push/PR `audit/xtrax-rewire-xa` → `main` (human) |
| debt | Promote epic-audit skill → Praxia PCW (out of scope) |
| lesson | Exception 1-4 scales must live on EnsemblePlan energy path |
| lesson | Vacuum unconstrained-H needs γ≥50 @ dt=0.5 fs (or smaller dt @ γ=10) |
| lesson | GitHub CI marker is `not (slow or integration or dynamics)` — `openmm` alone does not deselect |
| lesson | Bathos: campaign before run; `[experiment]` sidecar; do not trust `outcome` alone when gate JSON says pass |

## Audit leaf rollup

| Leaf | Status | Gate |
|------|--------|------|
| XA-CI | completed | AC1 |
| XA-SYNC | completed | AC2 |
| XA-HYGIENE | completed | AC3 |
| XA-DRIFT | completed | AC4 |
| XA-CLOSEOUT | completed | AC5–AC8 |

## Verdict

**VERIFY PASS**

Blockers for REQUEST_CHANGES (CI green ∧ hygiene ∧ dt audit) are cleared. Next: human TRIAGE for paper / B1-full / HP4; execute `XA-REHOME` for cheap CI re-admits; do not reopen XR-BUCKET without silent-drop repro.

## Citations index

- Audit spec: `.praxia/docs/specs/260710_xtrax-rewire_audit_spec.md`
- Phase-0 research (superseded for verdict): `.praxia/docs/research/260710_xtrax-rewire_audit.md`
- Next steps / DRIFT freeze: `.praxia/docs/audits/260710_xtrax-rewire_next_steps.md`
- ROLLUP: `.praxia/docs/audits/xr_rewire_challenges/ROLLUP.md`
- Closed epic: `.praxia/docs/specs/260709_xtrax-rewire-epic.md`
