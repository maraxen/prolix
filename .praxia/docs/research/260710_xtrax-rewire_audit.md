# Epic closeout audit — research memo

**task_id:** `260710_epic-audit_xtrax-rewire`  
**closed_epic:** `#3269` / `XR-EPIC` — Prolix × xtrax 0.4 rewire (`260709_xtrax_rewire`)  
**next_epic:** TBD (paper / B1-full / HP4 per epic AC8; triage after audit)  
**date:** 2026-07-10  
**status:** Phase 0 research (cited claims only)

## Scope anchors

| Anchor | Evidence |
|--------|----------|
| Epic spec | `.praxia/docs/specs/260709_xtrax-rewire-epic.md` |
| Challenge rollup | `.praxia/docs/audits/xr_rewire_challenges/ROLLUP.md` |
| Backlog | `.praxia/backlog.jsonl` — XR-EPIC + 17 children `status=completed` |
| HEAD | `6b9e588` (pre-epic baseline on `main`) |
| Working tree | **~99 dirty paths** — epic code/docs **uncommitted** (`git status`) |

## Shipped vs claimed (epic ACs)

| Acceptance criterion | Status | Evidence |
|---------------------|--------|----------|
| AC1 Pin: `xtrax` floor; CI/lock agree | PASS | `uv run python scripts/ci/check_xtrax_pin.py` → OK `0.4.0a5`; `import xtrax` → `0.4.0a5`; `.github/workflows/ci.yml` + `pyproject.toml` modified |
| AC2 Adapter uses `MemoryBudget`; no host demotion loop | PASS | `src/prolix/tiling/xtrax_adapter.py` imports `MemoryBudget`; `rg` under `src/prolix/tiling` finds no `while.*estimate` / `secondary demotion` / `greedy` |
| AC3 Shadow: strategy type + `batch_size` + dispatch | PASS (leaf) | ROLLUP XR-SHADOW; tests under `tests/tiling/` / ensemble (subset green below) |
| AC4 Fitting does not call prolix greedy `.plan()` | PASS | XR-FIT-FLIP / XR-KILL-FORK: `BatchPlanner.plan` → `plan_with_xtrax` (`260709_xr-kill-fork.md`) |
| AC5 TIP3P OpenMM bathos confirmatory | **UNVERIFIED** (XA-SYNC 2026-07-10) | Challenge summary claims campaign `5ffe2644` pass (\|ΔE\|=0.040, force_rmse=0.011, mean_T=303.6 K). After `bth sync engaging` fail + catalog rsync: local SQL still only `cbea8064…` `outcome=unknown`. Do not cite as locally confirmed. |
| AC6 Post-kill: no competing greedy planner | PASS | Structural kill leaf complete; planner docstring authority = xtrax `MemoryBudget` |
| AC7 V7 rewritten to joint-budget | PASS | XR-KILL-FORK V7 contracts in spec; `tests/physics/test_batch_planner_v7.py` modified |
| AC8 Paper epic unblocked after AC6 | PASS (process) | Kill-fork challenge_summary: “Paper/B1-full AC8 gate lifted” |

### Leaf rollup (all completed 2026-07-09…10)

| Leaf | Status | Key evidence |
|------|--------|--------------|
| XR-PIN … XR-KILL-FORK, CARRY, DEDUP, SINK-XTC, A2A3 | completed | ROLLUP.md L9–20 |
| XR-VACUUM-DT | completed | `tests/api/test_xr_vacuum_dt.py`; dt=fs, γ=ps⁻¹ |
| XR-PARITY-OMM-PROTEIN | completed | `exception_*` in `bundle_md.py`; `tests/physics/test_xr_parity_omm_protein.py` |
| XR-PARITY-KUPS | completed | bathos campaign `991b1851` / run `35136cb4…` `outcome=pass`; earlier runs `error` |
| XR-PARITY-TORCH | completed | run `1d4c5cb0…` `outcome=pass`; TorchMD out of CI |
| XR-DISPATCH-MULTI | completed | `dispatch_vmap_safemap` + `dispatch_n_atoms`; `tests/api/test_xr_dispatch_multi.py` |
| XR-BUCKET (#746) | completed | `compute_dense_tiling_dims` + `tile_reduction` assert; `tests/physics/test_xr_bucket_tiling.py` |

## Regression status

**Default CI (from project docs / CLAUDE.md):** `uv run pytest -m "not slow"` (no `loop_priorities.toml` `[invariants]` present — **UNVERIFIED** formal default_ci).

**Audit smoke (2026-07-10):**
```text
uv run pytest -m "not slow and not openmm and not kups" \
  tests/tiling/test_xtrax_adapter.py \
  tests/api/test_ensemble_vmap_dispatch.py \
  tests/api/test_xr_dispatch_multi.py \
  tests/physics/test_xr_bucket_tiling.py \
  tests/tiling/test_xr_parity_torch_planner.py \
  tests/physics/test_xr_parity_kups_adapter.py -q
→ 36 passed in 1.97s
```

Full default suite **not yet run** in this audit session — track for Phase 5/6.

## Open risks for next epic

1. **Uncommitted epic delta on `main`** — ~99 paths dirty; no merge/PR. Risk: loss / unreviewable history. Mitigation: atomic commit series or PR before paper/B1 epic.
2. **OMM-WATER local bathos outcome `unknown`** — Titanix pass claimed; local catalog incomplete. Mitigation: `bth sync` + re-query before citing in paper.
3. **API unit change (`dt` = fs default)** — VACUUM-DT escape `dt_unit="akma"`. Risk: silent wrong timestep for callers assuming AKMA. Mitigation: changelog + contract tests already present; audit track should grep call sites.
4. **NL tiling assert asymmetry** — dense path asserts N%tile_size; NL path only N%inner_tile_size. Risk: residual silent drop. Mitigation: document in XR-BUCKET follow-up or debt.
5. **Cross-FF protein parity out of scope** — PROT used shared amber14; ff19SB≠amber14 remains. Mitigation: do not claim cross-FF in paper.
6. **ROLLUP header stale** — early “all 8 = not_ready” vs body “all closed”. Risk: triage confusion. Mitigation: refresh ROLLUP in audit sprint track b.

## Debt / lessons (candidates for Phase 8)

| Kind | Item | Source |
|------|------|--------|
| debt | Promote epic-audit skill → Praxia PCW template | skill header |
| debt | Commit / PR the XR working tree | git status |
| debt | Sync bathos OMM-WATER claim locally | catalog `unknown` |
| lesson | Exception 1-4 scales must live on EnsemblePlan energy path | PROT root cause |
| lesson | Vacuum unconstrained-H needs γ≥50 @ dt=0.5 fs (or dt≤0.1 @ γ=10) | VACUUM-DT |
| lesson | Bathos: campaign before run; `[experiment]` sidecar | KUPS early `error` runs |

## Verdict (Phase 0 only)

**REQUEST_CHANGES for “shipped to main”** — functional epic ACs appear met in the working tree and leaf gates, but **nothing is committed**, full default CI not re-run here, and OMM-WATER local catalog is incomplete.

Recommended audit sprint tracks (preview):
- **a** mechanical: full `pytest -m "not slow"` (+ yellow-gate policy if any)
- **b** research: sync bathos water; refresh ROLLUP; closeout memo
- **c** drift: VACUUM-DT call-site audit; NL tiling note; commit hygiene plan
- **d** lesson/debt capture + route `loop` → TRIAGE for next epic

## Citations index

- Epic ACs: `.praxia/docs/specs/260709_xtrax-rewire-epic.md` L32–41  
- ROLLUP: `.praxia/docs/audits/xr_rewire_challenges/ROLLUP.md`  
- Kill-fork AC8: `.praxia/docs/specs/260709_xr-kill-fork.md` L39  
- Pin script: `scripts/ci/check_xtrax_pin.py` (runtime OK)  
- Protein exception fix: `src/prolix/api/bundle_md.py` (modified)  
- Bucket helper: `src/prolix/physics/tiling.py` (modified)  
