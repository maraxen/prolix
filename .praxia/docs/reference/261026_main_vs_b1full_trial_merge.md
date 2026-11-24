# Trial merge: `b1/full` -> `main` (2026-10-26)

Done in a throwaway worktree (aborted and removed). `main` carries #34's rebased copy of earlier work, so the histories diverge.

**Result: 22 conflicts.**

| Group | Files |
| :--- | :--- |
| Docs/state | `.praxia/{daily.jsonl,docs/INDEX.md,plans.jsonl}`, `CLAUDE.md` |
| Build | `pyproject.toml` (1 hunk), `uv.lock` |
| Slurm (6) | `scripts/slurm/{b1_init_exec,bench_tip3p_dt_gamma_matrix,bench_tip3p_ke_compare_prolix,bench_tip3p_langevin_preemptable,p2b_slow_gate,p5_nvt_gate_diag}.slurm` |
| Source | `ensemble_dispatch` (2 hunks), `ensemble_planner` (1), `constraints` (1), `optimization` (9), `pme` (3), `settle` (1), `solvation` (2), `system` (10), `tiling/axis_dispatch` (add/add, 2), `tiling/xtrax_adapter` (1) |

## Recommendation

1. Do not merge `b1/full` into `main` directly. Merge `main` into `b1/full` on a dedicated branch and resolve there, so the PR to `main` is clean.
2. Source conflicts: `b1/full` is the superset for physics and tiling (Pallas/xtrax, Phase 1 leaves), so prefer its side. Re-apply only genuine `main`-only fixes.
3. Regenerate `uv.lock` (`uv lock`), and take the union of `.praxia` jsonl lines (append-only state).
4. After resolving, run the guardrails (`tests/test_import_architecture.py`, `tests/test_stablehlo_export_guardrails.py`) and the wide suite on titanix and engaging.
