---
task_id: 260527_p2b-nonbonded-comparator
sprint_id: 3
date: 260527
status: complete
verdict: PASS (auditor)
---

# Sprint 3 / P2b Handoff: OpenMM Parity Harness (Nonbonded)

## What Landed

### Files created/extended
| File | Action | Tasks |
|---|---|---|
| `tests/physics/fixtures_openmm_parity.py` | +312 lines (append-only) | f1+f2+f3+fix |
| `tests/physics/test_openmm_parity_nonbonded.py` | new, 214 lines | f4+f5 |
| `.praxia/docs/audits/260526_p2a-bonded-field-audit.md` | extended | f6 |
| `.praxia/docs/INDEX.md` | 1-line update | f6 |

### Gates achieved

| Gate | Result | Observed |
|---|---|---|
| `\|dE(LJ)\| < 1.0 kcal/mol` | **PASS** | ~1e-6 kcal/mol |
| `\|dE(Coul)\| < 1.0 kcal/mol` | **PASS** | ~3e-6 kcal/mol |
| `\|dE(exc14)\| < 0.2 kcal/mol` (self-consistency) | **PASS** | ~0 (machine-epsilon) |
| Force RMS < 0.5 kcal/mol/Å | **PASS** | 5e-6 kcal/mol/Å |
| P2a regression (5/5 PASS) | **PASS** | 5/5 |
| Audit doc f6 gate | **PASS** | all 6 fields present |
| Auditor verdict | **PASS** | minor doc finding applied |

## Key Implementation Finding (Non-trivial)

**Root cause of exclusion double-counting** (discovered during implementation):

`prolix.physics.neighbor_list.map_exclusions_to_dense_padded` only processes `ExclusionSpec.idx_12_13` and `idx_14` when building the dense exclusion mask. It does NOT process `exception_pairs`. This means 1-4 exception pairs are included in BOTH:
1. The main pairwise sum (`chunked_lj_energy`, `chunked_coulomb_energy`) — because they're not in the exclusion mask
2. The exception energy function (`make_exception_pair_energy_fn`) — because they're in `exception_pairs`

**Fix:** In `build_exclusion_spec`, concatenate exception_pairs into `idx_12_13` (scale=0) so they are zeroed from the main sum, then re-added with correct parameters via `exception_pairs`.

```python
all_idx_12_13 = idx_12_13 + exception_pairs  # exclude from main sum too
```

This pattern applies to any future caller of `make_energy_fn` with `exclusion_spec`. The fix is documented in `build_exclusion_spec` (fixtures file, lines 570-576) and in the audit doc.

## Decomposition Semantics (f3 → f4 mapping)

`get_prolix_nonbonded_energies` returns:
- `'lj'` = `chunked_lj_energy (1-5+ only) + exception_energy (1-4 LJ+Coulomb)` ← combined; matches OpenMM charge-zeroed E_LJ
- `'coulomb'` = `chunked_coulomb_energy (1-5+ only)` ← matches OpenMM E_Coul = E_nb − E_LJ_zeroed
- `'exception_14'` = `exception_energy` (1-4 LJ+Coulomb raw) ← for self-consistency gate only

## What Was Deferred

**f7 (PME stretch):** Not dispatched. f1–f6 all PASS on first auditor pass, satisfying the f7 dispatch condition. However, time budget was not available in this session. PME test (vacuum → periodic box, `pme_alpha ≈ 0.334 Å⁻¹`, `pme_grid_points=32`) is fully specified in `.praxia/docs/specs/260527_p2b-nonbonded-comparator.md` §f7. Estimated 2–4h.

## Known Non-Blocking Issues (from auditor)

1. **Repeated `jax.config.update("jax_enable_x64", True)` inside multiple fixture functions** (5 sites). Global state mutation that can leak; move to conftest session fixture in a follow-up. Not blocking.
2. **No explicit test for `lj + coulomb ≈ total_nb < 1e-4`** (f1-f3 self-consistency gate from plan). Implicit in data flow but not asserted as a standalone pytest. Add in follow-up if desired.
3. **`sys.path.insert` in test module** (line 14 of test file). Works but conftest-based discovery is cleaner. Low priority.

## Next Steps

1. **f7 (PME stretch):** Dispatch when time allows; spec is complete.
2. **§7.1 figure:** P2b closes the nonbonded leg. Both bonded (P2a) and nonbonded (P2b) prolix-vs-OpenMM parity gates are green. The external comparator figure can now be populated with prolix timing on ala-dip at the kernel level.
3. **Exclusion-pattern documentation:** Consider adding a design note to `src/prolix/physics/neighbor_list.py` or `system.py` explaining that callers must include exception_pairs in idx_12_13 when using the ExclusionSpec path.

## Worktree Commit History

```
git log worktree-p2b-nonbonded --oneline
866xxxx fix(p2b): f6 audit doc — clarify observed vs required gate thresholds
865c66e feat(p2b): f6 audit doc extended for nonbonded fields + INDEX updated
9da71d8 feat(p2b): f5 nonbonded force parity test via FD (RMS=5e-6 kcal/mol/Å, gate=0.5 PASS)
c00db60 feat(p2b): f4 nonbonded energy parity tests
bc44841 fix(p2b): f2+f3 exclusion double-counting — include exception pairs in idx_12_13 mask; return combined lj+exc14
5c85510 feat(p2b): f1+f2+f3 fixture nonbonded extension
```

To merge: `git -C /home/marielle/projects/prolix cherry-pick 5c85510..HEAD` (from worktree) or merge the branch `worktree-p2b-nonbonded` into main.
