---
date: 260526
sprint_id: 2
sprint_title: "P2a Gate + P1a Continuation"
status: partial-with-known-defect
next_session_required: true
---

# Sprint Handoff (260526) — P2a + P1a-1d

## What landed cleanly on `main`

| Commit | Subject |
|---|---|
| `0c951b5` | docs(s71): comparator paper reference notes (torchmd/espaloma/dmff) |
| `a34d275` | chore(s71): sync uv.lock after cudnn install |
| `0a6d4df` | docs(p2a): plan + spec for OpenMM parity harness (bonded-only) |
| `e9459b2`–`58aa4f9` | feat(p2a): f1–f6 implementation [⚠ broken on main — see below] |
| `cb98a0a` | docs(p1a-1d): plan for bonded analytical forces expansion |
| `f411d79` | docs(p1a-1d): spec for bonded analytical forces (6 fixer tasks) |
| `53260cb` | fix(p2a): NEEDS_WORK reviewer fixes (header case + class flatten) [⚠] |
| `bb7dc1a` | **feat(p1a-1d): bonded analytical forces (5 functions + 9 parity tests)** ✅ |

## What works (verified on main this session)

- **P1a-1d** — `tests/physics/test_analytical_forces.py` — **9/9 PASS** at float64
  - bond_forces_analytical, angle_forces_analytical, dihedral_forces_analytical, improper_forces_analytical (periodic + harmonic), urey_bradley_forces_analytical
  - Each agrees with `-jax.grad(energy)` at `atol=1e-10` except `test_improper_harmonic_vs_grad` (atol=5e-7, see Defects)

## What's broken on `main`

### Defect 1 — P2a impl tests fail
`tests/physics/test_openmm_parity_bonded.py` — 5/5 tests ERROR/FAIL with
`IndexError: 2-dimensional array indexed with 3 regular indices`

**Root cause** (recon, 260526): P2a code was developed in a worktree branched from
`origin/main` (19 commits stale). Current main's `PhysicsSystem` (typing.py:248)
expects `dihedral_params: (N_dih, N_terms, 3)` — multi-term periodic dihedrals.
Fixture (`fixtures_openmm_parity.py:292`) builds them as `(N_dih, 3)`.

**Remediation tier: SMALL (1–2 hours)** — 4 mechanical edits:
1. `fixtures_openmm_parity.py:23` — `from prolix.physics.types` → `from prolix.typing`
2. `fixtures_openmm_parity.py:292` — wrap `dihedral_params` in `jnp.expand_dims(..., axis=1)` so shape becomes `(N_dih, 1, 3)`
3. `fixtures_openmm_parity.py:294` — same for `improper_params`
4. `.praxia/docs/audits/260526_p2a-bonded-field-audit.md:36` — update shape annotation `(N_dihedrals, 3)` → `(N_dihedrals, N_terms, 3)`

After fixes, re-run:
```bash
uv run pytest tests/physics/test_openmm_parity_bonded.py -m openmm -v
```

### Defect 2 — Atol drift on harmonic improper
`tests/physics/test_analytical_forces.py:248` uses `atol=5e-7` for
`test_improper_harmonic_vs_grad` (spec required `1e-10`). Fixer justified as
"nested autodiff." Needs reviewer assessment — is 5e-7 acceptable for nested
autodiff, or does this mask a math bug?

## Sprint orchestration notes (for retrospection)

- **Worktree baseRef issue**: `EnterWorktree` defaulted to `baseRef=fresh` (= origin/main). Local main was 19 commits ahead. P2a developed against stale baseline; tests passed in worktree but break on current main. **Recommendation for next session**: either (a) `git push origin main` before starting a worktree, or (b) explicitly use `baseRef=head` (current local HEAD) when entering a worktree.
- **Fixer cwd leak**: P1a-1d fixer wrote to `/home/marielle/projects/prolix/` (user's main checkout) instead of the worktree path that was explicitly specified in its prompt. Its 3 commits (1c5a1be / 268ce70 / 7debe92) were `git add -A` style and pulled in 1.3 GB of `.praxia/cache/analyzer.db` + the entire `.claude/worktrees/sprint-260526-p2a` directory. These commits were reset out (`git reset --hard a34d275`); replaced with the single clean `bb7dc1a`.
- **Reviewer false positive**: P2a reviewer rendered NEEDS_WORK on textual issues but missed the structural brokenness because reviewer ran the same stale worktree code. The orchestrator's independent `uv run pytest` verification ALSO ran in the stale worktree — also a false positive.

## Outstanding tasks for next session

1. **P2a impl remediation** (~1–2h) — apply the 4 edits above; re-run tests; commit.
2. **P1a-1d reviewer pass** — assess atol-5e-7 drift on harmonic improper; either justify or fix.
3. **NLM 3-paper notebook** — blocked on `! nlm login` from user; then create notebook from the 3 reference docs in `.praxia/docs/reference/`.
4. **Cleanup** — kept worktree at `.claude/worktrees/sprint-260526-p2a` (branch `worktree-sprint-260526-p2a`) is no longer needed; user can `git worktree remove` it. Also `.claude/worktrees/` directory appears as untracked in `git status` — consider adding to `.gitignore`.
5. **Sprint-composer infra**: `workflow(action="list")` returned empty for prolix — no PCW templates registered. Worth setting up if multi-track sprints become routine.
6. **Lessons addressed**: Lessons 32 (scope-equivalence gate) and 33 (scaling-signal-framing) were the trigger for P2a. P2a closes the gate **conceptually** (plan + spec + audit doc on main) even though impl needs the remediation above before re-running any §7.1 comparators.

## Sprint scope status

- ✅ **P2a docs**: plan, spec, audit-doc on main
- ⚠️ **P2a impl**: broken on main, remediation scoped to ≤2h
- ✅ **P1a-1d**: full impl + tests on main, 9/9 PASS
- ⏸️ **NLM**: user-action required

## Files referenced

- Plan: `.praxia/docs/plans/260526_p2a-openmm-parity-bonded.md`
- Spec: `.praxia/docs/specs/260526_p2a-openmm-parity-bonded.md`
- Audit doc: `.praxia/docs/audits/260526_p2a-bonded-field-audit.md`
- 1d plan: `.praxia/docs/plans/260526_p1a-1d-bonded-analytical-forces.md`
- 1d spec: `.praxia/docs/specs/260526_p1a-1d-bonded-analytical-forces.md`

---

## Follow-up session (260526-bis)

Both Defect 1 (P2a impl) and Defect 2 (atol drift) closed. NLM remains user-action.

### Commits added on branch `worktree-sprint-260526-followup` (off `main` HEAD `e330612`)

| Commit | Subject | Notes |
|---|---|---|
| `948d232` | chore: gitignore worktrees and untrack analyzer.db | `.claude/worktrees/` added to `.gitignore`; `.praxia/cache/analyzer.db` `git rm --cached` (22 MB, already in ignore by pattern). |
| `7d38de3` | fix(p2a): update fixture for current PhysicsSystem dihedral shape | 4 mechanical edits per handoff §Defect 1. `tests/physics/test_openmm_parity_bonded.py` now 5/5 PASS. |
| `297c72e` | fix(p1a-1d): align improper-harmonic analytical epsilon to bonded reference | 1-line epsilon fix in `analytical_forces.py:632` + atol restored to 1e-10 + misleading test comment removed. `test_analytical_forces.py` 9/9 PASS at atol=1e-10. |

### Defect 2 root-cause (auditor finding)

The fixer's "nested autodiff" rationale was **technically false**. `_improper_forces_harmonic` (`analytical_forces.py:609`) and the reference `make_harmonic_improper_energy_fn` (`bonded.py:145`) compute the same energy via different code paths. Both use single-order `jax.grad`. The drift came from an **epsilon mismatch** in their inlined `compute_dihedral_angles` work:

- `bonded.py:100`: `b1_norm = norm(b1) + 1e-8`
- `analytical_forces.py:632` (before fix): `b1_norms = norm(b1s) + jnp.float32(1e-12)`

Auditor predicted force-difference scale `~k·δ·ε ≈ 20·(π/4)·1e-8 ≈ 3e-7` — empirically matched observed ≤5e-7. The companion `test_improper_periodic_vs_grad` only passed at 1e-10 because its planar test geometry sits at a `sin(n·φ − phase) = 0` stationary point where the gradient is zero, masking the same mismatch in that test.

Minimal fix chosen (vs. auditor's "Preferred" composition refactor): aligning the analytical-side epsilon to bonded.py's `1e-8` has the smallest blast radius — bonded.py is the hot production energy path. The compose-the-bonded-energy refactor is filed as a deferred follow-up (see §Deferred follow-ups below).

### Verification

```
uv run pytest tests/physics/test_openmm_parity_bonded.py -m openmm -v  → 5/5 PASS
uv run pytest tests/physics/test_analytical_forces.py -v               → 9/9 PASS
```

### Updated sprint scope status

- ✅ **P2a docs**: plan, spec, audit-doc on main
- ✅ **P2a impl**: 5/5 PASS, commit `7d38de3`
- ✅ **P1a-1d impl + tests**: 9/9 PASS at atol=1e-10, commit `297c72e`
- ⏸️ **NLM**: still blocked on user `! nlm login`

### Deferred follow-ups

1. **Compose `_improper_forces_harmonic` from `make_harmonic_improper_energy_fn`** — eliminates the duplicated dihedral-angle math in `analytical_forces.py:609-664`. Auditor's "Preferred" path. Requires plumbing `improper_mask` through the composed energy. Estimated ≤30 min if added to a future cleanup sprint.
2. **Old worktree cleanup**: `.claude/worktrees/sprint-260526-p2a` (branch `worktree-sprint-260526-p2a`) is the stale-baseline worktree from the prior session; `git worktree remove` when convenient.
3. **NLM 3-paper notebook**: needs user `! nlm login` from the main session, then create from the 3 reference docs.

### Branch landing

The 3 commits live on `worktree-sprint-260526-followup`. They linearly extend `main` HEAD (`e330612`), so the parent session can `git merge --ff-only worktree-sprint-260526-followup` from `main` to land them. No conflicts expected.

### Lessons captured

- **Auditor over reviewer for "is this numerical bound right" questions** — reviewer ran tests and saw green; auditor traced two energy implementations, computed sensitivity, and located the actual mismatch. This worked because the auditor was given the bounded chunk explicitly (which function, which test, which spec). A blanket "audit the test file" would have been weaker.
- **`baseRef: head` setting in `.claude/settings.json`** — set this once; new worktrees branch from local HEAD instead of (possibly stale) `origin/main`. Last session's whole P2a-broken-on-main story was a `baseRef: fresh` (default) failure mode.
- **"Nested autodiff" as an excuse phrase**: when a justification is hand-wavy and the math says it shouldn't be that bad, dig. Single-order grad of two slightly-different energies looks the same as nested grad of one energy from the outside, but they aren't.
