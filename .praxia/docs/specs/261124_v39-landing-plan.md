---
title: Landing the v3.9 dev line on main
description: Plan to bring prolix's local-only dev line feat/v3.9-complete onto main, superseding PRs #41/#42/#44 and the b1/full integration branch.
task_id: 261124_sprint-review-sync
status: draft
date: 2026-11-24
---

# Landing the v3.9 dev line on main

## Why

prolix has four overlapping lines of unmerged work and none of them targets `main` cleanly:

- `b1/full`, an integration branch with no PR into `main`.
- PRs #41 and #42, which target `b1/full`.
- PR #44 (263k lines), which targets `main`.
- The local-only `feat/v3.6` → `v3.9-complete` chain, which is the actual development head.

`main` is 64 commits behind all of them. This plan lands one line, v3.9, and retires the rest. **No merge happens without the owner's sign-off on this document.**

## Facts (verified 261124 unless tagged otherwise)

Commands are reproducible from any prolix checkout after `git fetch --all`.

| # | Fact | How verified |
|---|---|---|
| F1 | `feat/v3.9-complete` is **local only** (no upstream), 99 commits ahead of `origin/main` and 64 behind, last commit 2026-11-18. The same is true of v3.6, v3.7 and v3.8. | `git for-each-ref`; `rev-list --count` |
| F2 | v3.9 contains #41 and #42 (`merge-base --is-ancestor` = yes for both). | git |
| F3 | v3.9 contains #44's **content**, but not its commits. `ddfec0b chore(v3.5-snapshot)` (2026-11-17) folded it in. `git cherry` marks all 13 #44-only commits as non-equivalent. All five `src/prolix/openmm/{config,context,converter,lepton_compiler,pattern_matcher}.py` exist on v3.9, and the DEF-23..DEF-32 fix markers are present on both sides. v3.9's `src/prolix/openmm` is about 13k lines larger than #44's. | `git cherry`, `git grep` |
| F4 | v3.9 lacks b1/full's two merge commits (#37 phase-3 evidence/kUPS, #39 knob governance). Their files partly diverge on v3.9: 3 of 20 and 25 of 51 files differ. **Unverified:** whether v3.9 evolved those files further or dropped them. | `git log`, per-file `git diff` |
| F5 | `git merge-tree main × v3.9` gives **73 conflicted paths**: 34 under `scripts/` (mostly `scripts/slurm` add/add), 25 under `src/`, 7 under `.praxia/`, plus `uv.lock`, `myxcel.toml`, `README.md`, `CLAUDE.md`, one `tests/` path and one `docs/` path. | `git merge-tree --write-tree --name-only` |
| F6 | v3.9 vs main in lines added: `data/pdb` 141.7k in 2 files (a generated OpenMM XML and a PDB, not in LFS); `src/prolix` 86k in 101 files; `docs/benchmarks` 46k in 45 files; `tests/openmm` 22k; `tests/physics` 18k; `scripts/benchmarks` 15k; the rest under 6k per directory. | `git diff --numstat` |
| F7 | `main` already tracks 439 `.praxia/` telemetry files, and v3.9 tracks 430. This hygiene debt predates v3.9. | `git ls-tree` |
| F8 | `main` has orphan gitlinks `submodules/gromacs` and `submodules/openmm` with no `.gitmodules` entry, which breaks git-URL installs (debt #3609). | `git ls-tree`, failed `uv tool install` |
| F9 | Review of #41 (comment on PR #41, 261124) found two confirmed majors that are **still present on v3.9**:<br>(a) `pallas_bonded.py` `_triton_gpu_available` only rejects `cc < 8.0`, so sm120 takes compiled `plt.atomic_add` force accumulation. v3.9 has 76 such call sites in `pallas_bonded.py` (45 in #41) and atomic_add in 7 `src` files.<br>(b) The GB and SASA references `del box, has_box`, so radius gradients ignore PBC. | `git show v3.9:<file>`, grep |
| F10 | Pallas Triton `atomic_add` leaves the output all zero on RTX PRO 6000 Blackwell (sm120) under jax 0.11.1. Interpret mode is exact. The force-accuracy record on v3.9 (`docs/benchmarks/261111_l40s_pallas_v35_physics_validation.json`) is from an L40S (sm89), where atomics work. v3.9 and `main` both lock **jax 0.10.2**, not the probed 0.11.1. **Unverified:** sm120 behaviour on 0.10.2, and H200 (sm90). The P3(a) canary settles it either way. | GPU probe 261026 (throwaway); v3.9 benchmark JSON; `uv.lock` |
| F11 | Cursor `surface_dispatch` died twice without writing a log in a prolix review worktree; the same call worked in demistify. **Unverified:** the cause. | dispatch records `ad52ba3f…`, `21bff46e…` |

## Goal

`main` carries the v3.9 code, and the sm120 and PBC defects (F9) are fixed and verified on hardware. #41, #42 and #44 are closed as superseded, `b1/full` is retired, and the v3.x chain is no longer the only copy of the work.

## Non-goals

- Re-reviewing every v3.9 line from scratch. Review is sliced (P5) and focused on kernels, OpenMM parity and anything changed while resolving conflicts.
- Performance claims. Benchmarks under `docs/benchmarks` stay labelled exploratory until each has a bathos sidecar.

## Plan

**P0. Back up the dev line now.** Push `feat/v3.6-cupotential-fixes`, `v3.7`, `v3.8` and `v3.9-complete` to origin as plain branches (no PRs). This is cheap and reversible, and removes the single-copy risk in F1. *Owner approval needed; nothing is pushed yet.*

**P1. Hygiene on main, as small separate PRs:**
1. Remove the orphan gitlinks (#3609).
2. Untrack `.praxia/` telemetry: `git rm -r --cached` plus `.gitignore`, keeping `.praxia/docs/`.
3. Decide where `data/pdb` goes: git-lfs, or a fixtures download. This must happen before 142k lines of generated XML enter `main`'s history.

**P2. Integration branch.** Create `land/v3.9` from `origin/main`, then `git merge feat/v3.9-complete` and resolve the 73 conflicts (F5):
- `scripts/slurm` add/add conflicts: keep both variants under distinct names, or take v3.9's and list what was dropped.
- `src/` conflicts: resolve by hand and record each resolution in the PR body.
- `uv.lock`: regenerate it, don't hand-merge.
- Then reconcile F4: for each of the 28 diverging #37/#39 files, record whether v3.9 supersedes it or the b1/full change must be ported.

**P3. Fix before landing, on `land/v3.9`:**
- (a) sm120. Replace the capability gate with a one-time GPU canary: a small `atomic_add` into a zeroed buffer, checked for row sums equal to n. If the canary fails, fall back to the interpret path or the plain-JAX reference, and log it loudly. Apply the same check to every atomic_add site found by `git grep atomic_add src` (7 files).
- (b) Pass the box into `_gb_ace_nl_ref_one` and `_sasa_nl_ref_one` (or raise when `has_box`), and add a periodic parity test.
- (c) The #41 minors that survived onto v3.9:
  - Born-radii kernel: NL flag bits and the self-pair.
  - Duplicate `make_spread_op` / `make_gather_op`.
  - Document the soft-core exponent change and the erf approximation.

**P4. Verify, pre-registered with bathos (`bth`):**
- Local: narrow CPU parity tests only (single files, with thread caps).
- **Compiled GPU path on Engaging** RTX PRO 6000 (sm120) and H200 (sm90). Titanix's Turing GPUs run no compiled Pallas, so they cannot verify P3(a).
- One `.bth.toml` with `[outcomes]` covers:
  - canary verdict per architecture;
  - force parity, compiled vs reference, for bonded, PME spread and GB/SASA (max abs and relative error thresholds fixed up front);
  - energy drift on one small solvated system.

**P5. Review in slices.** `land/v3.9` is too large for one review. Review it as stacked or diff-scoped passes: physics kernels; OpenMM; tests; scripts, benchmarks and docs; and conflict resolutions. Each pass gets a code review plus an independent audit. Use Cursor if F11 is resolved, otherwise two Sonnet reviewers on disjoint files.

**P6. Land and retire.** Merge `land/v3.9` into `main` (owner authorises the merge). Then:
- Close #41, #42 and #44 with a "superseded by" link.
- Close the drafts #40 and #43 or rebase them.
- Delete `b1/full` and the v3.x backup branches only after `main` contains them (`merge-base --is-ancestor`).

## Risks

| Risk | Mitigation |
|---|---|
| The v3.x chain is lost before P0 (single local copy). | Do P0 first. |
| Conflict resolution silently drops b1/full #37/#39 work (F4). | Per-file reconciliation table in the P2 PR body. |
| The P3(a) canary passes on one architecture and the kernels still fail on another. | Canary per process plus compiled-GPU parity on both sm120 and sm90 in P4. |
| 142k-line data files bloat `main`'s history. | P1.3 before P2. |
| Review fatigue on an 86k-line source diff. | P5 slices; review conflict resolutions and P3 changes in full. |

## Decisions for the owner

1. Approve P0 (push the four v3.x branches as backups).
2. `data/pdb`: git-lfs or a fixtures download?
3. Land v3.9 as one integration merge (recommended, keeps history) or as a series of squashed slices?
4. Verify P4 on the locked jax 0.10.2 only, or also on 0.11.x ahead of a planned upgrade.
