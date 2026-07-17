# Next steps — after B1-XTRAX-WIRE (superseded — Claim-1 closed, re-tested under supposed physics parity, PME bug found, re-run pending)

**date:** 2026-07-15 (original), updated 2026-07-16/17 (Phase 3 physics-parity re-run), corrected 2026-07-17 (PME stacked-dispatch bug)
**campaign:** `2115b4dd` (`b1-full`, confounded physics) — **CONCLUDED: fail**; `c3644ac7` (`b1-full-parity`, believed-genuine physics) — **CONCLUDED: fail, more decisively — but see correction below, this campaign's water-class physics was NOT actually PME-complete**
**status:** Claim-1 (H1) **FALSIFIED** under both the original confounded comparison and the Phase 3 re-run. **However, Phase 3's "genuine physics parity" claim is INCORRECT for the water class** — see "Correction (2026-07-17)" below. A further re-run is needed before Phase 3's 7.28x number can be cited as the physics-matched result.
**leaves:** B1-INFER, B1-XTRAX-WIRE, B1-AOT-RATIO, B1-FINITE-GATE, B1-SETTLE-STACK, B1-NONBONDED-PARITY all completed. Full re-run (Phase 3) completed 2026-07-16. PME stacked-dispatch bug found+fixed 2026-07-17 (commit `395995b`).
**invariants:** `.praxia/loop_priorities.toml`

## Correction (2026-07-17) — PME reciprocal energy was silently disabled for the entire Phase 3 re-run

While building a Phase 4 profiling script, found the compiled HLO for B1's real production dispatch (`EnsemblePlan`'s stacked/vmapped path, the *only* path B1 ever uses at B>1) contained zero PME/FFT instructions. Confirmed via 4 independent tests: real `EnsemblePlan.run()` gave bit-identical trajectories regardless of `pme_alpha`/`box_size` whenever dispatched through the stacked path, while single-bundle dispatch and hand-wired `settle_langevin` calls both showed genuine PME sensitivity. Two root causes, both required: (1) `physics_system_from_bundle`'s `box_size` derivation had no vmap-safe fallback (unlike `pme_alpha`/`nonbonded_cutoff`, which already had one) — a traced value landed in a field declared `static=True`; (2) the PME exclusion-correction branch derived 1-2/1-3/1-4 pairs via pure Python/numpy graph traversal (`topology.find_bonded_exclusions`), fundamentally unable to run under `jax.vmap`/`jit` tracing. Both raised exceptions silently caught by `single_padded_energy`'s (deliberately broad, to catch a legitimate one-time shape-probe) except clause. Fixed in commit `395995b`: `_host_box_size` mirrors the existing `_host_float` pattern; the exclusion correction now reuses `bundle.excl_indices`/`excl_scales_elec` (already precomputed host-side, already vmap-safe) instead of the host-only traversal. Verified 4 ways post-fix + a regression sweep (target parity test 3/3 pass, broader 8-file sweep 15 passed / 3 pre-existing-and-unrelated failures confirmed by direct comparison against the pre-fix code). Full record: memory `project_b1_pme_stacked_dispatch_fix_260717`, research record `260717_pme_reciprocal_silently_disabled_under_stacked_dispatch`.

**Implication: SETTLE constraints genuinely ran in Phase 3 (independently verified), but PME reciprocal did not.** The reported 7.28x slowdown reflects prolix's water class *without* real PME reciprocal cost for 16/64 systems. Since real PME only adds compute, not removes it, the true physics-matched gap is likely **larger**, not smaller — but this needs a re-run to confirm, not just asserted. Do not cite 7.28x as the physics-matched number until that re-run lands.

## Phase 3 outcome (2026-07-16/17) — genuine physics parity re-run

After the original `2115b4dd` falsification, prolix and OpenMM were found to be running different physics in B1 (prolix: zero constraints, no PBC/cutoff for water; OpenMM: HBonds+rigidWater, CutoffPeriodic). `B1-SETTLE-STACK` (`f73aea4`) and `B1-NONBONDED-PARITY` (`e014ecc`) implemented real SETTLE rigid-water constraints and periodic PME nonbonded for prolix; a post-implementation audit (`3cbe258`) found and fixed two more bugs that would have silently invalidated the re-run (a `pme_alpha` vmap-fallback mismatch, and an unconditional `EnsemblePlan` construction crash under periodic PME that meant Phase 2's own tests never exercised the real dispatch path).

Full re-run (64 systems × 200k steps × 3 seeds, H200-class `pi_so3` nodes, jobs `18069084`/`18069190`): **6/6 tasks COMPLETED, `finite_fraction=1.0` on every task** — no NaN blowups under real constraints + periodic PME.

| | prolix | openmm |
|---|---|---|
| median `t_total` | 4963.03 s (82.7 min) | 681.37 s (11.4 min) |
| range | 4961.6 – 4982.9 s | 680.1 – 691.1 s |
| `aot_ratio` | 0.0025 – 0.0054 | 0.0 |

**Ratio: prolix ≈7.28× slower than OpenMM** — worse than the original confounded ratio (≈4.35×), not better. Prolix's steady-state cost grew ~80% (2755s → 4963s) once real SETTLE constraint-solving and periodic PME reciprocal-space sums actually ran; OpenMM's grew only ~7.6% (633s → 681s) from `CutoffPeriodic`→`PME` alone, since it already paid for rigid-water constraints in the original run. Compile overhead is negligible for both (`aot_ratio` <1%) — this is a genuine steady-state throughput gap, not a JIT artifact.

**Verdict: Claim-1 (H1) remains FALSIFIED, and the physics-matched comparison closes the door on "the original result was an artifact of missing prolix features" — it wasn't. Prolix is genuinely slower per-step once it does the same physics OpenMM does.** FF-parameter-source mismatch (`ff19SB`/classic-TIP3P vs `amber14-all`/TIP3P-FB) remains a documented, out-of-scope residual and does not affect this timing conclusion.

**Provenance caveat (does not affect the result above, but flag for infra follow-up):** the run manifest's `git_sha` field reads `b6aa00b` on branch `main` — a June 11 commit that predates `b1_init_exec.py` entirely. This is misleading metadata, not a wrong-code run: direct diff of the remote working tree against local `b1/full`@`3cbe258` confirms `scripts/benchmarks/b1_init_exec.py`, `src/prolix/physics/settle.py`, `src/prolix/batched_energy.py`, `src/prolix/api/bundle_md.py`, and `src/prolix/api/ensemble_plan.py` are byte-identical to the intended commit. The remote's git-checked-out branch/HEAD is stale relative to whatever rsync-based mechanism actually deploys code there — `git_sha` in `run_manifest.json` should not be trusted as provenance until this is fixed. Same root-cause family as the separately-discovered `CAMPAIGN_ID`-not-threaded-through-`sbatch` bug (the local `sbatch` proxy wrapper doesn't forward env vars over SSH) — both are cluster-submission provenance gaps, not physics bugs.

## Where we are

| Leaf | Status | Gate |
|------|--------|------|
| B1-INFER | **completed** | while_loop + streaming sink |
| B1-XTRAX-WIRE | **completed** | same-bucket `can_jit_vmap`; host `partition_bundles_by_shape`; DedupGather topology-keyed only |
| B1-AOT-RATIO | **completed** | `aot_ratio = t_aot / t_total` (was inverted) |
| B1-FINITE-GATE | **completed** | `finite_fraction >= 0.9` (tolerates 5/64 1AKE NaN minority, diag 17905437) |
| Claim-1 medians | **LANDED — FALSIFIED** | prolix 2755.3s (17928274/17930333/17924116) vs openmm 633.0s (17982750/17983798/17982521); ratio ≈4.35× against H1 |

## Result

H1 ("prolix cold-start `t_total` < OpenMM N-Context", B=64/100ps/3seeds) is falsified under the locked prereg protocol on H200. Root cause is a genuine prolix steady-state throughput gap, not compile dominance — `aot_ratio` ≈0.003–0.004, far below the 0.5 explanatory threshold. Full analysis and asset links: `scripts/benchmarks/b1_init_exec.py.db3e7612-812b-4b22-a230-a757bb861c83.bth.postmortem.toml`.

## Next steps (per postmortem + 2026-07-15 direction: profile before deciding)

1. Profile prolix `t_steady_state` breakdown (per-class vmap cost, energy/force, SETTLE restore) to check for a fixable bottleneck before deciding whether to drop or re-scope Claim-1.
2. Do not cite a prolix cold-start win from campaign `2115b4dd` — it is concluded `fail`.
3. `B1-SETTLE-STACK` (P2) and `B1-1AKE-NAN` (P3) remain parked unless the paper needs vacuum long-horizon all-finite.
4. Future OpenMM baselines: use the conda-forge CUDA env (`b1-openmm`), not the PyPI wheel — it has no `libOpenMMCUDA`.

## Dispatch rule (locked, still valid for any follow-up runs)

- Host partition by `shape_spec` (K classes; Python over K only).
- Within class: Vmap/SafeMap, `integration_prefix` = atom bucket, real masses.
- **Never** DedupGather seeded Langevin (would copy one traj × R).

## Submit (for any follow-up profiling runs)

```bash
export CAMPAIGN_ID=2115b4dd
sbatch --array=0-2%1 --export=ALL,CAMPAIGN_ID,FOOTPRINT=prereg,B1_PATH=inference \
  scripts/slurm/b1_init_exec.slurm
```
