# Next steps — after B1-XTRAX-WIRE (superseded — Claim-1 closed)

**date:** 2026-07-15
**campaign:** `2115b4dd` (`b1-full`) — **CONCLUDED: fail**
**status:** Claim-1 (H1) **FALSIFIED** — see postmortem `db3e7612-812b-4b22-a230-a757bb861c83`
**leaves:** B1-INFER, B1-XTRAX-WIRE, B1-AOT-RATIO, B1-FINITE-GATE all completed; B1-SETTLE-STACK (P2) / B1-1AKE-NAN (P3) parked
**invariants:** `.praxia/loop_priorities.toml`

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
