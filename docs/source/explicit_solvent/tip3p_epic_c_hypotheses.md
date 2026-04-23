# Epic C — ranked hypotheses (P2a-B2-R failure on `tip3p_ke_compare` Tier 1)

**When to use:** P2a-B2-R fails for **`openmm_ref_linear_com_on`** on a committed Tier‑1 JSON (or equivalent B-e2e artifact). **Not** automatically opened for P2a-B2-X (G4) alone — see [`tip3p_benchmark_policy.md`](tip3p_benchmark_policy.md).

## Observed symptom (2026-04-21 smoke)

Prolix rigid thermometer mean `T` can sit **far above** the bath (e.g. ~395–410 K vs ~305 K OpenMM) even with linear COM removal aligned; OpenMM stays near target.

## Hypotheses (most actionable first)

1. **Different effective DOF / thermometer definition mismatch** — Confirm both sides still use `dof_rigid = 6 * n_waters - 3` and the same kinetic energy definitions documented in the rerun memo; probe whether Prolix `rigid_tip3p_box_ke_kcal` vs per-atom KE used anywhere in the chain disagrees with OpenMM’s constrained `getKineticEnergy` mapping beyond COM.

2. **BAOAB + SETTLE splitting vs OpenMM integrator + SETTLE ordering** — `settle_langevin` applies SETTLE after a specific BAOAB substep sequence; OpenMM couples constraints with `LangevinMiddleIntegrator` differently. Expect **distribution** shifts, not just COM drift.

3. **Stochastic / friction unit mapping** — Re-verify `gamma_reduced = gamma_ps * AKMA_TIME_UNIT_FS * 1e-3` matches OpenMM’s `gamma` in 1/ps at the same `dt` (already fail-fast in JSON meta); extend to logging effective collision frequency per step if needed.

4. **Insufficient burn-in for Prolix rigid modes** — Prolix may need longer burn than OpenMM for the same nominal burn step count; sensitivity scan (e.g. 2× burn) as an A/B on **one** replica before touching physics code.

5. **JAX dtype / scatter path** — Watch `float64` vs `float32` scatter warnings under x64; confirm energy masks and masses are dtype-consistent end-to-end.

## Minimal reproducers

- **Fast:** `tests/physics/test_tip3p_com_linear_momentum.py` (paired short slice).
- **Medium:** Tier 0 `tip3p_ke_compare` (`n_waters=4`, fewer steps) with `--verbose-samples` to inspect COM and atomic T series.
- **Slow:** Single-replica Tier 1 subset before full `replicas=5` matrix.

## Next tickets (suggested)

- Instrument `settle_langevin` vs OpenMM step timeline (documentation + optional logging hooks).
- Optional: align burn / sample cadence between engines in the benchmark script for fairer window means.
- If R passes but X fails: treat as cross-engine residual under current protocol; decide whether to tighten physics or relax X for this script only.
