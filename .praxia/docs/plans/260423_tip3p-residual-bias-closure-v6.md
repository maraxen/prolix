# TIP3P Residual Temperature-Bias Closure Plan (v6)

Date: 2026-04-22  
Status: Oracle-approved for execution (cycle 6)

## Objective

Close the residual Prolix/OpenMM rigid-temperature mismatch under `openmm_ref_linear_com_on` and end in one of two explicit states:

- A (preferred): both engines pass P2a-B2-R.
- B (fallback): bounded exception memo with explicit external-claim prohibition and dated remediation.

## Non-Negotiable Constraints

- Do not change benchmark thresholds during this effort.
- Keep instrumentation PR separate from physics-change PR.
- Keep all new dynamics behavior behind rollbackable flags.

## PR-0: Instrumentation + Compatibility (No Physics Semantics Change)

### A. Integrator Controls and Metadata Plumbing

- [ ] `scripts/benchmarks/tip3p_langevin_tightening.py`
  - [ ] Add `--gamma-ps` (default 1.0).
  - [ ] Add `--projection-site {post_o,post_settle_vel,both}`.
  - [ ] Add `--settle-velocity-iters` (default 10).
  - [ ] Add `--diagnostics-level {off,light,full}`.
  - [ ] Add `--diagnostics-decimation` (default 10 for light/full).
- [ ] `src/prolix/physics/settle.py`
  - [ ] Extend `settle_langevin(..., projection_site=..., settle_velocity_iters=...)`.
  - [ ] Extend `settle_velocities(..., n_iters=10)` and remove hard-coded 10-loop.
- [ ] Persist minimum per-run metadata:
  - [ ] `git_sha`
  - [ ] `jax_backend`
  - [ ] `profile_id`
  - [ ] `gamma_ps`
  - [ ] `projection_site`
  - [ ] `settle_velocity_iters`
  - [ ] `diagnostics_level`
  - [ ] `diagnostics_decimation`
  - [ ] `effective_integrator_config`

### B. Schema Migration Contract (v1 -> v2)

- [ ] Keep aggregate writer default: `tip3p_tightening_aggregate/v1`.
- [ ] Add dual-read support in `scripts/benchmarks/tip3p_ke_gates.py` for v1/v2.
- [ ] Pre-register v2 contract:
  - [ ] v1 required fields remain unchanged.
  - [ ] v2 requires `meta.run_metadata` with keys listed above.
  - [ ] Optional diagnostics block defaults to null when absent.
  - [ ] Unknown schema triggers explicit hard error.
- [ ] Add fixtures/tests in `tests/physics/test_tip3p_ke_gates.py`:
  - [ ] v1 pass/fail (existing coverage retained).
  - [ ] v2 pass/fail.
  - [ ] v2 missing optional diagnostics.
  - [ ] unknown schema hard-fail.
- [ ] Add aggregator `--schema-version {v1,v2}` only after dual-read tests exist.

### C. Deterministic Replay Gate for v2 Default Promotion

Promotion to v2 default requires all:

- [ ] 2 consecutive CI green runs with dual-read enabled.
- [ ] Deterministic replay passes 3/3 attempts on frozen baseline artifact tuple.
- [ ] Metadata completeness check: 100% required fields present.

Replay acceptance:

- [ ] Exact-match fields: schema tag, profile_id, run_metadata keys, replica ordering, engine labels.
- [ ] Numeric tolerance fields: absolute diff <= 1e-6 for stored means/SEMs.
- [ ] Any replay failure blocks promotion.

### D. Diagnostics Non-Perturbation Gate

Run A/B with diagnostics `off` vs `light` on >=2 seeds (or replicas):

- [ ] mean abs delta(`mean_T_K`) <= 2 K
- [ ] max delta(`mean_T_K`) <= 4 K
- [ ] abs delta(`std_T_K`) <= 2 K
- [ ] no guard metric drift >10% (M3/M4/M5 summaries)

## PR-1: DOE + Paired Analysis Artifact

### A. Frozen Baseline Tuple (Immutable)

- [ ] Freeze tuple:
  - [ ] commit SHA
  - [ ] profile_id (`openmm_ref_linear_com_on`)
  - [ ] hardware class
  - [ ] full CLI command
  - [ ] seed policy
  - [ ] steps/burn/sample
  - [ ] schema version

### B. New Paired Analysis Artifact

- [ ] Add `scripts/benchmarks/tip3p_tightening_paired_analysis.py`.
- [ ] Emit `tip3p_tightening_paired_analysis/v1` with:
  - [ ] per-replica-index deltas for M1-M5
  - [ ] mean delta, SEM, one-sided 95% bound fields
  - [ ] source aggregate artifact paths
  - [ ] frozen baseline tuple hash
- [ ] Require all escalation/promotion decisions to consume this artifact only.

### C. Statistical Contract (Pre-Registered)

- [ ] Pairing definition: replica-index matched across variants.
- [ ] Primary endpoint:
  - [ ] `delta = M1_incumbent - M1_candidate` (higher is better).
- [ ] Inference:
  - [ ] one-sided paired t bound at alpha=0.05.
  - [ ] advancement only if lower 95% one-sided bound > 0.
- [ ] Minimum paired replicas:
  - [ ] smoke n >= 4
  - [ ] decision n >= 5
- [ ] Missing/NaN handling:
  - [ ] any missing paired delta => candidate invalid => auto-prune.
- [ ] Tie handling:
  - [ ] lower bound <= 1e-9 treated as no improvement.

### D. Multiplicity and Guardrails

- [ ] Use fixed hierarchical gatekeeping:
  - [ ] evaluate primary endpoint first,
  - [ ] evaluate secondary improvements only if primary passes.
- [ ] Treat M3/M4/M5 as hard constraints (not discovery tests).

### E. Budget and Pruning (Hard Enforced)

Decision-equivalent accounting:

- [ ] `eq_runs = decision_runs + 0.5 * smoke_runs`
- [ ] enforce global cap `eq_runs <= 30.0` with pre-check and post-run ledger update.
- [ ] persist in every artifact:
  - [ ] `budget.eq_consumed`
  - [ ] `budget.eq_remaining`
  - [ ] `budget.phase`

Stage ladder:

- [ ] S1 smoke prefilter: 6 cells x 4 replicas = 24 smoke-runs.
- [ ] Advance top 2 candidates by primary endpoint.
- [ ] S1 decision for top 2: 2 x 5 replicas = 10 decision runs.
- [ ] Remaining budget for S2/S3 decision work: max 20 eq-runs.

Mandatory early prune:

- [ ] no positive primary improvement (lower bound <= 0), or
- [ ] guardrail violation:
  - [ ] M3 p95 > 1.2x baseline
  - [ ] M4 p95 > 1.2x baseline
  - [ ] M5 > 1.5x baseline

## PR-2: Physics Changes (Escalation-Controlled)

- [ ] D1 (low risk): reorder/dual projection placement only.
- [ ] D2 (medium risk): constrained OU update for rigid water triplets.
- [ ] D3 (high risk): LFMiddle-aligned constrained sequence redesign.

Escalation logic:

- [ ] Escalate to D2 if best S1 has <30% M1 improvement or any guardrail fails.
- [ ] Escalate to D3 if best D2 still has Prolix R-relative error >3.0% (decision runs).
- [ ] Tie-break rule: absolute residual (>3.0%) dominates conflicting relative criteria.

Required gates for any D-option:

- [ ] Unit tests: projection idempotence, manifold residual non-increase, COM invariants.
- [ ] Integration tests: short paired OpenMM/Prolix checks on M1-M5 trends.
- [ ] Cluster progression: 1-rep smoke -> 4-rep mini -> 5-rep decision.

## PR-3: Docs + Claim Governance

- [ ] Add stage-timing mapping table:
  - [ ] OpenMM `LangevinMiddleIntegrator` + `CMMotionRemover`
  - [ ] Prolix BAOAB + SETTLE + projection placement.
- [ ] If exception path B:
  - [ ] write signed exception memo (owner + reviewer),
  - [ ] include full artifact tuple and bounded claim language,
  - [ ] include dated D3 remediation milestone.
- [ ] Keep blocking check that forbids release-note parity claims unless R-both pass is true.

## Rollback Flag Matrix

- [ ] Defaults preserve legacy behavior unless new flags explicitly enabled.
- [ ] Precedence: `both` > `post_settle_vel` > `post_o`.
- [ ] Validate illegal combinations at CLI parse-time with deterministic error text.
- [ ] Regression test when all new flags disabled:
  - [ ] behavior must match legacy within established tolerance bands.

## Owner Decisions (Confirmed Recommendations)

- **OD1 (High):** Internal milestone can proceed via exception path; external parity claims remain forbidden.
- **OD2 (High):** Keep D3 trigger at >3.0% residual after D2 decision runs.
- **OD3 (Medium):** Promote schema v2 default only after dual-read + deterministic replay proof.

