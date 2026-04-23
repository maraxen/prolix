# TIP3P KE / Langevin benchmark policy (pre-registered)

**Effective date:** 2026-04-21  
**Related:** [`tip3p_ke_x64_rerun_decision.md`](tip3p_ke_x64_rerun_decision.md), [`tip3p_p2a_b2_e2e.md`](tip3p_p2a_b2_e2e.md), [`tip3p_epic_c_hypotheses.md`](tip3p_epic_c_hypotheses.md), `scripts/benchmarks/tip3p_ke_compare.py`, `scripts/benchmarks/tip3p_ke_gates.py`, `scripts/benchmarks/tip3p_langevin_tightening.py`

This document binds profile IDs, gate definitions, and anti–cherry-picking rules **before** Tier‑1 artifacts used for release-style claims are treated as authoritative.

## Primary claim profile

- **Binding profile for product / external parity language:** `openmm_ref_linear_com_on`
  - OpenMM: `ForceField.createSystem(..., removeCMMotion=True)` (explicit in harnesses).
  - Prolix: `settle_langevin(..., remove_linear_com_momentum=True)` after the SETTLE velocity projection each step.

## Diagnostic profile (CI / hypothesis closure)

- **`diag_linear_com_off`:** OpenMM `removeCMMotion=False`; Prolix `remove_linear_com_momentum=False`. Used to close the historical G4 narrative under **matched linear-COM policy** without changing Prolix until Phase B2.

## Profile IDs (immutable strings)

| `profile_id` | OpenMM `removeCMMotion` | Prolix `remove_linear_com_momentum` |
|----------------|-------------------------|--------------------------------------|
| `diag_linear_com_off` | `false` | `false` |
| `openmm_ref_linear_com_on` | `true` | `true` |

## Rule: no post-hoc profile switching

After the **first** Tier‑1 JSON (or checkpoint / `temps_wip` aggregate) consumed toward a gate for a given study, **do not** change `profile_id`, tolerances, or primary profile without a new dated policy revision and a fresh matrix.

## G4 (soft gate) — definition

As in the rerun decision memo:

- Replicate-mean temperatures: OpenMM uses `2 * KE_getState / (dof_rigid * k_B)` with `dof_rigid = 6 * n_waters - 3`; Prolix uses `2 * rigid_tip3p_box_ke_kcal / (dof_rigid * k_B)` with the same `dof_rigid`.
- **G4 pass:** `|mean_T_omm - mean_T_plx| <= 2 * sqrt(SEM_omm² + SEM_plx²)` on replicate means.

G4 is evaluated **per `profile_id`**. Diagnostic G4 under `diag_linear_com_off` does **not** substitute for the primary profile.

## Paired short-run sanity (primary profile)

For `openmm_ref_linear_com_on` only, CI runs a **short** OpenMM vs Prolix slice (`tests/physics/test_tip3p_com_linear_momentum.py::test_openmm_prolix_short_window_mean_t_primary_profile`) with:

- **Observables:** window-mean rigid thermometer `T` (same definitions as G4); total momentum magnitude `|sum_i p_i|` (Prolix atomic momenta; OpenMM constrained velocities via state).
- **Pre-registered tolerances (n_waters = 2, ~800 steps post burn, dt = 2 fs):**
  - `|mean_T_omm - mean_T_plx| < 80` K (loose bound; detects gross thermostat / COM regression).
  - Prolix with COM removal: `mean(|sum p|) / sum(m) < 1e-2` in AKMA momentum units after burn (order-of-magnitude drift control).

**RNG:** OpenMM uses `setVelocitiesToTemperature(..., seed)` with fixed seed; Prolix uses `jax.random.PRNGKey(seed)` for `init_fn`. **Bitwise trajectories are not expected to match**; only the above observables are gated.

## Artifact hygiene

- JSON summaries MUST include top-level `meta.profile_id` (for `tip3p_ke_compare.py`) or per-run `profile_id` (for tightening summaries).
- `tip3p_ke_compare.py` emits `meta.schema: "tip3p_ke_compare/v1"` for fixture and gate tooling version alignment.
- `aggregate_tip3p_tightening_logs.py` prints top-level **`schema`: `"tip3p_tightening_aggregate/v1"`** plus `meta.benchmark_policy.temperature_k` (inferred from per-run `target_T_K`) and optional `meta.profile_id` when unique across runs. **Normative gate semantics** for that layout match **P2a-B2-R / P2a-B2-X** using `per_replica_mean_T_K` (mean = `statistics.fmean`, SEM = `pstdev/sqrt(n)` over replicates, same as `tip3p_ke_compare` replicate summaries).
- Aggregators MUST use `--require-profile-id` when combining **tightening** tee logs for gates (see `scripts/benchmarks/aggregate_tip3p_tightening_logs.py`). That flag applies to **per-run `profile_id`** inside `runs[]` — do not conflate with `tip3p_ke_compare` top-level `meta.profile_id` (same string semantics, different JSON layout).

## P2a-B2-R and P2a-B2-X (normative implementation)

Canonical evaluation lives in **`scripts/benchmarks/tip3p_ke_gates.py`** (also callable as `python scripts/benchmarks/tip3p_ke_gates.py <artifact.json>`). The CLI auto-detects **`tip3p_ke_compare/v1`**-shaped JSON vs **`tip3p_tightening_aggregate/v1`** output from the aggregator.

- **P2a-B2-R (per-engine, stretch):** For **`tip3p_ke_compare`**, let `T_eng` be `diagnostics[].replicate_temperature.mean` and `T_bath` be `meta.benchmark_policy.temperature_k` (must equal `config.temperature_k` when both exist). For **`tip3p_tightening_aggregate/v1`**, let `T_eng` be `statistics.fmean` of `aggregated[].per_replica_mean_T_K` for that engine (must match `mean_T_K_mean` in the row), and `T_bath` be `meta.benchmark_policy.temperature_k`. **Pass** if `|T_eng - T_bath| / T_bath <= 0.0167` (±1.67%). **Both** OpenMM and Prolix must pass for “R-both.”

- **P2a-B2-X (cross-engine, informational unless elevated):** Same inequality as **G4** above: `|mean_T_omm - mean_T_plx| <= 2 * sqrt(SEM_omm² + SEM_plx²)` using replicate-level means and SEMs (`replicate_temperature` for compare JSON; `per_replica_mean_T_K` for aggregate JSON). **Sprint / merge policy:** X may fail while R is tracked separately; treat X as informational for release wording unless the team explicitly elevates it to a blocking gate.

## Operator runbook (profile ↔ CLI ↔ defaults)

| `profile_id` | `tip3p_ke_compare.py` | OpenMM `removeCMMotion` | Prolix `remove_linear_com_momentum` |
|----------------|-------------------------|-------------------------|-------------------------------------|
| `diag_linear_com_off` | `--remove-cmmotion false` | false | false |
| `openmm_ref_linear_com_on` | `--remove-cmmotion true` | true | true |

**Production `SimulationSpec`:** `remove_linear_com_momentum` defaults to **False** until a separate product decision; benchmark “primary profile” behavior is **opt-in** via CLI / explicit spec fields.

**Epic C trigger:** If **P2a-B2-R** fails for `openmm_ref_linear_com_on` on a committed Tier‑1 JSON (or documented B-e2e artifact), run the short hypothesis pass in [`tip3p_epic_c_hypotheses.md`](tip3p_epic_c_hypotheses.md). **Do not** auto-start Epic C on P2a-B2-X failure alone unless the team requests deeper cross-engine investigation.

## Non-goals

- **Angular** center-of-mass motion removal is out of scope for this policy (linear momentum only, analogous to OpenMM `CMMotionRemover` default behavior).
- **Bitwise** phase-space identity between OpenMM `CMMotionRemover` scheduling and Prolix post-SETTLE subtraction is **not** claimed; observables within the registered tolerances are the contract.
