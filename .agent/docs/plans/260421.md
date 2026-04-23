# Explicit-solvent validation — execution plan (2026-04-21)

**Status:** Oracle-approved (cycle 4). **Companion critique:** `.agent/critiques/260421.md`  
**Master reference:** `.agent/docs/plans/explicit_solvent_validation_comprehensive.md`  
**TIP3P policy:** `docs/source/explicit_solvent/tip3p_benchmark_policy.md`

---

## 1. Strategic frame

**Release blockers:** **P1 + P5a** (and **P5b** for the release artifact). Per `tip3p_benchmark_policy.md`, **P2a-B2-R / P2a-B2-X** are **not** merge/release blockers unless leadership elevates them.

**Two lanes for TIP3P Langevin / thermometer work:**

| Lane | Role | Artifacts |
|------|------|-----------|
| **CI / regression / tooling** | Normative code path for gates in tests and short reproducible runs | `tip3p_ke_compare.py` → JSON `tip3p_ke_compare/v1` → `tip3p_ke_gates.py` |
| **Long equilibrium / cluster** | Preferred substrate for credible long-run and **external** claims | `tip3p_langevin_tightening.py` → tee logs → `aggregate_tip3p_tightening_logs.py` → aggregate gate path + versioned schema |

**Epic C:** Only if **P2a-B2-R fails for OpenMM** on a committed **primary-profile** Tier‑1 artifact — not on **P2a-B2-X** alone.

---

## 2. Repo ground truth

- **P1a-style hard gate:** `.github/workflows/openmm-nightly.yml` job **`explicit_solvent_parity`** has **no** `continue-on-error`; runs 1UAO integration parity, full `test_explicit_langevin_tip3p_parity.py`, and `export_regression_pme.py --check`.
- **Soft / informational:** job **`openmm`** uses `continue-on-error: true`.
- **PME SSOT:** `REGRESSION_EXPLICIT_PME` lives in **`src/prolix/physics/regression_explicit_pme.py`**; `conftest.py` and benchmark scripts import it (no duplicated literals).
- **`tip3p_ke_gates.evaluate_payload`:** Consumes **compare-shaped** JSON only; **tightening aggregates** need an adapter + `tip3p_tightening_aggregate/v1` (W3).
- **Estimator alignment:** Per-replica `mean_T_K` (tightening) matches inputs to `tip3p_ke_compare`’s `replicate_temperature` (fmeans); **X** uses SEM on replicate means as in `tip3p_ke_gates.p2a_b2_x_g4_passes`.

---

## 3. Workstreams

| ID | Scope | Exit |
|----|--------|------|
| **W1** | Release CP: nightly green, P1b, P5a runbook, P5b | Master plan checklist |
| **W2** | PME dedup → `regression_explicit_pme.py` + export script reads it | Done when merged |
| **W3** | Aggregate schema `tip3p_tightening_aggregate/v1`; extend `tip3p_ke_gates`; golden tests; **required** step in `explicit_solvent_parity` or fast PR CI | PR |
| **W4** | Slurm Tier‑1 tightening matrix; archive SHA + logs + aggregate | Artifact memo |
| **W5** | Epic C if OpenMM R fails | Hypotheses doc |
| **W6** | P2b / P3 / P4 — **post-CP, capacity-gated** | Non-blocking |

---

## 4. Release / comms tree

Policy: **P2a-B2 gates do not block release.**

| Situation | Engineering | Comms |
|-----------|-------------|--------|
| **OpenMM R fails** | Epic C | Do **not** imply OpenMM bath equilibrium for that profile until closed |
| **Prolix R fails, OpenMM passes** | May ship if P1+P5a pass | **Forbidden:** rigid-thermometer mean T matches bath under **primary profile** for Prolix. **Allowed:** P1 parity, policy link, which leg passed **R** |
| **Both R pass, X fails** | Informational | Per policy; no Epic C from X alone |

**Product default:** `SimulationSpec.remove_linear_com_momentum` is **False** until a product decision; benchmark primary profile is **opt-in**.

---

## 5. Public Tier‑1 reproducibility tuple

Freeze for papers/slides: `{ git SHA, profile_id, n_waters, total_steps, burn_in, sample_every, schema ids (compare + aggregate), full CLI including aggregate --require-profile-id }`.

Richer equilibrium stats (e.g. ESS): defer to P2b / L3 era.

---

## 6. Pointer table (blockers)

| ID | Artifact |
|----|----------|
| **P1a** | `openmm-nightly` → `explicit_solvent_parity` |
| **P1b** | `REGRESSION_EXPLICIT_PME` + `export_regression_pme` + protocol doc |
| **P5a** | `explicit_solvent_runbook.md` |
| **P5b** | Release notes |

---

## 7. Resolved decision (D1)

**External / long-equilibrium claims** → tightening + aggregate + gate output. **CI / regression** → `tip3p_ke_compare` + `tip3p_ke_gates`.

---

## 8. Implementation follow-up (post-merge)

- [x] **W2 (2026-04-21):** `REGRESSION_EXPLICIT_PME` centralized in `src/prolix/physics/regression_explicit_pme.py`; `conftest`, `tip3p_ke_compare`, `tip3p_langevin_tightening`, and `export_regression_pme.py` updated; runbook + protocol prose SSOT pointers refreshed.
- [x] **W3 (2026-04-21):** `tip3p_tightening_aggregate/v1` on aggregator stdout; `evaluate_tightening_aggregate` + CLI auto-detect in `tip3p_ke_gates.py`; fixtures under `tests/fixtures/tip3p_tightening_aggregate/`; `openmm-nightly` `explicit_solvent_parity` runs `test_tip3p_ke_gates.py`.
- [ ] **Doc sweep:** Historical `.agent` / root copies of the comprehensive plan may still cite `conftest.py:140`; update when those files are next edited.
