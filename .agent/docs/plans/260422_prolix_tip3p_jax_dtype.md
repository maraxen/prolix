# Prolix TIP3P cluster thermometer — JAX / PME dtype follow-up (2026-04-22)

**Status:** Oracle cycle 3 **APPROVE** (execution-ready with suggestions).  
**Context:** Engaging runs with `JAX_ENABLE_X64=1` still show Prolix `mean_T_K` ~430–440 K vs bath 300 K; OpenMM ~303 K passes P2a-B2-R. JAX **`FutureWarning`** on scatter: `float64` values into **`float32`** `Q` in `spread_charges`.

---

## 1. Ground truth (repo)

- **`physics/pme.py` `spread_charges`:** `Q = zeros(..., float32)`; `q = charges * atom_mask.astype(float32)` — if **`charges` is float64**, `q` / `w` promote to float64 → **`Q.at[...].add(w)`** warns (JAX scatter promotion).
- **`tip3p_langevin_tightening.py`:** builds water **`charges`/`sigmas`/`epsilons` as float64**; no **`jax.config.update`** today (relies on env). **`tip3p_ke_compare`** uses **`_configure_jax`** with **`requested_jax_x64` / `effective_jax_x64` / `jax_backend`**.
- **`tests/physics/test_explicit_langevin_tip3p_parity.py::test_openmm_prolix_tip3p_force_rmse_one_step`:** sets **`jax_enable_x64` True**; RMSE gate **&lt; 3.0** kcal/mol/Å — good **Tier A regression** after PME tweak.
- **`tip3p_epic_c_hypotheses.md`:** hypothesis **5** = JAX dtype / scatter; hypotheses **2–4** remain if dtype fixes do not move **T**.

---

## 2. Goals

1. Remove **silent f64→f32 scatter** on the scalar explicit PME path used by tightening (or document waiver).
2. Bring **Prolix P2a-B2-R** into policy band on Engaging **or** falsify dtype and pivot Epic C to integrator/SETTLE/burn-in.

---

## 3. Workstreams (ordered)

### W0 — Harness observability (small PR, blocking for Phase 0)

- Add **`--jax-x64 {off,on}`** to **`tip3p_langevin_tightening.py`**.
- Call **`jax.config.update("jax_enable_x64", …)` at the start of `main()` after `parse_args`**, before **`run_prolix` / JIT** (not only at module import — CLI must apply).
- Extend final printed JSON with **`jax`**: `requested_jax_x64`, `effective_jax_x64`, `jax_backend` — **same semantics as** **`tip3p_ke_compare._configure_jax`**.
- **Slurm / env:** Keep **`JAX_ENABLE_X64`** documented; state that **env must match CLI** for logs (or unset env when using CLI-only).

**Note:** **`tip3p_ke_profile`** is imported before **`jax`** today and does **not** import JAX; if that changes, reorder or configure JAX earlier.

### W1 — Tier A (surgical PME)

- **Single change:** In **`spread_charges`**, ensure **`w`** (and/or **`q`**) is cast to **`Q.dtype`** before **`Q.at[...].add(w)`**.
- **Gates:**
  1. Scatter **FutureWarning** gone (or explicitly waived with reason).
  2. **CI:** **`test_openmm_prolix_tip3p_force_rmse_one_step`** still passes.
  3. **Dynamics smoke:** Prolix **P2a-B2-R** on a short Engaging run:  
     `|mean_T_K - T_bath| / T_bath ≤ 0.0167` with **`T_bath`** / **`profile_id`** per **`tip3p_benchmark_policy.md`** (not a hand-wavy “moves toward bath”).

### W2 — Phase 0a (falsification, before claiming Tier A fixes T)

- **Single replica**, **`n_waters=33`**, **~5k steps**, same cluster settings.
- Two checkpoint roots: e.g. **`TIP3P_RUN_TAG`** / **`TIP3P_RUN_ROOT`** **`ab_x64_off`** vs **`ab_x64_on`**; vary **only** **`JAX_ENABLE_X64`** (**0** vs **1**) or **`--jax-x64`**.
- Compare **Prolix** `mean_T_K` in printed summaries. If **no delta**, dtype/x64 is **unlikely** sole cause → prioritize Epic C **2–4**.

### W3 — Phase 0b (Tier A verification)

- Paired runs at **`merge-base` SHA** vs **Tier A merge commit SHA** (record in memo).
- Same hardware class; isolate checkpoints with a fresh **`TIP3P_RUN_TAG`** (e.g. `post_tierA_<shortsha>`).
- Gates: (1) warnings; (2) CI force RMSE; (3) Prolix **P2a-B2-R** on agreed smoke **or** explicit failure → Epic C.

### W4 — Tier B / C (conditional)

- **Tier B:** `work_dtype` through **`make_energy_fn`** only if Tier A passes (1)(2) but fails (3) **or** audit demands end-to-end dtype — requires design note and subsystem tests.
- **Tier C:** **`JAX_PLATFORMS=cpu`** bisect **GPU vs policy**; any CPU “gold” run must be followed by **GPU re-check** before Engaging Tier‑1 claims.

### W5 — Rollout

- Full array **`submit_tip3p_chain`**, **`aggregate_tip3p_tightening_logs.py --require-profile-id`**, **`tip3p_ke_gates.py`**.

---

## 4. Operator defaults

- Prefer **unified `both`** engine per array task for paired OpenMM/Prolix JSON.
- Use **one job family** for x64 A/Bs to reduce environmental variance; split jobs only for throughput **after** hypothesis is locked.

---

## 5. Oracle history

| Cycle | Verdict | gist |
|-------|---------|------|
| 1 | REVISE | Add falsification before Tier B; Tier A necessary but not sufficient for ~130 K bias |
| 2 | REVISE | Quantify gate (3) with P2a-B2-R; name mergeable Tier A; align `jax_meta` keys; note force test is x64-on |
| 3 | APPROVE | v3 + `main()` config timing + SHA-paired verification |

---

## 6. Owner decisions

| ID | Topic | Importance | Tradeoffs | Recommendation |
|----|--------|------------|-----------|----------------|
| **D1** | Tier A only vs Tier B if A insufficient | **High** | A is fast, low blast radius; B touches much of explicit path | **Ship Tier A first**; open Tier B only if gates (3) still fail with clean warnings + passing force RMSE |
| **D2** | Slurm: env-only x64 vs `--jax-x64` | **Medium** | Duplication if both set | **CLI as source of truth in logs**; Slurm sets `JAX_ENABLE_X64` **or** passes flag consistently — **document one blessed pattern** |
| **D3** | Smoke step count (5k vs 50k) for gate (3) | **Medium** | Shorter = faster falsification; may miss slow bias | **5k for Phase 0a/b**; **50k** for final Tier‑1 aggregate |

---

## 7. Artifacts to archive

`{commit SHA, profile_id, JAX flags, jax block in JSON, TIP3P_RUN_TAG, aggregate schema, tip3p_ke_gates stdout}` per **`tip3p_benchmark_policy.md`** reproducibility tuple.
