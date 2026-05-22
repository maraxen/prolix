# TIP3P KE diagnostic — x64 rerun decision (local CPU)

**Date:** 2026-04-21  
**Script:** `scripts/benchmarks/tip3p_ke_compare.py`  
**Artifacts:** `outputs/logs/local_tip3p_ke/tier{0,1}_*.json` (generated on the machine that ran the matrix)

**Pre-registered policy:** [`tip3p_benchmark_policy.md`](tip3p_benchmark_policy.md) (primary profile `openmm_ref_linear_com_on`, diagnostic `diag_linear_com_off`). New JSON summaries include `meta.profile_id` for gate hygiene; use filename tags such as `*_diag_linear_com_off.json` / `*_openmm_ref_linear_com_on.json` when archiving.

## What changed in the diagnostic

- **Pre-import JAX x64:** `--jax-x64 {off,on}` is applied *before* importing JAX/JAX-MD, so x64 mode is trustworthy.
- **Prolix step JIT:** `apply_s` from `settle_langevin` is wrapped in `jax.jit` so CPU benchmarks are not dominated by Python overhead.
- **Replicate-level inference:** For `replicas>1`, ratio and temperature summaries use the **mean per replica** (one scalar per seed), then **SEM** and **95% CI** with `t_{0.975, n-1}` (`replicate_t95` in JSON).
- **Frame-level diagnostic:** Pooled ratio samples use **block SEM** when `n_blocks >= 8`; otherwise IID SEM with a warning.
- **Timing:** `timing_mode` = `cold` / `steady` / `both`; steady throughput uses `measure_steps` after `warmup_steps`.
- **OpenMM-only path:** Boltzmann constant is inlined so OpenMM runs do not import `prolix.simulate` only for a constant.

### Option B calibration (COM / dynamics parity controls, post-2026-04-21)

These flags make the benchmark **self-describing** and align OpenMM with Prolix on **center-of-mass policy** (the leading hypothesis for the historical G4 soft failure).

- **`--remove-cmmotion {true,false}`:** forwarded to OpenMM `ForceField.createSystem(removeCMMotion=...)`. Prolix uses the same boolean for `settle_langevin(remove_linear_com_momentum=...)` so `profile_id` stays self-consistent. Default **`false`** matches historical diagnostic parity (no OpenMM `CMMotionRemover`, no Prolix COM subtraction). The JSON records both the requested OpenMM flag and whether a `CMMotionRemover` force is actually present.
- **`--dt-fs`:** integrator step in femtoseconds; **same** value is used for OpenMM and Prolix (default `2.0`, unchanged from the original hard-coded Tier‑1 protocol).
- **`--temperature-k`, `--gamma-ps`:** explicit bath temperature and friction (1/ps); Prolix uses the same `gamma_reduced = gamma_ps * AKMA_TIME_UNIT_FS * 1e-3` mapping as production `prolix.simulate` logging.
- **`meta.benchmark_policy.gamma_dt_consistency`:** fail-fast check that `gamma_reduced * dt_akma == gamma_ps * dt_ps` (dimensionless damping per step).
- **`--openmm-integrator {middle,langevin}`:** diagnostic toggle between `LangevinMiddleIntegrator` and legacy `LangevinIntegrator` (default `middle`).
- **`--verbose-samples`:** adds per-sample `temperature_atomic` (uses `dof_atomic = 3 * N_atoms`, **not** the G4 gate), plus COM-related series where applicable.

**Classification:** `diag_linear_com_off` remains **`diagnostic_calibration`** relative to typical OpenMM app defaults. The **`openmm_ref_linear_com_on`** lane matches `removeCMMotion=True` with Prolix linear COM subtraction (see `tip3p_benchmark_policy.md`); it is the binding profile for production-style claims once B2 matrices are complete.

### Tier 1 matrix note (historical vs Option B default)

The Tier‑1 numbers in the table below were produced **before** explicit `removeCMMotion` control: OpenMM used the **application-layer default** `removeCMMotion=True` (adds `CMMotionRemover`), while Prolix had **no** COM remover. The Option B default (`--remove-cmmotion false`) is **not comparable** to those JSON files without relabeling. Re-run Tier‑1 under the new flags to refresh G4.

**G4 after Option B:** *pending* full Tier‑1 re-execution on a machine with OpenMM installed. A short local smoke of the Prolix path after the change succeeds and emits the new `meta.benchmark_policy` block.

**B1 (`diag_linear_com_off`):** Tier‑0/1 re-runs with `--remove-cmmotion false` on OpenMM and matching Prolix (`remove_linear_com_momentum` off) — **diagnostic / CI lane**; refresh G4 for this profile only.

**B2 (`openmm_ref_linear_com_on`):** After Prolix `settle_langevin(..., remove_linear_com_momentum=True)` is available, re-run Tier‑0/1 with `--remove-cmmotion true` and the same flag on Prolix (wired automatically in `tip3p_ke_compare.py`). **Primary** release-style G4 language applies to this profile only.

**CI:** `tests/physics/test_tip3p_com_linear_momentum.py` encodes the short paired OpenMM vs Prolix check for the primary profile (see benchmark policy tolerances).

### Tier 1 smoke (single replica, `replicas=1`, 2026-04-21, local CPU)

Artifacts under `outputs/logs/local_tip3p_ke/`: `tier1_both_diag_linear_com_off_x64_r1.json`, `tier1_both_openmm_ref_linear_com_on_x64_r1.json`. Each run includes `meta.profile_id` and `benchmark_policy.prolix_remove_linear_com_momentum`.

| Profile | Mean T OpenMM (K) | Mean T Prolix x64 (K) | G4 (soft) |
|---------|-------------------|------------------------|-----------|
| `diag_linear_com_off` | ~304.2 | ~409.8 | FAIL |
| `openmm_ref_linear_com_on` | ~304.7 | ~395.5 | FAIL |

Linear COM removal on Prolix moves mean T slightly but does **not** alone close G4 at this protocol. Decision-grade runs should use `replicas=5` per [`tip3p_benchmark_policy.md`](tip3p_benchmark_policy.md).

### Tier 1 decision-grade (Epic A, `replicas=5`, Prolix x64, 2026-04-21)

Repro: `scripts/benchmarks/run_tip3p_ke_tier1_sprint.sh` (sets `OPENMM_INSTALL_MODE=ephemeral`, writes JSON + `tier1_sprint_sha256sums.txt` under `outputs/logs/local_tip3p_ke/`).

SHA256 (see also `outputs/logs/local_tip3p_ke/tier1_sprint_sha256sums.txt`):

- `tier1_both_diag_linear_com_off_x64_r5.json` — `8527414d8ceb79d25a9c9d69a72ee94df7e8b60963b6efe4e6149f55cafa3b85`
- `tier1_both_openmm_ref_linear_com_on_x64_r5.json` — `20ec6e1547c28ee70999cc78d2538f85b4358505ccac930fb699512ab3a5d35d`

| Profile | G1 OpenMM ratio CI∋1 | G2 Prolix ratio gate | G4 soft (mean-T SEM) | P2a-B2-R (both engines ±1.67% bath) | P2a-B2-X (=G4) |
|---------|----------------------|----------------------|------------------------|--------------------------------------|----------------|
| `diag_linear_com_off` | PASS | PASS | FAIL | FAIL | FAIL |
| `openmm_ref_linear_com_on` | PASS | PASS | FAIL | FAIL (Prolix) | FAIL |

**G3:** not refreshed this sprint (optional Prolix x32 Tier 1 deferred). **G5:** no `replicate_warning` / frame warnings in these JSON summaries.

Means (K): diagnostic lane — OpenMM `≈308.3`, Prolix `≈416.1`; primary lane — OpenMM `≈304.6`, Prolix `≈403.6`. Evaluate gates with `uv run python scripts/benchmarks/tip3p_ke_gates.py <path.json>` (see [`tip3p_benchmark_policy.md`](tip3p_benchmark_policy.md)).

## Run matrix (executed)

| Tier | OpenMM | Prolix x32 | Prolix x64 |
|------|--------|------------|------------|
| 0 (`n_waters=4`, `steps=3000`, `burn=1200`, `sample_every=20`, `replicas=1`) | `tier0_openmm.json` | `tier0_prolix_x32.json` | `tier0_prolix_x64.json` |
| 1 (`n_waters=33`, `steps=30000`, `burn=10000`, `sample_every=10`, `replicas=5`) | `tier1_openmm.json` | `tier1_prolix_x32.json` | `tier1_prolix_x64.json` |

OpenMM local runs used `OPENMM_INSTALL_MODE=ephemeral` and `uv run --with openmm` where the project venv lacks OpenMM.

## Tier 0 snapshot (exploratory)

| Engine | Replicate mean `KE_atomic/KE_rigid` (Prolix) or `KE_getState/Σ½mv²` (OpenMM) | Replicate mean T (K) |
|--------|-----------------------------------------------------------------------------|------------------------|
| OpenMM | 1.0 | ~331.5 |
| Prolix x32 | ~1.000035 | ~326.6 |
| Prolix x64 | ~1.000038 | ~333.4 |

Tier 0 is under-powered for formal gates (`replicas=1`) but shows **no large x32 vs x64 mismatch** in the KE ratio at 4 waters.

## Tier 1 gates (decision-grade, `replicas=5`)

Definitions:

- **G1 (hard):** OpenMM ratio replicate CI contains 1.0.
- **G2 (hard):** Prolix x64: `|mean - 1| <= max(3 * SEM, 2e-4)` on replicate-mean ratios.
- **G3 (hard):** x64 non-regression: `|mean_x64 - 1| <= |mean_x32 - 1| + 2 * max(SEM_x32, SEM_x64)`.
- **G4 (soft):** `|mean_T(OpenMM) - mean_T(Prolix x64)| <= 2 * sqrt(SEM_omm² + SEM_plx²)` using replicate-mean temperatures.
- **G5:** Sample adequacy warnings (replicas < 5, block fallback) — none on Tier 1.

### Observed Tier 1 outcomes

- **G1:** PASS (ratio mean 1.0, CI `[1,1]` — OpenMM self-consistency is exact in double arithmetic on Reference).
- **G2:** PASS for Prolix x64 (`mean ≈ 1.0000254`, SEM `≈ 6.35e-7`).
- **G3:** PASS (x64 mean slightly *closer* to 1 than x32).
- **G4:** FAIL — mean temperatures diverge strongly (OpenMM `≈ 305 K` vs Prolix rigid thermometer `≈ 416 K` on this protocol). This is **not** explained by the KE ratio gate; it points to **thermostat / trajectory / observable mismatch** between engines under this script’s dynamics setup, not a `float32` KE bookkeeping bug.

### Precedence verdict

- **Hard gates:** PASS (G1–G3).  
- **Soft gate:** FAIL (G4).  
- **Overall:** **Proceed with risk** — treat the original `~0.00004` ratio residual as **numerical / sampling** at small `n_waters`; **do not** use cross-engine mean-T agreement from this script alone as a parity claim until the Langevin + SETTLE path is aligned (see follow-on).

## Follow-on implemented

- **Static force sanity (OpenMM vs Prolix PME):** `tests/physics/test_explicit_langevin_tip3p_parity.py::test_openmm_prolix_tip3p_force_rmse_one_step`  
  - `@pytest.mark.slow` + `@pytest.mark.openmm`  
  - One configuration snapshot, force RMSE `< 3.0` kcal/mol/Å  
  - Writes `tip3p_parity_params.json` under `tmp_path` for parameter equivalence capture in CI logs.

## Residual risks / next actions

1. **Cross-engine Langevin mean-T** on this diagnostic still disagrees at Tier 1 — investigate `settle_langevin` vs OpenMM `LangevinMiddleIntegrator` + constraints (friction units, constraint coupling, COM motion, burn-in length), or promote the existing P2a-B2 tightening harness (`tip3p_langevin_tightening.py`) as the authoritative thermostat comparison.
2. **JAX scatter dtype warning** under x64 (`float64` vs `float32` in scatter) — monitor; may require dtype alignment in `settle` / exclusion masks for future JAX versions.
3. **Re-run OpenMM JSON** after `openmm_version` was added to summaries if you need version metadata in old Tier 1 files (field is present on new runs).

## Commands to reproduce Tier 1

### Option B default (explicit COM parity: `removeCMMotion=false`)

```bash
mkdir -p outputs/logs/local_tip3p_ke

OPENMM_INSTALL_MODE=ephemeral uv run --with openmm python scripts/benchmarks/tip3p_ke_compare.py \
  --engine openmm --jax-x64 off --n-waters 33 --steps 30000 --burn 10000 --sample-every 10 --replicas 5 \
  --timing-mode both --warmup-steps 100 --measure-steps 500 \
  --remove-cmmotion false --dt-fs 2.0 --temperature-k 300 --gamma-ps 1.0 --openmm-integrator middle \
  > outputs/logs/local_tip3p_ke/tier1_openmm_optionB.json

uv run python scripts/benchmarks/tip3p_ke_compare.py \
  --engine prolix --jax-x64 off --n-waters 33 --steps 30000 --burn 10000 --sample-every 10 --replicas 5 \
  --timing-mode both --warmup-steps 100 --measure-steps 500 \
  --remove-cmmotion false --dt-fs 2.0 --temperature-k 300 --gamma-ps 1.0 \
  > outputs/logs/local_tip3p_ke/tier1_prolix_x32_optionB.json

uv run python scripts/benchmarks/tip3p_ke_compare.py \
  --engine prolix --jax-x64 on --n-waters 33 --steps 30000 --burn 10000 --sample-every 10 --replicas 5 \
  --timing-mode both --warmup-steps 100 --measure-steps 500 \
  --remove-cmmotion false --dt-fs 2.0 --temperature-k 300 --gamma-ps 1.0 \
  > outputs/logs/local_tip3p_ke/tier1_prolix_x64_optionB.json
```

### Legacy OpenMM application default (for comparison only: `removeCMMotion=true`)

```bash
OPENMM_INSTALL_MODE=ephemeral uv run --with openmm python scripts/benchmarks/tip3p_ke_compare.py \
  --engine openmm --jax-x64 off --n-waters 33 --steps 30000 --burn 10000 --sample-every 10 --replicas 5 \
  --timing-mode both --warmup-steps 100 --measure-steps 500 \
  --remove-cmmotion true --dt-fs 2.0 --temperature-k 300 --gamma-ps 1.0 --openmm-integrator middle \
  > outputs/logs/local_tip3p_ke/tier1_openmm_legacy_cm.json
```
