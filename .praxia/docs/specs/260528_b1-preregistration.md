# B1 Pre-Registration — Init-Bound Heterogeneous MD Throughput (Claim 1)

**Status:** Binding pre-registration (2026-05-28)  
**Task ID:** `260528_s5_paper_gate`  
**Amendments:** Any deviation from pinned fields below requires an explicit spec amendment PR before cluster spend.

---

## Scope boundary

| In scope | Out of scope |
|----------|----------------|
| Claim 1 headline: heterogeneous batching + init-bound wall-clock win | §7.1 external comparator (bonded Scope A; campaign `edbd0b84`) |
| `EnsemblePlan` / `BatchPlanner` / `Bundle` / `safe_map` | Long single-trajectory throughput (Lane B MD production) |
| Cold-start wall-clock vs OpenMM multi-Context baseline | NPT / SETTLE production stability (v1.1 tracks) |

§7.1 figure gate is **complete** when `outputs/analysis/s71_external_comparator.png` and `s71_compatibility_matrix.md` pass the 0.5× prolix-vs-external rule on primary hardware.

---

## Hypothesis (falsifiable)

**H1:** For a fixed ensemble of 64 heterogeneous systems (4 topology classes × 16 replicas), prolix cold-start `t_total` (median over 3 seeds) is lower than OpenMM’s equivalent cold-start protocol with **N independent Contexts** (no Context reuse), in the short-trajectory regime where initialization and compilation dominate (100 ps per system).

**H0:** No statistically meaningful wall-clock advantage, or prolix advantage is entirely explained by `t_aot_compile / t_total > 0.5` (compile-dominated artifact).

---

## Cadence split (locked)

| Cadence | B | Harness | Tracking | Purpose |
|---------|---|---------|----------|---------|
| **B1-smoke** | 4 mixed bundles | pytest (`tests/bench/` or dedicated module) | CI nightly | Regression + AOT-ratio alert |
| **B1-full** | 64 (4 types × 16) | `scripts/benchmarks/b1_init_exec.py` + `.bth.toml` | `bth run` + confirmation campaign | Paper headline number |

Do **not** use pytest for B1-full cluster runs (bathos provenance required).

---

## Pinned protocol (B1-full)

| Field | Value |
|-------|--------|
| Ensemble | 64 systems: `{1ake (~3k atoms), 1ubq (~1.4k), 2gb1 (~560), 4-water (~12)}` × 16 each |
| Trajectory | 100 ps per system (short-trajectory / init-bound regime) |
| Seeds | 3 minimum (report median + min/max envelope) |
| Primary metric | `t_total` (wall-clock, seconds) |
| Sub-metrics (reported separately) | `t_ff_load`, `t_bundle_construct`, `t_aot_compile`, `t_first_step`, `t_steady_state` |
| Cold-start | Fresh Python process; `JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_$(uuidgen)`; OS page cache drop (`sync && echo 3 > /proc/sys/vm/drop_caches` on Linux) |
| OpenMM baseline | N independent `Context` objects (varying topology); FF load + Context creation per system; **no reuse** |
| XLA flags | `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_FLAGS=--xla_gpu_enable_triton_softmax_fusion=false` |
| Hardware (primary) | H100 |
| Hardware (secondary) | RTX 4090 |
| Output CSV | `outputs/bench/b1_full.csv` |
| Re-run trigger | Any change to `EnsemblePlan`, `BatchPlanner`, `Bundle`, or `safe_map` integration |

---

## Success criteria

1. **Headline:** median `t_total` (B1-full, 3 seeds) prolix < OpenMM baseline.
2. **AOT guard:** `t_aot_compile / t_total < 0.5` on B1-full. Breach → escalate to R4 sub-spec (roadmap §2.4) before paper submit.
3. **B1-smoke:** green in nightly CI with same AOT-ratio alert threshold.

---

## Bathos (B1-full only)

- **Script:** `scripts/benchmarks/b1_init_exec.py` (scaffold when implementation sprint starts)
- **Sidecar:** `[benchmark]` or `[experiment]` with `result_schema`: `t_total`, `t_aot_compile`, `aot_ratio`, `seed`, `hardware_tag`
- **Campaign:** `bth campaign create "b1-full" --mode confirmation --hypothesis "<H1 text>"`
- **Cluster:** slurm wrapper following `scripts/slurm/fit_bonded_hp4.slurm` (`_bth_env.sh` + `uv run bth run`)

---

## Plan-auditor checklist

- [x] Distinct from §7.1 external comparator
- [x] B1-smoke vs B1-full cadence explicit
- [x] Falsifiable hypothesis + H0
- [x] AOT-ratio guard pinned
- [x] Deviations require amendment

**Verdict:** PASS (document-only phase; implementation deferred to follow-on sprint)
