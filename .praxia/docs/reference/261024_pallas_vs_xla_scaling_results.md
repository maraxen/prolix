# Pallas vs XLA scaling results (Phase 3 harness) — 2026-10-24

Harness: `scripts/benchmarks/pallas_vs_baseline.py` (`--kernel nonbonded|pme`).
Raw JSON: `docs/benchmarks/261023_engaging_nonbonded_pallas_vs_xla.json`,
`docs/benchmarks/261024_engaging_pme_pallas_vs_xla.json`.

**Method.** Identical inputs to both paths: jittered cubic lattice at 0.1 atoms/Å^3
(water-like), LJ sigma 1.6–2.0 Å, neutral random charges, cutoff 9 Å, switch 1.5 Å,
PME alpha 0.34, float32. Timing is the median of 5 steady-state `jit(value_and_grad)`
calls after a warm-up call (compile excluded). Host: engaging `pi_so3` GPU node
(node5101, JAX 0.10.2, Triton lowering). Parity is the XLA-vs-Pallas difference in
energy and forces on the same positions.

## Fused NL nonbonded kernel (`pallas_nonbonded_nl`) vs chunked XLA reference

| atoms | K | XLA ms | Pallas ms | speedup | rel dE | max abs dF |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 392 | 0.84 | 1.24 | 0.67x | 1.1e-6 | 4.3e-5 |
| 2,048 | 424 | 1.39 | 1.19 | 1.17x | 3.2e-7 | 5.3e-5 |
| 8,192 | 408 | 2.79 | 1.35 | 2.07x | 4.5e-7 | 9.5e-5 |
| 32,768 | 448 | 9.55 | 2.99 | 3.19x | 1.2e-7 | 2.7e-4 |
| 131,072 | 448 | 35.5 | 9.34 | 3.80x | 2.1e-7 | 3.9e-4 |

Fitted time-vs-N exponent: XLA 0.68, Pallas 0.36. Both are below the ideal 1.0 because
the GPU is under-occupied at small N, and the Pallas curve is flatter. Crossover is near
1k atoms; below that XLA wins (launch-bound). Parity is at float32 round-off throughout.

## SPME spread/gather (`use_pallas_pme`) vs XLA scatter/gather

| atoms | XLA ms | Pallas ms | speedup | rel dE |
|---:|---:|---:|---:|---:|
| 512 | 0.095 | 0.102 | 0.93x | 0 |
| 2,048 | 0.120 | 0.110 | 1.09x | 1.8e-7 |
| 8,192 | 0.145 | 0.142 | 1.02x | 2.4e-7 |
| 32,768 | 0.186 | 0.167 | 1.12x | 0 |
| 131,072 | 0.311 | 0.257 | 1.21x | 0 |

**Conclusion: the PME Pallas kernels are parity-correct but not a performance lever.**
Reciprocal-space time is dominated by the FFT and XLA's scatter is already efficient;
spread/gather speedup is within 0.93–1.21x. The nonbonded NL kernel is where Pallas pays
(up to 3.8x at 131k atoms), so further Pallas coverage effort should target
neighbor-list-side O(N*K) work, not PME.

## Neighbor-list rebuild cost vs the Pallas nonbonded step (`--kernel nl_build`)

Raw JSON: `docs/benchmarks/261024_engaging_nl_build.json`. Times a `jax.jit`'d
`make_neighbor_list_fn(...).update` (jax_md cell list, 9 Å cutoff, `dr_threshold` 0.5) on
the same engaging GPU. The timed call is a **genuine rebuild** (all atoms shifted 0.4 Å per
axis, past the skin); `nl_skin_check_only` is the no-op branch at unmoved positions.

| atoms | rebuild ms | skin check ms | Pallas nonbonded step ms | rebuild / step | share of step @10 / 20 / 50 steps per rebuild |
|---:|---:|---:|---:|---:|---|
| 512 | 0.080 | 0.061 | 0.445 | 0.18 | 1.8% / 0.9% / 0.4% |
| 2,048 | 0.101 | 0.063 | 0.471 | 0.21 | 2.1% / 1.1% / 0.4% |
| 8,192 | 1.08 | 0.070 | 0.614 | 1.76 | 14.9% / 8.1% / 3.4% |
| 32,768 | 4.27 | 0.074 | 2.14 | 1.99 | 16.6% / 9.1% / 3.8% |
| 131,072 | 16.5 | 0.398 | 8.42 | 1.96 | 16.4% / 8.9% / 3.8% |

**Decision: no-go on a Pallas neighbor-list builder (for now).** A rebuild costs ~2x one
fused nonbonded step at >= 8k atoms, but it runs only every 10–50 steps, so even a
zero-cost builder removes at most ~16% of step time (1.2x) at 10 steps/rebuild and under 4%
at 50. A Pallas builder would also need a cell sort, a capacity-overflow protocol and
dynamic shapes, none of which Pallas Triton supports natively. Cheaper levers first: a
larger skin / `dr_threshold` to lengthen the rebuild interval, and tuning
`capacity_multiplier`. Revisit only if profiling shows rebuilds every < 10 steps.

**Pitfall recorded:** the first version of this benchmark timed `update()` at unchanged
positions, which hits the no-op `lax.cond` branch and made the rebuild look 5–13% of a step
instead of ~200%. Always move atoms past `dr_threshold / 2` to time a real rebuild.

## Caveats (Resolved in 2026-10-26 Phase 3 Completion)

- `peak_mem_*` fields in the non-isolated JSON were cumulative across paths in one process; resolved below via `--isolate-memory` and `--kernel mem_scaling` (spawning a fresh subprocess per `(path, N)` with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_ALLOCATOR=platform`).
- Single GPU model caveat resolved below by repeating nonbonded, Flash, and kUPS propagator benchmarks across **three GPU generations** on Engaging: **Blackwell (`sm_120`, RTX PRO 6000 Blackwell Server Edition)**, **Hopper (`sm_90`, NVIDIA H200)**, and **Ada Lovelace (`sm_89`, NVIDIA L40S)**.
- The first harness version used uniform-random positions, which produced overlapping atoms and ~1e13 energies where float32 parity is meaningless. Always use a physically-spaced configuration for parity benchmarks.

---

## Per-Process Isolated Memory Scaling (`--kernel mem_scaling`, 2026-10-26)

Raw JSON: [`docs/benchmarks/261026_engaging_mem_scaling.json`](file:///home/marielle/projects/prolix/docs/benchmarks/261026_engaging_mem_scaling.json).
Each `(path, N)` entry runs in a fresh Python subprocess (`--isolate-memory`) with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, recording `baseline_bytes` (after input allocation), `working_peak_bytes` (`peak_bytes_in_use` after `jit(value_and_grad)` execution), and `activation_peak_bytes = working_peak_bytes - baseline_bytes`.

| $N$ (atoms) | `dense_xla` $O(N^2)$ working MB | `flash_xla` $O(N\cdot T)$ working MB | `flash_pallas` $O(N)$ working MB | `nl_xla` $O(N\cdot K)$ working MB (act MB) | `nl_pallas` $O(N\cdot K)$ working MB (act MB) |
|---:|---:|---:|---:|---:|---:|
| 512 | 67.1 | 67.1 | 67.1 | 67.1 (65.8) | 67.1 (65.8) |
| 2,048 | 67.1 | 67.1 | 67.1 | 67.1 (63.6) | 67.1 (63.6) |
| 8,192 | 275.3 | 67.1 | 67.1 | 67.1 (46.6) | 67.1 (46.6) |
| 16,384 | **1,100.1** | 68.1 | **67.1** | 67.1 (26.2) | 67.1 (26.2) |
| 32,768 | OOM / >4.4 GB | 136.6 | **67.1** | 125.8 (65.9) | **125.8 (65.9)** |
| 131,072 | OOM / >68 GB | — | — | 555.0 (281.2) | **536.9 (263.1)** |

- **Quadratic blowup of un-tiled $O(N^2)$ XLA**: `dense_xla` scales with `working_mem_exponent = 2.00` from 8k to 16k (`275.3 MB -> 1,100.1 MB`).
- **Flat $O(N)$ memory in `flash_pallas`**: `pallas_flash_explicit_nonbonded` keeps working peak memory flat at **`67.1 MB`** (`working_mem_exponent = 0.0`) up to 32k atoms ($2.04\times$ lower than `flash_xla`'s `136.6 MB`).
- **Sub-linear / linear $O(N \cdot K)$ memory in `nl_xla` and `nl_pallas`**: At $N=131,072$ ($K=448$), `nl_pallas` uses **`536.9 MB` working (`263.1 MB` activations)** vs `nl_xla`'s `555.0 MB` (`281.2 MB` activations), saving `18.1 MB` by accumulating forces in SRAM registers without materializing $(N, K, 3)$ intermediate force tensors in HBM.

---

## Dense $O(N^2)$ FlashMD Kernel (`--kernel flash`, 2026-10-26)

Raw JSON: [`docs/benchmarks/261026_engaging_flash_pallas_vs_xla.json`](file:///home/marielle/projects/prolix/docs/benchmarks/261026_engaging_flash_pallas_vs_xla.json) (RTX PRO 6000), [`docs/benchmarks/261026_engaging_l40s_flash.json`](file:///home/marielle/projects/prolix/docs/benchmarks/261026_engaging_l40s_flash.json) (L40S), [`docs/benchmarks/261026_engaging_h200_flash.json`](file:///home/marielle/projects/prolix/docs/benchmarks/261026_engaging_h200_flash.json) (H200).
Compares `chunked_explicit_nonbonded_energy` (`lax.scan` over $T=256$ tiles with `jax.checkpoint`) against `pallas_flash_explicit_nonbonded` ($(32, 32)$ tile loop in Triton SRAM).

| $N$ (atoms) | XLA Flash ms (RTX PRO 6000) | Pallas Flash ms (RTX PRO 6000) | speedup | XLA Working MB | Pallas Working MB | rel dE | max abs dF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.096 | 0.530 | 0.18x | 67.1 | 67.1 | 3.0e-7 | 1.5e-5 |
| 2,048 | 0.351 | 1.809 | 0.19x | 67.1 | 67.1 | 3.1e-7 | 2.3e-5 |
| 8,192 | 2.917 | 9.634 | 0.30x | 67.1 | 67.1 | 4.4e-7 | 4.2e-5 |
| 16,384 | 9.914 | 41.32 | 0.24x | 68.1 | 67.1 | 7.5e-7 | 5.3e-5 |
| 32,768 | 19.22 | 117.2 | 0.16x | 136.6 | **67.1** | 1.4e-6 | 7.2e-5 |

**Conclusion**: `pallas_flash_explicit_nonbonded` achieves float32 round-off parity (`rel_dE <= 1.4e-6`) and $2.04\times$ lower peak memory at 32k (`67.1 MB` vs `136.6 MB`), but XLA's `lax.scan` over $(N, 256)$ tiles is $3.3\times\text{–}6.0\times$ faster because XLA fuses the $(N, 256)$ tile broadcast across all SMs at high occupancy whereas the $(32, 32)$ Pallas kernel executes $N/32$ serial inner loop steps per program (`1,024` sequential steps at 32k). More importantly, at $N=32,768$, the $O(N \cdot K)$ `pallas_nonbonded_nl` kernel (`2.16 ms`) is **$8.9\times$ faster than Flash XLA (`19.22 ms`) and $54\times$ faster than Flash Pallas (`117.2 ms`)**. Use `pallas_nonbonded_nl` for all explicit/periodic systems $\ge 2\text{k}$ atoms.

---

## Neighbor-List Skin (`dr_threshold`) & `capacity_multiplier` Sweep (`--kernel nl_skin`, 2026-10-26)

Raw JSON: [`docs/benchmarks/261026_engaging_nl_skin.json`](file:///home/marielle/projects/prolix/docs/benchmarks/261026_engaging_nl_skin.json).
Sweeps `dr_threshold` $\in \{0.5, 1.0, 1.5, 2.0\}$ Å and `capacity_multiplier` $\in \{1.15, 1.25, 1.50\}$ across $N \in \{2048, 8192, 32768\}$ on RTX PRO 6000 Blackwell. Assuming a thermal displacement rate of `0.025 Å/step`, `est_steps_between_rebuilds = max(1, (dr_threshold / 2) / 0.025)`.

| $N$ | `dr_threshold` (Å) | `cap_mult` | Padded $K$ | Est. steps / rebuild | Rebuild ms | Skin check ms | Pallas step ms | **Net amortized step ms** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.5 | **1.15** | **376** | 10.0 | 0.996 | 0.072 | **0.606** | **0.770** |
| 8,192 | 0.5 | 1.25 | 408 | 10.0 | 1.097 | 0.070 | 0.625 | 0.798 |
| 8,192 | 1.0 | **1.15** | **424** | 20.0 | 1.095 | 0.071 | **0.654** | **0.776** |
| 8,192 | 1.5 | 1.15 | 480 | 30.0 | 1.069 | 0.071 | 0.695 | 0.799 |
| 8,192 | 2.0 | 1.25 | 600 | 40.0 | 1.116 | 0.070 | 0.808 | 0.904 |
| 32,768 | 0.5 | **1.15** | **416** | 10.0 | 4.212 | 0.077 | **2.126** | 2.616 |
| 32,768 | 0.5 | 1.25 | 448 | 10.0 | 4.273 | 0.074 | 2.257 | 2.750 |
| 32,768 | 1.0 | **1.15** | **432** | 20.0 | 4.199 | 0.078 | **2.090** | **2.374** |
| 32,768 | 1.0 | 1.25 | 464 | 20.0 | 4.236 | 0.076 | 2.248 | 2.532 |
| 32,768 | 1.5 | 1.25 | 576 | 30.0 | 4.300 | 0.076 | 2.640 | 2.857 |
| 32,768 | 2.0 | 1.25 | 656 | 40.0 | 4.383 | 0.076 | 2.912 | 3.096 |

**Takeaway**:
1. **`capacity_multiplier = 1.15` strictly dominates `1.25` and `1.50`** for homogeneous liquid boxes, trimming padded $K$ by 8–24% and reducing `pallas_nonbonded_nl` step time at every size without overflow.
2. **`dr_threshold = 0.5–1.0` Å is the sweet spot**: at $N=32,768$, increasing `dr_threshold` from `0.5` to `1.0` Å with `capacity_multiplier = 1.15` halves the rebuild frequency ($10 \to 20$ steps) while barely increasing $K$ ($416 \to 432$), cutting net amortized step time from `2.62 ms` to **`2.37 ms`**. Pushing `dr_threshold` to `1.5–2.0` Å inflates $(r_c + r_{\text{skin}})^3$, growing $K$ to `576–656` and slowing every nonbonded step by `25–39%`.

---

## Cross-GPU Generation Scaling (`sm_120` Blackwell vs `sm_89` Ada L40S vs `sm_90` Hopper H200)

Raw JSON:
- Blackwell (`sm_120`, NVIDIA RTX PRO 6000 Blackwell Server Edition, 98 GB GDDR7): [`docs/benchmarks/261026_engaging_rtxpro6000_nonbonded.json`](file:///home/marielle/projects/prolix/docs/benchmarks/261026_engaging_rtxpro6000_nonbonded.json)
- Ada Lovelace (`sm_89`, NVIDIA L40S, 46 GB GDDR6): [`docs/benchmarks/261026_engaging_l40s_nonbonded.json`](file:///home/marielle/projects/prolix/docs/benchmarks/261026_engaging_l40s_nonbonded.json)
- Hopper (`sm_90`, NVIDIA H200, 144 GB HBM3e): [`docs/benchmarks/261026_engaging_h200_nonbonded.json`](file:///home/marielle/projects/prolix/docs/benchmarks/261026_engaging_h200_nonbonded.json)

| $N$ (atoms) | $K$ | RTX PRO 6000 (`sm_120`) XLA / Pallas ms (**speedup**) | L40S (`sm_89`) XLA / Pallas ms (**speedup**) | H200 (`sm_90`) XLA / Pallas ms (**speedup**) | Isolated Working MB (XLA / Pallas) |
|---:|---:|---:|---:|---:|---:|
| 512 | 392 | 0.258 / 0.453 (0.57x) | 0.257 / 0.459 (0.56x) | 0.251 / 0.592 (0.42x) | 67.1 / 67.1 |
| 2,048 | 424 | 0.531 / 0.483 (**1.10x**) | 0.602 / 0.498 (**1.21x**) | 0.493 / 0.606 (0.81x) | 67.1 / 67.1 |
| 8,192 | 408 | 2.138 / 0.630 (**3.39x**) | 2.187 / 0.845 (**2.59x**) | 1.783 / 1.281 (**1.39x**) | 67.1 / 67.1 |
| 32,768 | 448 | 9.046 / 2.164 (**4.18x**) | 9.801 / 3.940 (**2.49x**) | 6.758 / 5.673 (**1.19x**) | 125.8 / 125.8 |
| 131,072 | 448 | 36.32 / 8.591 (**4.23x**) | 37.39 / 15.56 (**2.40x**) | 25.75 / 23.94 (**1.08x**) | 554.5 / **536.9** |

**Key insights across GPU architectures**:
1. **`pallas_nonbonded_nl` outperforms XLA across all three GPU generations for $N \ge 8,192$**: up to **`4.23x`** on RTX PRO 6000 Blackwell (`sm_120`), **`2.59x`** on L40S (`sm_89`), and **`1.39x`** on H200 (`sm_90`).
2. **Why speedup is largest on GDDR6/GDDR7 architectures (`sm_89` L40S, `sm_120` RTX PRO 6000) vs HBM3e (`sm_90` H200)**: XLA's chunked neighbor-list path is memory-bandwidth bound because it materializes intermediate $(N_{\text{chunk}}, K)$ gather/mask/force tensors in global memory. H200's `4.8 TB/s` HBM3e bandwidth accelerates XLA's memory-bound gathers (`6.76 ms` at 32k vs `9.80 ms` on L40S), whereas `pallas_nonbonded_nl` keeps pair distances, exclusion binary searches, and force accumulation in SRAM registers, achieving its highest absolute throughput (`2.16 ms` at 32k, `8.59 ms` at 131k) on Blackwell `sm_120`'s high-clocked SMs.
3. **Device memory footprint is architecture-invariant**: across `sm_89`, `sm_90`, and `sm_120`, isolated working memory at $N=131,072$ is `536.9 MB` (`263.1 MB` activation) for Pallas vs `554.5–555.0 MB` (`280.7–281.2 MB` activation) for XLA (`working_mem_exponent = 0.345`).

---

## Prolix vs kUPS 3-Tier High-Precision & Propagator Benchmark (`scripts/benchmarks/kups_benchmark.py`, 2026-10-26)

Harness: [`scripts/benchmarks/kups_benchmark.py`](file:///home/marielle/projects/prolix/scripts/benchmarks/kups_benchmark.py).
Raw JSON: [`docs/benchmarks/261026_engaging_kups_benchmark.json`](file:///home/marielle/projects/prolix/docs/benchmarks/261026_engaging_kups_benchmark.json) (RTX PRO 6000), [`docs/benchmarks/261026_engaging_l40s_kups.json`](file:///home/marielle/projects/prolix/docs/benchmarks/261026_engaging_l40s_kups.json) (L40S), [`docs/benchmarks/261026_engaging_h200_kups.json`](file:///home/marielle/projects/prolix/docs/benchmarks/261026_engaging_h200_kups.json) (H200).
Unit-conversion contract implemented and tested in [`src/prolix/physics/kups_adapter.py`](file:///home/marielle/projects/prolix/src/prolix/physics/kups_adapter.py) and [`tests/bench/test_kups_benchmark.py`](file:///home/marielle/projects/prolix/tests/bench/test_kups_benchmark.py).

### Tier 1: Single-Step & Potential Parity (`float64` Machine Precision + `float32` Production)

- **Tier 1A (Harmonic oscillator + single-step BAOAB & CSVR at $\Delta t \in \{0.5, 1.0, 2.0\}$ fs, $N=256$, `float64`)**:
  - Potential energy relative difference (`AKMA` vs `kUPS eV` converted via `energy_ev_to_kcal_mol`): **`0.0` to `1.28e-16`**.
  - Force max absolute difference (`force_ev_per_angstrom_to_kcal_mol`): **`2.78e-17` kcal/(mol·Å)**.
  - Single-step BAOAB & CSVR propagator state parity (identical standard normal draws & PRNG keys): position max abs diff **`1.11e-16` Å**, momentum max abs diff **`5.33e-15`**.
- **Tier 1B (Periodic nonbonded LJ + reaction-field/PME-direct fluid, $N \in \{512, 2048\}$)**:
  - `float64` Prolix XLA vs kUPS (eV, Å): `rel_dE_f64 = 0.0` ($N=512$) and `1.96e-16` ($N=2048$); `max_abs_dF_f64 = 3.55e-15` to `2.84e-14` kcal/(mol·Å).
  - `float32` Prolix Pallas (`pallas_nonbonded_nl`) vs kUPS (eV, Å): `rel_dE_f32 = 1.05e-7` to `4.46e-7`; `max_abs_dF_f32 = 4.39e-5` to `9.16e-5` kcal/(mol·Å).

### Tier 2: Thermostat Ensemble & Canonical Fluctuation Parity ($N=256$, $T=300$ K, `10,000` `lax.scan` Steps)

| Integrator | $\Delta t$ (fs) | Prolix $\langle T \rangle$ (K) | kUPS $\langle T \rangle$ (K) | $|\Delta \langle T \rangle|$ (K) | Max stepwise $|\Delta T|$ (K) | $\sigma_K / \sigma_{K,\text{canonical}}$ |
|---|---:|---:|---:|---:|---:|---:|
| BAOAB | 0.5 | 299.64 | 299.64 | **5.68e-14** | 5.68e-13 | 0.996 |
| BAOAB | 1.0 | 300.64 | 300.64 | **5.68e-14** | 6.25e-13 | 0.933 |
| BAOAB | 2.0 | 300.51 | 300.51 | **1.14e-13** | 5.68e-13 | 0.961 |
| CSVR | 0.5 | 298.89 | 298.89 | **5.68e-14** | 5.12e-13 | 0.932 |
| CSVR | 1.0 | 300.09 | 300.09 | **0.00e+00** | 5.12e-13 | 0.946 |

### Tier 3: 20-Step `lax.scan` BAOAB Propagator Throughput ($\Delta t = 2.0$ fs)

Compares four compiled 20-step `lax.scan` propagators on identical periodic nonbonded systems:
1. `naive_split_xla`: calls `energy_fn(r)` and `grad(energy_fn)(r)` separately on every step.
2. `kups_cached_xla`: kUPS eV/Å propagator carrying cached `(U_eV, F_eV_A)` across steps via `jax.value_and_grad` (1 force evaluation per step).
3. `prolix_bundle_xla`: Prolix AKMA propagator carrying `EnergyForceBundle` (`@jax.tree_util.register_dataclass`) across steps via `make_energy_force_bundle_fn` on XLA.
4. `prolix_bundle_pallas`: Prolix AKMA propagator carrying `EnergyForceBundle` backed by `pallas_nonbonded_nl` (forward pass emits both `E` and `F` in one Pallas kernel launch).

| GPU Architecture | $N$ (atoms) | `kups_cached_xla` ns/day (ms/step) | `prolix_bundle_xla` ns/day (ms/step) | `prolix_bundle_pallas` ns/day (ms/step) | Prolix XLA / kUPS XLA ratio | **Prolix Pallas / kUPS XLA speedup** | 20-step Trajectory `rel_dE` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 (`sm_120`)** | 512 | 762.5 (0.227) | 753.0 (0.229) | 405.8 (0.426) | 0.987x | 0.53x | 8.7e-7 |
| **RTX PRO 6000 (`sm_120`)** | 2,048 | 341.0 (0.507) | 341.6 (0.506) | **411.1 (0.420)** | 1.002x | **1.21x** | 9.6e-7 |
| **RTX PRO 6000 (`sm_120`)** | 8,192 | 90.4 (1.911) | 90.3 (1.914) | **306.7 (0.564)** | 0.998x | **3.39x** | 9.7e-7 |
| **RTX PRO 6000 (`sm_120`)** | 32,768 | 20.2 (8.538) | 20.3 (8.511) | **82.5 (2.094)** | 1.003x | **4.08x** | 9.7e-7 |
| **NVIDIA L40S (`sm_89`)** | 512 | 725.5 (0.238) | 724.7 (0.238) | 389.6 (0.444) | 0.999x | 0.54x | 8.7e-7 |
| **NVIDIA L40S (`sm_89`)** | 2,048 | 286.4 (0.603) | 287.4 (0.601) | **346.4 (0.499)** | 1.003x | **1.21x** | 7.6e-7 |
| **NVIDIA L40S (`sm_89`)** | 8,192 | 77.1 (2.240) | 77.5 (2.229) | **202.9 (0.852)** | 1.005x | **2.63x** | 8.9e-7 |
| **NVIDIA L40S (`sm_89`)** | 32,768 | 17.6 (9.813) | 17.6 (9.819) | **41.4 (4.174)** | 0.999x | **2.35x** | 9.8e-7 |
| **NVIDIA H200 (`sm_90`)** | 512 | 860.8 (0.201) | 860.1 (0.201) | 318.2 (0.543) | 0.999x | 0.37x | 8.7e-7 |
| **NVIDIA H200 (`sm_90`)** | 2,048 | 395.0 (0.438) | 394.5 (0.438) | 287.2 (0.602) | 0.999x | 0.73x | 8.6e-7 |
| **NVIDIA H200 (`sm_90`)** | 8,192 | 98.1 (1.761) | 98.0 (1.763) | **127.5 (1.355)** | 0.999x | **1.30x** | 9.8e-7 |
| **NVIDIA H200 (`sm_90`)** | 32,768 | 24.7 (7.007) | 24.7 (6.985) | **28.0 (6.171)** | 1.003x | **1.14x** | 9.8e-7 |

