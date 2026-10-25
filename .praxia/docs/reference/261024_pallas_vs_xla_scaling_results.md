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

## Caveats

- `peak_mem_*` fields in the JSON are the device high-water mark, which is cumulative
  across both paths in one process; they do **not** isolate per-path memory. Use separate
  processes (or `jax.profiler` device-memory profiles) for a per-path memory claim.
- The titanix run fell back to CPU (no CUDA jaxlib, no Triton), so Pallas ran in interpret
  mode there; those numbers are not comparable and are not archived.
- Single GPU model/node; repeat on a second GPU generation before quoting speedups externally.
- The first harness version used uniform-random positions, which produced overlapping
  atoms and ~1e13 energies where float32 parity is meaningless. Always use a
  physically-spaced configuration for parity benchmarks.
