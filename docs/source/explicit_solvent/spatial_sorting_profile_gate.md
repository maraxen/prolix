# Phase 6 — Spatial sorting: profile gate (decision record)

**Status:** Profiling-first; no default Morton reorder pass is enabled in `src/prolix` until a bottleneck is measured on representative hardware.

## Objective

Avoid speculative optimization (scatter order vs cell-list occupancy vs neighbor-list rebuild) by requiring a **short profiling artifact** before merging spatial-sorting or cell-tuning code.

## What to profile (candidates)

| Candidate | Typical symptom | Entry points |
|-----------|-----------------|--------------|
| PME charge spreading / gather-scatter | FFT or spline spread dominates step time | `physics/pme.py`, hot paths under `make_energy_fn` |
| Neighbor list rebuild | Large fraction of step in `neighbor_list` update | `physics/neighbor_list.py`, JAX-MD neighbor API |
| Cell list / tiled kernels | Used when exercising `cell_list` / `cell_nonbonded` | `physics/cell_list.py` |

## Minimal workflow

1. Fix system size **N** and box (e.g. solvated protein bucket from production).
2. Use **JAX profiler** or `scripts/profile_step.py` (if applicable) to capture one or more MD steps after JIT warmup.
3. Record: device (CPU/GPU), JAX version, **where time went** (top ops or custom ranges).
4. **Decision:** implement at most **one** of: Morton/Z-order sort for a specific hot scatter, cell occupancy tuning, or neighbor-list policy — tied to the measured bottleneck.

## Out of scope until profiling says otherwise

- Blind Morton ordering of all coordinates every step.
- Parallel changes to PME grid policy and neighbor policy in the same PR without measurements.

## Related

- [current_implementation](current_implementation.md) — as-built modules.
- [gpu_optimization_strategies](gpu_optimization_strategies.md) — broader GPU notes.
