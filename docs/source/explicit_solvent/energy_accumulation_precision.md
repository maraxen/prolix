# Decision: Energy Accumulation Precision Strategy

**Date:** 2026-03-18
**Status:** Accepted
**Context:** noised_cb / prolix Langevin dynamics engine

---

## Background

Our JAX-based MD engine sums N×K pairwise energy terms in `float32` using
`jnp.sum()`. For a typical system (N=4096, K=2000), this is ~8M terms per
replica. Under massive vmap parallelism (1024 replicas), each GPU step
reduces ~1 billion elements.

The question: does naive `float32` summation lose enough precision to
corrupt production Langevin dynamics trajectories?

## NVE vs Langevin: Why This Matters

| Property | NVE (Microcanonical) | Langevin (NVT) |
|---|---|---|
| **Energy** | Strictly conserved | Fluctuates (thermostat) |
| **Temperature** | Emergent, fluctuates | Controlled at target |
| **Dynamics** | True Newtonian | Damped + stochastic kicks |
| **Precision sensitivity** | **Very high** — rounding acts as spurious forces | **Low** — thermal noise >> rounding |
| **Use case** | Transport properties, diffusion | **Conformational sampling, free energy** |

**NVE** integrates Newton's equations without any thermostat. Every bit of
rounding error in the energy sum acts as a spurious force that accumulates
over time, causing destructive energy drift. This is where `float64`
accumulators are essential.

**Langevin dynamics** (our integrator) explicitly injects random thermal
noise and applies viscous friction at every step. The thermostat continuously
adds and removes kinetic energy to maintain the target temperature. This
stochastic noise is **orders of magnitude larger** than `float32` rounding
error (~5.7e-7 relative), effectively masking numerical drift.

**Our use case is Langevin NVT for conformational sampling** — we only care
that the ensemble accurately reflects the Boltzmann distribution at the
target temperature, not the exact trajectory. Kinetic dampening from
Langevin is an acceptable tradeoff for sampling robustness.

> **Sources:** NotebookLM grounded from "Molecular Dynamics: Theory, Methods
> & Practice" notebook (165 sources, queried 2026-03-18).

## What Production Engines Do

| Engine | GPU Precision Model | Energy Accumulation |
|---|---|---|
| **AMBER (SPFP)** | Default. Forces in f32. | **64-bit fixed-point integer** for PME accumulation. Perfectly associative. |
| **GROMACS** | Mixed precision. Positions/velocities in f32. | **f64 for energy accumulators and virials** only. |
| **OpenMM** | Configurable per-platform. | f64 accumulators optional (`'Precision': 'double'`). |

All three use higher-precision accumulators — but they also target NVE
validation and strict energy conservation benchmarks. For Langevin-only
workflows, the f32 hot loop is sufficient.

## Benchmark Results

Tested on node4007 (2× NVIDIA RTX PRO 6000 Blackwell, 98 GB each).
B=1024 replicas, N=1024 atoms, K=1024 neighbors = **1.07 billion elements**.

With `JAX_ENABLE_X64=True` for valid ground truth:

| Strategy | Time (ms) | Mean Rel Error | Sig Digits | Memory |
|---|---|---|---|---|
| **naive_f32** (jnp.sum) | 1.814 | 5.72e-07 | 6.2 | 10.2 GB |
| **pairwise_tree** (manual binary tree) | 1.824 | 5.72e-07 | 6.2 | 10.2 GB |
| **f64_accumulator** (cast to f64, sum, cast back) | 2.144 | 1.29e-07 | 6.9 | 10.2 GB |

### Key Observations

1. **XLA already performs tree reduction** — naive and pairwise have *identical*
   error. The manual tree sum adds zero value.
2. **f64 accumulator is 18% slower** (0.33 ms overhead per step).
3. **0.33 ms × 1M steps = 330 seconds** of added wall time per production run.
4. **All strategies scale linearly** under massive vmap parallelism — no
   degradation at 1024 concurrent replicas across 2 GPUs.

### Strategies Eliminated Early

| Strategy | Why Eliminated |
|---|---|
| **Kahan f32** (compensated sum) | 4600 ms (2500× slower) — `lax.scan` prevents XLA fusion |
| **int64 fixed-point** | 100% error — scale factor overflow at our energy magnitudes |

## Decision

**Use naive `jnp.sum()` in the simulation hot loop. Use `f64` accumulation
only for energy values written to disk (logging/checkpoint).**

### Rationale

1. **Langevin thermostat noise >> f32 rounding** — the stochastic kicks at
   each step add/remove ~kT of energy, dwarfing the ~1e-7 relative
   accumulation error.
2. **XLA already optimizes `jnp.sum` to tree reduction** — we get the best
   possible f32 accuracy for free.
3. **0.33 ms/step compounds** — over million-step production runs across
   hundreds of systems, this adds up to hours of GPU time.
4. **Logged energies should be precise** — when writing energies to
   trajectory files for analysis, the f64 cast costs nothing (single
   reduction per write interval, not per step).

### When to Revisit

- If we ever implement **NVE dynamics** (transport properties, diffusion
  coefficients) — f64 accumulators become essential.
- If we observe **systematic energy drift** in production Langevin runs
  exceeding thermostat fluctuations.
- If we move to **free energy perturbation** (FEP) where energy differences
  between states must be precise to ~0.1 kcal/mol.

## Implementation

```python
# Hot loop (every step) — f32 is fine
total_energy = jnp.sum(e_pair)  # XLA tree-reduces this

# Logging (every N steps) — use f64 for precision
logged_energy = jnp.sum(e_pair.astype(jnp.float64)).astype(jnp.float32)
```

No changes to `batched_energy.py` hot path required.
