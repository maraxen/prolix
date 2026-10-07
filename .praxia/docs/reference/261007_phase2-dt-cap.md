---
title: "Why the dt cap was 0.5 fs (and how it was lifted to 1.0 fs)"
description: "SETTLE-thermostat kinetic-energy feedback explanation and the production-scale dt lift, moved verbatim from CLAUDE.md."
status: reference
date: '261007'
---

### Why the dt cap was 0.5 fs (and how it was lifted to 1.0 fs)

SETTLE velocity constraints remove kinetic energy from constrained degrees of freedom. Standard thermostats expect to regulate total kinetic energy, creating a feedback loop:

1. SETTLE constraints remove KE from rigid-body DOF
2. Thermostat tries to add KE back
3. SETTLE removes it again next step
4. Oscillation/divergence emerges

At smaller timesteps (dt ≤ 0.5 fs), the per-step constraint impulse magnitude is small, giving the Langevin thermostat (friction + noise) time to re-equilibrate before the next SETTLE constraint applies.

**Resolution (2026-06-13):** the C3 AM-conservation correction (678c9cb) plus adequate
friction (gamma ≈ 10 ps⁻¹) lifted this to **dt ≤ 1.0 fs at production scale**. Gate job
15870804 (895 waters) holds T_rot = 299.6 K, and size sweep ba334c1f shows the residual
warm bias is **translational finite-size** — concentrated in the 3·N−3 translational DOF,
so it only bites for very small systems (n ≲ 16) and washes out by n ≳ 16 (T_total within
±15 K) / n ≳ 64 (within ±5 K). T_rot is faithful at every size. Use dt ≤ 0.5 fs only for
n ≲ 16 or weak friction (gamma ≈ 1 ps⁻¹). See
`.praxia/docs/research/260612_p5-dt1fs-size-crossover.md`.
