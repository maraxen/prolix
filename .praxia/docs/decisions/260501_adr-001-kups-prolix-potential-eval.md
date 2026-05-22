# ADR 001: kUPs-style potential evaluation in Prolix

## Status

Accepted (2026-04-29). Documents Prolix potential/force evaluation invariants (`make_energy_fn`, `PaddedSystem` batching). **The Prolix runtime does not depend on the kUPS Python package**; ADR filename is historical. Prior art: kUPs-style bundling of scalar energy with gradients at the propagator boundary.

## Context

- **kUPs** bundles scalar energy and gradients in one evaluation, composes potentials linearly, and JITs the **propagator** boundary (`jit(as_result_function(...), donate_argnums=(1,))` in application code).
- **Prolix** builds energies from **proxide** `Protein` / dict via [`make_energy_fn`](../../src/prolix/physics/system.py), uses **PME custom VJP** and batched paths [`single_padded_force`](../../src/prolix/batched_energy.py) / [`single_padded_energy_nl_cvjp`](../../src/prolix/batched_energy.py) where reverse-mode through dense scatter is unsafe or slow.

## Decision

### A. Force convention (JAX-MD aligned)

- For **scalar energy** `E(R, **kwargs) -> scalar`, MD **forces** are **`F = -∇_R E`**, matching `jax_md.quantity.force` / `canonicalize_force`.
- Some batched integrator paths historically stored **`+∇E`** on `LangevinState.force` while using **`p ← p - (dt/2) ∇E`**. That is dynamically equivalent to storing **`F = -∇E`** with **`p ← p + (dt/2) F`**. New **`value_energy_and_grad`** helpers document **physical `F`**; call sites that keep the minus-form must use **`grad_E = -forces`**.

### B. Shapes

| Mode | `energy` | `forces` | Notes |
|------|----------|----------|--------|
| Single system | `()` | `(N, 3)` | `make_energy_fn` |
| Batched padded `(B, N, 3)` | `(B,)` | `(B, N, 3)` | one scalar per replica |
| Logging reductions | `()` | — | mask-weighted sum over real atoms |

### C. When to use `value_and_grad` vs custom VJP / analytical force

| Path | Mechanism | Rationale |
|------|-----------|-----------|
| `make_energy_fn` + PME | `jax.value_and_grad` **or** `quantity.force` | PME registers **custom VJP**; `value_and_grad` uses it. |
| `single_padded_force` (implicit) | **Analytical / structured** | Avoids AD through padded exclusions / scatter. **Do not** wrap in `value_and_grad` for production force unless fused energy exists. |
| `single_padded_energy_nl_cvjp` | **Custom VJP** + optional `checkpoint` | Preferred batched explicit-solvent AD path. |
| `fused_energy_and_forces_nl` | **Fused kernel** | Fastest when applicable. |

### D. Compatibility matrix (MVP)

| Ensemble / integrator | Energy+force bundle | Notes |
|------------------------|---------------------|--------|
| SETTLE Langevin (`settle_langevin`) | `value_energy_and_forces` at constrained `R` | Same AD site as [`settle_langevin_potential_propagator`](../../src/prolix/physics/settle_langevin_potential_propagator.py). |
| Implicit batched (`make_langevin_step`) | Force-only (`single_padded_force`) | No scalar total without extra reduction. |
| NL batched (`make_langevin_step_nl`) | `value_energy_and_grad_energy` → store `grad_E` + optional `energy` | Matches prior `jax.grad(energy)` semantics for the O step. |

## Consequences

- New helpers live under [`md_potential_bundle.py`](../../src/prolix/physics/md_potential_bundle.py); SETTLE step under [`settle_langevin_potential_propagator.py`](../../src/prolix/physics/settle_langevin_potential_propagator.py).
- **`donate_argnums`** on `jit` + `lax.scan` is **opt-in** only after profiling; donation failed on heterogeneous SETTLE harness (see sprint Step 2).
- Full **kUPs `Table` / `Lens` / `Patch`** stack is **out of scope** until neighbor-incremental (ADR follow-up) is justified.
