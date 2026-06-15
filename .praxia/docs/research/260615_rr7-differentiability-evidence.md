---
name: rr7-differentiability-evidence
description: RR7 reviewer rebuttal evidence — differentiability × hetero-batch = S71 novelty; D1/D2 S1 test synthesis
metadata:
  type: project
---

# RR7 Differentiability Evidence Synthesis

**Date:** 2026-06-15
**Task:** 260615_sprint39 (#311)
**Purpose:** §Related Work rebuttal for Reviewer Comment RR7 ("differentiability is not novel")

---

## Summary

Prolix's novelty is not `jax.grad` itself, but differentiability through a **heterogeneous-batch trajectory** — a capability no prior tool provides. D1 and D2 tests provide quantitative evidence that the autodiff graph is correct through all bonded terms. The S71 figure (PENDING, blocked on #260) will be the empirical proof of the multiplicativity claim.

---

## Evidence Status Table

| Evidence | Label | Location | Precision | Status |
|----------|-------|----------|-----------|--------|
| Analytical force parity (bond/angle/dihedral/improper/UB) | D1 | `tests/physics/test_analytical_forces.py` | float64, atol=1e-10 | **PASS** |
| jax.grad vs finite-diff parity (positions + params) | D2 | `tests/fitting/test_bonded_loss.py:test_numerical_gradient_parity_*` | float32, rtol=2e-3 | **PASS** |
| Prior-tool comparison (DMFF/TorchMD/espaloma/jax-md) | Ref | `.praxia/docs/reference/` | — | confirmed |
| §7.1 figure (hetero-batch gradient fit) | S71 | PENDING — blocked on #260 (HP4) | — | **PENDING** |

**Note on canonical filenames:** Sprint 39 target filenames (`tests/physics/test_s1_force_parity.py`, `tests/api/test_s1_jaxgrad_parity.py`) do not yet exist on disk. Underlying tests are at the locations above.

---

## D1 — Analytical Force Parity

**7 test functions; float64 precision; atol=1e-10**

Covers:
- Bond forces: 3 atoms, 2 bonds
- Angle forces: 4 atoms, 2 angles
- Dihedral angle helper: 5 atoms
- Dihedral forces: 4 atoms, 2 Fourier terms
- Improper periodic: 4 atoms
- Improper harmonic: 4 atoms with out-of-plane geometry
- Urey-Bradley: 3 atoms

All verify: `analytical_forces.{bond,angle,...}_forces_analytical == -jax.grad(energy_fn)(positions)` at machine-epsilon float64 precision.

**Rebuttal use:** When reviewer cites "differentiability is not novel," D1 shows the autodiff graph is correct through every bonded term at machine-epsilon precision — this is the foundational evidence that `jax.grad` through the energy function is trustworthy.

---

## D2 — jax.grad Through Bonded Energy: Finite-Diff Cross-Validation

**Location:** `tests/fitting/test_bonded_loss.py` (commit `96b9d1c`):
- `test_numerical_gradient_parity_positions`
- `test_numerical_gradient_parity_params`

**Setup:** 3-atom water-like system (2 bonds, 1 angle); float32.

**D2a (position gradient):** `jax.grad(bonded_energy)(positions)` vs finite-diff (eps=1e-4); agreement rtol=2e-3, atol=1e-4.

**D2b (parameter gradient):** `jax.grad` w.r.t. bonded parameters (`k_bond`, `r0`); confirms the autodiff graph is differentiable through the full bonded energy functional with respect to learnable parameters.

**Rebuttal use:** D2 provides quantitative finite-diff cross-validation. When reviewer claims "differentiability is asserted but not demonstrated," D2 is the quantitative evidence.

---

## Prior-Tool Comparison

| Tool | Differentiable | Hetero-batch | Notes |
|------|---------------|--------------|-------|
| DMFF (JCTC 2023) | Y (JAX) | N | Single-system; authors recommend OpenMM export for production |
| TorchMD (JCTC 2021) | Y (autograd) | N | 60× slower than ACEMD3; O(N²) pairwise |
| espaloma (Chem Sci 2022/2024) | At parameterization time only | N | MD execution is non-differentiable OpenMM |
| jax-md | Y (JAX) | N | `vmap` requires fixed shapes; no bucketed compile |

**Source:** `.praxia/docs/reference/` (internal reference docs).

All cited tools lack the batch dimension. jax-md lacks heterogeneous topology support. No prior tool provides both differentiability + bucketed heterogeneous batch in a single JIT boundary.

---

## Multiplicativity Argument

The S71 experiment (training bonded parameters on N=512 chemically distinct molecules in a single backward pass) requires **both** capabilities simultaneously:

- **Differentiability alone** (DMFF, TorchMD): cannot batch heterogeneous systems under one JIT compile.
- **Heterogeneous batching alone** (prolix without grad): can simulate but cannot train.
- **The combination** is what enables the S71 result and constitutes the novel contribution.

**Paper-draft positioning:** "We do not claim `jax.grad` itself is novel. We claim that differentiability through a heterogeneous-batch trajectory is novel, and that the S71 result is enabled by the combination rather than either capability alone."

---

## Cross-Surface Notes

- D1 uses float64 (atol=1e-10); D2 uses float32 (rtol=2e-3, atol=1e-4) — different precision regimes are intentional. D1 is analytical identity verification; D2 is gradient-chain validation in production dtype.
- DMFF uses JAX (same language as prolix) — the throughput and batch gap is a design decision, not a language artifact. This distinction is load-bearing in the rebuttal.
- S71 figure is PENDING (HP4 sub-spec #260) while D1/D2 are PASS — the evidence table has a live gap that weakens the multiplicativity argument until S71 is produced.

---

## Open Questions

1. Should `test_s1_force_parity.py` and `test_s1_jaxgrad_parity.py` be created as new files (reorganizing existing tests) or should the evidence document simply reference the actual test paths?
2. Does the D2 test (3-atom system, 1 conformer, rtol=2e-3) adequately represent "jax.grad through a 10-step trajectory" as the task brief specifies, or is a proper trajectory-length test needed?
3. Is jax-md's vmap (homogeneous-only) heterogeneous batching absence documented in an internal reference file, or does the rebuttal rely solely on this synthesis?
4. Once S71 figure is produced (#260), should this document be updated in-place or should a new evidence entry be created?
