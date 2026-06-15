---
task_id: 260615_autonomous-loop
backlog_item: 257
date: 2026-06-15
---

# RR1 Rebuttal Evidence: Differentiation from Prior Work

**Purpose:** §Related Work + RR1 rebuttal evidence for the anticipated reviewer objection: "How is this different from just using batched inputs in jax-md or OpenMM?"

**Prior-art base:** See `.praxia/docs/research/260615_claim1-hetero-batch-precedent.md` for the full per-tool analysis table. This document focuses on the three specific comparators in the §7.1 figure (kUPS/jax-md, OpenMM, GROMACS multi-sim) and distills the rebuttal-ready evidence.

---

## kUPS / jax-md

kUPS is a JAX-native MD engine (CuspAI, open-sourced April 2026) built around composable, differentiable primitives (`jit`, `vmap`, `grad`). Its documentation explicitly conditions performance on fixed shapes: "aggressive XLA compilation settings can be enabled when certain tensors have fixed shapes — e.g., the number of atoms will be constant." PR #95 ("Fix: per-system optimizer states for batched relaxation of independent structures") reveals the practical consequence of this assumption — batched relaxation of independent systems required an explicit fix because each system's LBFGS optimizer state needed its own fixed-shape storage, confirming a system homogeneity assumption in the compiled path.

jax-md (Schoenholz & Cubuk, NeurIPS 2020), which prolix uses as a substrate for neighbor-list and space primitives, has the same structural limitation. GitHub Issue #101 shows that `vmap(neighbor_fn)` raises `ConcretizationTypeError` when atom counts differ across systems in the batch. Issue #192 shows that neighbor list reallocation in a `vmap` batch requires knowing the largest capacity upfront, which forces homogeneous sizing. All published jax-md batch examples use homogeneous N.

The key limitation shared by kUPS and jax-md is that `vmap` in JAX requires static shapes at trace time. Running two systems with different atom counts under one `jit`-compiled function requires one of: (a) padding both to the same size and accepting wasted compute, (b) separate compilations per size, or (c) a bucketed JIT key that maps coarse shape classes to compiled functions. Neither kUPS nor jax-md implements (c); prolix does.

**What kUPS measures (§7.1 context):** The §7.1 figure uses kUPS as an external throughput comparator for bonded parameter fitting. kUPS reports throughput over RASPA on packing calculations, but this is for homogeneous-topology ensembles. The comparison in §7.1 is throughput on a mixed-topology bonded ensemble, where kUPS must either serialize (one system at a time) or pad to the largest system (wasting compute). Neither mode is documented as kUPS's intended use case.

---

## OpenMM

OpenMM is a CUDA-native MD engine with no autodiff support. It can parallelize a single simulation across multiple GPUs (via `DeviceIndex` comma-list) or run one simulation per GPU. NVIDIA MPS (Multi-Process Service), highlighted by the OMSF blog (2024), is an OS-level mechanism that allows multiple CUDA processes to share one GPU — it reduces context-switching overhead but does not fuse independent simulations into a single compiled kernel. Each distinct topology still triggers its own CUDA kernel compilation, and each simulation context remains independent.

MultiOpenMM (third-party) extends OpenMM by CPU-side concatenation of multiple independent systems into one OpenMM `System` object, including topologically distinct systems. However, MultiOpenMM's stacking approach uses sequential CPU-side assembly + serial or MPS dispatch, not a single compiled JIT graph over a heterogeneous batch. The `openmm-torch` issue #13 ("Multiple simulations on the same GPU are interfering with each other") confirms that concurrent multi-simulation on OpenMM is managed at the OS/CUDA-context level, not at the level of a fused compiled kernel.

OpenMM has no autodiff capability. A reviewer who cites OpenMM as prior art for "batched multi-system execution" is correct that OpenMM can run multiple systems on one GPU (via MPS), but is incorrect that it achieves a single compiled forward or backward pass over heterogeneous topologies. The distinction that matters for §7.1 is that prolix's gradient-based parameter fitting requires a differentiable backward pass through the entire heterogeneous batch — a capability OpenMM cannot provide at any level.

---

## GROMACS multi-sim

GROMACS `-multidir` (mdrun multi-simulation) launches N independent simulations under a single `mpirun` invocation, with one MPI rank group and one subdirectory (`.tpr` + topology) per simulation. MPI ranks from different simulations can share a GPU via OS-level MPS, reducing context-switching, but each simulation processes its own force kernels independently. The GROMACS 2026.1 documentation states: "The MPI ranks doing PP work on a node are mapped to the GPUs even though they come from more than one simulation" — this describes MPS-level sharing, not kernel fusion.

GROMACS multi-sim is architecturally designed for same-topology replicas (replica exchange is its primary use case). Running different-topology systems requires independent MPI processes with no shared compiled kernel path and no communication between them. There is no concept of a bucketed JIT key or a shared XLA/CUDA compiled function that accepts multiple topology layouts.

GROMACS is not autodiff-native. The multi-sim mechanism is pure execution-level parallelism across independent C++ MD trajectories. It has no backward pass and no parameter gradient support. For the §7.1 use case (gradient-based bonded parameter fitting over a mixed-topology ensemble), GROMACS multi-sim provides no viable path even as a baseline.

---

## Prolix claim + codebase evidence

The prolix differentiator is **compile-level sharing across heterogeneous topologies**, not just execution-level parallelism. The architectural mechanism is `MolecularShapeSpec` — a frozen dataclass carrying bucket indices (not real atom counts) as plain Python ints, serving as the XLA JIT cache key.

**Evidence 1 — `src/prolix/types/bundles.py:66-113`:**

`MolecularShapeSpec` (line 66) is the sole `eqx.field(static=True)` in `MolecularBundle`. The design invariant is documented explicitly at lines 107–113:

> "All topology arrays are DYNAMIC (not static=True) to avoid XLA recompilation per distinct topology. Instead, XLA caches on bucketed size. shape_spec is the ONLY eqx.field(static=True) — it carries bucket indices (coarse) as plain Python ints, serving as the JIT cache key. Two systems with different real counts but same bucket size produce identical shape_spec (enabling Claim 1: heterogeneous batch substrate)."

This means two systems with, e.g., 73 and 119 real atoms both hash to `ATOM_BUCKET[1] = 128`, and share one compiled XLA function. Real atom counts are tracked via dynamic `atom_mask` arrays; padding waste is bounded by bucket granularity.

**Evidence 2 — `src/prolix/tiling/planner.py:55-63` and `128-140`:**

`AxisSpec.heterogeneous: bool` (line 63) is the flag that marks an axis as containing elements with varying shapes. `BatchPlanner.plan()` Phase 1 (lines 128–140) unconditionally pre-demotes all heterogeneous axes to `safe_map` with the reasoning: `"heterogeneous axis: element shapes vary; vmap invalid; safe_map required"`. This is the planner-level enforcement of the design invariant: heterogeneous axes cannot use `vmap` (which requires static shapes) and are always tiled via `safe_map`.

**Evidence 3 — `src/prolix/types/bundles.py:20-33`:**

`ATOM_BUCKETS = (64, 128, 256, 1_024, 5_000, 25_000, 60_000)`. The comment at line 20–22: "Bucketed array size thresholds: systems are padded to the smallest bucket that fits their actual count. This enables XLA to cache on bucket size, not distinct topology." The bucket ladder was refined in HP4 (2026-05-20) to allow the Lane B ensemble to span 4 distinct buckets for cross-bucket heterogeneity evidence in the §7.1 figure.

**The combination:** kUPS and jax-md provide `jax.grad`-compatible differentiability but cannot batch heterogeneous systems under one compiled function. TorchSim (2025, closest prior art) achieves device-level heterogeneous batching via sequential bin-packing but without a bucketed JIT cache key and without differentiable trajectories through the batch. Prolix's `MolecularShapeSpec + BatchPlanner` architecture is the only design that enables both simultaneously.

---

## Gaps / open questions

1. **TorchSim torch.compile behavior across bins** — The TorchSim paper (arXiv 2508.06628) does not confirm whether `torch.compile` with `dynamic=True` is used internally for `BinningAutobatcher`. If it is, TorchSim may be closer to prolix's bucketed approach than the paper implies. Reading the torch-sim source (github.com/Radical-AI/torch-sim) before paper submission is recommended to sharpen this distinction.

2. **kUPS v0.2+ heterogeneous support** — PR #95 (April 2026) suggests kUPS is actively developing per-system batching. Monitor kUPS releases. If a future kUPS release adds bucketed heterogeneous MD, prolix's differentiability advantage (`jax.grad` through the hetero-batch) becomes the sole differentiator.

3. **MultiOpenMM single-GPU dispatch** — Whether MultiOpenMM achieves true concurrent execution on one GPU vs. sequential dispatch is not confirmed. The OMSF MPS blog implies OS-level time-multiplexing. §Related Work should hedge: "MultiOpenMM supports heterogeneous topology via CPU-side concatenation with MPS-based concurrent scheduling; true single-kernel forward/backward pass is not claimed."

4. **Compile-time benchmark** — No benchmark numbers exist comparing prolix's O(B_max) bucket compilations vs. per-topology recompilation in jax-md/kUPS for a mixed-topology dataset of N=512 systems. This would make the efficiency claim quantitative and harder to dispute.

5. **jax-md issue recency** — Issues #101 and #192 date to 2021. A reviewer may argue that jax-md has since added heterogeneous support. Verify against the current jax-md GitHub before submission (no evidence of fix found in this scan).

---

## Sources

1. kUPS documentation — Packaged Simulations. https://cusp-ai-oss.github.io/kUPS/simulations/ (2026-04)
2. kUPS PR #95 — Fix: per-system optimizer states for batched relaxation. https://github.com/cusp-ai-oss/kUPS/pull/95 (2026)
3. kUPS GitHub README. https://github.com/cusp-ai-oss/kUPS (2026)
4. CuspAI blog — "kUPS: a molecular simulation engine for the AI era." Medium, April 2026.
5. jax-md Issue #101 (vmap ConcretizationTypeError). https://github.com/jax-md/jax-md/issues/101 (2021)
6. jax-md Issue #192 (batch neighbor list). https://github.com/jax-md/jax-md/issues/192
7. OpenMM User Guide (8.x) — Platform-Specific Properties. https://docs.openmm.org/latest/userguide/library/04_platform_specifics.html
8. OMSF blog — "Maximizing OpenMM Throughput with NVIDIA MPS." 2024.
9. openmm-torch Issue #13 — Multiple simulations interfering. https://github.com/openmm/openmm-torch/issues/13
10. GROMACS 2026.1 — Useful mdrun features. https://manual.gromacs.org/current/user-guide/mdrun-features.html
11. Cohen, O. et al. "TorchSim: An efficient atomistic simulation engine in PyTorch." arXiv 2508.06628, 2025.
12. prolix codebase — MolecularBundle, MolecularShapeSpec, ATOM_BUCKETS. src/prolix/types/bundles.py (2026-06-15)
13. prolix codebase — BatchPlanner, AxisSpec. src/prolix/tiling/planner.py (2026-06-15)
14. prolix prior-art scan (base document). .praxia/docs/research/260615_claim1-hetero-batch-precedent.md (2026-06-15)
