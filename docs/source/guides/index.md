# User Guides

In-depth architectural and reference guides covering configuration governance, GPU megakernel design, physical unit conventions, and multi-engine validation.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} KnobSurface & ProlixOpenMMConfig Reference
:link: knob_governance
:link-type: doc

Complete reference for all 9 `KnobSurface` control dataclasses, `ProlixOpenMMConfig` execution options, OpenMM platform/integrator property mappings, and `redsox` U1/U5 static governance guarantees.
:::

:::{grid-item-card} Pallas v3.5 Megakernel Architecture
:link: pallas_megakernel
:link-type: doc

How Prolix composes Morton $32 \times 32$ Newton-pair tiles, SRAM SPME, persistent inter-step BAOAB+SETTLE+RATTLE, two-pass OBC2 GBSA, `stride=0` congeneric deduplication, and the 1D ragged CTA work queue.
:::

:::{grid-item-card} Physics, Units, & Multi-Engine Validation
:link: physics_validation
:link-type: doc

Canonical AKMA / OpenMM / kUPS unit conversion tables, potential functional forms, `openmm_cuda_guard` anti-fallback provenance, and the 4-tier OpenMM CUDA & kUPS audit summary.
:::

:::{grid-item-card} Explicit-Solvent OpenMM Comparison Protocol
:link: ../explicit_solvent/openmm_comparison_protocol
:link-type: doc

Canonical `REGRESSION_EXPLICIT_PME` anchor parameters, force-field pairing rules, constraint handling, and benchmark JSON schema conventions.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

knob_governance
pallas_megakernel
physics_validation
../explicit_solvent/openmm_comparison_protocol
```
