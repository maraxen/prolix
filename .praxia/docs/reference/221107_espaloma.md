---
name: espaloma
description: "End-to-End Differentiable Construction of Molecular Mechanics Force Fields — Wang et al., Chemical Science, 2022; espaloma-0.3.0 Takaba et al., Chemical Science, 2024"
metadata:
  type: reference
---

## Citation
**Primary (espaloma 0.2):** Yuanqing Wang, Josh Fass, Benjamin Kaminow, et al. "End-to-end differentiable construction of molecular mechanics force fields." *Chemical Science* 13, no. 41 (2022): 12016–12033. DOI: 10.1039/D2SC02739A. PMC: PMC9600499.

**Secondary (espaloma 0.3):** Kenichiro Takaba, Iván Pulido, Pavan Kumar Behara, et al. "Machine-learned molecular mechanics force fields from large-scale quantum chemical data." *Chemical Science* 15 (2024): 12861–12878. DOI: 10.1039/D4SC00690A.

**EspalomaCharge:** "EspalomaCharge: Machine Learning-Enabled Ultrafast Partial Charge Assignment." *JPCA* (2024). DOI: 10.1021/acs.jpca.4c01287.

## What it claims
- Replaces hand-crafted atom-type rules with a GNN that maps chemical structure directly to MM FF parameters (bond, angle, torsion force constants and equilibrium values).
- End-to-end differentiable with respect to model parameters, enabling gradient-based FF optimization against QC data.
- espaloma 0.2: optimizes valence parameters only (bonded terms); nonbonded terms fixed using OpenFF 1.2 Parsley + AM1-BCC charges from OpenEye.
- espaloma 0.3: extends to proteins, peptides, and nucleic acids; trained in a single GPU-day on >1.1 million QC energy/force calculations.
- Once parameters are assigned, MD simulation speed is identical to classical MM FF (same functional form; GNN runs only at parameterization time, not per-step).
- EspalomaCharge: 300–3000× faster than AmberTools, 15–75× faster than OpenEye for partial charge assignment.

## Benchmark methodology
- **Hardware tested:** Not explicitly stated for MD throughput
- **N tested:** Individual molecule parameterization (single molecules, peptides up to ~hundreds of residues). No batched N-molecule throughput benchmark (e.g., N=512 simultaneous molecules) is reported
- **Metric reported:** (1) Accuracy vs. QC reference energies/geometries; (2) relative free energy predictions vs. experiment; (3) parameterization wall-clock time vs. AmberTools/OpenEye
- **Baseline compared against:** OpenFF Parsley, GAFF2, SMIRNOFF for accuracy; AmberTools + OpenEye for parameterization speed
- **No per-step MD throughput benchmark is reported** — the speed claim is qualitative ("same as classical MM")

## Force field scope
- **Bonded terms:** Yes — bonds (harmonic), angles (harmonic), proper torsions (Fourier series), improper torsions
- **Nonbonded terms:** In 0.2 — no; nonbonded uses legacy OpenFF/AM1-BCC. In 0.3 — partial; LJ from a base FF, charges via EspalomaCharge GNN
- **Differentiable?** Yes — the GNN parameter-assignment step is differentiable; the resulting MM simulation is not itself differentiable through espaloma

## Relevance to prolix §7.1
- Espaloma is a **parameterization tool**, not an MD engine. Comparing prolix (MD throughput) to espaloma must be carefully scoped — the "espaloma" baseline in our benchmark is likely MD execution using espaloma-assigned parameters in a reference backend (e.g., OpenMM). Clarify this in §7.1.
- espaloma 0.2 is bonded-only (no nonbonded training), which is closest to prolix Scope A — the comparison is relatively apples-to-apples for bonded terms.
- The "same speed as classical MM" claim means espaloma adds no per-step overhead; speedup comparison is prolix vs. whatever backend runs espaloma parameters.
- espaloma has no concept of batching across N=512 molecules in the MD sense.
- Our 56–65× prolix GPU advantage vs. espaloma at N=512 should be explained: prolix batches N=512 simultaneously under JIT while the espaloma baseline likely runs molecules sequentially in OpenMM.

## Notes
- Three distinct espaloma papers: (1) 2022 Chem Sci original, (2) 2024 Chem Sci espaloma-0.3, (3) EspalomaCharge 2024 JPCA — do not conflate.
- Espaloma training on 1.1M QC calculations in "a single GPU-day" is a training cost claim, not inference or MD throughput.
- "same speed as classical MM" means: once you have parameters, simulation cost = OpenMM cost. Our benchmark captures this correctly if the espaloma script runs OpenMM (or equivalent) per molecule.
