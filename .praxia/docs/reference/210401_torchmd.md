---
name: torchmd
description: "TorchMD: A Deep Learning Framework for Molecular Simulations — Doerr, Majewski, Pérez et al., JCTC, 2021"
metadata:
  type: reference
---

## Citation
Stefan Doerr, Maciej Majewski, Adrià Pérez, Andreas Krämer, Cecilia Clementi, Frank Noe, Toni Giorgino, and Gianni De Fabritiis. "TorchMD: A Deep Learning Framework for Molecular Simulations." *Journal of Chemical Theory and Computation* 17, no. 4 (2021): 2355–2363. DOI: 10.1021/acs.jctc.0c01343. PMC: PMC8486166.

## What it claims
- TorchMD provides a unified framework for molecular simulations with mixed classical and ML potentials, all expressed as PyTorch tensors.
- All classical force computations — bonds, angles, dihedrals, Lennard-Jones, Coulomb — are implemented as differentiable PyTorch operations.
- AMBER force field parameters can be loaded via `parmed`.
- The framework enables straightforward integration of neural network potentials alongside classical terms.
- TorchMD is described as a research and prototyping tool, not a production-throughput engine; the paper explicitly acknowledges it is ~60× slower than ACEMD3.
- Performance bottleneck is the absence of neighbor lists for nonbonded interactions; all pairwise distances must fit in GPU memory.

## Benchmark methodology
- **Hardware tested:** NVIDIA Titan V GPU
- **N tested:** Three systems — (1) periodic water box of 97 water molecules (~291 atoms), (2) alanine dipeptide in vacuum, (3) trypsin with benzamidine ligand (~3000+ atoms)
- **Metric reported:** Relative throughput vs. ACEMD3; simulation wall-clock time for 50,000 steps at 1 fs/step (50 ps)
- **Baseline compared against:** ACEMD3 (highly optimized CUDA MD engine from the same group)
- **Result:** TorchMD is approximately 60× slower than ACEMD3 on the same Titan V GPU across all test systems
- **Note:** No absolute throughput (ns/day) is stated for TorchMD itself; only the ratio to ACEMD3 is reported

## Force field scope
- **Bonded terms:** Yes — bonds, angles, proper and improper dihedrals
- **Nonbonded terms:** Yes — Lennard-Jones (12-6) and Coulomb; no neighbor list, so O(N²) pairwise enumeration
- **Differentiable?** Yes — all terms are PyTorch autograd-compatible

## Relevance to prolix §7.1
- TorchMD includes nonbonded terms (LJ + Coulomb); prolix Scope A is bonded-only. TorchMD is doing more per step than prolix in our benchmark — the 2800× prolix advantage is a conservative lower bound.
- TorchMD's benchmark GPU (Titan V, 2017) is far older than A100/H200/Blackwell hardware; hardware generation mismatch must be acknowledged in §7.1.
- TorchMD's stated design goal is flexibility for ML research, not throughput. The comparison is fair framing-wise as long as we note context.
- TorchMD benchmarks single-system scaling (up to ~3000 atoms); prolix benchmarks N=512 independent molecules — orthogonal scaling dimensions.

## Notes
- Paper's preprint on arXiv: 2012.12106 (December 2020); published JCTC April 2021.
- TorchMD-Net (neural network potentials) is distinct from TorchMD the MD engine — do not conflate.
- The 60× slowdown vs. ACEMD3 is for full classical FF including nonbonded. A bonded-only comparison would be more favorable to TorchMD; no such number is reported.
- Missing features at time of publication: hydrogen bond constraints, neighbor lists.
