# L3 observables protocol (skeleton, post-release)

This document is a **placeholder** for research-grade explicit-solvent validation
beyond L2 dynamics (e.g., structural observables).

## Intended scope (future work)

- **Radial distribution functions (RDF):** water–water, water–protein, with bin
  width ~0.1 Å unless documented otherwise.
- **Reference data:** long OpenMM trajectories (10–100 ps) exported in a stable
  format (PDB/DCD/XYZ — to be chosen when implementing).
- **Tolerances (proposal):** peak positions within **1 bin**; peak heights
  within **10%** (to be revisited against measurement noise).

## Current status

No RDF tests are executed in CI yet. See `tests/physics/test_explicit_l3_observables.py`
for a skipped scaffold matching this plan.
