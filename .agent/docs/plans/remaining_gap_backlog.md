# Remaining gap backlog (tracked)

Companion to [`explicit_solvent_gaps_and_espaloma_charges.md`](explicit_solvent_gaps_and_espaloma_charges.md). Items below are **not** all scheduled; each needs a ticket when prioritized.

| Priority | Gap | Validation / done criteria |
|----------|-----|----------------------------|
| P2 | Phase 7 L2/L3 (e.g. RDF, observables) | Define metric, reference trajectory (OpenMM/GROMACS export), tolerance |
| P3 | OPC3 pre-equilibrated `opc3.npz` | Asset checksum + one solvation or box test using OPC3 |
| P4 | Phase 8 cluster automation / checkpoints | SLURM template + smoke vs full matrix (see explicit_solvent_benchmarks) |
| P5 | Spatial sorting (Morton / PME scatter) | [spatial_sorting_profile_gate](../../docs/source/explicit_solvent/spatial_sorting_profile_gate.md) — profile first |
| P6 | Cell-list vs JAX-MD NL default | Profiling; keep NL default until data says otherwise |
| Deferred | RESPA / multi-rate integration | After charge + implicit paths stable |

**Espaloma accuracy:** tiers T0–T4 and golden files are described in proxide `README.md` and `tests/data/espaloma_golden/README.md`.
