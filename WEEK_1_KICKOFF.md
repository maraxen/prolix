# Week 1 Kickoff: Explicit Solvent Validation

## Tasks (Week of April 16, 2026)

### 1. P1a: Solvated Protein Anchor Test
- **Objective:** Finalize a robust energy parity test for a solvated protein (1UAO).
- **Sub-tasks:**
    - Verify 1-4 scaling and Dispersion parity.
    - Verify terminal residue (NALA, CALA) and capping (ACE, NME) parity.
    - Perform component-wise energy breakdown comparison.
- **Owner:** [Lead Engineer]
- **Deliverable:** `tests/test_explicit_solvation_parity.py` passing with < 1e-5 error.

### 2. P1b: PME Policy Documentation
- **Objective:** Establish the canonical PME grid and alpha parameters for all regression tests.
- **Owner:** [MD Specialist]
- **Deliverable:** `regression_pme_config.yaml` and policy doc.

### 3. SETTLE Wiring Verification
- **Objective:** Confirm `rigid_water` flag in `SimulationSpec` correctly enables SETTLE in the MD loop.
- **Owner:** [Integrator Lead]
- **Deliverable:** `test_settle_integration.py` verifying distance constraints over 1ps.

## Meeting Agenda
1. Review Comprehensive Plan v1.0.
2. Discuss PME alpha/grid trade-offs.
3. Align on statistical sampling requirements (Number of water boxes, simulation length).
