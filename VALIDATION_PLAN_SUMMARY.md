# Validation Plan Summary: Explicit Solvent

**Executive Overview:**
Ensuring high-fidelity physical correctness for explicit solvent molecular dynamics in `prolix`.

**Key Objectives:**
- **Energy Parity:** < 1e-5 relative error vs OpenMM (Component-wise: Bonded, Non-bonded, PME, etc.).
- **Force Parity:** MAE < 1e-4 kcal/mol/Å; NL vs Dense consistency; Triclinic support.
- **Topology:** Full parity for Amber terminal residues and ACE/NME capping groups.
- **Statistical Fidelity:** BAOAB parity; Multi-water (TIP3P, TIP4P, OPC) RDFs.
- **Stability:** 10ns simulation across CPU/GPU/TPU.

**Timeline:**
- Phase 1 (Energy): Weeks 1-2.
- Phase 2 (Force): Weeks 3-4.
- Phase 3 (Statistical): Weeks 5-7.
- Phase 4 (Performance): Weeks 8-9.

**Core Requirements:**
- Access to 1UAO solvated PDB.
- A100/H100 for long-term trajectory validation.
- Standardized PME configuration policy.
