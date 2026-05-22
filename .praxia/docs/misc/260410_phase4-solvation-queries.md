# NLM Query Set: Phase 4 — Explicit Solvation Pipeline (Revised)

## Target Notebook
- **Name**: Molecular Dynamics: Theory, Methods & Practice
- **ID**: `9230d5f7-cff8-49a1-9ccd-8b65e8e207a7`
- **Sources**: 161

## Purpose
Ground Phase 4 implementation (solvation pipeline enhancement) with authoritative parameters, topology merger rules, and validation targets. Focus on exact numerical values that cannot be derived from first principles.

## Oracle Critique Status
- **Round 1**: REVISE — merged Q3/Q4 overlap, added OPC3 geometry query to Q1, added ff14SB+OPC3 compatibility query. Forward-looking queries (Q7, Q8) retained but labeled.

---

## Query Set (8 queries)

### Q1: OPC3 vs TIP3P Water Model Parameters (Complete)
**Query**: "What are the exact force field parameters for the OPC3 3-site water model compared to TIP3P? For each model I need: (1) partial charges on oxygen and hydrogen, (2) Lennard-Jones sigma and epsilon for the oxygen atom, (3) O-H bond length, (4) H-O-H angle, (5) the original citation. Critically: does OPC3 use the same geometry as TIP3P (r_OH=0.9572 Å, θ=104.52°) or SPC/E geometry (r_OH=1.0 Å, θ=109.47°)? This determines SETTLE constraint parameters."

**Rationale**: Complete parameter table for both models + geometry determination for SETTLE.

### Q2: Joung-Cheatham Ion Parameters
**Query**: "What are the Joung-Cheatham monovalent ion parameters for Na+ and Cl- optimized for use with the SPC/E and TIP3P water models in AMBER? I need the Lennard-Jones sigma, epsilon, and charge for each ion. Also, are there Joung-Cheatham parameters specifically re-optimized for the OPC3 water model, or should one use the SPC/E-fit parameters?"

**Rationale**: Ion parameters must be consistent with the water model. Need to determine correct parameter set for OPC3.

### Q3: Water Bonded Terms and Exclusions (Merged)
**Query**: "When solvating a protein with AMBER ff14SB + TIP3P or OPC3 explicit water, how are water intramolecular interactions handled in practice? Specifically: (1) Are O-H bonds and H-O-H angles included in the bonded term lists, or handled exclusively through SETTLE/SHAKE constraints? (2) What nonbonded exclusions are required for water molecules — are all intramolecular pairs (O-H₁, O-H₂, H₁-H₂) fully excluded from LJ and Coulomb evaluation? (3) How do 1-4 scaling factors apply to water — does a 3-site water molecule even have 1-4 pairs? (4) Are protein-water cross-exclusions ever needed beyond standard cutoff neighbor lists?"

**Rationale**: Determines whether water bonds/angles appear in PaddedSystem arrays and what exclusions are required. Incorrect handling causes double-counted energy or catastrophic forces.

### Q4: ff14SB + OPC3 Compatibility
**Query**: "Is the AMBER ff14SB protein force field validated for use with the OPC3 water model? Does OPC3 require special Lennard-Jones combining rules or re-parameterized protein-water cross terms beyond standard Lorentz-Berthelot combining rules? What force field and water model combinations have been published and validated with OPC3?"

**Rationale**: Using the wrong force field + water model combination corrupts system thermodynamics. Must confirm ff14SB+OPC3 is a validated pairing.

### Q5: Neutralization and Ion Placement Protocols
**Query**: "What is the standard protocol for neutralizing a solvated protein system and adding ionic strength? How should counterions be placed — random water replacement vs. Coulombic potential-based placement? What is the recommended minimum distance between placed ions and between ions and solute? Does Joung-Cheatham assume a specific ion placement protocol?"

**Rationale**: Current `add_ions()` uses random placement. Need to verify this is acceptable.

### Q6: Box Sizing and PME Cutoff Compatibility
**Query**: "For explicit solvent protein simulations with PME electrostatics, what is the recommended minimum padding between the protein and box edge? What is the relationship between the PME real-space cutoff and the minimum box dimension? Why must the box dimension be at least 2× the real-space cutoff?"

**Rationale**: Validate our default 10 Å padding and ensure box dimensions are compatible with PME cutoff.

### Q7: Equilibration Protocol (Forward-Looking: Phase 9)
**Query**: "What is the recommended multi-stage equilibration protocol for a solvated protein system before production MD? Include: (1) minimization stages with and without restraints, (2) NVT heating schedule (temperature ramp), (3) NPT density equilibration parameters, (4) restraint protocol and force constants, (5) typical durations for each stage. Focus on AMBER ff14SB with explicit solvent."

**Rationale**: Needed for Phase 9 production integration. Querying now to have grounding data ready.

### Q8: Water Model Validation Targets (Forward-Looking: Phase 7)
**Query**: "What experimental observables should be compared when validating an explicit solvent MD implementation? For both TIP3P and OPC3, what are the expected values for: (1) bulk water density at 298K and 1 atm, (2) self-diffusion coefficient, (3) O-O radial distribution function first peak position and height, (4) dielectric constant? What numerical tolerances indicate a correctly implemented water model vs. a subtly broken one?"

**Rationale**: Phase 7 validation needs quantitative targets. Querying now to have reference values ready.
