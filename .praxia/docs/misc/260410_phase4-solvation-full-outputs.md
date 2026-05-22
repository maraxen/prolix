# NLM Query 1: OPC3 Water Model Geometry

**Query**: What are the analytical geometry parameters for the OPC3 water model required for the SETTLE algorithm? Specifically, provide the O-H bond length (r_OH) and the H-O-H angle (theta).

**Result**:
- O-H bond length ($r_{OH}$): 0.9782 Å
- H-O-H angle ($\theta$): 109.47°
- H-H distance ($r_{HH}$): ~1.5533 Å (computed as $2 \times r_{OH} \times \sin(\theta/2)$)

**Source**: Izadi & Onufriev (2016), confirmed via OpenMM `opc3.xml`.

---

# NLM Query 2: OPC3-Compatible Ion Parameters

**Query**: Which ion parameter sets (Na+, Cl-) are recommended for use with the OPC3 water model in AMBER-style simulations? Are Joung-Cheatham parameters compatible, or are there model-specific sets like Li/Merz?

**Result**:
- **Recommended**: Li/Merz (Sengupta et al., JCIM 2021) ion parameters are specifically optimized for OPC3.
- **Compatibility**: Joung-Cheatham parameters (optimized for TIP3P/TIP4P-Ew) will lead to incorrect hydration free energies and ion-pairing behavior when used with OPC3.

---

# NLM Query 3: Water Intramolecular bonded terms in topology

**Query**: In a solvated AMBER system (Protein + OPC3/TIP3P), should the water intramolecular bonds and angles be included in the topology for minimization, or are they exclusively handled by constraints (SETTLE/SHAKE)?

**Result**:
- **Minimization**: Included. Bonds and angles should be present in the topology with standard force constants (e.g., $k_{bond} \approx 553$ kcal/mol/Å²) to prevent unrealistic geometries before SETTLE is active.
- **Dynamics**: Skipped. The SETTLE algorithm analytical solution overrides these terms by fixing distances.

---

# NLM Query 4: Nonbonded Exclusions for Water

**Query**: What are the standard nonbonded exclusion rules for 3-site water models (TIP3P, OPC3)? Should the 1-2 (O-H) and 1-3 (H-H) pairs be fully excluded?

**Result**:
- **Exclusion Rule**: All 3 intramolecular pairs (O-H1, O-H2, H1-H2) must be **fully excluded** (scale=0.0) from both Lennard-Jones and Electrostatic evaluations.

---

# NLM Query 5: Protein-Water Cross-Exclusions

**Query**: Are there any standard nonbonded exclusions between protein atoms and water atoms in the AMBER ff14SB force field?

**Result**:
- **Exclusion Rule**: None. Protein-water interactions are entirely nonbonded (LJ + Coulomb) with no covalent exclusions.

---

# NLM Query 6: PME Reciprocal Force Calculation

**Query**: What is the recommended strategy for calculating reciprocal space forces in a JAX-native PME implementation to avoid massive VRAM usage during backpropagation?

**Result**:
- **Strategy**: Use `jax.custom_vjp`. Manually define the gradients of the reciprocal energy with respect to charges and grid coordinates to avoid JAX's default checkpointing of large 3D FFT grids.

---

# NLM Query 7: Explicit Solvent Equilibration Restraints

**Query**: What are the standard restraint protocols for equilibrating a solvated protein system starting from a crystal structure? Specify restraint targets and force constants.

**Result**:
- **Protocol**: Apply harmonic restraints to **protein heavy atoms**.
- **Force Constant**: Start at ~500 kcal/mol/Å² (10 kcal/mol/Å² per atom approx) and stage down (125 → 25 → 0) during minimization and NVT heating.

---

# NLM Query 8: Water Model Validation Benchmarks

**Query**: What are the primary physical properties used to validate TIP3P and OPC3 water model implementations in a new MD engine? Provide reference values for density and diffusion.

**Result**:
- **Density**: TIP3P (0.98 g/cm³), OPC3 (0.997 g/cm³, target 1.0).
- **Diffusion**: TIP3P (5.5e-5 cm²/s), OPC3 (2.3e-5 cm²/s).
- **Other**: O-O RDF peaks (2.75 Å, 4.5 Å).

---

# NLM Query 9: FF14SB + OPC3 Compatibility

**Query**: Does AMBER ff14SB require any special nonbonded scaling or parameter adjustments when used with the OPC3 water model compared to TIP3P?

**Result**:
- **Compatibility**: No special adjustments. Both use standard Lorentz-Berthelot combining rules. Just ensure the OPC3-specific oxygen LJ parameters and Li/Merz ions are used.
