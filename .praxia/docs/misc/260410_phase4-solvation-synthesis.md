# Phase 4 — Solvation Pipeline NLM Research Synthesis

**Date**: 2026-03-31  
**Source**: NotebookLM — "Molecular Dynamics: Theory, Methods & Practice" (161 sources)  
**Queries**: 9 (8 planned + 1 supplementary on implicit-vs-explicit minimization)

---

## Critical Findings

### 1. OPC3 Water Model — Parameters & Geometry

| Parameter | TIP3P | OPC3 | Source |
|-----------|-------|------|--------|
| q(O) | -0.834 e | -0.8952 e | NLM Q1 (sourced + external) |
| q(H) | +0.417 e | +0.4476 e | NLM Q1 |
| σ(O) | 3.15061 Å | 3.16655 Å | External knowledge — verify |
| ε(O) | 0.1521 kcal/mol | 0.1553 kcal/mol | External knowledge — verify |
| r(O-H) | 0.9572 Å | **0.9782 Å** | NLM Q1 |
| θ(H-O-H) | 104.52° | **109.47°** | NLM Q1 |
| r(H-H) | 1.5139 Å | **1.5533 Å** (computed) | Derived from geometry |
| Citation | Jorgensen et al. 1983 | Izadi & Onufriev 2016 | NLM Q1 |

> **⚠️ CRITICAL**: OPC3 geometry differs from both TIP3P and SPC/E. SETTLE constraints MUST be parameterized per water model. The existing `settle.py` hardcodes TIP3P geometry constants — this needs a `WaterModel` dispatch.

### 2. Ion Parameters — Model-Specific Requirement

| Water Model | Ion Parameter Set | Citation |
|------------|-------------------|----------|
| TIP3P | Joung-Cheatham (`frcmod.ionsjc_tip3p`) | Joung & Cheatham, JPCB 2008 |
| SPC/E | Joung-Cheatham (`frcmod.ionsjc_spce`) | Joung & Cheatham, JPCB 2008 |
| **OPC3** | **Li/Merz** (`frcmod.ionslm_126_opc3`*) | Sengupta et al., JCIM 2021 |

> **⚠️ WARNING**: Do NOT use Joung-Cheatham ions with OPC3. The Li/Merz parameterization was specifically developed for OPC3/OPC/TIP3P-FB/TIP4P-FB. Exact σ/ε values for Na⁺/Cl⁻ not in notebook sources — must extract from AMBER `frcmod` files or Sengupta 2021 paper.

### 3. ff14SB + OPC3 Compatibility — Confirmed

- OpenMM officially bundles `amber14/opc3.xml` for use with `amber14/protein.ff14SB.xml`
- Standard Lorentz-Berthelot combining rules apply — no special cross terms
- The ff19SB paper notes TIP3P introduces helicity bias; OPC3/OPC corrects this
- Both `amber14/` and `amber19/` directories contain identical OPC3 XML files

### 4. Water Bonded Terms & Exclusions

**Key rules for topology merger**:
1. Water O-H bonds and H-O-H angles **ARE** in the topology for minimization
2. During dynamics, SETTLE replaces bonded evaluation (AMBER `ntf=2` skips H-bond energy)
3. All intramolecular water pairs (O-H₁, O-H₂, H₁-H₂) are **fully excluded** from nonbonded (1-2 and 1-3)
4. 3-site water has **no 1-4 pairs** — 1-4 scaling is irrelevant
5. **No protein-water cross-exclusions** — no covalent bonds between them

**Implementation implication**: Water bonds/angles go into `PaddedSystem` bonded arrays for minimization, but the MD integrator should skip evaluating them when SETTLE is active. Exclusion lists must include all 3 intramolecular pairs per water.

### 5. Ion Placement Protocol

- **Recommended**: Coulombic grid-based placement (AMBER `addIons` default)
- **Acceptable**: Random placement with post-hoc randomization (`randomizeions`)
- **Min ion-solute distance**: 5.0 Å buffer
- **Min ion-ion distance**: 3.0 Å 
- JC parameters do NOT require specific placement — ions equilibrate during minimization/dynamics
- Our current `add_ions()` random placement is acceptable; Coulombic placement is better but not critical

### 6. Box Sizing

- **Minimum padding**: 10-12 Å from protein to box edge
- **PME constraint**: Box dimension ≥ 2 × real-space cutoff (minimum image convention)
- Default 10 Å padding with 9-10 Å cutoff is within best practices

### 7. Equilibration Protocol (Phase 9 Forward-Looking)

Standard AMBER ff14SB protocol:
1. **Minimization** (staged restraint release):
   - 500 kcal/mol/Å² heavy atoms → SD 500 steps + CG 1000-4500 steps
   - Step down: 500 → 125 → 25 → 0 kcal/mol/Å² on backbone
2. **NVT Heating**: 0K → 300K over 100-300 ps, Langevin γ=1.0-2.0 ps⁻¹, backbone restraints 5-10 kcal/mol/Å²
3. **NPT Density Equilibration**: 1 atm, release restraints (5→1→0.1→0 kcal/mol/Å²), 1-2 ns
4. **Global settings**: dt=2 fs (4 fs with HMR), SHAKE/SETTLE, PME, cutoff 8-10 Å

### 8. Validation Metrics (Phase 7 Forward-Looking)

| Property | TIP3P | OPC3 | Experiment | Tolerance |
|----------|-------|------|------------|-----------|
| Density (g/cm³) | 0.97-0.99 | ~0.999 | 0.997 | ±2% |
| Diffusion (10⁻⁵ cm²/s) | 5.1-5.8 | ~2.4 | 2.3 | ±5-10% |
| O-O RDF 1st peak (Å) | ~2.75 | ~2.78 | 2.76 | ±0.05 Å |
| Dielectric | 82-97 | ~78 | 78.5 | ±10% |

**NVE energy drift** target: ≤ 0.02 kBT/ns per DOF (cross-code comparison shows systematic offsets between packages are common).

### 9. Explicit vs Implicit Minimization/Thermalization

| Aspect | Implicit (GB) | Explicit |
|--------|--------------|----------|
| Initial restraints | Heavy atoms to resolve steric clashes | Heavy atoms + **solvent relaxation** (water must reorganize around protein) |
| Why restrain | Local clash resolution | Prevent artificial solvent-solute forces from tearing protein apart |
| NVT heating | Introduces KE to protein DOFs | Thermalizes water molecules out of artificial packing + prerequisite for NPT |
| NPT phase | Not needed (no physical box volume) | **Required** — box must contract to physical water density |
| Staged release | During temperature ramp | During **NPT density equilibration** (after NVT) |
| Protein freezing | Optional | **Critical** during initial solvent relaxation (water dipole reorientation) |

---

## Implementation Decisions

Based on this grounding, the following architectural decisions are confirmed:

1. **`WaterModel` enum**: `TIP3P` and `OPC3` with separate parameter sets (charges, LJ, geometry for SETTLE)
2. **Ion parameters**: TIP3P → Joung-Cheatham, OPC3 → Li/Merz (Sengupta 2021)
3. **Combining rules**: Standard Lorentz-Berthelot for both models
4. **Topology merger**: Include water bonds/angles in PaddedSystem for minimization; skip during dynamics
5. **Water exclusions**: All 3 intramolecular pairs fully excluded (1-2 and 1-3), no 1-4 pairs
6. **No cross-exclusions**: Protein-water interactions governed purely by cutoff + PME
7. **Box sizing**: Validate padding ≥ 10 Å and box ≥ 2 × cutoff
8. **Ion placement**: Keep random placement (acceptable), add min-distance validation

## Open Items Requiring External Verification

- [ ] OPC3 LJ parameters (σ=3.16655, ε=0.1553) — flagged as external knowledge by NLM, need to verify against Izadi 2016 paper or AMBER XML
- [ ] Li/Merz Na⁺/Cl⁻ σ/ε for OPC3 — need to extract from `frcmod.ionslm_126_opc3` or Sengupta 2021
- [ ] OPC3 H-H distance — computed as 2×0.9782×sin(109.47°/2) = 1.5533 Å, needs verification
