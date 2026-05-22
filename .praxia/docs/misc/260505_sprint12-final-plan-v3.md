# Sprint 12: Zero-Buffer Pivot & High-Precision Parity Restoration (FINAL)

## Objective
Restore < 0.1 kcal/mol parity with OpenMM by eliminating numerical noise (Zero-Buffer Pivot) and implementing missing physical components (ACE nonpolar, Harmonic impropers, Multi-term torsions).

## Phase 1: Zero-Buffer Pivot (Numerical Integrity)
- **Step 1.1**: Implement bitmask-based exclusion logic in `src/prolix/physics/kernels.py` and `src/prolix/physics/neighbor_list.py`. 
    - Implement sparse-to-tile bitmask mapping for efficient O(N) masking.
    - Instead of adding 1e-12/1e-10 to distances, use `jnp.where` masking to skip excluded pairs exactly.
- **Step 1.2**: Refactor `src/prolix/physics/optimization.py` to remove all epsilon stability hacks in the forward pass of LJ and Coulomb kernels.
- **Verification**: `pytest tests/physics/test_numerical_stability.py` (Verify energy parity on 1-2 bonds vs high-precision analytical).

## Phase 2: Force Field Infrastructure (Multi-term & Improper Routing)
- **Step 2.1**: Refactor `src/prolix/physics/force_fields/amber.py` and `topologies.py` to support "multi-row" topology data. 
    - Ensure multi-term torsions are summed correctly.
    - Update Amber parser to correctly route improper torsion parameters to the Harmonic kernel instead of defaulting to periodic.
- **Verification**: `pytest tests/physics/test_amber_multi_term.py` (Verify 321 proper + 9 harmonic improper torsions loaded for 1UAO).

## Phase 3: Missing Physics Kernels
- **Step 3.1**: Implement a dedicated Harmonic Improper kernel ($E = k(\phi - \phi_0)^2$) in `src/prolix/physics/bonded.py`.
- **Step 3.2**: Integrate the ACE nonpolar solvation term into the `total_energy` sum in `src/prolix/physics/system.py` when `implicit_solvent=True`.
- **Verification**: `pytest tests/physics/test_improper_harmonic.py` and `pytest tests/physics/test_gb_ace_parity.py`.

## Phase 4: High-Precision Validation (The Gold Standard)
- **Step 4.1**: Verification of Force vs. Parameter Gradient signs. 
    - Assert $F = -\nabla_r E$ (negative gradient).
    - Assert $\Delta_{param} E = +\nabla_p E$ (positive gradient, per Lesson #31).
- **Step 4.2**: Component-wise parity audit against OpenMM for Chignolin (1UAO). Target $< 0.1$ kcal/mol gap for all components.
- **Verification**: `pytest tests/physics/test_openmm_parity.py` (Full suite).

## Affected Files
- `src/prolix/physics/optimization.py`
- `src/prolix/physics/kernels.py`
- `src/prolix/physics/neighbor_list.py`
- `src/prolix/physics/generalized_born.py`
- `src/prolix/physics/force_fields/amber.py`
- `src/prolix/physics/topologies.py`
- `src/prolix/physics/bonded.py`
- `src/prolix/physics/system.py`
