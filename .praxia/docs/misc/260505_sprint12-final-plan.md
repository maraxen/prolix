# Sprint 12: Zero-Buffer Pivot & High-Precision Parity Restoration

## Objective
Restore < 1 kcal/mol parity with OpenMM by eliminating numerical noise (Zero-Buffer Pivot) and implementing missing physical components (ACE nonpolar, Harmonic impropers, Multi-term torsions).

## Phase 1: Zero-Buffer Pivot (Numerical Stability)
- **Step 1.1**: Implement bitmask-based exclusion logic in `src/prolix/physics/kernels.py`. Instead of adding 1e-12/1e-10 to distances, use `jnp.where` masking to skip excluded pairs exactly.
- **Step 1.2**: Refactor `src/prolix/physics/optimization.py` to remove all epsilon stability hacks in the forward pass of LJ and Coulomb kernels.
- **Verification**: `pytest tests/physics/test_numerical_stability.py` (Verify energy parity on 1-2 bonds vs high-precision analytical).

## Phase 2: Force Field Infrastructure (Multi-term Support)
- **Step 2.1**: Refactor `src/prolix/physics/force_fields/amber.py` and `topologies.py` to support "multi-row" topology data. Ensure that the same 4 atoms can have multiple periodic torsion terms summed correctly instead of being overwritten.
- **Verification**: `pytest tests/physics/test_amber_multi_term.py` (Verify all 321+9 torsions for Chignolin are loaded).

## Phase 3: Missing Physics Kernels
- **Step 3.1**: Implement a dedicated Harmonic Improper kernel ($E = k(\phi - \phi_0)^2$) in `src/prolix/physics/bonded.py`.
- **Step 3.2**: Integrate the ACE nonpolar solvation term into the `total_energy` sum in `src/prolix/physics/system.py` when `implicit_solvent=True`.
- **Verification**: `pytest tests/physics/test_improper_harmonic.py` and `pytest tests/physics/test_gb_ace_parity.py`.

## Phase 4: High-Precision Validation
- **Step 4.1**: Verification of Force vs. Parameter Gradient signs. Ensure $F = -\nabla E$ holds for all new kernels to $<10^{-7}$ precision.
- **Step 4.2**: Component-wise parity audit against OpenMM for Chignolin (1UAO). Target $< 0.1$ kcal/mol gap for all bonded terms.
- **Verification**: `pytest tests/physics/test_openmm_parity.py` (Full suite).

## Affected Files
- `src/prolix/physics/optimization.py`
- `src/prolix/physics/kernels.py`
- `src/prolix/physics/generalized_born.py`
- `src/prolix/physics/force_fields/amber.py`
- `src/prolix/physics/topologies.py`
- `src/prolix/physics/bonded.py`
- `src/prolix/physics/system.py`
