# Oracle Critique Report (Cycle 4)

**Verdict: Changes Requested**

## 1. Numerical Stability Gaps: Virtual Site "Shadow" Forces
- **Finding:** The plan (P1e) mentions force redistribution but lacks detail on integrator synchronization.
- **Requirement:** Virtual sites must be reconstructed after every coordinate update in the integrator (e.g., mid-step in BAOAB) to avoid unphysical "lagging" charges.
- **Action:** Add "Integrator-Virtual Site Synchronization" to Phase 2.

## 2. PBC Wrapping Policy: Molecule Integrity
- **Finding:** No mention of molecule-aware wrapping for triclinic cells or pressure/virial calculations.
- **Requirement:** Atomic wrapping breaks molecules at boundaries, causing pressure spikes. Molecule-aware wrapping is mandatory for virial parity.
- **Action:** Add "Molecule-Aware Wrapping Policy" to Phase 2.

## 3. Padded vs. Non-Padded Parity
- **Finding:** P2b focuses on NL vs Dense but misses padding-specific validation.
- **Requirement:** Verify that JAX-MD's padded neighbor lists (for GPU) match non-padded results exactly.
- **Action:** Amend P2b to include "Padded-vs-Unpadded Parity".

## 4. Tooling & DX Status
- **Finding:** `just` commands mentioned in the plan are not in the `Justfile`.
- **Requirement:** Explicitly list the creation of the validation CLI harness as a deliverable.
- **Action:** Update Phase 1 roadmap to include `Justfile` expansion.

