# Prolix Parity Testing Roadmap

## Current Status: ✅ Primary Goals Achieved

- ✅ **Hydrogen Addition:** Enabled and validated.
- ✅ **Explicit Solvent Parity:** Initial energy parity achieved.
- ✅ **Simulation Stability:** SETTLE + Langevin stable.

---

## Phase 1: Energy Parity (COMPLETED)

- ✅ Bonded energy parity (bond, angle, dihedral).
- ✅ Basic nonbonded validation.
- ✅ CMAP support.

---

## Phase 2: Force and Trajectory Parity (ACTIVE)

### Goals

Full validation of forces and dynamics against OpenMM.

### Tasks

- [ ] `test_force_parity` - Vector comparison of forces.
- [ ] `test_14_nonbonded_parity` - Detailed 1-4 scaling validation.
- [ ] `test_trajectory_parity` - Comparison of N-step dynamics.

---

## Phase 3: Implicit Solvent (GBSA) Parity

### Goals

GBSA energy and force parity between Prolix and OpenMM.

### Tasks

- [ ] `test_gbsa_born_radii_parity`
- [ ] `test_gbsa_total_energy_parity`

---

## Phase 4: Extended Conditions

- [ ] Test multiple forcefields (ff14SB, ff19SB).
- [ ] Test different water models (TIP3P, SPC/E).
