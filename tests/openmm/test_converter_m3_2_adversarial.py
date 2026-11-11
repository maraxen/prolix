"""Adversarial stress-testing suite for Milestone 3 (OpenMM System Converter).

Focus areas (Challenger M3-2):
1. NonbondedForce Exclusions & Exceptions:
   - Dense 32 exclusion cap enforcement (>32 per atom raises clear ValueError).
   - Unified 1-4 exception handling (scales 0.0 in excl_indices, modified pairs in exception_*).
   - Nonbonded methods: NoCutoff, CutoffNonPeriodic, CutoffPeriodic (Reaction Field), Ewald, PME.
   - Exact analytical parity of reaction-field coefficients (k_rf, c_rf) vs OpenMM formulas.
   - Switching function activation and dispersion correction parameter propagation.
2. Custom Forces Routing:
   - Custom force mode with all 3 settings ("auto", "pattern_only", "lepton_only").
   - Native bundle routing for CHARMM impropers, periodic torsions, harmonic angles, harmonic bonds.
   - Fallback to Lepton compilation and isolation of non-default force groups.
   - Position restraints extraction with isotropic and custom expressions, coordinate conversion.
3. Force Groups & Global Parameters:
   - Systems with 8+ distinct forces across different forceGroup indices (0..31).
   - Verification of force_groups mapping and force_group_by_component.
   - Global parameter extraction and default values across positive, negative, zero, and extreme values.
4. Fail-Fast Catalogue:
   - Systematic validation of all 24 unsupported OpenMM force classes from UNSUPPORTED_OPENMM_FORCES.
   - NonbondedForce LJPME fail-fast.
   - Arbitrary unrecognized force fail-fast.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import openmm
import pytest
from openmm import unit

from prolix.openmm.config import ProlixOpenMMConfig
from prolix.openmm.converter import (
    _EXCL_DENSE_MAX,
    ConvertedOpenMMSystem,
    PositionRestraints,
    from_openmm_system,
)
from prolix.openmm.exceptions import (
    UNSUPPORTED_OPENMM_FORCES,
    UnsupportedOpenMMForceError,
)
from prolix.openmm.units import (
    ANGLE_K_OMM_TO_AKMA,
    BOND_K_OMM_TO_AKMA,
    KJ_PER_MOL_TO_KCAL_PER_MOL,
    NM_TO_ANGSTROM,
    TORSION_K_OMM_TO_AKMA,
)
from prolix.physics.electrostatic_methods import openmm_reaction_field_coefficients
from prolix.physics.neighbor_list import pair_list_to_dense_padded


# =============================================================================
# Helper Utilities
# =============================================================================


def _make_dummy_system(n_particles: int = 4, mass: float = 12.0) -> openmm.System:
    """Construct minimal OpenMM system with specified particle count."""
    sys = openmm.System()
    for _ in range(n_particles):
        sys.addParticle(mass)
    return sys


# =============================================================================
# 1. NonbondedForce Exclusions & Exceptions
# =============================================================================


class TestNonbondedExclusionsAndExceptions:
    """Adversarial stress tests for NonbondedForce exclusions, exceptions, and methods."""

    def test_excl_cap_exactly_32_passes(self):
        """Atom with exactly 32 exclusions must succeed and populate all 32 slots."""
        n_atoms = 33
        sys = _make_dummy_system(n_atoms)
        nb = openmm.NonbondedForce()
        for _ in range(n_atoms):
            nb.addParticle(0.0, 0.3, 0.0)

        # Atom 0 excluded with atoms 1..32 (exactly 32 exclusions)
        for other in range(1, 33):
            nb.addException(0, other, 0.0, 0.1, 0.0)
        sys.addForce(nb)

        conv = from_openmm_system(sys)
        bundle = conv.bundle

        # Atom 0 row in excl_dense_indices must have exactly 32 valid entries (none -1)
        row_0 = np.asarray(bundle.excl_dense_indices[0])
        assert len(row_0) == _EXCL_DENSE_MAX
        assert not np.any(row_0 == -1)
        assert set(row_0) == set(range(1, 33))

        # Each other atom should have exactly 1 exclusion (with atom 0) and 31 padding slots (-1)
        for other in range(1, 33):
            row_other = np.asarray(bundle.excl_dense_indices[other])
            assert row_other[0] == 0
            assert np.all(row_other[1:] == -1)

    def test_excl_cap_33_raises_clear_value_error(self):
        """Atom with 33 exclusions (> 32 cap) must raise ValueError with detailed message."""
        n_atoms = 34
        sys = _make_dummy_system(n_atoms)
        nb = openmm.NonbondedForce()
        for _ in range(n_atoms):
            nb.addParticle(0.0, 0.3, 0.0)

        # Atom 0 excluded with atoms 1..33 (33 exclusions)
        for other in range(1, 34):
            nb.addException(0, other, 0.0, 0.1, 0.0)
        sys.addForce(nb)

        with pytest.raises(ValueError) as exc_info:
            from_openmm_system(sys)

        err_msg = str(exc_info.value)
        assert "exclusion topology needs 33 slots per atom" in err_msg
        assert "max_exclusions=32" in err_msg

    def test_excl_cap_interior_atom_exceeded_raises(self):
        """Interior atom (e.g. index 25) with >32 exclusions must trigger the cap."""
        n_atoms = 60
        sys = _make_dummy_system(n_atoms)
        nb = openmm.NonbondedForce()
        for _ in range(n_atoms):
            nb.addParticle(0.0, 0.3, 0.0)

        target_atom = 25
        # Add 35 exclusions to atom 25
        count = 0
        for other in range(n_atoms):
            if other != target_atom and count < 35:
                nb.addException(min(target_atom, other), max(target_atom, other), 0.0, 0.1, 0.0)
                count += 1
        sys.addForce(nb)

        with pytest.raises(ValueError) as exc_info:
            from_openmm_system(sys)

        err_msg = str(exc_info.value)
        assert "exclusion topology needs 35 slots per atom" in err_msg
        assert "max_exclusions=32" in err_msg

    def test_excl_cap_bidirectional_accumulation(self):
        """Verify that bidirectional accumulation of exclusions correctly tracks per-atom degree."""
        n_atoms = 40
        sys = _make_dummy_system(n_atoms)
        nb = openmm.NonbondedForce()
        for _ in range(n_atoms):
            nb.addParticle(0.0, 0.3, 0.0)

        # Atom 10 added as index in 20 exceptions (other < 10 or 10 < other < 21)
        # Exclude atom 10 with 0..9 (10 atoms) and 11..20 (10 atoms) -> 20 exclusions
        for other in list(range(10)) + list(range(11, 21)):
            nb.addException(min(10, other), max(10, other), 0.0, 0.1, 0.0)
        # Now add 13 more with distinct atoms 21..33 (13 atoms) -> total 33 exclusions
        for other in range(21, 34):
            nb.addException(10, other, 0.0, 0.1, 0.0)
        sys.addForce(nb)

        with pytest.raises(ValueError) as exc_info:
            from_openmm_system(sys)

        err_msg = str(exc_info.value)
        assert "exclusion topology needs 33 slots per atom" in err_msg
        assert "max_exclusions=32" in err_msg

    def test_unified_14_pure_exclusions_only(self):
        """Zero-charge and zero-epsilon exclusions appear in excl_indices with scale 0.0 and NOT in exception_*."""
        sys = _make_dummy_system(5)
        nb = openmm.NonbondedForce()
        for _ in range(5):
            nb.addParticle(0.2, 0.3, 1.0)

        # 4 pure exclusions (1-2 and 1-3 pairs)
        nb.addException(0, 1, 0.0, 0.1, 0.0)
        nb.addException(1, 2, 0.0, 0.1, 0.0)
        nb.addException(2, 3, 0.0, 0.1, 0.0)
        nb.addException(3, 4, 0.0, 0.1, 0.0)
        sys.addForce(nb)

        conv = from_openmm_system(sys)
        bundle = conv.bundle

        assert int(bundle.n_excl) == 4
        assert int(bundle.n_exception_pairs) == 0

        # All 4 pairs in excl_indices with scales 0.0
        for i in range(4):
            assert float(bundle.excl_scales_vdw[i]) == 0.0
            assert float(bundle.excl_scales_elec[i]) == 0.0

    def test_unified_14_modified_pairs_in_both(self):
        """Modified 1-4 exception pairs must appear in excl_indices (scale 0.0) AND exception_* arrays."""
        sys = _make_dummy_system(6)
        nb = openmm.NonbondedForce()
        for _ in range(6):
            nb.addParticle(0.3, 0.3, 1.5)

        # Pair (0, 1): pure exclusion
        nb.addException(0, 1, 0.0, 0.1, 0.0)
        # Pair (0, 3): full 1-4 exception (qprod != 0 and eps != 0)
        qprod = 0.09  # e^2
        sig_nm = 0.34
        eps_kj = 0.75  # kJ/mol
        nb.addException(0, 3, qprod, sig_nm, eps_kj)
        # Pair (1, 4): pure electrostatic exception (eps == 0, qprod != 0)
        nb.addException(1, 4, -0.05, 0.2, 0.0)
        # Pair (2, 5): pure LJ exception (qprod == 0, eps != 0)
        nb.addException(2, 5, 0.0, 0.31, 0.45)
        sys.addForce(nb)

        conv = from_openmm_system(sys)
        bundle = conv.bundle

        # All 4 pairs are in excl_indices with 0.0 scales (bulk nonbonded exclusion)
        assert int(bundle.n_excl) == 4
        for i in range(4):
            assert float(bundle.excl_scales_vdw[i]) == 0.0
            assert float(bundle.excl_scales_elec[i]) == 0.0

        # Exactly 3 non-zero pairs must be in exception_*
        assert int(bundle.n_exception_pairs) == 3

        # Pair (0, 3)
        assert tuple(bundle.exception_pairs[0]) == (0, 3)
        assert np.isclose(float(bundle.exception_chargeprods[0]), qprod, atol=1e-6)
        assert np.isclose(float(bundle.exception_sigmas[0]), sig_nm * NM_TO_ANGSTROM, atol=1e-5)
        assert np.isclose(
            float(bundle.exception_epsilons[0]),
            eps_kj * KJ_PER_MOL_TO_KCAL_PER_MOL,
            atol=1e-5,
        )

        # Pair (1, 4) - negative chargeprod preserved
        assert tuple(bundle.exception_pairs[1]) == (1, 4)
        assert np.isclose(float(bundle.exception_chargeprods[1]), -0.05, atol=1e-6)
        assert np.isclose(float(bundle.exception_epsilons[1]), 0.0, atol=1e-6)

        # Pair (2, 5) - pure LJ exception
        assert tuple(bundle.exception_pairs[2]) == (2, 5)
        assert np.isclose(float(bundle.exception_chargeprods[2]), 0.0, atol=1e-6)
        assert np.isclose(
            float(bundle.exception_epsilons[2]),
            0.45 * KJ_PER_MOL_TO_KCAL_PER_MOL,
            atol=1e-5,
        )

    def test_unified_14_reversed_pair_ordering(self):
        """Exceptions added with p1 > p2 must be canonically ordered min(p1, p2), max(p1, p2)."""
        sys = _make_dummy_system(5)
        nb = openmm.NonbondedForce()
        for _ in range(5):
            nb.addParticle(0.1, 0.3, 1.0)

        # Reverse ordering: (4, 1) and (3, 0)
        nb.addException(4, 1, 0.0, 0.1, 0.0)
        nb.addException(3, 0, 0.05, 0.35, 0.8)
        sys.addForce(nb)

        conv = from_openmm_system(sys)
        bundle = conv.bundle

        # excl_indices must have (1, 4) and (0, 3)
        excl_pairs = [tuple(p) for p in bundle.excl_indices[:2]]
        assert (1, 4) in excl_pairs
        assert (0, 3) in excl_pairs

        # exception_pairs must have (0, 3)
        assert tuple(bundle.exception_pairs[0]) == (0, 3)

    @pytest.mark.parametrize(
        ("omm_method", "expected_method", "has_pbc", "expected_cutoff_A"),
        [
            (openmm.NonbondedForce.NoCutoff, "coulomb", False, float("inf")),
            (openmm.NonbondedForce.CutoffNonPeriodic, "cutoff", False, 11.0),
            (openmm.NonbondedForce.CutoffPeriodic, "reaction_field", True, 9.5),
            (openmm.NonbondedForce.Ewald, "pme", True, 8.5),
            (openmm.NonbondedForce.PME, "pme", True, 10.0),
        ],
    )
    def test_nonbonded_methods_mapping(
        self, omm_method, expected_method, has_pbc, expected_cutoff_A
    ):
        """Verify nonbonded method translation across all 5 standard methods."""
        sys = _make_dummy_system(2)
        if has_pbc:
            sys.setDefaultPeriodicBoxVectors(
                openmm.Vec3(3.0, 0, 0), openmm.Vec3(0, 3.0, 0), openmm.Vec3(0, 0, 3.0)
            )

        nb = openmm.NonbondedForce()
        nb.addParticle(0.1, 0.3, 1.0)
        nb.addParticle(-0.1, 0.3, 1.0)
        nb.setNonbondedMethod(omm_method)
        if expected_cutoff_A != float("inf"):
            nb.setCutoffDistance((expected_cutoff_A / 10.0) * unit.nanometer)
        sys.addForce(nb)

        conv = from_openmm_system(sys)
        assert conv.nonbonded_method == expected_method
        if math.isinf(expected_cutoff_A):
            assert math.isinf(conv.nonbonded_cutoff)
        else:
            assert np.isclose(conv.nonbonded_cutoff, expected_cutoff_A, atol=1e-5)

    @pytest.mark.parametrize(
        ("cutoff_A", "solvent_dielectric"),
        [
            (7.5, 78.3),
            (8.5, 80.0),
            (9.0, 78.5),
            (10.0, 1.0),  # Vacuum limit
            (12.0, 20.0),
            (15.0, 109.0),  # High dielectric
        ],
    )
    def test_reaction_field_coefficients_analytical_parity(
        self, cutoff_A: float, solvent_dielectric: float
    ):
        """Reaction-field coefficients k_rf and c_rf must match OpenMM analytical formulas."""
        k_rf, c_rf = openmm_reaction_field_coefficients(cutoff_A, solvent_dielectric)

        rc = cutoff_A
        eps = solvent_dielectric
        expected_k_rf = (1.0 / (rc**3)) * ((eps - 1.0) / (2.0 * eps + 1.0))
        expected_c_rf = (1.0 / rc) * (3.0 * eps / (2.0 * eps + 1.0))

        assert np.isclose(k_rf, expected_k_rf, rtol=1e-14, atol=1e-15)
        assert np.isclose(c_rf, expected_c_rf, rtol=1e-14, atol=1e-15)

        # Vacuum edge case: eps = 1.0 gives k_rf = 0 and c_rf = 1 / rc
        if math.isclose(solvent_dielectric, 1.0):
            assert k_rf == 0.0
            assert np.isclose(c_rf, 1.0 / cutoff_A, atol=1e-14)

    def test_switching_function_and_dispersion_parameters(self):
        """Verify switching function width and dispersion correction extraction."""
        sys = _make_dummy_system(2)
        sys.setDefaultPeriodicBoxVectors(
            openmm.Vec3(4.0, 0, 0), openmm.Vec3(0, 4.0, 0), openmm.Vec3(0, 0, 4.0)
        )

        nb = openmm.NonbondedForce()
        nb.addParticle(0.0, 0.3, 1.0)
        nb.addParticle(0.0, 0.3, 1.0)
        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(1.2 * unit.nanometer)  # 12.0 Å
        nb.setUseSwitchingFunction(True)
        nb.setSwitchingDistance(1.0 * unit.nanometer)  # 10.0 Å
        nb.setUseDispersionCorrection(True)
        sys.addForce(nb)

        conv = from_openmm_system(sys)
        assert conv.use_dispersion_correction is True
        assert np.isclose(conv.lj_switch_width, 2.0, atol=1e-5)  # 12 - 10 = 2 Å

        # Now test with switching function disabled
        nb.setUseSwitchingFunction(False)
        nb.setUseDispersionCorrection(False)
        conv2 = from_openmm_system(sys)
        assert conv2.use_dispersion_correction is False
        assert conv2.lj_switch_width == 0.0


# =============================================================================
# 2. Custom Forces Routing
# =============================================================================


class TestCustomForcesRouting:
    """Adversarial tests for custom force modes, pattern matching vs Lepton, and position restraints."""

    def test_custom_force_mode_auto_routes_recognized_patterns(self):
        """In 'auto' mode, recognized custom potentials fold into native bundle arrays."""
        sys = _make_dummy_system(4)

        # Custom harmonic bond
        cbf = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
        cbf.addPerBondParameter("k")
        cbf.addPerBondParameter("r0")
        cbf.addBond(0, 1, [250000.0, 0.15])
        sys.addForce(cbf)

        # Custom harmonic angle
        caf = openmm.CustomAngleForce("0.5 * k * (theta - theta0)^2")
        caf.addPerAngleParameter("k")
        caf.addPerAngleParameter("theta0")
        caf.addAngle(0, 1, 2, [400.0, 1.9])
        sys.addForce(caf)

        # Custom periodic torsion
        ctf = openmm.CustomTorsionForce("k * (1 + cos(periodicity * theta - phase))")
        ctf.addPerTorsionParameter("k")
        ctf.addPerTorsionParameter("periodicity")
        ctf.addPerTorsionParameter("phase")
        ctf.addTorsion(0, 1, 2, 3, [15.0, 3.0, 0.0])
        sys.addForce(ctf)

        conv = from_openmm_system(sys, config=ProlixOpenMMConfig(custom_force_mode="auto"))
        bundle = conv.bundle

        # All 3 folded into bundle
        assert int(bundle.n_bonds) == 1
        assert np.isclose(float(bundle.bond_params[0, 0]), 1.5, atol=1e-5)
        assert np.isclose(float(bundle.bond_params[0, 1]), 250000.0 * BOND_K_OMM_TO_AKMA, atol=1e-4)

        assert int(bundle.n_angles) == 1
        assert np.isclose(float(bundle.angle_params[0, 0]), 1.9, atol=1e-5)
        assert np.isclose(float(bundle.angle_params[0, 1]), 400.0 * ANGLE_K_OMM_TO_AKMA, atol=1e-4)

        assert int(bundle.n_dihedrals) == 1
        assert np.isclose(float(bundle.dihedral_params[0, 0]), 3.0, atol=1e-5)
        assert np.isclose(float(bundle.dihedral_params[0, 2]), 15.0 * TORSION_K_OMM_TO_AKMA, atol=1e-4)

        # None left in custom_forces
        assert len(conv.custom_forces) == 0

    def test_custom_force_mode_charmm_improper(self):
        """In 'auto' mode, CHARMM improper harmonic potential folds into improper bundle arrays."""
        sys = _make_dummy_system(4)

        # CHARMM improper: 0.5 * k * (theta - theta0)^2
        cif = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
        cif.addPerTorsionParameter("k")
        cif.addPerTorsionParameter("theta0")
        cif.addTorsion(0, 1, 2, 3, [200.0, 0.0])
        sys.addForce(cif)

        conv = from_openmm_system(sys, config=ProlixOpenMMConfig(custom_force_mode="auto"))
        bundle = conv.bundle

        assert int(bundle.n_impropers) == 1
        # In pattern_matcher.py, improper_params are [k_akma, theta0_akma, 0.0]
        # where k_akma = mult * (k_val * ANGLE_K_OMM_TO_AKMA) = 0.5 * 200 / 4.184
        assert np.isclose(float(bundle.improper_params[0, 0]), 0.5 * 200.0 * ANGLE_K_OMM_TO_AKMA, atol=1e-4)
        assert np.isclose(float(bundle.improper_params[0, 1]), 0.0, atol=1e-5)
        assert len(conv.custom_forces) == 0

    def test_custom_force_mode_lepton_only_never_folds(self):
        """In 'lepton_only' mode, recognized custom potentials compile to Lepton and stay out of bundle."""
        sys = _make_dummy_system(2)

        cbf = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
        cbf.addPerBondParameter("k")
        cbf.addPerBondParameter("r0")
        cbf.addBond(0, 1, [300000.0, 0.14])
        sys.addForce(cbf)

        conv = from_openmm_system(sys, config=ProlixOpenMMConfig(custom_force_mode="lepton_only"))

        # Bundle must NOT receive the bond
        assert int(conv.bundle.n_bonds) == 0
        # Compiled to custom_forces
        assert len(conv.custom_forces) == 1
        assert conv.custom_forces[0].name == "CustomBondForce"

    def test_custom_force_mode_pattern_only_unmatched_raises(self):
        """In 'pattern_only' mode, non-standard potentials raise UnsupportedOpenMMForceError."""
        sys = _make_dummy_system(2)

        # Morse potential: D * (1 - exp(-a*(r - r0)))^2
        cbf = openmm.CustomBondForce("D * (1 - exp(-a * (r - r0)))^2")
        cbf.addPerBondParameter("D")
        cbf.addPerBondParameter("a")
        cbf.addPerBondParameter("r0")
        cbf.addBond(0, 1, [350.0, 20.0, 0.15])
        sys.addForce(cbf)

        cfg = ProlixOpenMMConfig(custom_force_mode="pattern_only")
        with pytest.raises(UnsupportedOpenMMForceError) as exc_info:
            from_openmm_system(sys, config=cfg)

        assert "could not be matched to a native pattern" in str(exc_info.value)

    def test_custom_force_nondefault_force_group_isolated_to_lepton(self):
        """Recognized custom potential with non-default forceGroup must NOT fold into bundle."""
        sys = _make_dummy_system(3)

        # Standard bond in forceGroup 0
        b = openmm.HarmonicBondForce()
        b.addBond(0, 1, 0.15, 200000.0)
        b.setForceGroup(0)
        sys.addForce(b)

        # Custom bond in forceGroup 5 (isolated component)
        cbf = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
        cbf.addPerBondParameter("k")
        cbf.addPerBondParameter("r0")
        cbf.addBond(1, 2, [300000.0, 0.14])
        cbf.setForceGroup(5)
        sys.addForce(cbf)

        conv = from_openmm_system(sys, config=ProlixOpenMMConfig(custom_force_mode="auto"))

        # Only the standard bond is in bundle.n_bonds
        assert int(conv.bundle.n_bonds) == 1
        # The custom bond in forceGroup 5 is compiled into custom_forces to preserve decomposed energy
        assert len(conv.custom_forces) == 1
        assert conv.custom_forces[0].force_group == 5

    def test_position_restraints_custom_expressions_and_coordinates(self):
        """Verify position restraint extraction with custom coordinate names and spring constants."""
        sys = _make_dummy_system(3)

        # Multiple particles with different reference coordinates
        cef = openmm.CustomExternalForce("0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
        cef.addPerParticleParameter("k")
        cef.addPerParticleParameter("x0")
        cef.addPerParticleParameter("y0")
        cef.addPerParticleParameter("z0")
        cef.addParticle(0, [1000.0, 0.1, 0.2, 0.3])  # nm
        cef.addParticle(2, [2500.0, -0.4, 0.5, 0.0])  # nm
        cef.setForceGroup(12)
        sys.addForce(cef)

        conv = from_openmm_system(sys)
        assert len(conv.position_restraints) == 1
        pr = conv.position_restraints[0]

        assert pr.force_group == 12
        assert list(pr.indices) == [0, 2]

        # Coordinates converted nm -> Å
        assert np.allclose(pr.x0, [1.0, -4.0], atol=1e-5)
        assert np.allclose(pr.y0, [2.0, 5.0], atol=1e-5)
        assert np.allclose(pr.z0, [3.0, 0.0], atol=1e-5)

        # Springs in AKMA: 0.5 * k / 418.4
        expected_k = np.array([1000.0, 2500.0]) * 0.5 * BOND_K_OMM_TO_AKMA
        assert np.allclose(pr.k_x, expected_k, atol=1e-4)
        assert np.allclose(pr.k_y, expected_k, atol=1e-4)
        assert np.allclose(pr.k_z, expected_k, atol=1e-4)

    def test_position_restraints_periodic_distance(self):
        """Verify that position restraints using periodicdistance() extract into PositionRestraints."""
        sys = _make_dummy_system(2)
        sys.setDefaultPeriodicBoxVectors(
            openmm.Vec3(4.0, 0, 0), openmm.Vec3(0, 4.0, 0), openmm.Vec3(0, 0, 4.0)
        )

        cef = openmm.CustomExternalForce("0.5 * k * periodicdistance(x, y, z, x0, y0, z0)^2")
        cef.addPerParticleParameter("k")
        cef.addPerParticleParameter("x0")
        cef.addPerParticleParameter("y0")
        cef.addPerParticleParameter("z0")
        cef.addParticle(0, [1200.0, 0.4, 0.6, 0.8])
        sys.addForce(cef)

        conv = from_openmm_system(sys)
        assert len(conv.position_restraints) == 1
        pr = conv.position_restraints[0]
        assert np.isclose(float(pr.x0[0]), 4.0, atol=1e-5)
        assert np.isclose(float(pr.y0[0]), 6.0, atol=1e-5)
        assert np.isclose(float(pr.z0[0]), 8.0, atol=1e-5)
        assert np.isclose(float(pr.k_x[0]), 0.5 * 1200.0 * BOND_K_OMM_TO_AKMA, atol=1e-4)

    def test_position_restraints_anisotropic_lepton_fallback(self):
        """Anisotropic position restraints (kx != ky != kz) fall back to Lepton compilation."""
        sys = _make_dummy_system(2)
        cef = openmm.CustomExternalForce(
            "0.5 * kx * (x - x0)^2 + 0.5 * ky * (y - y0)^2 + 0.5 * kz * (z - z0)^2"
        )
        cef.addPerParticleParameter("kx")
        cef.addPerParticleParameter("ky")
        cef.addPerParticleParameter("kz")
        cef.addPerParticleParameter("x0")
        cef.addPerParticleParameter("y0")
        cef.addPerParticleParameter("z0")
        cef.addParticle(0, [1000.0, 2000.0, 3000.0, 0.1, 0.2, 0.3])
        sys.addForce(cef)

        conv = from_openmm_system(sys)
        assert len(conv.position_restraints) == 0
        assert len(conv.custom_forces) == 1
        assert conv.custom_forces[0].name == "CustomExternalForce"

    def test_custom_force_dynamic_parameters_fallback_to_lepton(self):
        """Custom forces with global parameters or energy parameter derivatives fall back to Lepton."""
        sys = _make_dummy_system(3)

        # Force 1 with dynamic global parameter
        cbf1 = openmm.CustomBondForce("lambda_scale * 0.5 * k * (r - r0)^2")
        cbf1.addGlobalParameter("lambda_scale", 1.0)
        cbf1.addPerBondParameter("k")
        cbf1.addPerBondParameter("r0")
        cbf1.addBond(0, 1, [250000.0, 0.15])
        sys.addForce(cbf1)

        # Force 2 with energy parameter derivative
        cbf2 = openmm.CustomBondForce("alpha * 0.5 * k * (r - r0)^2")
        cbf2.addGlobalParameter("alpha", 1.0)
        cbf2.addEnergyParameterDerivative("alpha")
        cbf2.addPerBondParameter("k")
        cbf2.addPerBondParameter("r0")
        cbf2.addBond(1, 2, [300000.0, 0.14])
        sys.addForce(cbf2)

        conv = from_openmm_system(sys, config=ProlixOpenMMConfig(custom_force_mode="auto"))

        # Both remain dynamic in custom_forces, not folded into bundle
        assert int(conv.bundle.n_bonds) == 0
        assert len(conv.custom_forces) == 2
        assert "lambda_scale" in conv.global_parameters
        assert "alpha" in conv.global_parameters

    def test_custom_compound_bond_combined_angle_ub_and_pure_ub(self):
        """CustomCompoundBondForce with combined Angle+UB and pure UB both fold into bundle."""
        sys = _make_dummy_system(4)

        # 1. Combined Angle + UB
        ccbf1 = openmm.CustomCompoundBondForce(
            3,
            "0.5 * k_ang * (angle(p1, p2, p3) - theta0)^2 + 0.5 * k_ub * (distance(p1, p3) - r_ub)^2",
        )
        ccbf1.addPerBondParameter("k_ang")
        ccbf1.addPerBondParameter("theta0")
        ccbf1.addPerBondParameter("k_ub")
        ccbf1.addPerBondParameter("r_ub")
        ccbf1.addBond([0, 1, 2], [450.0, 1.9, 150000.0, 0.25])
        sys.addForce(ccbf1)

        # 2. Pure UB
        ccbf2 = openmm.CustomCompoundBondForce(
            3, "0.5 * k_ub * (distance(p1, p3) - r_ub)^2"
        )
        ccbf2.addPerBondParameter("k_ub")
        ccbf2.addPerBondParameter("r_ub")
        ccbf2.addBond([1, 2, 3], [120000.0, 0.24])
        sys.addForce(ccbf2)

        conv = from_openmm_system(sys, config=ProlixOpenMMConfig(custom_force_mode="auto"))
        bundle = conv.bundle

        assert int(bundle.n_angles) == 1
        assert int(bundle.n_urey_bradley) == 2
        assert len(conv.custom_forces) == 0


# =============================================================================
# 3. Force Groups & Global Parameters
# =============================================================================


class TestForceGroupsAndGlobalParameters:
    """Adversarial tests for force group mapping (0..31) and global parameter extraction."""

    def test_force_groups_8_plus_distinct_forces(self):
        """Construct system with 9 distinct forces across different forceGroup indices (0..31)."""
        sys = _make_dummy_system(6)
        sys.setDefaultPeriodicBoxVectors(
            openmm.Vec3(4.0, 0, 0), openmm.Vec3(0, 4.0, 0), openmm.Vec3(0, 0, 4.0)
        )

        # 1. HarmonicBondForce (fg 1)
        f1 = openmm.HarmonicBondForce()
        f1.addBond(0, 1, 0.15, 200000.0)
        f1.setForceGroup(1)
        sys.addForce(f1)

        # 2. HarmonicAngleForce (fg 3)
        f2 = openmm.HarmonicAngleForce()
        f2.addAngle(0, 1, 2, 1.9, 400.0)
        f2.setForceGroup(3)
        sys.addForce(f2)

        # 3. PeriodicTorsionForce (fg 6)
        f3 = openmm.PeriodicTorsionForce()
        f3.addTorsion(0, 1, 2, 3, 3, 0.0, 10.0)
        f3.setForceGroup(6)
        sys.addForce(f3)

        # 4. RBTorsionForce (fg 10)
        f4 = openmm.RBTorsionForce()
        f4.addTorsion(0, 1, 2, 3, 10.0, 2.0, 1.0, 0.0, 0.0, 0.0)
        f4.setForceGroup(10)
        sys.addForce(f4)

        # 5. NonbondedForce (fg 14)
        f5 = openmm.NonbondedForce()
        for _ in range(6):
            f5.addParticle(0.1, 0.3, 1.0)
        f5.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        f5.setForceGroup(14)
        sys.addForce(f5)

        # 6. GBSAOBCForce (fg 19)
        f6 = openmm.GBSAOBCForce()
        for _ in range(6):
            f6.addParticle(0.1, 0.15, 0.8)
        f6.setForceGroup(19)
        sys.addForce(f6)

        # 7. CustomBondForce (fg 22)
        f7 = openmm.CustomBondForce("k_glob * (r - r0)^4")
        f7.addGlobalParameter("k_glob", 5500.0)
        f7.addPerBondParameter("r0")
        f7.addBond(1, 2, [0.15])
        f7.setForceGroup(22)
        sys.addForce(f7)

        # 8. CustomExternalForce (fg 27)
        f8 = openmm.CustomExternalForce("0.5 * k_ext * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
        f8.addGlobalParameter("k_ext", 1250.0)
        f8.addPerParticleParameter("x0")
        f8.addPerParticleParameter("y0")
        f8.addPerParticleParameter("z0")
        f8.addParticle(0, [0.0, 0.0, 0.0])
        f8.setForceGroup(27)
        sys.addForce(f8)

        # 9. MonteCarloBarostat (fg 31)
        f9 = openmm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25)
        f9.setForceGroup(31)
        sys.addForce(f9)

        conv = from_openmm_system(sys)

        # Verify force_groups mapping
        expected_mappings = {
            1: "HarmonicBondForce",
            3: "HarmonicAngleForce",
            6: "PeriodicTorsionForce",
            10: "RBTorsionForce",
            14: "NonbondedForce",
            19: "GBSAOBCForce",
            22: "CustomBondForce",
            27: "CustomExternalForce",
            31: "MonteCarloBarostat",
        }

        for fg, expected_name in expected_mappings.items():
            assert fg in conv.force_groups, f"Missing forceGroup {fg}"
            assert expected_name in conv.force_groups[fg]

        # Verify global parameter extraction
        assert "k_glob" in conv.global_parameters
        assert np.isclose(conv.global_parameters["k_glob"], 5500.0)
        assert "k_ext" in conv.global_parameters
        assert np.isclose(conv.global_parameters["k_ext"], 1250.0)

        # Verify default_parameters matches global_parameters
        assert conv.default_parameters == conv.global_parameters

    def test_multiple_forces_same_force_group(self):
        """Multiple forces assigned to the same forceGroup must all be cataloged under that group."""
        sys = _make_dummy_system(4)

        b = openmm.HarmonicBondForce()
        b.addBond(0, 1, 0.15, 200000.0)
        b.setForceGroup(7)
        sys.addForce(b)

        a = openmm.HarmonicAngleForce()
        a.addAngle(0, 1, 2, 1.9, 400.0)
        a.setForceGroup(7)
        sys.addForce(a)

        conv = from_openmm_system(sys)
        assert 7 in conv.force_groups
        assert "HarmonicBondForce" in conv.force_groups[7]
        assert "HarmonicAngleForce" in conv.force_groups[7]

    def test_force_groups_full_32_group_sweep(self):
        """Verify that all 32 OpenMM force groups (0..31) are mapped without omission."""
        sys = _make_dummy_system(4)

        for fg in range(32):
            b = openmm.HarmonicBondForce()
            b.addBond(0, 1, 0.15, 1000.0)
            b.setForceGroup(fg)
            sys.addForce(b)

        conv = from_openmm_system(sys)
        assert len(conv.force_groups) == 32
        for fg in range(32):
            assert fg in conv.force_groups
            assert "HarmonicBondForce" in conv.force_groups[fg]

    def test_global_parameters_extreme_and_negative_values(self):
        """Global parameters with negative, zero, and extreme values must be accurately captured."""
        sys = _make_dummy_system(2)

        cbf = openmm.CustomBondForce("k_pos * (r - r0)^2 + k_neg * (r - r0) + k_zero")
        cbf.addGlobalParameter("k_pos", 1.5e6)
        cbf.addGlobalParameter("k_neg", -42.8)
        cbf.addGlobalParameter("k_zero", 0.0)
        cbf.addGlobalParameter("k_tiny", 1.2e-15)
        cbf.addPerBondParameter("r0")
        cbf.addBond(0, 1, [0.15])
        sys.addForce(cbf)

        conv = from_openmm_system(sys)
        gp = conv.global_parameters

        assert np.isclose(gp["k_pos"], 1.5e6)
        assert np.isclose(gp["k_neg"], -42.8)
        assert gp["k_zero"] == 0.0
        assert np.isclose(gp["k_tiny"], 1.2e-15)


# =============================================================================
# 4. Fail-Fast Catalogue
# =============================================================================


class TestFailFastCatalogue:
    """Adversarial verification of immediate rejection for unsupported forces and features."""

    def _make_unsupported_force(self, name: str) -> Any:
        cls = getattr(openmm, name, None)
        if cls is None:
            cls = type(name, (), {"getForceGroup": lambda self: 0})
            return cls()
        if name == "CustomCentroidBondForce":
            return cls(2, "0.5*distance(g1, g2)^2")
        if name == "CustomCVForce":
            return cls("0.5*r^2")
        if name == "CustomHbondForce":
            return cls("0.5*r^2")
        if name == "CustomManyParticleForce":
            return cls(2, "0.5*r^2")
        if name == "CustomVolumeForce":
            return cls("0.5*V^2")
        if name == "RPMDMonteCarloBarostat":
            return cls(1.0 * unit.bar)
        if name == "MonteCarloAnisotropicBarostat":
            return cls((1.0, 1.0, 1.0) * unit.bar, 300 * unit.kelvin)
        if name == "MonteCarloFlexibleBarostat":
            return cls(1.0 * unit.bar, 300 * unit.kelvin)
        if name == "MonteCarloMembraneBarostat":
            return cls(
                1.0 * unit.bar,
                0.0 * unit.bar * unit.nanometer,
                300 * unit.kelvin,
                openmm.MonteCarloMembraneBarostat.XYIsotropic,
                openmm.MonteCarloMembraneBarostat.ZFree,
            )
        if name == "ATMForce":
            return cls("0.5*r^2")
        if name == "PythonForce":
            class DummyCalc:
                def computeForce(self, pos, vel, forces):
                    return 0.0
            return cls(DummyCalc())
        if name == "OrientationRestraintForce":
            return cls(1.0, [openmm.Vec3(0, 0, 0)])
        if name == "RMSDForce":
            return cls([openmm.Vec3(0, 0, 0)])

        return cls()

    @pytest.mark.parametrize("force_name", list(UNSUPPORTED_OPENMM_FORCES.keys()))
    def test_all_24_unsupported_forces_fail_fast(self, force_name: str, monkeypatch):
        """Every force in UNSUPPORTED_OPENMM_FORCES must immediately raise UnsupportedOpenMMForceError."""
        f = self._make_unsupported_force(force_name)

        sys = _make_dummy_system(2)
        if isinstance(f, openmm.Force):
            sys.addForce(f)
        else:
            sys.addForce(openmm.HarmonicBondForce())
            monkeypatch.setattr(sys, "getForce", lambda idx: f)

        with pytest.raises(UnsupportedOpenMMForceError) as exc_info:
            from_openmm_system(sys)

        err = exc_info.value
        assert err.force_name == force_name
        assert UNSUPPORTED_OPENMM_FORCES[force_name] in str(err)

    def test_fail_fast_nonbonded_ljpme(self):
        """NonbondedForce with method LJPME (method 5) must raise UnsupportedOpenMMForceError."""
        sys = _make_dummy_system(2)
        nb = openmm.NonbondedForce()
        nb.addParticle(0.0, 0.3, 1.0)
        nb.addParticle(0.0, 0.3, 1.0)
        nb.setNonbondedMethod(openmm.NonbondedForce.LJPME)
        sys.addForce(nb)

        with pytest.raises(UnsupportedOpenMMForceError) as exc_info:
            from_openmm_system(sys)

        assert "LJPME" in str(exc_info.value)

    def test_fail_fast_arbitrary_unrecognized_force(self, monkeypatch):
        """Any unrecognized third-party or mock OpenMM Force subclass must immediately raise."""
        class NovelExperimentalForce:
            def getForceGroup(self):
                return 0

        mock_force = NovelExperimentalForce()
        sys = _make_dummy_system(2)
        # Add a placeholder force and monkeypatch getForce to return mock_force
        sys.addForce(openmm.HarmonicBondForce())
        monkeypatch.setattr(sys, "getForce", lambda idx: mock_force)

        with pytest.raises(UnsupportedOpenMMForceError) as exc_info:
            from_openmm_system(sys)

        assert "NovelExperimentalForce" in str(exc_info.value)
        assert "not supported" in str(exc_info.value)
