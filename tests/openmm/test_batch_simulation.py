"""Tests for ProlixBatchSimulation and Pallas v3.5 OpenMM physics remediation.

Covers:
1. `ProlixBatchSimulation` (`BatchSimulation`) homogeneous multi-replica stepping
   with `positions_list` (matching `bench_v35_release_ladder.py` signature) and
   `EnsemblePlan` compilation sharing.
2. Congeneric parameter-differentiated OpenMM systems (same topology, distinct
   charges/LJ parameters) sharing a single compiled `EnsemblePlan` (`MK-64`).
3. Cross-size ragged OpenMM systems aligned to a common `MolecularShapeSpec`
   bucket (`len({b.shape_spec for b in bsim.bundles}) == 1`).
4. Differentiable `value_and_grad_positions` and Lepton `compute_parameter_derivatives`.
5. Physics audit regressions (`BUG-OMM1..BUG-OMM5`, `BUG-CMAP1`, `BUG-CMAP2`,
   `precision="mixed"` accumulation).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import openmm
from openmm import unit

from prolix.openmm import (
    BatchSimulation,
    ProlixBatchSimulation,
    ProlixContext,
    ProlixOpenMMConfig,
    from_openmm_system,
)
from prolix.openmm.units import AKMA_PRESSURE_PER_BAR, to_akma
from prolix.openmm_xml_importer import parse_openmm_system_xml


def _make_periodic_lj_dimer_system(
    charge: float = 0.25,
    box_nm: float = 2.5,
) -> tuple[openmm.System, np.ndarray]:
    """Create a minimal periodic 4-atom bonded + PME system in OpenMM."""
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    )
    for _ in range(4):
        sys.addParticle(12.0)

    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.15, 250000.0)
    bond.addBond(2, 3, 0.15, 250000.0)
    sys.addForce(bond)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9)
    nb.setEwaldErrorTolerance(5e-4)
    charges = [charge, -charge, charge, -charge]
    for q in charges:
        nb.addParticle(q, 0.34, 0.45)
    nb.createExceptionsFromBonds([(0, 1), (2, 3)], 1.0, 1.0)
    sys.addForce(nb)

    pos_nm = np.array(
        [
            [0.50, 0.50, 0.50],
            [0.65, 0.50, 0.50],
            [1.30, 1.30, 1.30],
            [1.45, 1.30, 1.30],
        ],
        dtype=np.float64,
    )
    return sys, pos_nm


def _make_vacuum_chain_system(n_atoms: int, charge_scale: float = 1.0) -> tuple[openmm.System, np.ndarray]:
    """Create a vacuum harmonic chain with GBSA implicit solvent."""
    sys = openmm.System()
    for _ in range(n_atoms):
        sys.addParticle(14.0)

    bond = openmm.HarmonicBondForce()
    for i in range(n_atoms - 1):
        bond.addBond(i, i + 1, 0.152, 220000.0)
    sys.addForce(bond)

    angle = openmm.HarmonicAngleForce()
    for i in range(n_atoms - 2):
        angle.addAngle(i, i + 1, i + 2, 1.95, 400.0)
    sys.addForce(angle)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for i in range(n_atoms):
        q = (0.15 if i % 2 == 0 else -0.15) * charge_scale
        nb.addParticle(q, 0.33, 0.40)
    nb.createExceptionsFromBonds([(i, i + 1) for i in range(n_atoms - 1)], 1.0, 1.0)
    sys.addForce(nb)

    gb = openmm.GBSAOBCForce()
    gb.setSolventDielectric(78.5)
    gb.setSoluteDielectric(1.0)
    for _ in range(n_atoms):
        gb.addParticle(0.0, 0.17, 0.85)
    for i in range(n_atoms):
        q, _, _ = nb.getParticleParameters(i)
        gb.setParticleParameters(i, q, 0.17, 0.85)
    sys.addForce(gb)

    xs = np.arange(n_atoms, dtype=np.float64) * 0.152
    ys = 0.03 * np.sin(np.arange(n_atoms, dtype=np.float64))
    zs = 0.02 * np.cos(np.arange(n_atoms, dtype=np.float64))
    pos_nm = np.stack([xs, ys, zs], axis=-1)
    return sys, pos_nm


class TestProlixBatchSimulation:
    """Verify multi-replica, congeneric, and ragged OpenMM batch execution."""

    def test_alias_export(self):
        assert BatchSimulation is ProlixBatchSimulation

    def test_homogeneous_pme_replicas_positions_list_contract(self):
        """Matches exact `bench_v35_release_ladder.py` signature and stateful evolution."""
        sys, pos_nm = _make_periodic_lj_dimer_system()
        pos_q = pos_nm * unit.nanometers
        b_count = 4
        integs = [
            openmm.LangevinMiddleIntegrator(
                300.0 * unit.kelvin, 10.0 / unit.picoseconds, 1.0 * unit.femtoseconds
            )
            for _ in range(b_count)
        ]
        cfg = ProlixOpenMMConfig(precision="single", nl_update_every=10)

        bsim = ProlixBatchSimulation(
            [sys for _ in range(b_count)],
            integs,
            positions_list=[pos_q for _ in range(b_count)],
            config=cfg,
        )
        assert bsim.batch_size == b_count
        assert bsim.plan is not None
        # Shared topology array identity (MK-64)
        assert bsim.bundles[0].bond_idx is bsim.bundles[1].bond_idx

        bsim.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
        bsim.step(3)
        states_1 = bsim.getStates(getPositions=True, getEnergy=True)
        assert len(states_1) == b_count
        assert bsim.currentStep == 3

        p1 = states_1[0].getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        assert p1.shape == (4, 3)
        assert np.all(np.isfinite(p1))
        assert np.isfinite(
            states_1[0].getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        )

        # Second step call must advance state from p1 rather than resetting to pos_nm
        bsim.step(3)
        states_2 = bsim.getStates(getPositions=True)
        p2 = states_2[0].getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        assert bsim.currentStep == 6
        assert not np.allclose(p1, p2)

    def test_congeneric_parameter_differentiated_systems(self):
        """Distinct per-ligand charges share a single shape_spec and EnsemblePlan."""
        systems = []
        positions = []
        for q_scale in (0.5, 1.0, 1.5):
            s, p = _make_vacuum_chain_system(10, charge_scale=q_scale)
            systems.append(s)
            positions.append(p)

        bsim = ProlixBatchSimulation(
            systems,
            openmm.LangevinMiddleIntegrator(300.0, 5.0, 0.001),
            positions=positions,
        )
        assert len({b.shape_spec for b in bsim.bundles}) == 1

        e_kj, grads = bsim.value_and_grad_positions()
        assert e_kj.shape == (3,)
        assert isinstance(grads, jnp.ndarray)
        assert grads.shape == (3, 10, 3)
        # Distinct charge scales produce distinct potential energies
        assert not np.isclose(float(e_kj[0]), float(e_kj[2]))

    def test_ragged_cross_size_systems_align_to_single_bucket(self):
        """Cross-bucket systems (12 vs 80 atoms) pad to one shared MolecularShapeSpec."""
        s_small, p_small = _make_vacuum_chain_system(12)
        s_large, p_large = _make_vacuum_chain_system(80)

        bsim = ProlixBatchSimulation(
            [s_small, s_large],
            openmm.LangevinMiddleIntegrator(300.0, 5.0, 0.001),
            positions_list=[p_small, p_large],
        )
        assert len({b.shape_spec for b in bsim.bundles}) == 1

        bsim.step(2)
        states = bsim.getStates(getPositions=True, getForces=True)
        assert states[0].getPositions(asNumpy=True).shape == (12, 3)
        assert states[1].getPositions(asNumpy=True).shape == (80, 3)
        assert states[0].getForces(asNumpy=True).shape == (12, 3)
        assert states[1].getForces(asNumpy=True).shape == (80, 3)

        e_kj, grads = bsim.value_and_grad_positions()
        assert e_kj.shape == (2,)
        assert isinstance(grads, list)
        assert grads[0].shape == (12, 3)
        assert grads[1].shape == (80, 3)

    def test_batch_parameter_derivatives_lepton(self):
        """`compute_parameter_derivatives` differentiates OpenMM global parameters across replicas."""
        systems = []
        positions = []
        for k_scale in (1.0, 2.0):
            sys = openmm.System()
            sys.addParticle(12.0)
            sys.addParticle(12.0)
            cf = openmm.CustomBondForce("lambda_b * 0.5 * k * (r - r0)^2")
            cf.addGlobalParameter("lambda_b", k_scale)
            cf.addPerBondParameter("k")
            cf.addPerBondParameter("r0")
            cf.addBond(0, 1, [1000.0, 0.15])
            cf.addEnergyParameterDerivative("lambda_b")
            sys.addForce(cf)
            systems.append(sys)
            positions.append(np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=np.float64))

        bsim = ProlixBatchSimulation(
            systems,
            openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
            positions_list=positions,
        )
        derivs = bsim.compute_parameter_derivatives()
        assert len(derivs) == 2
        # 0.5 * 1000 * (0.25 - 0.15)^2 = 5.0 kJ/mol
        assert abs(derivs[0]["lambda_b"] - 5.0) < 1e-3
        assert abs(derivs[1]["lambda_b"] - 5.0) < 1e-3


class TestPhysicsAuditRegressions:
    """Regression suite for BUG-OMM1..BUG-OMM5, BUG-CMAP1, BUG-CMAP2, and mixed precision."""

    def test_bug_omm1_rigid_water_equipartition_temperature(self):
        """`setVelocitiesToTemperature(300 K)` on rigid TIP3P waters yields ~300 K, not ~200 K."""
        n_waters = 64
        sys = openmm.System()
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        pos = []
        r_oh = 0.09572
        theta = np.deg2rad(104.52)
        r_hh = 2.0 * r_oh * np.sin(0.5 * theta)
        for w in range(n_waters):
            base = 3 * w
            sys.addParticle(15.9994)
            sys.addParticle(1.00794)
            sys.addParticle(1.00794)
            sys.addConstraint(base, base + 1, r_oh)
            sys.addConstraint(base, base + 2, r_oh)
            sys.addConstraint(base + 1, base + 2, r_hh)
            nb.addParticle(-0.834, 0.31507, 0.63597)
            nb.addParticle(0.417, 0.04, 0.0)
            nb.addParticle(0.417, 0.04, 0.0)
            nb.addException(base, base + 1, 0.0, 0.1, 0.0)
            nb.addException(base, base + 2, 0.0, 0.1, 0.0)
            nb.addException(base + 1, base + 2, 0.0, 0.1, 0.0)
            ox, oy, oz = (w % 4) * 0.4, ((w // 4) % 4) * 0.4, (w // 16) * 0.4
            pos.append([ox, oy, oz])
            pos.append([ox + r_oh, oy, oz])
            pos.append([ox + r_oh * np.cos(theta), oy + r_oh * np.sin(theta), oz])
        sys.addForce(nb)

        ctx = ProlixContext(
            sys,
            openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
            config=ProlixOpenMMConfig(precision="double"),
        )
        ctx.setPositions(np.array(pos, dtype=np.float64))
        ctx.setVelocitiesToTemperature(300.0, 1234)
        ke_kj = ctx.getState(getEnergy=True).getKineticEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        # Active DOF = 3*(3*64) - 3*64 - 3 = 381 -> expected KE = 0.5 * 381 * R * 300
        r_kj = 8.314462618e-3
        n_dof = 6 * n_waters - 3
        temp_inst = 2.0 * ke_kj / (n_dof * r_kj)
        assert 255.0 < temp_inst < 345.0, f"Expected ~300 K after SETTLE RATTLE, got {temp_inst:.1f} K"

    def test_bug_omm2_cold_and_verlet_ke_not_corrupted_by_forces(self):
        """Cold context KE is 0.0, and Velocity-Verlet KE equals exact 0.5*m*v^2."""
        sys = openmm.System()
        sys.addParticle(10.0)
        sys.addParticle(10.0)
        bond = openmm.HarmonicBondForce()
        bond.addBond(0, 1, 0.10, 500000.0)
        sys.addForce(bond)

        # Stretched bond produces huge force, yet cold v=0 must have KE == 0.0
        ctx_lf = ProlixContext(sys, openmm.VerletIntegrator(0.002))
        ctx_lf.setPositions(np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]))
        ke_cold = ctx_lf.getState(getEnergy=True).getKineticEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        assert ke_cold == 0.0

        # Velocity-Verlet (`integrator_scheme="verlet"`) stores synchronous integer-step velocities
        ctx_vv = ProlixContext(
            sys,
            openmm.VerletIntegrator(0.002),
            config=ProlixOpenMMConfig(integrator_scheme="verlet", precision="double"),
        )
        ctx_vv.setPositions(np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]))
        v_set = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])  # nm/ps
        ctx_vv.setVelocities(v_set)
        ke_vv = ctx_vv.getState(getEnergy=True).getKineticEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        # 2 particles * 0.5 * 10 amu * (0.5 nm/ps)^2 = 2.5 kJ/mol
        assert abs(ke_vv - 2.5) < 1e-8

    def test_bug_omm4_wrapped_bond_mc_barostat_and_adaptive_scale(self):
        """MC barostat unwraps split molecules across PBC and maintains finite adaptive scale."""
        sys, _ = _make_periodic_lj_dimer_system(box_nm=2.5)
        sys.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 1))

        # Split bond (0, 1) across the x=0/2.5 nm boundary: 2.45 nm and 0.05 nm -> MIC = 0.15 nm
        pos_wrapped = np.array(
            [
                [2.45, 0.50, 0.50],
                [0.05, 0.50, 0.50],
                [1.20, 1.20, 1.20],
                [1.35, 1.20, 1.20],
            ],
            dtype=np.float64,
        )
        ctx = ProlixContext(
            sys,
            openmm.LangevinMiddleIntegrator(300.0, 5.0, 0.001),
            config=ProlixOpenMMConfig(precision="double"),
        )
        ctx.setPositions(pos_wrapped)
        assert abs(ctx._mc_volume_scale - 0.01) < 1e-12

        pe0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        ctx.step(12)
        st = ctx.getState(getPositions=True, getEnergy=True)
        pe1 = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isfinite(pe1)
        # If bond (0, 1) had been scaled without PBC unwrapping, bond energy would explode >1e5 kJ/mol
        assert abs(pe1 - pe0) < 500.0

    def test_bug_omm5_atm_pressure_and_gbsa_zero_surface_area_energy(self):
        """1 atm == 1.01325 bar and `GBSAOBCForce.setSurfaceAreaEnergy(0.0)` disables ACE nonpolar."""
        p_bar = to_akma(1.0, "bar")
        p_atm = to_akma(1.0, "atm")
        assert abs(p_bar - AKMA_PRESSURE_PER_BAR) < 1e-18
        assert abs(p_atm / p_bar - 1.01325) < 1e-12

        s_ace, pos = _make_vacuum_chain_system(6)
        s_no_ace, _ = _make_vacuum_chain_system(6)
        for i in range(s_no_ace.getNumForces()):
            f = s_no_ace.getForce(i)
            if isinstance(f, openmm.GBSAOBCForce):
                f.setSurfaceAreaEnergy(0.0)

        ref_plat = openmm.Platform.getPlatformByName("Reference")
        ref_ctx = openmm.Context(s_no_ace, openmm.VerletIntegrator(0.001), ref_plat)
        ref_ctx.setPositions(pos)
        e_ref = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )

        prx_ctx = ProlixContext(
            s_no_ace,
            openmm.VerletIntegrator(0.001),
            config=ProlixOpenMMConfig(precision="double"),
        )
        prx_ctx.setPositions(pos)
        e_prx = prx_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        assert abs(e_prx - e_ref) < 0.05

        prx_ace = ProlixContext(
            s_ace,
            openmm.VerletIntegrator(0.001),
            config=ProlixOpenMMConfig(precision="double"),
        )
        prx_ace.setPositions(pos)
        e_prx_ace = prx_ace.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        assert abs(e_prx_ace - e_prx) > 0.5

    def test_bug_cmap1_and_cmap2_asymmetric_grid_parity(self):
        """Asymmetric CMAP grid matches OpenMM Reference (XLA + Pallas `first_is_psi=False`) and XML `<Energy e=.../>`."""
        size = 12
        # Asymmetric grid where E(phi, psi) != E(psi, phi): OpenMM flat idx = phi_i + size * psi_j
        phis = np.linspace(-np.pi, np.pi, size, endpoint=False)
        psis = np.linspace(-np.pi, np.pi, size, endpoint=False)
        flat_kj = []
        for j in range(size):
            for i in range(size):
                val = 12.0 * np.sin(phis[i]) + 3.5 * np.cos(2.0 * psis[j]) + 4.0 * np.sin(phis[i] - 2.0 * psis[j])
                flat_kj.append(float(val))

        sys = openmm.System()
        for _ in range(5):
            sys.addParticle(12.0)
        cmap = openmm.CMAPTorsionForce()
        cmap.addMap(size, flat_kj)
        cmap.addTorsion(0, 0, 1, 2, 3, 1, 2, 3, 4)
        sys.addForce(cmap)

        # Pick asymmetric backbone coordinates so phi != psi
        pos = np.array(
            [
                [0.00, 0.12, 0.05],
                [0.15, 0.00, 0.00],
                [0.24, 0.13, 0.00],
                [0.39, 0.13, 0.04],
                [0.47, 0.25, -0.09],
            ],
            dtype=np.float64,
        )

        ref_ctx = openmm.Context(
            sys,
            openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("Reference"),
        )
        ref_ctx.setPositions(pos)
        ref_st = ref_ctx.getState(getEnergy=True, getForces=True)
        e_ref = ref_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        f_ref = ref_st.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )

        prx_ctx = ProlixContext(
            sys,
            openmm.VerletIntegrator(0.001),
            config=ProlixOpenMMConfig(precision="double"),
        )
        prx_ctx.setPositions(pos)
        prx_st = prx_ctx.getState(getEnergy=True, getForces=True)
        e_prx = prx_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        f_prx = prx_st.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        assert abs(e_prx - e_ref) < 1e-2
        assert np.max(np.abs(f_prx - f_ref)) < 0.2

        # Also verify BUG-CMAP2 XML importer `<Energy e="..."/>` + Fortran reshape
        xml_serialized = openmm.XmlSerializer.serialize(sys)
        parsed_xml = parse_openmm_system_xml(xml_serialized)
        assert parsed_xml.cmap_energy_grids is not None
        conv = from_openmm_system(sys)
        np.testing.assert_allclose(
            np.asarray(parsed_xml.cmap_energy_grids[0]),
            np.asarray(conv.bundle.cmap_energy_grids[0]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_mixed_precision_context_steps_cleanly(self):
        """`precision='mixed'` evaluates forces in fp32 and integrates positions/velocities cleanly."""
        sys, pos = _make_periodic_lj_dimer_system()
        ctx = ProlixContext(
            sys,
            openmm.LangevinMiddleIntegrator(300.0, 5.0, 0.001),
            config=ProlixOpenMMConfig(precision="mixed"),
        )
        ctx.setPositions(pos)
        ctx.setVelocitiesToTemperature(300.0, 7)
        ctx.step(5)
        st = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True)
        assert np.all(np.isfinite(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)))
        assert np.isfinite(st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))

    def test_bug_omm9_nocutoff_gbsa_no_9a_truncation_and_pallas_tiles(self):
        """`NoCutoff` + GBSAOBCForce keeps all-pairs Coulomb/LJ (> 9 Å) and builds `GBSATilePairList`."""
        n_atoms = 68
        sys = openmm.System()
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        gb = openmm.GBSAOBCForce()
        gb.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
        gb.setSoluteDielectric(1.0)
        gb.setSolventDielectric(78.5)
        gb.setSurfaceAreaEnergy(2.25936)

        rng = np.random.default_rng(2026)
        pos = np.zeros((n_atoms, 3), dtype=np.float64)
        for i in range(n_atoms):
            sys.addParticle(12.0)
            q = 0.35 if i < n_atoms // 2 else -0.35
            nb.addParticle(q, 0.34, 0.45)
            gb.addParticle(q, 0.17, 0.85)
            # Place cluster A near origin and cluster B ~1.5 nm (15 Å > 9 Å) away
            offset_x = 0.0 if i < n_atoms // 2 else 1.5
            pos[i] = [
                offset_x + 0.22 * (i % 6),
                0.22 * ((i // 6) % 6),
                0.22 * (i // 36) + 0.01 * rng.standard_normal(),
            ]
        sys.addForce(nb)
        sys.addForce(gb)

        ref_ctx = openmm.Context(
            sys,
            openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("Reference"),
        )
        ref_ctx.setPositions(pos)
        e_ref = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )

        ctx_pallas = ProlixContext(
            sys,
            openmm.VerletIntegrator(0.001),
            config=ProlixOpenMMConfig(
                precision="single",
                use_pallas=True,
                pallas_interpret=True,
            ),
        )
        ctx_pallas.setPositions(pos)
        assert ctx_pallas._nonbonded_engine == "gbsa_tile"
        assert ctx_pallas._neighbor_idx is None
        assert ctx_pallas._gbsa_tile_list is not None
        e_pallas = ctx_pallas.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        assert abs(e_pallas - e_ref) < 2.0, f"Expected NoCutoff GBSA parity ({e_ref:.2f}), got {e_pallas:.2f}"

        ctx_xla = ProlixContext(
            sys,
            openmm.VerletIntegrator(0.001),
            config=ProlixOpenMMConfig(precision="double", use_pallas=False),
        )
        ctx_xla.setPositions(pos)
        assert ctx_xla._neighbor_idx is None
        e_xla = ctx_xla.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        assert abs(e_xla - e_ref) < 2.0, f"Expected XLA NoCutoff GBSA parity ({e_ref:.2f}), got {e_xla:.2f}"

    def test_bug_omm10_float32_covalent_exclusion_no_catastrophic_cancellation(self):
        """Short 0.96 Å covalent exclusions with sigma=3.4 Å do not lose digits in single/mixed precision."""
        n_pairs = 120
        sys = openmm.System()
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        pos = np.zeros((2 * n_pairs, 3), dtype=np.float64)
        for p in range(n_pairs):
            i, j = 2 * p, 2 * p + 1
            sys.addParticle(12.0)
            sys.addParticle(12.0)
            nb.addParticle(0.25, 0.34, 0.80)
            nb.addParticle(-0.25, 0.34, 0.80)
            # Excluded 0.096 nm (0.96 Å) covalent pair -> unmasked (3.4/0.96)^12 ~ 3.8e6 kcal/mol per pair!
            nb.addException(i, j, 0.0, 0.34, 0.0)
            base_x = (p % 10) * 0.65
            base_y = ((p // 10) % 6) * 0.65
            base_z = (p // 60) * 0.65
            pos[i] = [base_x, base_y, base_z]
            pos[j] = [base_x + 0.096, base_y, base_z]
        sys.addForce(nb)

        ref_ctx = openmm.Context(
            sys,
            openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("Reference"),
        )
        ref_ctx.setPositions(pos)
        e_ref = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )

        for prec in ("single", "mixed"):
            ctx = ProlixContext(
                sys,
                openmm.VerletIntegrator(0.001),
                config=ProlixOpenMMConfig(precision=prec, use_pallas=False),
            )
            ctx.setPositions(pos)
            e_prx = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            assert abs(e_prx - e_ref) < 0.05, (
                f"precision={prec!r} suffered float32 exclusion cancellation: "
                f"expected {e_ref:.4f} kJ/mol, got {e_prx:.4f} kJ/mol"
            )

    def test_bug_omm11_large_system_exceeding_bucket_ladders_auto_fallback(self):
        """Systems exceeding `ATOM_BUCKETS[-1]` (60,000) auto-fallback to `use_size_buckets=False`."""
        n_large = 60_010
        sys = openmm.System()
        for _ in range(n_large):
            sys.addParticle(12.0)
        conv = from_openmm_system(sys)
        assert conv.bundle.shape_spec.use_size_buckets is False
        assert int(conv.bundle.n_atoms) == n_large
        assert conv.bundle.masses.shape[0] == n_large
