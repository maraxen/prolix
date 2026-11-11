"""Adversarial stress and dynamics challenge test suite for PR #44 (feat/openmm-backend).

Author: Challenger r2_2
Focus Areas:
1. Multi-Integrator Trajectory Differentiation & Stability (lfmiddle, baoab, verlet, brownian).
2. 1000-Step NVE Energy Conservation (|delta E / E0| < 1e-4).
3. NeighborList O(N * K) Memory Scaling at N=4000 (sparse shapes, peak RAM, zero dense [4000, 4000]).
4. Barostat & Rollout Stability (MonteCarloBarostat with 'mc' and 'scr' modes).
5. Defect reproduction: DEF-23 ('lfmiddle' config rejection) and DEF-24 ('baoab' without Langevin integrator drops thermal noise).
"""

from __future__ import annotations

import re
import tracemalloc
import jax
import numpy as np
import pytest

try:
    import openmm
    from openmm import app, unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

from prolix.openmm import ProlixContext, ProlixOpenMMConfig, ProlixSimulation
from prolix.openmm.context import _eval_system_potential_energy

pytestmark = pytest.mark.openmm


# =============================================================================
# 1. Multi-Integrator Trajectory Differentiation & Physical Dynamics
# =============================================================================


class TestMultiIntegratorDynamicsAndStability:
    """Adversarially challenge trajectory differentiation across integrators and NVE drift."""

    def test_multi_integrator_trajectory_differentiation(self):
        """Verify that OpenMM integrators (LFMiddle, Verlet, Brownian, and Langevin) produce distinct, valid dynamics."""
        def make_test_system():
            sys = openmm.System()
            for _ in range(4):
                sys.addParticle(12.0)
            bonds = openmm.HarmonicBondForce()
            bonds.addBond(0, 1, 0.15 * unit.nanometer, 20000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
            bonds.addBond(1, 2, 0.15 * unit.nanometer, 20000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
            bonds.addBond(2, 3, 0.15 * unit.nanometer, 20000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
            sys.addForce(bonds)
            return sys

        init_pos = [
            openmm.Vec3(0.0, 0.0, 0.0),
            openmm.Vec3(0.15, 0.02, 0.0),
            openmm.Vec3(0.30, -0.01, 0.02),
            openmm.Vec3(0.45, 0.03, -0.01),
        ] * unit.nanometer

        # Test distinct integrator implementations
        integrators = {
            "lfmiddle": openmm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picosecond),
            "verlet": openmm.VerletIntegrator(0.001 * unit.picosecond),
            "brownian": openmm.BrownianIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picosecond),
        }

        final_trajectories: dict[str, np.ndarray] = {}

        for name, integ in integrators.items():
            sys = make_test_system()
            cfg_kwargs = {"deterministic_seed": 42}
            if name == "brownian":
                cfg_kwargs["integrator_scheme"] = "brownian"

            cfg = ProlixOpenMMConfig(**cfg_kwargs)
            ctx = ProlixContext(sys, integ, config=cfg)
            ctx.setPositions(init_pos)
            ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)

            # Roll out 10 steps
            ctx.step(10)

            st = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True)
            pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            vel = st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
            pe = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            ke = st.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)

            # Assert valid physical observables (no NaNs or Infs)
            assert np.all(np.isfinite(pos)), f"Integrator {name} produced NaN/Inf positions"
            assert np.all(np.isfinite(vel)), f"Integrator {name} produced NaN/Inf velocities"
            assert np.isfinite(pe), f"Integrator {name} produced NaN/Inf potential energy"
            assert np.isfinite(ke), f"Integrator {name} produced NaN/Inf kinetic energy"

            final_trajectories[name] = pos

        # Assert mutually distinct trajectories between all pairs
        names = list(integrators.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s_a, s_b = names[i], names[j]
                diff = np.linalg.norm(final_trajectories[s_a] - final_trajectories[s_b])
                assert diff > 1e-4, (
                    f"Integrators '{s_a}' and '{s_b}' produced identical trajectories "
                    f"(norm diff = {diff:.6e} nm). Integrators must produce distinct dynamics!"
                )

    def test_nve_energy_conservation_1000_steps(self):
        """Run 1000-step NVE simulation and verify relative energy drift |(E(t) - E(0)) / E(0)| < 1e-4."""
        sys = openmm.System()
        sys.addParticle(12.0)
        sys.addParticle(12.0)

        b = openmm.HarmonicBondForce()
        b.addBond(0, 1, 0.15 * unit.nanometer, 20000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
        sys.addForce(b)

        top = app.Topology()
        ch = top.addChain()
        res = top.addResidue("RES", ch)
        top.addAtom("A1", app.Element.getBySymbol("C"), res)
        top.addAtom("A2", app.Element.getBySymbol("C"), res)

        integ = openmm.VerletIntegrator(0.0005 * unit.picosecond)  # 0.5 fs timestep
        sim = ProlixSimulation(top, sys, integ)
        sim.context.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(0.155, 0, 0)] * unit.nanometer)
        sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=42)

        s0 = sim.context.getState(getEnergy=True)
        e0 = (
            s0.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            + s0.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        )

        energies: list[float] = []
        for _ in range(10):
            sim.step(100)
            s = sim.context.getState(getEnergy=True)
            e = (
                s.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                + s.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
            )
            energies.append(e)

        max_drift = max(abs(e - e0) for e in energies)
        rel_drift = max_drift / abs(e0)
        assert rel_drift < 1e-4, f"NVE energy drift too large: rel_drift={rel_drift:.2e} (expected < 1e-4)"


# =============================================================================
# 2. Defect Reproduction: DEF-23 ('lfmiddle' rejection) & DEF-24 (BAOAB c2=0)
# =============================================================================


class TestIntegratorDefectReproduction:
    """Verify remediation of DEF-24 ('lfmiddle' support) and DEF-25 (BAOAB thermal noise kick)."""

    def test_def23_lfmiddle_rejected_by_config(self):
        """DEF-24 (formerly numbered DEF-23): ProlixOpenMMConfig accepts integrator_scheme='lfmiddle' and runs."""
        cfg = ProlixOpenMMConfig(integrator_scheme="lfmiddle")
        assert cfg.integrator_scheme == "lfmiddle"
        sys = openmm.System()
        sys.addParticle(12.0)
        integ = openmm.VerletIntegrator(0.001)
        ctx = ProlixContext(sys, integ, config=cfg)
        ctx.setPositions([openmm.Vec3(0, 0, 0)] * unit.nanometer)
        ctx.step(5)
        st = ctx.getState(getPositions=True)
        assert np.all(np.isfinite(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)))

    def test_def24_baoab_scheme_with_verlet_integrator_drops_thermal_noise(self):
        """DEF-25 (formerly numbered DEF-24): ProlixContext with scheme='baoab' and Verlet integrator applies thermal noise kicks."""
        sys = openmm.System()
        sys.addParticle(12.0)
        pos = [openmm.Vec3(0, 0, 0)] * unit.nanometer

        # Run with seed=1
        ctx1 = ProlixContext(
            sys, openmm.VerletIntegrator(0.001),
            config=ProlixOpenMMConfig(integrator_scheme="baoab", deterministic_seed=1),
        )
        ctx1.setPositions(pos)
        ctx1.step(10)
        p1 = ctx1.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        # Run with seed=2
        ctx2 = ProlixContext(
            sys, openmm.VerletIntegrator(0.001),
            config=ProlixOpenMMConfig(integrator_scheme="baoab", deterministic_seed=2),
        )
        ctx2.setPositions(pos)
        ctx2.step(10)
        p2 = ctx2.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        # Remediated: is_langevin is True for scheme='baoab', applying thermal noise kicks (c2 > 0).
        # Consequently, p1 and p2 diverge across different PRNG seeds!
        diff = np.linalg.norm(p1 - p2)
        assert diff > 0.0, (
            f"Expected thermal noise divergence with distinct seeds (diff > 0.0), but got diff={diff}"
        )


# =============================================================================
# 3. NeighborList O(N * K) Memory Scaling at N=4000 (DEF-18)
# =============================================================================


class TestNeighborListMemoryScalingN4000:
    """Adversarially challenge sparse memory allocation and ensure zero dense (4000, 4000) buffers."""

    def test_n4000_neighbor_list_sparse_shape_and_no_dense_allocation(self):
        """Instantiate system with N=4000 particles and verify sparse (4000, K) neighbor list with K << 4000."""
        n_atoms = 4000
        sys = openmm.System()
        sys.setDefaultPeriodicBoxVectors(
            openmm.Vec3(16.0, 0, 0),
            openmm.Vec3(0, 16.0, 0),
            openmm.Vec3(0, 0, 16.0),
        )
        for _ in range(n_atoms):
            sys.addParticle(39.948)

        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        cutoff_nm = 0.9
        nb.setCutoffDistance(cutoff_nm * unit.nanometer)
        for _ in range(n_atoms):
            nb.addParticle(0.0, 0.33, 0.8)
        sys.addForce(nb)

        cfg = ProlixOpenMMConfig(
            nonbonded_engine="neighbor_list",
            nl_skin=0.2,
            nl_capacity_multiplier=1.25,
            nl_update_every=10,
        )
        ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)

        rng = np.random.RandomState(42)
        pos_np = rng.uniform(0.0, 16.0, size=(n_atoms, 3)) * unit.nanometer
        ctx.setPositions(pos_np)

        # 1. Inspect neighbor list allocation and shapes
        assert ctx._neighbor is not None, "NeighborList object was not created in ProlixContext!"
        assert ctx._neighbor_idx is not None, "Neighbor index array is None!"
        n_rows, k_capacity = ctx._neighbor_idx.shape
        assert n_rows == n_atoms, f"Expected neighbor list rows {n_atoms}, got {n_rows}"
        assert k_capacity == 13, f"Expected sparse cell list capacity K=13, got {k_capacity}"
        assert k_capacity < 500, (
            f"Sparse neighbor list capacity K={k_capacity} is too large (expected K << {n_atoms})!"
        )

        # 2. Memory allocation profiling via tracemalloc
        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()
        state = ctx.getState(getEnergy=True, getForces=True)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        assert np.isfinite(pe), f"Potential energy is non-finite: {pe}"
        assert np.all(np.isfinite(forces)), "Forces contain NaN/Inf values"
        assert forces.shape == (n_atoms, 3)

        peak_mem_mb = sum(stat.size for stat in snap_after.compare_to(snap_before, "lineno")) / (1024 * 1024)
        assert peak_mem_mb < 50.0, f"Peak memory allocation {peak_mem_mb:.2f} MB is too large for sparse evaluation!"

        # 3. JAX jaxpr inspection: verify zero dense [4000, 4000] pair matrices
        def pe_fn(p):
            return _eval_system_potential_energy(
                p,
                ctx._box_vectors,
                ctx._bundle,
                ctx._custom_forces,
                ctx._position_restraints,
                ctx._get_dynamic_global_params(),
                ctx._converted_system.force_group_by_component,
                -1,
                True,
                9.0,
                nonbonded_engine="neighbor_list",
                neighbor_idx=ctx._neighbor_idx,
            )

        jaxpr = jax.make_jaxpr(pe_fn)(ctx._positions)
        jaxpr_str = str(jaxpr)
        dense_matches = re.findall(r"\[4000,\s*4000[^\]]*\]", jaxpr_str)
        assert len(dense_matches) == 0, (
            f"Detected dense [4000, 4000] arrays in jaxpr: {dense_matches}. "
            "NeighborList nonbonded engine must never materialize dense pair matrices!"
        )


# =============================================================================
# 4. Barostat & Rollout Stability (MonteCarloBarostat with 'mc' and 'scr')
# =============================================================================


class TestBarostatStabilityAndRollouts:
    """Adversarially challenge barostat stability, box geometry, and multi-step rollouts."""

    @pytest.mark.parametrize("baro_mode", ["mc", "scr"])
    def test_barostat_rollout_stability_and_volume_fluctuations(self, baro_mode: str):
        """Verify MonteCarloBarostat in 'mc' and 'scr' modes completes rollouts stably without NaNs."""
        box_size_nm = 3.0
        sys = openmm.System()
        sys.setDefaultPeriodicBoxVectors(
            openmm.Vec3(box_size_nm, 0, 0),
            openmm.Vec3(0, box_size_nm, 0),
            openmm.Vec3(0, 0, box_size_nm),
        )

        n_particles = 27  # 3x3x3 grid
        for _ in range(n_particles):
            sys.addParticle(39.948)

        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(0.9 * unit.nanometer)
        for _ in range(n_particles):
            nb.addParticle(0.0, 0.34, 1.0)
        sys.addForce(nb)

        # Barostat with frequency = 5 steps
        barostat = openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 5)
        sys.addForce(barostat)

        top = app.Topology()
        ch = top.addChain()
        res = top.addResidue("ARG", ch)
        for i in range(n_particles):
            top.addAtom(f"AR{i}", app.Element.getBySymbol("Ar"), res)

        integ = openmm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin,
            2.0 / unit.picosecond,
            0.001 * unit.picosecond,
        )

        cfg = ProlixOpenMMConfig(
            barostat_mode=baro_mode,
            deterministic_seed=42,
        )
        sim = ProlixSimulation(top, sys, integ, config=cfg)

        # Initialize particles on a 3x3x3 grid inside box
        grid_coords = []
        for x in [0.75, 1.5, 2.25]:
            for y in [0.75, 1.5, 2.25]:
                for z in [0.75, 1.5, 2.25]:
                    grid_coords.append(openmm.Vec3(x, y, z))
        sim.context.setPositions(grid_coords * unit.nanometer)
        sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)

        initial_state = sim.context.getState(getPositions=True, getEnergy=True)
        init_box = initial_state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        init_vol = np.linalg.det(init_box)
        assert abs(init_vol - (box_size_nm ** 3)) < 1e-4

        # Run 50 steps
        sim.step(50)

        final_state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        final_pos = final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        final_vel = final_state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        final_box = final_state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        final_vol = np.linalg.det(final_box)
        pe = final_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        ke = final_state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)

        # Assertions
        assert np.all(np.isfinite(final_pos)), f"Barostat [{baro_mode}] produced NaN/Inf positions"
        assert np.all(np.isfinite(final_vel)), f"Barostat [{baro_mode}] produced NaN/Inf velocities"
        assert np.all(np.isfinite(final_box)), f"Barostat [{baro_mode}] produced NaN/Inf box vectors"
        assert np.isfinite(pe), f"Barostat [{baro_mode}] produced NaN/Inf potential energy: {pe}"
        assert np.isfinite(ke), f"Barostat [{baro_mode}] produced NaN/Inf kinetic energy: {ke}"

        # Box geometry preservation
        assert final_vol > 0.0, f"Volume must remain positive (got {final_vol})"
        assert np.all(np.diag(final_box) > 0.5), f"Box vectors degraded: {final_box}"
        assert 0.5 * init_vol < final_vol < 2.0 * init_vol, (
            f"Volume fluctuation out of bounds: init={init_vol:.3f}, final={final_vol:.3f}"
        )
