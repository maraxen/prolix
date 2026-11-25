"""CPU unit tests for (I, 3) harmonic impropers, EnsemblePlan OpenMM knob preservation, and BatchSimulation guards."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import openmm
from jax_md import space
from openmm import app, unit

from prolix.api.bundle_md import _bonded_energy_core, single_padded_energy
from prolix.api.ensemble_plan import EnsemblePlan
from prolix.openmm import ProlixBatchSimulation, ProlixOpenMMConfig, ProlixSimulation
from prolix.typing import PhysicsSystem


def _nonplanar_improper_openmm_system():
    """4-atom non-planar molecule with a single OpenMM CustomTorsionForce harmonic improper."""
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("MOL", ch)
    elem_c = app.Element.getBySymbol("C")
    for idx in range(4):
        top.addAtom(f"C{idx}", elem_c, res)

    imp = openmm.CustomTorsionForce("0.5*k*(theta-theta0)^2")
    imp.addPerTorsionParameter("k")
    imp.addPerTorsionParameter("theta0")
    imp.addTorsion(0, 1, 2, 3, [400.0, 0.25])
    sys.addForce(imp)

    pos_nm = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.15, 0.15, 0.0],
            [0.05, 0.15, 0.12],
        ],
        dtype=np.float64,
    )
    return top, sys, pos_nm


def test_harmonic_improper_i3_parity_across_paths():
    top, sys, pos_nm = _nonplanar_improper_openmm_system()
    integ = openmm.VerletIntegrator(0.001 * unit.picosecond)
    cfg = ProlixOpenMMConfig(use_pallas=False)
    sim = ProlixSimulation(top, sys, integ, config=cfg)
    sim.context.setPositions(pos_nm * unit.nanometer)
    omm_e_kcal = (
        sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    )
    assert omm_e_kcal > 1.0

    # Pure-JAX and Pallas-interpret bundle_md must match ProlixContext
    bun = sim.context._converted_system.bundle
    assert bun.improper_params.shape[-1] == 3
    assert not bool(np.asarray(bun.improper_is_periodic))
    pos_A = jnp.asarray(pos_nm * 10.0, dtype=jnp.float32)

    e_bmd_jax = float(_bonded_energy_core(bun, use_pallas=False)(pos_A))
    np.testing.assert_allclose(e_bmd_jax, omm_e_kcal, rtol=1e-4, atol=1e-4)

    # Pure-JAX batched_energy single_padded_energy path with (I, 2) harmonic impropers must match
    empty2 = jnp.zeros((0, 2), dtype=jnp.int32)
    empty3 = jnp.zeros((0, 3), dtype=jnp.int32)
    empty_p2 = jnp.zeros((0, 2), dtype=jnp.float32)
    empty_m = jnp.zeros((0,), dtype=bool)
    ones_b = jnp.ones((4,), dtype=bool)
    ps = PhysicsSystem(
        positions=pos_A,
        charges=jnp.zeros((4,), dtype=jnp.float32),
        sigmas=jnp.ones((4,), dtype=jnp.float32),
        epsilons=jnp.zeros((4,), dtype=jnp.float32),
        radii=jnp.ones((4,), dtype=jnp.float32),
        scaled_radii=jnp.ones((4,), dtype=jnp.float32),
        masses=jnp.full((4,), 12.0, dtype=jnp.float32),
        element_ids=jnp.full((4,), 6, dtype=jnp.int32),
        atom_mask=ones_b,
        is_hydrogen=jnp.zeros((4,), dtype=bool),
        is_backbone=jnp.zeros((4,), dtype=bool),
        is_heavy=ones_b,
        protein_atom_mask=ones_b,
        water_atom_mask=jnp.zeros((4,), dtype=bool),
        bonds=empty2,
        bond_params=empty_p2,
        bond_mask=empty_m,
        angles=empty3,
        angle_params=empty_p2,
        angle_mask=empty_m,
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((0, 3), dtype=jnp.float32),
        dihedral_mask=empty_m,
        impropers=bun.improper_idx,
        improper_params=bun.improper_params[..., :2],
        improper_mask=bun.improper_mask,
        urey_bradley_bonds=jnp.zeros((0, 3), dtype=jnp.int32),
        urey_bradley_params=empty_p2,
        urey_bradley_mask=empty_m,
        n_real_atoms=jnp.array(4, dtype=jnp.int32),
        n_padded_atoms=4,
        pme_alpha=0.0,
    )
    e_be_jax = float(single_padded_energy(ps, space.free()[0], implicit_solvent=False, use_pallas=False))
    np.testing.assert_allclose(e_be_jax, omm_e_kcal, rtol=1e-4, atol=1e-4)

    # Pallas-interpret path on bundle must also match
    e_bmd_pallas = float(_bonded_energy_core(bun, use_pallas=True)(pos_A))
    np.testing.assert_allclose(e_bmd_pallas, omm_e_kcal, rtol=1e-4, atol=1e-4)


def test_ensemble_plan_from_openmm_systems_preserves_non_pme_knobs():
    sys = openmm.System()
    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("MOL", ch)
    elem_c = app.Element.getBySymbol("C")
    for i in range(4):
        sys.addParticle(12.0)
        top.addAtom(f"C{i}", elem_c, res)

    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(3.0, 0.0, 0.0),
        openmm.Vec3(0.0, 3.0, 0.0),
        openmm.Vec3(0.0, 0.0, 3.0),
    )
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setUseDispersionCorrection(False)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(0.8 * unit.nanometer)
    nb.setExceptionsUsePeriodicBoundaryConditions(True)
    for i in range(4):
        nb.addParticle(0.1 * ((-1) ** i), 0.34, 0.1)
    sys.addForce(nb)

    pos_nm = np.array([[0.1 * i, 0.2, 0.3] for i in range(4)], dtype=np.float64)
    plan = EnsemblePlan.from_openmm_systems(
        [sys], [pos_nm], use_pallas=False, use_pallas_pme=False, nonbonded_engine="dense"
    )
    assert plan.knob_surface is not None
    assert plan.knob_surface.nonbonded.dispersion_correction is False
    assert plan.knob_surface.nonbonded.exceptions_use_pbc is True
    assert abs(plan.knob_surface.nonbonded.switch_width_angstrom - 1.0) < 1e-5


def test_batch_simulation_intermolecular_exception_fallback_and_use_size_buckets():
    sys = openmm.System()
    top = app.Topology()
    ch = top.addChain()
    r0 = top.addResidue("M0", ch)
    r1 = top.addResidue("M1", ch)
    elem_c = app.Element.getBySymbol("C")
    atoms = []
    for i in range(4):
        sys.addParticle(12.0)
        atoms.append(top.addAtom(f"C{i}", elem_c, r0 if i < 2 else r1))

    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(3.0, 0.0, 0.0),
        openmm.Vec3(0.0, 3.0, 0.0),
        openmm.Vec3(0.0, 0.0, 3.0),
    )
    bf = openmm.HarmonicBondForce()
    bf.addBond(0, 1, 0.15, 100000.0)
    bf.addBond(2, 3, 0.15, 100000.0)
    sys.addForce(bf)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setExceptionsUsePeriodicBoundaryConditions(False)
    for _ in range(4):
        nb.addParticle(0.1, 0.34, 0.1)
    # Non-zero intermolecular exception between molecule 0 (atom 0) and molecule 1 (atom 2)
    nb.addException(0, 2, 0.05, 0.34, 0.1)
    sys.addForce(nb)

    integ = openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
    cfg = ProlixOpenMMConfig(use_pallas=False, use_size_buckets=False)
    sim = ProlixBatchSimulation([sys], [integ], topologies=[top], config=cfg)
    assert sim._contexts is not None
    assert sim.plan is None
    assert sim._build_knob_surface("dense", False, False).pallas.use_size_buckets is False
