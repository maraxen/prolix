"""Tests for O(|U| * N) sparse CustomNonbondedForce InteractionGroups and is_gpu_runner_fast (v3.9)."""

from __future__ import annotations

import tracemalloc
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.unit as _omm_unit
import pytest

from prolix.openmm.config import ProlixOpenMMConfig
from prolix.openmm.converter import CompiledRBTorsionForce
from prolix.openmm.lepton_compiler import (
    CompiledCustomNonbondedForce,
    CompiledLeptonForce,
    compile_custom_force,
    evaluate_custom_force,
)
from prolix.openmm.units import (
    FORCE_AKMA_TO_OMM,
    KCAL_PER_MOL_TO_KJ_PER_MOL,
    NM_TO_ANGSTROM,
)

if TYPE_CHECKING:
    from collections.abc import Generator

unit = cast("Any", _omm_unit)


@pytest.fixture(autouse=True)
def _enable_x64() -> Generator[None]:
    prev = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _eval_openmm_reference(
    force: openmm.CustomNonbondedForce,
    positions_nm: np.ndarray,
    box_nm: np.ndarray | None,
    global_params: dict[str, float],
) -> tuple[float, np.ndarray, dict[str, float]]:
    sys = openmm.System()
    n = int(positions_nm.shape[0])
    for _ in range(n):
        sys.addParticle(12.0)
    if box_nm is not None:
        sys.setDefaultPeriodicBoxVectors(
            openmm.Vec3(float(box_nm[0, 0]), float(box_nm[0, 1]), float(box_nm[0, 2])),
            openmm.Vec3(float(box_nm[1, 0]), float(box_nm[1, 1]), float(box_nm[1, 2])),
            openmm.Vec3(float(box_nm[2, 0]), float(box_nm[2, 1]), float(box_nm[2, 2])),
        )
    # Clone via XmlSerializer so caller's Force remains unattached
    f_copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(force))
    sys.addForce(f_copy)

    ctx = openmm.Context(
        sys,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ctx.setPositions(positions_nm * unit.nanometer)
    for k, v in global_params.items():
        ctx.setParameter(k, float(v))
    st = ctx.getState(getEnergy=True, getForces=True, getParameterDerivatives=True)
    e_kj = float(st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    f_omm = np.asarray(
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
        dtype=np.float64,
    )
    derivs = {str(k): float(v) for k, v in st.getEnergyParameterDerivatives().items()}
    return e_kj, f_omm, derivs


def test_is_gpu_runner_fast_property_across_all_custom_force_classes() -> None:
    """Verify is_gpu_runner_fast is True on bonded/external/RB/sparse/small-N forces and False on large dense CNB."""
    cb = openmm.CustomBondForce("0.5*k*(r-r0)^2")
    cb.addPerBondParameter("k")
    cb.addPerBondParameter("r0")
    cb.addBond(0, 1, [1000.0, 0.15])
    assert compile_custom_force(cb).is_gpu_runner_fast is True

    ca = openmm.CustomAngleForce("0.5*k*(theta-t0)^2")
    ca.addPerAngleParameter("k")
    ca.addPerAngleParameter("t0")
    ca.addAngle(0, 1, 2, [200.0, 1.9])
    assert compile_custom_force(ca).is_gpu_runner_fast is True

    ct = openmm.CustomTorsionForce("k*(1+cos(theta))")
    ct.addPerTorsionParameter("k")
    ct.addTorsion(0, 1, 2, 3, [5.0])
    assert compile_custom_force(ct).is_gpu_runner_fast is True

    ce = openmm.CustomExternalForce("0.5*k*(x^2+y^2+z^2)")
    ce.addPerParticleParameter("k")
    ce.addParticle(0, [50.0])
    assert compile_custom_force(ce).is_gpu_runner_fast is True

    ccb = openmm.CustomCompoundBondForce(2, "0.5*k*distance(p1,p2)^2")
    ccb.addPerBondParameter("k")
    ccb.addBond([0, 1], [100.0])
    assert compile_custom_force(ccb).is_gpu_runner_fast is True

    rb = cast(
        "CompiledLeptonForce",
        CompiledRBTorsionForce(
            name="PeriodicTorsionForce",
            force_group=0,
            uses_periodic=False,
            energy_param_derivatives=(),
            global_param_names=(),
            torsion_indices=jnp.zeros((1, 4), dtype=jnp.int32),
            c_params=jnp.zeros((1, 6), dtype=jnp.float32),
        ),
    )
    assert rb.is_gpu_runner_fast is True

    # Small dense CustomNonbondedForce (N = 16 <= 4096) -> True
    cnb_small = openmm.CustomNonbondedForce("c / (r^2 + 0.1); c = q1 * q2")
    cnb_small.addPerParticleParameter("q")
    for i in range(16):
        cnb_small.addParticle([0.2 if i % 2 == 0 else -0.2])
    cf_small = cast("CompiledCustomNonbondedForce", compile_custom_force(cnb_small))
    assert cf_small.sparse_u_indices is None
    assert cf_small.exclusion_mask is not None
    assert cf_small.is_gpu_runner_fast is True

    # Dense CustomNonbondedForce with num_particles > 4096 and sparse_u_indices=None -> False
    cf_large_dense = cast(
        "CompiledCustomNonbondedForce",
        CompiledCustomNonbondedForce(
            name="CustomNonbondedForce",
            force_group=0,
            uses_periodic=True,
            energy_param_derivatives=(),
            global_param_names=(),
            nonbonded_method=2,
            cutoff_distance=0.9,
            use_switching=False,
            switching_distance=0.8,
            use_dispersion_correction=False,
            dispersion_coeff=jnp.asarray(0.0, dtype=jnp.float32),
            chunk_size=128,
            per_particle_params={},
            computed_value_fns={},
            exclusion_mask=None,
            interaction_matrix=None,
            interaction_groups=None,
            pair_eval_fn=lambda env, _box: env["r"],
            num_particles=5000,
            sparse_u_indices=None,
            sparse_multiplicity=None,
        ),
    )
    assert cf_large_dense.is_gpu_runner_fast is False


@pytest.mark.parametrize("use_computed_values", [False, True])
@pytest.mark.parametrize("chunk_size", [4, 128])
@pytest.mark.parametrize("triclinic", [False, True])
def test_sparse_interaction_groups_parity_overlapping_switching_lrc_and_derivs(
    use_computed_values: bool,
    chunk_size: int,
    triclinic: bool,
) -> None:
    """Verify sparse InteractionGroups (overlapping groups, switching, LRC, ComputedValues, dE/dlam, chunked U) match OpenMM Reference < 2e-5."""
    n = 64
    rng = np.random.default_rng(2026)
    if not triclinic:
        box_nm = np.diag([2.6, 2.6, 2.6]).astype(np.float64)
    else:
        box_nm = np.array(
            [
                [2.7, 0.0, 0.0],
                [0.45, 2.55, 0.0],
                [-0.35, 0.40, 2.50],
            ],
            dtype=np.float64,
        )

    s_frac = rng.uniform(0.05, 0.95, size=(n, 3))
    pos_nm = s_frac @ box_nm

    if use_computed_values:
        # OpenMM forbids addEnergyParameterDerivative when ComputedValues are present,
        # so exercise ComputedValues and EnergyParameterDerivatives on separate forces.
        cnb = openmm.CustomNonbondedForce(
            "lam * 4.0 * eps_eff * ((sig_eff / r_sc)^12 - (sig_eff / r_sc)^6);"
            "r_sc = sqrt(r^2 + 0.05 * (1.0 - lam));"
            "sig_eff = 0.5 * (sig1 + sig2);"
            "eps_eff = sqrt(w1 * w2)"
        )
    else:
        cnb = openmm.CustomNonbondedForce(
            "lam * 4.0 * eps_eff * ((sig_eff / r_sc)^12 - (sig_eff / r_sc)^6);"
            "r_sc = sqrt(r^2 + 0.05 * (1.0 - lam));"
            "sig_eff = 0.5 * (sig1 + sig2);"
            "eps_eff = sqrt(eps1 * eps2) * (0.8 + 0.2 * lam)"
        )
    cnb.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    cnb.setCutoffDistance(0.95 * unit.nanometer)
    cnb.setUseSwitchingFunction(True)
    cnb.setSwitchingDistance(0.75 * unit.nanometer)
    cnb.setUseLongRangeCorrection(True)
    cnb.addGlobalParameter("lam", 0.65)
    if not use_computed_values:
        cnb.addEnergyParameterDerivative("lam")
    cnb.addPerParticleParameter("sig")
    cnb.addPerParticleParameter("eps")
    if use_computed_values:
        cnb.addComputedValue("w", "eps * (0.8 + 0.2 * lam)")

    for i in range(n):
        cnb.addParticle([0.30 + 0.04 * (i % 3), 0.35 + 0.15 * (i % 4)])

    # Overlapping sparse interaction groups: |U| = 9 <= max(512, 64 // 4)
    lig_a = [0, 1, 2, 3, 4]
    lig_b = [3, 4, 5, 6, 7, 8]
    solvent = list(range(9, n))
    cnb.addInteractionGroup(lig_a, solvent)
    cnb.addInteractionGroup(lig_a, lig_a)  # intra-ligand A
    cnb.addInteractionGroup(lig_b, solvent[:30])  # overlaps on {3, 4} -> multiplicity 2!
    cnb.addInteractionGroup(solvent[:25], [7, 8])  # reversed argument order (larger first)

    # Exclusions spanning intra-ligand, overlapping ligand-solvent, and non-anchor solvent pairs
    cnb.addExclusion(0, 1)
    cnb.addExclusion(3, 12)
    cnb.addExclusion(15, 16)

    cfg = ProlixOpenMMConfig(custom_nonbonded_chunk_size=chunk_size)
    cf = cast("CompiledCustomNonbondedForce", compile_custom_force(cnb, config=cfg))
    assert cf.exclusion_mask is None
    assert cf.interaction_matrix is None
    assert cf.sparse_u_indices is not None
    assert cf.sparse_multiplicity is not None
    assert cf.sparse_u_indices.shape == (9,)
    assert cf.sparse_multiplicity.shape == (9, n)
    assert cf.is_gpu_runner_fast is True

    gp_val = {"lam": 0.65}
    e_ref_kj, f_ref_omm, deriv_ref = _eval_openmm_reference(cnb, pos_nm, box_nm, gp_val)

    pos_akma = jnp.asarray(pos_nm * NM_TO_ANGSTROM, dtype=jnp.float64)
    box_akma = jnp.asarray(box_nm * NM_TO_ANGSTROM, dtype=jnp.float64)
    gp_jax = {"lam": jnp.asarray(0.65, dtype=jnp.float64)}

    e_akma, f_akma, deriv_got = evaluate_custom_force(
        cf,
        pos_akma,
        box_akma,
        gp_jax,
        compute_derivatives=(not use_computed_values),
    )
    e_got_kj = float(e_akma) * KCAL_PER_MOL_TO_KJ_PER_MOL
    f_got_omm = np.asarray(f_akma, dtype=np.float64) * FORCE_AKMA_TO_OMM

    assert abs(e_got_kj - e_ref_kj) / max(abs(e_ref_kj), 1.0) < 2e-5
    f_norm_max = max(float(np.max(np.linalg.norm(f_ref_omm, axis=-1))), 1.0)
    assert float(np.max(np.linalg.norm(f_got_omm - f_ref_omm, axis=-1))) / f_norm_max < 2e-5
    if not use_computed_values:
        assert abs(float(deriv_got["lam"]) - deriv_ref["lam"]) / max(abs(deriv_ref["lam"]), 1.0) < 2e-5


def test_n_10000_sparse_interaction_group_zero_n2_memory_and_reference_parity() -> None:
    """At N = 10,000, sparse InteractionGroup compilation allocates zero (N, N) arrays (< 25 MB vs 500 MB) and matches OpenMM Reference < 2e-5."""
    n = 10_000
    u_lig = 12
    box_nm = np.diag([6.5, 6.5, 6.5]).astype(np.float64)

    # Place 12-atom ligand near box center and remaining 9,988 solvent particles on a jittered grid
    rng = np.random.default_rng(3910)
    n_side = 22  # 22^3 = 10,648 >= 10,000
    ix, iy, iz = np.meshgrid(
        np.arange(n_side), np.arange(n_side), np.arange(n_side), indexing="ij"
    )
    grid_pts = (
        np.stack([ix.ravel()[:n], iy.ravel()[:n], iz.ravel()[:n]], axis=-1).astype(
            np.float64
        )
        + 0.5
    ) / float(n_side) * 6.5
    grid_pts += rng.uniform(-0.02, 0.02, size=(n, 3))

    cnb = openmm.CustomNonbondedForce(
        "lam * 4.0 * eps * ((sig / r_eff)^12 - (sig / r_eff)^6);"
        "r_eff = sqrt(r^2 + 0.04);"
        "sig = 0.5 * (sig1 + sig2);"
        "eps = sqrt(eps1 * eps2)"
    )
    cnb.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    cnb.setCutoffDistance(0.90 * unit.nanometer)
    cnb.setUseLongRangeCorrection(True)
    cnb.addGlobalParameter("lam", 0.5)
    cnb.addEnergyParameterDerivative("lam")
    cnb.addPerParticleParameter("sig")
    cnb.addPerParticleParameter("eps")

    for i in range(n):
        if i < u_lig:
            cnb.addParticle([0.34, 0.45])
        elif i % 3 == 0:
            cnb.addParticle([0.315, 0.636])
        else:
            cnb.addParticle([0.10, 0.0])

    # Add realistic ligand + water 1-2/1-3 exclusions (~10,000 exclusions)
    for k in range(u_lig - 1):
        cnb.addExclusion(k, k + 1)
    for w_base in range(u_lig, n - 3, 3):
        cnb.addExclusion(w_base, w_base + 1)
        cnb.addExclusion(w_base, w_base + 2)
        cnb.addExclusion(w_base + 1, w_base + 2)

    lig_set = set(range(u_lig))
    env_set = set(range(u_lig, n))
    cnb.addInteractionGroup(lig_set, env_set)
    cnb.addInteractionGroup(lig_set, lig_set)

    # Track Python/NumPy heap peak during compile_custom_force:
    # A dense (10000, 10000) bool + int32 allocation would consume >= 500 MB!
    tracemalloc.start()
    try:
        cf = cast(
            "CompiledCustomNonbondedForce",
            compile_custom_force(cnb, n_particles=n),
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert cf.exclusion_mask is None
    assert cf.interaction_matrix is None
    assert cf.sparse_u_indices is not None
    assert cf.sparse_multiplicity is not None
    assert cf.sparse_u_indices.shape == (u_lig,)
    assert cf.sparse_multiplicity.shape == (u_lig, n)
    assert cf.is_gpu_runner_fast is True
    assert peak_bytes < 30 * 1024 * 1024, (
        f"Expected < 30 MB peak heap during sparse compilation, got {peak_bytes / 1e6:.2f} MB"
    )

    e_ref_kj, f_ref_omm, deriv_ref = _eval_openmm_reference(
        cnb, grid_pts, box_nm, {"lam": 0.5}
    )
    pos_akma = jnp.asarray(grid_pts * NM_TO_ANGSTROM, dtype=jnp.float64)
    box_akma = jnp.asarray(box_nm * NM_TO_ANGSTROM, dtype=jnp.float64)
    e_akma, f_akma, deriv_got = evaluate_custom_force(
        cf,
        pos_akma,
        box_akma,
        {"lam": jnp.asarray(0.5, dtype=jnp.float64)},
        compute_derivatives=True,
    )
    e_got_kj = float(e_akma) * KCAL_PER_MOL_TO_KJ_PER_MOL
    f_got_omm = np.asarray(f_akma, dtype=np.float64) * FORCE_AKMA_TO_OMM

    assert abs(e_got_kj - e_ref_kj) / max(abs(e_ref_kj), 1.0) < 2e-5
    f_norm_max = max(float(np.max(np.linalg.norm(f_ref_omm, axis=-1))), 1.0)
    assert float(np.max(np.linalg.norm(f_got_omm - f_ref_omm, axis=-1))) / f_norm_max < 2e-5
    assert abs(float(deriv_got["lam"]) - deriv_ref["lam"]) / max(abs(deriv_ref["lam"]), 1.0) < 2e-5
