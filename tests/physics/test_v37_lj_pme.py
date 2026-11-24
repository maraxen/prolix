"""v3.7 Track 2 validation: P=6 dispersion LJ-PME physics, gradients, StableHLO
portability, and OpenMM Reference NonbondedForce.LJPME parity.
"""

from __future__ import annotations

import math
import re
from typing import Any, cast

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import openmm
import pytest
from openmm import unit as _omm_unit

from prolix import export
from prolix.openmm.config import NonbondedEngineType, PrecisionType, ProlixOpenMMConfig
from prolix.openmm.context import ProlixContext
from prolix.physics.lj_pme import (
    ewald_p6_fourier_kernel_grad,
    ewald_p6_fourier_kernel_value,
    ewald_p6_long_range_value,
    ewald_p6_short_range_value,
    lj_pme_influence_function,
    lj_pme_total_dispersion_energy,
    resolve_ljpme_parameters,
)

unit = cast("Any", _omm_unit)
pytestmark = pytest.mark.fast

FORBIDDEN_STABLEHLO_SUBSTRINGS = (
    "triton",
    "mosaic",
    "pallas",
    "tpu_custom_call",
    "xla.gpu",
    "lapack",
    "cusolver",
    "cublas",
    "cudnn",
)
_CUSTOM_CALL_RE = re.compile(r"stablehlo\.custom_call\s+@([\w.$\-]+)")
_TARGET_NAME_RE = re.compile(r'call_target_name\s*=\s*"([^"]+)"')


# =============================================================================
# 1. Splitting Identity, DC k^2 = 0 Finite Mode, & Small-y Taylor Series
# =============================================================================


@pytest.mark.parametrize(
    "y_val",
    [1e-8, 1e-4, 0.05, 0.5, 1.99, 2.01, 5.0, 12.0],
)
def test_ewald_p6_short_plus_long_splitting_identity_and_r0_limit(y_val: float) -> None:
    with jax.enable_x64(True):
        alpha_lj = 0.35
        r2 = jnp.asarray(y_val / (alpha_lj**2), dtype=jnp.float64)
        short_v = float(ewald_p6_short_range_value(alpha_lj, r2))
        long_v = float(ewald_p6_long_range_value(alpha_lj, r2))
        exact_inv_r6 = float(1.0 / (r2**3))
        np.testing.assert_allclose(short_v + long_v, exact_inv_r6, rtol=1e-13, atol=0.0)

        # Verify r -> 0 finite value and NaN-free gradient
        r2_zero = jnp.asarray(0.0, dtype=jnp.float64)
        val0 = float(ewald_p6_long_range_value(alpha_lj, r2_zero))
        grad0 = float(jax.grad(lambda x: ewald_p6_long_range_value(alpha_lj, x))(r2_zero))
        np.testing.assert_allclose(val0, (alpha_lj**6) / 6.0, rtol=1e-15)
        np.testing.assert_allclose(grad0, -(alpha_lj**8) / 8.0, rtol=1e-15)


def test_ewald_p6_dc_mode_kept_and_box_grad_nan_free() -> None:
    with jax.enable_x64(True):
        alpha_lj = 0.31
        k2_vec = jnp.array([0.0, 0.01, 0.25], dtype=jnp.float64)
        k6_vals = ewald_p6_fourier_kernel_value(alpha_lj, k2_vec)
        expected_dc = (math.pi**1.5 / 3.0) * (alpha_lj**3)
        np.testing.assert_allclose(float(k6_vals[0]), expected_dc, rtol=1e-15)

        dk6_dk2 = ewald_p6_fourier_kernel_grad(alpha_lj, k2_vec)
        assert float(dk6_dk2[0]) == 0.0
        assert bool(jnp.all(jnp.isfinite(dk6_dk2)))

        # Verify lj_pme_influence_function keeps finite negative DC entry -K_6(0) / V
        box = jnp.array([25.0, 27.0, 30.0], dtype=jnp.float64)
        g6 = lj_pme_influence_function((14, 15, 16), box, alpha_lj, order=5)
        vol = float(np.prod(np.asarray(box)))
        np.testing.assert_allclose(float(g6[0, 0, 0]), -expected_dc / vol, rtol=1e-14)


def test_resolve_ljpme_parameters_matches_openmm_reference() -> None:
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors([3.0, 0.0, 0.0], [0.0, 3.2, 0.0], [0.0, 0.0, 3.8])
    sys.addParticle(12.0)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.LJPME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setEwaldErrorTolerance(5e-4)
    nb.addParticle(0.0, 0.3, 1.0)
    sys.addForce(nb)

    ctx = openmm.Context(
        sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    al_omm, nx_omm, ny_omm, nz_omm = nb.getLJPMEParametersInContext(ctx)
    al_res, dims_res = resolve_ljpme_parameters(
        np.array([30.0, 32.0, 38.0]),
        cutoff=9.0,
        ewald_error_tolerance=5e-4,
    )
    np.testing.assert_allclose(al_res, float(al_omm) / 10.0, rtol=1e-14)
    assert dims_res == (int(nx_omm), int(ny_omm), int(nz_omm))


# =============================================================================
# 2. Sparse O(N * K) Gradients & Portable StableHLO Export
# =============================================================================


def test_lj_pme_sparse_neighbor_list_grads_and_stablehlo_export() -> None:
    with jax.enable_x64(True):
        rng = np.random.default_rng(2026)
        n_real = 8
        n_pad = 10
        pos_np = rng.uniform(2.0, 22.0, size=(n_pad, 3))
        pos = jnp.asarray(pos_np, dtype=jnp.float64)
        # Include zero-epsilon hydrogen (index 1) and padded ghost atoms (8, 9)
        sig = jnp.array(
            [3.15, 0.40, 3.40, 2.95, 3.20, 3.05, 3.30, 2.85, 3.50, 3.50],
            dtype=jnp.float64,
        )
        eps = jnp.array(
            [0.15, 0.0, 0.12, 0.20, 0.18, 0.11, 0.16, 0.14, 0.25, 0.25],
            dtype=jnp.float64,
        )
        atom_mask = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
        box = jnp.array([25.0, 26.0, 27.0], dtype=jnp.float64)
        cutoff = 10.0
        alpha_lj = 0.30
        grid_dims = (16, 16, 18)

        # Build full neighbor list of active upper-triangular candidates (padded with -1)
        nb_rows = []
        for i in range(n_pad):
            row = list(range(i + 1, n_real))
            row += [-1] * (n_real - len(row))
            nb_rows.append(row)
        neighbor_idx = jnp.asarray(nb_rows, dtype=jnp.int32)
        excl_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        excl_mask = jnp.array([True, True], dtype=bool)
        excl_scale_vdw = jnp.ones_like(neighbor_idx, dtype=jnp.float64)
        # Zero (0, 1) and (2, 3) in direct neighbor list
        excl_scale_vdw = excl_scale_vdw.at[0, 0].set(0.0).at[2, 0].set(0.0)

        def _e_fn(p, s, ep, b):
            return lj_pme_total_dispersion_energy(
                p,
                s,
                ep,
                b,
                neighbor_idx,
                cutoff=cutoff,
                alpha_lj=alpha_lj,
                grid_dims=grid_dims,
                order=5,
                atom_mask=atom_mask,
                excl_indices=excl_indices,
                excl_mask=excl_mask,
                excl_scale_vdw=excl_scale_vdw,
                lj_combining_rule="lorentz_berthelot",
            )

        g_pos, g_sig, g_eps, g_box = jax.grad(_e_fn, argnums=(0, 1, 2, 3))(pos, sig, eps, box)
        assert bool(jnp.all(jnp.isfinite(g_pos)))
        assert bool(jnp.all(jnp.isfinite(g_sig)))
        assert bool(jnp.all(jnp.isfinite(g_eps)))
        assert bool(jnp.all(jnp.isfinite(g_box)))
        # Zero-epsilon H and masked padding atoms have exact 0 gradient
        assert float(g_eps[1]) == 0.0
        assert float(g_sig[8]) == 0.0
        assert float(g_eps[8]) == 0.0

        h = 1e-6
        ep_pos = float(_e_fn(pos.at[2, 1].add(h), sig, eps, box))
        em_pos = float(_e_fn(pos.at[2, 1].add(-h), sig, eps, box))
        np.testing.assert_allclose(float(g_pos[2, 1]), (ep_pos - em_pos) / (2.0 * h), rtol=1e-6)

        ep_sig = float(_e_fn(pos, sig.at[0].add(h), eps, box))
        em_sig = float(_e_fn(pos, sig.at[0].add(-h), eps, box))
        np.testing.assert_allclose(float(g_sig[0]), (ep_sig - em_sig) / (2.0 * h), rtol=1e-6)

        ep_eps = float(_e_fn(pos, sig, eps.at[0].add(h), box))
        em_eps = float(_e_fn(pos, sig, eps.at[0].add(-h), box))
        np.testing.assert_allclose(float(g_eps[0]), (ep_eps - em_eps) / (2.0 * h), rtol=1e-6)

        ep_box = float(_e_fn(pos, sig, eps, box.at[0].add(h)))
        em_box = float(_e_fn(pos, sig, eps, box.at[0].add(-h)))
        np.testing.assert_allclose(float(g_box[0]), (ep_box - em_box) / (2.0 * h), rtol=1e-6)

    # Portable StableHLO export guardrail (float32)
    pos32 = pos.astype(jnp.float32)
    sig32 = sig.astype(jnp.float32)
    eps32 = eps.astype(jnp.float32)
    box32 = box.astype(jnp.float32)
    esv32 = excl_scale_vdw.astype(jnp.float32)

    def _export_fn(p, s, ep, b):
        return jax.value_and_grad(
            lambda p_i, s_i, e_i, b_i: lj_pme_total_dispersion_energy(
                p_i,
                s_i,
                e_i,
                b_i,
                neighbor_idx,
                cutoff=cutoff,
                alpha_lj=alpha_lj,
                grid_dims=grid_dims,
                order=5,
                atom_mask=atom_mask,
                excl_indices=excl_indices,
                excl_mask=excl_mask,
                excl_scale_vdw=esv32,
            ),
            argnums=(0, 1, 2, 3),
        )(p, s, ep, b)

    with export.export_context():
        jitted = jax.jit(_export_fn)
        mlir_text = jitted.lower(pos32, sig32, eps32, box32).as_text()
        exported = jax.export.export(jitted)(pos32, sig32, eps32, box32)
    low = mlir_text.lower()
    assert not [tok for tok in FORBIDDEN_STABLEHLO_SUBSTRINGS if tok in low]
    targets = set(_CUSTOM_CALL_RE.findall(mlir_text)) | set(_TARGET_NAME_RE.findall(mlir_text))
    assert not targets
    rt_e, _ = exported.call(pos32, sig32, eps32, box32)
    assert bool(jnp.isfinite(rt_e))


# =============================================================================
# 3. OpenMM Reference Platform NonbondedForce.LJPME Parity (ProlixContext)
# =============================================================================


def _build_test_ljpme_openmm_system(
    *,
    method: int = openmm.NonbondedForce.LJPME,
    explicit_grids: bool = True,
    exceptions_use_pbc: bool = False,
    with_offset: bool = False,
    with_charges: bool = True,
) -> tuple[openmm.System, np.ndarray]:
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors([2.6, 0.0, 0.0], [0.0, 2.8, 0.0], [0.0, 0.0, 3.0])
    masses = [16.0, 1.008, 12.0, 14.0, 32.0, 19.0]
    for m in masses:
        sys.addParticle(m)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(method)
    nb.setCutoffDistance(0.95 * unit.nanometer)
    nb.setEwaldErrorTolerance(5e-4)
    # Set both True to verify LJPME ignores switching function and isotropic tail correction
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(0.75 * unit.nanometer)
    nb.setUseDispersionCorrection(True)
    nb.setExceptionsUsePeriodicBoundaryConditions(exceptions_use_pbc)
    if explicit_grids:
        # 64^3 electrostatic mesh (order-4 SPME in Prolix vs order-5 in OpenMM) paired with
        # a non-cubic 16x18x20 dispersion LJPME mesh (order-5 in both engines).
        nb.setPMEParameters(3.2, 64, 64, 64)
        if method == openmm.NonbondedForce.LJPME:
            nb.setLJPMEParameters(3.0, 16, 18, 20)

    params = [
        (-0.834, 0.31507, 0.63597),
        (0.417, 0.04000, 0.0),
        (0.417, 0.33997, 0.45773),
        (-0.350, 0.32500, 0.71128),
        (0.200, 0.35636, 1.04600),
        (0.150, 0.29500, 0.50208),
    ]
    for q, sig, eps in params:
        nb.addParticle(q if with_charges else 0.0, sig, eps)

    # Pure 1-2 exclusion (0, 1) and scaled 1-4 exception (2, 3)
    nb.addException(0, 1, 0.0, 1.0, 0.0)
    nb.addException(2, 3, -0.08 if with_charges else 0.0, 0.33, 0.35)

    if with_offset:
        nb.addGlobalParameter("lambda_vdw", 0.4)
        nb.addParticleParameterOffset("lambda_vdw", 4, 0.0, 0.02, 0.30)

    sys.addForce(nb)
    pos_nm = np.array(
        [
            [0.30, 0.40, 0.50],
            [0.39, 0.42, 0.51],
            [0.95, 0.85, 0.70],
            [1.32, 0.90, 0.78],
            [1.70, 1.50, 1.40],
            [2.15, 1.95, 2.10],
        ],
        dtype=np.float64,
    )
    return sys, pos_nm


@pytest.mark.parametrize("engine", ["dense", "neighbor_list"])
def test_prolix_context_pure_ljpme_matches_openmm_reference_tight(
    engine: NonbondedEngineType,
) -> None:
    """Zero-charge NonbondedForce.LJPME isolates pure P=6 dispersion PME to < 1e-7."""
    sys, pos_nm = _build_test_ljpme_openmm_system(explicit_grids=True, with_charges=False)
    ref_ctx = openmm.Context(
        sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    ref_ctx.setPositions(pos_nm * unit.nanometer)
    ref_state = ref_ctx.getState(getEnergy=True, getForces=True)
    e_ref = ref_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = ref_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    p_ctx = ProlixContext(
        sys,
        openmm.VerletIntegrator(0.001),
        config=ProlixOpenMMConfig(precision="double", nonbonded_engine=engine),
    )
    p_ctx.setPositions(pos_nm * unit.nanometer)
    p_state = p_ctx.getState(getEnergy=True, getForces=True)
    e_prolix = p_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_prolix = p_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    np.testing.assert_allclose(e_prolix, e_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(f_prolix, f_ref, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("explicit_grids", [True, False])
@pytest.mark.parametrize("engine", ["dense", "neighbor_list", "morton_tile_pairs"])
def test_prolix_context_ljpme_matches_openmm_reference(
    explicit_grids: bool,
    engine: NonbondedEngineType,
) -> None:
    sys, pos_nm = _build_test_ljpme_openmm_system(explicit_grids=explicit_grids)
    ref_ctx = openmm.Context(
        sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    ref_ctx.setPositions(pos_nm * unit.nanometer)
    ref_state = ref_ctx.getState(getEnergy=True, getForces=True)
    e_ref = ref_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = ref_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    prec: PrecisionType = "single" if engine == "morton_tile_pairs" else "double"
    cfg = ProlixOpenMMConfig(precision=prec, nonbonded_engine=engine)
    p_ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    assert p_ctx._can_use_morton_step_runner(False, None, None, "baoab") is False
    p_ctx.setPositions(pos_nm * unit.nanometer)
    p_state = p_ctx.getState(getEnergy=True, getForces=True)
    e_prolix = p_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_prolix = p_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    rtol = 5e-4
    atol = 1e-2 if prec == "single" else 5e-3
    np.testing.assert_allclose(e_prolix, e_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(f_prolix, f_ref, rtol=rtol * 5.0, atol=atol * 5.0)


def test_prolix_config_use_lj_pme_upgrade_and_alchemical_offset_deriv() -> None:
    """Verify config.use_lj_pme upgrades PME -> LJPME and getParameterDerivatives matches FD."""
    sys_pme, pos_nm = _build_test_ljpme_openmm_system(
        method=openmm.NonbondedForce.PME, explicit_grids=False, with_offset=True
    )
    sys_ljpme, _ = _build_test_ljpme_openmm_system(
        method=openmm.NonbondedForce.LJPME, explicit_grids=False, with_offset=True
    )

    ref_ctx = openmm.Context(
        sys_ljpme, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    ref_ctx.setPositions(pos_nm * unit.nanometer)
    e_ref = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )

    # Upgrade PME system via ProlixOpenMMConfig(use_lj_pme=True)
    p_ctx = ProlixContext(
        sys_pme,
        openmm.VerletIntegrator(0.001),
        config=ProlixOpenMMConfig(precision="double", use_lj_pme=True),
    )
    p_ctx.setPositions(pos_nm * unit.nanometer)
    st = p_ctx.getState(getEnergy=True, getParameterDerivatives=True)
    e_prolix = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    np.testing.assert_allclose(e_prolix, e_ref, rtol=5e-4, atol=5e-3)

    # Check analytical dU/d(lambda_vdw) against OpenMM Reference central FD
    derivs = st.getEnergyParameterDerivatives()
    dl = 1e-4
    ref_ctx.setParameter("lambda_vdw", 0.4 + dl)
    ep = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    ref_ctx.setParameter("lambda_vdw", 0.4 - dl)
    em = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    fd_deriv = (ep - em) / (2.0 * dl)
    np.testing.assert_allclose(derivs["lambda_vdw"], fd_deriv, rtol=1e-4, atol=1e-3)

    # Also verify step(2) rollout executes cleanly under LJPME
    p_ctx.step(2)
    st2 = p_ctx.getState(getPositions=True, getEnergy=True)
    assert np.all(np.isfinite(st2.getPositions(asNumpy=True).value_in_unit(unit.nanometer)))
