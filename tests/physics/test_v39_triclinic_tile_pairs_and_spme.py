"""v3.9 Track 1 validation: Reduced-form triclinic Morton 32x32 tile pairs, SPME spread/gather,
B==3 box disambiguation, analytical custom_vjp wrt (pos, q, sig, eps, box), StableHLO export,
and EnsemblePlan triclinic + HMR (dt = 4.0 fs) rollout.
"""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path
from typing import Any, cast

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import openmm
import pytest
from openmm import unit as _omm_unit

from prolix.api.ensemble_plan import EnsemblePlan, _spread_and_gather_ljpme_order5_soa
from prolix.constants import COULOMB_CONSTANT
from prolix.export import export_context
from prolix.knob_surface import ConstraintControls, KnobSurface, NonbondedControls, PallasControls
from prolix.physics.lj_pme import (
    dispersion_charges,
    lj_pme_influence_function,
    lj_pme_reciprocal_energy,
)
from prolix.physics.pallas_bonded import _triton_gpu_available
from prolix.physics.pallas_tile_pairs import (
    build_tile_pair_neighbor_list,
    pallas_bonded_and_tile_pairs_fused,
    pallas_build_morton_tile_pairs_gpu,
    pallas_morton_tile_pairs_energy,
    pallas_morton_tile_pairs_forces,
    pallas_tile_pairs_param_grads,
    precompute_dedup_excl_edges,
    prepare_fused_bonded_buffers,
)
from prolix.physics.pbc import minimum_image_distance, wrap_positions
from prolix.physics.pme import influence_function, spme_reciprocal_energy, spme_self_energy
from prolix.physics.pme_pallas import (
    compute_pme_spatial_perm,
    gather_pallas,
    make_gather_op,
    make_spread_op,
    spread_charges_pallas,
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


def _canonical_triclinic_boxes_angstrom() -> dict[str, np.ndarray]:
    """Return all 5 canonical OpenMM reduced lower-triangular lattice boxes (in Angstrom)."""
    d_ortho = 26.0
    d_rhomb = 34.0  # cz = 34 / sqrt(2) = 24.04 A > 2 * (cutoff + skin)
    d_oct = 30.0  # cz = 30 * sqrt(6)/3 = 24.49 A > 2 * (cutoff + skin)
    return {
        "cube": np.array(
            [[d_ortho, 0.0, 0.0], [0.0, d_ortho, 0.0], [0.0, 0.0, d_ortho]],
            dtype=np.float64,
        ),
        "shear_b": np.array(
            [[27.0, 0.0, 0.0], [8.5, 26.5, 0.0], [0.0, 0.0, 26.0]],
            dtype=np.float64,
        ),
        "shear_c": np.array(
            [[28.0, 0.0, 0.0], [6.5, 26.8, 0.0], [-7.2, 6.9, 25.5]],
            dtype=np.float64,
        ),
        "rhombic_dodecahedron": np.array(
            [
                [d_rhomb, 0.0, 0.0],
                [0.0, d_rhomb, 0.0],
                [0.5 * d_rhomb, 0.5 * d_rhomb, math.sqrt(0.5) * d_rhomb],
            ],
            dtype=np.float64,
        ),
        "truncated_octahedron": np.array(
            [
                [d_oct, 0.0, 0.0],
                [d_oct / 3.0, (2.0 * math.sqrt(2.0) / 3.0) * d_oct, 0.0],
                [-d_oct / 3.0, (math.sqrt(2.0) / 3.0) * d_oct, (math.sqrt(6.0) / 3.0) * d_oct],
            ],
            dtype=np.float64,
        ),
    }


def _build_two_tile_triclinic_openmm_system(
    box_mat_a: np.ndarray,
    *,
    cutoff_a: float = 8.5,
    pme_alpha_a: float = 0.35,
    grid_dims: tuple[int, int, int] = (60, 60, 60),
    hydrogen_mass_amu: float = 1.008,
    use_ljpme: bool = False,
    seed: int = 2039,
) -> dict[str, Any]:
    """Build a neutral 40-atom (2-tile) periodic system with solute bonds/angles/torsions/1-4s + 12 rigid waters."""
    rng = np.random.default_rng(seed)
    sys_omm = openmm.System()
    box_nm = box_mat_a * 0.1
    sys_omm.setDefaultPeriodicBoxVectors(box_nm[0].tolist(), box_nm[1].tolist(), box_nm[2].tolist())

    # 4-atom solute chain (0, 1, 2, 3) with C-H bond constraint on (0, 1) + 12 rigid TIP3P waters = 40 atoms (2 tiles)
    m_c_base = 12.011
    m_h = float(hydrogen_mass_amu)
    dm_h = m_h - 1.008
    m_o_water = 15.999 - 2.0 * dm_h
    solute_masses = [m_c_base - dm_h, m_h, 14.007, 15.999]
    for m_val in solute_masses:
        sys_omm.addParticle(m_val)
    for _ in range(12):
        sys_omm.addParticle(m_o_water)
        sys_omm.addParticle(m_h)
        sys_omm.addParticle(m_h)

    # Solute H-bond constraint on (0, 1) + harmonic bonds (1, 2), (2, 3) + angles + torsion (with triclinic PBC!)
    sys_omm.addConstraint(0, 1, 0.109)
    hb = openmm.HarmonicBondForce()
    hb.setUsesPeriodicBoundaryConditions(True)
    hb.addBond(1, 2, 0.146, 260000.0)
    hb.addBond(2, 3, 0.142, 250000.0)
    sys_omm.addForce(hb)

    ha = openmm.HarmonicAngleForce()
    ha.setUsesPeriodicBoundaryConditions(True)
    ha.addAngle(0, 1, 2, 1.91, 420.0)
    ha.addAngle(1, 2, 3, 1.95, 450.0)
    sys_omm.addForce(ha)

    pt = openmm.PeriodicTorsionForce()
    pt.setUsesPeriodicBoundaryConditions(True)
    pt.addTorsion(0, 1, 2, 3, 2, 0.35, 8.5)
    sys_omm.addForce(pt)

    # Rigid water constraints for all 12 waters
    for w_idx in range(12):
        o_i = 4 + 3 * w_idx
        sys_omm.addConstraint(o_i, o_i + 1, 0.09572)
        sys_omm.addConstraint(o_i, o_i + 2, 0.09572)
        sys_omm.addConstraint(o_i + 1, o_i + 2, 0.15139)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.LJPME if use_ljpme else openmm.NonbondedForce.PME)
    nb.setExceptionsUsePeriodicBoundaryConditions(True)
    nb.setCutoffDistance((cutoff_a * 0.1) * unit.nanometer)
    nb.setEwaldErrorTolerance(1e-4)
    nb.setUseSwitchingFunction(not use_ljpme)
    if not use_ljpme:
        nb.setSwitchingDistance(((cutoff_a - 1.5) * 0.1) * unit.nanometer)
    nb.setUseDispersionCorrection(False)
    nb.setPMEParameters(pme_alpha_a * 10.0, grid_dims[0], grid_dims[1], grid_dims[2])
    if use_ljpme:
        nb.setLJPMEParameters(0.30 * 10.0, 20, 20, 20)

    # Neutral solute charges: -0.18 + 0.09 + 0.34 - 0.25 = 0.0
    solute_nb = [
        (-0.18, 0.33997, 0.45773),
        (0.09, 0.24714, 0.06569),
        (0.34, 0.32500, 0.71128),
        (-0.25, 0.29599, 0.87864),
    ]
    for q_v, sig_v, eps_v in solute_nb:
        nb.addParticle(q_v, sig_v, eps_v)
    for _ in range(12):
        nb.addParticle(-0.834, 0.31507, 0.63597)
        nb.addParticle(0.417, 0.04000, 0.0)
        nb.addParticle(0.417, 0.04000, 0.0)

    # Solute 1-2/1-3 full exclusions + 1-4 scaled exception on (0, 3)
    nb.addException(0, 1, 0.0, 1.0, 0.0)
    nb.addException(1, 2, 0.0, 1.0, 0.0)
    nb.addException(2, 3, 0.0, 1.0, 0.0)
    nb.addException(0, 2, 0.0, 1.0, 0.0)
    nb.addException(1, 3, 0.0, 1.0, 0.0)
    sig_03 = 0.5 * (solute_nb[0][1] + solute_nb[3][1])
    eps_03 = 0.5 * math.sqrt(solute_nb[0][2] * solute_nb[3][2])
    qq_03 = (solute_nb[0][0] * solute_nb[3][0]) / 1.2
    nb.addException(0, 3, qq_03, sig_03, eps_03)

    for w_idx in range(12):
        o_i = 4 + 3 * w_idx
        nb.addException(o_i, o_i + 1, 0.0, 1.0, 0.0)
        nb.addException(o_i, o_i + 2, 0.0, 1.0, 0.0)
        nb.addException(o_i + 1, o_i + 2, 0.0, 1.0, 0.0)
    sys_omm.addForce(nb)

    # Place solute near the skewed triclinic corner (fractional [0.96, 0.94, 0.95]) so bonds & pairs cross PBC faces!
    pos_a = np.zeros((40, 3), dtype=np.float64)
    c0 = np.array([0.96, 0.94, 0.95], dtype=np.float64) @ box_mat_a
    pos_a[0] = c0
    pos_a[1] = c0 + np.array([1.09, 0.0, 0.0], dtype=np.float64)
    pos_a[2] = pos_a[1] + np.array([-0.48, 1.38, 0.05], dtype=np.float64)
    pos_a[3] = pos_a[2] + np.array([0.52, 0.45, 1.24], dtype=np.float64)

    # Distribute 12 waters on a well-separated 3x2x2 fractional lattice including boundary-straddling waters
    frac_centers = []
    for ix in range(3):
        for iy in range(2):
            for iz in range(2):
                frac_centers.append(
                    [(0.12 + 0.31 * ix) % 1.0, (0.18 + 0.42 * iy) % 1.0, (0.16 + 0.44 * iz) % 1.0]
                )
    r_oh = 0.9572
    r_hh = 1.5139
    hy = math.sqrt(r_oh * r_oh - (0.5 * r_hh) ** 2)
    for w_idx in range(12):
        o_i = 4 + 3 * w_idx
        f_c = np.array(frac_centers[w_idx], dtype=np.float64) + rng.uniform(-0.015, 0.015, size=(3,))
        o_pos = f_c @ box_mat_a
        pos_a[o_i] = o_pos
        pos_a[o_i + 1] = o_pos + np.array([0.5 * r_hh, hy, 0.0], dtype=np.float64)
        pos_a[o_i + 2] = o_pos + np.array([-0.5 * r_hh, hy, 0.0], dtype=np.float64)

    # Wrap solute & whole waters cleanly or even wrap per-atom to test PBC bond/angle/torsion minimum image!
    pos_wrapped_a = np.asarray(
        wrap_positions(jnp.asarray(pos_a, dtype=jnp.float64), jnp.asarray(box_mat_a, dtype=jnp.float64))
    )
    return {
        "system": sys_omm,
        "pos_a": pos_wrapped_a,
        "pos_nm": pos_wrapped_a * 0.1,
        "box_a": box_mat_a,
        "cutoff_a": cutoff_a,
        "lj_switch_width_a": 0.0 if use_ljpme else 1.5,
        "pme_alpha_a": pme_alpha_a,
        "grid_dims": grid_dims,
    }


def _eval_openmm_reference(sys_omm: openmm.System, pos_nm: np.ndarray, box_a: np.ndarray) -> tuple[float, np.ndarray]:
    ctx = openmm.Context(
        sys_omm,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    box_nm = box_a * 0.1
    ctx.setPeriodicBoxVectors(box_nm[0].tolist(), box_nm[1].tolist(), box_nm[2].tolist())
    ctx.setPositions(pos_nm * unit.nanometer)
    st = ctx.getState(getEnergy=True, getForces=True)
    e_kcal = float(st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
    f_akma = np.asarray(
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer) / 41.84,
        dtype=np.float64,
    )
    return e_kcal, f_akma


# =============================================================================
# 1. Signature Contract & Zero-Missed-Pairs Triclinic Discovery
# =============================================================================


def test_v39_triclinic_keyword_signature_contract() -> None:
    """Verify backward-compatible keyword-only is_triclinic=False across all public tile-pair & SPME APIs."""
    expected_src = Path(__file__).resolve().parents[2] / "src"
    assert Path(inspect.getfile(pallas_morton_tile_pairs_forces)).resolve().is_relative_to(expected_src)
    for fn in (
        build_tile_pair_neighbor_list,
        pallas_build_morton_tile_pairs_gpu,
        pallas_bonded_and_tile_pairs_fused,
        pallas_morton_tile_pairs_energy,
        pallas_morton_tile_pairs_forces,
        pallas_tile_pairs_param_grads,
        compute_pme_spatial_perm,
        spread_charges_pallas,
        gather_pallas,
        make_spread_op,
        make_gather_op,
    ):
        params = inspect.signature(fn).parameters
        assert "is_triclinic" in params, f"{fn.__name__} missing is_triclinic"
        assert params["is_triclinic"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["is_triclinic"].default is False


@pytest.mark.parametrize("lattice_name", ["rhombic_dodecahedron", "truncated_octahedron", "shear_c"])
def test_triclinic_tile_pair_discovery_misses_zero_cutoff_pairs(lattice_name: str) -> None:
    """Verify 32-atom + 4-octant triclinic AABB discovery never misses a pair within cutoff + skin."""
    rng = np.random.default_rng(991)
    box_a = _canonical_triclinic_boxes_angstrom()[lattice_name]
    n_atoms = 96  # 3 Morton tiles -> 6 candidate tile pairs
    frac = rng.uniform(0.0, 1.0, size=(n_atoms, 3))
    pos = jnp.asarray(frac @ box_a, dtype=jnp.float32)
    box_j = jnp.asarray(box_a, dtype=jnp.float32)
    cutoff = 8.5
    skin = 1.5
    r_cut = cutoff + skin

    modes = [True] + ([False] if _triton_gpu_available() else [])
    for interp in modes:
        tp = build_tile_pair_neighbor_list(
            pos, box_j, cutoff=cutoff, skin=skin, is_triclinic=True, interpret=interp
        )
        n_val = int(tp.n_valid_pairs[0])
        active_tiles = {
            (int(tp.pair_ti[0, k]), int(tp.pair_tj[0, k])) for k in range(n_val)
        }
        # Exact 27-image lattice check in float64
        pos_m_np = np.asarray(pos, dtype=np.float64)[np.asarray(tp.perm[0])]
        shifts = np.array(
            [[ix, iy, iz] for ix in (-1, 0, 1) for iy in (-1, 0, 1) for iz in (-1, 0, 1)],
            dtype=np.float64,
        ) @ box_a
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                diff_all = (pos_m_np[i] - pos_m_np[j])[None, :] + shifts
                d_min = float(np.sqrt(np.min(np.sum(diff_all * diff_all, axis=-1))))
                if d_min < r_cut - 1e-4:
                    ti, tj = i // 32, j // 32
                    pair_key = (min(ti, tj), max(ti, tj))
                    assert pair_key in active_tiles, (
                        f"Missed cutoff pair ({i}, {j}) in tile pair {pair_key} at d={d_min:.4f} A "
                        f"on {lattice_name} (interpret={interp})"
                    )


# =============================================================================
# 2. Five-Box OpenMM Reference Parity (Direct + Bonded + SPME Spread/Gather)
# =============================================================================


@pytest.mark.parametrize(
    "lattice_name",
    ["cube", "shear_b", "shear_c", "rhombic_dodecahedron", "truncated_octahedron"],
)
def test_five_canonical_lattices_match_openmm_reference(lattice_name: str) -> None:
    """Verify Morton tile pairs + Pallas SPME spread/gather match OpenMM Reference across all 5 lattices."""
    box_a = _canonical_triclinic_boxes_angstrom()[lattice_name]
    fine_dims = (72, 72, 72) if lattice_name == "rhombic_dodecahedron" else (60, 60, 60)
    fx = _build_two_tile_triclinic_openmm_system(box_a, grid_dims=fine_dims)
    e_ref, f_ref = _eval_openmm_reference(fx["system"], fx["pos_nm"], box_a)

    plan = EnsemblePlan.from_openmm_systems(
        [fx["system"]],
        positions_list=[fx["pos_nm"] * unit.nanometer],
        nonbonded_engine="morton_tile_pairs",
        precision="mixed",
    )
    b0 = plan.bundles[0]
    n_act = int(b0.n_atoms)
    bonded = prepare_fused_bonded_buffers(
        1,
        bonds=b0.bond_idx,
        bond_params=b0.bond_params,
        bond_mask=b0.bond_mask,
        angles=b0.angle_idx,
        angle_params=b0.angle_params,
        angle_mask=b0.angle_mask,
        dihedrals=b0.dihedral_idx,
        dihedral_params=b0.dihedral_params,
        dihedral_mask=b0.dihedral_mask,
        exception_pairs=b0.exception_pairs,
        exception_sigmas=b0.exception_sigmas,
        exception_epsilons=b0.exception_epsilons,
        exception_chargeprods=b0.exception_chargeprods,
        exception_mask=b0.exception_mask,
    )
    dedup_excl = precompute_dedup_excl_edges(
        b0.excl_dense_indices, b0.excl_dense_scales_vdw, b0.excl_dense_scales_elec, n_act
    )
    pos_f32 = jnp.asarray(fx["pos_a"], dtype=jnp.float32)
    q_f32 = b0.charges[:n_act].astype(jnp.float32)
    sig_f32 = b0.sigmas[:n_act].astype(jnp.float32)
    eps_f32 = b0.epsilons[:n_act].astype(jnp.float32)
    msk_bool = jnp.ones((n_act,), dtype=jnp.bool_)
    box_f32 = jnp.asarray(box_a, dtype=jnp.float32)
    grid_dims = fx["grid_dims"]
    alpha = fx["pme_alpha_a"]
    g_inf = influence_function(grid_dims, box_f32, alpha, 4).astype(jnp.float32)
    n_grid = float(grid_dims[0] * grid_dims[1] * grid_dims[2])
    f_scale = -float(COULOMB_CONSTANT) * n_grid
    e_self = float(spme_self_energy(q_f32, msk_bool, alpha=alpha))

    # 1. Exact order-4 stencil parity vs pure-JAX pme.spme_reciprocal_energy at coarse (24, 24, 24)
    gd_24 = (24, 24, 24)
    g_24 = influence_function(gd_24, box_f32, alpha, 4).astype(jnp.float32)
    n_grid_24 = float(24 * 24 * 24)
    f_scale_24 = -float(COULOMB_CONSTANT) * n_grid_24
    e_rec_jax24, g_rec_jax24 = jax.value_and_grad(
        lambda p: spme_reciprocal_energy(
            p, q_f32, msk_bool, box_f32, gd_24, alpha=alpha, order=4, use_pallas_pme=False
        )
    )(pos_f32)
    f_rec_jax24 = np.asarray(-g_rec_jax24, dtype=np.float64)

    modes = [True] + ([False] if _triton_gpu_available() else [])
    for interp in modes:
        mesh_24 = spread_charges_pallas(
            pos_f32, q_f32, msk_bool, box_f32, gd_24, interpret=interp, is_triclinic=True
        )
        theta_24 = jnp.fft.irfftn(
            jnp.fft.rfftn(mesh_24, s=gd_24, axes=(-3, -2, -1)) * g_24, s=gd_24, axes=(-3, -2, -1)
        ).astype(jnp.float32)
        f_rec_pal24, _ = gather_pallas(
            pos_f32,
            q_f32,
            msk_bool,
            box_f32,
            theta_24,
            gd_24,
            interpret=interp,
            compute_dq=False,
            force_scale=f_scale_24,
            is_triclinic=True,
        )
        e_rec_pal24 = 0.5 * float(COULOMB_CONSTANT) * float(jnp.sum(mesh_24 * theta_24)) * n_grid_24
        rel_e24 = abs(e_rec_pal24 - float(e_rec_jax24)) / max(abs(float(e_rec_jax24)), 1.0)
        rel_f24 = float(
            np.linalg.norm(np.asarray(f_rec_pal24) - f_rec_jax24) / max(np.linalg.norm(f_rec_jax24), 1.0)
        )
        assert rel_e24 < 1e-5, f"{lattice_name} (interpret={interp}) 24^3 SPME rel E err {rel_e24:.3e} >= 1e-5"
        assert rel_f24 < 1e-5, f"{lattice_name} (interpret={interp}) 24^3 SPME rel F err {rel_f24:.3e} >= 1e-5"

        # 2. Full system parity vs OpenMM Reference (PmeOrder=5) on fine (60, 60, 60) mesh
        e_dir_arr, f_dir = cast(
            "tuple[jax.Array, jax.Array]",
            pallas_morton_tile_pairs_forces(
                pos_f32,
                q_f32,
                sig_f32,
                eps_f32,
                box_f32,
                atom_mask=msk_bool,
                dedup_excl=dedup_excl,
                bonded=bonded,
                n_act=n_act,
                pme_alpha=alpha,
                cutoff=fx["cutoff_a"],
                lj_switch_width=fx["lj_switch_width_a"],
                include_pme_excl_erf=True,
                is_triclinic=True,
                return_energy=True,
                interpret=interp,
            ),
        )
        perm, inv_perm = compute_pme_spatial_perm(
            pos_f32, box_f32, grid_dims, atom_mask=msk_bool, is_triclinic=True
        )
        mesh = spread_charges_pallas(
            pos_f32,
            q_f32,
            msk_bool,
            box_f32,
            grid_dims,
            atom_perm=perm,
            interpret=interp,
            is_triclinic=True,
        )
        q_hat = jnp.fft.rfftn(mesh, s=grid_dims, axes=(-3, -2, -1))
        theta = jnp.fft.irfftn(q_hat * g_inf, s=grid_dims, axes=(-3, -2, -1)).astype(jnp.float32)
        f_tot, _ = gather_pallas(
            pos_f32,
            q_f32,
            msk_bool,
            box_f32,
            theta,
            grid_dims,
            atom_perm=perm,
            atom_inv_perm=inv_perm,
            interpret=interp,
            compute_dq=False,
            force_scale=f_scale,
            force_accum=f_dir,
            is_triclinic=True,
        )
        e_rec = 0.5 * float(COULOMB_CONSTANT) * float(jnp.sum(mesh * theta)) * n_grid + e_self
        e_tot = float(e_dir_arr) + e_rec

        rel_e = abs(e_tot - e_ref) / max(abs(e_ref), 1.0)
        rel_f = float(np.linalg.norm(np.asarray(f_tot) - f_ref) / max(np.linalg.norm(f_ref), 1.0))
        assert rel_e < 1e-4, f"{lattice_name} (interpret={interp}) rel energy err {rel_e:.3e} >= 1e-4"
        assert rel_f < 1e-3, f"{lattice_name} (interpret={interp}) rel force err {rel_f:.3e} >= 1e-3"

    # Also verify make_spread_op / make_gather_op factory parity on truncated_octahedron
    if lattice_name == "truncated_octahedron":
        spread_op = make_spread_op(grid_dims, is_triclinic=True)
        gather_op = make_gather_op(grid_dims, is_triclinic=True)
        m2 = spread_op(pos_f32, q_f32, msk_bool, box_f32)
        th2 = jnp.fft.irfftn(jnp.fft.rfftn(m2, s=grid_dims) * g_inf, s=grid_dims).astype(jnp.float32)
        dp2, dq2 = gather_op(pos_f32, q_f32, msk_bool, box_f32, th2)
        assert m2.shape == grid_dims
        assert dp2.shape == pos_f32.shape
        assert dq2.shape == q_f32.shape


# =============================================================================
# 3. B == 3 (3, 3) Orthorhombic vs Single/Batched Triclinic Disambiguation
# =============================================================================


def test_b3_orthorhombic_box_vs_triclinic_disambiguation() -> None:
    """Verify B==3 with box.shape==(3, 3) defaults to 3 orthorhombic boxes, while is_triclinic=True treats (3, 3) as H."""
    rng = np.random.default_rng(303)
    pos_b = jnp.asarray(rng.uniform(2.0, 20.0, size=(3, 18, 3)), dtype=jnp.float32)
    q_b = jnp.asarray(rng.uniform(-0.4, 0.4, size=(3, 18)), dtype=jnp.float32)
    sig_b = jnp.asarray(rng.uniform(2.8, 3.4, size=(3, 18)), dtype=jnp.float32)
    eps_b = jnp.asarray(rng.uniform(0.08, 0.20, size=(3, 18)), dtype=jnp.float32)
    msk_b = jnp.ones((3, 18), dtype=jnp.bool_)

    # 3 distinct orthorhombic box vectors stacked into shape (3, 3)
    ortho_boxes_3x3 = jnp.array(
        [[24.0, 25.0, 26.0], [25.5, 26.5, 27.5], [27.0, 28.0, 29.0]],
        dtype=jnp.float32,
    )
    e_b3, f_b3 = cast(
        "tuple[jax.Array, jax.Array]",
        pallas_morton_tile_pairs_forces(
            pos_b,
            q_b,
            sig_b,
            eps_b,
            ortho_boxes_3x3,
            atom_mask=msk_b,
            cutoff=8.5,
            pme_alpha=0.34,
            return_energy=True,
            interpret=True,
        ),
    )
    grid_dims = (16, 16, 16)
    mesh_b3 = spread_charges_pallas(pos_b, q_b, msk_b, ortho_boxes_3x3, grid_dims, interpret=True)
    f_g_b3, dq_b3 = gather_pallas(pos_b, q_b, msk_b, ortho_boxes_3x3, mesh_b3, grid_dims, interpret=True)

    for b in range(3):
        e_1, f_1 = cast(
            "tuple[jax.Array, jax.Array]",
            pallas_morton_tile_pairs_forces(
                pos_b[b],
                q_b[b],
                sig_b[b],
                eps_b[b],
                ortho_boxes_3x3[b],
                atom_mask=msk_b[b],
                cutoff=8.5,
                pme_alpha=0.34,
                return_energy=True,
                interpret=True,
            ),
        )
        mesh_1 = spread_charges_pallas(pos_b[b], q_b[b], msk_b[b], ortho_boxes_3x3[b], grid_dims, interpret=True)
        f_g_1, dq_1 = gather_pallas(pos_b[b], q_b[b], msk_b[b], ortho_boxes_3x3[b], mesh_1, grid_dims, interpret=True)

        np.testing.assert_allclose(float(e_b3[b]), float(e_1), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(f_b3[b]), np.asarray(f_1), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(mesh_b3[b]), np.asarray(mesh_1), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(f_g_b3[b]), np.asarray(f_g_1), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(dq_b3[b]), np.asarray(dq_1), rtol=1e-5, atol=1e-6)

    # Shared (3, 3) triclinic box across B==3 via explicit is_triclinic=True (or (3, 3, 3) without flag)
    tri_box = jnp.asarray(_canonical_triclinic_boxes_angstrom()["rhombic_dodecahedron"], dtype=jnp.float32)
    e_tri_shared, f_tri_shared = cast(
        "tuple[jax.Array, jax.Array]",
        pallas_morton_tile_pairs_forces(
            pos_b, q_b, sig_b, eps_b, tri_box, atom_mask=msk_b, cutoff=8.5, is_triclinic=True, return_energy=True, interpret=True
        ),
    )
    e_tri_333, f_tri_333 = cast(
        "tuple[jax.Array, jax.Array]",
        pallas_morton_tile_pairs_forces(
            pos_b,
            q_b,
            sig_b,
            eps_b,
            jnp.broadcast_to(tri_box[None, :, :], (3, 3, 3)),
            atom_mask=msk_b,
            cutoff=8.5,
            return_energy=True,
            interpret=True,
        ),
    )
    e_tri_single, f_tri_single = cast(
        "tuple[jax.Array, jax.Array]",
        pallas_morton_tile_pairs_forces(
            pos_b[0], q_b[0], sig_b[0], eps_b[0], tri_box, atom_mask=msk_b[0], cutoff=8.5, return_energy=True, interpret=True
        ),
    )
    np.testing.assert_allclose(np.asarray(e_tri_shared), np.asarray(e_tri_333), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(f_tri_shared), np.asarray(f_tri_333), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(float(e_tri_shared[0]), float(e_tri_single), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(f_tri_shared[0]), np.asarray(f_tri_single), rtol=1e-5, atol=1e-5)


# =============================================================================
# 4. Analytical @jax.custom_vjp wrt (pos, q, sig, eps, box) vs Central FD
# =============================================================================


def test_triclinic_custom_vjp_all_five_inputs_and_virial_match_central_fd() -> None:
    """Verify jax.grad wrt (pos, q, sig, eps, box) and fixed-fractional virial match float64 central FD."""
    prev_x64 = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        box_a = _canonical_triclinic_boxes_angstrom()["rhombic_dodecahedron"]
        fx = _build_two_tile_triclinic_openmm_system(box_a, seed=4040)
        plan = EnsemblePlan.from_openmm_systems(
            [fx["system"]],
            positions_list=[fx["pos_nm"] * unit.nanometer],
            nonbonded_engine="morton_tile_pairs",
            precision="mixed",
        )
        b0 = plan.bundles[0]
        n_act = 16  # Solute (4 atoms, crossing PBC!) + 4 boundary-straddling waters = 16 atoms
        bonded = prepare_fused_bonded_buffers(
            1,
            bonds=b0.bond_idx,
            bond_params=b0.bond_params,
            bond_mask=b0.bond_mask,
            angles=b0.angle_idx,
            angle_params=b0.angle_params,
            angle_mask=b0.angle_mask,
            dihedrals=b0.dihedral_idx,
            dihedral_params=b0.dihedral_params,
            dihedral_mask=b0.dihedral_mask,
            exception_pairs=b0.exception_pairs[:6],
            exception_sigmas=b0.exception_sigmas[:6],
            exception_epsilons=b0.exception_epsilons[:6],
            exception_chargeprods=b0.exception_chargeprods[:6],
            exception_mask=b0.exception_mask[:6],
        )
        dedup_excl = precompute_dedup_excl_edges(
            b0.excl_dense_indices[:n_act],
            b0.excl_dense_scales_vdw[:n_act],
            b0.excl_dense_scales_elec[:n_act],
            n_act,
        )
        pos64 = jnp.asarray(fx["pos_a"][:n_act], dtype=jnp.float64)
        q64 = b0.charges[:n_act].astype(jnp.float64)
        sig64 = jnp.maximum(b0.sigmas[:n_act].astype(jnp.float64), 0.8)
        eps64 = jnp.maximum(b0.epsilons[:n_act].astype(jnp.float64), 0.05)
        msk = jnp.ones((n_act,), dtype=jnp.bool_)
        box64 = jnp.asarray(box_a, dtype=jnp.float64)

        tp_fixed = build_tile_pair_neighbor_list(
            pos64,
            box64,
            charges=q64,
            sigmas=sig64,
            epsilons=eps64,
            atom_mask=msk,
            dedup_excl=dedup_excl,
            n_act=n_act,
            cutoff=8.5,
            skin=1.5,
            is_triclinic=True,
            interpret=True,
        )

        def _e_fn(p: jax.Array, q: jax.Array, s: jax.Array, ep: jax.Array, bx: jax.Array) -> jax.Array:
            return pallas_morton_tile_pairs_energy(
                p,
                q,
                s,
                ep,
                bx,
                atom_mask=msk,
                tile_state=tp_fixed,
                dedup_excl=dedup_excl,
                bonded=bonded,
                n_act=n_act,
                pme_alpha=0.35,
                cutoff=8.5,
                lj_switch_width=1.5,
                include_pme_excl_erf=True,
                is_triclinic=True,
                interpret=True,
            )

        g_pos, g_q, g_sig, g_eps, g_box = jax.grad(_e_fn, argnums=(0, 1, 2, 3, 4))(
            pos64, q64, sig64, eps64, box64
        )
        eps_fd = 1e-5

        # Central FD on a representative subset of coordinates (including boundary-crossing solute atoms 0..3)
        for ai in (0, 1, 2, 3, 5, 9):
            for xyz in range(3):
                dp = jnp.zeros_like(pos64).at[ai, xyz].set(eps_fd)
                fd = float(_e_fn(pos64 + dp, q64, sig64, eps64, box64) - _e_fn(pos64 - dp, q64, sig64, eps64, box64)) / (
                    2.0 * eps_fd
                )
                np.testing.assert_allclose(float(g_pos[ai, xyz]), fd, rtol=2e-5, atol=2e-5)

        for ai in (0, 2, 3, 4, 7):
            dq = jnp.zeros_like(q64).at[ai].set(eps_fd)
            fd_q = float(_e_fn(pos64, q64 + dq, sig64, eps64, box64) - _e_fn(pos64, q64 - dq, sig64, eps64, box64)) / (
                2.0 * eps_fd
            )
            np.testing.assert_allclose(float(g_q[ai]), fd_q, rtol=2e-5, atol=2e-5)

            ds = jnp.zeros_like(sig64).at[ai].set(eps_fd)
            fd_s = float(_e_fn(pos64, q64, sig64 + ds, eps64, box64) - _e_fn(pos64, q64, sig64 - ds, eps64, box64)) / (
                2.0 * eps_fd
            )
            np.testing.assert_allclose(float(g_sig[ai]), fd_s, rtol=2e-5, atol=2e-5)

            de = jnp.zeros_like(eps64).at[ai].set(eps_fd)
            fd_e = float(_e_fn(pos64, q64, sig64, eps64 + de, box64) - _e_fn(pos64, q64, sig64, eps64 - de, box64)) / (
                2.0 * eps_fd
            )
            np.testing.assert_allclose(float(g_eps[ai]), fd_e, rtol=2e-5, atol=2e-5)

        # Central FD on all 6 lower-triangular box entries (fixed Cartesian AND fixed fractional virial!)
        s_frac = pos64 @ jnp.linalg.inv(box64)

        def _e_virial(bx: jax.Array) -> jax.Array:
            return _e_fn(s_frac @ bx, q64, sig64, eps64, bx)

        g_virial = jax.grad(_e_virial)(box64)
        # Upper-triangle entries must be identically zero
        assert float(g_box[0, 1]) == 0.0
        assert float(g_box[0, 2]) == 0.0
        assert float(g_box[1, 2]) == 0.0
        for r_i, c_i in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)):
            db = jnp.zeros_like(box64).at[r_i, c_i].set(eps_fd)
            fd_box = float(_e_fn(pos64, q64, sig64, eps64, box64 + db) - _e_fn(pos64, q64, sig64, eps64, box64 - db)) / (
                2.0 * eps_fd
            )
            fd_vir = float(_e_virial(box64 + db) - _e_virial(box64 - db)) / (2.0 * eps_fd)
            np.testing.assert_allclose(float(g_box[r_i, c_i]), fd_box, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(float(g_virial[r_i, c_i]), fd_vir, rtol=2e-5, atol=2e-5)

        if _triton_gpu_available():
            tp32 = build_tile_pair_neighbor_list(
                pos64.astype(jnp.float32),
                box64.astype(jnp.float32),
                charges=q64.astype(jnp.float32),
                sigmas=sig64.astype(jnp.float32),
                epsilons=eps64.astype(jnp.float32),
                atom_mask=msk,
                dedup_excl=dedup_excl,
                n_act=n_act,
                cutoff=8.5,
                skin=1.5,
                is_triclinic=True,
                interpret=False,
            )
            gp_gpu, gq_gpu, gs_gpu, ge_gpu, gb_gpu = jax.grad(
                lambda p, q, s, ep, bx: pallas_morton_tile_pairs_energy(
                    p,
                    q,
                    s,
                    ep,
                    bx,
                    atom_mask=msk,
                    tile_state=tp32,
                    dedup_excl=dedup_excl,
                    bonded=bonded,
                    n_act=n_act,
                    pme_alpha=0.35,
                    cutoff=8.5,
                    lj_switch_width=1.5,
                    include_pme_excl_erf=True,
                    is_triclinic=True,
                    interpret=False,
                ),
                argnums=(0, 1, 2, 3, 4),
            )(
                pos64.astype(jnp.float32),
                q64.astype(jnp.float32),
                sig64.astype(jnp.float32),
                eps64.astype(jnp.float32),
                box64.astype(jnp.float32),
            )
            np.testing.assert_allclose(np.asarray(gp_gpu), np.asarray(g_pos), rtol=2e-4, atol=5e-4)
            np.testing.assert_allclose(np.asarray(gq_gpu), np.asarray(g_q), rtol=2e-4, atol=5e-4)
            np.testing.assert_allclose(np.asarray(gs_gpu), np.asarray(g_sig), rtol=2e-4, atol=5e-4)
            np.testing.assert_allclose(np.asarray(ge_gpu), np.asarray(g_eps), rtol=2e-4, atol=5e-4)
            np.testing.assert_allclose(np.asarray(gb_gpu), np.asarray(g_box), rtol=2e-4, atol=5e-4)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


# =============================================================================
# 5. Portable StableHLO Export Guardrail (Zero Custom Calls, Zero GPU Markers)
# =============================================================================


def test_triclinic_tile_pairs_and_spme_export_zero_custom_calls() -> None:
    """Verify triclinic tile-pair energy/grad and SPME spread/gather export with 0 custom_calls."""
    box_a = _canonical_triclinic_boxes_angstrom()["truncated_octahedron"]
    fx = _build_two_tile_triclinic_openmm_system(box_a, grid_dims=(16, 16, 16), seed=5050)
    plan = EnsemblePlan.from_openmm_systems(
        [fx["system"]],
        positions_list=[fx["pos_nm"] * unit.nanometer],
        nonbonded_engine="morton_tile_pairs",
        precision="single",
    )
    b0 = plan.bundles[0]
    n_act = 16
    dedup_excl = precompute_dedup_excl_edges(
        b0.excl_dense_indices[:n_act],
        b0.excl_dense_scales_vdw[:n_act],
        b0.excl_dense_scales_elec[:n_act],
        n_act,
    )
    pos = jnp.asarray(fx["pos_a"][:n_act], dtype=jnp.float32)
    q = b0.charges[:n_act].astype(jnp.float32)
    sig = jnp.maximum(b0.sigmas[:n_act].astype(jnp.float32), 0.8)
    eps = jnp.maximum(b0.epsilons[:n_act].astype(jnp.float32), 0.05)
    msk = jnp.ones((n_act,), dtype=jnp.bool_)
    box = jnp.asarray(box_a, dtype=jnp.float32)
    grid_dims = (16, 16, 16)
    g_inf = influence_function(grid_dims, box, 0.35, 4).astype(jnp.float32)

    with export_context():
        tp = build_tile_pair_neighbor_list(
            pos,
            box,
            charges=q,
            sigmas=sig,
            epsilons=eps,
            atom_mask=msk,
            dedup_excl=dedup_excl,
            n_act=n_act,
            cutoff=8.5,
            skin=1.5,
            is_triclinic=True,
        )

        def _exported_fn(
            p_in: jax.Array, q_in: jax.Array, s_in: jax.Array, ep_in: jax.Array, bx_in: jax.Array
        ) -> tuple[jax.Array, tuple[jax.Array, ...], jax.Array]:
            val_e, grads = jax.value_and_grad(
                lambda p_, q_, s_, e_, b_: pallas_morton_tile_pairs_energy(
                    p_,
                    q_,
                    s_,
                    e_,
                    b_,
                    atom_mask=msk,
                    tile_state=tp,
                    dedup_excl=dedup_excl,
                    n_act=n_act,
                    pme_alpha=0.35,
                    cutoff=8.5,
                    include_pme_excl_erf=True,
                    is_triclinic=True,
                ),
                argnums=(0, 1, 2, 3, 4),
            )(p_in, q_in, s_in, ep_in, bx_in)
            mesh = spread_charges_pallas(p_in, q_in, msk, bx_in, grid_dims, is_triclinic=True)
            theta = jnp.fft.irfftn(jnp.fft.rfftn(mesh, s=grid_dims) * g_inf, s=grid_dims).astype(jnp.float32)
            f_rec, _ = gather_pallas(
                p_in, q_in, msk, bx_in, theta, grid_dims, compute_dq=False, force_scale=-332.0637 * 4096.0, is_triclinic=True
            )
            return val_e, grads, f_rec

        lowered = jax.jit(_exported_fn).lower(pos, q, sig, eps, box)
        mlir_text = lowered.as_text()
        exported = jax.export.export(jax.jit(_exported_fn))(pos, q, sig, eps, box)

    non_loc = "\n".join(line.split(" loc(")[0] for line in exported.mlir_module().splitlines()).lower()
    non_loc_lowered = "\n".join(line.split(" loc(")[0] for line in mlir_text.splitlines()).lower()
    for tok in FORBIDDEN_STABLEHLO_SUBSTRINGS:
        assert tok not in non_loc_lowered, f"Forbidden marker {tok!r} in lowered HLO"
        assert tok not in non_loc, f"Forbidden marker {tok!r} in exported StableHLO"
    targets = set(_CUSTOM_CALL_RE.findall(mlir_text)) | set(_TARGET_NAME_RE.findall(mlir_text))
    assert not targets, f"Unexpected custom_call targets: {targets}"

    val_e, grads, f_rec = exported.call(pos, q, sig, eps, box)
    assert bool(jnp.isfinite(val_e))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
    assert bool(jnp.all(jnp.isfinite(f_rec)))


# =============================================================================
# 6. EnsemblePlan Triclinic + LJPME + HMR (m_H = 3.0 amu, dt = 4.0 fs) Rollout
# =============================================================================


def test_triclinic_order5_ljpme_spread_gather_matches_autodiff() -> None:
    """Verify _spread_and_gather_ljpme_order5_soa with is_triclinic=True matches lj_pme_reciprocal_energy."""
    box_a = _canonical_triclinic_boxes_angstrom()["truncated_octahedron"]
    fx = _build_two_tile_triclinic_openmm_system(box_a, use_ljpme=True, seed=6060)
    plan = EnsemblePlan.from_openmm_systems(
        [fx["system"]], positions_list=[fx["pos_nm"] * unit.nanometer], nonbonded_engine="morton_tile_pairs"
    )
    b0 = plan.bundles[0]
    n_act = int(b0.n_atoms)
    n_pad = 64
    lj_dims = (20, 20, 20)
    alpha_lj = 0.30
    pos_pad = jnp.pad(jnp.asarray(fx["pos_a"], dtype=jnp.float32), ((0, n_pad - n_act), (0, 0)))
    sig_pad = jnp.pad(b0.sigmas[:n_act].astype(jnp.float32), (0, n_pad - n_act), constant_values=1.0)
    eps_pad = jnp.pad(b0.epsilons[:n_act].astype(jnp.float32), (0, n_pad - n_act))
    msk_pad = jnp.arange(n_pad, dtype=jnp.int32) < n_act
    box_b = jnp.asarray(box_a, dtype=jnp.float32)[None, :, :]
    b_disp = dispersion_charges(sig_pad, eps_pad, msk_pad)[None, :]
    g6_b = lj_pme_influence_function(lj_dims, box_b[0], alpha_lj, order=5).astype(jnp.float32)[None, ...]

    e_soa, fx_s, fy_s, fz_s = _spread_and_gather_ljpme_order5_soa(
        1,
        n_pad,
        pos_pad[:, 0],
        pos_pad[:, 1],
        pos_pad[:, 2],
        b_disp.reshape(-1),
        msk_pad.astype(jnp.int32),
        box_b,
        g6_b,
        lj_dims,
        is_triclinic=True,
    )
    f_soa = jnp.stack([fx_s, fy_s, fz_s], axis=-1)[:n_act]

    e_ref, g_ref = jax.value_and_grad(
        lambda p: lj_pme_reciprocal_energy(
            p,
            b_disp[0, :n_act],
            jnp.ones((n_act,), dtype=bool),
            box_b[0],
            lj_dims,
            alpha_lj=alpha_lj,
            order=5,
        )
    )(pos_pad[:n_act])
    np.testing.assert_allclose(float(e_soa[0]), float(e_ref), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(f_soa), np.asarray(-g_ref), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("use_ljpme", [False, True])
def test_ensemble_plan_triclinic_hmr_dt4fs_nvt_and_npt_rollout(use_ljpme: bool) -> None:
    """Verify EnsemblePlan Morton SoA runner supports triclinic boxes + HMR (m_H=3.0 amu, dt=4.0 fs) in NVT & NPT."""
    box_a = _canonical_triclinic_boxes_angstrom()["rhombic_dodecahedron"]
    fx = _build_two_tile_triclinic_openmm_system(
        box_a, grid_dims=(48, 48, 48), hydrogen_mass_amu=3.0, use_ljpme=use_ljpme, seed=7070
    )
    e_ref, f_ref = _eval_openmm_reference(fx["system"], fx["pos_nm"], box_a)

    ks = KnobSurface(
        nonbonded=NonbondedControls(
            nonbonded_method="lj_pme" if use_ljpme else "pme",
            cutoff_angstrom=fx["cutoff_a"],
            pme_alpha=fx["pme_alpha_a"],
            pme_grid_dims=fx["grid_dims"],
            dispersion_correction=False,
            use_lj_pme=use_ljpme,
        ),
        constraints=ConstraintControls(r_oh_angstrom=0.9572, r_hh_angstrom=1.5139),
        pallas=PallasControls(nonbonded_engine="morton_tile_pairs", precision="mixed"),
    )
    plan = EnsemblePlan.from_openmm_systems(
        [fx["system"], fx["system"]],
        positions_list=[fx["pos_nm"] * unit.nanometer, fx["pos_nm"] * unit.nanometer],
        knob_surface=ks,
    )
    if use_ljpme:
        plan._explicit_lj_pme_alpha = 0.30
        plan._lj_pme_grid_dims = (20, 20, 20)

    assert plan._can_use_morton_soa_runner(plan.bundles, use_neighbor_list=True, use_flash_forces=False)
    entry = plan._build_morton_soa_runner(
        plan.bundles,
        integration_prefix=None,
        gamma=1.0,
        nl_update_every=10,
        nb_cfg=None,
    )
    assert entry["is_triclinic"] is True
    w_roh, w_rhh, w_mo, w_mh = entry["water_settle_params"]
    np.testing.assert_allclose(w_roh, 0.9572, rtol=1e-6)
    np.testing.assert_allclose(w_rhh, 1.5139, rtol=1e-6)
    np.testing.assert_allclose(w_mh, 3.0, rtol=1e-6)
    np.testing.assert_allclose(w_mo, 15.999 - 2.0 * (3.0 - 1.008), rtol=1e-6)

    # Verify initial Morton SoA energy & forces match OpenMM Reference
    tp0 = entry["tp0"]
    n_act = entry["n_act"]
    n_act_pad = entry["n_act_pad"]
    pos_act = jnp.pad(
        jnp.stack([b.positions[:n_act].astype(jnp.float32) for b in plan.bundles], axis=0),
        ((0, 0), (0, n_act_pad - n_act), (0, 0)),
    )
    pos_m = jnp.take_along_axis(pos_act, tp0.perm[:, :, None], axis=1)
    e_m, fx_m, fy_m, fz_m = entry["eval_morton_forces"](
        pos_m[:, :, 0].reshape(-1),
        pos_m[:, :, 1].reshape(-1),
        pos_m[:, :, 2].reshape(-1),
        tp0,
        entry["box_init_b"],
        entry["G_init"],
        entry["G6_init"],
        compute_energy=True,
    )
    f_m_b = jnp.stack([fx_m.reshape(2, n_act_pad), fy_m.reshape(2, n_act_pad), fz_m.reshape(2, n_act_pad)], axis=-1)
    f_unperm = jnp.take_along_axis(f_m_b, tp0.inv_perm[:, :n_act, None], axis=1)
    np.testing.assert_allclose(float(e_m[0]), e_ref, rtol=3e-4, atol=5e-3)
    np.testing.assert_allclose(np.asarray(f_unperm[0]), f_ref, rtol=1e-3, atol=2e-2)

    # Multi-step NVT rollout at dt = 4.0 fs with HMR (m_H = 3.0 amu) across skin-check chunks
    trajs_nvt = cast(
        "list[Any]",
        plan.run(n_steps=8, dt=4.0, kT=0.596, seed=123, nl_update_every=4, run_mode="inference"),
    )
    assert len(trajs_nvt) == 2
    box_j64 = jnp.asarray(box_a, dtype=jnp.float64)
    for tr in trajs_nvt:
        p_fin = jnp.asarray(tr.positions[0], dtype=jnp.float64)
        assert p_fin.shape == (40, 3)
        assert bool(jnp.all(jnp.isfinite(p_fin)))
        # Solute H-bond constraint (0, 1) == 1.09 A under triclinic MIC
        d_ch = float(minimum_image_distance(p_fin[1], p_fin[0], box_j64))
        np.testing.assert_allclose(d_ch, 1.09, atol=1e-4)
        # All 12 rigid waters conserve O-H (0.9572 A) and H-H (1.5139 A) under triclinic MIC at dt=4.0 fs!
        for w_idx in range(12):
            o_i = 4 + 3 * w_idx
            d_oh1 = float(minimum_image_distance(p_fin[o_i + 1], p_fin[o_i], box_j64))
            d_oh2 = float(minimum_image_distance(p_fin[o_i + 2], p_fin[o_i], box_j64))
            d_hh = float(minimum_image_distance(p_fin[o_i + 2], p_fin[o_i + 1], box_j64))
            np.testing.assert_allclose(d_oh1, 0.9572, atol=1e-4)
            np.testing.assert_allclose(d_oh2, 0.9572, atol=1e-4)
            np.testing.assert_allclose(d_hh, 1.5139, atol=1e-4)

    # Triclinic isotropic MC barostat NPT rollout at dt = 4.0 fs
    if not use_ljpme:
        plan_npt = EnsemblePlan.from_openmm_systems(
            [fx["system"]],
            positions_list=[fx["pos_nm"] * unit.nanometer],
            pressure_atm=1.0,
            barostat_interval=4,
            knob_surface=ks,
        )
        traj_npt_raw = plan_npt.run(n_steps=8, dt=4.0, kT=0.596, seed=77, nl_update_every=4, run_mode="inference")
        traj_npt = cast("Any", traj_npt_raw[0] if isinstance(traj_npt_raw, list) else traj_npt_raw)
        p_npt = np.asarray(traj_npt.positions[0], dtype=np.float64)
        box_npt = np.asarray(traj_npt.observable_values["box"], dtype=np.float64)
        assert np.all(np.isfinite(p_npt))
        assert box_npt.shape == (3, 3)
        # Isotropic scaling preserves exact lower-triangular aspect ratios of rhombic dodecahedron
        scale_est = float(box_npt[0, 0] / box_a[0, 0])
        assert 0.95 < scale_est < 1.05
        np.testing.assert_allclose(box_npt, box_a * scale_est, rtol=1e-6, atol=1e-6)
