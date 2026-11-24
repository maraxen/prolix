"""Track 2 (v3.9 Complete): Triclinic FltFlt BAOAB SETTLE+RATTLE & Cross-Shape 1D Ragged Nonbonded/SPME."""

from __future__ import annotations

import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.constants import COULOMB_CONSTANT
from prolix.export import export_context
from prolix.physics import pbc as pbc_mod
from prolix.physics.fltflt_pallas import FltFlt, ff_from_f32
from prolix.physics.pallas_ragged_queue import (
    RaggedMortonTilePairs,
    build_ragged_morton_tile_pairs,
    gather_pallas_ragged,
    pallas_nonbonded_tile_pairs_ragged,
    plan_pme_tiers,
    spme_energy_and_forces_ragged_tiered,
    spme_energy_ragged_tiered,
    spme_forces_ragged_tiered,
    spread_charges_pallas_ragged,
)
from prolix.physics.pme import spme_reciprocal_energy, spme_self_energy
from prolix.physics.settle import (
    TIP3P_RHH,
    TIP3P_ROH,
    _ff_choose,
    _lattice_shift_to_ref_ff,
    _min_image_ff,
    _normalize_box_batch,
    _pallas_baoab_post_soa,
    _wrap_ff,
    build_baoab_soa_static_plan,
    ff_double_rt,
    pallas_baoab_interstep,
    pallas_baoab_post_force,
    pallas_baoab_pre_force,
    settle_langevin,
)

pytestmark = pytest.mark.fast

_FORBIDDEN_EXPORT_SUBSTRINGS = (
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


def _assert_portable_stablehlo(hlo_text: str, label: str) -> None:
    ir_body = "\n".join(
        line for line in hlo_text.splitlines() if not line.lstrip().startswith("#loc")
    )
    low = ir_body.lower()
    leaked = [s for s in _FORBIDDEN_EXPORT_SUBSTRINGS if s in low]
    if leaked:
        pytest.fail(f"{label}: forbidden backend markers in StableHLO: {leaked}")
    targets = set(_CUSTOM_CALL_RE.findall(hlo_text)) | set(_TARGET_NAME_RE.findall(hlo_text))
    if targets:
        pytest.fail(f"{label}: unexpected custom_call targets in StableHLO: {sorted(targets)}")


def _rhombic_dodecahedron_box(edge: float, dtype=np.float64) -> np.ndarray:
    """Reduced OpenMM lower-triangular rhombic dodecahedron lattice matrix."""
    L = float(edge)
    return np.array(
        [
            [L, 0.0, 0.0],
            [0.0, L, 0.0],
            [0.5 * L, 0.5 * L, L / math.sqrt(2.0)],
        ],
        dtype=dtype,
    )


def _truncated_octahedron_box(edge: float, dtype=np.float64) -> np.ndarray:
    """Reduced OpenMM lower-triangular truncated octahedron lattice matrix."""
    L = float(edge)
    return np.array(
        [
            [L, 0.0, 0.0],
            [L / 3.0, (2.0 * math.sqrt(2.0) / 3.0) * L, 0.0],
            [-L / 3.0, (math.sqrt(2.0) / 3.0) * L, (math.sqrt(6.0) / 3.0) * L],
        ],
        dtype=dtype,
    )


def _split_f64_to_ff_np(vals_f64: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    v64 = np.asarray(vals_f64, dtype=np.float64)
    hi = v64.astype(np.float32)
    lo = (v64 - hi.astype(np.float64)).astype(np.float32)
    return jnp.asarray(hi, dtype=jnp.float32), jnp.asarray(lo, dtype=jnp.float32)


def _triclinic_mic_np(dr: np.ndarray, box_3x3: np.ndarray) -> np.ndarray:
    """Exact 27-image Euclidean minimum-image displacement in IEEE float64."""
    d64 = np.asarray(dr, dtype=np.float64)
    h64 = np.asarray(box_3x3, dtype=np.float64)
    orig_shape = d64.shape
    flat = d64.reshape(-1, 3)
    best = flat.copy()
    best_r2 = np.sum(best * best, axis=-1)
    for nz in (-1, 0, 1):
        for ny in (-1, 0, 1):
            for nx in (-1, 0, 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                shift = nx * h64[0] + ny * h64[1] + nz * h64[2]
                cand = flat + shift[None, :]
                r2 = np.sum(cand * cand, axis=-1)
                take = r2 < best_r2
                best = np.where(take[:, None], cand, best)
                best_r2 = np.where(take, r2, best_r2)
    return best.reshape(orig_shape)


def _wrap_brick_np(pos: np.ndarray, box_3x3: np.ndarray) -> np.ndarray:
    """Wrap coordinates sequentially z -> y -> x into [0, ax) x [0, by) x [0, cz)."""
    p = np.asarray(pos, dtype=np.float64).copy()
    h = np.asarray(box_3x3, dtype=np.float64)
    nz = np.floor(p[..., 2] / h[2, 2])
    p[..., 0] -= nz * h[2, 0]
    p[..., 1] -= nz * h[2, 1]
    p[..., 2] -= nz * h[2, 2]
    ny = np.floor(p[..., 1] / h[1, 1])
    p[..., 0] -= ny * h[1, 0]
    p[..., 1] -= ny * h[1, 1]
    nx = np.floor(p[..., 0] / h[0, 0])
    p[..., 0] -= nx * h[0, 0]
    return p


# =============================================================================
# Suite 1: ff_double_rt, _ff_choose & Triclinic FltFlt Lattice Primitives
# =============================================================================


def test_ff_double_rt_bit_exact_ties_and_regimes() -> None:
    """Verify ff_double_rt matches IEEE float64 rint to 0.0 ULP across all mantissa regimes."""
    eps30 = np.ldexp(1.0, -30)
    eps26 = np.ldexp(1.0, -26)
    test_vals: list[float] = [0.0, -0.0, 0.4999999, -0.4999999]
    for base in (-17, -4, -3, -2, -1, 0, 1, 2, 3, 4, 17, 1024):
        for delta in (-eps26, -eps30, 0.0, eps30, eps26):
            test_vals.append(float(base) + 0.5 + delta)
            test_vals.append(float(base) - 0.5 + delta)

    # Regime boundaries around 2^22, 2^23 (odd/even integer hi), and 2^24..2^42
    for exp_p in (22, 23, 24, 30, 40):
        b_val = float(1 << exp_p)
        for k in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
            for frac in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75):
                test_vals.append(b_val + k + frac)
                test_vals.append(-(b_val + k + frac))

    rng = np.random.default_rng(20261118)
    rand_small = rng.uniform(-500.0, 500.0, size=512)
    rand_mid = rng.integers(-(1 << 25), 1 << 25, size=256).astype(np.float64) + rng.choice(
        [-0.5, -0.25, 0.0, 0.25, 0.5], size=256
    )
    all_f64 = np.concatenate([np.array(test_vals, dtype=np.float64), rand_small, rand_mid])

    hi, lo = _split_f64_to_ff_np(all_f64)
    got_ff = jax.jit(lambda h, l: ff_double_rt(FltFlt(h, l), interpret=True))(hi, lo)
    got_f64 = np.asarray(got_ff.hi, dtype=np.float64) + np.asarray(got_ff.lo, dtype=np.float64)
    ref_f64 = np.rint(all_f64)
    np.testing.assert_array_equal(got_f64, ref_f64)

    # Verify signed zero preservation on +/-0.0 and +/-0.25 -> +/-0.0
    sz_in = np.array([0.0, -0.0, 0.25, -0.25], dtype=np.float64)
    sz_hi, sz_lo = _split_f64_to_ff_np(sz_in)
    sz_out = ff_double_rt(FltFlt(sz_hi, sz_lo), interpret=True)
    np.testing.assert_array_equal(np.signbit(np.asarray(sz_out.hi)), np.array([False, True, False, True]))

    # Unnormalized FltFlt inputs (|lo| > 0.5 ULP(hi))
    u_hi = jnp.array([3.0, 2.0, -3.0, -2.0], dtype=jnp.float32)
    u_lo = jnp.array([0.6, 0.5, -0.6, -0.5], dtype=jnp.float32)
    u_out = ff_double_rt(FltFlt(u_hi, u_lo), interpret=True)
    u_f64 = np.asarray(u_out.hi, dtype=np.float64) + np.asarray(u_out.lo, dtype=np.float64)
    np.testing.assert_array_equal(u_f64, np.array([4.0, 2.0, -4.0, -2.0], dtype=np.float64))

    # _ff_choose component selection & StableHLO export portability
    cond = jnp.array([True, False, True, False])
    chosen = _ff_choose(cond, FltFlt(u_hi, u_lo), ff_from_f32(jnp.zeros_like(u_hi)))
    np.testing.assert_allclose(np.asarray(chosen.hi), np.array([3.0, 0.0, -3.0, 0.0], dtype=np.float32))

    with export_context():
        exp = jax.export.export(jax.jit(lambda h, l: ff_double_rt(FltFlt(h, l)).hi))(
            jax.ShapeDtypeStruct((16,), jnp.float32),
            jax.ShapeDtypeStruct((16,), jnp.float32),
        )
    _assert_portable_stablehlo(exp.mlir_module(), "ff_double_rt_export")


def test_triclinic_fltflt_wrap_and_mic_primitives() -> None:
    """Verify _wrap_ff, _min_image_ff, and _lattice_shift_to_ref_ff on skewed triclinic cells."""
    for box_np in (_rhombic_dodecahedron_box(28.0), _truncated_octahedron_box(30.0)):
        b_f32 = box_np.astype(np.float32)
        box_spec = (
            jnp.asarray(b_f32[0, 0]),
            jnp.asarray(b_f32[1, 0]),
            jnp.asarray(b_f32[1, 1]),
            jnp.asarray(b_f32[2, 0]),
            jnp.asarray(b_f32[2, 1]),
            jnp.asarray(b_f32[2, 2]),
        )
        h_exact = np.array(
            [
                [float(b_f32[0, 0]), 0.0, 0.0],
                [float(b_f32[1, 0]), float(b_f32[1, 1]), 0.0],
                [float(b_f32[2, 0]), float(b_f32[2, 1]), float(b_f32[2, 2])],
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(77)
        pts = rng.uniform(-45.0, 75.0, size=(64, 3))
        px_h, px_l = _split_f64_to_ff_np(pts[:, 0])
        py_h, py_l = _split_f64_to_ff_np(pts[:, 1])
        pz_h, pz_l = _split_f64_to_ff_np(pts[:, 2])

        wx, wy, wz = _wrap_ff(FltFlt(px_h, px_l), FltFlt(py_h, py_l), FltFlt(pz_h, pz_l), box_spec)
        w_got = np.stack(
            [
                np.asarray(wx.hi, dtype=np.float64) + np.asarray(wx.lo, dtype=np.float64),
                np.asarray(wy.hi, dtype=np.float64) + np.asarray(wy.lo, dtype=np.float64),
                np.asarray(wz.hi, dtype=np.float64) + np.asarray(wz.lo, dtype=np.float64),
            ],
            axis=-1,
        )
        w_ref = _wrap_brick_np(pts, h_exact)
        np.testing.assert_allclose(w_got, w_ref, atol=1e-11, rtol=0.0)

        # True displacements strictly inside the inscribed sphere (< 0.5 * min(ax, by, cz))
        true_dr = rng.uniform(-6.0, 6.0, size=(64, 3))
        lattice_jumps = rng.integers(-3, 4, size=(64, 3)).astype(np.float64) @ h_exact
        raw_dr = true_dr + lattice_jumps
        dx_h, dx_l = _split_f64_to_ff_np(raw_dr[:, 0])
        dy_h, dy_l = _split_f64_to_ff_np(raw_dr[:, 1])
        dz_h, dz_l = _split_f64_to_ff_np(raw_dr[:, 2])
        mx, my, mz = _min_image_ff(FltFlt(dx_h, dx_l), FltFlt(dy_h, dy_l), FltFlt(dz_h, dz_l), box_spec)
        m_got = np.stack(
            [
                np.asarray(mx.hi, dtype=np.float64) + np.asarray(mx.lo, dtype=np.float64),
                np.asarray(my.hi, dtype=np.float64) + np.asarray(my.lo, dtype=np.float64),
                np.asarray(mz.hi, dtype=np.float64) + np.asarray(mz.lo, dtype=np.float64),
            ],
            axis=-1,
        )
        np.testing.assert_allclose(m_got, true_dr, atol=1e-11, rtol=0.0)

        # _lattice_shift_to_ref_ff brings shifted position back within true_dr of ref
        ref_pts = rng.uniform(0.0, 20.0, size=(64, 3))
        shifted_pts = ref_pts + raw_dr
        rx_h, rx_l = _split_f64_to_ff_np(ref_pts[:, 0])
        ry_h, ry_l = _split_f64_to_ff_np(ref_pts[:, 1])
        rz_h, rz_l = _split_f64_to_ff_np(ref_pts[:, 2])
        sx_h, sx_l = _split_f64_to_ff_np(shifted_pts[:, 0])
        sy_h, sy_l = _split_f64_to_ff_np(shifted_pts[:, 1])
        sz_h, sz_l = _split_f64_to_ff_np(shifted_pts[:, 2])
        shx, shy, shz = _lattice_shift_to_ref_ff(
            (FltFlt(sx_h, sx_l), FltFlt(sy_h, sy_l), FltFlt(sz_h, sz_l)),
            (FltFlt(rx_h, rx_l), FltFlt(ry_h, ry_l), FltFlt(rz_h, rz_l)),
            box_spec,
        )
        sh_got = np.stack(
            [
                np.asarray(shx.hi, dtype=np.float64) + np.asarray(shx.lo, dtype=np.float64),
                np.asarray(shy.hi, dtype=np.float64) + np.asarray(shy.lo, dtype=np.float64),
                np.asarray(shz.hi, dtype=np.float64) + np.asarray(shz.lo, dtype=np.float64),
            ],
            axis=-1,
        )
        np.testing.assert_allclose((shifted_pts + sh_got) - ref_pts, true_dr, atol=1e-11, rtol=0.0)


# =============================================================================
# Suite 2: Triclinic BAOAB SETTLE + RATTLE Holonomic Conservation & Equipartition
# =============================================================================


def _build_triclinic_water_and_star_system(box_np: np.ndarray):
    """Build rigid waters + CH/CH2/CH3/NH4+ star clusters straddling skewed triclinic boundaries."""
    half_hh = 0.5 * TIP3P_RHH
    oh_y = math.sqrt(TIP3P_ROH**2 - half_hh**2)
    h_vec = box_np

    # Place 8 waters right across a, b, c, and corner boundaries, then wrap per-atom into brick!
    anchors = [
        0.02 * h_vec[0] + 0.50 * h_vec[1] + 0.50 * h_vec[2],
        0.50 * h_vec[0] + 0.02 * h_vec[1] + 0.50 * h_vec[2],
        0.50 * h_vec[0] + 0.50 * h_vec[1] + 0.02 * h_vec[2],
        0.99 * h_vec[0] + 0.99 * h_vec[1] + 0.99 * h_vec[2],
        0.01 * h_vec[0] + 0.99 * h_vec[1] + 0.01 * h_vec[2],
        0.25 * h_vec[0] + 0.35 * h_vec[1] + 0.98 * h_vec[2],
        0.98 * h_vec[0] + 0.15 * h_vec[1] + 0.45 * h_vec[2],
        0.40 * h_vec[0] + 0.60 * h_vec[1] + 0.50 * h_vec[2],
    ]
    coords_list: list[np.ndarray] = []
    masses_list: list[float] = []
    water_idx_list: list[list[int]] = []

    rng = np.random.default_rng(1234)
    for w_i, anc in enumerate(anchors):
        # Random 3D rotation via QR so waters have general orientations across skewed faces
        q_rot, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(q_rot) < 0:
            q_rot[:, 0] = -q_rot[:, 0]
        r_o = anc
        r_h1 = anc + np.array([half_hh, oh_y, 0.0]) @ q_rot.T
        r_h2 = anc + np.array([-half_hh, oh_y, 0.0]) @ q_rot.T
        base = 3 * w_i
        coords_list.extend([r_o, r_h1, r_h2])
        masses_list.extend([15.999, 1.008, 1.008])
        water_idx_list.append([base, base + 1, base + 2])

    # Add 4 solute star clusters straddling boundaries: CH (1), CH2 (2), CH3 (3), NH4+ (4)
    c_pairs_list: list[list[int]] = []
    c_lens_list: list[float] = []
    star_Anchors = [
        (1, 12.011, 1.09, 0.995 * h_vec[2] + 0.30 * h_vec[0]),
        (2, 12.011, 1.09, 0.995 * h_vec[1] + 0.995 * h_vec[2]),
        (3, 12.011, 1.09, 0.005 * h_vec[0] + 0.005 * h_vec[1] + 0.995 * h_vec[2]),
        (4, 14.007, 1.01, 0.992 * h_vec[0] + 0.992 * h_vec[1] + 0.008 * h_vec[2]),
    ]
    tet_dirs = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    ) / math.sqrt(3.0)

    for deg, m_heavy, d_xh, s_anc in star_Anchors:
        idx_x = len(coords_list)
        coords_list.append(s_anc)
        masses_list.append(m_heavy)
        for k in range(deg):
            idx_h = len(coords_list)
            coords_list.append(s_anc + d_xh * tet_dirs[k])
            masses_list.append(1.008)
            c_pairs_list.append([idx_x, idx_h])
            c_lens_list.append(d_xh)

    # Add 4 unconstrained solute atoms + 2 masked padding atoms at the tail
    for s_i in range(4):
        coords_list.append((0.2 + 0.15 * s_i) * (h_vec[0] + h_vec[1] + h_vec[2]))
        masses_list.append(12.011)
    n_real = len(coords_list)
    for _ in range(2):
        coords_list.append(np.zeros(3, dtype=np.float64))
        masses_list.append(1.0)
    n_tot = len(coords_list)

    pos_unwrapped = np.stack(coords_list, axis=0)
    # Per-atom brick wrap intentionally splits boundary-straddling waters & star clusters!
    pos_wrapped = _wrap_brick_np(pos_unwrapped, box_np)
    pos_wrapped[n_real:] = 0.0

    atom_mask = np.zeros((n_tot,), dtype=bool)
    atom_mask[:n_real] = True
    masses = np.array(masses_list, dtype=np.float64)
    water_indices = np.array(water_idx_list, dtype=np.int32)
    water_mask = np.ones((len(water_idx_list),), dtype=bool)
    c_pairs = np.array(c_pairs_list, dtype=np.int32)
    c_lens = np.array(c_lens_list, dtype=np.float64)
    c_mask = np.ones((len(c_pairs_list),), dtype=bool)

    return pos_wrapped, masses, atom_mask, water_indices, water_mask, c_pairs, c_lens, c_mask, n_real


def _check_holonomic_and_velocity_constraints(
    pos: np.ndarray,
    mom: np.ndarray,
    masses: np.ndarray,
    box_np: np.ndarray,
    water_indices: np.ndarray,
    c_pairs: np.ndarray,
    c_lens: np.ndarray,
    *,
    pos_tol: float,
    vel_tol: float,
) -> None:
    vel = mom / masses[:, None]
    # Waters: O-H1, O-H2, H1-H2
    r_o = pos[water_indices[:, 0]]
    r_h1 = pos[water_indices[:, 1]]
    r_h2 = pos[water_indices[:, 2]]
    d_oh1 = _triclinic_mic_np(r_h1 - r_o, box_np)
    d_oh2 = _triclinic_mic_np(r_h2 - r_o, box_np)
    d_hh = _triclinic_mic_np(r_h2 - r_h1, box_np)

    err_oh1 = np.max(np.abs(np.linalg.norm(d_oh1, axis=-1) - TIP3P_ROH))
    err_oh2 = np.max(np.abs(np.linalg.norm(d_oh2, axis=-1) - TIP3P_ROH))
    err_hh = np.max(np.abs(np.linalg.norm(d_hh, axis=-1) - TIP3P_RHH))
    assert max(err_oh1, err_oh2, err_hh) < pos_tol, (
        f"SETTLE bond error exceeded {pos_tol}: OH1={err_oh1:.3e}, OH2={err_oh2:.3e}, HH={err_hh:.3e}"
    )

    v_o = vel[water_indices[:, 0]]
    v_h1 = vel[water_indices[:, 1]]
    v_h2 = vel[water_indices[:, 2]]
    u_oh1 = d_oh1 / np.linalg.norm(d_oh1, axis=-1, keepdims=True)
    u_oh2 = d_oh2 / np.linalg.norm(d_oh2, axis=-1, keepdims=True)
    u_hh = d_hh / np.linalg.norm(d_hh, axis=-1, keepdims=True)
    vproj_w = max(
        float(np.max(np.abs(np.sum((v_h1 - v_o) * u_oh1, axis=-1)))),
        float(np.max(np.abs(np.sum((v_h2 - v_o) * u_oh2, axis=-1)))),
        float(np.max(np.abs(np.sum((v_h2 - v_h1) * u_hh, axis=-1)))),
    )
    assert vproj_w < vel_tol, f"SETTLE velocity projection residual {vproj_w:.3e} >= {vel_tol}"

    # Solute star-cluster RATTLE constraints
    d_xh = _triclinic_mic_np(pos[c_pairs[:, 1]] - pos[c_pairs[:, 0]], box_np)
    err_xh = float(np.max(np.abs(np.linalg.norm(d_xh, axis=-1) - c_lens)))
    assert err_xh < pos_tol, f"RATTLE X-H bond error {err_xh:.3e} >= {pos_tol}"

    u_xh = d_xh / np.linalg.norm(d_xh, axis=-1, keepdims=True)
    v_xh = vel[c_pairs[:, 1]] - vel[c_pairs[:, 0]]
    vproj_c = float(np.max(np.abs(np.sum(v_xh * u_xh, axis=-1))))
    assert vproj_c < vel_tol, f"RATTLE velocity projection residual {vproj_c:.3e} >= {vel_tol}"


@pytest.mark.parametrize("cell_kind", ["rhombic_dodecahedron", "truncated_octahedron"])
@pytest.mark.parametrize("acc_mode", ["fltflt", "f32"])
def test_triclinic_baoab_settle_rattle_holonomic_and_interstep(
    cell_kind: str, acc_mode: str
) -> None:
    """Multi-step triclinic BAOAB (pre + post & interstep fused + SoA) conserves holonomic geometry."""
    box_np = (
        _rhombic_dodecahedron_box(26.0)
        if cell_kind == "rhombic_dodecahedron"
        else _truncated_octahedron_box(28.0)
    )
    (
        pos_init,
        masses_np,
        atom_mask_np,
        wi_np,
        wm_np,
        cp_np,
        cl_np,
        cm_np,
        _n_real,
    ) = _build_triclinic_water_and_star_system(box_np)

    dtype = jnp.float32
    pos = jnp.asarray(pos_init, dtype=dtype)
    mom = jnp.zeros_like(pos)
    masses = jnp.asarray(masses_np, dtype=dtype)
    atom_mask = jnp.asarray(atom_mask_np)
    wi = jnp.asarray(wi_np, dtype=jnp.int32)
    wm = jnp.asarray(wm_np)
    cp = jnp.asarray(cp_np, dtype=jnp.int32)
    cl = jnp.asarray(cl_np, dtype=dtype)
    cm = jnp.asarray(cm_np)
    box_j = jnp.asarray(box_np, dtype=dtype)

    dt_akma = 2.0 / 48.88821291839
    kT_akma = 0.59616
    gamma_akma = 5.0 * 0.0488882129
    rng = jax.random.PRNGKey(2026)

    # Mild harmonic tether force on wrapped MIC displacements so forces are non-zero
    ref_anchor = pos
    def _force_fn(r_cur: jax.Array) -> jax.Array:
        dr = jax.vmap(lambda a, b: pbc_mod.minimum_image_displacement(a, b, box_j))(r_cur, ref_anchor)
        return jnp.where(atom_mask[:, None], -0.25 * dr, 0.0).astype(dtype)

    frc = _force_fn(pos)
    pos_lo = jnp.zeros_like(pos) if acc_mode == "fltflt" else None
    mom_lo = jnp.zeros_like(mom) if acc_mode == "fltflt" else None

    # 12 steps using interstep fusion inside chunk + standalone post_force at chunk end
    if acc_mode == "fltflt":
        pos, mom, rng, pos_lo, mom_lo = pallas_baoab_pre_force(
            pos, mom, frc, masses, rng, wi, wm, box_j, dt_akma, kT_akma, gamma_akma,
            interpret=True, atom_mask=atom_mask, constraint_pairs=cp, constraint_lengths=cl,
            constraint_mask=cm, accumulator_mode="fltflt", positions_lo=pos_lo, momentum_lo=mom_lo,
            return_lo_accumulators=True, is_triclinic=True,
        )
        for _ in range(10):
            frc = _force_fn(pos)
            pos, mom, rng, pos_lo, mom_lo = pallas_baoab_interstep(
                pos, mom, frc, masses, rng, wi, wm, box_j, dt_akma, kT_akma, gamma_akma,
                interpret=True, atom_mask=atom_mask, constraint_pairs=cp, constraint_lengths=cl,
                constraint_mask=cm, accumulator_mode="fltflt", positions_lo=pos_lo,
                momentum_lo=mom_lo, return_lo_accumulators=True, is_triclinic=True,
            )
        frc = _force_fn(pos)
        mom, mom_lo = pallas_baoab_post_force(
            pos, mom, frc, masses, wi, wm, dt_akma,
            interpret=True, atom_mask=atom_mask, constraint_pairs=cp, constraint_lengths=cl,
            constraint_mask=cm, accumulator_mode="fltflt", positions_lo=pos_lo, momentum_lo=mom_lo,
            return_lo_accumulators=True, box=box_j, is_triclinic=True,
        )
        p_check = np.asarray(pos, dtype=np.float64) + np.asarray(pos_lo, dtype=np.float64)
        m_check = np.asarray(mom, dtype=np.float64) + np.asarray(mom_lo, dtype=np.float64)
        pos_tol = 1e-6
    else:
        pos, mom, rng = pallas_baoab_pre_force(
            pos, mom, frc, masses, rng, wi, wm, box_j, dt_akma, kT_akma, gamma_akma,
            interpret=True, atom_mask=atom_mask, constraint_pairs=cp, constraint_lengths=cl,
            constraint_mask=cm, accumulator_mode="f32", is_triclinic=True,
        )
        for _ in range(10):
            frc = _force_fn(pos)
            pos, mom, rng = pallas_baoab_interstep(
                pos, mom, frc, masses, rng, wi, wm, box_j, dt_akma, kT_akma, gamma_akma,
                interpret=True, atom_mask=atom_mask, constraint_pairs=cp, constraint_lengths=cl,
                constraint_mask=cm, accumulator_mode="f32", is_triclinic=True,
            )
        frc = _force_fn(pos)
        mom = pallas_baoab_post_force(
            pos, mom, frc, masses, wi, wm, dt_akma,
            interpret=True, atom_mask=atom_mask, constraint_pairs=cp, constraint_lengths=cl,
            constraint_mask=cm, accumulator_mode="f32", box=box_j, is_triclinic=True,
        )
        p_check = np.asarray(pos, dtype=np.float64)
        m_check = np.asarray(mom, dtype=np.float64)
        pos_tol = 5e-6

    # Padding atoms must remain identically untouched at 0
    np.testing.assert_array_equal(p_check[~atom_mask_np], 0.0)
    np.testing.assert_array_equal(m_check[~atom_mask_np], 0.0)

    _check_holonomic_and_velocity_constraints(
        p_check, m_check, masses_np, box_np, wi_np, cp_np, cl_np, pos_tol=pos_tol, vel_tol=1e-5
    )

    # Also verify _pallas_baoab_post_soa wrapper with Morton permutation & triclinic box
    N = int(pos.shape[0])
    N_pad = ((N + 31) // 32) * 32
    soa_plan = build_baoab_soa_static_plan(
        1, N, dtype, masses[None, :], wi[None, :, :], wm[None, :],
        atom_mask_b=atom_mask[None, :], constraint_pairs_b=cp[None, :, :],
        constraint_lengths_b=cl[None, :], constraint_mask_b=cm[None, :],
    )
    inv_perm_b = jnp.arange(N, dtype=jnp.int32)[None, :]
    frc_pad = jnp.pad(frc, ((0, N_pad - N), (0, 0)))
    mx_s, my_s, mz_s = _pallas_baoab_post_soa(
        pos[:, 0], pos[:, 1], pos[:, 2],
        mom[:, 0], mom[:, 1], mom[:, 2],
        frc_pad[:, 0], frc_pad[:, 1], frc_pad[:, 2],
        inv_perm_b, N_pad, jnp.asarray([dt_akma], dtype=dtype),
        15.999, 1.008, True, soa_plan,
        accumulator_mode=acc_mode, box=box_j[None, :, :], is_triclinic=True,
    )
    mom_soa = np.stack([np.asarray(mx_s), np.asarray(my_s), np.asarray(mz_s)], axis=-1).astype(np.float64)
    _check_holonomic_and_velocity_constraints(
        p_check, mom_soa, masses_np, box_np, wi_np, cp_np, cl_np, pos_tol=pos_tol, vel_tol=1e-5
    )


def test_triclinic_settle_langevin_equipartition_and_b3_disambiguation() -> None:
    """Verify settle_langevin Pallas fast-path on triclinic box (equipartition + holonomic) & B=3 disambiguation."""
    box_np = _rhombic_dodecahedron_box(26.0)
    (
        pos_init,
        masses_np,
        atom_mask_np,
        wi_np,
        wm_np,
        cp_np,
        cl_np,
        cm_np,
        n_real,
    ) = _build_triclinic_water_and_star_system(box_np)

    dtype = jnp.float32
    box_j = jnp.asarray(box_np, dtype=dtype)
    masses_j = jnp.asarray(masses_np, dtype=dtype)
    atom_mask_j = jnp.asarray(atom_mask_np)
    wi_j = jnp.asarray(wi_np, dtype=jnp.int32)
    wm_j = jnp.asarray(wm_np)
    cp_j = jnp.asarray(cp_np, dtype=jnp.int32)
    cl_j = jnp.asarray(cl_np, dtype=dtype)
    cm_j = jnp.asarray(cm_np)

    _, shift_fn = pbc_mod.create_periodic_space(box_j)
    dt_akma = 1.0 / 48.88821291839
    target_kT = 0.59616  # ~300 K in kcal/mol
    gamma_akma = 20.0 * 0.0488882129  # Strong friction for fast thermalization in unit test

    def zero_force_fn(r: jax.Array, **_kw) -> jax.Array:
        return jnp.zeros_like(r)

    init_fn, apply_fn = settle_langevin(
        zero_force_fn,
        shift_fn,
        dt=dt_akma,
        kT=target_kT,
        gamma=gamma_akma,
        mass=masses_j,
        water_indices=wi_j,
        box=box_j,
        constraints=(cp_j, cl_j, cm_j),
        water_mask=wm_j,
        atom_mask=atom_mask_j,
        is_force_shaped=True,
        use_pallas=True,
        accumulator_mode="f32",
    )
    assert apply_fn.__dict__.get("_pallas_settle_pins_ghosts") is True

    state = init_fn(jax.random.PRNGKey(99), jnp.asarray(pos_init, dtype=dtype))

    @jax.jit
    def _run_n_steps(st, n_steps: jax.Array):
        def _body(_, s):
            return apply_fn(s)
        return jax.lax.fori_loop(jnp.int32(0), n_steps, _body, st)

    # Equilibrate 60 steps, then sample 8 chunks of 15 steps using a single compiled loop
    state = _run_n_steps(state, jnp.int32(60))
    n_dof = 3 * n_real - 3 * int(wi_np.shape[0]) - int(cp_np.shape[0])
    kTs: list[float] = []
    for _ in range(8):
        state = _run_n_steps(state, jnp.int32(15))
        mom_np = np.asarray(state.momentum, dtype=np.float64)[:n_real]
        ke = 0.5 * np.sum((mom_np * mom_np) / masses_np[:n_real, None])
        kTs.append(float(2.0 * ke / float(n_dof)))

    mean_kT = float(np.mean(kTs))
    assert abs(mean_kT - target_kT) < 0.12, f"Equipartition kT={mean_kT:.4f} far from {target_kT:.4f}"
    pos_f32_np = np.asarray(state.positions, dtype=np.float64)
    mom_f32_np = np.asarray(state.momentum, dtype=np.float64)
    np.testing.assert_array_equal(pos_f32_np[~atom_mask_np], 0.0)
    np.testing.assert_array_equal(mom_f32_np[~atom_mask_np], 0.0)
    _check_holonomic_and_velocity_constraints(
        pos_f32_np,
        mom_f32_np,
        masses_np,
        box_np,
        wi_np,
        cp_np,
        cl_np,
        pos_tol=5e-6,
        vel_tol=1e-5,
    )

    # Also verify settle_langevin(..., use_pallas=True, accumulator_mode="fltflt") eagerly on CPU
    # (avoiding outer @jax.jit CPU LLVM codegen across ~3,500 Veltkamp optimization_barrier ops)
    init_ff, apply_ff = settle_langevin(
        zero_force_fn,
        shift_fn,
        dt=dt_akma,
        kT=target_kT,
        gamma=gamma_akma,
        mass=masses_j,
        water_indices=wi_j,
        box=box_j,
        constraints=(cp_j, cl_j, cm_j),
        water_mask=wm_j,
        atom_mask=atom_mask_j,
        is_force_shaped=True,
        use_pallas=True,
        accumulator_mode="fltflt",
    )
    assert apply_ff.__dict__.get("_pallas_settle_pins_ghosts") is True
    state_ff = init_ff(jax.random.PRNGKey(101), jnp.asarray(pos_init, dtype=dtype))
    state_ff = apply_ff(apply_ff(state_ff))
    pos_ff_np = np.asarray(state_ff.positions, dtype=np.float64)
    mom_ff_np = np.asarray(state_ff.momentum, dtype=np.float64)
    np.testing.assert_array_equal(pos_ff_np[~atom_mask_np], 0.0)
    np.testing.assert_array_equal(mom_ff_np[~atom_mask_np], 0.0)
    _check_holonomic_and_velocity_constraints(
        pos_ff_np,
        mom_ff_np,
        masses_np,
        box_np,
        wi_np,
        cp_np,
        cl_np,
        pos_tol=2e-6,
        vel_tol=1e-5,
    )

    # Verify B == 3 (3, 3) orthorhombic vs (B=3, 3, 3) triclinic disambiguation
    ortho_b3 = jnp.array([[24.0, 25.0, 26.0], [27.0, 28.0, 29.0], [30.0, 31.0, 32.0]], dtype=jnp.float32)
    bx_norm_o, bx_flat_o = _normalize_box_batch(3, True, ortho_b3, jnp.float32, is_triclinic=False)
    assert bx_norm_o.shape == (3, 3)
    assert bx_flat_o.shape == (9,)
    np.testing.assert_allclose(np.asarray(bx_norm_o), np.asarray(ortho_b3))

    tri_b3 = jnp.stack(
        [
            jnp.diag(jnp.array([26.0, 26.0, 26.0], dtype=jnp.float32)),
            jnp.asarray(_rhombic_dodecahedron_box(26.0), dtype=jnp.float32),
            jnp.asarray(_truncated_octahedron_box(28.0), dtype=jnp.float32),
        ],
        axis=0,
    )
    bx_norm_t, bx_flat_t = _normalize_box_batch(3, True, tri_b3, jnp.float32, is_triclinic=False)
    assert bx_norm_t.shape == (3, 3, 3)
    assert bx_flat_t.shape == (27,)

    # Portable StableHLO export guardrail for triclinic BAOAB pre & post
    with export_context():
        exp_pre = jax.export.export(
            jax.jit(
                lambda r, p, f: pallas_baoab_pre_force(
                    r, p, f, masses_j, jax.random.PRNGKey(1), wi_j, wm_j, box_j,
                    dt_akma, target_kT, gamma_akma, atom_mask=atom_mask_j,
                    accumulator_mode="fltflt", is_triclinic=True,
                )[0]
            )
        )(state.positions, state.momentum, state.force)
        exp_post = jax.export.export(
            jax.jit(
                lambda r, p, f: pallas_baoab_post_force(
                    r, p, f, masses_j, wi_j, wm_j, dt_akma,
                    atom_mask=atom_mask_j, accumulator_mode="fltflt", box=box_j, is_triclinic=True,
                )
            )
        )(state.positions, state.momentum, state.force)
    _assert_portable_stablehlo(exp_pre.mlir_module(), "triclinic_baoab_pre_export")
    _assert_portable_stablehlo(exp_post.mlir_module(), "triclinic_baoab_post_export")


# =============================================================================
# Suite 3: Cross-Size, Cross-Shape 1D Ragged Batching (Direct Tile-Pairs & SPME)
# =============================================================================


def _exact_27_image_nonbonded_f64_ref(
    pos_b: np.ndarray,
    q_b: np.ndarray,
    sig_b: np.ndarray,
    eps_b: np.ndarray,
    mask_b: np.ndarray,
    boxes_3x3: np.ndarray,
    excl_idx_b: np.ndarray,
    excl_sv_b: np.ndarray,
    excl_se_b: np.ndarray,
    *,
    cutoff: float,
    pme_alpha: float,
    include_pme_excl_erf: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent IEEE float64 27-image Bravais lattice sum reference for cross-shape batches."""
    from scipy.special import erf as sp_erf
    from scipy.special import erfc as sp_erfc

    B, N, _ = pos_b.shape
    e_out = np.zeros((B,), dtype=np.float64)
    f_out = np.zeros((B, N, 3), dtype=np.float64)
    two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)

    for b in range(B):
        h_mat = np.asarray(boxes_3x3[b], dtype=np.float64)
        pos = _wrap_brick_np(np.asarray(pos_b[b], dtype=np.float64), h_mat)
        q = np.asarray(q_b[b], dtype=np.float64)
        sig = np.asarray(sig_b[b], dtype=np.float64)
        eps = np.asarray(eps_b[b], dtype=np.float64)
        msk = np.asarray(mask_b[b], dtype=bool)

        scale_map: dict[tuple[int, int], tuple[float, float]] = {}
        idx_1 = excl_idx_b[b]
        sv_1 = excl_sv_b[b]
        se_1 = excl_se_b[b]
        for u in range(idx_1.shape[0]):
            if not msk[u]:
                continue
            for slot in range(idx_1.shape[1]):
                v = int(idx_1[u, slot])
                if 0 <= v < N and v != u and msk[v]:
                    key = (min(u, v), max(u, v))
                    scale_map[key] = (float(sv_1[u, slot]), float(se_1[u, slot]))

        act_atoms = np.nonzero(msk)[0]
        for ii in range(act_atoms.size):
            i = int(act_atoms[ii])
            for jj in range(ii + 1, act_atoms.size):
                j = int(act_atoms[jj])
                dr = _triclinic_mic_np(pos[i] - pos[j], h_mat)
                r2 = float(np.dot(dr, dr))
                if r2 >= cutoff * cutoff:
                    continue
                sv, se = scale_map.get((i, j), (1.0, 1.0))
                is_full = sv <= 1e-6 and se <= 1e-6
                if is_full and not include_pme_excl_erf:
                    continue

                dist = math.sqrt(r2 + 1e-12)
                inv_d = 1.0 / dist
                inv_r2 = inv_d * inv_d

                e_lj = 0.0
                de_lj_dr = 0.0
                if not is_full and sv > 0.0:
                    sig_ij = 0.5 * (sig[i] + sig[j])
                    eps_ij = math.sqrt(max(eps[i] * eps[j], 0.0))
                    if eps_ij > 0.0:
                        sr2 = (sig_ij * sig_ij) / (r2 + 1e-12)
                        sr6 = sr2 * sr2 * sr2
                        e_lj = 4.0 * eps_ij * (sr6 * sr6 - sr6) * sv
                        de_lj_dr = 4.0 * eps_ij * (-12.0 * sr6 * sr6 + 6.0 * sr6) * inv_d * sv

                qq = COULOMB_CONSTANT * q[i] * q[j]
                ar = pme_alpha * dist
                g_exp = pme_alpha * two_over_sqrt_pi * math.exp(-ar * ar)
                if is_full:
                    e_coul = -qq * float(sp_erf(ar)) * inv_d
                    de_coul_dr = qq * (float(sp_erf(ar)) * inv_r2 - g_exp * inv_d)
                elif se < 1.0 - 1e-6:
                    # 1-4 scaled PME direct-space term: se * erfc(a r)/r - (1 - se) * erf(a r)/r
                    om_se = 1.0 - se
                    e_coul = qq * (se * float(sp_erfc(ar)) - om_se * float(sp_erf(ar))) * inv_d
                    de_coul_dr = qq * (
                        -se * (float(sp_erfc(ar)) * inv_r2 + g_exp * inv_d)
                        + om_se * (float(sp_erf(ar)) * inv_r2 - g_exp * inv_d)
                    )
                else:
                    e_coul = qq * float(sp_erfc(ar)) * inv_d
                    de_coul_dr = -qq * (float(sp_erfc(ar)) * inv_r2 + g_exp * inv_d)

                e_out[b] += e_lj + e_coul
                f_vec = -(de_lj_dr + de_coul_dr) * inv_d * dr
                f_out[b, i] += f_vec
                f_out[b, j] -= f_vec

    return e_out, f_out


def _make_cross_shape_ragged_batch():
    """Create B=3 ragged batch [cubic, rhombic_dodecahedron, truncated_octahedron] with corner pairs."""
    B = 3
    n_reals = [24, 45, 66]
    n_pad = 96
    boxes_np = np.stack(
        [
            np.diag([32.0, 32.0, 32.0]),
            _rhombic_dodecahedron_box(32.0),
            _truncated_octahedron_box(32.0),
        ],
        axis=0,
    )
    rng = np.random.default_rng(20260309)
    pos_np = np.zeros((B, n_pad, 3), dtype=np.float64)
    q_np = np.zeros((B, n_pad), dtype=np.float64)
    sig_np = np.ones((B, n_pad), dtype=np.float64) * 3.1
    eps_np = np.zeros((B, n_pad), dtype=np.float64)
    mask_np = np.zeros((B, n_pad), dtype=bool)
    excl_idx = np.full((B, n_pad, 4), -1, dtype=np.int32)
    excl_sv = np.ones((B, n_pad, 4), dtype=np.float32)
    excl_se = np.ones((B, n_pad, 4), dtype=np.float32)

    for b, nr in enumerate(n_reals):
        h_b = boxes_np[b]
        mask_np[b, :nr] = True
        # First 8 atoms placed near opposite skewed faces/corners (~3.0-4.5 A apart across c & b tilts!)
        corner_fracs = np.array(
            [
                [0.04, 0.04, 0.04],
                [0.95, 0.95, 0.95],
                [0.04, 0.50, 0.95],
                [0.50, 0.04, 0.95],
                [0.95, 0.05, 0.05],
                [0.05, 0.95, 0.05],
                [0.47, 0.53, 0.04],
                [0.53, 0.47, 0.95],
            ],
            dtype=np.float64,
        )
        placed_list: list[np.ndarray] = [cf @ h_b for cf in corner_fracs]
        while len(placed_list) < nr:
            cand_f = rng.uniform(0.06, 0.94, size=(3,))
            cand_p = cand_f @ h_b
            prev_arr = np.stack(placed_list, axis=0)
            drs = _triclinic_mic_np(prev_arr - cand_p[None, :], h_b)
            if float(np.min(np.sum(drs * drs, axis=-1))) >= 2.75 * 2.75:
                placed_list.append(cand_p)
        base_pos = np.stack(placed_list, axis=0)
        # Shift atoms by +/-1 lattice vectors so input coords also test pre-wrapping across >1 cell!
        jumps = rng.integers(-1, 2, size=(nr, 3)).astype(np.float64)
        pos_np[b, :nr] = base_pos + jumps @ h_b

        raw_q = rng.uniform(-0.65, 0.65, size=(nr,))
        raw_q -= np.mean(raw_q)  # Charge-neutral per replica
        q_np[b, :nr] = raw_q
        sig_np[b, :nr] = rng.uniform(2.8, 3.5, size=(nr,))
        eps_np[b, :nr] = rng.uniform(0.05, 0.18, size=(nr,))

        # Add 1-2 full exclusions (0-1, 2-3, 4-5) and 1-4 scaled pairs (1-2, 3-4)
        for u, v in ((0, 1), (2, 3), (4, 5)):
            excl_idx[b, u, 0] = v
            excl_idx[b, v, 0] = u
            excl_sv[b, u, 0] = excl_sv[b, v, 0] = 0.0
            excl_se[b, u, 0] = excl_se[b, v, 0] = 0.0
        for u, v in ((1, 2), (3, 4)):
            excl_idx[b, u, 1] = v
            excl_idx[b, v, 1] = u
            excl_sv[b, u, 1] = excl_sv[b, v, 1] = 0.5
            excl_se[b, u, 1] = excl_se[b, v, 1] = float(1.0 / 1.2)

    return pos_np, q_np, sig_np, eps_np, mask_np, boxes_np, excl_idx, excl_sv, excl_se


def test_ragged_morton_tile_pairs_cross_shape_parity_determinism_and_grads() -> None:
    """Verify cross-size, cross-shape ragged Morton tile pairs match 27-image f64 ref & deterministic invariance."""
    (
        pos_np,
        q_np,
        sig_np,
        eps_np,
        mask_np,
        boxes_np,
        excl_idx,
        excl_sv,
        excl_se,
    ) = _make_cross_shape_ragged_batch()
    cutoff = 8.5
    pme_alpha = 0.34

    e_ref, f_ref = _exact_27_image_nonbonded_f64_ref(
        pos_np,
        q_np,
        sig_np,
        eps_np,
        mask_np,
        boxes_np,
        excl_idx,
        excl_sv,
        excl_se,
        cutoff=cutoff,
        pme_alpha=pme_alpha,
        include_pme_excl_erf=True,
    )

    pos_j = jnp.asarray(pos_np, dtype=jnp.float32)
    q_j = jnp.asarray(q_np, dtype=jnp.float32)
    sig_j = jnp.asarray(sig_np, dtype=jnp.float32)
    eps_j = jnp.asarray(eps_np, dtype=jnp.float32)
    mask_j = jnp.asarray(mask_np)
    boxes_j = jnp.asarray(boxes_np, dtype=jnp.float32)

    rmp: RaggedMortonTilePairs = build_ragged_morton_tile_pairs(
        pos_j,
        mask_j,
        boxes_j,
        cutoff=cutoff,
        skin=1.5,
        excl_indices=excl_idx,
        excl_scales_vdw=excl_sv,
        excl_scales_elec=excl_se,
        tp_group=1,
    )
    assert rmp.is_triclinic is True
    assert rmp.pair_queue.batch_size == 3
    # Replica 0 (24 atoms = 1 tile) has at most 1 upper-tri tile pair -> ghost CTAs eliminated!
    assert int(rmp.num_valid_pairs[0]) == 1

    q_s = rmp.reorder_atom_array(q_j)
    sig_s = rmp.reorder_atom_array(sig_j, pad_value=1.0)
    eps_s = rmp.reorder_atom_array(eps_j)

    for det_mode in (False, True):
        e_got, f_sorted = pallas_nonbonded_tile_pairs_ragged(
            rmp.positions_sorted,
            q_s,
            sig_s,
            eps_s,
            rmp.atom_mask_sorted,
            rmp.tile_pairs,
            rmp.excl_full_mask,
            rmp.excl_14_mask,
            pair_queue=rmp.pair_queue,
            box=boxes_j,
            cutoff=cutoff,
            pme_alpha=pme_alpha,
            scale_14_vdw=rmp.scale_14_vdw,
            scale_14_elec=rmp.scale_14_elec,
            include_pme_excl_erf=True,
            deterministic_forces=det_mode,
            return_forces=True,
            interpret=True,
        )
        f_got = rmp.unsort_atom_array(f_sorted, n_orig=pos_np.shape[1])
        np.testing.assert_allclose(np.asarray(e_got), e_ref, rtol=2e-5, atol=2e-4)
        np.testing.assert_allclose(np.asarray(f_got), f_ref, rtol=3e-5, atol=3e-4)

    # Bitwise permutation invariance of deterministic_forces=True across shuffled tile_pairs & tp_group=4
    rng_perm = np.random.default_rng(4242)
    tp_shuf = np.asarray(rmp.tile_pairs).copy()
    ef_shuf = np.asarray(rmp.excl_full_mask).copy()
    e14_shuf = np.asarray(rmp.excl_14_mask).copy()
    for b in range(3):
        nv = int(rmp.num_valid_pairs[b])
        perm_p = rng_perm.permutation(nv)
        tp_shuf[b, :nv] = tp_shuf[b, perm_p]
        ef_shuf[b, :nv] = ef_shuf[b, perm_p]
        e14_shuf[b, :nv] = e14_shuf[b, perm_p]

    _, f_det_1 = pallas_nonbonded_tile_pairs_ragged(
        rmp.positions_sorted, q_s, sig_s, eps_s, rmp.atom_mask_sorted,
        rmp.tile_pairs, rmp.excl_full_mask, rmp.excl_14_mask,
        num_valid_pairs=rmp.num_valid_pairs, box=boxes_j, cutoff=cutoff, pme_alpha=pme_alpha,
        scale_14_vdw=rmp.scale_14_vdw, scale_14_elec=rmp.scale_14_elec,
        include_pme_excl_erf=True, tp_group=1, deterministic_forces=True,
        return_forces=True, interpret=True,
    )
    _, f_det_shuf_4 = pallas_nonbonded_tile_pairs_ragged(
        rmp.positions_sorted, q_s, sig_s, eps_s, rmp.atom_mask_sorted,
        jnp.asarray(tp_shuf), jnp.asarray(ef_shuf), jnp.asarray(e14_shuf),
        num_valid_pairs=rmp.num_valid_pairs, box=boxes_j, cutoff=cutoff, pme_alpha=pme_alpha,
        scale_14_vdw=rmp.scale_14_vdw, scale_14_elec=rmp.scale_14_elec,
        include_pme_excl_erf=True, tp_group=4, deterministic_forces=True,
        return_forces=True, interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(f_det_1), np.asarray(f_det_shuf_4))

    # Custom VJP gradients wrt (positions, charges, sigmas, epsilons)
    weights = jnp.array([1.0, 0.7, 1.3], dtype=jnp.float32)
    gp, gq, gsig, geps = jax.grad(
        lambda p_in, q_in, s_in, ep_in: jnp.sum(
            weights
            * pallas_nonbonded_tile_pairs_ragged(
                p_in, q_in, s_in, ep_in, rmp.atom_mask_sorted,
                rmp.tile_pairs, rmp.excl_full_mask, rmp.excl_14_mask,
                pair_queue=rmp.pair_queue, box=boxes_j, cutoff=cutoff, pme_alpha=pme_alpha,
                scale_14_vdw=rmp.scale_14_vdw, scale_14_elec=rmp.scale_14_elec,
                include_pme_excl_erf=True, interpret=True,
            )
        ),
        argnums=(0, 1, 2, 3),
    )(rmp.positions_sorted, q_s, sig_s, eps_s)
    assert jnp.all(jnp.isfinite(gp))
    assert jnp.all(jnp.isfinite(gq))
    assert jnp.all(jnp.isfinite(gsig))
    assert jnp.all(jnp.isfinite(geps))
    # dE/dpos == -weights * forces
    np.testing.assert_allclose(
        np.asarray(gp),
        -np.asarray(weights[:, None, None] * f_det_1),
        rtol=5e-5,
        atol=5e-4,
    )

    # StableHLO export portability check
    with export_context():
        exp_nb = jax.export.export(
            jax.jit(
                lambda p_in, q_in: pallas_nonbonded_tile_pairs_ragged(
                    p_in, q_in, sig_s, eps_s, rmp.atom_mask_sorted,
                    rmp.tile_pairs, rmp.excl_full_mask, rmp.excl_14_mask,
                    num_valid_pairs=rmp.num_valid_pairs, box=boxes_j,
                    cutoff=cutoff, pme_alpha=pme_alpha, include_pme_excl_erf=True,
                    deterministic_forces=True, return_forces=True,
                )
            )
        )(rmp.positions_sorted, q_s)
    _assert_portable_stablehlo(exp_nb.mlir_module(), "triclinic_ragged_nonbonded_export")


def test_ragged_tiered_spme_cross_shape_parity_grads_and_export() -> None:
    """Verify 1D ragged spread/gather & multi-tier SPME across cubic + dodecahedron + octahedron."""
    B = 3
    n_reals = [20, 42, 68]
    n_pad = 96
    # Distinct cell scales so "grouped" / "auto" exercise multi-resolution FFT tiers!
    boxes_np = np.stack(
        [
            np.diag([24.0, 24.0, 24.0]),
            _rhombic_dodecahedron_box(34.0),
            _truncated_octahedron_box(46.0),
        ],
        axis=0,
    )
    rng = np.random.default_rng(9090)
    pos_np = np.zeros((B, n_pad, 3), dtype=np.float64)
    q_np = np.zeros((B, n_pad), dtype=np.float64)
    mask_np = np.zeros((B, n_pad), dtype=bool)
    for b, nr in enumerate(n_reals):
        mask_np[b, :nr] = True
        fracs = rng.uniform(-0.25, 1.25, size=(nr, 3))  # Intentionally outside [0, 1) to test s_wrap!
        pos_np[b, :nr] = fracs @ boxes_np[b]
        rq = rng.uniform(-0.8, 0.8, size=(nr,))
        rq -= np.mean(rq)
        q_np[b, :nr] = rq

    pos_j = jnp.asarray(pos_np, dtype=jnp.float32)
    q_j = jnp.asarray(q_np, dtype=jnp.float32)
    mask_j = jnp.asarray(mask_np)
    boxes_j = jnp.asarray(boxes_np, dtype=jnp.float32)

    alpha = 0.34
    cutoff = 8.5

    # Check direct spread_charges_pallas_ragged & gather_pallas_ragged on mixed (3, 3, 3) batch
    gd_direct = (24, 24, 24)
    q_mesh = spread_charges_pallas_ragged(
        pos_j, q_j, mask_j, boxes_j, gd_direct, interpret=True
    )
    for b in range(B):
        # Charge conservation: sum of B-spline weights per active atom == 1 -> sum(Q_mesh[b]) == sum(q[b])
        np.testing.assert_allclose(
            float(jnp.sum(q_mesh[b])), float(np.sum(q_np[b, : n_reals[b]])), atol=1e-4
        )
    f_gath, dq_gath = gather_pallas_ragged(
        pos_j, q_j, mask_j, boxes_j, q_mesh, gd_direct, compute_dq=True, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(f_gath)[~mask_np], 0.0)
    np.testing.assert_array_equal(np.asarray(dq_gath)[~mask_np], 0.0)

    # Check all 3 tier modes ("grouped", "auto", "uniform") against single-system pme.py reference
    for mode in ("grouped", "auto", "uniform"):
        plan = plan_pme_tiers(
            boxes_j,
            cutoff=cutoff,
            alpha=alpha,
            atom_mask=mask_j,
            pme_tier_mode=mode,
            tier_launch_overhead_us=5.0,
        )
        for tier in plan.tiers:
            assert tier.G_cached is not None, f"G_cached must be populated for triclinic tier in {mode=}"

        # Map each replica to the exact grid_dims assigned in this plan
        rep_gd: dict[int, tuple[int, int, int]] = {}
        for tier in plan.tiers:
            for r_i in tier.replica_indices_tuple:
                rep_gd[int(r_i)] = tier.grid_dims

        e_ref_list: list[float] = []
        f_ref_list: list[np.ndarray] = []
        for b in range(B):
            gd_b = rep_gd[b]
            pb = pos_j[b]
            qb = q_j[b]
            mb = mask_j[b]
            bb = boxes_j[b]

            def eb_fn(r_1, q_1=qb, m_1=mb, b_1=bb, g_1=gd_b):
                return spme_reciprocal_energy(
                    r_1, q_1, m_1, b_1, g_1, alpha=alpha, order=4, use_pallas_pme=False
                ) + spme_self_energy(q_1, m_1, alpha=alpha)

            eb_val, gb_pos = jax.value_and_grad(eb_fn)(pb)
            e_ref_list.append(float(eb_val))
            f_ref_list.append(np.asarray(-gb_pos))

        e_ref_arr = np.array(e_ref_list, dtype=np.float32)
        f_ref_arr = np.stack(f_ref_list, axis=0)

        e_pl = spme_energy_ragged_tiered(
            pos_j, q_j, mask_j, boxes_j, tier_plan=plan, use_pallas_pme=True, interpret=True
        )
        f_pl = spme_forces_ragged_tiered(
            pos_j, q_j, mask_j, boxes_j, tier_plan=plan, use_pallas_pme=True, interpret=True
        )
        e_both, f_both = spme_energy_and_forces_ragged_tiered(
            pos_j, q_j, mask_j, boxes_j, tier_plan=plan, use_pallas_pme=True, interpret=True
        )

        np.testing.assert_allclose(np.asarray(e_pl), e_ref_arr, rtol=2e-5, atol=2e-4)
        np.testing.assert_allclose(np.asarray(e_both), e_ref_arr, rtol=2e-5, atol=2e-4)
        np.testing.assert_allclose(np.asarray(f_pl), f_ref_arr, rtol=5e-5, atol=5e-4)
        np.testing.assert_allclose(np.asarray(f_both), f_ref_arr, rtol=5e-5, atol=5e-4)

    # Analytical custom_vjp grad wrt (positions, charges, box_size) vs pure-JAX pme.py AD & central FD
    plan_g = plan_pme_tiers(boxes_j, cutoff=cutoff, alpha=alpha, atom_mask=mask_j, pme_tier_mode="grouped")
    w_b = jnp.array([1.1, 0.8, 1.3], dtype=jnp.float32)
    g_pos, g_q, g_box = jax.grad(
        lambda r_a, q_a, b_a: jnp.sum(
            w_b * spme_energy_ragged_tiered(r_a, q_a, mask_j, b_a, tier_plan=plan_g, interpret=True)
        ),
        argnums=(0, 1, 2),
    )(pos_j, q_j, boxes_j)
    g_pos_ref, g_q_ref, g_box_ref = jax.grad(
        lambda r_a, q_a, b_a: jnp.sum(
            w_b * spme_energy_ragged_tiered(r_a, q_a, mask_j, b_a, tier_plan=plan_g, use_pallas_pme=False)
        ),
        argnums=(0, 1, 2),
    )(pos_j, q_j, boxes_j)
    assert jnp.all(jnp.isfinite(g_pos))
    assert jnp.all(jnp.isfinite(g_q))
    assert jnp.all(jnp.isfinite(g_box))
    np.testing.assert_allclose(np.asarray(g_pos), np.asarray(g_pos_ref), rtol=1e-4, atol=5e-4)
    np.testing.assert_allclose(np.asarray(g_q), np.asarray(g_q_ref), rtol=1e-4, atol=5e-4)
    np.testing.assert_allclose(np.asarray(g_box), np.asarray(g_box_ref), rtol=1e-4, atol=5e-4)

    # Central FD spot-checks (step 2.5e-2 avoids float32 ULP quantization on ~500 kcal/mol sums)
    eps_q = 2.5e-2
    q_p = q_j.at[1, 3].add(eps_q)
    q_m = q_j.at[1, 3].add(-eps_q)
    ep = float(jnp.sum(w_b * spme_energy_ragged_tiered(pos_j, q_p, mask_j, boxes_j, tier_plan=plan_g, use_pallas_pme=False)))
    em = float(jnp.sum(w_b * spme_energy_ragged_tiered(pos_j, q_m, mask_j, boxes_j, tier_plan=plan_g, use_pallas_pme=False)))
    fd_q = (ep - em) / (2.0 * eps_q)
    np.testing.assert_allclose(float(g_q[1, 3]), fd_q, rtol=5e-3, atol=5e-3)

    eps_b = 2.5e-2
    b_p = boxes_j.at[2, 1, 0].add(eps_b)
    b_m = boxes_j.at[2, 1, 0].add(-eps_b)
    ebp = float(jnp.sum(w_b * spme_energy_ragged_tiered(pos_j, q_j, mask_j, b_p, tier_plan=plan_g, use_pallas_pme=False)))
    ebm = float(jnp.sum(w_b * spme_energy_ragged_tiered(pos_j, q_j, mask_j, b_m, tier_plan=plan_g, use_pallas_pme=False)))
    fd_bx = (ebp - ebm) / (2.0 * eps_b)
    np.testing.assert_allclose(float(g_box[2, 1, 0]), fd_bx, rtol=5e-3, atol=5e-3)

    # B == 3 (3, 3) orthorhombic batch disambiguation in plan_pme_tiers & spme_energy_ragged_tiered
    ortho_3x3 = jnp.array([[24.0, 25.0, 26.0], [28.0, 29.0, 30.0], [34.0, 35.0, 36.0]], dtype=jnp.float32)
    plan_o3 = plan_pme_tiers(ortho_3x3, cutoff=cutoff, alpha=alpha, atom_mask=mask_j, is_triclinic=False)
    e_o3 = spme_energy_ragged_tiered(pos_j, q_j, mask_j, ortho_3x3, tier_plan=plan_o3, interpret=True)
    assert e_o3.shape == (3,)
    assert jnp.all(jnp.isfinite(e_o3))

    # Portable StableHLO export check for triclinic ragged SPME energy & forces
    with export_context():
        exp_spme_e = jax.export.export(
            jax.jit(lambda r_a, q_a: spme_energy_ragged_tiered(r_a, q_a, mask_j, boxes_j, tier_plan=plan_g))
        )(pos_j, q_j)
        exp_spme_f = jax.export.export(
            jax.jit(lambda r_a, q_a: spme_forces_ragged_tiered(r_a, q_a, mask_j, boxes_j, tier_plan=plan_g))
        )(pos_j, q_j)
    _assert_portable_stablehlo(exp_spme_e.mlir_module(), "triclinic_spme_energy_export")
    _assert_portable_stablehlo(exp_spme_f.mlir_module(), "triclinic_spme_forces_export")
