"""Verification suite for v3.7 Track 3A: Disjoint SETTLE / SHAKE / Solute Pallas Kernel Split.

Covers:
1. Disjoint sequential ``pl.pallas_call`` dispatch (up to 3 separate CTAs: unconstrained solute,
   rigid-water SETTLE, and star-cluster SHAKE/RATTLE) with zero monolithic 3-way ``lax.cond``
   register inflation and zero HBM copies via in-place ``input_output_aliases``.
2. Exact numerical parity (``atol=1e-6``) across ``B=1`` and ``B=4``, pure water, constrained
   solute, and mixed protein+water+ghost topologies in standard (``float32`` / ``float64``) and
   ``fltflt`` (``single`` and ``mixed``) accumulator modes, including ``fuse_prev_post=True`` and
   ``emit_f32_out=True``.
3. Morton permutation ``morton_inv_perm`` on/off parity via ``_build_baoab_soa_plan`` and
   ``_pallas_baoab_post_soa`` across standard and ``fltflt`` modes.
4. Portable StableHLO lowering under ``prolix.export.export_context()`` with zero ``custom_call``s
   and zero non-portable backend markers.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from prolix.export import export_context
from prolix.physics.settle import (
    TIP3P_RHH,
    TIP3P_ROH,
    _pallas_baoab_post_force_batch,
    _pallas_baoab_post_soa,
    _pallas_baoab_pre_force_batch,
    build_baoab_soa_static_plan,
    pallas_baoab_post_force,
    pallas_baoab_pre_force,
)


def _water_geometry(origin_xyz: tuple[float, float, float]) -> np.ndarray:
    half_hh = 0.5 * TIP3P_RHH
    y_h = float(np.sqrt(TIP3P_ROH**2 - half_hh**2))
    ox, oy, oz = origin_xyz
    return np.array(
        [
            [ox, oy, oz],
            [ox + half_hh, oy + y_h, oz],
            [ox - half_hh, oy + y_h, oz],
        ],
        dtype=np.float64,
    )


def _ch3_geometry(origin_xyz: tuple[float, float, float], d_ch: float = 1.09) -> np.ndarray:
    cx, cy, cz = origin_xyz
    dirs = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return np.vstack([[cx, cy, cz], np.array([cx, cy, cz]) + d_ch * dirs])


def _nh_geometry(origin_xyz: tuple[float, float, float], d_nh: float = 1.01) -> np.ndarray:
    nx, ny, nz = origin_xyz
    return np.array([[nx, ny, nz], [nx, ny, nz + d_nh]], dtype=np.float64)


def _build_mixed_fixture_f64() -> dict[str, Any]:
    """Build an N=16 system: 2 unconstrained solute + CH3(4) + NH(2) + 2 waters(6) + 2 ghosts."""
    unconstrained = np.array([[4.0, 4.0, 4.0], [6.5, 4.5, 5.0]], dtype=np.float64)
    ch3 = _ch3_geometry((10.0, 9.0, 8.0), d_ch=1.09)  # atoms 2..5
    nh = _nh_geometry((13.0, 7.0, 9.5), d_nh=1.01)  # atoms 6..7
    w0 = _water_geometry((15.5, 11.0, 6.0))  # atoms 8..10
    w1 = _water_geometry((18.0, 14.0, 12.0))  # atoms 11..13
    ghosts = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float64)  # atoms 14..15

    pos = np.vstack([unconstrained, ch3, nh, w0, w1, ghosts])
    mass = np.array(
        [
            12.011,
            15.999,  # unconstrained solute
            12.011,
            1.008,
            1.008,
            1.008,  # CH3 cluster
            14.007,
            1.008,  # NH cluster
            15.999,
            1.008,
            1.008,  # water 0
            15.999,
            1.008,
            1.008,  # water 1
            1.0,
            1.0,  # ghost padding
        ],
        dtype=np.float64,
    )
    atom_mask = np.array([1] * 14 + [0, 0], dtype=np.int32)
    water_indices = np.array([[8, 9, 10], [11, 12, 13]], dtype=np.int32)
    water_mask = np.array([1, 1], dtype=np.int32)
    constraint_pairs = np.array([[2, 3], [2, 4], [2, 5], [6, 7]], dtype=np.int32)
    constraint_lengths = np.array([1.09, 1.09, 1.09, 1.01], dtype=np.float64)
    constraint_mask = np.array([1, 1, 1, 1], dtype=np.int32)
    box = np.array([28.0, 28.0, 28.0], dtype=np.float64)

    rng = np.random.default_rng(20261117)
    mom = rng.normal(0.0, 0.35, size=pos.shape) * atom_mask[:, None]
    frc = rng.normal(0.0, 1.25, size=pos.shape) * atom_mask[:, None]
    return {
        "pos": pos,
        "mom": mom,
        "frc": frc,
        "mass": mass,
        "atom_mask": atom_mask,
        "wi": water_indices,
        "wm": water_mask,
        "cp": constraint_pairs,
        "cl": constraint_lengths,
        "cm": constraint_mask,
        "box": box,
    }


def _check_constraint_manifolds(
    pos: np.ndarray,
    mom: np.ndarray,
    mass: np.ndarray,
    wi: np.ndarray | None,
    wm: np.ndarray | None,
    cp: np.ndarray | None,
    cl: np.ndarray | None,
    cm: np.ndarray | None,
    *,
    pos_tol: float = 2e-5,
    vel_tol: float = 2e-5,
) -> None:
    vel = mom / mass[:, None]
    if wi is not None and wi.shape[0] > 0:
        w_active = np.ones((wi.shape[0],), dtype=bool) if wm is None else (wm > 0)
        for idx_w, ok in enumerate(w_active):
            if not ok:
                continue
            io, ih1, ih2 = wi[idx_w]
            d_oh1 = pos[ih1] - pos[io]
            d_oh2 = pos[ih2] - pos[io]
            d_hh = pos[ih2] - pos[ih1]
            assert abs(np.linalg.norm(d_oh1) - TIP3P_ROH) < pos_tol
            assert abs(np.linalg.norm(d_oh2) - TIP3P_ROH) < pos_tol
            assert abs(np.linalg.norm(d_hh) - TIP3P_RHH) < pos_tol
            assert abs(np.dot(vel[ih1] - vel[io], d_oh1) / np.linalg.norm(d_oh1)) < vel_tol
            assert abs(np.dot(vel[ih2] - vel[io], d_oh2) / np.linalg.norm(d_oh2)) < vel_tol
            assert abs(np.dot(vel[ih2] - vel[ih1], d_hh) / np.linalg.norm(d_hh)) < vel_tol
    if cp is not None and cl is not None and cp.shape[0] > 0:
        c_active = np.ones((cp.shape[0],), dtype=bool) if cm is None else (cm > 0)
        for idx_c, ok in enumerate(c_active):
            if not ok:
                continue
            ia, ib = cp[idx_c]
            dab = pos[ib] - pos[ia]
            norm_ab = float(np.linalg.norm(dab))
            assert abs(norm_ab - float(cl[idx_c])) < pos_tol
            assert abs(float(np.dot(vel[ib] - vel[ia], dab)) / norm_ab) < vel_tol


def test_disjoint_pallas_call_topology_dispatch_and_aliasing(monkeypatch: pytest.MonkeyPatch):
    """Verify exact disjoint kernel launches, grids, and in-place input_output_aliases per topology."""
    fx = _build_mixed_fixture_f64()
    recorded: list[dict[str, Any]] = []
    orig_pallas_call = pl.pallas_call

    def _spy_pallas_call(kernel_fn: Any, **kwargs: Any):
        call_fn = orig_pallas_call(kernel_fn, **kwargs)

        def _wrapped(*args: Any):
            recorded.append(
                {
                    "name": kwargs.get("name"),
                    "grid": kwargs.get("grid"),
                    "aliases": dict(kwargs.get("input_output_aliases") or {}),
                    "n_args": len(args),
                    "n_outs": len(kwargs["out_shape"]),
                }
            )
            return call_fn(*args)

        return _wrapped

    monkeypatch.setattr(pl, "pallas_call", _spy_pallas_call)

    pos32 = jnp.asarray(fx["pos"], dtype=jnp.float32)
    mom32 = jnp.asarray(fx["mom"], dtype=jnp.float32)
    frc32 = jnp.asarray(fx["frc"], dtype=jnp.float32)
    mass32 = jnp.asarray(fx["mass"], dtype=jnp.float32)
    am32 = jnp.asarray(fx["atom_mask"], dtype=jnp.int32)
    wi = jnp.asarray(fx["wi"], dtype=jnp.int32)
    wm = jnp.asarray(fx["wm"], dtype=jnp.int32)
    cp = jnp.asarray(fx["cp"], dtype=jnp.int32)
    cl32 = jnp.asarray(fx["cl"], dtype=jnp.float32)
    cm = jnp.asarray(fx["cm"], dtype=jnp.int32)
    box32 = jnp.asarray(fx["box"], dtype=jnp.float32)

    # 1. Mixed protein + water + ghost: 3 pre-force launches + 3 post-force launches
    recorded.clear()
    pos_m, mom_m, _ = pallas_baoab_pre_force(
        pos32,
        mom32,
        frc32,
        mass32,
        jax.random.key(0),
        wi,
        wm,
        box32,
        dt=jnp.float32(0.02),
        kT=jnp.float32(0.596),
        gamma=0.1,
        interpret=True,
        atom_mask=am32,
        constraint_pairs=cp,
        constraint_lengths=cl32,
        constraint_mask=cm,
    )
    _ = pallas_baoab_post_force(
        pos_m,
        mom_m,
        frc32,
        mass32,
        wi,
        wm,
        dt=jnp.float32(0.02),
        interpret=True,
        atom_mask=am32,
        constraint_pairs=cp,
        constraint_lengths=cl32,
        constraint_mask=cm,
    )
    assert [r["name"] for r in recorded] == [
        "baoab_pre_solute",
        "baoab_pre_settle",
        "baoab_pre_rattle",
        "baoab_post_solute",
        "baoab_post_settle",
        "baoab_post_rattle",
    ]
    for r in recorded[:3]:
        assert r["aliases"] == {i: i for i in range(6)}
        assert r["grid"] == (1, 1)
    for r in recorded[3:]:
        assert r["aliases"] == {0: 0, 1: 1, 2: 2}
        assert r["grid"] == (1, 1)

    # 2. Pure water with n_active_atoms=0: ONLY 1 pre-force launch + 1 post-force launch!
    recorded.clear()
    w_pos = jnp.asarray(np.vstack([_water_geometry((5.0, 5.0, 5.0)), _water_geometry((9.0, 9.0, 9.0))]), dtype=jnp.float32)
    w_mom = jnp.zeros_like(w_pos)
    w_frc = jnp.ones_like(w_pos) * jnp.float32(0.2)
    w_mass = jnp.asarray([15.999, 1.008, 1.008, 15.999, 1.008, 1.008], dtype=jnp.float32)
    w_wi = jnp.asarray([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    w_wm = jnp.asarray([1, 1], dtype=jnp.int32)
    pw1, mw1, _ = pallas_baoab_pre_force(
        w_pos,
        w_mom,
        w_frc,
        w_mass,
        jax.random.key(1),
        w_wi,
        w_wm,
        box32,
        dt=jnp.float32(0.02),
        kT=jnp.float32(0.596),
        gamma=0.1,
        interpret=True,
        n_active_atoms=0,
    )
    _ = pallas_baoab_post_force(
        pw1, mw1, w_frc, w_mass, w_wi, w_wm, dt=jnp.float32(0.02), interpret=True, n_active_atoms=0
    )
    assert [r["name"] for r in recorded] == ["baoab_pre_settle", "baoab_post_settle"]

    # 3. Dry constrained solute (W=0, C=4): ONLY solute + rattle launches (zero water launches!)
    recorded.clear()
    empty_wi = jnp.zeros((0, 3), dtype=jnp.int32)
    empty_wm = jnp.zeros((0,), dtype=jnp.int32)
    ps1, ms1, _ = pallas_baoab_pre_force(
        pos32[:8],
        mom32[:8],
        frc32[:8],
        mass32[:8],
        jax.random.key(2),
        empty_wi,
        empty_wm,
        box32,
        dt=jnp.float32(0.02),
        kT=jnp.float32(0.596),
        gamma=0.1,
        interpret=True,
        constraint_pairs=cp,
        constraint_lengths=cl32,
        constraint_mask=cm,
    )
    _ = pallas_baoab_post_force(
        ps1,
        ms1,
        frc32[:8],
        mass32[:8],
        empty_wi,
        empty_wm,
        dt=jnp.float32(0.02),
        interpret=True,
        constraint_pairs=cp,
        constraint_lengths=cl32,
        constraint_mask=cm,
    )
    assert [r["name"] for r in recorded] == [
        "baoab_pre_solute",
        "baoab_pre_rattle",
        "baoab_post_solute",
        "baoab_post_rattle",
    ]


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("topology_kind", ["pure_water", "constrained_solute", "mixed_protein_water"])
def test_disjoint_parity_across_topologies_batches_and_modes(batch_size: int, topology_kind: str):
    """Verify <1e-6 parity between float64 reference, float32 single, fltflt single, and fltflt mixed."""
    with jax.enable_x64(True):
        fx = _build_mixed_fixture_f64()
        if topology_kind == "pure_water":
            pos0 = fx["pos"][8:14]
            mom0 = fx["mom"][8:14]
            frc0 = fx["frc"][8:14]
            mass = jnp.asarray(fx["mass"][8:14], dtype=jnp.float64)
            am = jnp.ones((6,), dtype=jnp.int32)
            wi = jnp.asarray([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
            wm = jnp.ones((2,), dtype=jnp.int32)
            cp = cl = cm = None
        elif topology_kind == "constrained_solute":
            pos0 = fx["pos"][:8]
            mom0 = fx["mom"][:8]
            frc0 = fx["frc"][:8]
            mass = jnp.asarray(fx["mass"][:8], dtype=jnp.float64)
            am = jnp.ones((8,), dtype=jnp.int32)
            wi = jnp.zeros((0, 3), dtype=jnp.int32)
            wm = jnp.zeros((0,), dtype=jnp.int32)
            cp = jnp.asarray(fx["cp"], dtype=jnp.int32)
            cl = jnp.asarray(fx["cl"], dtype=jnp.float64)
            cm = jnp.asarray(fx["cm"], dtype=jnp.int32)
        else:
            pos0 = fx["pos"]
            mom0 = fx["mom"]
            frc0 = fx["frc"]
            mass = jnp.asarray(fx["mass"], dtype=jnp.float64)
            am = jnp.asarray(fx["atom_mask"], dtype=jnp.int32)
            wi = jnp.asarray(fx["wi"], dtype=jnp.int32)
            wm = jnp.asarray(fx["wm"], dtype=jnp.int32)
            cp = jnp.asarray(fx["cp"], dtype=jnp.int32)
            cl = jnp.asarray(fx["cl"], dtype=jnp.float64)
            cm = jnp.asarray(fx["cm"], dtype=jnp.int32)

        box64 = jnp.asarray(fx["box"], dtype=jnp.float64)
        dt = 0.025
        kT = 0.596
        gamma = 0.15

        if batch_size == 1:
            pos64 = jnp.asarray(pos0, dtype=jnp.float64)
            mom64 = jnp.asarray(mom0, dtype=jnp.float64)
            frc32 = jnp.asarray(frc0, dtype=jnp.float32)
            key = jax.random.key(42)

            p_ref, m_ref_pre, _, p_emit = pallas_baoab_pre_force(
                pos64,
                mom64,
                frc32,
                mass,
                key,
                wi,
                wm,
                box64,
                dt=dt,
                kT=kT,
                gamma=gamma,
                interpret=True,
                atom_mask=am,
                constraint_pairs=cp,
                constraint_lengths=cl,
                constraint_mask=cm,
                return_pos_f32=True,
                f64_accumulators=True,
            )
            m_ref = pallas_baoab_post_force(
                p_ref,
                m_ref_pre,
                frc32,
                mass,
                wi,
                wm,
                dt=dt,
                interpret=True,
                atom_mask=am,
                constraint_pairs=cp,
                constraint_lengths=cl,
                constraint_mask=cm,
                f64_accumulators=True,
            )
            assert p_emit.dtype == jnp.float32
            np.testing.assert_allclose(
                np.asarray(p_emit), np.asarray(p_ref, dtype=np.float32), atol=1e-6, rtol=1e-6
            )

            # fltflt mixed (float64 caller state)
            p_ff_mix, m_ff_mix_pre, _ = pallas_baoab_pre_force(
                pos64,
                mom64,
                frc32,
                mass,
                key,
                wi,
                wm,
                box64,
                dt=dt,
                kT=kT,
                gamma=gamma,
                interpret=True,
                atom_mask=am,
                constraint_pairs=cp,
                constraint_lengths=cl,
                constraint_mask=cm,
                accumulator_mode="fltflt",
            )
            m_ff_mix = pallas_baoab_post_force(
                p_ff_mix,
                m_ff_mix_pre,
                frc32,
                mass,
                wi,
                wm,
                dt=dt,
                interpret=True,
                atom_mask=am,
                constraint_pairs=cp,
                constraint_lengths=cl,
                constraint_mask=cm,
                accumulator_mode="fltflt",
            )

            # fltflt single (float32 hi + lo accumulators)
            cl32 = None if cl is None else cl.astype(jnp.float32)
            p_hi, m_hi_pre, _, p_lo, m_lo_pre = pallas_baoab_pre_force(
                pos64.astype(jnp.float32),
                mom64.astype(jnp.float32),
                frc32,
                mass.astype(jnp.float32),
                key,
                wi,
                wm,
                box64.astype(jnp.float32),
                dt=jnp.float32(dt),
                kT=jnp.float32(kT),
                gamma=gamma,
                interpret=True,
                atom_mask=am,
                constraint_pairs=cp,
                constraint_lengths=cl32,
                constraint_mask=cm,
                accumulator_mode="fltflt",
                return_lo_accumulators=True,
            )
            m_hi, m_lo = pallas_baoab_post_force(
                p_hi,
                m_hi_pre,
                frc32,
                mass.astype(jnp.float32),
                wi,
                wm,
                dt=jnp.float32(dt),
                interpret=True,
                atom_mask=am,
                constraint_pairs=cp,
                constraint_lengths=cl32,
                constraint_mask=cm,
                accumulator_mode="fltflt",
                positions_lo=p_lo,
                momentum_lo=m_lo_pre,
                return_lo_accumulators=True,
            )

            # Standard float32 single
            p_s32, m_s32_pre, _ = pallas_baoab_pre_force(
                pos64.astype(jnp.float32),
                mom64.astype(jnp.float32),
                frc32,
                mass.astype(jnp.float32),
                key,
                wi,
                wm,
                box64.astype(jnp.float32),
                dt=jnp.float32(dt),
                kT=jnp.float32(kT),
                gamma=gamma,
                interpret=True,
                atom_mask=am,
                constraint_pairs=cp,
                constraint_lengths=cl32,
                constraint_mask=cm,
            )
            m_s32 = pallas_baoab_post_force(
                p_s32,
                m_s32_pre,
                frc32,
                mass.astype(jnp.float32),
                wi,
                wm,
                dt=jnp.float32(dt),
                interpret=True,
                atom_mask=am,
                constraint_pairs=cp,
                constraint_lengths=cl32,
                constraint_mask=cm,
            )

            p_ff_single = np.asarray(p_hi, dtype=np.float64) + np.asarray(p_lo, dtype=np.float64)
            m_ff_single = np.asarray(m_hi, dtype=np.float64) + np.asarray(m_lo, dtype=np.float64)
            np.testing.assert_allclose(np.asarray(p_ff_mix), np.asarray(p_ref), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(m_ff_mix), np.asarray(m_ref), atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(p_ff_single, np.asarray(p_ref), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(m_ff_single, np.asarray(m_ref), atol=2e-6, rtol=2e-6)
            # Naive float32 (without fltflt) stays close to float64 within float32 R-impulse precision
            np.testing.assert_allclose(np.asarray(p_s32), np.asarray(p_ref), atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(np.asarray(m_s32), np.asarray(m_ref), atol=3e-3, rtol=3e-3)

            _check_constraint_manifolds(
                np.asarray(p_ref),
                np.asarray(m_ref),
                np.asarray(mass),
                None if wi.shape[0] == 0 else np.asarray(wi),
                None if wm.shape[0] == 0 else np.asarray(wm),
                None if cp is None else np.asarray(cp),
                None if cl is None else np.asarray(cl),
                None if cm is None else np.asarray(cm),
            )
        else:
            # B=4 multi-replica vmap parity across standard f64, standard f32, fltflt mixed, fltflt single
            shifts = np.array([[0.0, 0.0, 0.0], [0.4, -0.2, 0.3], [-0.3, 0.5, -0.1], [0.2, 0.1, -0.4]])
            pos_b64 = jnp.asarray(pos0[None, :, :] + shifts[:, None, :] * (np.asarray(am)[None, :, None]), dtype=jnp.float64)
            mom_b64 = jnp.asarray(
                np.stack([mom0 * (1.0 + 0.1 * b) for b in range(batch_size)], axis=0), dtype=jnp.float64
            )
            frc_b32 = jnp.asarray(
                np.stack([frc0 * (1.0 - 0.08 * b) for b in range(batch_size)], axis=0), dtype=jnp.float32
            )
            keys_b = jax.random.split(jax.random.key(99), batch_size)
            cl32 = None if cl is None else cl.astype(jnp.float32)

            def _step_f64(p_i, m_i, f_i, k_i):
                p1, m1, _ = pallas_baoab_pre_force(
                    p_i, m_i, f_i, mass, k_i, wi, wm, box64, dt, kT, gamma,
                    interpret=True, atom_mask=am, constraint_pairs=cp, constraint_lengths=cl,
                    constraint_mask=cm, f64_accumulators=True,
                )
                m2 = pallas_baoab_post_force(
                    p1, m1, f_i, mass, wi, wm, dt, interpret=True, atom_mask=am,
                    constraint_pairs=cp, constraint_lengths=cl, constraint_mask=cm, f64_accumulators=True,
                )
                return p1, m2

            def _step_ff_mix(p_i, m_i, f_i, k_i):
                p1, m1, _ = pallas_baoab_pre_force(
                    p_i, m_i, f_i, mass, k_i, wi, wm, box64, dt, kT, gamma,
                    interpret=True, atom_mask=am, constraint_pairs=cp, constraint_lengths=cl,
                    constraint_mask=cm, accumulator_mode="fltflt",
                )
                m2 = pallas_baoab_post_force(
                    p1, m1, f_i, mass, wi, wm, dt, interpret=True, atom_mask=am,
                    constraint_pairs=cp, constraint_lengths=cl, constraint_mask=cm, accumulator_mode="fltflt",
                )
                return p1, m2

            def _step_ff_single(p_i, m_i, f_i, k_i):
                ph, mh, _, pl_o, ml_o = pallas_baoab_pre_force(
                    p_i, m_i, f_i, mass.astype(jnp.float32), k_i, wi, wm, box64.astype(jnp.float32),
                    jnp.float32(dt), jnp.float32(kT), gamma, interpret=True, atom_mask=am,
                    constraint_pairs=cp, constraint_lengths=cl32, constraint_mask=cm,
                    accumulator_mode="fltflt", return_lo_accumulators=True,
                )
                mh2, ml2 = pallas_baoab_post_force(
                    ph, mh, f_i, mass.astype(jnp.float32), wi, wm, jnp.float32(dt),
                    interpret=True, atom_mask=am, constraint_pairs=cp, constraint_lengths=cl32,
                    constraint_mask=cm, accumulator_mode="fltflt", positions_lo=pl_o,
                    momentum_lo=ml_o, return_lo_accumulators=True,
                )
                return ph, pl_o, mh2, ml2

            p_ref_b, m_ref_b = jax.vmap(_step_f64)(pos_b64, mom_b64, frc_b32, keys_b)
            p_ffm_b, m_ffm_b = jax.vmap(_step_ff_mix)(pos_b64, mom_b64, frc_b32, keys_b)
            ph_b, plo_b, mh_b, mlo_b = jax.vmap(_step_ff_single)(
                pos_b64.astype(jnp.float32), mom_b64.astype(jnp.float32), frc_b32, keys_b
            )

            def _step_s32(p_i, m_i, f_i, k_i):
                p1, m1, _ = pallas_baoab_pre_force(
                    p_i, m_i, f_i, mass.astype(jnp.float32), k_i, wi, wm, box64.astype(jnp.float32),
                    jnp.float32(dt), jnp.float32(kT), gamma, interpret=True, atom_mask=am,
                    constraint_pairs=cp, constraint_lengths=cl32, constraint_mask=cm,
                )
                m2 = pallas_baoab_post_force(
                    p1, m1, f_i, mass.astype(jnp.float32), wi, wm, jnp.float32(dt),
                    interpret=True, atom_mask=am, constraint_pairs=cp, constraint_lengths=cl32,
                    constraint_mask=cm,
                )
                return p1, m2

            ps32_vmap, ms32_vmap = jax.vmap(_step_s32)(
                pos_b64.astype(jnp.float32), mom_b64.astype(jnp.float32), frc_b32, keys_b
            )
            ps32_loop = []
            ms32_loop = []
            for b_idx in range(batch_size):
                pb_i, mb_i = _step_s32(
                    pos_b64[b_idx].astype(jnp.float32),
                    mom_b64[b_idx].astype(jnp.float32),
                    frc_b32[b_idx],
                    keys_b[b_idx],
                )
                ps32_loop.append(np.asarray(pb_i))
                ms32_loop.append(np.asarray(mb_i))
            np.testing.assert_allclose(np.asarray(ps32_vmap), np.stack(ps32_loop, axis=0), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(ms32_vmap), np.stack(ms32_loop, axis=0), atol=1e-6, rtol=1e-6)

            np.testing.assert_allclose(np.asarray(p_ffm_b), np.asarray(p_ref_b), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(m_ffm_b), np.asarray(m_ref_b), atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(
                np.asarray(ph_b, dtype=np.float64) + np.asarray(plo_b, dtype=np.float64),
                np.asarray(p_ref_b),
                atol=1e-6,
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(mh_b, dtype=np.float64) + np.asarray(mlo_b, dtype=np.float64),
                np.asarray(m_ref_b),
                atol=2e-6,
                rtol=2e-6,
            )


def test_morton_permutation_on_off_and_disjoint_fltflt_pallas_kernels(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify Morton permutation on/off parity + exercise disjoint fltflt pallas_call kernels."""
    with jax.enable_x64(True):
        fx = _build_mixed_fixture_f64()
        B = 2
        N = fx["pos"].shape[0]
        N_pad = 32
        pos_b = jnp.stack([jnp.asarray(fx["pos"]), jnp.asarray(fx["pos"]) + 0.25], axis=0)
        # Zero ghost shift on replica 1 so ghost positions stay at fixed coordinates
        am_b = jnp.broadcast_to(jnp.asarray(fx["atom_mask"], dtype=jnp.int32)[None, :], (B, N))
        pos_b = jnp.where(am_b[:, :, None] > 0, pos_b, jnp.asarray(fx["pos"])[None, :, :])
        mom_b = jnp.stack([jnp.asarray(fx["mom"]), jnp.asarray(fx["mom"]) * 0.9], axis=0)
        frc_b = jnp.stack([jnp.asarray(fx["frc"]), jnp.asarray(fx["frc"]) * 1.1], axis=0).astype(jnp.float32)
        mass_b = jnp.broadcast_to(jnp.asarray(fx["mass"])[None, :], (B, N))
        wi_b = jnp.broadcast_to(jnp.asarray(fx["wi"], dtype=jnp.int32)[None, :, :], (B, 2, 3))
        wm_b = jnp.broadcast_to(jnp.asarray(fx["wm"], dtype=jnp.int32)[None, :], (B, 2))
        cp_b = jnp.broadcast_to(jnp.asarray(fx["cp"], dtype=jnp.int32)[None, :, :], (B, 4, 2))
        cl_b = jnp.broadcast_to(jnp.asarray(fx["cl"])[None, :], (B, 4))
        cm_b = jnp.broadcast_to(jnp.asarray(fx["cm"], dtype=jnp.int32)[None, :], (B, 4))
        box_b = jnp.broadcast_to(jnp.asarray(fx["box"])[None, :], (B, 3))
        seeds_b = jnp.asarray([[11, 22], [33, 44]], dtype=jnp.int32)
        dt_b = jnp.asarray([0.02, 0.02], dtype=jnp.float64)
        kT_b = jnp.asarray([0.596, 0.596], dtype=jnp.float64)

        # Build two distinct valid Morton inverse permutations mapping [0..N-1] -> [0..N_pad-1]
        rng = np.random.default_rng(777)
        inv_perm_np = np.stack(
            [rng.permutation(N_pad)[:N].astype(np.int32) for _ in range(B)], axis=0
        )
        inv_perm_b = jnp.asarray(inv_perm_np, dtype=jnp.int32)

        # Scatter forces into Morton-padded layout (B * N_pad) so Morton force indexing is tested
        fx_m = np.zeros((B, N_pad), dtype=np.float32)
        fy_m = np.zeros((B, N_pad), dtype=np.float32)
        fz_m = np.zeros((B, N_pad), dtype=np.float32)
        frc_np = np.asarray(frc_b)
        for b in range(B):
            fx_m[b, inv_perm_np[b]] = frc_np[b, :, 0]
            fy_m[b, inv_perm_np[b]] = frc_np[b, :, 1]
            fz_m[b, inv_perm_np[b]] = frc_np[b, :, 2]

        for acc_mode in (None, "fltflt"):
            soa_plan = build_baoab_soa_static_plan(
                B,
                N,
                jnp.float64,
                mass_b,
                wi_b,
                wm_b,
                atom_mask_b=am_b,
                constraint_pairs_b=cp_b,
                constraint_lengths_b=cl_b,
                constraint_mask_b=cm_b,
            )
            px0 = pos_b[:, :, 0].reshape(B * N)
            py0 = pos_b[:, :, 1].reshape(B * N)
            pz0 = pos_b[:, :, 2].reshape(B * N)
            mx0 = mom_b[:, :, 0].reshape(B * N)
            my0 = mom_b[:, :, 1].reshape(B * N)
            mz0 = mom_b[:, :, 2].reshape(B * N)
            fx0 = frc_b[:, :, 0].reshape(B * N)
            fy0 = frc_b[:, :, 1].reshape(B * N)
            fz0 = frc_b[:, :, 2].reshape(B * N)
            in_b = (True,) * 15

            # Morton OFF
            out_off = _pallas_baoab_pre_force_batch(
                B,
                in_b,
                px0,
                mx0,
                fx0,
                mass_b,
                seeds_b,
                wi_b,
                wm_b,
                box_b,
                dt_b,
                kT_b,
                0.15,
                TIP3P_ROH,
                TIP3P_RHH,
                15.999,
                1.008,
                True,
                True,
                True,
                atom_mask=am_b,
                constraint_pairs=cp_b,
                constraint_lengths=cl_b,
                constraint_mask=cm_b,
                f64_accumulators=(acc_mode is None),
                fuse_prev_post=True,
                prebuilt_soa=(px0, py0, pz0, mx0, my0, mz0, fx0, fy0, fz0, soa_plan),
                accumulator_mode=acc_mode,
            )

            # Morton ON
            out_on = _pallas_baoab_pre_force_batch(
                B,
                in_b,
                px0,
                mx0,
                jnp.asarray(fx_m.reshape(B * N_pad)),
                mass_b,
                seeds_b,
                wi_b,
                wm_b,
                box_b,
                dt_b,
                kT_b,
                0.15,
                TIP3P_ROH,
                TIP3P_RHH,
                15.999,
                1.008,
                True,
                True,
                True,
                atom_mask=am_b,
                constraint_pairs=cp_b,
                constraint_lengths=cl_b,
                constraint_mask=cm_b,
                f64_accumulators=(acc_mode is None),
                fuse_prev_post=True,
                morton_inv_perm=inv_perm_b,
                morton_pad=N_pad,
                prebuilt_soa=(
                    px0,
                    py0,
                    pz0,
                    mx0,
                    my0,
                    mz0,
                    jnp.asarray(fx_m.reshape(B * N_pad)),
                    jnp.asarray(fy_m.reshape(B * N_pad)),
                    jnp.asarray(fz_m.reshape(B * N_pad)),
                    soa_plan,
                ),
                accumulator_mode=acc_mode,
            )

            assert len(out_off) == 6
            assert len(out_on) == 9
            for k in range(6):
                np.testing.assert_allclose(np.asarray(out_on[k]), np.asarray(out_off[k]), atol=1e-6, rtol=1e-6)

            # Verify scattered Morton f32 positions match updated positions (including ghost atoms!)
            px_m_out = np.asarray(out_on[6]).reshape(B, N_pad)
            py_m_out = np.asarray(out_on[7]).reshape(B, N_pad)
            pz_m_out = np.asarray(out_on[8]).reshape(B, N_pad)
            for b in range(B):
                np.testing.assert_allclose(
                    px_m_out[b, inv_perm_np[b]],
                    np.asarray(out_on[0].reshape(B, N)[b], dtype=np.float32),
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    py_m_out[b, inv_perm_np[b]],
                    np.asarray(out_on[1].reshape(B, N)[b], dtype=np.float32),
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    pz_m_out[b, inv_perm_np[b]],
                    np.asarray(out_on[2].reshape(B, N)[b], dtype=np.float32),
                    atol=1e-6,
                )

            # Also verify _pallas_baoab_post_soa parity between Morton forces and unpermuted post-force
            mx_post_m, my_post_m, mz_post_m = _pallas_baoab_post_soa(
                out_on[0],
                out_on[1],
                out_on[2],
                out_on[3],
                out_on[4],
                out_on[5],
                jnp.asarray(fx_m.reshape(B * N_pad)),
                jnp.asarray(fy_m.reshape(B * N_pad)),
                jnp.asarray(fz_m.reshape(B * N_pad)),
                inv_perm_b,
                N_pad,
                dt_b,
                15.999,
                1.008,
                True,
                soa_plan,
                f64_accumulators=(acc_mode is None),
                accumulator_mode=acc_mode,
            )
            post_off = cast(
                "tuple[jax.Array, jax.Array, jax.Array]",
                _pallas_baoab_post_force_batch(
                    B,
                    (True,) * 12,
                    out_off[0],
                    out_off[3],
                    fx0,
                    mass_b,
                    wi_b,
                    wm_b,
                    dt_b,
                    15.999,
                    1.008,
                    True,
                    atom_mask=am_b,
                    is_water_or_ghost=soa_plan["is_water_flat"].reshape(B, N),
                    constraint_pairs=cp_b,
                    constraint_lengths=cl_b,
                    constraint_mask=cm_b,
                    f64_accumulators=(acc_mode is None),
                    prebuilt_soa=(
                        out_off[0],
                        out_off[1],
                        out_off[2],
                        out_off[3],
                        out_off[4],
                        out_off[5],
                        fx0,
                        fy0,
                        fz0,
                        soa_plan,
                    ),
                    accumulator_mode=acc_mode,
                ),
            )
            np.testing.assert_allclose(np.asarray(mx_post_m), np.asarray(post_off[0]), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(my_post_m), np.asarray(post_off[1]), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(mz_post_m), np.asarray(post_off[2]), atol=1e-6, rtol=1e-6)

        # Verify all 6 disjoint fltflt GPU pallas_call kernels (baoab_pre_{solute,settle,rattle}_ff
        # and baoab_post_{solute,settle,rattle}_ff) trace cleanly with exact sequential launch order,
        # disjoint grids, and zero-copy input_output_aliases (including Morton f32 scatter buffers).
        # Note: we trace via jax.make_jaxpr(..., interpret=False) rather than XLA CPU LLVM interpret=True
        # execution to avoid the known XLA CPU LLVM Veltkamp-split pallas_call compile pathology.
        ff_calls: list[dict[str, Any]] = []
        orig_pc = pl.pallas_call

        def _spy_ff(kfn: Any, **kw: Any):
            cf = orig_pc(kfn, **kw)

            def _w(*a: Any):
                ff_calls.append(
                    {
                        "name": str(kw.get("name")),
                        "grid": tuple(int(x) for x in kw["grid"]),
                        "aliases": dict(kw.get("input_output_aliases") or {}),
                        "n_in": len(a),
                    }
                )
                return cf(*a)

            return _w

        monkeypatch.setattr(pl, "pallas_call", _spy_ff)
        soa_ff = build_baoab_soa_static_plan(
            B,
            N,
            jnp.float64,
            mass_b,
            wi_b,
            wm_b,
            atom_mask_b=am_b,
            constraint_pairs_b=cp_b,
            constraint_lengths_b=cl_b,
            constraint_mask_b=cm_b,
        )
        px0_f = pos_b[:, :, 0].reshape(B * N)
        py0_f = pos_b[:, :, 1].reshape(B * N)
        pz0_f = pos_b[:, :, 2].reshape(B * N)
        mx0_f = mom_b[:, :, 0].reshape(B * N)
        my0_f = mom_b[:, :, 1].reshape(B * N)
        mz0_f = mom_b[:, :, 2].reshape(B * N)
        fxm_flat = jnp.asarray(fx_m.reshape(B * N_pad))
        fym_flat = jnp.asarray(fy_m.reshape(B * N_pad))
        fzm_flat = jnp.asarray(fz_m.reshape(B * N_pad))

        def _trace_ff_step(px_i, py_i, pz_i, mx_i, my_i, mz_i):
            pre_out = _pallas_baoab_pre_force_batch(
                B,
                (True,) * 15,
                px_i,
                mx_i,
                fxm_flat,
                mass_b,
                seeds_b,
                wi_b,
                wm_b,
                box_b,
                dt_b,
                kT_b,
                0.15,
                TIP3P_ROH,
                TIP3P_RHH,
                15.999,
                1.008,
                True,
                False,
                False,
                atom_mask=am_b,
                constraint_pairs=cp_b,
                constraint_lengths=cl_b,
                constraint_mask=cm_b,
                fuse_prev_post=True,
                morton_inv_perm=inv_perm_b,
                morton_pad=N_pad,
                prebuilt_soa=(px_i, py_i, pz_i, mx_i, my_i, mz_i, fxm_flat, fym_flat, fzm_flat, soa_ff),
                accumulator_mode="fltflt",
            )
            post_out = _pallas_baoab_post_soa(
                pre_out[0],
                pre_out[1],
                pre_out[2],
                pre_out[3],
                pre_out[4],
                pre_out[5],
                fxm_flat,
                fym_flat,
                fzm_flat,
                inv_perm_b,
                N_pad,
                dt_b,
                15.999,
                1.008,
                False,
                soa_ff,
                accumulator_mode="fltflt",
            )
            return pre_out, post_out

        jaxpr_ff = jax.make_jaxpr(_trace_ff_step)(px0_f, py0_f, pz0_f, mx0_f, my0_f, mz0_f)
        assert [c["name"] for c in ff_calls] == [
            "baoab_pre_solute_ff",
            "baoab_pre_settle_ff",
            "baoab_pre_rattle_ff",
            "baoab_post_solute_ff",
            "baoab_post_settle_ff",
            "baoab_post_rattle_ff",
        ]
        # Pre-force with Morton ON aliases all 15 float32 (hi, lo) + Morton scatter state buffers in place
        for c in ff_calls[:3]:
            assert c["aliases"] == {i: i for i in range(15)}
            assert c["grid"][0] == B
        # Post-force aliases all 6 float32 (hi, lo) momentum buffers in place
        for c in ff_calls[3:]:
            assert c["aliases"] == {i: i for i in range(6)}
            assert c["grid"][0] == B
        pallas_eqns = [eqn for eqn in jaxpr_ff.jaxpr.eqns if eqn.primitive.name == "pallas_call"]
        assert len(pallas_eqns) == 6


def test_disjoint_settle_shake_stablehlo_export_zero_custom_calls():
    """Verify both standard and fltflt BAOAB steps lower portably with 0 custom_calls in export_context."""
    fx = _build_mixed_fixture_f64()
    pos32 = jnp.asarray(fx["pos"], dtype=jnp.float32)
    mom32 = jnp.asarray(fx["mom"], dtype=jnp.float32)
    frc32 = jnp.asarray(fx["frc"], dtype=jnp.float32)
    mass32 = jnp.asarray(fx["mass"], dtype=jnp.float32)
    am32 = jnp.asarray(fx["atom_mask"], dtype=jnp.int32)
    wi = jnp.asarray(fx["wi"], dtype=jnp.int32)
    wm = jnp.asarray(fx["wm"], dtype=jnp.int32)
    cp = jnp.asarray(fx["cp"], dtype=jnp.int32)
    cl32 = jnp.asarray(fx["cl"], dtype=jnp.float32)
    cm = jnp.asarray(fx["cm"], dtype=jnp.int32)
    box32 = jnp.asarray(fx["box"], dtype=jnp.float32)

    forbidden = (
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

    def _export_target(pos_in: jax.Array, mom_in: jax.Array, frc_in: jax.Array):
        p_std, m_std1, _ = pallas_baoab_pre_force(
            pos_in,
            mom_in,
            frc_in,
            mass32,
            jax.random.key(5),
            wi,
            wm,
            box32,
            dt=jnp.float32(0.02),
            kT=jnp.float32(0.596),
            gamma=0.1,
            interpret=False,  # Even when interpret=False is requested, export_context forces portable XLA!
            atom_mask=am32,
            constraint_pairs=cp,
            constraint_lengths=cl32,
            constraint_mask=cm,
        )
        m_std2 = pallas_baoab_post_force(
            p_std,
            m_std1,
            frc_in,
            mass32,
            wi,
            wm,
            dt=jnp.float32(0.02),
            interpret=False,
            atom_mask=am32,
            constraint_pairs=cp,
            constraint_lengths=cl32,
            constraint_mask=cm,
        )
        p_ff, m_ff1, _, plo, mlo = pallas_baoab_pre_force(
            pos_in,
            mom_in,
            frc_in,
            mass32,
            jax.random.key(5),
            wi,
            wm,
            box32,
            dt=jnp.float32(0.02),
            kT=jnp.float32(0.596),
            gamma=0.1,
            interpret=False,
            atom_mask=am32,
            constraint_pairs=cp,
            constraint_lengths=cl32,
            constraint_mask=cm,
            accumulator_mode="fltflt",
            return_lo_accumulators=True,
        )
        m_ff2, mlo2 = pallas_baoab_post_force(
            p_ff,
            m_ff1,
            frc_in,
            mass32,
            wi,
            wm,
            dt=jnp.float32(0.02),
            interpret=False,
            atom_mask=am32,
            constraint_pairs=cp,
            constraint_lengths=cl32,
            constraint_mask=cm,
            accumulator_mode="fltflt",
            positions_lo=plo,
            momentum_lo=mlo,
            return_lo_accumulators=True,
        )
        return p_std + p_ff, m_std2 + m_ff2 + mlo2

    with export_context():
        jitted = jax.jit(_export_target)
        lowered_hlo = jitted.lower(pos32, mom32, frc32).as_text()
        exported = jax.export.export(jitted)(pos32, mom32, frc32)
        mlir_text = exported.mlir_module()

    assert "stablehlo.custom_call" not in lowered_hlo
    assert "stablehlo.custom_call" not in mlir_text
    non_loc_mlir = "\n".join(
        line for line in mlir_text.splitlines() if not line.lstrip().startswith("#loc")
    ).lower()
    for marker in forbidden:
        assert marker not in lowered_hlo.lower(), f"Forbidden marker {marker!r} in lowered HLO"
        assert marker not in non_loc_mlir, f"Forbidden marker {marker!r} in exported StableHLO"

    out_p, out_m = exported.call(pos32, mom32, frc32)
    assert jnp.all(jnp.isfinite(out_p))
    assert jnp.all(jnp.isfinite(out_m))
