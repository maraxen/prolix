"""v3.6 Track 2 verification: P0-B RATTLE velocity fix, P1-A Pallas fltflt, and conductor-limit RF."""

from __future__ import annotations

import ast
import inspect
import math
from typing import cast

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import pytest

import prolix.export
from prolix.constants import COULOMB_CONSTANT
from prolix.pallas_kernels import (
    build_tile_pair_neighbor_list,
    nonbonded_nl_reference,
    pallas_bonded_and_nonbonded_nl_fused,
    pallas_nonbonded_nl,
    pallas_nonbonded_tile_pairs,
    reaction_field_coefficients,
)
from prolix.physics import fltflt_pallas as ff_mod
from prolix.physics.fltflt_pallas import (
    FltFlt,
    fast_two_sum,
    ff_add,
    ff_div,
    ff_from_f32,
    ff_from_float64,
    ff_min_image_f32,
    ff_mod_f32,
    ff_mul,
    ff_norm3d,
    ff_sqrt,
    ff_sqrt_fast,
    ff_sub,
    ff_to_f32,
    ff_to_float64,
    fltflt_triton_asm_context,
    should_use_triton_asm,
    two_prod_fma,
    two_sum,
)
from prolix.physics.settle import (
    TIP3P_RHH,
    TIP3P_ROH,
    pallas_baoab_interstep,
    pallas_baoab_post_force,
    pallas_baoab_pre_force,
    settle_langevin,
)

FORBIDDEN_STABLEHLO_TARGETS = (
    "triton",
    "mosaic",
    "pallas",
    "tpu_custom_call",
    "xla.gpu",
    "lapack",
    "cusolver",
    "cublas",
    "cudnn",
    "inline_asm",
)


def _make_water_and_ch3_system(offset: float = 0.0, dtype=jnp.float64):
    """Build 1 NH3+-like 3-H star cluster + 1 CH cluster + 1 TIP3P water + 1 free solute atom (N=10)."""
    # Cluster 0: N(0) bonded to H(1), H(2), H(3) with bond length d_NH = 1.01 A
    d_nh = 1.01
    tet_z = -d_nh / 3.0
    tet_r = d_nh * math.sqrt(8.0 / 9.0)
    r_n = np.array([5.0, 5.0, 5.0], dtype=np.float64) + offset
    r_h1 = r_n + np.array([tet_r, 0.0, tet_z])
    r_h2 = r_n + np.array([-0.5 * tet_r, 0.5 * math.sqrt(3.0) * tet_r, tet_z])
    r_h3 = r_n + np.array([-0.5 * tet_r, -0.5 * math.sqrt(3.0) * tet_r, tet_z])

    # Cluster 1: C(4) bonded to H(5) with bond length d_CH = 1.09 A
    d_ch = 1.09
    r_c = np.array([9.0, 5.0, 5.0], dtype=np.float64) + offset
    r_hc = r_c + np.array([0.0, d_ch, 0.0])

    # Water: O(6), H(7), H(8) at exact TIP3P geometry
    half_hh = 0.5 * TIP3P_RHH
    y_h = math.sqrt(TIP3P_ROH * TIP3P_ROH - half_hh * half_hh)
    r_o = np.array([5.0, 9.5, 5.0], dtype=np.float64) + offset
    r_wh1 = r_o + np.array([half_hh, y_h, 0.0])
    r_wh2 = r_o + np.array([-half_hh, y_h, 0.0])

    # Unconstrained solute atom (9)
    r_sol = np.array([9.5, 9.5, 5.0], dtype=np.float64) + offset

    positions = jnp.asarray(
        np.stack([r_n, r_h1, r_h2, r_h3, r_c, r_hc, r_o, r_wh1, r_wh2, r_sol], axis=0),
        dtype=dtype,
    )
    mass = jnp.asarray(
        [14.007, 1.008, 1.008, 1.008, 12.011, 1.008, 15.999, 1.008, 1.008, 39.948],
        dtype=dtype,
    )
    water_indices = jnp.asarray([[6, 7, 8]], dtype=jnp.int32)
    water_mask = jnp.asarray([1], dtype=jnp.int32)
    constraint_pairs = jnp.asarray([[0, 1], [0, 2], [0, 3], [4, 5]], dtype=jnp.int32)
    constraint_lengths = jnp.asarray([d_nh, d_nh, d_nh, d_ch], dtype=dtype)
    constraint_mask = jnp.asarray([1, 1, 1, 1], dtype=jnp.int32)
    return (
        positions,
        mass,
        water_indices,
        water_mask,
        constraint_pairs,
        constraint_lengths,
        constraint_mask,
    )


def _max_bond_and_vel_errors(
    positions: jax.Array,
    momentum: jax.Array,
    mass: jax.Array,
    constraint_pairs: jax.Array,
    constraint_lengths: jax.Array,
) -> tuple[float, float]:
    pos64 = np.asarray(positions, dtype=np.float64)
    vel64 = np.asarray(momentum, dtype=np.float64) / np.asarray(mass, dtype=np.float64)[:, None]
    pairs = np.asarray(constraint_pairs, dtype=np.int32)
    lengths = np.asarray(constraint_lengths, dtype=np.float64)
    dr = pos64[pairs[:, 1]] - pos64[pairs[:, 0]]
    dist = np.linalg.norm(dr, axis=-1)
    unit = dr / dist[:, None]
    dv = vel64[pairs[:, 1]] - vel64[pairs[:, 0]]
    rad_vel = np.abs(np.sum(dv * unit, axis=-1))
    return float(np.max(np.abs(dist - lengths))), float(np.max(rad_vel))


# =============================================================================
# Deliverable 1: P0-B RATTLE Spurious Post-R2 Velocity Projection Fix
# =============================================================================


def test_p0b_rattle_single_step_no_spurious_post_r2_projection():
    """Verify _do_rattle does not damp rotational momentum via a post-R2 velocity projection."""
    with jax.enable_x64(True):
        pos, mass, wi, wm, cp, cl, cm = _make_water_and_ch3_system(dtype=jnp.float64)
        # Pure rigid-body rotation of the NH3+ cluster around z-axis through N(0):
        # v_i = omega x (r_i - r_N) satisfies (v_H - v_N) . (r_H - r_N) == 0 initially!
        omega = np.array([0.0, 0.0, 0.35], dtype=np.float64)  # rad / tau_AKMA
        pos_np = np.asarray(pos, dtype=np.float64)
        mass_np = np.asarray(mass, dtype=np.float64)
        vel_np = np.zeros_like(pos_np)
        for idx in (1, 2, 3):
            vel_np[idx] = np.cross(omega, pos_np[idx] - pos_np[0])
        vel_np[5] = np.array([0.25, 0.0, -0.15])  # orthogonal to C(4)-H(5) along +y
        mom0 = jnp.asarray(vel_np * mass_np[:, None], dtype=jnp.float64)
        zero_force = jnp.zeros_like(pos)
        dt_akma = 2.0 / 48.88821291839  # 2.0 fs in AKMA

        # Zero friction (gamma=0) + zero force: symplectic RATTLE step must conserve cluster KE!
        pos1, mom_pre_b2, _ = pallas_baoab_pre_force(
            pos,
            mom0,
            zero_force,
            mass,
            jax.random.key(0),
            wi,
            wm,
            box=None,
            dt=dt_akma,
            kT=0.0,
            gamma=0.0,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl,
            constraint_mask=cm,
            f64_accumulators=True,
        )
        mom1 = pallas_baoab_post_force(
            pos1,
            mom_pre_b2,
            zero_force,
            mass,
            wi,
            wm,
            dt=dt_akma,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl,
            constraint_mask=cm,
            f64_accumulators=True,
        )

        bond_err, vel_err = _max_bond_and_vel_errors(pos1, mom1, mass, cp, cl)
        assert bond_err < 1e-10, f"RATTLE bond length error too high: {bond_err}"
        assert vel_err < 1e-10, f"Post-B2 RATTLE velocity constraint error too high: {vel_err}"

        # Compare single-step kinetic energy against initial KE:
        # With F=0, gamma=0, before B2 (which adds 0 when F=0), a spurious post-R2 projection
        # PLUS post-B2 projection is idempotent on 1 step ONLY AFTER B2, wait: at R2 the impulse
        # is along ab1 (not ab2), so post-B2 projects onto ab2 ONCE. Under the old bug, when F != 0,
        # projecting onto ab2 BEFORE B2 discarded O(dt * omega^2) radial velocity that would have
        # interacted symplectically with B2! Test with nonzero central harmonic force too:
        ke0 = float(0.5 * jnp.sum((mom0 * mom0) / mass[:, None]))
        ke1 = float(0.5 * jnp.sum((mom1 * mom1) / mass[:, None]))
        assert abs(ke1 - ke0) / ke0 < 1e-5, f"Single-step rotational KE error: {ke0} -> {ke1}"

        # With nonzero conservative force, verify pre-B2 momentum is NOT orthogonally projected onto ab2
        # (because R2's Lagrange impulse is parallel to ab1, so (v_pre_b2[H] - v_pre_b2[X]) . ab2 != 0,
        # whereas post-B2 momentum IS orthogonal to ab2!).
        f_ext = jnp.asarray(
            np.array(
                [
                    [1.2, -0.8, 0.4],
                    [-0.5, 0.9, -0.3],
                    [0.4, -0.6, 0.7],
                    [-0.3, 0.5, -0.6],
                    [0.7, -0.4, 0.2],
                    [-0.7, 0.4, -0.2],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.1, -0.2, 0.3],
                ]
            ),
            dtype=jnp.float64,
        )
        pos_f, mom_pre_f, _ = pallas_baoab_pre_force(
            pos,
            mom0,
            f_ext,
            mass,
            jax.random.key(1),
            wi,
            wm,
            box=None,
            dt=dt_akma,
            kT=0.0,
            gamma=0.0,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl,
            constraint_mask=cm,
            f64_accumulators=True,
        )
        _, pre_b2_rad_vel = _max_bond_and_vel_errors(pos_f, mom_pre_f, mass, cp, cl)
        # Before B2, radial velocity along ab2 MUST be nonzero O(dt^2) (not zeroed by a spurious projection!)
        assert pre_b2_rad_vel > 1e-6, (
            f"Expected nonzero pre-B2 radial velocity along ab2 (symplectic R2 along ab1), got {pre_b2_rad_vel}"
        )
        mom_post_f = pallas_baoab_post_force(
            pos_f,
            mom_pre_f,
            f_ext,
            mass,
            wi,
            wm,
            dt=dt_akma,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl,
            constraint_mask=cm,
            f64_accumulators=True,
        )
        bond_err_f, post_b2_rad_vel = _max_bond_and_vel_errors(pos_f, mom_post_f, mass, cp, cl)
        assert bond_err_f < 1e-10
        assert post_b2_rad_vel < 1e-10


def test_p0b_rattle_multistep_energy_and_interstep_parity():
    """Verify multi-step NVE total energy conservation with harmonic forces + HBonds RATTLE + interstep fusion."""
    with jax.enable_x64(True):
        pos0, mass, wi, wm, cp, cl, cm = _make_water_and_ch3_system(dtype=jnp.float64)
        k_trap = 1.5  # kcal / (mol * A^2)
        dt_akma = 1.5 / 48.88821291839

        def potential_fn(r):
            dr = r - pos0
            return 0.5 * k_trap * jnp.sum(dr * dr)

        force_fn = jax.jit(lambda r: -jax.grad(potential_fn)(r))

        # Initialize velocities with rotational + translational excitation projected onto constraints
        rng = np.random.default_rng(42)
        raw_p = jnp.asarray(rng.normal(0.0, 1.2, size=pos0.shape), dtype=jnp.float64)
        mom0 = pallas_baoab_post_force(
            pos0,
            raw_p,
            jnp.zeros_like(pos0),
            mass,
            wi,
            wm,
            dt=dt_akma,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl,
            constraint_mask=cm,
            f64_accumulators=True,
        )
        f0 = force_fn(pos0)
        e0 = float(potential_fn(pos0) + 0.5 * jnp.sum((mom0 * mom0) / mass[:, None]))

        @jax.jit
        def run_split_steps(r_init, p_init, f_init):
            def body(i, carry):
                r_c, p_c, f_c, key_c = carry
                r_n, p_pre, key_n = pallas_baoab_pre_force(
                    r_c,
                    p_c,
                    f_c,
                    mass,
                    key_c,
                    wi,
                    wm,
                    box=None,
                    dt=dt_akma,
                    kT=0.0,
                    gamma=0.0,
                    interpret=True,
                    constraint_pairs=cp,
                    constraint_lengths=cl,
                    constraint_mask=cm,
                    f64_accumulators=True,
                )
                f_n = -jax.grad(potential_fn)(r_n)
                p_n = pallas_baoab_post_force(
                    r_n,
                    p_pre,
                    f_n,
                    mass,
                    wi,
                    wm,
                    dt=dt_akma,
                    interpret=True,
                    constraint_pairs=cp,
                    constraint_lengths=cl,
                    constraint_mask=cm,
                    f64_accumulators=True,
                )
                return (r_n, p_n, f_n, key_n)

            return jax.lax.fori_loop(0, 80, body, (r_init, p_init, f_init, jax.random.key(7)))

        pos_fin, mom_fin, _, _ = run_split_steps(pos0, mom0, f0)
        e_fin = float(potential_fn(pos_fin) + 0.5 * jnp.sum((mom_fin * mom_fin) / mass[:, None]))
        bond_err, vel_err = _max_bond_and_vel_errors(pos_fin, mom_fin, mass, cp, cl)
        assert bond_err < 1e-8
        assert vel_err < 1e-9
        # Symplectic RATTLE NVE conserves total shadow Hamiltonian without secular KE drain!
        assert abs(e_fin - e0) / max(abs(e0), 1.0) < 2e-3, f"Secular energy drift: {e0} -> {e_fin}"

        # Also verify MK-66 fused interstep (`pallas_baoab_interstep`) matches split post(t)+pre(t+1) identically!
        r1, p1_pre, k1 = pallas_baoab_pre_force(
            pos0,
            mom0,
            f0,
            mass,
            jax.random.key(99),
            wi,
            wm,
            box=None,
            dt=dt_akma,
            kT=0.596,
            gamma=0.2,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl,
            constraint_mask=cm,
            f64_accumulators=True,
        )
        f1 = force_fn(r1)
        p1_post = pallas_baoab_post_force(
            r1,
            p1_pre,
            f1,
            mass,
            wi,
            wm,
            dt=dt_akma,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl,
            constraint_mask=cm,
            f64_accumulators=True,
        )
        r2_split, p2_split, _ = pallas_baoab_pre_force(
            r1,
            p1_post,
            f1,
            mass,
            k1,
            wi,
            wm,
            box=None,
            dt=dt_akma,
            kT=0.596,
            gamma=0.2,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl,
            constraint_mask=cm,
            f64_accumulators=True,
        )
        r2_fused, p2_fused, _ = pallas_baoab_interstep(
            r1,
            p1_pre,
            f1,
            mass,
            k1,
            wi,
            wm,
            box=None,
            dt=dt_akma,
            kT=0.596,
            gamma=0.2,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl,
            constraint_mask=cm,
            f64_accumulators=True,
        )
        np.testing.assert_allclose(np.asarray(r2_fused), np.asarray(r2_split), atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(np.asarray(p2_fused), np.asarray(p2_split), atol=1e-10, rtol=1e-10)


# =============================================================================
# Deliverable 2: P1-A Pallas fltflt ~48-Bit Arithmetic & BAOAB Accumulator Mode
# =============================================================================


def test_p1a_fltflt_primitives_48bit_precision_vs_float64():
    """Verify two_sum, fast_two_sum, two_prod_fma, ff_add/sub/mul/div/sqrt/norm3d achieve ~48-bit precision."""
    rng = np.random.default_rng(20261117)
    a64 = rng.uniform(0.5, 50.0, size=(256,)) * rng.choice([-1.0, 1.0], size=(256,))
    b64 = rng.uniform(0.5, 50.0, size=(256,)) * rng.choice([-1.0, 1.0], size=(256,))

    # Exact float32 inputs: two_sum and two_prod_fma must be 0-error EFTs wrt float64!
    a32 = a64.astype(np.float32)
    b32 = b64.astype(np.float32)
    ts = two_sum(jnp.asarray(a32), jnp.asarray(b32))
    tp = two_prod_fma(jnp.asarray(a32), jnp.asarray(b32))
    with jax.enable_x64(True):
        np.testing.assert_allclose(
            np.asarray(ff_to_float64(ts)),
            a32.astype(np.float64) + b32.astype(np.float64),
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(ff_to_float64(tp)),
            a32.astype(np.float64) * b32.astype(np.float64),
            atol=0.0,
            rtol=0.0,
        )

        # Full ~48-bit FltFlt arithmetic (inputs carry populated lo limb!)
        aff = ff_from_float64(jnp.asarray(a64, dtype=jnp.float64))
        bff = ff_from_float64(jnp.asarray(b64, dtype=jnp.float64))
        p64 = np.abs(a64) + 0.25
        pff = ff_from_float64(jnp.asarray(p64, dtype=jnp.float64))

        for op_ff, ref64, name in (
            (ff_add(aff, bff), a64 + b64, "ff_add"),
            (ff_sub(aff, bff), a64 - b64, "ff_sub"),
            (ff_mul(aff, bff), a64 * b64, "ff_mul"),
            (ff_div(aff, bff), a64 / b64, "ff_div"),
            (ff_sqrt(pff), np.sqrt(p64), "ff_sqrt"),
            (ff_sqrt_fast(pff), np.sqrt(p64), "ff_sqrt_fast"),
        ):
            assert op_ff.hi.dtype == jnp.float32
            assert op_ff.lo.dtype == jnp.float32
            got64 = np.asarray(ff_to_float64(op_ff))
            rel_err = np.max(np.abs(got64 - ref64) / np.maximum(np.abs(ref64), 1e-12))
            assert rel_err < 2e-13, f"{name} relative error {rel_err:.3e} exceeded ~45-48 bit bound"

        # 3D Euclidean norm and periodic minimum-image / modulo wrapping at large coordinates
        x_ff = ff_from_float64(jnp.asarray(128.0 + 1e-6 * rng.standard_normal(64), dtype=jnp.float64))
        y_ff = ff_from_float64(jnp.asarray(-96.0 + 1e-6 * rng.standard_normal(64), dtype=jnp.float64))
        z_ff = ff_from_float64(jnp.asarray(64.0 + 1e-6 * rng.standard_normal(64), dtype=jnp.float64))
        n_ff = ff_norm3d(x_ff, y_ff, z_ff)
        x64 = np.asarray(ff_to_float64(x_ff))
        y64 = np.asarray(ff_to_float64(y_ff))
        z64 = np.asarray(ff_to_float64(z_ff))
        n_ref = np.sqrt(x64 * x64 + y64 * y64 + z64 * z64)
        np.testing.assert_allclose(np.asarray(ff_to_float64(n_ff)), n_ref, rtol=2e-13, atol=1e-12)

        L = jnp.float32(32.0)
        wrapped_ff = ff_mod_f32(x_ff, L)
        wrapped_ref = np.mod(x64, 32.0)
        np.testing.assert_allclose(np.asarray(ff_to_float64(wrapped_ff)), wrapped_ref, rtol=1e-13, atol=1e-12)


def test_p1a_fltflt_sub_ulp_serial_accumulation():
    """Demonstrate fltflt resolves sub-float32-ULP coordinate increments where float32 stalls at 0."""
    x0 = np.float32(128.0)  # ULP(128.0f) = 2^-16 = 1.5258789e-5
    dx = np.float32(3.0e-6)  # < 0.2 ULP -> x0 + dx == x0 in IEEE-754 float32!
    n_steps = 5000

    @jax.jit
    def accumulate_both():
        def body(_, carry):
            x_f32, hi, lo = carry
            x_f32_next = x_f32 + jnp.float32(dx)
            ff_next = ff_mod.ff_add_f32(FltFlt(hi, lo), jnp.float32(dx))
            return (x_f32_next, ff_next.hi, ff_next.lo)

        return jax.lax.fori_loop(
            0, n_steps, body, (jnp.float32(x0), jnp.float32(x0), jnp.float32(0.0))
        )

    x_f32_fin, hi_fin, lo_fin = accumulate_both()
    # Naive float32 completely stalls (0.0 total displacement!)
    assert float(x_f32_fin) == float(x0)
    # FltFlt captures all 5,000 sub-ULP increments to ~48 bits!
    expected = float(x0) + n_steps * float(dx)
    got = float(np.float64(hi_fin) + np.float64(lo_fin))
    assert abs(got - expected) < 1e-11, f"Expected {expected:.12f}, got {got:.12f}"


def test_p1a_fltflt_ptx_inline_asm_contract_and_ast_safety():
    """Verify PTX rn.f32 opcodes (without .ftz), context gating, and Pallas AST rules on fltflt_pallas."""
    src = inspect.getsource(ff_mod)
    for expected_ptx in (
        "add.rn.f32 $0, $1, $2;",
        "sub.rn.f32 $0, $1, $2;",
        "mul.rn.f32 $0, $1, $2;",
        "fma.rn.f32 $0, $1, $2, $3;",
    ):
        assert expected_ptx in src, f"Missing IEEE-754 round-to-nearest PTX instruction {expected_ptx!r}"
    assert ".ftz" not in src.split("def fadd_rn", 1)[1], "PTX instructions must not enable .ftz"

    # AST check: zero ast.Pow (`**`) inside fltflt_pallas (required for Pallas Triton lowering)
    tree = ast.parse(src)
    pow_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pow)]
    assert not pow_nodes, f"Found forbidden ast.Pow inside fltflt_pallas.py: {pow_nodes}"

    # Context gating: always False under export_context() or CPU backend
    assert not should_use_triton_asm()
    with prolix.export.export_context(), fltflt_triton_asm_context(True):
        assert not should_use_triton_asm()


def test_p1a_settle_baoab_fltflt_accumulator_mode():
    """Verify settle.py accumulator_mode='fltflt' preserves sub-ULP drifts and exact constraints."""
    with jax.enable_x64(True):
        # Place whole system at +256.0 A (where float32 ULP is 3.05e-5 A!)
        pos64, mass64, wi, wm, cp, cl64, cm = _make_water_and_ch3_system(
            offset=256.0, dtype=jnp.float64
        )
        # Tiny uniform velocity v = 8e-5 A/tau_AKMA, dt = 0.04 tau_AKMA -> half_dt * v = 1.6e-6 A
        # (< 0.06 float32 ULP at 256 A! Pure float32 stalls completely on the solute atom!)
        v_tiny = 8.0e-5
        dt_akma = 0.04
        mom64 = mass64[:, None] * jnp.broadcast_to(
            jnp.asarray([v_tiny, -v_tiny, 0.5 * v_tiny], dtype=jnp.float64), pos64.shape
        )
        zero_f32 = jnp.zeros(pos64.shape, dtype=jnp.float32)

        # 1. Pure float32 state stalls on sub-ULP drift at +256 A
        pos_f32_1, _, _ = pallas_baoab_pre_force(
            pos64.astype(jnp.float32),
            mom64.astype(jnp.float32),
            zero_f32,
            mass64.astype(jnp.float32),
            jax.random.key(0),
            wi,
            wm,
            box=None,
            dt=jnp.float32(dt_akma),
            kT=jnp.float32(0.0),
            gamma=0.0,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl64.astype(jnp.float32),
            constraint_mask=cm,
        )
        # Unconstrained atom 9 did not move a single bit in naive float32!
        assert float(jnp.max(jnp.abs(pos_f32_1[9] - pos64.astype(jnp.float32)[9]))) == 0.0

        # 2. Pure float32 state + return_lo_accumulators=True (zero float64 arrays anywhere!)
        pos_hi, mom_hi, _, pos_lo, mom_lo = pallas_baoab_pre_force(
            pos64.astype(jnp.float32),
            mom64.astype(jnp.float32),
            zero_f32,
            mass64.astype(jnp.float32),
            jax.random.key(0),
            wi,
            wm,
            box=None,
            dt=jnp.float32(dt_akma),
            kT=jnp.float32(0.0),
            gamma=0.0,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl64.astype(jnp.float32),
            constraint_mask=cm,
            accumulator_mode="fltflt",
            return_lo_accumulators=True,
        )
        assert pos_hi.dtype == jnp.float32
        assert pos_lo.dtype == jnp.float32
        mom_hi_2, mom_lo_2 = pallas_baoab_post_force(
            pos_hi,
            mom_hi,
            zero_f32,
            mass64.astype(jnp.float32),
            wi,
            wm,
            dt=jnp.float32(dt_akma),
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl64.astype(jnp.float32),
            constraint_mask=cm,
            accumulator_mode="fltflt",
            positions_lo=pos_lo,
            momentum_lo=mom_lo,
            return_lo_accumulators=True,
        )
        assert mom_hi_2.dtype == jnp.float32
        assert mom_lo_2.dtype == jnp.float32

        # Check sub-ULP displacement of unconstrained solute atom 9 was captured to ~48 bits!
        disp_atom9 = (
            np.asarray(pos_hi[9], dtype=np.float64) + np.asarray(pos_lo[9], dtype=np.float64)
        ) - np.asarray(pos64[9].astype(jnp.float32), dtype=np.float64)
        expected_disp9 = dt_akma * np.array([v_tiny, -v_tiny, 0.5 * v_tiny], dtype=np.float64)
        np.testing.assert_allclose(disp_atom9, expected_disp9, rtol=1e-5, atol=1e-12)

        # 3. Float64 caller interface with accumulator_mode="fltflt" vs f64_accumulators=True reference
        pos_ff64, mom_ff64_pre, pos_f32_emit = None, None, None
        pos_ff64, mom_ff64_pre, _, pos_f32_emit = pallas_baoab_pre_force(
            pos64,
            mom64,
            zero_f32,
            mass64,
            jax.random.key(11),
            wi,
            wm,
            box=jnp.asarray([512.0, 512.0, 512.0], dtype=jnp.float64),
            dt=dt_akma,
            kT=0.596,
            gamma=0.2,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl64,
            constraint_mask=cm,
            return_pos_f32=True,
            accumulator_mode="fltflt",
        )
        mom_ff64 = pallas_baoab_post_force(
            pos_ff64,
            mom_ff64_pre,
            zero_f32,
            mass64,
            wi,
            wm,
            dt=dt_akma,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl64,
            constraint_mask=cm,
            accumulator_mode="fltflt",
        )
        assert pos_ff64.dtype == jnp.float64
        assert pos_f32_emit.dtype == jnp.float32
        assert mom_ff64.dtype == jnp.float64

        pos_ref64, mom_ref64_pre, _ = pallas_baoab_pre_force(
            pos64,
            mom64,
            zero_f32,
            mass64,
            jax.random.key(11),
            wi,
            wm,
            box=jnp.asarray([512.0, 512.0, 512.0], dtype=jnp.float64),
            dt=dt_akma,
            kT=0.596,
            gamma=0.2,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl64,
            constraint_mask=cm,
            f64_accumulators=True,
        )
        mom_ref64 = pallas_baoab_post_force(
            pos_ref64,
            mom_ref64_pre,
            zero_f32,
            mass64,
            wi,
            wm,
            dt=dt_akma,
            interpret=True,
            constraint_pairs=cp,
            constraint_lengths=cl64,
            constraint_mask=cm,
            f64_accumulators=True,
        )
        # Note: O-step noise is float32 in fltflt mode, so positions match float64 reference to ~1e-7 A,
        # while rigid water and RATTLE bond lengths hold to ~1e-7..1e-11 A even at x = +256 A!
        np.testing.assert_allclose(np.asarray(pos_ff64), np.asarray(pos_ref64), atol=2e-6, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(mom_ff64), np.asarray(mom_ref64), atol=2e-5, rtol=1e-5)

        bond_err_ff, vel_err_ff = _max_bond_and_vel_errors(pos_ff64, mom_ff64, mass64, cp, cl64)
        assert bond_err_ff < 2e-7, f"fltflt RATTLE bond error at +256 A: {bond_err_ff}"
        assert vel_err_ff < 2e-6, f"fltflt RATTLE vel error at +256 A: {vel_err_ff}"
        # Water O-H1, O-H2, H1-H2 distances at +256 A
        r_o, r_h1, r_h2 = np.asarray(pos_ff64[6]), np.asarray(pos_ff64[7]), np.asarray(pos_ff64[8])
        assert abs(np.linalg.norm(r_h1 - r_o) - TIP3P_ROH) < 1e-10
        assert abs(np.linalg.norm(r_h2 - r_o) - TIP3P_ROH) < 1e-10
        assert abs(np.linalg.norm(r_h2 - r_h1) - TIP3P_RHH) < 1e-10

        # End-to-end settle_langevin(..., use_pallas=True, accumulator_mode="fltflt") smoke check
        init_fn, apply_fn = settle_langevin(
            lambda r: jnp.zeros_like(r),
            lambda r, dr: r + dr,
            dt=dt_akma,
            kT=0.596,
            gamma=0.2,
            mass=mass64,
            water_indices=wi,
            water_mask=wm,
            constraints=(cp, cl64),
            is_force_shaped=True,
            use_pallas=True,
            accumulator_mode="fltflt",
        )
        st0 = init_fn(jax.random.key(3), pos64)
        st1 = apply_fn(st0)
        assert jnp.all(jnp.isfinite(st1.positions))
        assert jnp.all(jnp.isfinite(st1.momentum))


# =============================================================================
# Deliverable 3: Reaction Field Conductor Limit (rf_dielectric = inf) & Parity
# =============================================================================


def _dense_reaction_field_energy_f64(
    positions: np.ndarray,
    charges: np.ndarray,
    sigmas: np.ndarray,
    epsilons: np.ndarray,
    box: np.ndarray,
    excl_indices: np.ndarray,
    excl_vdw: np.ndarray,
    excl_elec: np.ndarray,
    cutoff: float,
    rf_dielectric: float,
) -> float:
    N = positions.shape[0]
    k_rf, c_rf = reaction_field_coefficients(cutoff, rf_dielectric)
    total = 0.0
    for i in range(N):
        w_v_map = {}
        w_e_map = {}
        for m in range(excl_indices.shape[1]):
            j_ex = int(excl_indices[i, m])
            if j_ex >= 0:
                w_v_map[j_ex] = float(excl_vdw[i, m])
                w_e_map[j_ex] = float(excl_elec[i, m])
        for j in range(i + 1, N):
            dr = positions[i] - positions[j]
            dr = dr - box * np.floor(dr / box + 0.5)
            r2 = float(np.dot(dr, dr))
            if r2 >= cutoff * cutoff:
                continue
            r = math.sqrt(r2 + 1e-12)
            wv = w_v_map.get(j, 1.0)
            we = w_e_map.get(j, 1.0)
            sig = 0.5 * (float(sigmas[i]) + float(sigmas[j]))
            eps = math.sqrt(max(float(epsilons[i]) * float(epsilons[j]), 0.0))
            u6 = (r / sig) ** 6
            x6 = 1.0 / u6
            e_lj = wv * 4.0 * eps * (x6 * x6 - x6)
            qq = COULOMB_CONSTANT * float(charges[i]) * float(charges[j])
            e_coul = qq * we * (1.0 / r + k_rf * r2 - c_rf)
            total += e_lj + e_coul
    return total


def test_reaction_field_coefficients_conductor_limit_and_validation():
    """Verify k_rf and c_rf at rf_dielectric=inf (metallic conductor) produce zero NaNs."""
    rc = 9.0
    k_inf, c_inf = reaction_field_coefficients(rc, math.inf)
    assert math.isfinite(k_inf)
    assert math.isfinite(c_inf)
    assert math.isclose(k_inf, 1.0 / (2.0 * rc**3), rel_tol=0.0, abs_tol=1e-16)
    assert math.isclose(c_inf, 3.0 / (2.0 * rc), rel_tol=0.0, abs_tol=1e-16)

    # Continuity with large finite dielectric (1e15)
    k_big, c_big = reaction_field_coefficients(rc, 1e15)
    assert math.isclose(k_inf, k_big, rel_tol=1e-14, abs_tol=1e-16)
    assert math.isclose(c_inf, c_big, rel_tol=1e-14, abs_tol=1e-16)

    # At r = r_c with rf_dielectric=inf, BOTH V(r_c) and V'(r_c) vanish identically!
    assert abs((1.0 / rc) + k_inf * (rc * rc) - c_inf) < 1e-15
    assert abs((-1.0 / (rc * rc)) + 2.0 * k_inf * rc) < 1e-15

    # Invalid inputs raise ValueError cleanly
    with pytest.raises(ValueError, match="rf_dielectric"):
        reaction_field_coefficients(rc, 0.0)
    with pytest.raises(ValueError, match="rf_dielectric"):
        reaction_field_coefficients(rc, -5.0)
    with pytest.raises(ValueError, match="rf_dielectric"):
        reaction_field_coefficients(rc, float("nan"))
    with pytest.raises(ValueError, match="cutoff"):
        reaction_field_coefficients(0.0, 78.3)
    with pytest.raises(ValueError, match="cutoff"):
        reaction_field_coefficients(math.inf, 78.3)


@pytest.mark.parametrize("rf_dielectric", [math.inf, 78.3, 1.0])
def test_pallas_kernels_reaction_field_parity_and_grads(rf_dielectric: float):
    """Verify tile-pairs, NL, and fused megakernel energy/forces/param-grads match dense float64 RF."""
    rng = np.random.default_rng(1337)
    N = 36
    box_np = np.array([22.0, 22.0, 22.0], dtype=np.float64)
    cutoff = 8.5
    # Place atoms on a jittered grid with minimum separation > 1.8 A so LJ never overflows float32
    grid = np.stack(np.meshgrid(np.linspace(1.5, 19.5, 4), np.linspace(1.5, 19.5, 3), np.linspace(1.5, 19.5, 3)), axis=-1).reshape(-1, 3)[:N]
    pos_np = grid + rng.uniform(-0.25, 0.25, size=(N, 3))
    q_np = rng.uniform(-0.8, 0.8, size=(N,))
    q_np -= np.mean(q_np)
    sig_np = rng.uniform(2.8, 3.6, size=(N,))
    eps_np = rng.uniform(0.04, 0.18, size=(N,))

    # Add both a full 1-2 exclusion (0, 1) and a 1-4 scaled pair (2, 3)
    excl_idx_np = np.full((N, 2), -1, dtype=np.int32)
    excl_vdw_np = np.ones((N, 2), dtype=np.float64)
    excl_elec_np = np.ones((N, 2), dtype=np.float64)
    s14_v, s14_e = 0.5, 1.0 / 1.2
    excl_idx_np[0, 0], excl_idx_np[1, 0] = 1, 0
    excl_vdw_np[0, 0] = excl_vdw_np[1, 0] = 0.0
    excl_elec_np[0, 0] = excl_elec_np[1, 0] = 0.0
    excl_idx_np[2, 0], excl_idx_np[3, 0] = 3, 2
    excl_vdw_np[2, 0] = excl_vdw_np[3, 0] = s14_v
    excl_elec_np[2, 0] = excl_elec_np[3, 0] = s14_e

    # All-pairs dense neighbor list (N, N-1) so every interacting pair within cutoff is included
    nbr_np = np.stack([np.delete(np.arange(N, dtype=np.int32), i) for i in range(N)], axis=0)

    e_ref64 = _dense_reaction_field_energy_f64(
        pos_np, q_np, sig_np, eps_np, box_np, excl_idx_np, excl_vdw_np, excl_elec_np, cutoff, rf_dielectric
    )

    pos32 = jnp.asarray(pos_np, dtype=jnp.float32)
    q32 = jnp.asarray(q_np, dtype=jnp.float32)
    sig32 = jnp.asarray(sig_np, dtype=jnp.float32)
    eps32 = jnp.asarray(eps_np, dtype=jnp.float32)
    mask32 = jnp.ones((N,), dtype=jnp.bool_)
    box32 = jnp.asarray(box_np, dtype=jnp.float32)
    nbr32 = jnp.asarray(nbr_np, dtype=jnp.int32)
    ei32 = jnp.asarray(excl_idx_np, dtype=jnp.int32)
    ev32 = jnp.asarray(excl_vdw_np, dtype=jnp.float32)
    ee32 = jnp.asarray(excl_elec_np, dtype=jnp.float32)

    tp_nl = build_tile_pair_neighbor_list(
        nbr32,
        excl_indices=ei32,
        excl_scales_vdw=ev32,
        excl_scales_elec=ee32,
        atom_mask=mask32,
        positions=pos32,
        box=box32,
    )

    # 1. Reference pure-JAX NL energy & exact autodiff gradients
    def ref_fn(p_in, q_in, s_in, ep_in):
        return nonbonded_nl_reference(
            p_in,
            q_in,
            s_in,
            ep_in,
            mask32,
            nbr32,
            ei32,
            ev32,
            ee32,
            jnp.concatenate([box32, jnp.zeros((3,), dtype=jnp.float32)]),
            jnp.ones((1,), dtype=jnp.float32),
            has_box=True,
            pme_alpha=0.0,
            cutoff=cutoff,
            scale_14_vdw=s14_v,
            scale_14_elec=s14_e,
            rf_dielectric=rf_dielectric,
        )

    e_ref_jax, (gp_ref, gq_ref, gs_ref, ge_ref) = jax.value_and_grad(
        ref_fn, argnums=(0, 1, 2, 3)
    )(pos32, q32, sig32, eps32)
    assert math.isclose(float(e_ref_jax), e_ref64, rel_tol=2e-5, abs_tol=2e-4)

    # 2. pallas_nonbonded_tile_pairs, pallas_nonbonded_nl, and pallas_bonded_and_nonbonded_nl_fused
    e_tp, f_tp = pallas_nonbonded_tile_pairs(
        pos32,
        q32,
        sig32,
        eps32,
        tp_nl,
        atom_mask=mask32,
        box=box32,
        pme_alpha=0.0,
        cutoff=cutoff,
        scale_14_vdw=s14_v,
        scale_14_elec=s14_e,
        rf_dielectric=rf_dielectric,
        return_forces=True,
        interpret=True,
    )
    assert jnp.all(jnp.isfinite(f_tp))
    np.testing.assert_allclose(float(e_tp), e_ref64, rtol=2e-5, atol=2e-4)
    np.testing.assert_allclose(np.asarray(f_tp), np.asarray(-gp_ref), rtol=3e-5, atol=3e-4)

    e_fused, f_fused = pallas_bonded_and_nonbonded_nl_fused(
        pos32,
        q32,
        sig32,
        eps32,
        mask32,
        nbr32,
        ei32,
        ev32,
        ee32,
        box=box32,
        pme_alpha=0.0,
        cutoff=cutoff,
        scale_14_vdw=s14_v,
        scale_14_elec=s14_e,
        rf_dielectric=rf_dielectric,
        return_forces=True,
        interpret=True,
    )
    np.testing.assert_allclose(float(e_fused), e_ref64, rtol=2e-5, atol=2e-4)
    np.testing.assert_allclose(np.asarray(f_fused), np.asarray(-gp_ref), rtol=3e-5, atol=3e-4)

    if math.isinf(rf_dielectric):
        # Full 4-arg custom_vjp check wrt (pos, charges, sigmas, epsilons) on the conductor limit
        e_tp_vjp, (gp_tp, gq_tp, gs_tp, ge_tp) = jax.value_and_grad(
            lambda p_in, q_in, s_in, ep_in: pallas_nonbonded_tile_pairs(
                p_in,
                q_in,
                s_in,
                ep_in,
                tp_nl,
                atom_mask=mask32,
                box=box32,
                pme_alpha=0.0,
                cutoff=cutoff,
                scale_14_vdw=s14_v,
                scale_14_elec=s14_e,
                rf_dielectric=rf_dielectric,
                interpret=True,
            ),
            argnums=(0, 1, 2, 3),
        )(pos32, q32, sig32, eps32)
        assert jnp.all(jnp.isfinite(gq_tp))
        np.testing.assert_allclose(float(e_tp_vjp), e_ref64, rtol=2e-5, atol=2e-4)
        np.testing.assert_allclose(np.asarray(gp_tp), np.asarray(gp_ref), rtol=3e-5, atol=3e-4)
        np.testing.assert_allclose(np.asarray(gq_tp), np.asarray(gq_ref), rtol=3e-5, atol=3e-4)
        np.testing.assert_allclose(np.asarray(gs_tp), np.asarray(gs_ref), rtol=3e-5, atol=3e-4)
        np.testing.assert_allclose(np.asarray(ge_tp), np.asarray(ge_ref), rtol=3e-5, atol=3e-4)

        e_nl_vjp, (gp_nl, gq_nl, gs_nl, ge_nl) = jax.value_and_grad(
            lambda p_in, q_in, s_in, ep_in: pallas_nonbonded_nl(
                p_in,
                q_in,
                s_in,
                ep_in,
                mask32,
                nbr32,
                ei32,
                ev32,
                ee32,
                box=box32,
                pme_alpha=0.0,
                cutoff=cutoff,
                scale_14_vdw=s14_v,
                scale_14_elec=s14_e,
                rf_dielectric=rf_dielectric,
                interpret=True,
            ),
            argnums=(0, 1, 2, 3),
        )(pos32, q32, sig32, eps32)
        np.testing.assert_allclose(float(e_nl_vjp), e_ref64, rtol=2e-5, atol=2e-4)
        np.testing.assert_allclose(np.asarray(gp_nl), np.asarray(gp_ref), rtol=3e-5, atol=3e-4)
        np.testing.assert_allclose(np.asarray(gq_nl), np.asarray(gq_ref), rtol=3e-5, atol=3e-4)
        np.testing.assert_allclose(np.asarray(gs_nl), np.asarray(gs_ref), rtol=3e-5, atol=3e-4)
        np.testing.assert_allclose(np.asarray(ge_nl), np.asarray(ge_ref), rtol=3e-5, atol=3e-4)
    else:
        e_nl, f_nl = pallas_nonbonded_nl(
            pos32,
            q32,
            sig32,
            eps32,
            mask32,
            nbr32,
            ei32,
            ev32,
            ee32,
            box=box32,
            pme_alpha=0.0,
            cutoff=cutoff,
            scale_14_vdw=s14_v,
            scale_14_elec=s14_e,
            rf_dielectric=rf_dielectric,
            return_forces=True,
            interpret=True,
        )
        np.testing.assert_allclose(float(e_nl), e_ref64, rtol=2e-5, atol=2e-4)
        np.testing.assert_allclose(np.asarray(f_nl), np.asarray(-gp_ref), rtol=3e-5, atol=3e-4)


# =============================================================================
# Deliverable 4: Portable StableHLO Export Lowering Guardrails (0 Custom Calls)
# =============================================================================


def test_v36_fltflt_and_conductor_rf_stablehlo_export_zero_custom_calls():
    """Verify fltflt primitives and conductor-limit RF lower to pure portable StableHLO."""
    N = 16
    pos = jnp.linspace(1.0, 12.0, N * 3, dtype=jnp.float32).reshape(N, 3)
    q = jnp.linspace(-0.5, 0.5, N, dtype=jnp.float32)
    sig = jnp.full((N,), 3.1, dtype=jnp.float32)
    eps = jnp.full((N,), 0.1, dtype=jnp.float32)
    mask = jnp.ones((N,), dtype=jnp.bool_)
    box = jnp.asarray([20.0, 20.0, 20.0], dtype=jnp.float32)
    nbr = jnp.stack([jnp.delete(jnp.arange(N, dtype=jnp.int32), i) for i in range(N)], axis=0)
    tp_nl = build_tile_pair_neighbor_list(nbr, atom_mask=mask, positions=pos, box=box)

    def exported_fn(p_in, q_in):
        # Exercise every fltflt primitive inside export_context()
        a = ff_from_f32(p_in[:, 0])
        b = ff_from_f32(p_in[:, 1] + jnp.float32(10.0))
        s = ff_add(two_sum(p_in[:, 0], p_in[:, 1]), two_prod_fma(p_in[:, 0], p_in[:, 2]))
        d = ff_div(ff_mul(a, b), ff_add(ff_sqrt(b), ff_from_f32(jnp.float32(1.0))))
        nrm = ff_norm3d(a, b, ff_sub(b, a))
        wrapped = ff_min_image_f32(fast_two_sum(b.hi, a.hi * jnp.float32(1e-4)), jnp.float32(20.0))
        ff_term = jnp.sum(ff_to_f32(ff_add(ff_add(s, d), ff_add(nrm, wrapped))))

        e_tp = cast(
            "jax.Array",
            pallas_nonbonded_tile_pairs(
                p_in,
                q_in,
                sig,
                eps,
                tp_nl,
                atom_mask=mask,
                box=box,
                pme_alpha=0.0,
                cutoff=8.0,
                rf_dielectric=math.inf,
            ),
        )
        e_nl = cast(
            "jax.Array",
            pallas_nonbonded_nl(
                p_in,
                q_in,
                sig,
                eps,
                mask,
                nbr,
                box=box,
                pme_alpha=0.0,
                cutoff=8.0,
                rf_dielectric=math.inf,
            ),
        )
        return ff_term + e_tp + e_nl

    with prolix.export.export_context():
        val_and_grad_fn = jax.jit(jax.value_and_grad(exported_fn, argnums=(0, 1)))
        lowered_text = val_and_grad_fn.lower(pos, q).as_text()
        exported = jax.export.export(val_and_grad_fn)(pos, q)
        mlir_text = exported.mlir_module()
        out_val, (g_pos, g_q) = val_and_grad_fn(pos, q)

    assert jnp.isfinite(out_val)
    assert jnp.all(jnp.isfinite(g_pos))
    assert jnp.all(jnp.isfinite(g_q))
    assert "stablehlo.custom_call" not in mlir_text
    assert "stablehlo.custom_call" not in lowered_text
    lower_clean = lowered_text.lower()
    for bad in FORBIDDEN_STABLEHLO_TARGETS:
        assert bad not in lower_clean, f"Forbidden target {bad!r} found in lowered StableHLO"
    mlir_non_loc = "\n".join(
        line for line in mlir_text.splitlines() if not line.lstrip().startswith("#loc")
    ).lower()
    for bad in FORBIDDEN_STABLEHLO_TARGETS:
        assert bad not in mlir_non_loc, f"Forbidden target {bad!r} found in exported StableHLO"
