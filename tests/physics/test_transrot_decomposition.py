"""Synthetic ground-truth tests for the trans/rot decomposition and the BAOAB A-step.

Task 260806_p5_measurement_pipeline_audit.

These tests use momenta with *exactly known* translational and rotational content,
constructed analytically. They require no MD, no forces and no trajectory, so they
run in the fast CI gate.

Two independent defects are pinned here:

1. DOF MISCOUNT. Every Phase 5 copy of the decomposition computes ``ke_trans`` from
   per-water COM velocities *without* subtracting the system COM velocity, then
   divides by ``dof_trans = 3N - 3`` -- a count that is only correct if the system
   COM has been removed. The integrator's ``remove_linear_com_momentum`` defaults to
   False (settle.py:1022) and the Phase 5 scripts pass False explicitly, so the
   system COM is a live, thermalized 3-DOF mode (settle.py:2120 draws each water's
   COM velocity independently from N(0, kT/M)). The reported T_trans is therefore
   inflated by 3*T_target/(3N-3): +300 K at n=2, +20 K at n=16.

2. HALF-RATE POSITION ADVANCE. ``_langevin_step_a`` (settle.py:2069-2074) already
   multiplies by 0.5 internally, so callers must pass the FULL dt. ``settle_langevin``
   passes ``half_dt`` at settle.py:1233 and :1288, making each A-step advance
   0.25*dt*v and the full cycle 0.5*dt*v instead of BAOAB's dt*v. Every other
   ``_langevin_step_a`` call site in the file passes ``_dt``.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import pbc, settle
from prolix.simulate import BOLTZMANN_KCAL

jax.config.update("jax_enable_x64", True)

R_OH = settle.TIP3P_ROH
R_HH = settle.TIP3P_RHH
M_O, M_H = 15.999, 1.008
M_WATER = M_O + M_H + M_H
T_TARGET = 300.0
KT = T_TARGET * BOLTZMANN_KCAL


def _ideal_water_geometry(n_waters: int, spacing: float = 6.0) -> np.ndarray:
    """n rigid TIP3P waters at ideal geometry, laid out on a cubic lattice."""
    half_angle = math.asin(R_HH / (2.0 * R_OH))
    single = np.array([
        [0.0, 0.0, 0.0],
        [+R_OH * math.sin(half_angle), R_OH * math.cos(half_angle), 0.0],
        [-R_OH * math.sin(half_angle), R_OH * math.cos(half_angle), 0.0],
    ])
    per_side = int(math.ceil(n_waters ** (1.0 / 3.0)))
    out = []
    for w in range(n_waters):
        i, j, k = w % per_side, (w // per_side) % per_side, w // (per_side ** 2)
        out.append(single + np.array([i, j, k]) * spacing)
    return np.concatenate(out, axis=0)


def _masses(n_waters: int) -> np.ndarray:
    return np.array([M_O, M_H, M_H] * n_waters)


def _decompose_current(momentum, mass_flat, n_waters):
    """Verbatim reproduction of the Phase 5 decomposition under audit.

    Matches p5_nvt_gate_dt1fs.py:90-109, p5_nve_transrot_bisect.py:125-137 and the
    six other hand-copied instances.
    """
    p3 = np.asarray(momentum).reshape(n_waters, 3, 3)
    v_com = p3.sum(axis=1) / M_WATER
    ke_trans = 0.5 * M_WATER * np.sum(v_com**2)
    ke_tot = float(np.sum(np.sum(np.asarray(momentum) ** 2, axis=-1) / (2.0 * mass_flat)))
    ke_rot = ke_tot - ke_trans
    return {
        "t_trans": 2.0 * ke_trans / ((3 * n_waters - 3) * BOLTZMANN_KCAL),
        "t_rot": 2.0 * ke_rot / ((3 * n_waters) * BOLTZMANN_KCAL),
        "t_total": 2.0 * ke_tot / ((6 * n_waters - 3) * BOLTZMANN_KCAL),
        "ke_trans": ke_trans,
        "ke_rot": ke_rot,
    }


def _decompose_corrected(momentum, mass_flat, n_waters):
    """COM-subtracted decomposition: valid regardless of remove_linear_com_momentum."""
    p = np.asarray(momentum)
    m_tot = float(np.sum(mass_flat))
    p_sys = p.sum(axis=0)
    v_sys = p_sys / m_tot
    ke_com = 0.5 * float(p_sys @ p_sys) / m_tot

    p3 = p.reshape(n_waters, 3, 3)
    v_com = p3.sum(axis=1) / M_WATER
    ke_trans_rel = 0.5 * M_WATER * np.sum((v_com - v_sys) ** 2)
    ke_tot = float(np.sum(np.sum(p**2, axis=-1) / (2.0 * mass_flat)))
    ke_rot = ke_tot - ke_trans_rel - ke_com
    return {
        "t_trans": 2.0 * ke_trans_rel / ((3 * n_waters - 3) * BOLTZMANN_KCAL),
        "t_rot": 2.0 * ke_rot / ((3 * n_waters) * BOLTZMANN_KCAL),
        "t_com": 2.0 * ke_com / (3 * BOLTZMANN_KCAL),
        "t_total": 2.0 * (ke_tot - ke_com) / ((6 * n_waters - 3) * BOLTZMANN_KCAL),
        "ke_com": ke_com,
        "ke_rot": ke_rot,
    }


# --------------------------------------------------------------------------
# Channel isolation on exactly-known synthetic momenta
# --------------------------------------------------------------------------


def test_pure_com_boost_has_zero_relative_translation_and_zero_rotation():
    """Every atom given the same velocity: all KE is system-COM, none is rot."""
    n = 8
    mass = _masses(n)
    u = np.array([0.3, -0.2, 0.15])
    momentum = mass[:, None] * u

    d = _decompose_corrected(momentum, mass, n)
    assert abs(d["ke_rot"]) < 1e-10 * d["ke_com"]
    assert d["t_trans"] == pytest.approx(0.0, abs=1e-6)
    assert d["ke_com"] == pytest.approx(0.5 * mass.sum() * float(u @ u), rel=1e-12)


def test_pure_rotation_has_zero_translation_of_either_kind():
    """p_i = m_i * (omega x r_rel): zero net linear momentum per water."""
    n = 4
    pos = _ideal_water_geometry(n)
    mass = _masses(n)
    rng = np.random.default_rng(0)

    momentum = np.zeros_like(pos)
    for a in range(n):
        sl = slice(3 * a, 3 * a + 3)
        m_w, r_w = mass[sl], pos[sl]
        com = (m_w[:, None] * r_w).sum(axis=0) / m_w.sum()
        omega = rng.normal(size=3) * 0.05
        momentum[sl] = m_w[:, None] * np.cross(omega, r_w - com)

    d = _decompose_corrected(momentum, mass, n)
    assert d["t_rot"] > 0.0
    assert abs(d["ke_com"]) < 1e-10 * d["ke_rot"]
    assert d["t_trans"] == pytest.approx(0.0, abs=1e-6)


def test_ke_rot_is_galilean_invariant():
    """Boosting the whole system must leave ke_rot exactly unchanged.

    This is why T_rot is unaffected by the DOF miscount -- "T_rot is faithful at
    every size" is a theorem about this estimator, not evidence about the physics.
    """
    n = 8
    pos = _ideal_water_geometry(n)
    mass = _masses(n)
    rng = np.random.default_rng(1)
    momentum = rng.normal(size=pos.shape) * mass[:, None] * 0.01

    base = _decompose_corrected(momentum, mass, n)["ke_rot"]
    for _ in range(5):
        u = rng.normal(size=3)
        boosted = momentum + mass[:, None] * u
        assert _decompose_corrected(boosted, mass, n)["ke_rot"] == pytest.approx(base, rel=1e-10)


# --------------------------------------------------------------------------
# The killer: equipartition round-trip through the project's own initializer
# --------------------------------------------------------------------------


def _mb_draw_temperatures(n_waters):
    """Mean temperatures from many exact rigid-body Maxwell-Boltzmann draws at 300 K."""
    pos = jnp.asarray(_ideal_water_geometry(n_waters), dtype=jnp.float64)
    mass = _masses(n_waters)
    m_jax = jnp.asarray(mass, dtype=jnp.float64)

    n_draws = 4000
    keys = jax.random.split(jax.random.key(0), n_draws)

    def one_draw(key):
        def one_water(carry, inputs):
            k = carry
            r_w, m_w = inputs
            p_w, k = settle._init_momentum_one_water_rigid(k, r_w, m_w, KT)
            return k, p_w

        _, p = jax.lax.scan(
            one_water, key, (pos.reshape(n_waters, 3, 3), m_jax.reshape(n_waters, 3))
        )
        return p.reshape(-1, 3)

    draws = np.asarray(jax.jit(jax.vmap(one_draw))(keys))

    cur = [_decompose_current(d, mass, n_waters) for d in draws]
    cor = [_decompose_corrected(d, mass, n_waters) for d in draws]
    return {
        "t_com": float(np.mean([c["t_com"] for c in cor])),
        "t_trans_cur": float(np.mean([c["t_trans"] for c in cur])),
        "t_trans_cor": float(np.mean([c["t_trans"] for c in cor])),
        "t_rot_cur": float(np.mean([c["t_rot"] for c in cur])),
    }


@pytest.mark.parametrize("n_waters", [2, 16, 64])
def test_corrected_estimator_recovers_target_temperature(n_waters):
    """Draw from the repo's own rigid-body MB initializer at 300 K; demand 300 K back.

    No integrator, no forces, no trajectory -- the ground truth is exact by
    construction. Also pins the predicted artifact size, 3*T_target/(3N-3).
    """
    t = _mb_draw_temperatures(n_waters)
    predicted_bias = 3.0 * T_TARGET / (3 * n_waters - 3)

    # The system COM really is a live, thermalized 3-DOF mode at the target T.
    assert t["t_com"] == pytest.approx(T_TARGET, abs=25.0), f"T_com={t['t_com']:.1f}"

    # T_rot is unbiased under BOTH estimators (Galilean invariance).
    assert t["t_rot_cur"] == pytest.approx(T_TARGET, abs=5.0), f"T_rot={t['t_rot_cur']:.1f}"

    # The corrected estimator recovers the target.
    assert t["t_trans_cor"] == pytest.approx(T_TARGET, abs=5.0)

    # ...and the gap to the current estimator is exactly the predicted artifact.
    assert (t["t_trans_cur"] - t["t_trans_cor"]) == pytest.approx(predicted_bias, rel=0.10), (
        f"n={n_waters}: observed bias {t['t_trans_cur'] - t['t_trans_cor']:.2f} K, "
        f"predicted {predicted_bias:.2f} K"
    )


@pytest.mark.xfail(
    strict=True,
    reason="DOF miscount (task 260806_p5_measurement_pipeline_audit): the Phase 5 "
    "estimator divides un-COM-subtracted ke_trans by 3N-3, inflating T_trans by "
    "3*T_target/(3N-3). Reproduces the documented size sweep (603.8 K vs recorded "
    "600.6 K at n=2; 319.6 K vs 319.67 K at n=16) with no dynamics at all. "
    "Fix: subtract the system COM velocity (see scripts/explore/p5_dof_artifact_correction.py).",
)
@pytest.mark.parametrize("n_waters", [2, 16])
def test_current_estimator_recovers_target_temperature(n_waters):
    """The Phase 5 estimator should report 300 K on exact 300 K draws. It does not."""
    t = _mb_draw_temperatures(n_waters)
    assert t["t_trans_cur"] == pytest.approx(T_TARGET, abs=5.0), (
        f"n={n_waters}: current estimator reports T_trans={t['t_trans_cur']:.1f} K"
    )


# --------------------------------------------------------------------------
# BAOAB propagator: does one step advance positions by dt*v?
# --------------------------------------------------------------------------


def _free_translation_step(integrator_factory, dt_akma, n_waters=2):
    """One step of uniform free translation: zero force, zero friction."""
    # Offset well inside the box: the ideal geometry puts hydrogens at negative x,
    # which the periodic shift_fn would wrap to ~L, swamping the dt*v displacement.
    pos_np = _ideal_water_geometry(n_waters, spacing=8.0) + 50.0
    mass_np = _masses(n_waters)
    box_vec = jnp.array([200.0] * 3, dtype=jnp.float64)
    _, shift_fn = pbc.create_periodic_space(box_vec)

    def zero_energy(R, **_kwargs):
        return jnp.sum(R) * 0.0

    mass = jnp.asarray(mass_np, dtype=jnp.float64)
    init_fn, apply_fn = integrator_factory(zero_energy, shift_fn, dt_akma, mass, box_vec, n_waters)

    state = init_fn(jax.random.key(0), jnp.asarray(pos_np, dtype=jnp.float64), mass=mass)

    u = jnp.array([0.01, -0.007, 0.013], dtype=jnp.float64)
    momentum = state.mass * u  # state.mass is (n_atoms, 1)
    state = eqx.tree_at(lambda s: s.momentum, state, momentum)
    state = eqx.tree_at(lambda s: s.force, state, jnp.zeros_like(state.force))

    new_state = jax.jit(apply_fn)(state)
    dx = np.asarray(new_state.positions - state.positions)
    return dx, np.asarray(u), float(dt_akma)


def _advance_ratio(dx: np.ndarray, u: np.ndarray, dt: float) -> float:
    """Least-squares scale c in dx ~ c * (dt*u), with a uniform-motion check."""
    expected = np.asarray(dt * u)
    assert np.allclose(dx, dx[0], atol=1e-12), f"motion is not uniform: {dx}"
    return float(np.dot(dx, expected).sum() / np.dot(expected, expected) / dx.shape[0])


def test_settle_langevin_advances_positions_by_full_dt():
    """Free particles, gamma=0, zero force: dx must equal dt*v, not dt*v/2.

    For zero force and zero friction the exact solution is x(t+dt) = x(t) + dt*v,
    so any correct splitting must reproduce it. This is convention-independent.
    """

    def factory(energy_fn, shift_fn, dt, mass, box_vec, n_waters):
        return settle.settle_langevin(
            energy_fn, shift_fn, dt=dt, kT=KT, gamma=0.0, mass=mass,
            water_indices=settle.get_water_indices(0, n_waters), box=box_vec,
            remove_linear_com_momentum=False, project_ou_momentum_rigid=True,
            projection_site="post_o", settle_velocity_iters=10,
        )

    ratio = _advance_ratio(*_free_translation_step(factory, 0.02))
    assert ratio == pytest.approx(1.0, rel=1e-6), (
        f"settle_langevin advanced positions by {ratio:.4f} * dt*v "
        f"(expected 1.0; 0.5 indicates the half_dt double-halving)"
    )


@pytest.mark.xfail(
    strict=True,
    reason="LFMiddle has THREE A-steps (settle.py:1430, :1439, :1456), giving 1.5*dt*v "
    "per cycle; measured ratio is exactly 1.5000 against the free-particle invariant. "
    "NOT fixed alongside settle_langevin: its O-step is correctly split into two "
    "half_dt halves (:1434 and :1466) that compose to exp(-gamma*dt), so which A-step "
    "is spurious is a design question about the intended LF-Middle splitting, not a "
    "typo. Campaign 89c9a900 ('LFMiddle FALSIFIED', 46 runs, 0 passes) tested this "
    "propagator, so that falsification cannot be trusted until the scheme is settled.",
)
def test_lfmiddle_langevin_advances_positions_by_full_dt():
    """settle_lfmiddle_langevin must also advance a free particle by exactly dt*v."""

    def factory(energy_fn, shift_fn, dt, mass, box_vec, n_waters):
        return settle.settle_lfmiddle_langevin(
            energy_fn, shift_fn, dt=dt, kT=KT, gamma=0.0, mass=mass,
            water_indices=settle.get_water_indices(0, n_waters), box=box_vec,
        )

    ratio = _advance_ratio(*_free_translation_step(factory, 0.02))
    assert ratio == pytest.approx(1.0, rel=1e-6), (
        f"settle_lfmiddle_langevin advanced positions by {ratio:.4f} * dt*v"
    )
