"""Claim 1 hetero SETTLE batching: 64-water mixed-system end-to-end (#262).

Extends #161 (pure 64-water homo batch) to a mixed solvent+solute system
(64 TIP3P waters + one unconstrained solute atom) under ``settle_langevin``
+ ``jax.vmap``. Validates SETTLE geometry on waters and long-horizon stability
at dt=0.5 fs for 10 ps with batch_size=2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prolix.physics import settle
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

TARGET_ROH = 0.9572
TARGET_RHH = 1.5139
CONSTRAINT_TOL = 0.01


def _build_64water_mixed_system():
    """64 TIP3P waters on a 4×4×4 grid plus one free solute atom."""
    n_waters = 64
    tip3p_roh = TARGET_ROH
    tip3p_theta = 104.52 * jnp.pi / 180.0
    h_offset_x = tip3p_roh
    h_offset_y_h2 = tip3p_roh * jnp.sin(tip3p_theta)
    h_offset_x_h2 = tip3p_roh * jnp.cos(tip3p_theta)

    spacing = 3.0
    waters = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                ox = 2.0 + spacing * ix
                oy = 2.0 + spacing * iy
                oz = 2.0 + spacing * iz
                o_pos = jnp.array([ox, oy, oz], dtype=jnp.float64)
                h1_pos = jnp.array([ox + h_offset_x, oy, oz], dtype=jnp.float64)
                h2_pos = jnp.array(
                    [ox + h_offset_x_h2, oy + h_offset_y_h2, oz], dtype=jnp.float64
                )
                waters.append([o_pos, h1_pos, h2_pos])

    water_positions = jnp.stack([atom for water in waters for atom in water])
    n_water_atoms = water_positions.shape[0]
    solute_pos = jnp.array([spacing * 4 + 4.0, spacing * 4 + 4.0, spacing * 4 + 4.0])
    positions = jnp.concatenate([water_positions, solute_pos[None, :]], axis=0)

    water_indices = jnp.arange(n_water_atoms, dtype=jnp.int32).reshape(n_waters, 3)
    mass = jnp.concatenate(
        [
            jnp.array([15.999, 1.008, 1.008] * n_waters, dtype=jnp.float64),
            jnp.array([12.0], dtype=jnp.float64),
        ]
    )
    o_targets = jnp.stack([water[0] for water in waters])
    solute_ref = solute_pos

    def energy_fn(positions, box=None):
        o_idx = jnp.arange(0, n_water_atoms, 3)
        o_positions = positions[o_idx]
        e_water = 0.05 * jnp.sum((o_positions - o_targets) ** 2)
        solute = positions[-1]
        e_solute = 0.02 * jnp.sum((solute - solute_ref) ** 2)
        e_couple = 0.01 * jnp.sum((solute - o_positions[0]) ** 2)
        return e_water + e_solute + e_couple

    def shift_fn(positions, box=None):
        return positions

    box_edge = 2.0 + spacing * 3 + 8.0
    box = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    return {
        "positions": positions,
        "water_indices": water_indices,
        "mass": mass,
        "energy_fn": energy_fn,
        "shift_fn": shift_fn,
        "box": box,
        "n_waters": n_waters,
        "n_atoms": positions.shape[0],
    }


def _compute_water_geometry(positions, water_indices):
    if positions.ndim == 3:
        return jax.vmap(lambda p: _compute_water_geometry(p, water_indices))(positions)

    o_idx = water_indices[:, 0]
    h1_idx = water_indices[:, 1]
    h2_idx = water_indices[:, 2]
    o_pos = positions[o_idx]
    h1_pos = positions[h1_idx]
    h2_pos = positions[h2_idx]
    oh1 = jnp.linalg.norm(h1_pos - o_pos, axis=-1)
    oh2 = jnp.linalg.norm(h2_pos - o_pos, axis=-1)
    hh = jnp.linalg.norm(h2_pos - h1_pos, axis=-1)
    return oh1, oh2, hh


@pytest.mark.slow
def test_settle_64water_mixed_batched_10ps_trajectory():
    """64-water + solute batched SETTLE: 10 ps at dt=0.5 fs without divergence."""
    jax.config.update("jax_enable_x64", True)
    sys = _build_64water_mixed_system()
    batch_size = 2
    dt_fs = 0.5
    dt_akma = dt_fs / AKMA_TIME_UNIT_FS
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = 10.0 * AKMA_TIME_UNIT_FS * 1e-3
    n_steps = int(10_000.0 / dt_fs)
    dof = float(6 * sys["n_waters"] - 3)
    box = sys["box"]
    water_mass = sys["mass"][: 3 * sys["n_waters"]]

    init_fn, apply_fn = settle.settle_langevin(
        sys["energy_fn"],
        sys["shift_fn"],
        dt=dt_akma,
        kT=kT,
        gamma=gamma,
        mass=sys["mass"],
        water_indices=sys["water_indices"],
        box=box,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
    )

    keys = jax.random.split(jax.random.key(262), batch_size)
    singles = [init_fn(k, sys["positions"], mass=sys["mass"]) for k in keys]
    state = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *singles)

    def apply_one(s):
        return apply_fn(s, box=box)

    batched_apply = jax.vmap(apply_one)

    def body(state, _):
        new_state = batched_apply(state)
        water_pos = new_state.position[..., : 3 * sys["n_waters"], :]
        water_mom = new_state.momentum[..., : 3 * sys["n_waters"], :]
        ke = jax.vmap(
            lambda pos, mom: rigid_tip3p_box_ke_kcal(
                pos, mom, water_mass, sys["n_waters"]
            ),
            in_axes=(0, 0),
        )(water_pos, water_mom)
        temp_k = 2.0 * ke / (dof * BOLTZMANN_KCAL)
        return new_state, temp_k

    final_state, temps_k = jax.lax.scan(body, state, None, length=n_steps)

    assert jnp.all(jnp.isfinite(final_state.position))
    assert jnp.all(jnp.isfinite(final_state.momentum))

    water_pos = final_state.position[..., : 3 * sys["n_waters"], :]
    oh1, oh2, hh = _compute_water_geometry(water_pos, sys["water_indices"])
    assert float(jnp.max(jnp.abs(oh1 - TARGET_ROH))) < CONSTRAINT_TOL
    assert float(jnp.max(jnp.abs(oh2 - TARGET_ROH))) < CONSTRAINT_TOL
    assert float(jnp.max(jnp.abs(hh - TARGET_RHH))) < CONSTRAINT_TOL

    burn = (3 * n_steps) // 4
    prod_temps = temps_k[burn:]
    mean_t = float(jnp.mean(prod_temps))
    max_t = float(jnp.max(prod_temps))
    assert max_t < 500.0, f"temperature runaway: max T={max_t:.1f} K"
    assert 285.0 <= mean_t <= 315.0, f"mean T={mean_t:.1f} K outside 300±15 K band"
