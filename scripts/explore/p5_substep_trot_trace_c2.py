"""Substep T_rot/T_trans diagnostic trace — C2 variant (gamma=1, production O step).

Identical to p5_substep_trot_trace.py except:
  - gamma_ps = 1.0  (matches production settle_langevin call)
  - O step uses _langevin_step_o_constrained directly (project + rigid OU),
    NOT the C1 approach of isotropic OU + project.

This gives substep contributions under exact production parameters so we can
identify which substep drains T_rot in the gamma=1 regime.
"""

import csv
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

jax.config.update("jax_enable_x64", True)

from prolix.physics import settle, system, pbc
from prolix.physics.settle import _langevin_step_o_constrained
from prolix.physics.solvation import load_water_box
from prolix.physics.water_models import WaterModelType, get_water_params
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL


def _proxide_params_pure_water(n_waters: int) -> dict:
    tip = get_water_params(WaterModelType.TIP3P)
    qo, qh = float(tip.charge_O), float(tip.charge_H)
    sig_o = float(tip.sigma_O)
    eps_o = float(tip.epsilon_O)
    n = n_waters * 3
    charges: list[float] = []
    sigmas: list[float] = []
    epsilons: list[float] = []
    for _ in range(n_waters):
        charges.extend([qo, qh, qh])
        sigmas.extend([sig_o, 1.0, 1.0])
        epsilons.extend([eps_o, 0.0, 0.0])
    mask = jnp.ones((n, n), dtype=jnp.float64) - jnp.eye(n, dtype=jnp.float64)
    for w in range(n_waters):
        b = w * 3
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            a, c = b + i, b + j
            mask = mask.at[a, c].set(0.0).at[c, a].set(0.0)
    return {
        "charges": jnp.array(charges, dtype=jnp.float64),
        "sigmas": jnp.array(sigmas, dtype=jnp.float64),
        "epsilons": jnp.array(epsilons, dtype=jnp.float64),
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "exclusion_mask": mask,
    }


def _make_tip3p_excl_indices(n_waters: int) -> jnp.ndarray:
    n_atoms = n_waters * 3
    excl = np.zeros((n_atoms, 2), dtype=np.int32)
    for w in range(n_waters):
        o, h1, h2 = w * 3, w * 3 + 1, w * 3 + 2
        excl[o] = [h1, h2]
        excl[h1] = [o, h2]
        excl[h2] = [o, h1]
    return jnp.array(excl, dtype=jnp.int32)


def _equil_water_positions(n_waters: int, seed: int = 0) -> tuple[np.ndarray, float]:
    box = load_water_box()
    positions = np.array(box.positions)
    box_edge = float(np.array(box.box_size)[0])
    n_total = len(positions) // 3
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n_total, size=n_waters, replace=False))
    atom_idx = np.concatenate([np.array([3*i, 3*i+1, 3*i+2]) for i in chosen])
    return positions[atom_idx], box_edge


def compute_T_rot_trans_all_waters(
    momentum: jnp.ndarray,
    positions: jnp.ndarray,
    water_indices: jnp.ndarray,
    mass_o: float = 15.999,
    mass_h: float = 1.008,
) -> tuple[float, float]:
    kB = BOLTZMANN_KCAL
    n_waters = water_indices.shape[0]
    T_rots = []
    T_transs = []

    for w in range(n_waters):
        o_idx, h1_idx, h2_idx = water_indices[w]
        m_total = mass_o + 2 * mass_h

        p_o = momentum[o_idx]
        p_h1 = momentum[h1_idx]
        p_h2 = momentum[h2_idx]
        r_o = positions[o_idx]
        r_h1 = positions[h1_idx]
        r_h2 = positions[h2_idx]

        p_cm = p_o + p_h1 + p_h2
        T_trans = float(jnp.sum(p_cm**2) / (3.0 * m_total * kB))

        r_com = (mass_o * r_o + mass_h * r_h1 + mass_h * r_h2) / m_total
        r_rel_o = r_o - r_com
        r_rel_h1 = r_h1 - r_com
        r_rel_h2 = r_h2 - r_com

        v_o = p_o / mass_o
        v_h1 = p_h1 / mass_h
        v_h2 = p_h2 / mass_h

        L = (mass_o * jnp.cross(r_rel_o, v_o) +
             mass_h * jnp.cross(r_rel_h1, v_h1) +
             mass_h * jnp.cross(r_rel_h2, v_h2))

        eye3 = jnp.eye(3, dtype=jnp.float64)
        I_tensor = (mass_o * (jnp.sum(r_rel_o**2) * eye3 - jnp.outer(r_rel_o, r_rel_o)) +
                    mass_h * (jnp.sum(r_rel_h1**2) * eye3 - jnp.outer(r_rel_h1, r_rel_h1)) +
                    mass_h * (jnp.sum(r_rel_h2**2) * eye3 - jnp.outer(r_rel_h2, r_rel_h2)))
        reg = 1e-12 * (jnp.trace(I_tensor) / 3.0 + 1.0)
        I_inv = jnp.linalg.inv(I_tensor + reg * eye3)
        T_rot = float(jnp.dot(L, jnp.dot(I_inv, L)) / (3.0 * kB))

        T_rots.append(T_rot)
        T_transs.append(T_trans)

    return float(np.mean(T_transs)), float(np.mean(T_rots))


def main():
    # --- Setup: gamma=1 to match production ---
    n_waters = 64
    n_atoms = n_waters * 3

    n_steps = 500
    dt_fs = 0.5
    temperature_k = 300.0
    gamma_ps = 1.0  # PRODUCTION VALUE (C1 used 10.0)

    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3

    c1 = float(jnp.exp(-gamma_reduced * dt_akma))
    c2 = float(jnp.sqrt(1.0 - c1**2))
    print(f"gamma_ps={gamma_ps}, c1={c1:.6f}, c2={c2:.6f}, c2^2={c2**2:.6f}")

    positions_init, box_edge = _equil_water_positions(n_waters)
    box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)
    positions = jnp.array(positions_init, dtype=jnp.float64)

    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)

    sys_dict = {k: v for k, v in _proxide_params_pure_water(n_waters).items() if k != "exclusion_mask"}
    sys_dict["excl_indices"] = _make_tip3p_excl_indices(n_waters)
    grid = max(16, round(box_edge / 1.0))
    energy_fn = system.make_energy_fn(
        displacement_fn, sys_dict, box=box_vec, use_pbc=True,
        implicit_solvent=False, pme_grid_points=grid, pme_alpha=0.34,
        cutoff_distance=9.0, strict_parameterization=False
    )

    mass_array = np.array([[15.999], [1.008], [1.008]] * n_waters, dtype=np.float64).reshape(n_atoms)
    mass = jnp.array(mass_array, dtype=jnp.float64)
    mass_col = mass[:, None]

    water_indices = settle.get_water_indices(0, n_waters)

    # Initialize with production integrator so we start from a thermalized state
    init_s, apply_s = settle.settle_langevin(
        energy_fn, shift_fn, dt=dt_akma, kT=kT,
        gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec,
        remove_linear_com_momentum=False, project_ou_momentum_rigid=True,
        projection_site="post_o", settle_velocity_iters=10
    )

    key = jax.random.PRNGKey(42)
    state = init_s(key, positions, mass=mass)

    # Warm up for 2000 steps to reach steady state at gamma=1
    print("Warming up 2000 steps with production integrator...")
    for _ in range(2000):
        state = apply_s(state)
    print("Warmup done.")

    # Check T_rot/T_trans after warmup
    T_trans_wu, T_rot_wu = compute_T_rot_trans_all_waters(state.momentum, state.positions, water_indices)
    print(f"After warmup: T_rot={T_rot_wu:.2f}K, T_trans={T_trans_wu:.2f}K")

    # --- Open CSV output ---
    output_path = Path("scripts/explore/p5_substep_trot_trace_c2.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = output_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "substep", "T_rot", "T_trans", "delta_T_rot", "delta_T_trans"])

    force_fn = jax.jit(lambda r: -jax.grad(energy_fn)(r))

    momentum = state.momentum
    force = state.force
    positions = state.positions
    positions_old = state.positions
    key = state.key

    for step in range(n_steps):
        T_trans_prev, T_rot_prev = compute_T_rot_trans_all_waters(momentum, positions, water_indices)

        # --- B1: half force kick ---
        momentum = momentum + 0.5 * dt_akma * force
        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        writer.writerow([step, "B1", T_rot, T_trans, T_rot - T_rot_prev, T_trans - T_trans_prev])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- A: position update (first half) ---
        velocity = momentum / mass_col
        positions = shift_fn(positions, 0.5 * dt_akma * velocity)
        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        writer.writerow([step, "A", T_rot, T_trans, T_rot - T_rot_prev, T_trans - T_trans_prev])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- SETTLE_pos1 + R-step ---
        positions_constrained = settle.settle_positions(positions, positions_old, water_indices, box=box_vec)
        dx = positions_constrained - positions
        dx = dx - box_vec * jnp.round(dx / box_vec)
        dp = mass_col * dx / (0.5 * dt_akma)
        momentum = momentum + dp
        positions = positions_constrained
        positions_mid = positions

        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        writer.writerow([step, "SETTLE1", T_rot, T_trans, T_rot - T_rot_prev, T_trans - T_trans_prev])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- O step: PRODUCTION algorithm (project + rigid OU, gamma=1) ---
        momentum, key = _langevin_step_o_constrained(
            momentum, positions, mass_col, gamma_reduced, dt_akma, kT, key, water_indices
        )
        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        writer.writerow([step, "O", T_rot, T_trans, T_rot - T_rot_prev, T_trans - T_trans_prev])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- A: position update (second half) ---
        velocity = momentum / mass_col
        positions = shift_fn(positions, 0.5 * dt_akma * velocity)
        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        writer.writerow([step, "A2", T_rot, T_trans, T_rot - T_rot_prev, T_trans - T_trans_prev])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- SETTLE_pos2 + R-step ---
        positions_constrained = settle.settle_positions(positions, positions_mid, water_indices, box=box_vec)
        dx = positions_constrained - positions
        dx = dx - box_vec * jnp.round(dx / box_vec)
        dp = mass_col * dx / (0.5 * dt_akma)
        momentum = momentum + dp
        positions = positions_constrained

        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        writer.writerow([step, "SETTLE2", T_rot, T_trans, T_rot - T_rot_prev, T_trans - T_trans_prev])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- Force update ---
        force = force_fn(positions)

        # --- B2: final half force kick ---
        momentum = momentum + 0.5 * dt_akma * force
        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        writer.writerow([step, "B2", T_rot, T_trans, T_rot - T_rot_prev, T_trans - T_trans_prev])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- SETTLE_vel (RATTLE) ---
        velocity_constrained = settle.settle_velocities(
            momentum / mass_col, positions_mid, positions, water_indices, dt_akma, n_iters=10
        )
        momentum = velocity_constrained * mass_col
        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        writer.writerow([step, "SETTLE_vel", T_rot, T_trans, T_rot - T_rot_prev, T_trans - T_trans_prev])

        positions_old = positions

    csv_file.close()
    print(f"Trace written to {output_path}")

    import pandas as pd
    df = pd.read_csv(output_path)
    summary = df.groupby("substep")[["delta_T_rot", "delta_T_trans"]].mean()
    print("\nMean ΔT per substep (gamma=1, production O step):")
    print(summary)
    net_rot = summary["delta_T_rot"].sum()
    net_trans = summary["delta_T_trans"].sum()
    print(f"\nNet per step: T_rot={net_rot:+.4f}K, T_trans={net_trans:+.4f}K")

    # Expected SS T_rot from simple OU model: T_rot_ss = T_target + drain_mech / c2^2
    c2_sq = c2**2
    drain_mech = net_rot - df[df.substep == "O"]["delta_T_rot"].mean()
    print(f"\nc2^2 = {c2_sq:.6f}")
    print(f"Mechanical drain (excl O) = {drain_mech:+.4f} K/step")
    print(f"Implied T_rot_ss = {300.0 + drain_mech / c2_sq:.1f} K")


if __name__ == "__main__":
    main()
