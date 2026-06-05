"""Substep T_rot/T_trans diagnostic trace for settle_langevin.

Instruments the BAOAB integration loop at each sub-step to pinpoint where
rotational temperature drains. Output: CSV file with substep-by-substep T_rot,
T_trans, and their deltas.
"""

import csv
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Enable 64-bit precision for water simulations
jax.config.update("jax_enable_x64", True)

from prolix.physics import settle, system, pbc
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL


# --- Water geometry setup (copied from test_p2b_nvt_216water.py) ---
def _proxide_params_pure_water(n_waters: int) -> dict:
    """Build TIP3P water system parameters."""
    # TIP3P parameters
    sigma_oh = 0.315365  # Å
    sigma_hh = 0.0  # No LJ on H-H
    sigma_oo = 3.15365  # Å

    eps_oh = 0.0  # No LJ on O-H
    eps_hh = 0.0
    eps_oo = 0.1521  # kcal/mol

    # Charges (elementary units)
    q_o = -0.834
    q_h = 0.417

    # Build arrays for n_waters * 3 atoms (O, H1, H2 per water)
    n_atoms = n_waters * 3
    sigma = np.zeros(n_atoms)
    epsilon = np.zeros(n_atoms)
    charges = np.zeros(n_atoms)

    for w in range(n_waters):
        o_idx = w * 3
        h1_idx = w * 3 + 1
        h2_idx = w * 3 + 2

        sigma[o_idx] = sigma_oo
        sigma[h1_idx] = sigma_hh
        sigma[h2_idx] = sigma_hh

        epsilon[o_idx] = eps_oo
        epsilon[h1_idx] = eps_oh
        epsilon[h2_idx] = eps_oh

        charges[o_idx] = q_o
        charges[h1_idx] = q_h
        charges[h2_idx] = q_h

    return {
        "sigma": jnp.array(sigma, dtype=jnp.float64),
        "epsilon": jnp.array(epsilon, dtype=jnp.float64),
        "charges": jnp.array(charges, dtype=jnp.float64),
        "excl_indices": _make_tip3p_excl_indices(n_waters),
    }


def _make_tip3p_excl_indices(n_waters: int) -> jnp.ndarray:
    """Per-atom sparse exclusion list for TIP3P."""
    n_atoms = n_waters * 3
    excl = np.zeros((n_atoms, 2), dtype=np.int32)
    for w in range(n_waters):
        o, h1, h2 = w * 3, w * 3 + 1, w * 3 + 2
        excl[o] = [h1, h2]
        excl[h1] = [o, h2]
        excl[h2] = [o, h1]
    return jnp.array(excl, dtype=jnp.int32)


def _equil_water_positions(n_waters: int, box_edge: float) -> np.ndarray:
    """Generate equilibrium TIP3P water positions on a grid."""
    n_per_side = int(np.ceil(n_waters ** (1.0 / 3.0)))
    spacing = box_edge / n_per_side

    positions = []
    water_count = 0

    for i in range(n_per_side):
        for j in range(n_per_side):
            for k in range(n_per_side):
                if water_count >= n_waters:
                    break

                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing
                z = (k + 0.5) * spacing

                # O atom
                positions.append([x, y, z])
                # H1 atom (offset along x)
                positions.append([x + 0.9572, y, z])
                # H2 atom (offset along y)
                positions.append([x, y + 0.9572, z])

                water_count += 1
            if water_count >= n_waters:
                break
        if water_count >= n_waters:
            break

    return np.array(positions[:n_waters * 3], dtype=np.float64)


def compute_T_rot_trans_one_water(
    p_o: jnp.ndarray,
    p_h1: jnp.ndarray,
    p_h2: jnp.ndarray,
    r_o: jnp.ndarray,
    r_h1: jnp.ndarray,
    r_h2: jnp.ndarray,
    mass_o: float,
    mass_h: float,
) -> tuple[float, float]:
    """Compute T_rot and T_trans for one water molecule.

    T_trans = (1/3) * (p_cm^2 / M) / kB
    T_rot = (1/3) * (L^2 / I) / kB  (via inertia-weighted angular velocity)
    """
    kB = BOLTZMANN_KCAL  # kcal/mol/K
    m_total = mass_o + 2 * mass_h

    # Center of mass velocity
    v_o = p_o / mass_o
    v_h1 = p_h1 / mass_h
    v_h2 = p_h2 / mass_h

    p_cm = mass_o * v_o + mass_h * v_h1 + mass_h * v_h2
    T_trans = jnp.sum(p_cm**2) / (3.0 * m_total * kB)

    # COM
    r_com = (mass_o * r_o + mass_h * r_h1 + mass_h * r_h2) / m_total

    # Angular momentum
    r_rel_o = r_o - r_com
    r_rel_h1 = r_h1 - r_com
    r_rel_h2 = r_h2 - r_com

    L = (mass_o * jnp.cross(r_rel_o, v_o) +
         mass_h * jnp.cross(r_rel_h1, v_h1) +
         mass_h * jnp.cross(r_rel_h2, v_h2))

    # Inertia tensor
    eye3 = jnp.eye(3)
    r_sq_o = jnp.sum(r_rel_o**2)
    r_sq_h1 = jnp.sum(r_rel_h1**2)
    r_sq_h2 = jnp.sum(r_rel_h2**2)

    I_tensor = (mass_o * (r_sq_o * eye3 - jnp.outer(r_rel_o, r_rel_o)) +
                mass_h * (r_sq_h1 * eye3 - jnp.outer(r_rel_h1, r_rel_h1)) +
                mass_h * (r_sq_h2 * eye3 - jnp.outer(r_rel_h2, r_rel_h2)))

    # Regularize for inversion
    reg = 1e-12 * (jnp.trace(I_tensor) / 3.0 + 1.0)
    I_inv = jnp.linalg.inv(I_tensor + reg * eye3)

    T_rot = 0.5 * jnp.dot(L, jnp.dot(I_inv, L)) / (3.0 * kB)

    return float(T_trans), float(T_rot)


def compute_T_rot_trans_all_waters(
    momentum: jnp.ndarray,
    positions: jnp.ndarray,
    water_indices: jnp.ndarray,
    mass_o: float = 15.999,
    mass_h: float = 1.008,
) -> tuple[float, float]:
    """Compute mean T_rot and T_trans across all waters."""
    n_waters = water_indices.shape[0]
    T_rots = []
    T_transs = []

    for w in range(n_waters):
        o_idx, h1_idx, h2_idx = water_indices[w]

        T_trans, T_rot = compute_T_rot_trans_one_water(
            momentum[o_idx], momentum[h1_idx], momentum[h2_idx],
            positions[o_idx], positions[h1_idx], positions[h2_idx],
            mass_o, mass_h
        )
        T_transs.append(T_trans)
        T_rots.append(T_rot)

    return float(np.mean(T_transs)), float(np.mean(T_rots))


def main():
    # --- Setup ---
    n_waters = 64
    box_edge = 20.0  # Å
    box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)
    n_atoms = n_waters * 3

    n_steps = 500
    dt_fs = 0.5
    temperature_k = 300.0
    gamma_ps = 10.0

    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3

    # --- Create energy and displacement functions ---
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)

    sys_dict = _proxide_params_pure_water(n_waters)
    energy_fn = system.make_energy_fn(
        displacement_fn, sys_dict, box=box_vec, use_pbc=True,
        implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34,
        cutoff_distance=9.0, strict_parameterization=False
    )

    # --- Build initial positions and masses ---
    positions_init = _equil_water_positions(n_waters, box_edge)
    positions = jnp.array(positions_init, dtype=jnp.float64)

    mass_array = np.array([[15.999], [1.008], [1.008]] * n_waters, dtype=np.float64).reshape(n_atoms)
    mass = jnp.array(mass_array, dtype=jnp.float64)
    mass_col = mass[:, None]

    water_indices = settle.get_water_indices(0, n_waters)

    # --- Initialize integrator (for force computation) ---
    init_s, apply_s = settle.settle_langevin(
        energy_fn, shift_fn, dt=dt_akma, kT=kT,
        gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec,
        remove_linear_com_momentum=False, project_ou_momentum_rigid=True,
        projection_site="post_o", settle_velocity_iters=10
    )

    key = jax.random.PRNGKey(42)
    state = init_s(key, positions, mass=mass)

    # --- Open CSV output ---
    output_path = Path("scripts/explore/p5_substep_trot_trace.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file = output_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "substep", "T_rot", "T_trans", "delta_T_rot", "delta_T_trans"])

    # --- Run integration with instrumentation ---
    momentum = state.momentum
    force = state.force
    positions_old = state.positions
    key = state.key

    # Helper for recording
    last_T_trans = None
    last_T_rot = None

    for step in range(n_steps):
        # Compute T at start of step
        T_trans_prev, T_rot_prev = compute_T_rot_trans_all_waters(
            momentum, state.positions, water_indices
        )

        # --- B1: half force kick ---
        momentum = momentum + 0.5 * dt_akma * force
        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, state.positions, water_indices)
        delta_T_trans = T_trans - T_trans_prev
        delta_T_rot = T_rot - T_rot_prev
        writer.writerow([step, "B1", T_rot, T_trans, delta_T_rot, delta_T_trans])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- A: position update (half) ---
        velocity = momentum / mass_col
        positions = shift_fn(state.positions, 0.5 * dt_akma * velocity)
        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        delta_T_trans = T_trans - T_trans_prev
        delta_T_rot = T_rot - T_rot_prev
        writer.writerow([step, "A", T_rot, T_trans, delta_T_rot, delta_T_trans])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- SETTLE_pos (first) ---
        positions_constrained = settle.settle_positions(
            positions, positions_old, water_indices, box=box_vec
        )
        # R-step: momentum correction
        dx = positions_constrained - positions
        if box_vec is not None:
            dx = dx - box_vec * jnp.round(dx / box_vec)
        dp = mass_col * dx / (0.5 * dt_akma)
        momentum = momentum + dp
        positions = positions_constrained

        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        delta_T_trans = T_trans - T_trans_prev
        delta_T_rot = T_rot - T_rot_prev
        writer.writerow([step, "SETTLE1", T_rot, T_trans, delta_T_rot, delta_T_trans])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- O: stochastic step with rigid projection ---
        c1 = jnp.exp(-gamma_reduced * dt_akma)
        c2 = jnp.sqrt(1 - c1**2)

        key, split = jax.random.split(key)
        noise = jax.random.normal(split, momentum.shape)
        momentum_ou = c1 * momentum + c2 * jnp.sqrt(mass_col * kT) * noise

        # Project onto rigid subspace
        momentum = settle.project_tip3p_waters_momentum_rigid(
            momentum_ou, positions, mass, water_indices
        )

        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        delta_T_trans = T_trans - T_trans_prev
        delta_T_rot = T_rot - T_rot_prev
        writer.writerow([step, "O", T_rot, T_trans, delta_T_rot, delta_T_trans])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- A: position update (second half) ---
        velocity = momentum / mass_col
        positions = shift_fn(positions, 0.5 * dt_akma * velocity)
        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        delta_T_trans = T_trans - T_trans_prev
        delta_T_rot = T_rot - T_rot_prev
        writer.writerow([step, "A2", T_rot, T_trans, delta_T_rot, delta_T_trans])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- SETTLE_pos (second) ---
        positions_mid = positions
        positions_constrained = settle.settle_positions(
            positions, positions_mid, water_indices, box=box_vec
        )
        # R-step: momentum correction
        dx = positions_constrained - positions
        if box_vec is not None:
            dx = dx - box_vec * jnp.round(dx / box_vec)
        dp = mass_col * dx / (0.5 * dt_akma)
        momentum = momentum + dp
        positions = positions_constrained

        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        delta_T_trans = T_trans - T_trans_prev
        delta_T_rot = T_rot - T_rot_prev
        writer.writerow([step, "SETTLE2", T_rot, T_trans, delta_T_rot, delta_T_trans])
        T_trans_prev, T_rot_prev = T_trans, T_rot

        # --- Compute force at new positions ---
        force = jax.grad(energy_fn)(positions)

        # --- B2: final half force kick ---
        momentum = momentum + 0.5 * dt_akma * force
        T_trans, T_rot = compute_T_rot_trans_all_waters(momentum, positions, water_indices)
        delta_T_trans = T_trans - T_trans_prev
        delta_T_rot = T_rot - T_rot_prev
        writer.writerow([step, "B2", T_rot, T_trans, delta_T_rot, delta_T_trans])

        # Update state for next step
        state = settle.NVTLangevinState(positions, momentum, force, mass_col, key)
        positions_old = positions

    csv_file.close()
    print(f"Trace written to {output_path}")

    # Print summary statistics
    import pandas as pd
    df = pd.read_csv(output_path)
    summary = df.groupby("substep")[["delta_T_rot", "delta_T_trans"]].mean()
    print("\nMean ΔT per substep:")
    print(summary)


if __name__ == "__main__":
    main()
