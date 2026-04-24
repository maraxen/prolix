"""Sprint 2 evaluation harness for EFA vs PME comparison.

Provides utilities for force accuracy metrics, NVE/NVT simulation,
and energy consistency checks across electrostatic methods.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from prolix.padding import PaddedSystem
from prolix.physics import pbc
from prolix.physics.electrostatic_methods import ElectrostaticMethod
from prolix.physics.flash_explicit import (
    flash_explicit_energy,
    flash_explicit_forces,
)
from prolix.physics.water_models import WaterModelType, get_water_params

if TYPE_CHECKING:
    from jax_md.util import Array


def _tip3p_local_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct local coordinate frame for a TIP3P water molecule.

    Returns:
        (o_pos, h1_pos, h2_pos): positions relative to origin for O, H1, H2.
    """
    tip = get_water_params(WaterModelType.TIP3P)
    r = float(tip.r_OH)
    theta = 104.52 * math.pi / 180.0
    o = np.zeros(3, dtype=np.float64)
    h1 = np.array([r, 0.0, 0.0])
    h2 = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])
    return o, h1, h2


def _grid_water_positions(
    n_waters: int, spacing: float = 3.1
) -> tuple[np.ndarray, float]:
    """Generate a grid of water molecule positions.

    Args:
        n_waters: Number of water molecules to place.
        spacing: Spacing between grid points in Angstroms.

    Returns:
        (positions_array, box_edge): Array of shape (3*n_waters, 3) with
            O, H1, H2 coords for each water, and cubic box edge length.
    """
    o0, h1l, h2l = _tip3p_local_frame()
    sites: list[tuple[int, int, int]] = []
    n = int(math.ceil(n_waters ** (1.0 / 3.0))) + 3
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                sites.append((ix, iy, iz))
                if len(sites) >= n_waters:
                    break
            if len(sites) >= n_waters:
                break
        if len(sites) >= n_waters:
            break
    sites = sites[:n_waters]

    base = np.array([3.0, 3.0, 3.0], dtype=np.float64)
    pos: list[np.ndarray] = []
    for ix, iy, iz in sites:
        o = base + np.array(
            [ix * spacing, iy * spacing, iz * spacing], dtype=np.float64
        )
        pos.append(o + o0)
        pos.append(o + h1l)
        pos.append(o + h2l)
    arr = np.vstack(pos)
    span = np.max(arr, axis=0) - np.min(arr, axis=0)
    box_edge = float(np.max(span) + 16.0)
    return arr, box_edge


def make_tip3p_water_system(
    n_waters: int, spacing: float = 3.1, seed: int = 0
) -> PaddedSystem:
    """Create a PaddedSystem for n_waters TIP3P molecules.

    Args:
        n_waters: Number of water molecules.
        spacing: Spacing in Angstroms.
        seed: Random seed (for future velocity initialization).

    Returns:
        PaddedSystem with all required fields for flash_explicit_* functions.
    """
    tip = get_water_params(WaterModelType.TIP3P)
    qo, qh = float(tip.charge_O), float(tip.charge_H)
    sig_o = float(tip.sigma_O)
    eps_o = float(tip.epsilon_O)

    positions_np, box_edge = _grid_water_positions(n_waters, spacing=spacing)
    positions = jnp.array(positions_np, dtype=jnp.float64)

    n_atoms = n_waters * 3
    charges_list = []
    sigmas_list = []
    epsilons_list = []
    for _ in range(n_waters):
        charges_list.extend([qo, qh, qh])
        sigmas_list.extend([sig_o, 1.0, 1.0])
        epsilons_list.extend([eps_o, 0.0, 0.0])

    charges = jnp.array(charges_list, dtype=jnp.float64)
    sigmas = jnp.array(sigmas_list, dtype=jnp.float64)
    epsilons = jnp.array(epsilons_list, dtype=jnp.float64)
    masses = jnp.array(
        [[15.999], [1.008], [1.008]] * n_waters, dtype=jnp.float64
    ).reshape(n_atoms)

    atom_mask = jnp.ones(n_atoms, dtype=jnp.bool_)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

    # Create sparse exclusion for water (O-H bonds only, all intramolecular)
    from prolix.utils import topology
    from prolix.padding import pad_array, select_bucket

    bonds_list = []
    for w in range(n_waters):
        b = w * 3
        bonds_list.append([b, b + 1])  # O-H1
        bonds_list.append([b, b + 2])  # O-H2
    bonds = jnp.array(bonds_list, dtype=jnp.int32) if bonds_list else jnp.zeros((0, 2), dtype=jnp.int32)

    # Find bonded exclusions: 1-2 (bonds), 1-3 (H-H pair), 1-4 (none for water)
    if bonds.shape[0] > 0:
        excl = topology.find_bonded_exclusions(bonds, n_atoms)
        idx_12 = excl.idx_12
        idx_13 = excl.idx_13
    else:
        idx_12 = jnp.zeros((0, 2), dtype=jnp.int32)
        idx_13 = jnp.zeros((0, 2), dtype=jnp.int32)

    # Build PaddedSystem using PaddedSystemBuilder or direct construction
    bucket = select_bucket(n_atoms)

    # Pad all arrays to bucket size
    charges_padded = pad_array(charges, bucket, 0.0)
    sigmas_padded = pad_array(sigmas, bucket, 1.0)
    epsilons_padded = pad_array(epsilons, bucket, 0.0)
    masses_padded = pad_array(masses, bucket, 1.0)
    atom_mask_padded = pad_array(atom_mask, bucket, False)
    positions_padded = jnp.pad(
        positions, ((0, bucket - n_atoms), (0, 0)), constant_values=0.0
    )

    # Pad bonds to bucket size
    bonds_padded = jnp.pad(
        bonds, ((0, bucket - bonds.shape[0]), (0, 0)), constant_values=0
    )
    bond_mask_padded = pad_array(
        jnp.ones(bonds.shape[0], dtype=jnp.bool_) if bonds.shape[0] > 0 else jnp.zeros(0, dtype=jnp.bool_),
        bucket,
        False,
    )

    # Populate sparse exclusion arrays (6 exclusions per atom max)
    max_excl = 6
    excl_indices_padded = jnp.full((bucket, max_excl), -1, dtype=jnp.int32)
    excl_scales_vdw_padded = jnp.ones((bucket, max_excl), dtype=jnp.float32)
    excl_scales_elec_padded = jnp.ones((bucket, max_excl), dtype=jnp.float32)

    # Fill exclusion arrays from idx_12 and idx_13 using vmap-compatible approach
    # 1-2 and 1-3 pairs: scale = 0.0 (fully excluded)
    if idx_12.shape[0] > 0 or idx_13.shape[0] > 0:
        idx_12_13 = jnp.concatenate([idx_12, idx_13], axis=0) if idx_12.shape[0] > 0 and idx_13.shape[0] > 0 else (
            idx_12 if idx_12.shape[0] > 0 else idx_13
        )
        # Simple approach: for each exclusion pair, scatter into the sparse arrays
        for pair_idx in range(idx_12_13.shape[0]):
            i_pair = int(idx_12_13[pair_idx, 0])
            j_pair = int(idx_12_13[pair_idx, 1])
            # Find first slot with -1 and fill it
            if i_pair < bucket and j_pair < bucket:
                for slot in range(max_excl):
                    if excl_indices_padded[i_pair, slot] == -1:
                        excl_indices_padded = excl_indices_padded.at[i_pair, slot].set(j_pair)
                        excl_scales_vdw_padded = excl_scales_vdw_padded.at[i_pair, slot].set(0.0)
                        excl_scales_elec_padded = excl_scales_elec_padded.at[i_pair, slot].set(0.0)
                        break

    # Build water indices for SETTLE
    water_indices_list = []
    for w in range(n_waters):
        b = w * 3
        water_indices_list.append([b, b + 1, b + 2])
    water_indices = jnp.array(water_indices_list, dtype=jnp.int32)

    sys = PaddedSystem(
        positions=positions_padded,
        charges=charges_padded,
        sigmas=sigmas_padded,
        epsilons=epsilons_padded,
        radii=jnp.zeros(bucket, dtype=jnp.float64),
        scaled_radii=jnp.zeros(bucket, dtype=jnp.float64),
        masses=masses_padded,
        element_ids=jnp.zeros(bucket, dtype=jnp.int32),
        atom_mask=atom_mask_padded,
        is_hydrogen=jnp.zeros(bucket, dtype=jnp.bool_),
        is_backbone=jnp.zeros(bucket, dtype=jnp.bool_),
        is_heavy=jnp.zeros(bucket, dtype=jnp.bool_),
        protein_atom_mask=jnp.zeros(bucket, dtype=jnp.bool_),
        water_atom_mask=atom_mask_padded,
        bonds=bonds_padded,
        bond_params=jnp.zeros((bucket, 2), dtype=jnp.float64),
        bond_mask=bond_mask_padded,
        angles=jnp.zeros((bucket, 3), dtype=jnp.int32),
        angle_params=jnp.zeros((bucket, 2), dtype=jnp.float64),
        angle_mask=jnp.zeros(bucket, dtype=jnp.bool_),
        dihedrals=jnp.zeros((bucket, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((bucket, 3), dtype=jnp.float64),
        dihedral_mask=jnp.zeros(bucket, dtype=jnp.bool_),
        impropers=jnp.zeros((bucket, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((bucket, 3), dtype=jnp.float64),
        improper_mask=jnp.zeros(bucket, dtype=jnp.bool_),
        urey_bradley_bonds=jnp.zeros((bucket, 2), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((bucket, 2), dtype=jnp.float64),
        urey_bradley_mask=jnp.zeros(bucket, dtype=jnp.bool_),
        cmap_torsions=None,
        cmap_indices=None,
        cmap_mask=None,
        cmap_coeffs=None,
        excl_indices=excl_indices_padded,
        excl_scales_vdw=excl_scales_vdw_padded,
        excl_scales_elec=excl_scales_elec_padded,
        constraint_pairs=jnp.zeros((bucket, 2), dtype=jnp.int32),
        constraint_lengths=jnp.zeros(bucket, dtype=jnp.float64),
        constraint_mask=jnp.zeros(bucket, dtype=jnp.bool_),
        n_real_atoms=n_atoms,
        n_padded_atoms=bucket,
        bucket_size=bucket,
        water_indices=jnp.pad(
            water_indices, ((0, bucket - n_waters), (0, 0)), constant_values=0
        ),
        water_mask=jnp.pad(
            jnp.ones(n_waters, dtype=jnp.bool_), (0, bucket - n_waters), constant_values=False
        ),
        box_size=box_vec,
        pme_alpha=0.34,
        pme_grid_points=32,
        nonbonded_cutoff=9.0,
        dense_excl_scale_vdw=None,
        dense_excl_scale_elec=None,
    )

    return sys


def velocity_verlet_step(
    pos: jnp.ndarray, vel: jnp.ndarray, forces_fn, dt: float, mass: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Perform one velocity Verlet integration step.

    Args:
        pos: Positions (N, 3).
        vel: Velocities (N, 3).
        forces_fn: Function that returns forces given positions.
        dt: Timestep.
        mass: Masses (N,).

    Returns:
        (new_pos, new_vel): Updated positions and velocities.
    """
    # F = ma => a = F/m
    f_old = forces_fn(pos)
    a_old = f_old / mass[:, None]

    # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    pos_new = pos + vel * dt + 0.5 * a_old * (dt ** 2)

    # Compute forces at new position
    f_new = forces_fn(pos_new)
    a_new = f_new / mass[:, None]

    # v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
    vel_new = vel + 0.5 * (a_old + a_new) * dt

    return pos_new, vel_new


def run_nve(
    system: PaddedSystem,
    forces_fn,
    n_steps: int,
    dt: float,
    freeze_geometry: bool = False,
) -> dict:
    """Run NVE (microcanonical) simulation without thermostat.

    Args:
        system: Initial PaddedSystem with positions and masses.
        forces_fn: Function sys -> forces (N, 3).
        n_steps: Number of integration steps.
        dt: Timestep (AKMA units).
        freeze_geometry: If True, don't update positions/velocities;
            just compute forces at fixed geometry. Returns energy dict
            with single snapshot. If False, standard velocity Verlet.

    Returns:
        Dictionary with:
        - 'positions': (n_snapshots, N, 3) array of positions.
        - 'velocities': (n_snapshots, N, 3) array of velocities.
        - 'kinetic_energy': (n_snapshots,) KE array.
        - 'potential_energy': (n_snapshots,) PE array.
        - 'total_energy': (n_snapshots,) E_total array.

        For freeze_geometry=True, n_snapshots=1; otherwise n_snapshots=n_steps+1.
    """
    from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal

    pos = system.positions[: system.n_real_atoms]
    mass = system.masses[: system.n_real_atoms]
    n_atoms = pos.shape[0]

    # Initialize velocities
    vel = jnp.zeros_like(pos)

    # Storage
    positions_list = [jnp.array(pos)]
    velocities_list = [jnp.array(vel)]
    ke_list = []
    pe_list = []

    if freeze_geometry:
        # Compute energy only at initial geometry
        sys_updated = system.replace(
            positions=jnp.pad(
                pos,
                ((0, system.n_padded_atoms - n_atoms), (0, 0)),
                constant_values=0.0,
            )
        )
        f = forces_fn(sys_updated)
        e = flash_explicit_energy(sys_updated, electrostatic_method=ElectrostaticMethod.PME)
        ke = float(0.0)  # No motion
        pe = float(e)
        ke_list.append(ke)
        pe_list.append(pe)
    else:
        # Standard NVE integration
        # Define closure to recompute forces at current position
        def forces_fn_at_pos(pos_local):
            sys_updated = system.replace(
                positions=jnp.pad(
                    pos_local,
                    ((0, system.n_padded_atoms - n_atoms), (0, 0)),
                    constant_values=0.0,
                )
            )
            f = forces_fn(sys_updated)
            return f[:n_atoms]

        for step in range(n_steps):
            # Create a temporary system with updated positions for energy calculation
            sys_updated = system.replace(
                positions=jnp.pad(
                    pos,
                    ((0, system.n_padded_atoms - n_atoms), (0, 0)),
                    constant_values=0.0,
                )
            )

            # Compute energy at current state
            e = flash_explicit_energy(sys_updated, electrostatic_method=ElectrostaticMethod.PME)

            # KE
            ke = float(rigid_tip3p_box_ke_kcal(pos, mass * vel, mass, n_atoms // 3))

            ke_list.append(ke)
            pe_list.append(float(e))

            # Integrate with proper closure that recomputes forces
            pos, vel = velocity_verlet_step(pos, vel, forces_fn_at_pos, dt, mass)

            positions_list.append(jnp.array(pos))
            velocities_list.append(jnp.array(vel))

    return {
        "positions": jnp.stack(positions_list),
        "velocities": jnp.stack(velocities_list),
        "kinetic_energy": jnp.array(ke_list),
        "potential_energy": jnp.array(pe_list),
        "total_energy": jnp.array(ke_list) + jnp.array(pe_list),
    }


def run_nvt(
    system: PaddedSystem,
    settle_langevin_init_fn,
    settle_langevin_apply_fn,
    n_steps: int,
    dt: float = 0.5,
) -> dict:
    """Run NVT (canonical) simulation using SETTLE + Langevin thermostat.

    Args:
        system: Initial PaddedSystem.
        settle_langevin_init_fn: Init function from settle.settle_langevin.
        settle_langevin_apply_fn: Apply function from settle.settle_langevin.
        n_steps: Number of steps.
        dt: Timestep (AKMA units, default 0.5 = 0.5 fs).

    Returns:
        Dictionary with temperature and energy traces.
    """
    from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal

    n_waters = system.water_indices[: system.n_real_atoms // 3].shape[0]
    dof_rigid = 6 * n_waters - 3

    mass = system.masses[: system.n_real_atoms]
    pos = system.positions[: system.n_real_atoms]

    state = settle_langevin_init_fn(
        jax.random.PRNGKey(0), pos, mass=mass
    )
    apply_j = jax.jit(settle_langevin_apply_fn)

    temps = []
    for step in range(n_steps):
        state = apply_j(state)
        ke_r = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
        temp = 2.0 * ke_r / (dof_rigid * 0.001987204)  # BOLTZMANN_KCAL
        temps.append(temp)

    return {
        "temperature_trace": jnp.array(temps),
        "mean_temperature": float(jnp.mean(jnp.array(temps))),
        "std_temperature": float(jnp.std(jnp.array(temps))),
    }


def force_rmse(
    f1: jnp.ndarray, f2: jnp.ndarray, mask: jnp.ndarray
) -> float:
    """Compute RMSE of forces over masked atoms.

    Args:
        f1, f2: Force arrays (N, 3).
        mask: Boolean mask (N,).

    Returns:
        Scalar RMSE value.
    """
    diff = f1[mask] - f2[mask]
    mse = jnp.mean(diff ** 2)
    return float(jnp.sqrt(mse))


def force_cosine_similarity(
    f1: jnp.ndarray, f2: jnp.ndarray, mask: jnp.ndarray
) -> float:
    """Compute cosine similarity of masked force vectors.

    Args:
        f1, f2: Force arrays (N, 3).
        mask: Boolean mask (N,).

    Returns:
        Scalar cosine similarity in [-1, 1].
    """
    f1_masked = f1[mask].reshape(-1)
    f2_masked = f2[mask].reshape(-1)
    dot = jnp.dot(f1_masked, f2_masked)
    norm1 = jnp.linalg.norm(f1_masked)
    norm2 = jnp.linalg.norm(f2_masked)
    cos_sim = dot / (norm1 * norm2 + 1e-12)
    return float(cos_sim)


def compute_temperature(
    vel: jnp.ndarray, mass: jnp.ndarray, mask: jnp.ndarray
) -> float:
    """Compute temperature from velocities.

    Args:
        vel: Velocities (N, 3).
        mass: Masses (N,).
        mask: Boolean atom mask (N,).

    Returns:
        Temperature in Kelvin.
    """
    BOLTZMANN_KCAL = 0.001987204
    n_dof = 3 * jnp.sum(mask) - 3  # Subtract COM
    ke = jnp.sum(0.5 * mass[mask, None] * vel[mask] ** 2)
    temp = 2.0 * ke / (n_dof * BOLTZMANN_KCAL)
    return float(temp)


def compute_total_energy(ke: float, pe: float) -> float:
    """Compute total energy as sum of kinetic and potential.

    Args:
        ke: Kinetic energy.
        pe: Potential energy.

    Returns:
        Total energy.
    """
    return ke + pe


def make_comparison_energies(
    system: PaddedSystem,
    n_seeds: int = 16,
    n_features: int = 512,
    alpha: float = 0.34,
) -> dict:
    """Compute EFA vs PME forces and energies for comparison.

    Args:
        system: PaddedSystem with fixed geometry.
        n_seeds: Number of RFF seeds to sample.
        n_features: Number of RFF features (D).
        alpha: Ewald damping parameter.

    Returns:
        Dictionary with:
        - 'pme_forces': (N, 3) PME reference forces.
        - 'efa_forces_per_seed': (n_seeds, N, 3) EFA forces, one per seed.
        - 'pme_energy': scalar PME energy.
        - 'efa_energies_per_seed': (n_seeds,) EFA energies.
    """
    # PME reference
    pme_forces = flash_explicit_forces(system, electrostatic_method=ElectrostaticMethod.PME)
    pme_energy = flash_explicit_energy(system, electrostatic_method=ElectrostaticMethod.PME)

    # EFA with multiple seeds
    efa_forces_list = []
    efa_energies_list = []

    for seed in range(n_seeds):
        efa_f = flash_explicit_forces(
            system,
            electrostatic_method=ElectrostaticMethod.EFA,
            n_rff_features=n_features,
            rff_seed=seed,
        )
        efa_e = flash_explicit_energy(
            system,
            electrostatic_method=ElectrostaticMethod.EFA,
            n_rff_features=n_features,
            rff_seed=seed,
        )
        efa_forces_list.append(efa_f)
        efa_energies_list.append(float(efa_e))

    return {
        "pme_forces": pme_forces,
        "efa_forces_per_seed": jnp.stack(efa_forces_list),
        "pme_energy": float(pme_energy),
        "efa_energies_per_seed": jnp.array(efa_energies_list),
    }
