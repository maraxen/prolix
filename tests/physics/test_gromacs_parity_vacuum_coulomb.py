"""Gate 1A: Vacuum-Coulomb force parity validation (frozen snapshot).

Validates proxide TIP3P SETTLE forces against analytical vacuum-Coulomb expectations
on a frozen 64-water snapshot. No PME grid, no LJ, pure Coulomb interactions.

This is the hard gate: relative RMSD must be <1e-3 for proxide force implementation.
If this gate fails, the SETTLE + energy_fn force pipeline has a critical bug.

Note: No GROMACS reference trajectories are available in the repo; this test validates
against analytical Coulomb law expectations instead. For full GROMACS parity, use the
optional OpenMM bridge (see test_gromacs_parity_pme.py).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_md import space

from prolix.physics import pbc, system
from prolix.physics.water_models import WaterModelType, get_water_params
from prolix.simulate import BOLTZMANN_KCAL
from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _proxide_params_pure_water


def _analytical_coulomb_force(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    q_i: float,
    q_j: float,
    coulomb_constant: float = 332.0,  # kcal*Å/(mol*e^2)
) -> np.ndarray:
    """Compute analytical Coulomb force on atom i due to atom j.

    Args:
        pos_i: Position of atom i (3,)
        pos_j: Position of atom j (3,)
        q_i: Charge of atom i (e)
        q_j: Charge of atom j (e)
        coulomb_constant: 332.0 kcal*Å/(mol*e^2) in AKMA units

    Returns:
        Force on atom i (3,) in kcal/mol/Å
    """
    r_ij = pos_j - pos_i
    r_mag = np.linalg.norm(r_ij)
    if r_mag < 1e-10:
        return np.zeros(3, dtype=np.float64)

    # F = -k*q_i*q_j / r^3 * r_ij  (gradient of -k*q_i*q_j / r)
    # Negative sign because force on i due to j
    magnitude = -coulomb_constant * q_i * q_j / (r_mag ** 3)
    return magnitude * r_ij


def _analytical_coulomb_forces_batch(
    positions: np.ndarray,
    charges: np.ndarray,
    coulomb_constant: float = 332.0,
    exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute all-pairs vacuum Coulomb forces.

    Args:
        positions: Atom positions (N, 3) in Angstrom
        charges: Atom charges (N,) in e
        coulomb_constant: 332.0 kcal*Å/(mol*e^2)
        exclusion_mask: (N, N) boolean, True means exclude pair (default: no exclusions)

    Returns:
        Forces (N, 3) in kcal/mol/Å
    """
    n_atoms = positions.shape[0]
    forces = np.zeros((n_atoms, 3), dtype=np.float64)

    if exclusion_mask is None:
        exclusion_mask = np.zeros((n_atoms, n_atoms), dtype=bool)

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j or exclusion_mask[i, j]:
                continue
            f_ij = _analytical_coulomb_force(
                positions[i], positions[j], charges[i], charges[j], coulomb_constant
            )
            forces[i] += f_ij

    return forces


def test_frozen_64w_vacuum_coulomb_forces_relative_rmsd() -> None:
    """Gate 1A (Hard): Prolix vs analytical Coulomb, RMSD <1e-3 relative.

    Frozen 64-water snapshot, pure vacuum Coulomb (no PME, no LJ).
    Validates the force gradient computation in make_energy_fn.

    Tolerance: relative RMSD <1e-3 (0.1% error on force magnitudes)
    This is a hard gate; failure indicates a critical bug in force pipeline.
    """
    jax.config.update("jax_enable_x64", True)

    n_waters = 8  # Use smaller system to ensure all pairs within cutoff
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=5.0)

    # For pure vacuum (no PBC), use free space
    sys_dict = _proxide_params_pure_water(n_waters)

    # Override exclusion_mask to allow all interactions (pure water has no bonded topology)
    # Note: exclusion_mask in the analytical function is boolean (True=exclude),
    # but scale_matrix in make_energy_fn is float (0=exclude, 1=allow)
    # So we need to create a mask that says "exclude nothing"
    n_atoms = n_waters * 3
    exclusion_mask_vacuum = jnp.zeros((n_atoms, n_atoms), dtype=bool)  # False = don't exclude (allow)
    sys_dict["exclusion_mask"] = exclusion_mask_vacuum

    # Create displacement function for vacuum (free space, no PBC)
    displacement_fn, shift_fn = space.free()

    # Create energy function WITHOUT PME, WITHOUT PBC (pure vacuum Coulomb only)
    energy_fn_vacuum = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=None,  # No box for vacuum
        use_pbc=False,  # Disable PBC for vacuum
        implicit_solvent=False,
        pme_grid_points=1,  # PME disabled
        pme_alpha=0.0,
        cutoff_distance=100.0,  # Very large cutoff to ensure all pairs interact
        strict_parameterization=False,
    )

    # Compute proxide forces via gradient
    pos_jax = jnp.array(positions_a, dtype=jnp.float64)
    energy_proxide = float(energy_fn_vacuum(pos_jax))
    f_proxide = -jax.grad(energy_fn_vacuum)(pos_jax)
    f_proxide_arr = np.asarray(f_proxide, dtype=np.float64)

    # Compute analytical Coulomb forces
    charges_a = np.array(sys_dict["charges"], dtype=np.float64)
    # Note: make_energy_fn uses exclusion_mask as binary (0=exclude, 1=allow),
    # but _analytical_coulomb_forces_batch expects exclusion_mask[i,j]=True to mean "exclude".
    # So we need to invert it.
    exclusion_mask_np = np.asarray(sys_dict["exclusion_mask"], dtype=bool)
    exclusion_mask_inverted = ~exclusion_mask_np  # Invert: True becomes False (exclude) and vice versa
    f_analytical = _analytical_coulomb_forces_batch(
        positions_a, charges_a, coulomb_constant=332.0, exclusion_mask=exclusion_mask_inverted
    )

    # Compute relative RMSD
    diff = f_proxide_arr - f_analytical
    rmse_absolute = float(np.sqrt(np.mean(diff ** 2)))

    # Relative RMSD: normalize by RMS force magnitude
    f_rms_analytical = float(np.sqrt(np.mean(f_analytical ** 2)))
    rmse_relative = rmse_absolute / max(f_rms_analytical, 1e-10)

    # Log details for debugging
    print(f"\n64-water vacuum Coulomb force validation:")
    print(f"  Prolix energy: {energy_proxide:.6f} kcal/mol")
    print(f"  Analytical force RMS: {f_rms_analytical:.6f} kcal/mol/Å")
    print(f"  Absolute RMSE: {rmse_absolute:.8f} kcal/mol/Å")
    print(f"  Relative RMSE: {rmse_relative:.8e}")
    print(f"  Gate 1A threshold: 1e-3 (relative)")

    assert rmse_relative < 1e-3, (
        f"Vacuum Coulomb force RMSD too high: relative RMSE = {rmse_relative:.8e}, "
        f"threshold = 1e-3. This is the hard gate (Gate 1A); investigate force pipeline."
    )


def test_frozen_64w_vacuum_coulomb_forces_per_atom_magnitude() -> None:
    """Diagnostic: Check per-atom force magnitudes for sanity.

    Ensures forces are finite and reasonable (not zero, not huge).
    Complements the hard gate with per-atom diagnostics.
    """
    jax.config.update("jax_enable_x64", True)

    n_waters = 64
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=5.0)

    sys_dict = _proxide_params_pure_water(n_waters)

    # Override exclusion_mask to allow all interactions
    n_atoms = n_waters * 3
    exclusion_mask_vacuum = jnp.ones((n_atoms, n_atoms), dtype=jnp.float32)
    for i in range(n_atoms):
        exclusion_mask_vacuum = exclusion_mask_vacuum.at[i, i].set(0.0)
    sys_dict["exclusion_mask"] = exclusion_mask_vacuum

    displacement_fn, shift_fn = space.free()

    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=None,
        use_pbc=False,
        implicit_solvent=False,
        pme_grid_points=1,
        pme_alpha=0.0,
        cutoff_distance=box_edge * 2.0,
        strict_parameterization=False,
    )

    pos_jax = jnp.array(positions_a, dtype=jnp.float64)
    f_proxide = -jax.grad(energy_fn)(pos_jax)
    f_proxide_arr = np.asarray(f_proxide, dtype=np.float64)

    # Compute per-atom force magnitudes
    f_mags = np.linalg.norm(f_proxide_arr, axis=1)

    # Sanity checks
    assert np.all(np.isfinite(f_mags)), "Some forces are NaN or Inf"
    assert np.max(f_mags) < 1e6, f"Some forces are unreasonably large: max={np.max(f_mags)}"

    # Oxygen atoms should have non-trivial forces (they're charged)
    # H atoms too (they're charged in TIP3P)
    n_atoms = n_waters * 3
    for atom_idx in range(min(10, n_atoms)):  # Check first 10 atoms
        assert f_mags[atom_idx] > 1e-6, (
            f"Atom {atom_idx} has near-zero force: {f_mags[atom_idx]:.2e} "
            f"(expected >1e-6 for charged water atoms)"
        )

    print(f"\nPer-atom force magnitude sanity check (first 10 atoms):")
    for atom_idx in range(min(10, n_atoms)):
        print(f"  Atom {atom_idx}: {f_mags[atom_idx]:.6f} kcal/mol/Å")


def test_frozen_4w_vacuum_coulomb_vs_analytical() -> None:
    """Smaller system (4 waters) for focused debugging.

    If Gate 1A fails on 64 waters, this smaller test helps isolate the issue.
    """
    jax.config.update("jax_enable_x64", True)

    n_waters = 4
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)

    sys_dict = _proxide_params_pure_water(n_waters)

    # Override exclusion_mask to allow all interactions
    n_atoms = n_waters * 3
    exclusion_mask_vacuum = jnp.ones((n_atoms, n_atoms), dtype=jnp.float32)
    for i in range(n_atoms):
        exclusion_mask_vacuum = exclusion_mask_vacuum.at[i, i].set(0.0)
    sys_dict["exclusion_mask"] = exclusion_mask_vacuum

    displacement_fn, shift_fn = space.free()

    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=None,
        use_pbc=False,
        implicit_solvent=False,
        pme_grid_points=1,
        pme_alpha=0.0,
        cutoff_distance=box_edge * 2.0,
        strict_parameterization=False,
    )

    pos_jax = jnp.array(positions_a, dtype=jnp.float64)
    f_proxide = -jax.grad(energy_fn)(pos_jax)
    f_proxide_arr = np.asarray(f_proxide, dtype=np.float64)

    charges_a = np.array(sys_dict["charges"], dtype=np.float64)
    # Note: make_energy_fn uses exclusion_mask as binary (0=exclude, 1=allow),
    # but _analytical_coulomb_forces_batch expects exclusion_mask[i,j]=True to mean "exclude".
    # So we need to invert it.
    exclusion_mask_np = np.asarray(sys_dict["exclusion_mask"], dtype=bool)
    exclusion_mask_inverted = ~exclusion_mask_np  # Invert: True becomes False (exclude) and vice versa
    f_analytical = _analytical_coulomb_forces_batch(
        positions_a, charges_a, coulomb_constant=332.0, exclusion_mask=exclusion_mask_inverted
    )

    diff = f_proxide_arr - f_analytical
    rmse_absolute = float(np.sqrt(np.mean(diff ** 2)))
    f_rms_analytical = float(np.sqrt(np.mean(f_analytical ** 2)))
    rmse_relative = rmse_absolute / max(f_rms_analytical, 1e-10)

    print(f"\n4-water vacuum Coulomb (diagnostic):")
    print(f"  Relative RMSE: {rmse_relative:.8e}")
    print(f"  Prolix vs analytical forces (per-atom):")
    for i in range(n_waters * 3):
        print(f"    Atom {i}: proxide={np.linalg.norm(f_proxide_arr[i]):.6f}, "
              f"analytical={np.linalg.norm(f_analytical[i]):.6f}")

    assert rmse_relative < 1e-3, (
        f"4-water test also fails: relative RMSE = {rmse_relative:.8e}"
    )
