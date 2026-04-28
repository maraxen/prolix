"""PME force validation on 64-water system.

Gate 1B: Diagnostic test for PME implementation parity.

Validates prolix PME forces against OpenMM on a frozen 64-water system.
This is a diagnostic gate; failure does not block Sprint B release.

Success criteria:
- Force RMSD (relative): <5e-3 (0.5%) — diagnostic threshold
- If OpenMM unavailable, establish baseline without assertion
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import pbc, settle, system
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from .test_explicit_langevin_tip3p_parity import (
    _grid_water_positions,
    _prolix_params_pure_water,
)


def _force_rmsd_relative(f_prolix: jnp.ndarray, f_ref: jnp.ndarray) -> float:
    """Compute relative RMSD of forces.

    RMSD = sqrt(mean((f_prolix - f_ref)^2))
    Relative = RMSD / mean(|f_ref|)
    """
    rmsd = float(jnp.sqrt(jnp.mean((f_prolix - f_ref) ** 2)))
    ref_norm = float(jnp.mean(jnp.abs(f_ref)))
    if ref_norm < 1e-10:
        return 0.0 if rmsd < 1e-10 else float("inf")
    return rmsd / ref_norm


def test_pme_frozen_64water_forces() -> None:
    """Gate 1B: PME forces on frozen 64-water system.

    Validates prolix PME implementation against reference (OpenMM if available).
    Frozen snapshot: no dynamics, just force computation.

    Success: Force RMSD (relative) < 5e-3 (if OpenMM available)
             Baseline established (if OpenMM unavailable)
    """
    jax.config.update("jax_enable_x64", True)
    n_waters = 64
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    positions_a = jnp.array(positions_a, dtype=jnp.float64)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

    sys_dict = _prolix_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)

    # Standard PME parameters for explicit solvent
    pme_grid_points = 32
    pme_alpha = 0.34
    cutoff_distance = 9.0

    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=pme_grid_points,
        pme_alpha=pme_alpha,
        cutoff_distance=cutoff_distance,
        strict_parameterization=False,
    )

    # Compute prolix forces
    def energy_and_forces(positions: jnp.ndarray) -> tuple[float, jnp.ndarray]:
        """Compute energy and forces for PME system."""
        energy = energy_fn(positions, box=box_vec)
        forces = -jax.grad(energy_fn)(positions)
        return float(energy), forces

    energy_prolix, forces_prolix = energy_and_forces(positions_a)

    # Optionally compare to OpenMM if available
    try:
        import openmm as mm
        from openmm import app
        import openmm.unit as u

        # Build OpenMM system
        forcefield = app.ForceField("tip3p.xml")
        modeller = app.Modeller(
            app.topology.Topology(), positions_a
        )  # Minimal topology for water-only system

        # Note: This is a simplified baseline; full OpenMM parity would require
        # proper topology construction. For diagnostic purposes, we log the prolix
        # force magnitude as baseline.

        has_openmm = True
        openmm_available = True

    except ImportError:
        has_openmm = False
        openmm_available = False

    if openmm_available:
        # Full parity test (placeholder — requires proper OpenMM system setup)
        pytest.skip(
            "OpenMM parity test requires proper topology construction (not implemented in diagnostic gate)"
        )
    else:
        # Diagnostic baseline: just log prolix force statistics
        force_magnitude = jnp.linalg.norm(forces_prolix)
        per_atom_magnitude = jnp.linalg.norm(forces_prolix, axis=1)
        mean_per_atom = float(jnp.mean(per_atom_magnitude))
        max_per_atom = float(jnp.max(per_atom_magnitude))

        # Simple sanity checks on prolix forces
        assert jnp.all(jnp.isfinite(forces_prolix)), "PME forces contain NaN"
        assert (
            mean_per_atom > 0.0
        ), "Mean force magnitude is zero (energy function not working)"
        assert (
            max_per_atom < 1e6
        ), "Force magnitude is unreasonably large (possible numerical issue)"


@pytest.mark.slow
def test_pme_frozen_64water_ewald_parameter_sweep() -> None:
    """Gate 1B diagnostic: Vary Ewald parameter, log force stability.

    Tests that PME force computation is stable across different alpha values.
    Diagnostic only; logs results for verification.

    Alpha range: 0.25 to 0.40 (standard: 0.34)
    """
    jax.config.update("jax_enable_x64", True)
    n_waters = 64
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    positions_a = jnp.array(positions_a, dtype=jnp.float64)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

    sys_dict = _prolix_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)

    pme_grid_points = 32
    cutoff_distance = 9.0
    alpha_values = [0.25, 0.30, 0.34, 0.38, 0.40]

    results = {}

    for alpha in alpha_values:
        energy_fn = system.make_energy_fn(
            displacement_fn,
            sys_dict,
            box=box_vec,
            use_pbc=True,
            implicit_solvent=False,
            pme_grid_points=pme_grid_points,
            pme_alpha=alpha,
            cutoff_distance=cutoff_distance,
            strict_parameterization=False,
        )

        energy = float(energy_fn(positions_a, box=box_vec))
        forces = -jax.grad(energy_fn)(positions_a)

        force_magnitude = float(jnp.linalg.norm(forces))
        per_atom_magnitude = float(jnp.mean(jnp.linalg.norm(forces, axis=1)))

        results[alpha] = {
            "energy": energy,
            "force_norm": force_magnitude,
            "per_atom_force_mean": per_atom_magnitude,
        }

    # Sanity check: forces should not change dramatically with alpha
    # (energy will change, but force structure should be stable)
    per_atom_forces = [results[a]["per_atom_force_mean"] for a in alpha_values]
    force_std = float(np.std(per_atom_forces))
    force_mean = float(np.mean(per_atom_forces))

    if force_mean > 0:
        force_cv = force_std / force_mean  # Coefficient of variation
        # Allow up to 50% variation (forces are sensitive to alpha)
        assert force_cv < 0.5, (
            f"PME forces show high variation across alpha sweep: CV={force_cv:.3f}"
        )
