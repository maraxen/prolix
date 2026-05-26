"""OpenMM parity tests for bonded energy and forces.

Phase 2 (energy): per-term |dE| < 0.05 kcal/mol
Phase 3 (force): RMS(|df|) < 0.01 kcal/mol/Å

Tests are module-level functions (not class-wrapped) so the spec's literal
gate commands resolve correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable x64 for float64 support
jax.config.update("jax_enable_x64", True)

# Register openmm marker
pytestmark = pytest.mark.openmm

from fixtures_openmm_parity import (
    ala_dip_reference,
    get_prolix_per_term_energies,
)


# Phase 2 — Energy Parity Tests
# Gate: per-term |dE| < 0.05 kcal/mol


@pytest.fixture
def parity_bundle(ala_dip_reference):
    """Compute parity metrics for the reference system."""
    omm_energies = ala_dip_reference['energies']
    prolix_system = ala_dip_reference['prolix_system']
    displacement_fn = ala_dip_reference['displacement_fn']
    positions_ang = ala_dip_reference['positions_ang']

    prolix_energies = get_prolix_per_term_energies(
        prolix_system, displacement_fn, positions_ang
    )

    return {
        'omm': omm_energies,
        'prolix': prolix_energies,
        'positions': positions_ang,
    }


def test_bond_energy_parity(parity_bundle):
    """Phase 2a: Bond energy parity."""
    omm_e = parity_bundle['omm']['bonds']
    pro_e = parity_bundle['prolix']['bonds']
    delta = abs(omm_e - pro_e)

    print(f"\nBond energy parity:")
    print(f"  OpenMM:  {omm_e:.6f} kcal/mol")
    print(f"  Prolix:  {pro_e:.6f} kcal/mol")
    print(f"  Delta:   {delta:.6e} kcal/mol")

    assert delta < 0.05, (
        f"Bond energy mismatch: prolix={pro_e:.6f}, omm={omm_e:.6f}, "
        f"delta={delta:.6f} (gate: 0.05)"
    )


def test_angle_energy_parity(parity_bundle):
    """Phase 2b: Angle energy parity."""
    omm_e = parity_bundle['omm']['angles']
    pro_e = parity_bundle['prolix']['angles']
    delta = abs(omm_e - pro_e)

    print(f"\nAngle energy parity:")
    print(f"  OpenMM:  {omm_e:.6f} kcal/mol")
    print(f"  Prolix:  {pro_e:.6f} kcal/mol")
    print(f"  Delta:   {delta:.6e} kcal/mol")

    assert delta < 0.05, (
        f"Angle energy mismatch: prolix={pro_e:.6f}, omm={omm_e:.6f}, "
        f"delta={delta:.6f} (gate: 0.05)"
    )


def test_dihedral_energy_parity(parity_bundle):
    """Phase 2c: Dihedral energy parity (proper + improper)."""
    omm_e = parity_bundle['omm']['dihedrals']
    pro_e = parity_bundle['prolix']['dihedrals']
    delta = abs(omm_e - pro_e)

    print(f"\nDihedral energy parity:")
    print(f"  OpenMM:  {omm_e:.6f} kcal/mol")
    print(f"  Prolix:  {pro_e:.6f} kcal/mol")
    print(f"  Delta:   {delta:.6e} kcal/mol")

    assert delta < 0.05, (
        f"Dihedral energy mismatch: prolix={pro_e:.6f}, omm={omm_e:.6f}, "
        f"delta={delta:.6f} (gate: 0.05)"
    )


# Phase 3 — Force Parity Test
# Gate: RMS(|df|) < 0.01 kcal/mol/Å


def test_force_parity(ala_dip_reference):
    """Phase 3: Force parity via jax.grad.

    Computes forces on prolix side via jax.grad(total_bonded_energy)(positions).
    Compares against OpenMM reference forces at the same conformation.
    Includes FD sanity check (eps=1e-4) to verify jax.grad matches FD.
    """
    prolix_system = ala_dip_reference['prolix_system']
    displacement_fn = ala_dip_reference['displacement_fn']
    positions_ang = ala_dip_reference['positions_ang']
    omm_forces = ala_dip_reference['forces']

    # Convert to JAX array
    positions = jnp.array(positions_ang, dtype=jnp.float64)

    # Define total bonded energy function
    def total_bonded_energy(r):
        # Bonds
        e_bonds = 0.0
        if (prolix_system.bonds is not None and
            prolix_system.bonds.shape[0] > 0):
            from prolix.physics import bonded
            bond_fn = bonded.make_bond_energy_fn(
                displacement_fn, prolix_system.bonds
            )
            e_bonds = bond_fn(r, prolix_system.bond_params)

        # Angles
        e_angles = 0.0
        if (prolix_system.angles is not None and
            prolix_system.angles.shape[0] > 0):
            from prolix.physics import bonded
            angle_fn = bonded.make_angle_energy_fn(
                displacement_fn, prolix_system.angles
            )
            e_angles = angle_fn(r, prolix_system.angle_params)

        # Dihedrals
        e_dihedrals = 0.0
        if (prolix_system.dihedrals is not None and
            prolix_system.dihedrals.shape[0] > 0):
            from prolix.physics import bonded
            dih_fn = bonded.make_dihedral_energy_fn(
                displacement_fn, prolix_system.dihedrals
            )
            e_dihedrals = dih_fn(r, prolix_system.dihedral_params)

        return e_bonds + e_angles + e_dihedrals

    # Compute forces via jax.grad (note: F = -dE/dr)
    grad_fn = jax.grad(total_bonded_energy)
    prolix_forces = -grad_fn(positions)

    # FD sanity check: verify jax.grad matches finite difference (F = -dE/dr)
    eps = 1e-4
    fd_forces = np.zeros_like(positions_ang)
    for i in range(positions.shape[0]):
        for j in range(3):
            r_plus = positions.at[i, j].add(eps)
            r_minus = positions.at[i, j].add(-eps)
            e_plus = total_bonded_energy(r_plus)
            e_minus = total_bonded_energy(r_minus)
            # F = -dE/dr, so negate the gradient
            fd_forces[i, j] = -(float(e_plus) - float(e_minus)) / (2.0 * eps)

    # Check FD vs jax.grad agreement
    fd_grad_delta = np.linalg.norm(np.array(prolix_forces) - fd_forces)
    print(f"\nFD sanity check (eps={eps}):")
    print(f"  RMS(|grad_jax - grad_FD|) = {fd_grad_delta:.6e}")
    assert fd_grad_delta < 0.01, (
        f"jax.grad does not match FD: RMS={fd_grad_delta:.6e}"
    )

    # Main parity check: prolix vs OpenMM
    prolix_forces_np = np.array(prolix_forces)
    force_delta = prolix_forces_np - omm_forces
    rms_delta = float(np.sqrt(np.mean(force_delta**2)))

    print(f"\nForce parity (prolix vs OpenMM):")
    print(f"  Prolix forces shape: {prolix_forces_np.shape}")
    print(f"  OpenMM forces shape: {omm_forces.shape}")
    print(f"  RMS(|dF|) = {rms_delta:.6e} kcal/mol/Å")
    print(f"  Max |dF| = {np.max(np.abs(force_delta)):.6e} kcal/mol/Å")

    assert rms_delta < 0.01, (
        f"Force RMS mismatch: {rms_delta:.4e} kcal/mol/Å (gate: 0.01)"
    )


# Phase 4 — Field-Audit Smoke Test
# Gate: field_audit module imports; test runs without error


def test_field_audit_smoke(ala_dip_reference):
    """Phase 4: Field-audit module smoke test."""
    from prolix.physics.field_audit import audit_bonded_fields

    prolix_system = ala_dip_reference['prolix_system']
    displacement_fn = ala_dip_reference['displacement_fn']
    positions_ang = ala_dip_reference['positions_ang']

    # Run audit
    fields_accessed = audit_bonded_fields(
        prolix_system, displacement_fn, positions_ang
    )

    print(f"\nField audit (bonded path):")
    print(f"  Fields accessed: {sorted(fields_accessed)}")

    # Basic validation: should access at least bonds/angles/dihedrals
    assert 'bonds' in fields_accessed, "bonds field not accessed"
    assert 'angles' in fields_accessed, "angles field not accessed"
    assert 'dihedrals' in fields_accessed, "dihedrals field not accessed"
