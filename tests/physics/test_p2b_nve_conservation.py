"""NVE energy conservation test for SETTLE integrator.

Test validates that settle_langevin with gamma=0 (free evolution, no thermostat)
conserves total energy (kinetic + potential) to within 1% over 1000 steps on a
64-particle harmonic oscillator system.
"""

import jax
import jax.numpy as jnp
import pytest
from jax_md import space
from prolix.physics import settle
from prolix.physics.kups_adapter import spring_constant_ev_per_angstrom_sq_to_kcal_mol
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL


def _harmonic_ke_kcal(momentum, mass):
    """Compute harmonic oscillator kinetic energy in kcal/mol.

    Args:
        momentum: shape (N, 3), velocities * mass
        mass: shape (N,), atomic masses

    Returns:
        Total kinetic energy in kcal/mol
    """
    # momentum is actually mv, so KE = (mv)^2 / (2*m) = p^2 / (2*m)
    # With broadcasting: (N, 3) / (N, 1) → per-atom KE, then sum
    ke = jnp.sum(momentum**2 / (2.0 * mass[:, None]))
    return ke


def _harmonic_pe_kcal(positions, k):
    """Compute harmonic oscillator potential energy in kcal/mol.

    Args:
        positions: shape (N, 3), particle positions in Angstroms
        k: spring constant in kcal/mol/Ų

    Returns:
        Total potential energy: 0.5 * k * sum(r^2)
    """
    return 0.5 * k * jnp.sum(positions**2)


@pytest.mark.slow
def test_nve_energy_conservation():
    """NVE energy conservation test.

    64-particle harmonic oscillator with zero thermostat (gamma=0).
    No SETTLE constraints (water_indices=None for unconstrained test).
    Energy drift over 1000 steps must be < 1% of initial energy.
    """
    jax.config.update("jax_enable_x64", True)

    # System parameters
    N = 64  # particles
    m_amu = 1.0  # atomic mass unit (amu)
    k_ev = 0.01  # spring constant in eV/Å²
    k_kcal = spring_constant_ev_per_angstrom_sq_to_kcal_mol(k_ev)  # kcal/mol/Å²

    # Time parameters
    dt_akma = 0.5 / AKMA_TIME_UNIT_FS  # 0.5 fs in AKMA units
    kT = 300.0 * BOLTZMANN_KCAL  # 300 K in kcal/mol
    gamma = 0.0  # No thermostat for NVE

    # If gamma=0.0 causes issues, fall back to near-zero
    if gamma == 0.0:
        # Use a very small gamma as effective zero
        # Note: Some integrators may have numerical issues with exactly 0
        gamma = 1e-10

    # Initial positions: random, small displacement from origin
    key = jax.random.PRNGKey(42)
    positions = jax.random.normal(key, (N, 3), dtype=jnp.float64) * 0.5  # 0.5 Å amplitude

    # Masses: uniform
    mass = jnp.full(N, m_amu, dtype=jnp.float64)

    # Displacement and shift functions (free space, no PBC)
    disp_fn, shift_fn = space.free()

    # Energy function: V(r) = 0.5 * k * sum(r^2)
    def energy_fn(pos):
        return _harmonic_pe_kcal(pos, k_kcal)

    # Initialize integrator (no water constraints)
    init_s, apply_s = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma,
        mass=mass,
        water_indices=None,
    )

    # Initialize state
    state = init_s(jax.random.key(0), positions, mass=mass)

    # Compute initial energy
    e_total_0 = energy_fn(positions) + _harmonic_ke_kcal(state.momentum, mass)

    # Run 1000 steps via scan: all on-device, single host transfer at end.
    def step_fn(state, _):
        state = apply_s(state)
        e = energy_fn(state.position) + _harmonic_ke_kcal(state.momentum, mass)
        return state, e

    _, e_traj = jax.lax.scan(step_fn, state, None, length=1000)
    e_total_trajectory = jnp.concatenate([jnp.array([e_total_0]), e_traj])

    # Compute max energy drift
    max_e_drift = jnp.max(jnp.abs(e_total_trajectory - e_total_0))
    fractional_drift = max_e_drift / jnp.abs(e_total_0)

    # Gate: max drift < 1% of initial energy
    assert fractional_drift < 0.01, (
        f"NVE energy conservation failed: "
        f"max drift = {max_e_drift:.6e} kcal/mol, "
        f"fractional drift = {fractional_drift:.4f} ({100*fractional_drift:.2f}%), "
        f"initial energy = {e_total_0:.6e} kcal/mol"
    )
