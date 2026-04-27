"""Sprint 5: Cross-validation of prolix thermostats against kUPs.

This test suite runs identical harmonic oscillator systems in both prolix (AKMA units)
and kUPs (eV/Å/amu units) to determine whether temperature biases are VV discretization
artifacts (present in both engines) or prolix-specific bugs.

Key question: Does prolix's +8K CSVR temperature bias at dt>=1fs appear in kUPs too?
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Skip test file if kUPs not available
kups = pytest.importorskip(
    "kups",
    reason="kUPs not installed; run: uv pip install /tmp/kups",
)

from kups.core.constants import BOLTZMANN_CONSTANT as KUPS_KB, FEMTO_SECOND
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import HasLensFields, LensField, lens
from kups.core.propagator import CachePropagator
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, jit
from kups.md.integrators import (
    make_baoab_langevin_step,
    make_csvr_step,
    euclidean_flow,
)
from kups.md.observables import particle_kinetic_energy

from jax_md import space
from prolix.physics import settle
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

# ============================================================================
# Unit Conversion Constants
# ============================================================================

# 1 eV = 23.0605 kcal/mol (from CODATA 2018)
EV_TO_KCAL = 23.060549

# prolix AKMA time unit in femtoseconds (imported from prolix.simulate)

# Validate conversion consistency
assert abs(AKMA_TIME_UNIT_FS - 48.88821291839) < 1e-6, "AKMA time unit mismatch"
assert abs(EV_TO_KCAL - 23.060549) < 1e-6, "eV→kcal/mol conversion mismatch"

# ============================================================================
# Test System Parameters
# ============================================================================

N_PARTICLES = 64  # Harmonic oscillator particles
K_EV = 0.01  # Spring constant in eV/Å² (kUPs)
K_KCAL = K_EV * EV_TO_KCAL  # Spring constant in kcal/mol/Å² (prolix)
M_AMU = 1.0  # Mass in amu (both engines)
T_TARGET_K = 300.0  # Target temperature in Kelvin
KT_EV = T_TARGET_K * KUPS_KB  # kUPs thermal energy in eV
KT_KCAL = T_TARGET_K * BOLTZMANN_KCAL  # prolix thermal energy in kcal/mol
GAMMA_PS = 10.0  # Langevin friction in ps⁻¹; chosen so τ≈100 fs to match kUPs effective τ (γ=FEMTO_SECOND kUPs-time⁻¹)
TAU_PS = 0.1  # CSVR time constant in ps
N_EQUIL_STEPS = 40_000  # Equilibration steps (at dt=0.5fs: 40k*0.5fs = 20ps)
N_SAMPLE_STEPS = 60_000  # Production steps (at dt=0.5fs: 60k*0.5fs = 30ps)
DOF = 3 * N_PARTICLES  # Unconstrained DOF (no COM removal)

# ============================================================================
# kUPs Data Classes (Copied from /tmp/kups/test/md/test_integrators.py)
# ============================================================================


@dataclass
class ParticleData:
    positions: jnp.ndarray
    momenta: jnp.ndarray
    forces: jnp.ndarray
    masses: jnp.ndarray
    system: Index[SystemId]
    position_gradients: jnp.ndarray


@dataclass
class SystemData:
    time_step: jnp.ndarray
    temperature: jnp.ndarray
    friction_coefficient: jnp.ndarray
    thermostat_time_constant: jnp.ndarray


@dataclass
class SimpleState(HasLensFields):
    particles: LensField[Table[ParticleId, ParticleData]]
    systems: LensField[Table[SystemId, SystemData]]


# ============================================================================
# kUPs Helper Functions (Copied Verbatim)
# ============================================================================


def compute_temperature(state, dof):
    """Compute instantaneous temperature from state."""
    ke = jnp.sum(
        particle_kinetic_energy(
            state.particles.data.momenta, state.particles.data.masses
        )
    )
    return 2.0 * ke / dof


def get_systems(s: SimpleState) -> Table[SystemId, SystemData]:
    """Extract Table SystemData from state."""
    return s.systems


def run_simulation(integrator, state, key, n_equil, n_sample, extract_fn):
    """Run equilibration + sampling with jax.lax.scan."""

    def step_fn(carry, _):
        key, s = carry
        key, subkey = jax.random.split(key)
        s = integrator(subkey, s)
        return (key, s), extract_fn(s)

    @jit
    def run(key, state):
        (key, state), _ = jax.lax.scan(step_fn, (key, state), None, length=n_equil)
        (_, state), samples = jax.lax.scan(
            step_fn, (key, state), None, length=n_sample
        )
        return state, samples

    return run(key, state)


def create_harmonic_system(
    n_particles=10, k=1.0, m=1.0, kT=1.0, dt=0.01, tau=0.1, gamma=1.0, key=None
):
    """Create harmonic oscillator system for testing (kUPs).

    Copied verbatim from /tmp/kups/test/md/test_integrators.py.
    """
    if key is None:
        key = jax.random.key(42)
    key1, key2 = jax.random.split(key)

    positions = jax.random.normal(key1, (n_particles, 3)) * 0.1
    momenta = jax.random.normal(key2, (n_particles, 3)) * jnp.sqrt(m * kT)
    forces = -k * positions
    masses = jnp.full((n_particles,), m)

    system_index = Index.new([SystemId(0)] * n_particles)
    particles = Table.arange(
        ParticleData(
            positions=positions,
            momenta=momenta,
            forces=forces,
            masses=masses,
            system=system_index,
            position_gradients=-forces,
        ),
        label=ParticleId,
    )

    systems = Table.arange(
        SystemData(
            time_step=jnp.array([dt]),
            temperature=jnp.array([kT / KUPS_KB]),
            friction_coefficient=jnp.array([gamma]),
            thermostat_time_constant=jnp.array([tau]),
        ),
        label=SystemId,
    )

    state = SimpleState(particles=particles, systems=systems)

    def compute_forces_fn(s):
        forces = -k * s.particles.data.positions
        return Table(
            s.particles.keys,
            ParticleData(
                positions=s.particles.data.positions,
                momenta=s.particles.data.momenta,
                forces=forces,
                masses=s.particles.data.masses,
                system=s.particles.data.system,
                position_gradients=-forces,
            ),
        )

    derivative_computation = CachePropagator(
        lambda key, state: compute_forces_fn(state).data.forces,
        lens(lambda s: s.particles, cls=SimpleState)
        .focus(lambda p: p.data.forces)
        .set,
    )

    return state, derivative_computation, compute_forces_fn


# ============================================================================
# Prolix Helper Functions
# ============================================================================


def _run_prolix_harmonic_baoab(
    n_particles, k_kcal, m_amu, kT_kcal, dt_fs, gamma_ps, n_equil, n_sample, seed
):
    """Run prolix settle_langevin (water_indices=None) on harmonic oscillator.

    Falls back to jax_md BAOAB when no water constraints are specified.
    """
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(seed)

    # Convert units: fs → AKMA; ps⁻¹ → AKMA⁻¹
    # 1 AKMA = AKMA_TIME_UNIT_FS fs = AKMA_TIME_UNIT_FS/1000 ps
    # => gamma [AKMA^-1] = gamma_ps * (AKMA_TIME_UNIT_FS / 1000)
    dt_akma = dt_fs / AKMA_TIME_UNIT_FS
    gamma_akma = gamma_ps * AKMA_TIME_UNIT_FS / 1000  # ps⁻¹ → AKMA⁻¹ (~0.0489 at gamma=1)

    # Harmonic energy: E = 0.5 * k * sum(r_i^2)
    def energy_fn(R, **kwargs):
        return 0.5 * k_kcal * jnp.sum(R**2)

    # Free space (no PBC)
    displacement_fn, shift_fn = space.free()

    mass = jnp.full(n_particles, m_amu)

    # settle_langevin with water_indices=None → delegates to jax_md.simulate.nvt_langevin
    init_fn, apply_fn = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT_kcal,
        gamma=gamma_akma,
        mass=mass,
        water_indices=None,
    )

    # Initialize: positions ~ N(0, 0.1Å), momenta ~ N(0, sqrt(m*kT))
    key, k1, k2 = jax.random.split(key, 3)
    init_pos = jax.random.normal(k1, (n_particles, 3)) * 0.1
    state = init_fn(k2, init_pos, mass=mass)

    apply_j = jax.jit(apply_fn)

    # Equilibration
    for _ in range(n_equil):
        state = apply_j(state)

    # Production: collect temperature each step
    temps = []
    for _ in range(n_sample):
        state = apply_j(state)
        # KE = sum(p_i^2 / (2*m_i)); T = 2*KE / (dof * kB)
        ke = float(jnp.sum(state.momentum**2 / (2 * mass[:, None])))
        temps.append(2.0 * ke / (DOF * BOLTZMANN_KCAL))

    return np.mean(temps)


def _run_prolix_harmonic_csvr(
    n_particles, k_kcal, m_amu, kT_kcal, dt_fs, tau_ps, n_equil, n_sample, seed
):
    """Run prolix settle_csvr (water_indices=None) on harmonic oscillator.

    Falls back to jax_md CSVR when no water constraints are specified.
    """
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(seed)

    # Convert units
    dt_akma = dt_fs / AKMA_TIME_UNIT_FS
    tau_akma = tau_ps * 1000 / AKMA_TIME_UNIT_FS  # Convert ps → AKMA

    # Harmonic energy
    def energy_fn(R, **kwargs):
        return 0.5 * k_kcal * jnp.sum(R**2)

    # Free space
    displacement_fn, shift_fn = space.free()

    mass = jnp.full(n_particles, m_amu)

    # settle_csvr with water_indices=None → delegates to jax_md CSVR
    init_fn, apply_fn = settle.settle_csvr(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT_kcal,
        tau=tau_akma,
        mass=mass,
        water_indices=None,
        n_constraint_pairs=0,
        remove_com=False,
    )

    # Initialize
    key, k1, k2 = jax.random.split(key, 3)
    init_pos = jax.random.normal(k1, (n_particles, 3)) * 0.1
    state = init_fn(k2, init_pos, mass=mass)

    apply_j = jax.jit(apply_fn)

    # Equilibration
    for _ in range(n_equil):
        state = apply_j(state)

    # Production: collect temperature
    temps = []
    for _ in range(n_sample):
        state = apply_j(state)
        ke = float(jnp.sum(state.momentum**2 / (2 * mass[:, None])))
        temps.append(2.0 * ke / (DOF * BOLTZMANN_KCAL))

    return np.mean(temps)


# ============================================================================
# Cross-Validation Tests
# ============================================================================


@pytest.mark.slow
@pytest.mark.dynamics
@pytest.mark.parametrize(
    "integrator_name,dt_fs,T_tolerance_K",
    [
        ("BAOAB", 0.5, 2.0),  # Both engines agree within 2K at small dt
        ("BAOAB", 1.0, 2.0),  # Same at larger dt
        ("CSVR", 0.5, 2.0),  # CSVR at small dt: no bias expected
        ("CSVR", 1.0, 10.0),  # CSVR at dt>=1fs: allow ±8K bias; test consistency
    ],
)
def test_kups_prolix_temperature_crossval(integrator_name, dt_fs, T_tolerance_K):
    """Cross-validate prolix thermostats against kUPs on harmonic oscillator.

    Determines whether prolix's +8K CSVR temperature bias at dt>=1fs is a
    velocity-Verlet discretization artifact (present in kUPs too) or prolix-specific.

    For CSVR at dt=1.0fs, if both engines show similar bias (e.g., both +7K to +9K),
    the bias is a VV artifact; if prolix is +8K and kUPs is ~300K, it's prolix-specific.
    """
    jax.config.update("jax_enable_x64", True)

    # Convert dt_fs to kUPs time units
    dt_kups = dt_fs * FEMTO_SECOND

    # Create kUPs system and run
    if integrator_name == "BAOAB":
        state, deriv, _ = create_harmonic_system(
            n_particles=N_PARTICLES,
            k=K_EV,
            m=M_AMU,
            kT=KT_EV,
            dt=dt_kups,
            gamma=GAMMA_PS * FEMTO_SECOND,
            tau=TAU_PS * FEMTO_SECOND,
            key=jax.random.key(42),
        )
        integrator = make_baoab_langevin_step(
            particles=SimpleState.particles,
            systems=SimpleState.systems.get,
            derivative_computation=deriv,
            flow=euclidean_flow,
        )
    elif integrator_name == "CSVR":
        state, deriv, _ = create_harmonic_system(
            n_particles=N_PARTICLES,
            k=K_EV,
            m=M_AMU,
            kT=KT_EV,
            dt=dt_kups,
            tau=TAU_PS * FEMTO_SECOND,
            gamma=GAMMA_PS * FEMTO_SECOND,
            key=jax.random.key(42),
        )
        integrator = make_csvr_step(
            particles=SimpleState.particles,
            systems=get_systems,
            derivative_computation=deriv,
            flow=euclidean_flow,
        )
    else:
        raise ValueError(f"Unknown integrator: {integrator_name}")

    # Run kUPs simulation
    final_state, temps_kups = run_simulation(
        integrator,
        state,
        jax.random.key(42),
        n_equil=N_EQUIL_STEPS,
        n_sample=N_SAMPLE_STEPS,
        extract_fn=lambda s: compute_temperature(s, DOF),
    )

    # Convert kUPs temperature (eV) to Kelvin
    T_kups_K = float(np.mean(temps_kups)) / KUPS_KB

    # Run prolix equivalent
    if integrator_name == "BAOAB":
        T_prolix_K = _run_prolix_harmonic_baoab(
            n_particles=N_PARTICLES,
            k_kcal=K_KCAL,
            m_amu=M_AMU,
            kT_kcal=KT_KCAL,
            dt_fs=dt_fs,
            gamma_ps=GAMMA_PS,
            n_equil=N_EQUIL_STEPS,
            n_sample=N_SAMPLE_STEPS,
            seed=42,
        )
    elif integrator_name == "CSVR":
        T_prolix_K = _run_prolix_harmonic_csvr(
            n_particles=N_PARTICLES,
            k_kcal=K_KCAL,
            m_amu=M_AMU,
            kT_kcal=KT_KCAL,
            dt_fs=dt_fs,
            tau_ps=TAU_PS,
            n_equil=N_EQUIL_STEPS,
            n_sample=N_SAMPLE_STEPS,
            seed=42,
        )
    else:
        raise ValueError(f"Unknown integrator for prolix runner: {integrator_name}")

    # Check cross-engine agreement
    T_diff = T_prolix_K - T_kups_K
    assert (
        abs(T_diff) < T_tolerance_K
    ), f"[CROSSVAL {integrator_name} dt={dt_fs}fs] T_prolix={T_prolix_K:.1f}K != T_kUPs={T_kups_K:.1f}K (diff={T_diff:+.1f}K, tolerance={T_tolerance_K}K)"

    # Emit diagnostic message
    print(
        f"\n[CROSSVAL {integrator_name} dt={dt_fs}fs] "
        f"T_prolix={T_prolix_K:.1f}K  T_kUPs={T_kups_K:.1f}K  "
        f"diff={T_diff:+.1f}K (tolerance={T_tolerance_K}K)"
    )

    # For CSVR at dt=1.0fs, assert that any bias is consistent across engines.
    # The VV-artifact hypothesis: if the bias is discretization-driven, both engines
    # must show similar bias directions and magnitudes. If prolix diverges by >5K from
    # kUPs bias, the bias is prolix-specific (a bug), not a shared VV artifact.
    if integrator_name == "CSVR" and dt_fs >= 1.0:
        prolix_bias = T_prolix_K - T_TARGET_K
        kups_bias = T_kups_K - T_TARGET_K
        print(
            f"[CSVR BIAS CHECK] T_prolix-{T_TARGET_K}K={prolix_bias:+.1f}K, "
            f"T_kUPs-{T_TARGET_K}K={kups_bias:+.1f}K"
        )
        assert abs(prolix_bias - kups_bias) < 5.0, (
            f"CSVR dt={dt_fs}fs bias mismatch: prolix={prolix_bias:+.1f}K vs kUPs={kups_bias:+.1f}K — "
            f"diff={prolix_bias - kups_bias:+.1f}K exceeds 5K; bias may be prolix-specific, not a VV artifact"
        )
