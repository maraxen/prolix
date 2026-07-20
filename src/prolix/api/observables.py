"""Observable protocol and Trajectory module for simulation output.

Observable: @runtime_checkable Protocol defining compute(state) -> Array interface.
Trajectory: eqx.Module storing positions and observable values for a simulation run.
Temperature: Example Observable computing kinetic temperature from state.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


@runtime_checkable
class Observable(Protocol):
    """Protocol for observable quantities computed from integrator state.

    An Observable defines a method compute(state) that extracts or computes
    a scalar or array quantity from an IntegratorState. Implementations
    should be pure functions with no side effects.

    Example:
        class Temperature(eqx.Module):
            dof: int = eqx.field(static=True)

            def compute(self, state):
                # state is LangevinState or similar
                # Return scalar temperature or array of temperatures
                return temperature_array
    """

    def compute(self, state: Any) -> Array:
        """Compute observable from integrator state.

        Args:
            state: IntegratorState (LangevinState, CSVRState, NHCState, etc.)

        Returns:
            Float Array of the computed observable (any shape).
        """
        ...


class Trajectory(eqx.Module):
    """Trajectory output module: positions + observables over time.

    Stores the result of a simulation run: sampled positions at each step
    and computed values for multiple observables.

    Attributes:
        positions: Shape (steps, atoms, 3) array of atomic positions
        observable_values: Dict mapping observable names to value arrays
        n_steps: Number of trajectory steps (static, not traced)
    """

    positions: Float[Array, "steps atoms 3"]
    observable_values: dict[str, Array]
    n_steps: int = eqx.field(static=True)


class Temperature(eqx.Module):
    """Observable computing kinetic temperature from integrator state.

    T = (2 * KE) / (dof * k_B)

    where KE is kinetic energy, dof is degrees of freedom, and k_B is Boltzmann.

    Attributes:
        dof: Degrees of freedom for temperature calculation
        atom_mask: Optional (N,) bool, True for real (non-padding) atoms.
            Padding atoms carry unit mass, not zero (see
            `masses_with_prefix`), and draw genuine nonzero momentum over
            the course of integration, so they must be excluded from the sum
            on the batched/stacked EnsemblePlan dispatch path or the result
            is polluted by however many padding atoms the shape bucket
            carries (debt 841). `None` (default) preserves prior behavior
            for callers whose state already contains only real atoms (e.g.
            the single-bundle path).
    """

    dof: int = eqx.field(static=True)
    atom_mask: Bool[Array, "N"] | None = None

    def compute(self, state: Any) -> Float[Array, ""]:
        """Compute temperature from state momenta.

        Uses the kinetic temperature formula: T = (2 * KE) / (dof * k_B)
        where KE = sum(p_i^2 / (2*m_i)) is the kinetic energy.

        Args:
            state: IntegratorState with momentum and mass fields
                   (e.g., from prolix.typing.IntegratorState)

        Returns:
            Scalar temperature in Kelvin
        """
        from prolix.simulate import BOLTZMANN_KCAL

        # Ensure masses are available
        if not hasattr(state, "mass") or state.mass is None:
            return jnp.array(float("nan"))

        # Extract momenta and masses
        momentum = state.momentum  # Shape (N, 3) or (B, N, 3)
        mass = state.mass  # Shape (N,) or (N, 1)

        # Reshape mass if needed to broadcast with momentum
        if mass.ndim == 1:
            mass_expanded = mass[:, None]  # (N, 1)
        else:
            mass_expanded = mass

        # Compute kinetic energy: KE = sum(p^2 / (2*m))
        # Handle both batched and unbatched cases
        ke_per_atom = jnp.sum(momentum**2 / (2.0 * mass_expanded), axis=-1)  # Remove coord axis
        if self.atom_mask is not None:
            ke_per_atom = jnp.where(self.atom_mask, ke_per_atom, 0.0)
        total_ke = jnp.sum(ke_per_atom)

        # Compute temperature: T = (2 * KE) / (dof * k_B)
        temperature = (2.0 * total_ke) / (self.dof * BOLTZMANN_KCAL)

        return temperature


class Energy(eqx.Module):
    """Observable computing total potential energy from integrator state.

    The energy function is evaluated at the current state positions and a
    captured MolecularBundle. This observable bridges the gap between the
    state (positions) and the potential energy function.

    Attributes:
        energy_fn: Callable taking (positions, bundle) -> scalar energy.
            The function should accept positions (N, 3) and a MolecularBundle,
            returning a scalar potential energy value.
        bundle: MolecularBundle or similar system description (captured at
            construction). Stored as a pytree leaf (not static) to allow
            batching and pytree operations.
    """

    energy_fn: Callable[..., Float[Array, ""]] = eqx.field(static=True)
    bundle: Any  # MolecularBundle or system descriptor

    def compute(self, state: Any) -> Float[Array, ""]:
        """Compute total potential energy at current state.

        Args:
            state: IntegratorState with positions attribute
                (e.g., LangevinState, CSVRState, NHCState)

        Returns:
            Scalar potential energy value
        """
        positions = (
            state.positions
            if hasattr(state, "positions")
            else state.position
        )
        return self.energy_fn(positions, self.bundle)


class KineticEnergy(eqx.Module):
    """Observable computing total kinetic energy from integrator state.

    KE = sum_i p_i^2 / (2 * m_i)

    The kinetic energy is computed from momentum and mass directly,
    without the Boltzmann constant (unlike Temperature which scales by dof).

    Attributes:
        atom_mask: Optional (N,) bool, True for real (non-padding) atoms.
            See `Temperature.atom_mask` for why this is needed on the
            batched/stacked EnsemblePlan dispatch path (debt 841).
    """

    atom_mask: Bool[Array, "N"] | None = None

    def compute(self, state: Any) -> Float[Array, ""]:
        """Compute total kinetic energy from state.

        Args:
            state: IntegratorState with momentum and mass attributes
                (e.g., LangevinState, CSVRState, NHCState)

        Returns:
            Scalar kinetic energy in kcal/mol
        """
        momentum = state.momentum  # Shape (N, 3)
        mass = state.mass          # Shape (N,) or (N, 1)

        # Expand mass to broadcast with momentum coords
        if mass.ndim == 1:
            mass_expanded = mass[:, None]
        else:
            mass_expanded = mass

        # Compute KE per atom: sum(p^2 / (2*m)) along coordinate axis
        ke_per_atom = jnp.sum(momentum**2 / (2.0 * mass_expanded), axis=-1)
        if self.atom_mask is not None:
            ke_per_atom = jnp.where(self.atom_mask, ke_per_atom, 0.0)

        # Sum over all (real) atoms
        return jnp.sum(ke_per_atom)


class RMSD(eqx.Module):
    """Observable computing RMSD vs a stored reference structure.

    RMSD = sqrt(mean over atoms of ||r_i - ref_i||^2)

    Attributes:
        reference: Shape (atoms, 3) reference positions
    """

    reference: Float[Array, "atoms 3"]

    def compute(self, state: Any) -> Float[Array, ""]:
        """Compute RMSD from reference structure.

        Args:
            state: IntegratorState with positions attribute
                (e.g., LangevinState, CSVRState, NHCState)

        Returns:
            Scalar RMSD in Angstroms (same units as positions)
        """
        positions = (
            state.positions
            if hasattr(state, "positions")
            else state.position
        )

        # Compute displacement from reference
        diff = positions - self.reference

        # RMSD = sqrt(mean of squared displacements)
        squared_distances = jnp.sum(diff**2, axis=-1)  # Sum over coordinates
        mean_squared = jnp.mean(squared_distances)     # Mean over atoms

        return jnp.sqrt(mean_squared)


class Pressure(eqx.Module):
    """Observable computing instantaneous pressure (ideal-gas approximation).

    P = N * k_B * T / V  (ideal gas)

    Note: virial contribution deferred to v1.1 (requires per-pair force decomposition).

    Attributes:
        n_atoms: Number of atoms in system (static)
        volume_angstrom3: System volume in Angstroms^3 (static)
        atom_mask: Optional (N,) bool, True for real (non-padding) atoms.
            See `Temperature.atom_mask` for why this is needed on the
            batched/stacked EnsemblePlan dispatch path (debt 841).
    """

    n_atoms: int = eqx.field(static=True)
    volume_angstrom3: float = eqx.field(static=True)
    atom_mask: Bool[Array, "N"] | None = None

    def compute(self, state: Any) -> Float[Array, ""]:
        """Compute pressure from state using ideal gas approximation.

        Uses P = 2*KE / (3*V) where KE is kinetic energy and V is volume.

        Args:
            state: IntegratorState with momentum and mass attributes
                (e.g., LangevinState, CSVRState, NHCState)

        Returns:
            Scalar pressure in bar
        """

        momentum = state.momentum  # Shape (N, 3)
        mass = state.mass          # Shape (N,) or (N, 1)

        # Expand mass to broadcast with momentum coords
        if mass.ndim == 1:
            mass_expanded = mass[:, None]
        else:
            mass_expanded = mass

        # Compute kinetic energy
        ke_per_atom = jnp.sum(momentum**2 / (2.0 * mass_expanded), axis=-1)
        if self.atom_mask is not None:
            ke_per_atom = jnp.where(self.atom_mask, ke_per_atom, 0.0)
        total_ke = jnp.sum(ke_per_atom)

        # Ideal gas: T = 2*KE / (3*N*k_B), so P = N*k_B*T/V = 2*KE / (3*V)
        # Volume in Angstrom^3; BOLTZMANN_KCAL in kcal/mol/K
        # Pressure in kcal/mol/Angstrom^3; convert to bar
        # 1 kcal/mol/A^3 = 68568 bar (derived from unit conversion)
        KCAL_MOL_PER_A3_TO_BAR = 68568.0

        pressure_kcal = (2.0 * total_ke) / (3.0 * self.volume_angstrom3)
        return pressure_kcal * KCAL_MOL_PER_A3_TO_BAR
