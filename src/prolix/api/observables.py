"""Observable protocol and Trajectory module for simulation output.

Observable: @runtime_checkable Protocol defining compute(state) -> Array interface.
Trajectory: eqx.Module storing positions and observable values for a simulation run.
Temperature: Example Observable computing kinetic temperature from state.
"""

from typing import Protocol, runtime_checkable
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


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

    def compute(self, state) -> Array:
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
    observable_values: dict  # name -> Array
    n_steps: int = eqx.field(static=True)


class Temperature(eqx.Module):
    """Observable computing kinetic temperature from integrator state.

    T = (2 * KE) / (dof * k_B)

    where KE is kinetic energy, dof is degrees of freedom, and k_B is Boltzmann.

    Attributes:
        dof: Degrees of freedom for temperature calculation
    """

    dof: int = eqx.field(static=True)

    def compute(self, state) -> Float[Array, ""]:
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
        if not hasattr(state, 'mass') or state.mass is None:
            return jnp.nan

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

    energy_fn: any = eqx.field(static=True)  # Callables are static (not traced)
    bundle: any  # MolecularBundle or system descriptor

    def compute(self, state) -> Float[Array, ""]:
        """Compute total potential energy at current state.

        Args:
            state: IntegratorState with positions attribute
                (e.g., LangevinState, CSVRState, NHCState)

        Returns:
            Scalar potential energy value
        """
        return self.energy_fn(state.positions, self.bundle)
