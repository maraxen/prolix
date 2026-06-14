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

        Placeholder implementation returning NaN. Real implementation
        will be fixed in Sprint 38 once state type is finalized.

        Args:
            state: IntegratorState (type finalized in Sprint 38)

        Returns:
            Scalar temperature in Kelvin
        """
        # Placeholder: will be implemented when state type is fixed in Sprint 38
        return jnp.nan
