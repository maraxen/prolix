
from collections.abc import Callable

import jax

from prolix.typing import EnergyParams


def wrap_energy_fn_pure(
    energy_fn: Callable[..., jax.Array]
) -> Callable[[EnergyParams, jax.Array, jax.Array | None], jax.Array]:
    """
    Creates a pure energy function from a closure-based energy function.

    Args:
        energy_fn: A closure-based function with signature (positions, box, **kwargs).
                   Expected to use values from its closure.

    Returns:
        A function with signature (params: EnergyParams, positions: jax.Array, box: jax.Array | None).
    """

    def pure_energy_fn(params: EnergyParams, positions: jax.Array, box: jax.Array | None) -> jax.Array:
        # Pass params as kwargs to the energy_fn, assuming energy_fn can consume them.
        # This structure allows energy_fn to be pure by not capturing external state.
        return energy_fn(positions, box, **params.params)

    return pure_energy_fn
