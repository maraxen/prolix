"""Training state container for bonded-parameter fitting.

Holds trainable parameters, optimizer state, step counter, and RNG key.
Designed as a pure pytree for use with jax.lax.scan.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Int

from prolix.fitting.params import BondedParams


class TrainState(eqx.Module):
    """Immutable training state for bonded-parameter optimization.

    All fields are immutable (eqx.Module semantics). Use eqx.tree_at()
    or .replace() for updates.

    Attributes:
        params: BondedParams (trainable, updated per step).
        opt_state: optax.OptState for the optimizer (e.g., Adam).
        step: Int scalar, current training step (JAX array for scan compatibility).
        rng: JAX PRNG key (split per step if needed).
    """

    params: BondedParams
    opt_state: optax.OptState
    step: Int[Array, ""]
    rng: jax.Array

    @classmethod
    def init(
        cls,
        params_init: BondedParams,
        optimizer: optax.GradientTransformation,
        rng_seed: int = 42,
    ) -> "TrainState":
        """Initialize a TrainState from initial parameters.

        Args:
            params_init: BondedParams (starting point).
            optimizer: optax optimizer (e.g., optax.adam(1e-3)).
            rng_seed: Random seed for PRNG key.

        Returns:
            New TrainState with step=0, opt_state initialized.
        """
        opt_state = optimizer.init(params_init)
        rng = jax.random.PRNGKey(rng_seed)

        return cls(
            params=params_init,
            opt_state=opt_state,
            step=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )

    def split_rng(self) -> tuple["TrainState", jax.Array]:
        """Split the RNG key and return a new state + fresh key.

        Useful for stochastic steps (e.g., conformer sampling).

        Returns:
            (new_state_with_split_rng, fresh_key_for_use)
        """
        key1, key2 = jax.random.split(self.rng)
        new_state = eqx.tree_at(lambda s: s.rng, self, key1)
        return new_state, key2
