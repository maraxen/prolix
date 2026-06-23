"""Conformer mini-batch scheduler for stochastic gradient descent.

For each molecule, randomly sample 1 conformer per training step.
Uses JAX random key for reproducibility.
"""

from typing import NamedTuple

import jax
from jaxtyping import Array, Float


class ConformerBatch(NamedTuple):
    """A single conformer sampled for one molecule.

    Attributes:
        positions: [N_atoms, 3] atomic positions (Å).
        forces_ref: [N_atoms, 3] reference forces (kcal/mol/Å).
        energy_ref: scalar reference energy (Hartree).
    """

    positions: Float[Array, "N_atoms 3"]
    forces_ref: Float[Array, "N_atoms 3"]
    energy_ref: Float[Array, ""]


def sample_one_conformer(
    positions_all: Float[Array, "N_conf N_atoms 3"],
    forces_all: Float[Array, "N_conf N_atoms 3"],
    energies_all: Float[Array, "N_conf"],
    rng: jax.Array,
) -> ConformerBatch:
    """Sample a single random conformer from a molecule's ensemble.

    Args:
        positions_all: [N_conf, N_atoms, 3] conformer ensemble.
        forces_all: [N_conf, N_atoms, 3] reference forces.
        energies_all: [N_conf] reference energies.
        rng: JAX PRNG key.

    Returns:
        ConformerBatch with randomly selected conformer.
    """
    n_conf = positions_all.shape[0]
    idx = jax.random.randint(rng, shape=(), minval=0, maxval=n_conf)

    positions = positions_all[idx]
    forces_ref = forces_all[idx]
    energy_ref = energies_all[idx]

    return ConformerBatch(
        positions=positions,
        forces_ref=forces_ref,
        energy_ref=energy_ref,
    )
