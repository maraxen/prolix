"""Trainable bonded parameters for force-field fitting.

BondedParams is an Equinox Module holding per-molecule bonded parameters
(force constants and equilibrium geometry). All fields are trainable by default.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class BondedParams(eqx.Module):
    """Trainable bonded parameters (per-molecule, no padding).

    All arrays are JAX arrays (jnp), not numpy, so gradients can flow through them.
    Shapes are real-length (not padded), matching one molecule's topology.

    Attributes:
        k_bond: Float[Array, "N_bonds"] in kcal/mol/Å²
        r0: Float[Array, "N_bonds"] in Å
        k_theta: Float[Array, "N_angles"] in kcal/mol/rad²
        theta0_rad: Float[Array, "N_angles"] in radians (NOT degrees)
        k_phi: Float[Array, "N_torsions n_terms"] in kcal/mol
    """

    k_bond: Float[Array, "N_bonds"]
    r0: Float[Array, "N_bonds"]
    k_theta: Float[Array, "N_angles"]
    theta0_rad: Float[Array, "N_angles"]
    k_phi: Float[Array, "N_torsions n_terms"]

    @classmethod
    def from_init(cls, init: "BondedParams") -> "BondedParams":
        """Clone init for use as θ_init (frozen reference for harmonic prior).

        Args:
            init: BondedParams to clone.

        Returns:
            New BondedParams with copied arrays.
        """
        return jax.tree_util.tree_map(jnp.array, init)

    @property
    def n_bonds(self) -> int:
        return len(self.k_bond)

    @property
    def n_angles(self) -> int:
        return len(self.k_theta)

    @property
    def n_torsions(self) -> int:
        return self.k_phi.shape[0]

    @property
    def n_torsion_terms(self) -> int:
        """Number of Fourier terms per torsion."""
        if len(self.k_phi.shape) == 1:
            return 1
        return self.k_phi.shape[1]
