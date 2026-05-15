"""ShimMode: analytical force shim via custom_jvp for bonded terms.

ShimMode.ANALYTICAL registers a @jax.custom_jvp rule that computes bonded
forces analytically instead of tracing through AD. This speeds up non-
differentiable MD runs by avoiding unnecessary autodiff overhead for
bonded terms while keeping LJ/PME AD-traced (dense N² broadcast OOMs
at benchmark sizes like N=50k).

Example:
    from prolix.types.shims import ShimMode, energy_with_analytical_shim
    from prolix.physics.analytical_forces import bond_forces

    # Wrap energy fn with analytical bond forces
    shim_energy = energy_with_analytical_shim(base_energy, bond_forces)

    # Use in gradient computation — custom_jvp intercepts VJP
    grad_fn = jax.grad(shim_energy)
    result = grad_fn(bundle)
"""

from __future__ import annotations

import enum
from typing import Callable

import jax
import jax.numpy as jnp

from prolix.types.bundles import MolecularBundle


class ShimMode(enum.Enum):
    """Mode for bonded force computation.

    AUTOGRAD: Standard JAX autograd (slower but flexible).
    ANALYTICAL: @jax.custom_jvp shim with analytical forces (faster, non-diff).
    """

    AUTOGRAD = "autograd"
    ANALYTICAL = "analytical"


def energy_with_analytical_shim(
    energy_fn: Callable[[MolecularBundle], jnp.ndarray],
    analytical_forces_bonded: Callable[[MolecularBundle], jnp.ndarray],
) -> Callable[[MolecularBundle], jnp.ndarray]:
    """Register @custom_jvp on energy_fn using analytical bonded forces.

    This shim intercepts the vector-Jacobian product (VJP) to provide
    analytical forces instead of computing them via autodiff. The primal
    is unchanged; only the tangent computation uses the analytical path.

    Args:
        energy_fn: Primal energy function, takes MolecularBundle → scalar.
        analytical_forces_bonded: Returns -dE_bonded/dr, shape (N, 3).
            Should compute analytical forces for all bonded terms
            (bonds, angles, dihedrals, etc.).

    Returns:
        Wrapped energy function with @jax.custom_jvp registered.
        Primal behavior identical; gradient uses analytical forces.

    Note:
        LJ and PME forces are NOT included in the shim because their
        dense N² implementations would OOM at benchmark sizes (N=50k).
        Only bonded terms (which scale as O(bonds)) are shimmed.

    Example:
        >>> import jax
        >>> from jax_md import space
        >>> shim_energy = energy_with_analytical_shim(base_energy, bond_forces)
        >>> grad_fn = jax.grad(shim_energy)
        >>> result = grad_fn(bundle)  # Uses analytical bond forces
    """

    @jax.custom_jvp
    def energy(bundle: MolecularBundle) -> jnp.ndarray:
        """Primal: evaluate energy normally."""
        return energy_fn(bundle)

    @energy.defjvp
    def energy_jvp(primals, tangents):
        """Custom JVP: tangent uses analytical forces."""
        (bundle,) = primals
        (bundle_dot,) = tangents

        # Primal: standard energy evaluation
        e = energy_fn(bundle)

        # Tangent: dot product of analytical forces with bundle tangent
        f_bonded = analytical_forces_bonded(bundle)
        jvp_val = -jnp.sum(f_bonded * bundle_dot.positions)

        return e, jvp_val

    return energy
