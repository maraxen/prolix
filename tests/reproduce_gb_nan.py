import jax
import jax.numpy as jnp
import numpy as np

# Force NaN to dump immediately if XLA config allows it
jax.config.update("jax_debug_nans", True)

from proxide.physics import constants
from prolix.physics.generalized_born import (
    compute_gb_energy_neighbor_list,
    compute_ace_nonpolar_energy
)

def main():
    print("Setting up minimal padding system...")
    # 2 real atoms, 2 padded atoms
    N = 4
    K = 4
    
    # Real atoms at distance 2.0
    # Padded atoms at 0,0,0
    pos = jnp.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],  # pad
        [0.0, 0.0, 0.0],  # pad
    ])
    
    # Radii: real=1.5, pad=1e-6
    radii = jnp.array([1.5, 1.5, 1e-6, 1e-6])
    
    # Charges: real=0.5, pad=0.0
    charges = jnp.array([0.5, -0.5, 0.0, 0.0])
    
    # Atom mask: 1 for real, 0 for pad
    mask = jnp.array([1.0, 1.0, 0.0, 0.0])
    
    # Neighbor list (include all for testing)
    neighbor_idx = jnp.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])
    
    # But mask out self and padded
    # Usually we use sentinel N for padding. Let's use N for padded interactions
    # 0 interacts with 1
    # 1 interacts with 0
    # 2 interacts with nothing
    # 3 interacts with nothing
    neighbor_idx = jnp.array([
        [1, 4, 4, 4],
        [0, 4, 4, 4],
        [4, 4, 4, 4],
        [4, 4, 4, 4],
    ])

    def energy_fn(positions):
        e_gb, born_radii = compute_gb_energy_neighbor_list(
            positions=positions,
            charges=charges,
            radii=radii,
            neighbor_idx=neighbor_idx,
            dielectric_offset=0.09
        )
        e_np = compute_ace_nonpolar_energy(radii, born_radii)
        e_np = jnp.sum(e_np * mask)
        return e_gb + e_np

    print("Computing energy...")
    e = energy_fn(pos)
    print(f"Energy: {e}")
    
    print("Computing gradient...")
    grad_fn = jax.grad(energy_fn)
    g = grad_fn(pos)
    
    print("Gradient:")
    print(g)
    
    if jnp.isnan(g).any():
        print("FAIL: Gradient contains NaN!")
    else:
        print("PASS: Gradient is clean.")

if __name__ == "__main__":
    main()
