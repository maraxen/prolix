import jax.numpy as jnp
import numpy as np
from prolix.physics.neighbor_list import ExclusionSpec, map_exclusions_to_dense_padded

def test_mapping():
    # Setup simple exclusion spec
    # Atom 0 bonded to 1. Atom 1 bonded to 2.
    # 1-2: (0, 1), (1, 2)
    # 1-3: (0, 2)
    idx_12_13 = jnp.array([[0, 1], [1, 2], [0, 2]])
    idx_14 = jnp.array([[0, 3]]) # 1-4

    spec = ExclusionSpec(
        idx_12_13=idx_12_13,
        idx_14=idx_14,
        scale_14_elec=0.833,
        scale_14_vdw=0.5,
        n_atoms=10,
        exception_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros((0,), dtype=jnp.float32),
        exception_epsilons=jnp.zeros((0,), dtype=jnp.float32),
        exception_chargeprods=jnp.zeros((0,), dtype=jnp.float32),
    )

    ei, sv, se = map_exclusions_to_dense_padded(spec, max_exclusions=5)

    print(f"Exclusions for atom 0: {ei[0]}")
    print(f"Scales VDW for atom 0: {sv[0]}")
    print(f"Scales ELEC for atom 0: {se[0]}")

if __name__ == "__main__":
    test_mapping()
