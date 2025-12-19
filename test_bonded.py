
import jax.numpy as jnp
from jax_md import space

from prolix.physics import bonded


def test_bonded():
    print("Testing bonded module...")
    displacement_fn, shift_fn = space.free()

    indices = jnp.array([[0, 1, 2, 3]])
    params = jnp.array([[1, 0.0, 10.0]]) # periodicity, phase, k

    try:
        fn = bonded.make_dihedral_energy_fn(displacement_fn, indices, params)
        print("Function created.")

        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0]
        ])

        e = fn(positions)
        print(f"Energy: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_bonded()
