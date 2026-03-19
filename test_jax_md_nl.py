import jax
import jax.numpy as jnp
from jax_md import space, partition

def test_nl():
    N = 100
    key = jax.random.PRNGKey(0)
    R = jax.random.uniform(key, (N, 3), minval=-10.0, maxval=10.0)

    displacement_fn, shift_fn = space.free()

    print("--- Test 1: disable_cell_list=True, box=None ---")
    try:
        neighbor_fn = partition.neighbor_list(
            displacement_fn,
            box=None,
            r_cutoff=5.0,
            disable_cell_list=True,
            format=partition.Dense,
            capacity_multiplier=1.25,
            dr_threshold=1.0, 
        )
        nbrs = neighbor_fn.allocate(R)
        print("Success! Shape:", nbrs.idx.shape)
    except Exception as e:
        print("Error:", e)

    print("\n--- Test 2: bounding box ---")
    try:
        box_size = 50.0  # Large enough bounding box
        neighbor_fn = partition.neighbor_list(
            displacement_fn,
            box=box_size,
            r_cutoff=5.0,
            disable_cell_list=False,
            format=partition.Dense,
            capacity_multiplier=1.25,
            dr_threshold=1.0, 
        )
        nbrs = neighbor_fn.allocate(R)
        print("Success! Shape:", nbrs.idx.shape)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_nl()
