import jax
import jax.numpy as jnp
import time
import argparse
import pandas as pd
from prolix.physics import pbc, system
from prolix.physics.system import PhysicsSystem, make_energy_fn_pure

def setup_system(name, n_atoms):
    if name == "argon":
        positions = jax.random.uniform(jax.random.PRNGKey(0), (n_atoms, 3)) * 10.0
        box = jnp.array([20.0, 20.0, 20.0])
        sys_dict = {
            "charges": jnp.zeros(n_atoms),
            "sigmas": jnp.full(n_atoms, 3.405),
            "epsilons": jnp.full(n_atoms, 0.238),
            "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
            "bond_params": jnp.zeros((0, 2), dtype=jnp.float32),
            "angles": jnp.zeros((0, 3), dtype=jnp.int32),
            "angle_params": jnp.zeros((0, 2), dtype=jnp.float32),
            "proper_dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
            "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float32),
            "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
            "improper_params": jnp.zeros((0, 3), dtype=jnp.float32),
        }
    elif name == "tip3p":
        # Mock TIP3P-like system
        n_waters = n_atoms // 3
        positions = jax.random.uniform(jax.random.PRNGKey(0), (n_waters * 3, 3)) * 30.0
        box = jnp.array([30.0, 30.0, 30.0])
        sys_dict = {
            "charges": jnp.tile(jnp.array([-0.834, 0.417, 0.417]), n_waters),
            "sigmas": jnp.tile(jnp.array([3.1507, 1.0, 1.0]), n_waters),
            "epsilons": jnp.tile(jnp.array([0.1521, 0.0, 0.0]), n_waters),
            "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
            "bond_params": jnp.zeros((0, 2), dtype=jnp.float32),
            "angles": jnp.zeros((0, 3), dtype=jnp.int32),
            "angle_params": jnp.zeros((0, 2), dtype=jnp.float32),
            "proper_dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
            "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float32),
            "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
            "improper_params": jnp.zeros((0, 3), dtype=jnp.float32),
        }
    else:
        raise ValueError(f"Unknown system: {name}")

    displacement_fn, _ = pbc.create_periodic_space(box)
    phys_sys = PhysicsSystem.from_dict(sys_dict, positions, box)
    return positions, displacement_fn, phys_sys

def run_benchmark(name, n_atoms, tile_sizes):
    pos, disp, phys_sys = setup_system(name, n_atoms)
    results = []

    for ts in tile_sizes:
        print(f"Benchmarking {name} (N={n_atoms}, tile_size={ts})...")
        params, energy_fn = make_energy_fn_pure(disp, phys_sys, tile_size=ts)
        
        # JIT compile
        energy_jit = jax.jit(energy_fn)
        energy_jit(params, pos).block_until_ready()
        
        # Warmup
        for _ in range(5):
            energy_jit(params, pos).block_until_ready()
            
        # Timed run
        start = time.time()
        n_iters = 50
        for _ in range(n_iters):
            energy_jit(params, pos).block_until_ready()
        end = time.time()
        
        avg_ms = (end - start) * 1000 / n_iters
        results.append({"system": name, "n_atoms": n_atoms, "tile_size": ts, "ms_per_eval": avg_ms})
        print(f"  -> {avg_ms:.2f} ms/eval")
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    all_results = []
    # System configs: (name, n_atoms)
    configs = [
        ("argon", 128),
        ("argon", 1024),
        ("tip3p", 3072),  # ~1024 waters
    ]
    tile_sizes = [32, 64, 128, 256]

    for name, n_atoms in configs:
        all_results.extend(run_benchmark(name, n_atoms, tile_sizes))

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
