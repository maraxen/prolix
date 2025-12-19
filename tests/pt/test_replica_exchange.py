"""Tests for Parallel Tempering."""
import jax.numpy as jnp
import pytest
from proxide.md.jax_md_bridge import SystemParams

from prolix.pt import replica_exchange, temperature


def test_temperature_ladder():
    temps = temperature.generate_temperature_ladder(4, 300, 400, geometric=True)
    assert len(temps) == 4
    assert jnp.allclose(temps[0], 300.0)
    assert jnp.allclose(temps[-1], 400.0)

    # Check geometric property
    ratios = temps[1:] / temps[:-1]
    assert jnp.allclose(ratios, ratios[0])

def test_replica_exchange_run():
    # Mock system
    # 2 particles, harmonic bond
    # R: (2, 3)

    system_params = SystemParams(
        masses=1.0, # Simple mass
        charges=None,
        atom_names=["A", "B"],
        bonds=[[0, 1]],
        bond_params={"k": jnp.array([100.0]), "length": jnp.array([1.0])}
    )

    # We need to ensure system.make_energy_fn works with this dummy params
    # Or strict checks might fail if it expects more fields.
    # prolix.physics.system usually handles sparse params.
    # But wait, system.make_energy_fn constructs ALL terms including nonbonded.
    # If charges are None, it might fail in electrostatics.
    # Let's mock a simpler energy function by mocking system.make_energy_fn?
    # Or just provide minimal valid params.

    params = {
        "masses": 1.0,
        "charges": jnp.array([0.0, 0.0]),
        "bonds": jnp.array([[0, 1]]),
        # bond_params: (N, 2) -> (length, k)
        "bond_params": jnp.array([[1.0, 100.0]]),

        # Mandatory fields for system.make_energy_fn
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2)),

        "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3)),

        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3)),

        "sigmas": jnp.array([1.0, 1.0]),
        "epsilons": jnp.array([1.0, 1.0]),

        "exclusion_mask": jnp.ones((2, 2)) - jnp.eye(2), # Exclude self, include pair

        # Additional fields usually present
        "res_names": ["ALA"],
        "res_ids": [0, 0],
    }

    # Positions
    r_init = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    spec = replica_exchange.ReplicaExchangeSpec(
        n_replicas=4,
        min_temp=300.0,
        max_temp=1000.0, # Large range to encourage swaps
        total_time_ns=0.0001, # very short
        step_size_fs=2.0,
        exchange_interval_ps=0.01, # very frequent
        save_path="" # no save
    )

    # Run
    # Warning: system.make_energy_fn might complain about missing fields if params aren't perfect.
    try:
        final_state = replica_exchange.run_replica_exchange(params, r_init, spec)

        # Check integrity
        assert final_state.positions.shape == (4, 2, 3)
        assert final_state.walker_indices.shape == (4,)

        # Check energy calculation
        assert final_state.potential_energy.shape == (4,)

        # We can't guarantee swaps in short run but logic ran
        print("Exchange attempts:", final_state.exchange_attempts)
        print("Exchange successes:", final_state.exchange_successes)

    except Exception as e:
        # If system construction fails, we might need a better mock.
        pytest.fail(f"Simulation failed: {e}")

if __name__ == "__main__":
    test_temperature_ladder()
    test_replica_exchange_run()
    print("All tests passed!")
