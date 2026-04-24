"""Smoke test for EFA electrostatics integration."""
import pytest
import jax
import jax.numpy as jnp
from prolix.physics.rff_coulomb import rff_frequency_sample, erfc_rff_coulomb_energy
from prolix.physics.electrostatic_methods import ElectrostaticMethod


@pytest.mark.smoke
def test_efa_imports():
  assert hasattr(ElectrostaticMethod, 'EFA')
  assert ElectrostaticMethod.EFA.value == 'efa'


@pytest.mark.smoke
def test_efa_minimal_energy():
  """Run EFA on a tiny box; assert finite output."""
  key = jax.random.PRNGKey(0)
  N = 16
  positions = jax.random.normal(key, (N, 3)) * 3.0
  charges = jnp.array([-0.834, 0.417, 0.417] * 5 + [0.0])
  atom_mask = jnp.array([1.0] * 15 + [0.0])
  omega = rff_frequency_sample(alpha=0.34, n_features=64, key=key)
  E = erfc_rff_coulomb_energy(positions, charges, atom_mask, omega, alpha=0.34)
  assert jnp.isfinite(E), f"EFA energy not finite: {E}"
