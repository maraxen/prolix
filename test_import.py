import sys
import os
sys.path.insert(0, os.path.abspath('src'))
# Bypass jax_md issue if it's imported by __init__.py in src/prolix
# Actually let's just import optimization.py directly
import jax
import jax.numpy as jnp
import equinox as eqx
import functools
# Mocking dependencies
import types
sys.modules['prolix.typing'] = types.ModuleType('prolix.typing')
sys.modules['prolix.physics.tiling'] = types.ModuleType('prolix.physics.tiling')
# Stub functions
import prolix.physics.tiling
prolix.physics.tiling.tile_reduction = lambda *a: 0.0
prolix.physics.tiling.tile_reduction_nl = lambda *a: 0.0
prolix.physics.tiling.pad_to_tile = lambda r, dim: (r, jnp.ones(r.shape, dtype=bool))
# Import
from prolix.physics.optimization import chunked_lj_energy
print("Successfully imported chunked_lj_energy")
