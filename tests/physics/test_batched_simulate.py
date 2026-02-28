import jax
import jax.numpy as jnp
from jax import random
import pytest

from prolix.padding import bucket_proteins, collate_batch
from prolix.batched_simulate import (
    LangevinState, 
    make_langevin_step, 
    safe_map, 
    batched_minimize,
    batched_equilibrate,
    batched_produce
)
from proxide import OutputSpec, CoordFormat
from proxide.io.parsing.backend import parse_structure
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pdb"
FF_PATH = Path(__file__).parent.parent.parent.parent / "proxide" / "src" / "proxide" / "assets" / "protein.ff19SB.xml"

def pytest_configure():
    # enable x64 just in case
    jax.config.update("jax_enable_x64", True)

@pytest.fixture
def fake_padded_batch():
    """Create a batch of 2 identical small proteins padded."""
    pdb_path = DATA_DIR / "1CRN.pdb"
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        parameterize_md=True,
        force_field=str(FF_PATH),
        add_hydrogens=True,
    )
    # create two copies for batching
    p1 = parse_structure(str(pdb_path), spec)
    p2 = parse_structure(str(pdb_path), spec)
    
    # We alter coordinates of p2 slightly so they differ
    import dataclasses
    p2_coords = [c + 0.1 for c in p2.coordinates]
    p2 = dataclasses.replace(p2, coordinates=p2_coords)
    
    # Bucket them (this returns a dict {bucket_size: [PaddedSystem, ...]})
    buckets = bucket_proteins([p1, p2])
    
    bucket_size = list(buckets.keys())[0]
    padded_list = buckets[bucket_size]
    
    return collate_batch(padded_list)


def test_safe_map():
    # test dummy
    x = jnp.array([1, 2, 3, 4, 5, 6])
    def fn(a): return a * 2
    res_none = safe_map(fn, x, chunk_size=None)
    assert jnp.allclose(res_none, x * 2)
    
    res_chunk_2 = safe_map(fn, x, chunk_size=2)
    assert jnp.allclose(res_chunk_2, x * 2)
    
    res_chunk_1 = safe_map(fn, x, chunk_size=1)
    assert jnp.allclose(res_chunk_1, x * 2)

def test_langevin_step_finite(fake_padded_batch):
    sys = fake_padded_batch
    dt = 2.0 / 48.88821  # 2fs
    kT = 0.6
    gamma = 1.0
    
    step_fn = make_langevin_step(dt, kT, gamma)
    
    B, N, _ = sys.positions.shape
    key = random.PRNGKey(42)
    keys = random.split(key, B)
    
    state = LangevinState(
        positions=sys.positions,
        momentum=jnp.zeros_like(sys.positions),
        force=jnp.zeros_like(sys.positions),
        mass=sys.masses,
        key=keys
    )
    
    def single_step(args):
        sys_inner, state_inner = args
        return step_fn(sys_inner, state_inner)
        
    new_state = safe_map(single_step, (sys, state), chunk_size=1)
    
    assert jnp.all(jnp.isfinite(new_state.positions))
    assert jnp.all(jnp.isfinite(new_state.force))
    
    # Ghost atoms remain at (9999,9999,9999)
    for b in range(B):
        n_real = int(sys.n_real_atoms[b])
        ghost_pos = new_state.positions[b, n_real:]
        assert jnp.allclose(ghost_pos, 9999.0, atol=1e-1)


def test_batched_minimize(fake_padded_batch):
    minimized_pos = batched_minimize(fake_padded_batch, max_steps=10, chunk_size=1)
    assert jnp.all(jnp.isfinite(minimized_pos))
    assert minimized_pos.shape == fake_padded_batch.positions.shape
    
    B = fake_padded_batch.positions.shape[0]
    for b in range(B):
        n_real = int(fake_padded_batch.n_real_atoms[b])
        ghost_pos = minimized_pos[b, n_real:]
        assert jnp.allclose(ghost_pos, 9999.0, atol=1e-1)


def test_batched_equilibrate(fake_padded_batch):
    state = batched_equilibrate(fake_padded_batch, key=random.PRNGKey(0), n_steps=5, chunk_size=1)
    
    assert jnp.all(jnp.isfinite(state.positions))
    assert jnp.all(jnp.isfinite(state.force))
    
    B = state.positions.shape[0]
    for b in range(B):
        n_real = int(fake_padded_batch.n_real_atoms[b])
        ghost_pos = state.positions[b, n_real:]
        assert jnp.allclose(ghost_pos, 9999.0, atol=1e-1)


def test_batched_produce(fake_padded_batch):
    state = batched_equilibrate(fake_padded_batch, key=random.PRNGKey(0), n_steps=5, chunk_size=1)
    
    final_state, traj = batched_produce(fake_padded_batch, state, n_saves=2, steps_per_save=3, chunk_size=1)
    
    assert jnp.all(jnp.isfinite(final_state.positions))
    assert jnp.all(jnp.isfinite(traj))
    assert traj.shape == (2, 2, fake_padded_batch.positions.shape[1], 3)
