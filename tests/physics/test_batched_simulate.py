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
    batched_equilibrate,  # imported to test deprecation warning only
    batched_produce,
    batched_produce_streaming,
)
import proxide
from proxide import OutputSpec, CoordFormat
from proxide.io.parsing.backend import parse_structure
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pdb"
# Resolve packaged ff19SB (works for venv/pip installs; sibling-repo paths break in CI).
FF_PATH = Path(proxide.__file__).resolve().parent / "assets" / "protein.ff19SB.xml"

def pytest_configure():
    # Production runs at float32 (JAX_ENABLE_X64=False)
    pass

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

def test_safe_map_heterogeneous_pytree():
    """Test safe_map with compound pytrees containing heterogeneous leaf shapes.

    Regression test for Sprint 7 Step 3: safe_map must validate that all leaves
    have the same leading (batch) dimension. Heterogeneous pytrees where one leaf
    lacks a batch dimension should raise ValueError.
    """
    import jax.tree_util
    import dataclasses

    # Case (a): Homogeneous compound pytree — all leaves have batch dim B=3
    @jax.tree_util.register_pytree_node_class
    @dataclasses.dataclass(frozen=True)
    class HomogeneousNode:
        x: jnp.ndarray  # (B, 5)
        y: jnp.ndarray  # (B, 3)
        def tree_flatten(self):
            return (self.x, self.y), None
        @classmethod
        def tree_unflatten(cls, aux, children):
            x, y = children
            return cls(x, y)

    B = 3
    homog = HomogeneousNode(
        x=jnp.ones((B, 5)),
        y=jnp.ones((B, 3)) * 2.0,
    )
    def fn_homog(node):
        return HomogeneousNode(node.x * 2, node.y * 3)
    result = safe_map(fn_homog, homog, chunk_size=1)
    assert result.x.shape == (B, 5)
    assert jnp.allclose(result.x, 2.0)
    assert jnp.allclose(result.y, 6.0)

    # Case (b): Heterogeneous compound pytree — one leaf has no batch dim
    @jax.tree_util.register_pytree_node_class
    @dataclasses.dataclass(frozen=True)
    class HeterogeneousNode:
        positions: jnp.ndarray  # (B, 3) — batch leading dim
        warn_counts: jnp.ndarray  # (4,) — NO batch dim (scalar config)
        def tree_flatten(self):
            return (self.positions, self.warn_counts), None
        @classmethod
        def tree_unflatten(cls, aux, children):
            pos, wc = children
            return cls(pos, wc)

    hetero = HeterogeneousNode(
        positions=jnp.ones((B, 3)),
        warn_counts=jnp.zeros(4, dtype=jnp.int32),  # No batch dim
    )
    def fn_hetero(node):
        return HeterogeneousNode(node.positions * 2, node.warn_counts)

    # Should raise ValueError due to heterogeneous leaf shapes
    try:
        result = safe_map(fn_hetero, hetero, chunk_size=1)
        assert False, "Expected ValueError for heterogeneous pytree, but got none"
    except ValueError as e:
        assert "same leading (batch) dimension" in str(e)

    # Case (c): Existing test_safe_map still passes (simple array case)
    x = jnp.array([1, 2, 3, 4, 5, 6])
    def fn(a): return a * 2
    res = safe_map(fn, x, chunk_size=2)
    assert jnp.allclose(res, x * 2)

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
        key=keys,
        cap_count=jnp.zeros(B, dtype=jnp.int32),
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
    # Short schedules keep first-JIT compile tractable in CI (defaults are 500×4 FIRE + L-BFGS).
    minimized_pos, converged, rms_grad = batched_minimize(
        fake_padded_batch,
        max_steps=10,
        chunk_size=1,
        fire_stage_steps=(2, 2, 2, 2),
        lbfgs_steps=0,
    )
    assert jnp.all(jnp.isfinite(minimized_pos))
    assert minimized_pos.shape == fake_padded_batch.positions.shape
    
    B = fake_padded_batch.positions.shape[0]
    for b in range(B):
        n_real = int(fake_padded_batch.n_real_atoms[b])
        ghost_pos = minimized_pos[b, n_real:]
        assert jnp.allclose(ghost_pos, 9999.0, atol=1e-1)


def _cold_start_state(batch) -> LangevinState:
    """Build a valid LangevinState from a batch using cold-start initialization.

    Computes initial forces from the energy function so that the first
    production step does not start from stale zero-force fields (the root
    cause of the batched_equilibrate NaN bug).
    """
    import dataclasses as _dc
    from jax_md import space as _space
    from prolix.batched_energy import single_padded_energy
    from prolix.physics.md_potential_bundle import value_energy_and_grad_energy

    displacement_fn, _ = _space.free()
    B = batch.positions.shape[0]

    def _init_force(sys):
        def _e(r):
            return single_padded_energy(_dc.replace(sys, positions=r), displacement_fn)
        _, f = value_energy_and_grad_energy(_e, sys.positions)
        return f

    initial_forces = jax.vmap(_init_force)(batch)
    keys = random.split(random.PRNGKey(0), B)
    return LangevinState(
        positions=batch.positions,
        momentum=jnp.zeros_like(batch.positions),
        force=initial_forces,
        mass=batch.masses,
        key=keys,
        cap_count=jnp.zeros(B, dtype=jnp.int32),
    )


def test_batched_equilibrate_deprecated(fake_padded_batch):
    """batched_equilibrate must emit DeprecationWarning."""
    import pytest
    B = fake_padded_batch.positions.shape[0]
    system_index = jnp.arange(B)
    with pytest.warns(DeprecationWarning, match="batched_equilibrate is deprecated"):
        batched_equilibrate(
            fake_padded_batch, system_index, fake_padded_batch.positions,
            key=random.PRNGKey(0), duration_ps=0.02, temp=300.0, chunk_size=1
        )


def test_cold_start_state_finite(fake_padded_batch):
    """Cold-start LangevinState has finite positions and forces (replaces test_batched_equilibrate)."""
    state = _cold_start_state(fake_padded_batch)

    assert jnp.all(jnp.isfinite(state.positions))
    assert jnp.all(jnp.isfinite(state.force))

    B = state.positions.shape[0]
    for b in range(B):
        n_real = int(fake_padded_batch.n_real_atoms[b])
        ghost_pos = state.positions[b, n_real:]
        assert jnp.allclose(ghost_pos, 9999.0, atol=1e-1)


def test_batched_produce(fake_padded_batch):
    state = _cold_start_state(fake_padded_batch)
    final_state, traj = batched_produce(fake_padded_batch, state, n_saves=2, steps_per_save=3, chunk_size=1)

    assert jnp.all(jnp.isfinite(final_state.positions))
    assert jnp.all(jnp.isfinite(traj))
    assert traj.shape == (2, 2, fake_padded_batch.positions.shape[1], 3)


def test_batched_produce_streaming(fake_padded_batch):
    """Streaming production via io_callback — no GPU trajectory accumulation.

    Verifies the callback mechanism: correct number of invocations, correct
    shapes, and numerical equivalence with the accumulation path.
    Constructs a cold-start state directly to avoid pre-existing NaN issues
    in the equilibration path.
    """
    import numpy as np
    import dataclasses

    from jax_md import space
    from prolix.batched_energy import single_padded_energy
    from prolix.physics.md_potential_bundle import value_energy_and_grad_energy

    batch = fake_padded_batch
    B = batch.positions.shape[0]
    N = batch.positions.shape[1]

    # Build a cold-start LangevinState directly (no equilibration)
    displacement_fn, _ = space.free()

    def compute_initial_force(sys):
        def energy_fn(r):
            sys_with_r = dataclasses.replace(sys, positions=r)
            return single_padded_energy(sys_with_r, displacement_fn)

        _, grad_e = value_energy_and_grad_energy(energy_fn, sys.positions)
        return grad_e

    initial_forces = jax.vmap(compute_initial_force)(batch)
    keys = random.split(random.PRNGKey(42), B)

    state = LangevinState(
        positions=batch.positions,
        momentum=jnp.zeros_like(batch.positions),
        force=initial_forces,
        mass=batch.masses,
        key=keys,
        cap_count=jnp.zeros(B, dtype=jnp.int32),
    )

    # Collect frames in a list via the callback
    collected_frames = []

    def write_fn(positions, batch_idx, save_idx):
        """Host-side callback — receives numpy arrays."""
        collected_frames.append({
            "positions": np.array(positions),
            "batch_idx": int(batch_idx),
            "save_idx": int(save_idx),
        })

    n_saves = 3
    steps_per_save = 1  # minimal steps to keep physics from diverging

    system_index = jnp.arange(B)
    final_state_stream = batched_produce_streaming(
        batch, system_index, state,
        n_saves=n_saves,
        steps_per_save=steps_per_save,
        write_fn=write_fn,
        chunk_size=1,
    )

    # --- Verify IO mechanism ---

    # Correct number of callback invocations
    assert len(collected_frames) == B * n_saves, (
        f"Expected {B * n_saves} frames, got {len(collected_frames)}"
    )

    # All frames have correct shape
    for frame in collected_frames:
        assert frame["positions"].shape == (N, 3)

    # Batch indices cover all systems
    batch_ids = {frame["batch_idx"] for frame in collected_frames}
    assert batch_ids == set(range(B))

    # Save indices cover all saves per system
    for b in range(B):
        save_ids = {
            frame["save_idx"]
            for frame in collected_frames
            if frame["batch_idx"] == b
        }
        assert save_ids == set(range(n_saves))

    # --- Verify numerical equivalence with accumulation path ---

    final_state_accum, traj_accum = batched_produce(
        batch, state,
        n_saves=n_saves, steps_per_save=steps_per_save, chunk_size=1,
    )

    # Final states must be bitwise identical (same computation graph)
    pos_stream = np.asarray(final_state_stream.positions)
    pos_accum = np.asarray(final_state_accum.positions)
    # Use allclose with equal_nan to handle any NaN identically
    assert np.allclose(pos_stream, pos_accum, equal_nan=True), (
        "Streaming and accumulation final states diverged!"
    )

    # Streamed frames must match accumulated trajectory frame-by-frame
    for frame in collected_frames:
        b = frame["batch_idx"]
        s = frame["save_idx"]
        expected = np.asarray(traj_accum[b, s])
        np.testing.assert_allclose(
            frame["positions"], expected,
            rtol=0, atol=0,
            err_msg=f"Frame mismatch at batch={b}, save={s}",
        )


def test_batched_produce_streaming_write_batch_size(fake_padded_batch):
    """``write_batch_size>1`` passes (W,N,3) once per batch of saves; matches ``batched_produce``."""
    import numpy as np
    import dataclasses

    from jax_md import space
    from prolix.batched_energy import single_padded_energy
    from prolix.physics.md_potential_bundle import value_energy_and_grad_energy

    batch = fake_padded_batch
    B = batch.positions.shape[0]
    N = batch.positions.shape[1]

    displacement_fn, _ = space.free()

    def compute_initial_force(sys):
        def energy_fn(r):
            sys_with_r = dataclasses.replace(sys, positions=r)
            return single_padded_energy(sys_with_r, displacement_fn)

        _, grad_e = value_energy_and_grad_energy(energy_fn, sys.positions)
        return grad_e

    initial_forces = jax.vmap(compute_initial_force)(batch)
    state = LangevinState(
        positions=batch.positions,
        momentum=jnp.zeros_like(batch.positions),
        force=initial_forces,
        mass=batch.masses,
        key=random.split(random.PRNGKey(1), B),
        cap_count=jnp.zeros(B, dtype=jnp.int32),
    )

    _, traj_accum = batched_produce(
        batch, state, n_saves=2, steps_per_save=1, chunk_size=1,
    )

    chunks: dict[int, list] = {b: [] for b in range(B)}

    def write_fn(positions, batch_idx, start_save_idx):
        arr = np.array(positions)
        b = int(batch_idx)
        chunks[b].append((int(start_save_idx), arr))

    wbatch = 2
    system_index = jnp.arange(B)
    _ = batched_produce_streaming(
        batch,
        system_index,
        state,
        n_saves=2,
        steps_per_save=1,
        write_fn=write_fn,
        chunk_size=1,
        write_batch_size=wbatch,
    )

    for b in range(B):
        assert len(chunks[b]) == 1
        sidx, pos_chunk = chunks[b][0]
        assert sidx == 0
        assert pos_chunk.shape == (wbatch, N, 3)
        np.testing.assert_array_equal(pos_chunk[0], np.asarray(traj_accum[b, 0]))
        np.testing.assert_array_equal(pos_chunk[1], np.asarray(traj_accum[b, 1]))

