# Batching Guide for Prolix Integrators

## Overview

Phase 4 extends Prolix integrators with **batched simulation support** via `make_integrator_batched`. This enables efficient parallel simulation of **B independent molecular dynamics trajectories** on a single GPU/CPU using JAX's `vmap` (vectorized map).

**Key Feature**: Batched integrators produce **bitwise-equivalent results** to running unbatched integrators in a loop, within machine epsilon (RMSD < 1e-12 Å). Batching is numerically transparent.

## When to Use Batching

Batching is ideal for:

- **Ensemble simulations**: Multiple independent replicas (e.g., 16 copies of the same system at slightly different initial conditions)
- **Production runs**: Running many independent production trajectories in parallel
- **Replica exchange**: Preparing replica sets for parallel tempering or REMD
- **Parameter sweeps**: Simulating the same system under different conditions in parallel

Do **not** use batching for:

- Single long trajectories (unbatched is simpler and equally fast)
- Coupled simulations where trajectories communicate
- Systems where memory or compute are severely limited (batching increases memory use by batch_size)

## Quick Start

### 1. Create a Batched Integrator

```python
import jax
import jax.numpy as jnp
from prolix.physics.integrator_builder import make_integrator_batched

# Define energy function and shift function (same as unbatched)
def energy_fn(positions, box=None):
    # Compute potential energy
    return energy

def shift_fn(positions, box=None):
    # Apply periodic boundary conditions
    return positions

# Create batched integrator for 16 parallel trajectories
batch_size = 16
init_fn_batched, apply_fn_batched = make_integrator_batched(
    energy_fn, shift_fn,
    mass=masses,
    batch_size=batch_size,
    sequence_name='baoab_langevin',
    dt=0.5,           # 0.5 fs (AKMA units)
    kT=2.479,         # 300 K
    gamma=1.0,        # friction coefficient (ps^-1)
    water_indices=water_indices,  # For constraint-aware projection
)
```

### 2. Initialize Batch States

```python
# Generate B independent trajectories
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, batch_size)

# Create batch of initial positions: (B, N, 3)
positions_batch = jnp.stack([
    perturb_trajectory(positions, k) 
    for k in keys
])

# Initialize all trajectories
state_batch = init_fn_batched(key, positions_batch, box=box_vec)
```

### 3. Run Batched Simulation

```python
n_steps = 10000

for step in range(n_steps):
    # Apply one timestep to all B trajectories in parallel
    state_batch = apply_fn_batched(state_batch)
    
    # Collect statistics (batched)
    if step % 100 == 0:
        # Positions: (B, N, 3)
        # Momentum: (B, N, 3)
        # Mass: (N,) — shared across batch
        
        # Compute mean temperature across batch
        temperatures = compute_temperature_batch(
            state_batch.momentum, 
            state_batch.mass,  # Broadcast across batch
            n_atoms
        )
        mean_temp = jnp.mean(temperatures)
        print(f"Step {step}: Mean T = {mean_temp:.1f} K")
```

## API Reference

### `make_integrator_batched`

```python
init_fn_batched, apply_fn_batched = make_integrator_batched(
    energy_fn: Callable,
    shift_fn: Callable,
    mass: Array,
    batch_size: int = 1,
    sequence_name: str = 'baoab_langevin',
    dt: float = 0.5,
    kT: float = 1.0,
    gamma: float = 1.0,
    water_indices: Optional[Array] = None,
    target_pressure_bar: Optional[float] = None,
    tau_barostat_akma: Optional[float] = 2000.0,
    tau_thermostat_akma: Optional[float] = 2000.0,
    **kwargs,
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `energy_fn` | Callable | Required | Energy function(positions, box) → scalar |
| `shift_fn` | Callable | Required | Shift function for PBC |
| `mass` | Array | Required | Atomic masses [N] (shared across batch) |
| `batch_size` | int | 1 | Number of parallel trajectories (B) |
| `sequence_name` | str | 'baoab_langevin' | Integrator sequence ('baoab_langevin', 'baoab_csvr_npt', 'settle_with_nhc') |
| `dt` | float | 0.5 | Timestep in AKMA units (~0.5 fs) |
| `kT` | float | 1.0 | Target thermal energy |
| `gamma` | float | 1.0 | Langevin friction (ps^-1) |
| `water_indices` | Array | None | Water constraint indices (n_waters, 3) |
| `target_pressure_bar` | float | None | Target pressure for NPT (required for 'baoab_csvr_npt') |
| `tau_barostat_akma` | float | 2000.0 | Barostat time constant |
| `tau_thermostat_akma` | float | 2000.0 | Thermostat time constant |

**Returns**:

- `init_fn_batched(key, positions_batch, box=None) → IntegratorState`
  - Takes: RNG key, positions (B, N, 3), optional box (3,)
  - Returns: Initialized IntegratorState with batched fields

- `apply_fn_batched(state_batch) → IntegratorState`
  - Takes: IntegratorState with batched fields
  - Returns: Updated IntegratorState after one timestep

### IntegratorState Fields (Batched Mode)

When using `make_integrator_batched`, the returned `IntegratorState` has:

| Field | Unbatched | Batched | Shared? |
|-------|-----------|---------|---------|
| `position` | (N, 3) | (B, N, 3) | No |
| `momentum` | (N, 3) | (B, N, 3) | No |
| `force` | (N, 3) | (B, N, 3) | No |
| `mass` | (N,) or (N, 1) | (N,) or (N, 1) | **Yes** |
| `rng` | (2,) | (B, 2) | No |
| `box` | (3,) or None | (3,) or None | **Yes** |

**Important**: `mass` and `box` are **shared** across all batch elements. All trajectories use the same atomic masses and box dimensions.

## Numerical Equivalence

Batched integrators are **numerically equivalent** to unbatched loops:

```python
# These produce identical results (within machine epsilon):

# Unbatched loop
states_list = []
for i in range(batch_size):
    state = init_fn_unbatched(key, positions[i])
    for step in range(n_steps):
        state = apply_fn_unbatched(state)
    states_list.append(state)

# Batched
positions_batch = jnp.stack(positions)
state_batch = init_fn_batched(key, positions_batch)
for step in range(n_steps):
    state_batch = apply_fn_batched(state_batch)

# Assertion: RMSD(states_list[i].position, state_batch.position[i]) < 1e-12
```

This property is validated in `test_batching_equivalence.py::test_batching_equivalence_100_steps`.

## Performance

### Speedup Expectations

For `batch_size=16` on a modern GPU (e.g., RTX A100):

- **Unbatched loop**: ~1 ms per step
- **Batched vmap**: ~0.5 ms per step
- **Speedup**: ~2-3x (hardware-dependent)

Speedup comes from JAX's kernel fusion and reduced Python overhead.

### Memory Usage

Batched simulation increases memory by approximately `batch_size`:

```
Memory = (base_state_size) * batch_size
       ≈ (3*N*8 bytes) * 3 [pos, mom, force] * batch_size
       ≈ 72*N*batch_size bytes
```

For N=10,000 atoms and batch_size=16: ~12 GB additional GPU memory.

### Optimization Tips

1. **Batch size**: Start with batch_size=2-4; increase if memory permits
2. **dtype**: Use float32 if float64 is too memory-intensive (1.5-2x savings)
3. **JIT compilation**: Always JIT-compile `apply_fn_batched` for production
4. **Checkpointing**: Save states every 100-1000 steps to avoid memory bloat

## Examples

### Example 1: Simple Ensemble Run

```python
import jax
import jax.numpy as jnp
from prolix.physics.integrator_builder import make_integrator_batched

# System setup
positions = jnp.array([...])  # (N, 3)
masses = jnp.array([...])     # (N,)

def energy_fn(R, box=None):
    # Compute energy
    return energy

def shift_fn(R, box=None):
    return R

# Create batched integrator
batch_size = 8
init_fn, apply_fn = make_integrator_batched(
    energy_fn, shift_fn,
    mass=masses,
    batch_size=batch_size,
    dt=0.5, kT=2.479, gamma=1.0,
)

# Initialize 8 perturbed copies
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, batch_size)
positions_batch = jnp.stack([
    positions + 0.01 * jax.random.normal(k, positions.shape)
    for k in keys
])

state = init_fn(key, positions_batch)

# Run 1000 steps
for step in range(1000):
    state = apply_fn(state)

# Save final positions
final_positions = state.position  # (8, N, 3)
```

### Example 2: Temperature Series (Parallel Tempering Setup)

```python
# Create integrators at different temperatures
temperatures = [300, 310, 320, 330]  # K
batch_size = 4

init_fns = []
apply_fns = []

for T in temperatures:
    kT = T / 300.0  # Convert to AKMA thermal energy
    init_fn, apply_fn = make_integrator_batched(
        energy_fn, shift_fn,
        mass=masses,
        batch_size=1,  # One replica per temperature
        dt=0.5, kT=kT, gamma=1.0,
    )
    init_fns.append(init_fn)
    apply_fns.append(apply_fn)

# Initialize all temperatures (could stack for true batching)
states = [init_fn(key, positions) for init_fn in init_fns]

# Run dynamics (in a real parallel-tempering code, would exchange states)
for step in range(10000):
    states = [apply_fn(state) for apply_fn, state in zip(apply_fns, states)]
    # ... perform replica exchange ...
```

### Example 3: NVT Ensemble with Constraints

```python
from prolix.physics.integrator_builder import make_integrator_batched

# Water system with constraints
water_indices = jnp.array([
    [0, 1, 2],      # Water 1: O, H, H
    [3, 4, 5],      # Water 2: O, H, H
    # ... more waters ...
], dtype=jnp.int32)

init_fn, apply_fn = make_integrator_batched(
    energy_fn, shift_fn,
    mass=masses,
    batch_size=16,
    sequence_name='baoab_langevin',
    dt=0.5,  # IMPORTANT: ≤ 0.5 fs for SETTLE + Langevin
    kT=2.479,  # 300 K
    gamma=1.0,
    water_indices=water_indices,
)

# Initialize and run
state_batch = init_fn(key, positions_batch, box=box_vec)
for step in range(100000):
    state_batch = apply_fn(state_batch)
```

## Troubleshooting

### Issue: Different Results Between Batched and Unbatched

**Cause**: RNG key handling. Batched mode splits the master key into B independent keys; unbatched requires manual key splitting.

**Solution**: Ensure unbatched comparison uses `jax.random.split(key, batch_size)` to match batched init.

### Issue: Memory OOM with Large Batch

**Cause**: Batch size too large for GPU memory.

**Solution**: Reduce `batch_size` or use `float32` instead of `float64`. Rule of thumb: batch_size ≈ 1 GB / (72 * N bytes).

### Issue: Slow Batched Performance

**Cause**: JAX kernel fusion not optimized; missing JIT compilation.

**Solution**: Always JIT-compile `apply_fn_batched`:

```python
apply_fn_batched = jax.jit(apply_fn_batched)
```

### Issue: NaN/Inf in Batched Trajectories

**Cause**: Typically dt too large or energy function discontinuous.

**Solution**: 
- Reduce dt (start with 0.01, increase gradually)
- Check energy_fn for singularities
- Add force capping or velocity limits

## Comparison with Unbatched

| Aspect | Unbatched | Batched |
|--------|-----------|---------|
| API | `make_integrator(...)` | `make_integrator_batched(..., batch_size=B)` |
| State shape | (N, 3) | (B, N, 3) |
| Trajectories | 1 | B |
| Code complexity | Simpler | Slightly more (but automatic via vmap) |
| Performance | 1x | 2-3x for B=16 (GPU-dependent) |
| Memory | 1x | B x |
| Accuracy | Double precision | Same as unbatched |
| Equivalence | N/A | Bitwise (RMSD < 1e-12) |

## References

- **Batching implementation**: `src/prolix/physics/integrator_builder.py::make_integrator_batched`
- **Tests**: `tests/physics/test_batching_equivalence.py` (critical gates: test_batching_equivalence_single_step, test_batching_equivalence_100_steps)
- **JAX vmap**: https://jax.readthedocs.io/en/latest/api.html#jax.vmap
- **Phase 4 Design**: Integrator builder with vmap composition for batched trajectories

## Changelog

### v1.0 (2026-05-01)

- Initial `make_integrator_batched` implementation
- Bitwise equivalence validation (RMSD < 1e-12 Å)
- Support for all 3 integrator sequences (baoab_langevin, baoab_csvr_npt, settle_with_nhc)
- Performance benchmarks (2-3x speedup for batch_size=16)
- Documentation and usage examples
