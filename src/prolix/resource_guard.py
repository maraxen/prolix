"""Resource management and memory estimation for Prolix simulations."""

import logging
import psutil
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def get_available_memory_bytes() -> int:
  """Get available memory on the JAX default device."""
  try:
    device = jax.devices()[0]
    if device.platform == "gpu":
      stats = device.memory_stats()
      # bytes_limit is total memory, but we care about available.
      # JAX preallocates usually, so we might check 'bytes_limit' - 'bytes_in_use'.
      limit = stats.get("bytes_limit", 0)
      in_use = stats.get("bytes_in_use", 0)
      if limit > 0:
        return limit - in_use
    elif device.platform == "cpu":
      # For CPU, check system memory
      return psutil.virtual_memory().available
  except Exception:
    pass

  # Fallback to a safe default? Or return max int (assume infinite)
  return 16 * 1024**3  # Assume 16GB if unknown


def estimate_simulation_memory(
  n_atoms: int,
  accumulate_steps: int = 500,
  use_neighbor_list: bool = False,
  neighbor_buffer_size: int = 400,  # Avg neighbors
  pme_grid_size: int = 64,
  use_pbc: bool = False,
  double_precision: bool = False,
) -> dict[str, int]:
  """Estimate memory usage for a simulation run.

  Returns a breakdown of estimated bytes.
  """

  float_size = 8 if double_precision else 4
  int_size = 4

  # 1. State Size (Positions, Velocities, Forces, Mass)
  # Positions: (N, 3), Velocities: (N, 3), Forces: (N, 3), Mass: (N,)
  state_size = n_atoms * (3 + 3 + 3 + 1) * float_size

  # 2. Accumulation Buffer (output of scan)
  # Stores 'accumulate_steps' frames of positions (N, 3) and maybe other fields?
  # run_simulation saves SimulationState (pos, vel, forces, mass, etc)
  # But usually we only care about positions for trajectory?
  # Actually simulate.py scan returns FULL SimulationState.
  # So it stores pos, vel, forces, mass, etc. for EACH step in the batch.
  # This is HUUUUGE if accumulate_steps is large.
  frame_size = n_atoms * (3 + 3 + 3 + 1) * float_size  # Full state
  accumulation_buffer = accumulate_steps * frame_size

  # 3. Neighbor List
  neighbor_list = 0
  if use_neighbor_list:
    # idx: (N, K) int32
    neighbor_list = n_atoms * neighbor_buffer_size * int_size
    # reference positions check? (N, 3)

  # 4. PME Grid (if valid)
  pme_grid = 0
  if use_pbc:
    # Complex grid (N, N, N)
    pme_grid = (pme_grid_size**3) * (float_size * 2)  # Complex number

  # 5. JIT / Overhead buffer
  # Rough heuristic: 2x state size for intermediate gradients + some constant
  jit_overhead = state_size * 3 + 500 * 1024**2  # 500MB baseline overhead

  total = state_size + accumulation_buffer + neighbor_list + pme_grid + jit_overhead

  return {
    "state_static": state_size,
    "accumulation": accumulation_buffer,
    "neighbor_list": neighbor_list,
    "pme_grid": pme_grid,
    "jit_overhead": jit_overhead,
    "total": total,
  }


def check_memory_budget(
  n_atoms: int,
  accumulate_steps: int,
  use_neighbor_list: bool = False,
  pme_grid_size: int = 64,
  use_pbc: bool = False,
) -> bool:
  """Check if simulation fits in memory. Logs warning if not."""

  estimates = estimate_simulation_memory(
    n_atoms,
    accumulate_steps,
    use_neighbor_list,
    neighbor_buffer_size=800 if use_neighbor_list else 0,  # Conservative
    pme_grid_size=pme_grid_size,
    use_pbc=use_pbc,
  )

  total_needed = estimates["total"]
  available = get_available_memory_bytes()

  usage_mb = total_needed / 1024**2
  avail_mb = available / 1024**2

  logger.info(f"Memory Estimation: {usage_mb:.1f} MB needed (Available: {avail_mb:.1f} MB)")
  logger.info(f"  - Accumulation Buffer: {estimates['accumulation'] / 1024**2:.1f} MB")

  if total_needed > available * 0.9:  # 90% threshold
    logger.warning(
      f"POTENTIAL OOM: Estimated memory {usage_mb:.1f} MB exceeds 90% of available {avail_mb:.1f} MB."
    )
    logger.warning(
      "Suggestion: Reduce 'accumulate_steps' (currently %d) or use smaller system.",
      accumulate_steps,
    )
    return False

  return True
