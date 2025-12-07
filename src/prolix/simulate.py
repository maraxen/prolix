"""Production simulation loop and state management."""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Callable, Optional, Sequence, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import msgpack
import msgpack_numpy as m
import numpy as np
import logging
from jax_md import util, space

# Try importing ArrayRecordWriter
try:
  from array_record.python.array_record_module import ArrayRecordWriter
except ImportError:
  # Fallback or mock for environments without array_record (dev/test)
  ArrayRecordWriter = None  # type: ignore

from prolix.physics import system, simulate as physics_simulate
from priox.md.jax_md_bridge import SystemParams

m.patch()

Array = util.Array
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SimulationSpec:
  """Configuration for a production simulation run."""
  total_time_ns: float
  step_size_fs: float = 2.0
  save_interval_ns: float = 0.001  # 1 ps
  accumulate_steps: int = 500  # Number of frames to accumulate before writing to disk
  save_path: str = "trajectory.array_record"
  temperature_k: float = 300.0
  gamma: float = 1.0  # friction coefficient (1/ps)
  
  # PBC / Explicit Solvent
  box: Optional[Array] = None
  use_pbc: bool = False
  pme_grid_size: int = 64

  
  def __post_init__(self):
      if self.save_interval_ns <= 0:
          raise ValueError("save_interval_ns must be positive")
      if self.step_size_fs <= 0:
          raise ValueError("step_size_fs must be positive")
      if self.accumulate_steps <= 0:
          raise ValueError("accumulate_steps must be positive")


class SimulationState(eqx.Module):
  """State of the simulation at a specific timepoint."""
  positions: Array
  velocities: Array
  
  # Scalar state (required)
  step: Union[int, Array]
  time_ns: Union[float, Array]

  # Optional fields (must come after required)
  forces: Optional[Array] = None
  mass: Optional[Array] = None
  
  # Energies (optional)
  potential_energy: Optional[Union[float, Array]] = None
  kinetic_energy: Optional[Union[float, Array]] = None
  
  def numpy(self) -> dict[str, Any]:
    """Convert state to a dictionary of numpy arrays (on CPU)."""
    # Use jax.device_put to ensure CPU
    def to_cpu(x):
      if x is None: return None
      if isinstance(x, (int, float)): return x
      return np.asarray(jax.device_put(x, jax.devices("cpu")[0]))

    cpu_state = jax.tree_util.tree_map(to_cpu, self)
    
    return {
      "positions": cpu_state.positions,
      "velocities": cpu_state.velocities,
      "forces": cpu_state.forces,
      "mass": cpu_state.mass,
      "step": cpu_state.step,
      "time_ns": cpu_state.time_ns,
      "potential_energy": cpu_state.potential_energy,
      "kinetic_energy": cpu_state.kinetic_energy,
    }

  def to_array_record(self) -> bytes:
    """Serialize the state to msgpack bytes for ArrayRecord."""
    data = self.numpy()
    # Filter None values to save space
    data = {k: v for k, v in data.items() if v is not None}
    return m.packb(data)

  @classmethod
  def from_array_record(cls, packed: bytes) -> "SimulationState":
    """Deserialize from msgpack bytes."""
    data = m.unpackb(packed)
    
    # helper to convert numpy to jax array
    def to_jax(x):
      return jnp.array(x) if x is not None else None

    # Handle potentially missing keys gracefully if schema evolves
    return cls(
      positions=to_jax(data["positions"]),
      velocities=to_jax(data["velocities"]),
      forces=to_jax(data.get("forces")),
      mass=to_jax(data.get("mass")), # Might not be saved every frame
      step=to_jax(data["step"]),
      time_ns=to_jax(data["time_ns"]),
      potential_energy=to_jax(data.get("potential_energy")),
      kinetic_energy=to_jax(data.get("kinetic_energy")),
    )


class TrajectoryWriter:
  """Writes simulation states to an ArrayRecord file."""
  
  def __init__(self, path: str):
      if ArrayRecordWriter is None:
          raise ImportError("array_record not installed")
      self.writer = ArrayRecordWriter(path, 'group_size:1')
      self.closed = False
      
  def write(self, states: Union[SimulationState, Sequence[SimulationState]]) -> None:
      """Write one or more states to the trajectory."""
      if self.closed:
          raise RuntimeError("Writer is closed")
          
      if isinstance(states, SimulationState):
          states = [states]
          
      # If states is a PyTree (e.g. from scan), we might need to unzip it?
      # Usually run_production_simulation returns a stack of states.
      # Which is a SimulationState where each leaf has an extra leading dimension.
      # We need to iterate over that dimension.
      
      # Check if positions has extra dimension
      # This logic depends on how 'states' is passed.
      # If passed as a list of objects, simple iteration.
      # If passed as a single object with stacked arrays, we need to unstack.
      
      # Let's assume the user handles unstacking OR we handle it here.
      # For efficiency from JAX, we likely get a stacked PyTree on CPU.
      
      # Let's support both.
      pass 
      # Actually, let's keep it simple: `write_batch` takes a stacked SimulationState.
      # But `write` declared above takes Sequence.
      
      # Implementation for stacked state (PyTree with leading dim):
      
      # But first, let's implement the loop.
      for state in states:
          self.writer.write(state.to_array_record())
          
  def write_batch(self, stacked_state: SimulationState) -> None:
      """Write a batch of states (stacked PyTree) to trajectory."""
      # We need to slice the PyTree.
      # Number of items is len(stacked_state.positions)
      n_items = stacked_state.positions.shape[0]
      
      # Convert to numpy once efficiently
      cpu_data = stacked_state.numpy() # Dict of arrays with leading dim
      
      for i in range(n_items):
          # Extract slice
          slice_data = {
              k: v[i] if v is not None and getattr(v, "ndim", 0) > 0 else v # Handle scalars if broadcasted?
              for k, v in cpu_data.items()
              if v is not None
          }
          # Pack and write
          self.writer.write(m.packb(slice_data))

  def close(self):
      if not self.closed:
          self.writer.close()
          self.closed = True
          
  def __del__(self):
      self.close()


def run_production_simulation(
  system_params: SystemParams,
  r_init: Array,
  spec: SimulationSpec,
  key: Optional[Array] = None,
) -> SimulationState:
  """Run full production simulation with trajectory saving."""
  
  if key is None:
      key = jax.random.PRNGKey(int(time.time()))
      
  # 1. Setup Physics
  if spec.use_pbc:
      if spec.box is None:
          raise ValueError("Must specify box when use_pbc=True")
      from prolix.physics import pbc
      displacement_fn, shift_fn = pbc.create_periodic_space(spec.box)
  else:
      displacement_fn, shift_fn = space.free()
  
  # Check for implicit solvent or vacuum
  # If use_pbc is True, we disable implicit solvent (GBSA)
  implicit_solvent = not spec.use_pbc
  
  energy_fn = system.make_energy_fn(
      displacement_fn, 
      system_params, 
      implicit_solvent=implicit_solvent,
      box=spec.box,
      use_pbc=spec.use_pbc,
      pme_grid_points=spec.pme_grid_size
  )
  
  # 2. Minimization & Thermalization (Initial setup)
  # We assume r_init is already minimized/thermalized? 
  # Or should we do it here?
  # "Simulation Loop Specification" usually implies the production run.
  # But we need a valid state (v, f) to start.
  # If r_init is just positions, we init velocities.
  
  kT = spec.temperature_k * 0.0019872041 # Boltzmann in kcal/mol/K
  masses = system_params.get("masses", 1.0)
  if not isinstance(masses, float): # Convert from amu
       masses = masses / 418.4
       
  # Canonicalize force
  force_fn = lambda R, **kwargs: -jax.grad(energy_fn)(R, **kwargs)
  
  # Initialize State
  key, split = jax.random.split(key)
  state = physics_simulate.NVTLangevinState(
      position=r_init,
      momentum=None, 
      force=force_fn(r_init), 
      mass=masses, 
      rng=split
  )
  # Init velocities
  state = jax_md_simulate.initialize_momenta(state, split, kT) # Error: define jax_md_simulate
  # Actually simulate.nvt_langevin returns init_fn.
  
  # Let's use the helpers in physics_simulate
  # We need to reconstruct the init_fn/apply_fn
  # The `physics_simulate.rattle_langevin` or `nvt_langevin` returns (init, apply).
  
  steps_per_save = int(round(spec.save_interval_ns * 1000000 / spec.step_size_fs)) # ns -> fs
  logger.info("Steps per save: %d", steps_per_save)
  
  # Setup Integrator
  dt = spec.step_size_fs * 1e-3 # fs -> ps
  gamma = spec.gamma
  
  constrained_bonds = system_params.get("constrained_bonds")
  constrained_lengths = system_params.get("constrained_bond_lengths")
  
  constraints = None
  if constrained_bonds is not None and constrained_lengths is not None and len(constrained_bonds) > 0:
      init_fn, apply_fn = physics_simulate.rattle_langevin(
          energy_fn, shift_fn, dt=dt, kT=kT, gamma=gamma, mass=masses, 
          constraints=(constrained_bonds, constrained_lengths)
      )
  else:
      init_fn, apply_fn = jax_md_simulate.nvt_langevin(
          energy_fn, shift_fn, dt=dt, kT=kT, gamma=gamma, mass=masses
      )

  # Re-initialize state properly with the integrator's init_fn
  state = init_fn(key, r_init)
  
  # 3. Trajectory Writer
  writer = TrajectoryWriter(spec.save_path)
  
  # 4. Simulation Loop
  
  # Total epochs (Python loop)
  total_saves = int(spec.total_time_ns / spec.save_interval_ns)
  accumulate = spec.accumulate_steps
  n_epochs = int(math.ceil(total_saves / accumulate))
  
  logger.info("Starting simulation: %d epochs, %d saves per epoch, total %d saves", n_epochs, accumulate, total_saves)

  # JIT the scan function
  
  @jax.jit
  def scan_fn(carrier, _):
      curr_state = carrier
      
      # Run inner loop (interval steps)
      def step_fn(i, s):
          return apply_fn(s)
          
      curr_state = jax.lax.fori_loop(0, steps_per_save, step_fn, curr_state)
      
      # Calculate Energy for saving (optional, but good for analysis)
      E = energy_fn(curr_state.position)
      K = jax_md_quantity.kinetic_energy(curr_state.momentum, curr_state.mass)
      
      # Convert NVTLangevinState to SimulationState for storage
      # Note: accumulated states in scan must match the return type structure.
      # We return (SimulationState)
      
      sim_state = SimulationState(
          positions=curr_state.position,
          velocities=curr_state.momentum / curr_state.mass, # velocity
          forces=curr_state.force,
          mass=curr_state.mass,
          step=jnp.array(0), # We need to track global step?
          time_ns=jnp.array(0.0), # TODO: track time
          potential_energy=E,
          kinetic_energy=K
      )
      
      return curr_state, sim_state

  # We need to track step count.
  # Let's augment the state or carrier?
  # The carrier is NVTLangevinState.
  # We can't easily add fields to it as it's a fixed dataclass in simulate.py?
  # Actually `simulate.py` defines `NVTLangevinState`.
  # We should probably assume step tracking in Python or add a wrapper.
  
  # Simplified: Just run loop.
  
  current_step = 0
  
  for epoch in range(n_epochs):
      start_time = time.time()
      
      # Determine batch size for this epoch (handle last partial epoch)
      # Simpler to just run fixed size and maybe overshoot slightly or slice?
      batch_size = accumulate
      remaining = total_saves - epoch * accumulate
      if remaining < batch_size:
          batch_size = remaining
      if batch_size <= 0:
          break
          
      # Run JAX Scan
      # We need a dummy xs for scan
      xs = jnp.arange(batch_size)
      
      final_state, stacked_sim_states = jax.lax.scan(scan_fn, state, xs)
      
      # Block until ready
      jax.block_until_ready(stacked_sim_states)
      
      # Update state for next epoch
      state = final_state
      
      # Fixup step/time in saved states if needed (since we didn't track them in scan)
      # Doing it on CPU is fast enough for trajectory metadata
      # Or we could have tracked it in scan with a structured carrier.
      
      # Write to disk
      # Move to CPU
      writer.write_batch(stacked_sim_states)
      
      end_time = time.time()
      duration = end_time - start_time
      sps = (batch_size * steps_per_save) / duration
      logger.info("Epoch %d/%d: %.2f steps/sec", epoch + 1, n_epochs, sps)
      
  writer.close()
  
  # Return final SimulationState (unpacked)
  # We construct one last state
  E = energy_fn(state.position)
  sim_state = SimulationState(
      positions=state.position,
      velocities=state.momentum / state.mass,
      forces=state.force,
      step=jnp.array(total_saves * steps_per_save),
      time_ns=jnp.array(spec.total_time_ns),
      potential_energy=E,
      kinetic_energy=None
  )
  return sim_state

# Import locally to avoid top-level circular imports if possible
from jax_md import simulate as jax_md_simulate
from jax_md import quantity as jax_md_quantity
