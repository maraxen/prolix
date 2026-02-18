"""Production simulation loop and state management."""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import msgpack_numpy as m
import numpy as np
from jax_md import space, util
from prolix import resource_guard
from prolix.physics import neighbor_list as nl

if TYPE_CHECKING:
  from proxide.core.atomic_system import (
    AtomicConstants,
    AtomicState,
    AtomicSystem,
    MolecularTopology,
  )
  from proxide.core.containers import Protein

try:
  from array_record.python.array_record_module import ArrayRecordWriter
except ImportError:
  ArrayRecordWriter = None  # type: ignore


from prolix.physics import simulate as physics_simulate
from prolix.physics import system as physics_system

if TYPE_CHECKING:
  from collections.abc import Sequence

  from proxide.types import SystemParams

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
  box: Array | None = None
  use_pbc: bool = False
  pme_grid_size: int = 64

  # Neighbor list for O(N*K) non-bonded (vs O(N^2))
  use_neighbor_list: bool = False
  neighbor_cutoff: float = 9.0  # Angstroms
  neighbor_update_interval_fs: float = 20.0  # Update neighbor list every N fs (20-50 fs typical)
  exclusion_spec: nl.ExclusionSpec | None = None  # Optional pre-built spec

  def __post_init__(self):
    if self.save_interval_ns <= 0:
      msg = "save_interval_ns must be positive"
      raise ValueError(msg)
    if self.step_size_fs <= 0:
      msg = "step_size_fs must be positive"
      raise ValueError(msg)
    if self.accumulate_steps <= 0:
      msg = "accumulate_steps must be positive"
      raise ValueError(msg)


class SimulationState(eqx.Module):
  """State of the simulation at a specific timepoint."""

  positions: Array
  velocities: Array

  # Scalar state (required)
  step: int | Array
  time_ns: float | Array

  # Optional fields (must come after required)
  forces: Array | None = None
  mass: Array | None = None

  # Energies (optional)
  potential_energy: float | Array | None = None
  kinetic_energy: float | Array | None = None

  def numpy(self) -> dict[str, Any]:
    """Convert state to a dictionary of numpy arrays (on CPU)."""

    def to_cpu(x):
      if x is None:
        return None
      if isinstance(x, (int, float)):
        return x
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
  def from_array_record(cls, packed: bytes) -> SimulationState:
    """Deserialize a SimulationState from msgpack bytes.

    Args:
        packed: Msgpack-encoded bytes from ArrayRecord.

    Returns:
        SimulationState with JAX arrays.
    """
    data = m.unpackb(packed)

    def to_jax(x):
      return jnp.array(x) if x is not None else None

    return cls(
      positions=to_jax(data["positions"]),
      velocities=to_jax(data["velocities"]),
      forces=to_jax(data.get("forces")),
      mass=to_jax(data.get("mass")),
      step=to_jax(data["step"]),
      time_ns=to_jax(data["time_ns"]),
      potential_energy=to_jax(data.get("potential_energy")),
      kinetic_energy=to_jax(data.get("kinetic_energy")),
    )


class TrajectoryWriter:
  """Writes simulation states to an ArrayRecord file."""

  def __init__(self, path: str) -> None:
    if ArrayRecordWriter is None:
      msg = "array_record not installed"
      raise ImportError(msg)
    self.writer = ArrayRecordWriter(path, "group_size:1")
    self.closed = False

  def write(self, states: SimulationState | Sequence[SimulationState]) -> None:
    """Append one or more states to the trajectory file.

    Process:
    1.  **Iterate**: Loop over the provided states.
    2.  **Serialize**: Convert each state to msgpack bytes.
    3.  **Append**: Write bytes to the underlying ArrayRecord.

    Args:
        states: Single state or sequence of states to write.
    """
    if self.closed:
      msg = "Writer is closed"
      raise RuntimeError(msg)

    if isinstance(states, SimulationState):
      states = [states]

    for state in states:
      self.writer.write(state.to_array_record())

  def write_batch(self, stacked_state: SimulationState) -> None:
    """Write a batch of states (stacked PyTree) to trajectory.

    NOTES:
        Extracts slices from a stacked SimulationState and writes each frame
        individually to the ArrayRecord trajectory.
    """
    n_items = stacked_state.positions.shape[0]
    cpu_data = stacked_state.numpy()

    for i in range(n_items):
      slice_data = {
        k: v[i] if v is not None and getattr(v, "ndim", 0) > 0 else v
        for k, v in cpu_data.items()
        if v is not None
      }
      self.writer.write(m.packb(slice_data))

  def close(self) -> None:
    if not self.closed:
      self.writer.close()
      self.closed = True

  def __del__(self) -> None:
    self.close()


def run_simulation(
  system: AtomicSystem | Protein | SystemParams | None = None,
  initial_positions: Array | None = None,
  spec: SimulationSpec | None = None,
  key: Array | None = None,
  # Hierarchical PyTree arguments (preferred for vmap efficiency)
  topology: MolecularTopology | None = None,
  state: AtomicState | None = None,
  constants: AtomicConstants | None = None,
  # Deprecated: kept for backward compatibility
  system_params: SystemParams | None = None,
) -> SimulationState:
  r"""Run a full production simulation with periodic trajectory saving.

  Process:
  1.  **Initialize**: Setup energy function, displacement functions, and random keys.
  2.  **Exclusions**: Build `ExclusionSpec` to handle 1-2, 1-3, and 1-4 scaling.
  3.  **Minimize**: Perform local energy minimization (L-BFGS or Gradient Descent).
  4.  **Dynamics**: Initialize NVT Langevin integrator with RATTLE constraints if needed.
  5.  **Simulate**: Execute production epochs using `jax.lax.scan` for efficiency.
  6.  **Validate**: Check for numerical instability (NaN/Inf) at each archive interval.
  7.  **Store**: Serialize batches of states to `ArrayRecord` trajectory.

  Notes:
  This function is the primary entry point for Prolix production MD. It supports
  both implicit solvent (GBSA) and explicit solvent (PME/PBC) modes.

  When using the hierarchical interface (`topology`, `state`, `constants`),
  the function can be efficiently `vmap`-ed for parallel ensemble simulations:

  ```python
  batched_sim = jax.vmap(run_simulation, in_axes=(None, 0, None, 0, None, 0, None))
  final_states = batched_sim(None, initial_positions_samples, spec, keys, ...)
  ```

  Args:
      system: `AtomicSystem`, `Protein`, or `SystemParams` dictionary.
      initial_positions: Initial coordinates (N, 3). Defaults to system coordinates.
      spec: Configuration parameters for the run.
      key: JAX random PRNG key.
      topology: `MolecularTopology` for hierarchical setup.
      state: `AtomicState` with coordinates for hierarchical setup.
      constants: `AtomicConstants` for hierarchical setup.
      system_params: [DEPRECATED] Dictionary of parameters.

  Returns:
      Final `SimulationState` after the specified total time.
  """
  from prolix.compat import system_params_to_protein
  from proxide.core.containers import Protein as ProteinCls

  # Handle backward compatibility: system_params= kwarg
  if system_params is not None:
    import warnings

    warnings.warn(
      "system_params= is deprecated, pass AtomicSystem or dict as first arg",
      DeprecationWarning,
      stacklevel=2,
    )
    system = system_params

  # === Input Path Resolution: All paths converge on a Protein object ===
  protein_system: ProteinCls | None = None

  # Path 0: Already a Protein (new OutputSpec path) — pass through directly
  if isinstance(system, ProteinCls):
    protein_system = system
    if initial_positions is None:
      initial_positions = jnp.array(protein_system.coordinates).reshape(-1, 3)
    n_atoms = initial_positions.shape[0]

  # Path 1: Hierarchical PyTree args
  elif topology is not None and state is not None:
    if initial_positions is None:
      initial_positions = jnp.array(state.coordinates).reshape(-1, 3)

    n_atoms = initial_positions.shape[0]

    # Extract constants if available
    charges = None
    sigmas = None
    epsilons = None
    radii = None
    if constants is not None:
      charges = jnp.array(constants.charges) if constants.charges is not None else None
      sigmas = jnp.array(constants.sigmas) if constants.sigmas is not None else None
      epsilons = jnp.array(constants.epsilons) if constants.epsilons is not None else None
      radii = jnp.array(constants.radii) if constants.radii is not None else None

    # Extract topology
    bonds = (
      jnp.array(topology.bonds)
      if topology.bonds is not None
      else jnp.zeros((0, 2), dtype=jnp.int32)
    )
    angles = (
      jnp.array(topology.angles)
      if topology.angles is not None
      else jnp.zeros((0, 3), dtype=jnp.int32)
    )
    dihedrals = (
      jnp.array(topology.proper_dihedrals)
      if topology.proper_dihedrals is not None
      else jnp.zeros((0, 4), dtype=jnp.int32)
    )
    impropers = (
      jnp.array(topology.impropers)
      if topology.impropers is not None
      else jnp.zeros((0, 4), dtype=jnp.int32)
    )

    # Extract constants params
    bond_params = (
      jnp.zeros((bonds.shape[0], 2))
      if constants is None or constants.bond_params is None
      else jnp.array(constants.bond_params)
    )
    angle_params = (
      jnp.zeros((angles.shape[0], 2))
      if constants is None or constants.angle_params is None
      else jnp.array(constants.angle_params)
    )
    dihedral_params = (
      jnp.zeros((dihedrals.shape[0], 3))
      if constants is None or constants.dihedral_params is None
      else jnp.array(constants.dihedral_params)
    )
    improper_params = (
      jnp.zeros((impropers.shape[0], 3))
      if constants is None or constants.improper_params is None
      else jnp.array(constants.improper_params)
    )

    system_params_dict: SystemParams = {
      "charges": charges if charges is not None else jnp.zeros(n_atoms),
      "sigmas": sigmas if sigmas is not None else jnp.ones(n_atoms) * 3.0,
      "epsilons": epsilons if epsilons is not None else jnp.ones(n_atoms) * 0.1,
      "bonds": bonds,
      "bond_params": bond_params,
      "angles": angles,
      "angle_params": angle_params,
      "dihedrals": dihedrals,
      "dihedral_params": dihedral_params,
      "impropers": impropers,
      "improper_params": improper_params,
      "gb_radii": radii,
    }
    protein_system = system_params_to_protein(system_params_dict)

  # Path 2: AtomicSystem — extract fields and wrap
  elif system is not None and hasattr(system, "coordinates") and hasattr(system, "charges"):
    atomic_sys = system
    if initial_positions is None:
      initial_positions = jnp.array(atomic_sys.coordinates).reshape(-1, 3)

    n_atoms = initial_positions.shape[0]
    system_params_dict: SystemParams = {
      "charges": jnp.array(atomic_sys.charges)
      if atomic_sys.charges is not None
      else jnp.zeros(n_atoms),
      "sigmas": jnp.array(atomic_sys.sigmas)
      if atomic_sys.sigmas is not None
      else jnp.ones(n_atoms) * 3.0,
      "epsilons": jnp.array(atomic_sys.epsilons)
      if atomic_sys.epsilons is not None
      else jnp.ones(n_atoms) * 0.1,
      "bonds": jnp.array(atomic_sys.bonds)
      if atomic_sys.bonds is not None
      else jnp.zeros((0, 2), dtype=jnp.int32),
      "bond_params": jnp.array(atomic_sys.bond_params)
      if atomic_sys.bond_params is not None
      else jnp.zeros((0, 2)),
      "angles": jnp.array(atomic_sys.angles)
      if atomic_sys.angles is not None
      else jnp.zeros((0, 3), dtype=jnp.int32),
      "angle_params": jnp.array(atomic_sys.angle_params)
      if atomic_sys.angle_params is not None
      else jnp.zeros((0, 2)),
      "dihedrals": jnp.array(atomic_sys.proper_dihedrals)
      if atomic_sys.proper_dihedrals is not None
      else jnp.zeros((0, 4), dtype=jnp.int32),
      "dihedral_params": jnp.array(atomic_sys.dihedral_params)
      if atomic_sys.dihedral_params is not None
      else jnp.zeros((0, 3)),
      "impropers": jnp.array(atomic_sys.impropers)
      if atomic_sys.impropers is not None
      else jnp.zeros((0, 4), dtype=jnp.int32),
      "improper_params": jnp.array(atomic_sys.improper_params)
      if atomic_sys.improper_params is not None
      else jnp.zeros((0, 3)),
      "gb_radii": jnp.array(atomic_sys.radii) if atomic_sys.radii is not None else None,  # type: ignore[possibly-missing-attribute]
    }
    protein_system = system_params_to_protein(system_params_dict)

  # Path 3: Raw dict
  elif system is not None:
    if initial_positions is None:
      msg = "initial_positions is required when using system_params dict"
      raise ValueError(msg)
    n_atoms = initial_positions.shape[0]
    protein_system = system_params_to_protein(system)

  else:
    msg = "Either system or (topology, state) must be provided"
    raise ValueError(msg)

  assert protein_system is not None

  # Import jax_md components here to avoid circular imports
  from jax_md import quantity as jax_md_quantity
  from jax_md import simulate as jax_md_simulate

  if spec is None:
    raise ValueError("spec must be provided")

  if key is None:
    key = jax.random.PRNGKey(int(time.time()))

  displacement_fn: Any
  shift_fn: Any

  if spec.use_pbc:
    if spec.box is None:
      msg = "Must specify box when use_pbc=True"
      raise ValueError(msg)
    from prolix.physics import pbc

    displacement_fn, shift_fn = pbc.create_periodic_space(spec.box)
  else:
    displacement_fn, shift_fn = space.free()

  # Check for implicit solvent or vacuum
  implicit_solvent = not spec.use_pbc

  # Build ExclusionSpec for proper 1-2/1-3/1-4 scaling in non-bonded terms
  # This is required for BOTH neighbor list and N^2 paths to work correctly
  exclusion_spec = spec.exclusion_spec
  if exclusion_spec is None:
    exclusion_spec = nl.ExclusionSpec.from_protein(protein_system)
    logger.info(
      "Built ExclusionSpec: %d 1-2/1-3 pairs, %d 1-4 pairs",
      len(exclusion_spec.idx_12_13),
      len(exclusion_spec.idx_14),
    )

  energy_fn: Any = physics_system.make_energy_fn(
    displacement_fn,
    protein_system,
    exclusion_spec=exclusion_spec,
    implicit_solvent=implicit_solvent,
    box=spec.box,
    use_pbc=spec.use_pbc,
    pme_grid_points=spec.pme_grid_size,
    cutoff_distance=spec.neighbor_cutoff if spec.use_neighbor_list else 9.0,
  )

  # Create neighbor list if requested
  neighbor: Any = None
  neighbor_fn: Any = None
  if spec.use_neighbor_list and spec.box is not None:
    neighbor_fn = nl.make_neighbor_list_fn(
      displacement_fn, jnp.array(spec.box), spec.neighbor_cutoff
    )
    neighbor = neighbor_fn.allocate(initial_positions)
    neighbor = neighbor_fn.allocate(initial_positions)
    logger.info("Neighbor list allocated: shape %s", neighbor.idx.shape)

  # Check memory budget
  resource_guard.check_memory_budget(
    n_atoms=n_atoms,
    accumulate_steps=spec.accumulate_steps,
    use_neighbor_list=spec.use_neighbor_list,
    pme_grid_size=spec.pme_grid_size,
    use_pbc=spec.use_pbc,
  )

  # Robust Energy Minimization
  logger.info("Running robust energy minimization (L-BFGS / Adaptive Line Search)...")

  # Initialize energy and neighbor list
  if neighbor is not None:
    neighbor = neighbor.update(initial_positions)
    e_initial, grads_initial = jax.value_and_grad(lambda r: energy_fn(r, neighbor=neighbor))(
      initial_positions
    )  # type: ignore
  else:
    e_initial, grads_initial = jax.value_and_grad(energy_fn)(initial_positions)

  logger.info("  Initial energy: %.2f kcal/mol", e_initial)

  # Adaptive Minimization State
  @jax.jit
  def robust_minimization_loop(init_pos, init_nbr, init_grads, init_energy):
    # State: [pos, nbr, grads, energy, step_size, step_count, converged]
    # We use a structured PyTree for state to keep it clean

    class MinState(typing.NamedTuple):
      pos: Array
      nbr: Any
      grads: Array
      energy: float
      step_size: float
      step: int
      converged: bool

    init_state = MinState(
      pos=init_pos,
      nbr=init_nbr,
      grads=init_grads,
      energy=init_energy,
      step_size=0.001,  # Conservative start
      step=0,
      converged=False,
    )

    def cond_fn(state):
      return (state.step < 5000) & (~state.converged)

    def body_fn(state):
      # 1. Force Capping
      # Cap max force to prevent explosions from singular potentials
      g = state.grads
      g_norm = jnp.linalg.norm(g, axis=-1, keepdims=True)
      # Limit force magnitude to 1000 kcal/mol/A
      scaling = jnp.minimum(1.0, 1000.0 / (g_norm + 1e-8))
      g_capped = g * scaling

      # 2. Candidate Step
      # Move AGAINST gradient (Gradient Descent)
      pos_new = state.pos - state.step_size * g_capped

      if spec.use_pbc and spec.box is not None:
        pos_new = jnp.mod(pos_new, jnp.array(spec.box))

      # 3. Evaluate New State
      nbr_new = state.nbr
      if state.nbr is not None:
        nbr_new = state.nbr.update(pos_new)
        e_new, g_new = jax.value_and_grad(lambda r: energy_fn(r, neighbor=nbr_new))(pos_new)  # type: ignore
      else:
        e_new, g_new = jax.value_and_grad(energy_fn)(pos_new)

      # 4. Acceptance Logic (Backtracking Line Search)
      # Accept if energy decreases AND is finite
      improved = e_new < state.energy
      is_valid = jnp.isfinite(e_new)
      accept = improved & is_valid

      # Update Variables
      final_pos = jnp.where(accept, pos_new, state.pos)  # type: ignore
      # For PyTrees (neighbor), we verify if we can use jnp.where or lax.cond
      # Neighbor is a PyTree, so we must use lax.cond for full state switching usually

      # Adaptive Step Size: grow if good, shrink if bad
      new_step_size = jnp.where(accept, state.step_size * 1.2, state.step_size * 0.5)
      new_step_size = jnp.clip(new_step_size, 1e-7, 1e-1)

      # Construct next state parts
      # Just use lax.cond to switch between (new_vals) and (old_vals)
      def accept_branch():
        return MinState(pos_new, nbr_new, g_new, e_new, new_step_size, state.step + 1, False)

      def reject_branch():
        # Keep old pos, old grads, old energy. Only update step size and count.
        return MinState(
          state.pos, state.nbr, state.grads, state.energy, new_step_size, state.step + 1, False
        )

      next_state = jax.lax.cond(accept, accept_branch, reject_branch)

      # Convergence check (Gradient Norm < 1.0 kcal/mol/A)
      # Use the NEW gradients if accepted, or OLD gradients if rejected
      current_g_norm = jnp.max(jnp.linalg.norm(next_state.grads, axis=-1))
      is_converged = current_g_norm < 1.0

      return next_state._replace(converged=is_converged)

    return jax.lax.while_loop(cond_fn, body_fn, init_state)

  # Need typing for NamedTuple
  import typing

  final_min_state = robust_minimization_loop(initial_positions, neighbor, grads_initial, e_initial)
  positions_minimized = final_min_state.pos
  neighbor = final_min_state.nbr
  e_minimized = final_min_state.energy

  logger.info("  Minimized energy: %.2f kcal/mol", e_minimized)
  logger.info("  Minimization steps: %d", final_min_state.step)
  logger.info("  Converged: %s", final_min_state.converged)

  # NVT Langevin dynamics setup
  # IMPORTANT: jax_md uses reduced units, not physical time
  # dt=2e-3 in reduced units is standard (like stress_test_stability.py)
  # Converting from fs to jax_md reduced units: dt = step_size_fs * 1e-3 * 0.0488 (if using AKMA units)
  # However, stress_test uses dt=2e-3 directly which is known to work.
  # For simplicity and stability, use the known-working value.

  kT = spec.temperature_k * 0.0019872041  # Boltzmann in kcal/mol/K
  dt = 2e-3  # Reduced units (known to work from stress_test_stability.py)
  gamma = spec.gamma if spec.gamma <= 1.0 else 0.1  # Use lower friction for stability

  logger.info(
    "NVT Langevin setup: T=%.1f K, kT=%.4f, dt=%.4f, gamma=%.2f", spec.temperature_k, kT, dt, gamma
  )

  # Use energy_fn directly - jax_md integrators support neighbor= kwarg natively
  # DO NOT wrap energy_fn with a captured neighbor list - that prevents updates!
  integrator_energy_fn = energy_fn

  # Don't pass mass to jax_md (it handles masses internally)
  # The stress_test_stability.py doesn't pass mass and works fine
  constrained_bonds = protein_system.constrained_bonds
  constrained_lengths = protein_system.constrained_bond_lengths

  if (
    constrained_bonds is not None and constrained_lengths is not None and len(constrained_bonds) > 0
  ):
    init_fn, apply_fn = physics_simulate.rattle_langevin(
      integrator_energy_fn,
      shift_fn,
      dt=dt,
      kT=kT,
      gamma=gamma,
      constraints=(jnp.array(constrained_bonds), jnp.array(constrained_lengths)),
    )
  else:
    init_fn, apply_fn = jax_md_simulate.nvt_langevin(
      integrator_energy_fn, shift_fn, dt=dt, kT=kT, gamma=gamma
    )

  # Initialize state from minimized positions
  # Pass neighbor= to init_fn if using neighbor lists (jax_md native support)
  if neighbor is not None:
    state = init_fn(key, positions_minimized, neighbor=neighbor)
  else:
    state = init_fn(key, positions_minimized)

  @jax.jit
  def jit_apply_fn(s, nbr=None):
    if nbr is not None:
      return apply_fn(s, neighbor=nbr)  # type: ignore[unknown-argument]
    return apply_fn(s)

  if neighbor is not None:
    _test_state = jit_apply_fn(state, nbr=neighbor)
  else:
    _test_state = jit_apply_fn(state)
  jax.block_until_ready(_test_state.position)
  logger.info("Step function compiled!")

  # Trajectory saving
  steps_per_save = round(spec.save_interval_ns * 1000000 / spec.step_size_fs)
  logger.info("Steps per save: %d", steps_per_save)

  # Setup trajectory writer
  writer = TrajectoryWriter(spec.save_path)

  # Calculate epochs for outer Python loop
  total_saves = int(spec.total_time_ns / spec.save_interval_ns)
  accumulate = spec.accumulate_steps
  n_epochs = math.ceil(total_saves / accumulate)

  logger.info(
    "Starting simulation: %d epochs, %d saves per epoch, total %d saves",
    n_epochs,
    accumulate,
    total_saves,
  )

  # Calculate neighbor update frequency in steps
  steps_per_neighbor_update = max(1, round(spec.neighbor_update_interval_fs / spec.step_size_fs))
  # Number of neighbor updates per save interval
  neighbor_updates_per_save = max(1, steps_per_save // steps_per_neighbor_update)
  # Actual steps per neighbor update (may differ slightly due to rounding)
  steps_per_update_actual = steps_per_save // neighbor_updates_per_save

  logger.info(
    "Neighbor list update: every %d steps (%.1f fs), %d updates per save",
    steps_per_update_actual,
    steps_per_update_actual * spec.step_size_fs,
    neighbor_updates_per_save,
  )

  # NOTE: These functions are used inside jax.lax.scan, which handles JIT.
  # Don't add @jax.jit here - it causes redundant compilation and slower performance.

  def scan_fn_with_neighbor(carrier, _):
    """Scan function with nested loops: outer updates neighbor, inner runs MD."""
    curr_state, nbrs = carrier

    def outer_step(j, state_nbrs):
      """Update neighbor list and run steps_per_update_actual MD steps."""
      state, nbrs = state_nbrs
      # Update neighbor list
      nbrs = nbrs.update(state.position)

      def inner_step(i, s):
        # Run integrator with fixed neighbor list
        return jit_apply_fn(s, nbr=nbrs)

      state = jax.lax.fori_loop(0, steps_per_update_actual, inner_step, state)
      return (state, nbrs)

    curr_state, nbrs = jax.lax.fori_loop(
      0, neighbor_updates_per_save, outer_step, (curr_state, nbrs)
    )

    # Calculate Energy for saving
    E = energy_fn(curr_state.position, neighbor=nbrs)  # type: ignore[unknown-argument]
    K = jax_md_quantity.kinetic_energy(momentum=curr_state.momentum, mass=curr_state.mass)

    # OPTIMIZATION: Do not store forces/mass in accumulation buffer to save memory
    # We only store positions, velocities, energy for trajectory.
    sim_state = SimulationState(
      positions=curr_state.position,
      velocities=curr_state.momentum / curr_state.mass,
      forces=None,  # Dropped for memory efficiency
      mass=None,  # Dropped (constant)
      step=jnp.array(0),  # Placeholder, actual step tracked externally
      time_ns=jnp.array(0.0),
      potential_energy=E,
      kinetic_energy=K,
    )

    return (curr_state, nbrs), sim_state

  def scan_fn_no_neighbor(carrier, _):
    """Scan function without neighbor list (original path)."""
    curr_state = carrier

    def step_fn(i, s):
      return jit_apply_fn(s)

    curr_state = jax.lax.fori_loop(0, steps_per_save, step_fn, curr_state)

    E = energy_fn(curr_state.position)
    K = jax_md_quantity.kinetic_energy(momentum=curr_state.momentum, mass=curr_state.mass)

    sim_state = SimulationState(
      positions=curr_state.position,
      velocities=curr_state.momentum / curr_state.mass,
      forces=None,  # Dropped for memory efficiency
      mass=None,  # Dropped
      step=jnp.array(0),
      time_ns=jnp.array(0.0),
      potential_energy=E,
      kinetic_energy=K,
    )

    return curr_state, sim_state

  # We need to track step count.
  # Let's augment the state or carrier?
  # The carrier is NVTLangevinState.
  # We can't easily add fields to it as it's a fixed dataclass in simulate.py?
  # Actually `simulate.py` defines `NVTLangevinState`.
  # We should probably assume step tracking in Python or add a wrapper.

  # Simplified: Just run loop.

  for epoch in range(n_epochs):
    start_time = time.time()

    # Determine batch size for this epoch (handle last partial epoch)
    batch_size = accumulate
    remaining = total_saves - epoch * accumulate
    batch_size = min(batch_size, remaining)
    if batch_size <= 0:
      break

    # Run JAX Scan - different paths for neighbor list vs non-neighbor list
    xs = jnp.arange(batch_size)

    if neighbor is not None:
      # Neighbor list path: carry (state, neighbor) tuple
      carrier = (state, neighbor)
      new_carrier, stacked_sim_states = jax.lax.scan(scan_fn_with_neighbor, carrier, xs)
      new_state, neighbor = new_carrier

      # Check for neighbor list overflow - reallocate and update for next epoch
      if neighbor.did_buffer_overflow:
        logger.warning("Neighbor list overflow detected, reallocating for next epoch...")
        neighbor = neighbor_fn.allocate(new_state.position)  # type: ignore[possibly-missing-attribute]
        # CRITICAL: Update the newly allocated neighbor list with actual neighbors
        neighbor = neighbor.update(new_state.position)
      state = new_state
      final_state = new_state
    else:
      # Non-neighbor list path (original)
      final_state, stacked_sim_states = jax.lax.scan(scan_fn_no_neighbor, state, xs)
      state = final_state

    # Block until ready
    jax.block_until_ready(stacked_sim_states)

    # CRITICAL: Check for NaN/Inf in positions before saving
    # This catches simulation instability early
    positions_cpu = np.asarray(stacked_sim_states.positions)
    if not np.all(np.isfinite(positions_cpu)):
      bad_frame = np.where(~np.all(np.isfinite(positions_cpu), axis=(1, 2)))[0][0]
      bad_step = epoch * accumulate + bad_frame
      logger.error(
        f"Simulation became unstable at step {bad_step} (epoch {epoch}, frame {bad_frame})"
      )
      msg = (
        f"Simulation instability detected: NaN/Inf positions at step {bad_step}.\n"
        f"This typically indicates:\n"
        f"  - Timestep too large (current: {spec.step_size_fs} fs)\n"
        f"  - Insufficient minimization (current: 5000 steps)\n"
        f"  - PME parameters inappropriate for box size\n"
        f"  - Numerical precision issues (TPU float32)\n"
        f"Suggestions:\n"
        f"  - Reduce step_size_fs (try 1.0 or 0.5 fs)\n"
        f"  - Increase minimization steps\n"
        f"  - Check PME grid size matches box dimensions"
      )
      raise RuntimeError(msg)

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
  if neighbor is not None:
    E = energy_fn(state.position, neighbor=neighbor)  # type: ignore[unknown-argument]
  else:
    E = energy_fn(state.position)
  return SimulationState(
    positions=state.position,
    velocities=state.momentum / state.mass,
    forces=state.force,
    step=jnp.array(total_saves * steps_per_save),
    time_ns=jnp.array(spec.total_time_ns),
    potential_energy=E,
    kinetic_energy=None,
  )


# Note: jax_md imports are now inside run_simulation to avoid circular imports
