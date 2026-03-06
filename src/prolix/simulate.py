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
from jax_md import minimize as jax_md_minimize, space, util
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

# AKMA unit system constants for JAX-MD
# JAX-MD is unitless. We operate in AMBER/AKMA units:
#   distance: Å, energy: kcal/mol, mass: g/mol (Daltons)
# Derived time unit: τ = sqrt(Da · Å² / (kcal/mol)) ≈ 48.888 fs
AKMA_TIME_UNIT_FS = 48.88821291839  # 1 AKMA time unit in femtoseconds
BOLTZMANN_KCAL = 0.0019872041       # kB in kcal/(mol·K)


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
      coords = protein_system.coordinates
      if coords.ndim == 2:
        # Flat format (CoordFormat.Full) — coordinates match parameter arrays
        initial_positions = jnp.array(coords)
      else:
        # Atom37 format — extract valid atoms via mask
        logger.warning(
          "Protein has Atom37 coordinates (ndim=3). For MD, use "
          "CoordFormat.Full to ensure coordinates match parameter arrays. "
          "Extracting valid atoms from mask, but this may mismatch parameter counts."
        )
        mask = protein_system.atom_mask
        flat_coords = coords.reshape(-1, 3)
        flat_mask = mask.reshape(-1) if mask.ndim > 1 else mask
        initial_positions = jnp.array(flat_coords[flat_mask > 0.5])
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

  # Assign GB radii if implicit solvent is requested but radii are missing.
  # This happens when parse_structure(OutputSpec(parameterize_md=True)) is used,
  # which populates charges/sigmas/epsilons but not radii/scaled_radii.
  if implicit_solvent and protein_system.radii is None:
    _atom_names = getattr(protein_system, "atom_names", None)
    _bonds = getattr(protein_system, "bonds", None)
    if _atom_names is not None and _bonds is not None and len(_atom_names) > 0:
      from proxide import assign_mbondi2_radii, assign_obc2_scaling_factors

      _radii = assign_mbondi2_radii(list(_atom_names), _bonds)
      _scaled = assign_obc2_scaling_factors(list(_atom_names))
      protein_system = dataclasses.replace(
        protein_system, radii=jnp.asarray(_radii), scaled_radii=jnp.asarray(_scaled)
      )
      logger.info(
        "Assigned mbondi2 radii to %d atoms (implicit solvent fallback)", len(_radii)
      )
    else:
      logger.warning(
        "Implicit solvent requested but cannot assign GB radii "
        "(atom_names=%s, bonds=%s). GB energy will use sigma/2 fallback.",
        _atom_names is not None,
        _bonds is not None,
      )

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

  # ── Per-atom masses (needed for both FIRE minimizer and NVT integrator) ──
  # Must be computed BEFORE minimization so FIRE can use them.
  raw_masses = protein_system.masses
  if raw_masses is not None:
    masses = jnp.array(raw_masses)
    if jnp.all(masses == 0):
      logger.warning("All masses are zero, will derive from elements")
      raw_masses = None  # trigger element-based derivation below

  if raw_masses is None:
    # Derive masses from element symbols if available
    from prolix.constants import masses_from_elements, DEFAULT_MASS
    elements = getattr(protein_system, "elements", None)
    if elements is not None and len(elements) == n_atoms:
      mass_list = masses_from_elements(list(elements))
      masses = jnp.array(mass_list, dtype=jnp.float32)
      logger.info("Derived masses from %d element symbols (range: %.1f–%.1f Da)",
                   n_atoms, float(jnp.min(masses)), float(jnp.max(masses)))
    else:
      logger.warning("No masses or elements found, defaulting to carbon mass (%.1f Da)", DEFAULT_MASS)
      masses = jnp.ones(n_atoms) * DEFAULT_MASS

  # ── Energy Minimization (FIRE) ──
  # Uses jax-md's built-in FIRE (Fast Inertial Relaxation Engine).
  # AMBER units: energy kcal/mol, distance Å, mass amu, time ps.
  # dt_start=0.0001 ps (0.1 fs), dt_max=0.001 ps (1.0 fs) — conservative
  # for systems with potentially extreme initial gradients.
  logger.info("Running FIRE energy minimization...")

  # Initialize energy and neighbor list
  if neighbor is not None:
    neighbor = neighbor.update(initial_positions)
    e_initial = energy_fn(initial_positions, neighbor=neighbor)
  else:
    e_initial = energy_fn(initial_positions)

  logger.info("  Initial energy: %.2f kcal/mol", e_initial)

  # Validate initial energy is finite before proceeding
  if not jnp.isfinite(e_initial):
    msg = (
      f"Initial energy is non-finite ({float(e_initial):.2f} kcal/mol). "
      f"This typically indicates severe steric clashes or unparameterized atoms. "
      f"Common causes:\n"
      f"  - Hydrogen addition placed H on top of existing atoms (use add_hydrogens=False)\n"
      f"  - Unparameterized atoms have sigma=0.0 (check force field coverage)\n"
      f"  - Crystal structure has severe clashes requiring preprocessing\n"
      f"Run diagnose_inf_energy.py for detailed energy decomposition."
    )
    raise RuntimeError(msg)

  # FIRE minimizer setup
  # NOTE: FIRE does NOT internally update neighbor lists. For our current
  # implicit solvent N² path (no neighbor list), this is fine. For future
  # PBC/explicit solvent support with neighbor lists, FIRE would need a
  # wrapper that calls nbr.update() every N steps.

  def _make_fire_energy(soft_core_lambda=None):
    """Create energy function for FIRE, optionally with soft-core LJ."""
    if neighbor is not None and soft_core_lambda is not None:
      return lambda r: energy_fn(r, neighbor=neighbor, soft_core_lambda=soft_core_lambda)
    elif neighbor is not None:
      return lambda r: energy_fn(r, neighbor=neighbor)
    elif soft_core_lambda is not None:
      return lambda r: energy_fn(r, soft_core_lambda=soft_core_lambda)
    else:
      return energy_fn

  def _make_capped_fire_apply(fire_apply_fn, force_cap):
    """Create force-capped FIRE apply function."""
    def capped_apply(state, **kwargs):
      f = state.force
      f_norm = jnp.linalg.norm(f, axis=-1, keepdims=True)
      cap = jnp.minimum(1.0, force_cap / (f_norm + 1e-8))
      f_capped = f * cap
      f_safe = jnp.where(jnp.isfinite(f_capped), f_capped, 0.0)
      state = state._replace(force=f_safe)
      return fire_apply_fn(state, **kwargs)
    return capped_apply

  # ── Dynamic step-size initialization (#2179) ──
  # Set dt_start based on initial gradient magnitude so that
  # max_grad * dt_start < 0.001 Å displacement.
  initial_grads = jax.grad(_make_fire_energy())(initial_positions)
  max_grad = float(jnp.max(jnp.linalg.norm(initial_grads, axis=-1)))
  dt_start = float(jnp.clip(0.001 / (max_grad + 1e-8), 1e-7, 0.001))
  dt_max = min(dt_start * 10.0, 0.01)
  logger.info("  Dynamic FIRE dt: dt_start=%.2e ps, dt_max=%.2e ps (max_grad=%.2e)",
              dt_start, dt_max, max_grad)

  # ── Staged Minimization (#2183) ──
  # Progressive softening resolves extreme steric clashes from crystal structures.
  # Stage 1: Soft-core λ=0.1, cap=10 kcal/mol/Å (500 steps) — gentle decompression
  # Stage 2: Soft-core λ=0.5, cap=100 (500 steps) — half-strength LJ
  # Stage 3: Soft-core λ=0.9, cap=1000 (1000 steps) — near-full LJ
  # Stage 4: Standard energy (NOT soft-core), no cap (≤3000 steps) — convergence
  #
  # IMPORTANT: Stage 4 uses the standard energy_fn, not soft-core with λ=1,
  # because soft-core with λ=1 recovers the singularity at r=0.

  stages = [
    {"name": "Stage 1 (λ=0.1)", "lambda": 0.1, "force_cap": 10.0,   "steps": 500},
    {"name": "Stage 2 (λ=0.5)", "lambda": 0.5, "force_cap": 100.0,  "steps": 500},
    {"name": "Stage 3 (λ=0.9)", "lambda": 0.9, "force_cap": 1000.0, "steps": 1000},
    {"name": "Stage 4 (full)",  "lambda": None, "force_cap": None,   "steps": 3000},
  ]

  current_positions = initial_positions

  for stage in stages:
    sc_lam = stage["lambda"]
    force_cap = stage["force_cap"]
    n_steps = stage["steps"]

    # Create energy function for this stage
    stage_energy_fn = _make_fire_energy(soft_core_lambda=sc_lam)

    # Create FIRE instance for this stage
    stage_init_fn, stage_apply_fn = jax_md_minimize.fire_descent(
      stage_energy_fn,
      shift_fn,
      dt_start=dt_start,
      dt_max=dt_max,
    )

    # Wrap with force capping if specified
    if force_cap is not None:
      stage_step_fn = _make_capped_fire_apply(stage_apply_fn, force_cap)
    else:
      # Stage 4: no cap, but still sanitize NaN
      def _nan_safe_apply(state, _apply=stage_apply_fn, **kwargs):
        f = state.force
        f_safe = jnp.where(jnp.isfinite(f), f, 0.0)
        state = state._replace(force=f_safe)
        return _apply(state, **kwargs)
      stage_step_fn = _nan_safe_apply

    # Initialize FIRE from current positions
    stage_state = stage_init_fn(current_positions, mass=masses)

    # Run FIRE loop
    @jax.jit
    def _run_stage(state, _step=stage_step_fn, _n=n_steps):
      def body_fn(i, s):
        return _step(s)
      return jax.lax.fori_loop(0, _n, body_fn, state)

    final_state = _run_stage(stage_state)
    current_positions = final_state.position

    # Log stage results
    stage_e = float(stage_energy_fn(current_positions))
    logger.info("  %s: E=%.2f kcal/mol (%d steps, cap=%s)",
                stage["name"], stage_e, n_steps,
                f"{force_cap:.0f}" if force_cap is not None else "none")

    # After stages with soft-core, ramp dt_start up for next stage
    dt_start = min(dt_start * 2.0, 0.001)
    dt_max = min(dt_start * 10.0, 0.01)

  positions_minimized = current_positions

  # Evaluate final energy with standard (non-soft-core) energy function
  fire_energy_fn = _make_fire_energy()  # standard, no soft-core
  e_minimized = fire_energy_fn(positions_minimized)

  logger.info("  Final minimized energy: %.2f kcal/mol", e_minimized)

  # NVT Langevin dynamics setup
  # Convert physical quantities to AKMA reduced units for JAX-MD.
  # JAX-MD integrators are unitless — they see dt, kT, gamma as plain floats.
  # Our energy function outputs kcal/mol and positions are in Å,
  # so we must use AKMA unit conversions to keep everything consistent.
  kT = spec.temperature_k * BOLTZMANN_KCAL               # kcal/mol
  dt = spec.step_size_fs / AKMA_TIME_UNIT_FS              # reduced time units
  gamma_reduced = spec.gamma * AKMA_TIME_UNIT_FS * 1e-3   # convert 1/ps → 1/τ

  logger.info(
    "NVT Langevin setup: T=%.1f K, kT=%.4f kcal/mol, "
    "dt=%.6f τ (%.1f fs), gamma=%.5f /τ (%.1f /ps)",
    spec.temperature_k, kT, dt, spec.step_size_fs,
    gamma_reduced, spec.gamma,
  )

  # Use energy_fn directly - jax_md integrators support neighbor= kwarg natively
  # DO NOT wrap energy_fn with a captured neighbor list - that prevents updates!
  integrator_energy_fn = energy_fn

  # (Masses already computed above, before FIRE minimization)

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
      gamma=gamma_reduced,
      constraints=(jnp.array(constrained_bonds), jnp.array(constrained_lengths)),
    )
  else:
    init_fn, apply_fn = jax_md_simulate.nvt_langevin(
      integrator_energy_fn, shift_fn, dt=dt, kT=kT, gamma=gamma_reduced
    )

  # Initialize state from minimized positions with per-atom masses
  # Pass neighbor= to init_fn if using neighbor lists (jax_md native support)
  if neighbor is not None:
    state = init_fn(key, positions_minimized, mass=masses, neighbor=neighbor)
  else:
    state = init_fn(key, positions_minimized, mass=masses)

  @jax.jit
  def jit_apply_fn(s, nbr=None):
    if nbr is not None:
      return apply_fn(s, neighbor=nbr)  # type: ignore[unknown-argument]
    return apply_fn(s)

  # ── Gentle-start equilibration (#2184) ──
  # Timestep ramping to catch residual instabilities from minimization.
  # Phase 1: 100 steps at dt/20 (0.1 fs if production is 2 fs)
  # Phase 2: 500 steps at dt/2 (1.0 fs)
  # Phase 3: Production dt (2.0 fs) — starts the main loop
  warmup_phases = [
    {"name": "warmup dt/20", "dt_scale": 1.0/20.0, "steps": 100},
    {"name": "warmup dt/2",  "dt_scale": 0.5,       "steps": 500},
  ]

  logger.info("Gentle-start equilibration...")
  for phase in warmup_phases:
    warmup_dt = dt * phase["dt_scale"]
    warmup_kT = kT  # same temperature

    # Create a temporary integrator at reduced timestep
    warmup_init, warmup_apply = jax_md_simulate.nvt_langevin(
      integrator_energy_fn, shift_fn, dt=warmup_dt, kT=warmup_kT, gamma=gamma_reduced
    )

    # Initialize from current state positions
    warmup_key = jax.random.fold_in(key, hash(phase["name"]))
    if neighbor is not None:
      warmup_state = warmup_init(warmup_key, state.position, mass=masses, neighbor=neighbor)
    else:
      warmup_state = warmup_init(warmup_key, state.position, mass=masses)

    # Run warmup phase
    @jax.jit
    def _warmup_loop(ws, _apply=warmup_apply, _n=phase["steps"]):
      def body(i, s):
        return _apply(s)
      return jax.lax.fori_loop(0, _n, body, ws)

    warmup_state = _warmup_loop(warmup_state)

    # NaN check
    has_nan = bool(jnp.any(~jnp.isfinite(warmup_state.position)))
    if has_nan:
      logger.warning("  %s: NaN detected after %d steps! Reverting to minimized positions.",
                     phase["name"], phase["steps"])
      # Revert to minimized positions — don't propagate NaN
      if neighbor is not None:
        state = init_fn(key, positions_minimized, mass=masses, neighbor=neighbor)
      else:
        state = init_fn(key, positions_minimized, mass=masses)
      break
    else:
      # Update state with warmup results, reinitialize with production integrator
      if neighbor is not None:
        state = init_fn(key, warmup_state.position, mass=masses, neighbor=neighbor)
      else:
        state = init_fn(key, warmup_state.position, mass=masses)
      logger.info("  %s: %d steps OK (dt=%.4e τ)", phase["name"], phase["steps"], warmup_dt)

  # Compile and test the production step function
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

    # Fixup step/time in saved states (tracked externally, not in scan)
    base_save_idx = epoch * accumulate
    frame_steps = jnp.array(
      [(base_save_idx + i + 1) * steps_per_save for i in range(batch_size)]
    )
    frame_times = frame_steps * spec.step_size_fs * 1e-6  # fs → ns
    stacked_sim_states = eqx.tree_at(
      lambda s: s.step, stacked_sim_states, frame_steps
    )
    stacked_sim_states = eqx.tree_at(
      lambda s: s.time_ns, stacked_sim_states, frame_times
    )

    # Write to disk
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


def simulate_frames(
  protein: "Protein",
  r_init: Array,
  n_steps: int,
  n_frames: int,
  key: Array,
) -> Array:
  """Production simulation loop isolating the core integrator logic.

  Process:
  1.  **Initialize**: Setup energy function and random keys.
  2.  **Integrator**: Initialize NVT Langevin integrator (with RATTLE if constrained).
  3.  **Simulate**: Execute production loop using `jax.lax.scan`.

  Args:
    protein: Protein object with physical parameters.
    r_init: Initial positions (N, 3).
    n_steps: Number of integration steps per frame.
    n_frames: Number of frames to generate.
    key: JAX random PRNG key.

  Returns:
    (n_frames, n_atoms, 3) array of positions.
  """
  from jax_md import simulate as jax_md_simulate

  # Physics parameters - use defaults consistent with run_simulation
  temperature_k = 300.0
  step_size_fs = 2.0
  gamma = 1.0

  kT = temperature_k * BOLTZMANN_KCAL
  dt = step_size_fs / AKMA_TIME_UNIT_FS
  gamma_reduced = gamma * AKMA_TIME_UNIT_FS * 1e-3

  displacement_fn, shift_fn = space.free()

  # Build ExclusionSpec if not present in protein
  exclusion_spec = nl.ExclusionSpec.from_protein(protein)

  # Energy function (implicit solvent by default for extracted loop)
  energy_fn = physics_system.make_energy_fn(
    displacement_fn,
    protein,
    exclusion_spec=exclusion_spec,
    implicit_solvent=True,
  )

  # Derived masses (matches run_simulation logic)
  n_atoms = r_init.shape[0]
  raw_masses = protein.masses
  if raw_masses is not None:
    masses = jnp.array(raw_masses)
    if jnp.all(masses == 0):
      raw_masses = None

  if raw_masses is None:
    # Element-based derivation fallback
    from prolix.constants import masses_from_elements, DEFAULT_MASS
    elements = getattr(protein, "elements", None)
    if elements is not None and len(elements) == n_atoms:
      masses = jnp.array(masses_from_elements(list(elements)), dtype=jnp.float32)
    else:
      masses = jnp.ones(n_atoms) * DEFAULT_MASS

  # Integrator setup (RATTLE support)
  constrained_bonds = protein.constrained_bonds
  constrained_lengths = protein.constrained_bond_lengths

  if (
    constrained_bonds is not None and constrained_lengths is not None and len(constrained_bonds) > 0
  ):
    init_fn, apply_fn = physics_simulate.rattle_langevin(
      energy_fn,
      shift_fn,
      dt=dt,
      kT=kT,
      gamma=gamma_reduced,
      constraints=(jnp.array(constrained_bonds), jnp.array(constrained_lengths)),
    )
  else:
    init_fn, apply_fn = jax_md_simulate.nvt_langevin(
      energy_fn, shift_fn, dt=dt, kT=kT, gamma=gamma_reduced
    )

  state = init_fn(key, r_init, mass=masses)

  @jax.jit
  def scan_fn(carrier, _):
    def step_fn(i, s):
      return apply_fn(s)

    carrier = jax.lax.fori_loop(0, n_steps, step_fn, carrier)
    return carrier, carrier.position

  _, trajectory = jax.lax.scan(scan_fn, state, jnp.arange(n_frames))
  return trajectory


def batched_simulate_frames(
  protein: "Protein",
  r_init: Array,
  n_steps: int,
  n_frames: int,
  keys: Array,
) -> Array:
  """Vmapped version of simulate_frames for ensemble simulations."""
  return jax.vmap(simulate_frames, in_axes=(None, 0, None, None, 0))(
    protein, r_init, n_steps, n_frames, keys
  )
