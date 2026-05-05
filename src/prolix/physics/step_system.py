"""Step system: reusable integrator primitives for composition into custom integrators.

This module provides a three-layer architecture for building integrators:

**Layer 1: Constraint Kinematics (ConstraintDOFMask)**
Formal constraint model and DOF decomposition for rigid bodies (SETTLE water molecules).
Reference: constraints.py, Phase 1.1.

**Layer 2: Step Primitives (V_Step, A_Step, O_Step, CSVR_Step, etc.)**
Pure, JIT-safe integrator steps that operate on system state (IntegratorState).
Each step is a class with an `apply` method implementing a single integration operation.
Reference: Phase 1.2.

**Layer 3: Integrator Sequences (StepSequence, step_sequences registry)**
Ordered lists of steps composing integrator variants (BAOAB_LANGEVIN, BAOAB_CSVR_NPT, etc.).
Each sequence captures the integrator structure and shared parameters.
Reference: Phase 1.3, ADR-005 v2.0.

**v1.0 Scope**: This release ships with BAOAB_LANGEVIN and BAOAB_CSVR_NPT integrators.
LFMiddle (Phase 3) and NHC thermostat (future work) are deferred to v1.1 pending
architectural changes. See oracle recommendation Path B and `.agent/docs/RELEASE_DECISION_v1.0.md`.

**Design Principles**:
- Each step is a class with an `apply` method
- `apply` is pure: no side effects, no state mutation, no control flow
- Steps work with free DOF subspace: constraint projection handled externally or within
- Steps are JIT-compiled without branches or Python loops (JAX lax.* loops OK)
- Sequences are immutable (frozen dataclasses) and validated at module load

**Why Three Layers?**
1. Orthogonal design: Constraints (Layer 1) are independent from steps (Layer 2) and sequences (Layer 3)
2. Composability: Sequences compose steps; steps can be reused in different sequences
3. Clarity: Each layer has one responsibility (kinematics, primitives, orchestration)
4. Extensibility: New constraints, steps, or sequences can be added without modifying others
5. Testing: Each layer validated independently, then composed

**Example: Building a Custom Integrator**
```python
# Get a sequence
sequence = make_sequence('baoab_langevin', dt=0.5, gamma=1.0, kT=2.479)

# Compose steps
for step_name in sequence.steps:
    step = make_step(step_name, **sequence.parameters)
    state = step.apply(state, constraint_dofs=constraint_dofs, **sequence.parameters)
```

**Reference**:
- Leimkuhler, B., & Shang, X. (2015). Adaptive thermostats for noisy gradient systems.
  SIAM Journal on Numerical Analysis, 54(2), 721-743.
- Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling through velocity rescaling.
  The Journal of Chemical Physics, 126(1), 014101.
- ADR-005 v2.0: Integrator Builder Architecture (Prolix internal reference)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array as ArrayType
from jaxtyping import PRNGKeyArray

from prolix.physics.constraints import ConstraintDOFMask
from prolix.typing import IntegratorParams, IntegratorState
from prolix.physics.settle import (
    _DEFAULT_CSVR_TAU_AKMA,
    _csvr_rescale_momenta,
    _langevin_step_a,
    _langevin_step_b,
    _langevin_step_o,
    _langevin_step_o_constrained,
    settle_velocities,
    settle_positions,
)
from prolix.typing import Array, WaterIndicesArray


def _csvr_compute_lambda_jit_safe(
    key: PRNGKeyArray,
    ke_current: ArrayType,
    n_dof: int | ArrayType,
    kT: float | ArrayType,
    dt: float | ArrayType,
    tau: float | ArrayType | None = None,
) -> tuple[ArrayType, PRNGKeyArray]:
  r"""JIT-safe CSVR lambda computation (Bussi et al. 2007).

  Implements stochastic velocity rescaling for canonical ensemble sampling.
  This version is designed to be JIT-safe, unlike the original in settle.py.

  Args:
      key: JAX PRNGKey.
      ke_current: Current kinetic energy.
      n_dof: Number of thermostated degrees of freedom.
      kT: Target thermal energy.
      dt: Timestep.
      tau: Relaxation time. Default ~0.1 ps.

  Returns:
      (lambda_factor, new_key).
  """
  if tau is None:
    tau = _DEFAULT_CSVR_TAU_AKMA

  # Convert n_dof to array for JIT compatibility
  n_dof = jnp.asarray(n_dof)
  n_dof = jnp.maximum(n_dof, 1)

  # Coupling strength
  c1 = jnp.exp(-dt / tau)
  c2_sq = 1.0 - c1

  # Random numbers
  key, split = jax.random.split(key)
  r_gaussian = jax.random.normal(split, dtype=ke_current.dtype)

  key, split = jax.random.split(key)
  # Sample (n_dof - 1) chi-squared components as sum of squares of standard normals
  n_chi = jnp.asarray(jnp.maximum(n_dof - 1, 1), dtype=jnp.int32)
  # To avoid dynamic shape issues, we use a sufficiently large fixed shape
  # and mask the excess samples
  max_chi = 512
  chi_samples = jax.random.normal(split, (max_chi,), dtype=ke_current.dtype)
  chi_mask = jnp.arange(max_chi) < n_chi
  s_chi_squared = jnp.sum(jnp.where(chi_mask, chi_samples**2, 0.0))

  target_ke = 0.5 * n_dof * kT
  ke_safe = jnp.maximum(ke_current, 1e-10)

  # Bussi formula (Eq. A7)
  n_dof_f = jnp.asarray(n_dof, dtype=ke_current.dtype)
  c2_corrected = c2_sq * (target_ke / ke_safe) / n_dof_f

  lambda_sq = (
      c1
      + c2_corrected * (r_gaussian**2 + s_chi_squared)
      + 2.0 * r_gaussian * jnp.sqrt(c1 * c2_corrected)
  )
  lambda_sq_safe = jnp.maximum(lambda_sq, 0.0)

  # Return 1.0 if kinetic energy is too small
  lambda_factor = jnp.where(ke_current <= 0.0, 1.0, jnp.sqrt(lambda_sq_safe))

  return lambda_factor, key


class Step(eqx.Module, ABC):

  """Abstract base class for integrator steps.

  All steps are Equinox Modules that implement a pure `apply` method.
  This ensures they are valid JAX PyTrees and can capture static configuration.

  Purity Guarantee:
  - No side effects (no global state mutation)
  - No Python control flow (use jax.lax.cond, jax.lax.scan, etc.)
  - Output deterministic given input state
  """

  @abstractmethod
  def apply(
      self,
      state: IntegratorState,
      params: IntegratorParams,
  ) -> IntegratorState:
    """Apply one step of the integrator.

    Args:
        state: Current integrator state.
        params: Standardized integrator parameters (dt, kT, etc.).

    Returns:
        Updated state (including updated PRNGKey if stochastic).
    """
    raise NotImplementedError


class O_Step(Step):
  r"""Ornstein-Uhlenbeck stochastic velocity update (O in BAOAB).

  Implements the stochastic friction step from Langevin dynamics:

  .. math::
      p_{new} = c_1 p_{old} + c_2 \sqrt{M \cdot k_T} \cdot \mathcal{N}(0, 1)

  where:
  - $c_1 = \exp(-\gamma \cdot dt)$ (friction damping)
  - $c_2 = \sqrt{1 - c_1^2}$ (noise amplitude)

  When constraint_dofs is provided and project_rigid=True, OU noise is sampled
  in the rigid-body subspace of water molecules (6D per water), preserving
  equipartition under constraints.

  **Reference**:
  Leimkuhler, B., & Shang, X. (2015). Adaptive thermostats for noisy gradient
  systems. SIAM Journal on Numerical Analysis, 54(2), 721-743.

  Attributes:
      fraction: Fraction of full step (default 1.0, or 0.5 for full step in BAOAB).
      project_rigid: If True and constraint_dofs provided, sample noise in rigid subspace.
  """
  fraction: float = eqx.field(static=True)
  project_rigid: bool = eqx.field(static=True)

  def __init__(self, fraction: float = 1.0, project_rigid: bool = True):
    """Initialize O_Step.

    Args:
        fraction: Fraction of full step to apply (0.5 for BAOAB).
        project_rigid: If True, project noise to rigid-body subspace.
    """
    self.fraction = fraction
    self.project_rigid = project_rigid

  def apply(
      self,
      state: IntegratorState,
      params: IntegratorParams,
  ) -> IntegratorState:
    """Apply O-step with optional constraint-aware noise sampling.

    Args:
        state: IntegratorState with momentum and RNG.
        params: Standardized parameters including dt, gamma, kT, and water_indices.

    Returns:
        Updated IntegratorState.
    """
    dt = params.dt * self.fraction
    gamma = params.gamma
    kT = params.kT

    # Check if we should use constrained OU step
    use_constrained = (
        self.project_rigid
        and params.constraint_dofs is not None
        and params.water_indices is not None
    )

    if use_constrained:
      momentum_new, key_out = _langevin_step_o_constrained(
          state.momentum,
          state.positions,
          state.mass,
          gamma,
          dt,
          kT,
          state.key,
          params.water_indices,
      )
    else:
      momentum_new, key_out = _langevin_step_o(
          state.momentum, state.mass, gamma, dt, kT, state.key
      )

    return state.__replace__(momentum=momentum_new, rng=key_out)



class V_Step(Step):
  r"""Velocity update (V in BAOAB, also called B-step or momentum step).

  Updates momenta via forces:

  .. math::
      p_{new} = p_{old} + fraction \cdot dt \cdot F

  Pure position-independent velocity update.

  Attributes:
      fraction: Fraction of timestep to apply (0.5 for BAOAB half-steps).
  """
  fraction: float = eqx.field(static=True)

  def __init__(self, fraction: float = 0.5):
    """Initialize V_Step.

    Args:
        fraction: Fraction of full timestep (0.5 for BAOAB).
    """
    self.fraction = fraction

  def apply(
      self,
      state: IntegratorState,
      params: IntegratorParams,
  ) -> IntegratorState:
    """Apply velocity update.

    Args:
        state: IntegratorState with momentum and force.
        params: Standardized parameters including dt.

    Returns:
        Updated state with new momentum.
    """
    dt = params.dt
    momentum_new = state.momentum + self.fraction * dt * state.force
    return state.__replace__(momentum=momentum_new)



class A_Step(Step):
  r"""Position update step (A in BAOAB).

  Updates positions via momenta:

  .. math::
      r_{new} = r_{old} + fraction \cdot dt \cdot \frac{p_{new}}{m}

  Attributes:
      fraction: Fraction of timestep (1.0 for full step).
  """
  fraction: float = eqx.field(static=True)
  shift_fn: Optional[Callable] = eqx.field(static=True)

  def __init__(self, fraction: float = 1.0, shift_fn: Callable | None = None):
    """Initialize A_Step.

    Args:
        fraction: Fraction of full timestep (0.5 for BAOAB half-steps).
        shift_fn: Optional shift function for PBC (jax_md.space.periodic_shift or similar).
    """
    self.fraction = fraction
    self.shift_fn = shift_fn

  def apply(
      self,
      state: IntegratorState,
      params: IntegratorParams,
  ) -> IntegratorState:
    """Apply position update.

    Args:
        state: IntegratorState with momentum and mass.
        params: Standardized parameters including dt.

    Returns:
        Updated state with new position.
    """
    dt = params.dt * self.fraction
    
    # If no shift_fn, use simple addition (no PBC wrapping)
    if self.shift_fn is None:
      velocity = state.momentum / state.mass
      position_new = state.positions + dt * velocity
    else:
      position_new = _langevin_step_a(
          state.positions, state.momentum, state.mass, dt, self.shift_fn
      )

    return state.__replace__(positions=position_new)


class SETTLE_Velocity_Step(Step):
  r"""Constraint-aware velocity projection (RATTLE for SETTLE).

  Applies velocity constraints to remove kinetic energy from constrained
  degrees of freedom, ensuring velocities are consistent with rigid water bonds.

  Calls settle_velocities() iteratively to project velocities onto the
  constraint-tangent subspace.

  Attributes:
      water_indices: Optional water indices. If None, must be provided in apply params.
      n_iters: Number of RATTLE iterations (default 10).
  """
  water_indices: Optional[tuple[tuple[int, ...], ...]] = eqx.field(static=True)
  n_iters: int = eqx.field(static=True)
  mass_oxygen: float = eqx.field(static=True)
  mass_hydrogen: float = eqx.field(static=True)

  def __init__(
      self,
      water_indices: Array | None = None,
      n_iters: int = 10,
      mass_oxygen: float = 15.999,
      mass_hydrogen: float = 1.008,
  ):
    """Initialize SETTLE_Velocity_Step.

    Args:
        water_indices: (n_waters, 3) array of [O, H1, H2] indices.
        n_iters: Number of RATTLE iterations.
        mass_oxygen: Oxygen mass (amu).
        mass_hydrogen: Hydrogen mass (amu).
    """
    if water_indices is not None:
        self.water_indices = tuple(tuple(int(x) for x in row) for row in np.asarray(water_indices))
    else:
        self.water_indices = None
    self.n_iters = n_iters
    self.mass_oxygen = mass_oxygen
    self.mass_hydrogen = mass_hydrogen

  def apply(
      self,
      state: IntegratorState,
      params: IntegratorParams,
  ) -> IntegratorState:
    """Apply velocity constraint projection.

    Args:
        state: IntegratorState with momentum, position (constrained).
        params: Standardized parameters including dt and positions_old.

    Returns:
        Updated state with constrained velocity.
    """
    water_indices = params.water_indices if params.water_indices is not None else self.water_indices
    if water_indices is None:
      # No water molecules, velocity constraint is no-op
      return state

    positions_old = params.positions_old
    if positions_old is None:
      # If positions_old not provided, assume positions haven't moved yet
      positions_old = state.positions

    dt = params.dt

    # Convert momentum to velocity for settle_velocities
    velocity = state.momentum / state.mass

    velocity_constrained = settle_velocities(
        velocity,
        positions_old,
        state.positions,
        jnp.asarray(water_indices),
        dt,
        mass_oxygen=self.mass_oxygen,
        mass_hydrogen=self.mass_hydrogen,
        n_iters=self.n_iters,
    )

    # Convert back to momentum
    momentum_constrained = velocity_constrained * state.mass

    return state.__replace__(momentum=momentum_constrained)


class SETTLE_Position_Step(Step):
  """Position constraint projection using SETTLE.

  Ensures water molecules maintain rigid geometry (O-H and H-H distances).
  Calls the iterative SETTLE algorithm to project unconstrained positions
  back onto the constraint manifold.
  """

  mass_oxygen: float = eqx.field(static=True)
  mass_hydrogen: float = eqx.field(static=True)
  r_OH: float = eqx.field(static=True)
  r_HH: float = eqx.field(static=True)

  def __init__(
      self,
      mass_oxygen: float = 15.999,
      mass_hydrogen: float = 1.008,
      r_OH: float = 0.9572,
      r_HH: float = 1.5139,
  ):
    """Initialize SETTLE_Position_Step."""
    self.mass_oxygen = mass_oxygen
    self.mass_hydrogen = mass_hydrogen
    self.r_OH = r_OH
    self.r_HH = r_HH

  def apply(
      self,
      state: IntegratorState,
      params: IntegratorParams,
  ) -> IntegratorState:
    """Apply SETTLE position constraints.

    Args:
        state: IntegratorState with unconstrained positions and box.
        params: IntegratorParams with water_indices and positions_old.

    Returns:
        Updated IntegratorState with constrained positions.
    """
    if params.water_indices is None or params.water_indices.shape[0] == 0:
      return state

    # Reference positions for SETTLE (used to fix orientation)
    r_old = jnp.where(
        params.positions_old is not None, params.positions_old, state.positions
    )

    position_constrained = settle_positions(
        state.positions,
        r_old,
        params.water_indices,
        self.r_OH,
        self.r_HH,
        self.mass_oxygen,
        self.mass_hydrogen,
        state.box,
    )

    return state.__replace__(positions=position_constrained)


class CSVR_Step(Step):
  r"""Canonical sampling via velocity rescaling (CSVR, Bussi thermostat).

  Stochastically rescales all momenta by a Langevin-coupled factor to
  drive kinetic energy toward a target value. Preserves constraint subspaces
  (scalar rescaling preserves linear subspaces).

  .. math::
      \lambda = \sqrt{
          c_1 + c_2 (R^2 + S) + 2 R \sqrt{c_1 c_2}
      }

  where $c_1 = \exp(-dt/\tau)$, $R \sim N(0,1)$, $S \sim \chi^2(n_{dof}-1)$.

  **Reference**:
  Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling through
  velocity rescaling. The Journal of Chemical Physics, 126(1), 014101.

  Attributes:
      tau: Relaxation time (default ~0.1 ps in AKMA).
  """
  tau: float = eqx.field(static=True)

  def __init__(self, tau: float = _DEFAULT_CSVR_TAU_AKMA):
    """Initialize CSVR_Step."""
    self.tau = tau

  def apply(
      self,
      state: IntegratorState,
      params: IntegratorParams,
  ) -> IntegratorState:
    """Apply CSVR velocity rescaling.

    Args:
        state: IntegratorState with momentum and RNG.
        params: Standardized parameters including dt, kT, and n_dof.

    Returns:
        Updated IntegratorState.
    """
    dt = params.dt
    kT = params.kT
    n_dof = params.n_dof
    
    if n_dof is None:
      raise ValueError("CSVR_Step requires n_dof in IntegratorParams")

    # Compute kinetic energy
    velocity = state.momentum / state.mass
    ke = 0.5 * jnp.sum(state.mass * velocity**2)

    # Compute rescaling factor and new RNG using JIT-safe version
    lambda_factor, key_out = _csvr_compute_lambda_jit_safe(
        state.key, ke, n_dof, kT, dt, tau=self.tau
    )

    # Rescale momenta
    momentum_new = _csvr_rescale_momenta(state.momentum, lambda_factor)

    return state.__replace__(momentum=momentum_new, rng=key_out)


class NHC_Step(Step):
  r"""Nosé-Hoover Chain thermostat step (deferred to v1.1).

  Not implemented in v1.0. Full implementation requires tracking chain state
  (xi_i, Q_i variables) outside the IntegratorState and architectural changes
  to the integrator builder.

  Deferred to v1.1 pending design completion. See oracle recommendation Path B.
  """

  def __init__(self):
    """Initialize NHC_Step."""
    pass

  def apply(
      self,
      state: IntegratorState,
      constraint_dofs: ConstraintDOFMask | None = None,
      **params,
  ) -> IntegratorState:
    """NHC integration not implemented in v1.0.

    Args:
        state: IntegratorState.
        constraint_dofs: Unused.
        **params: Unused.

    Raises:
        NotImplementedError: NHC thermostat deferred to v1.1.
    """
    raise NotImplementedError(
        "NHC (Nosé-Hoover Chain) thermostat integration is deferred to v1.1. "
        "Use O_Step (Langevin) or CSVR_Step for temperature control in v1.0. "
        "See oracle recommendation Path B and .agent/docs/RELEASE_DECISION_v1.0.md."
    )


# Step registry: map step names to step class constructors
# v1.0 contains O_Step, V_Step, A_Step, SETTLE_Velocity_Step, CSVR_Step.
# LFMiddle_Step, VV_Step (Phase 3) and NHC_Step deferred to v1.1.
step_registry: dict[str, type[Step]] = {
    "o_step": O_Step,
    "v_step": V_Step,
    "a_step": A_Step,
    "settle_velocity_step": SETTLE_Velocity_Step,
    "csvr_step": CSVR_Step,
    "nhc_step": NHC_Step,
    "mc_barostat_step": None,  # Dynamic import in make_step
    "scr_barostat_step": None, # Dynamic import in make_step
    "settle_position_step": SETTLE_Position_Step,
    "virtual_site_reconstruction_step": None,  # Dynamic import in make_step
}


def make_step(name: str, **kwargs) -> Step:
  """Construct a step instance from registry.

  Args:
      name: Step name (key in step_registry).
      **kwargs: Constructor arguments for the step class.

  Returns:
      Instantiated step.

  Raises:
      KeyError: If step name not found in registry.
  """
  if name == "mc_barostat_step":
    from prolix.physics.barostat import MC_Barostat_Step
    return MC_Barostat_Step(**kwargs)
  if name == "scr_barostat_step":
    from prolix.physics.barostat import SCR_Barostat_Step
    return SCR_Barostat_Step(**kwargs)
  if name == "virtual_site_reconstruction_step":
    from prolix.physics.virtual_sites_step import VirtualSiteReconstructionStep
    return VirtualSiteReconstructionStep(**kwargs)

  if name not in step_registry:
    raise KeyError(f"Unknown step '{name}'. Available: {list(step_registry.keys()) + ['mc_barostat_step', 'virtual_site_reconstruction_step']}")
  return step_registry[name](**kwargs)


@dataclass(frozen=True)
class StepSequence:
  """Ordered composition of steps defining an integrator variant.

  A StepSequence defines a complete integrator by specifying:
  1. An ordered list of step names (from step_registry)
  2. Shared parameters (dt, kT, gamma, etc.) for all steps
  3. Optional constraint DOF mask for projection operators
  4. Documentation string

  Immutable (frozen=True) to prevent accidental mutation and allow use as
  dict keys if needed.

  **Example**:
  ```python
  baoab_langevin = StepSequence(
      name='baoab_langevin',
      steps=['v_step', 'a_step', 'o_step', 'a_step', 'v_step'],
      parameters={'dt': 0.5, 'gamma': 1.0, 'kT': 2.479},
      description='BAOAB integrator with Langevin thermostat'
  )
  ```

  Attributes:
      name: Unique name for this integrator variant (e.g., 'baoab_langevin').
      steps: Ordered list of step names (keys in step_registry).
      parameters: Common parameters passed to all steps (dt, kT, etc.).
      constraint_dofs: Optional ConstraintDOFMask for shared projection operators.
      description: Human-readable description of integrator structure and use.
  """

  name: str
  steps: List[str]
  parameters: Dict[str, Any]
  constraint_dofs: Optional[ConstraintDOFMask] = None
  description: str = ""


# Step sequences registry: integrator variants and their step compositions
# Populated after step_registry is defined and validated
step_sequences: Dict[str, StepSequence] = {}


def _initialize_step_sequences() -> None:
  """Initialize step_sequences registry with validated sequences.

  This function is called at module load time to populate step_sequences
  and validate that all step names in each sequence exist in step_registry.

  Raises:
      KeyError: If any step name in a sequence is not in step_registry.
  """
  global step_sequences

  sequences = {
      "baoab_langevin": StepSequence(
          name="baoab_langevin",
          steps=["v_step", "a_step", "o_step", "a_step", "v_step"],
          parameters={
              "dt": 0.5,
              "gamma": 1.0,
              "kT": 2.479,
          },
          description=(
              "BAOAB integrator with Ornstein-Uhlenbeck (Langevin) thermostat. "
              "Structure: V(0.5) → A(0.5) → O(1.0) → A(0.5) → V(0.5). "
              "Symplectic, second-order accurate, preserves measures. "
              "Recommended for NVT (constant volume, temperature) ensemble. "
              "dt ≤ 0.5 fs required for stable rigid-body + thermostat coupling. "
              "Reference: Leimkuhler & Shang (2015)."
          ),
      ),
      "baoab_csvr_npt": StepSequence(
          name="baoab_csvr_npt",
          steps=["v_step", "a_step", "settle_position_step", "scr_barostat_step", "settle_position_step", "a_step", "v_step", "settle_velocity_step", "csvr_step"],
          parameters={
              "dt": 0.5,
              "kT": 2.479,
              "n_dof": 27,
              "tau": 2000.0,  # AKMA units, ~0.1 ps
              "target_pressure_bar": 1.0,
              "compressibility": 4.5e-5,
              "tau_barostat": 2000.0,
          },
          description=(
              "BAOAB integrator with CSVR thermostat and SCR barostat. "
              "Structure: V(0.5) → A(0.5) → SETTLE_pos → SCR → SETTLE_pos → A(0.5) → V(0.5) → SETTLE_vel → CSVR. "
              "Suitable for production NPT ensemble simulations. "
              "Reference: Bernetti & Bussi (2020)."
          ),
      ),
  }

  # Validate: all step names must exist in step_registry
  for seq_name, seq in sequences.items():
    for step_name in seq.steps:
      if step_name not in step_registry:
        raise KeyError(
            f"Sequence '{seq_name}' references unknown step '{step_name}'. "
            f"Available steps: {list(step_registry.keys())}"
        )

  step_sequences = sequences


def make_sequence(name: str, **kwargs) -> StepSequence:
  """Construct a StepSequence from the registry, with parameter overrides.

  Creates a copy of the named sequence with custom parameters merged in.
  Custom kwargs override sequence.parameters.

  Args:
      name: Sequence name (key in step_sequences).
      **kwargs: Parameters to override or add (e.g., dt=0.5, kT=2.479).

  Returns:
      A new StepSequence with merged parameters.

  Raises:
      KeyError: If sequence name not found in step_sequences.

  Example:
      ```python
      seq = make_sequence('baoab_langevin', dt=0.5, gamma=1.5)
      # seq.parameters now has dt=0.5, gamma=1.5, plus original kT=2.479
      ```
  """
  if name not in step_sequences:
    raise KeyError(
        f"Unknown sequence '{name}'. Available: {list(step_sequences.keys())}"
    )

  base_seq = step_sequences[name]
  merged_params = {**base_seq.parameters, **kwargs}

  return StepSequence(
      name=base_seq.name,
      steps=base_seq.steps,
      parameters=merged_params,
      constraint_dofs=base_seq.constraint_dofs,
      description=base_seq.description,
  )


# Initialize sequences at module load time
_initialize_step_sequences()
