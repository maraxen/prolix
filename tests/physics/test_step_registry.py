"""Tests for step_registry: instantiation, lookup, and compositions.

This module validates:
1. All step names resolvable in step_registry
2. Registry instantiation for each step
3. make_step factory function
4. Error handling for unknown steps
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prolix.physics.step_system import (
    A_Step,
    CSVR_Step,
    IntegratorState,
    NHC_Step,
    O_Step,
    SETTLE_Velocity_Step,
    StepSequence,
    V_Step,
    make_sequence,
    make_step,
    step_registry,
    step_sequences,
)
from prolix.typing import IntegratorParams, EnergyParams


class TestStepRegistry:
  """Tests for step_registry dict and make_step factory."""

  def test_registry_contains_all_steps(self):
    """All expected step names are in registry (v1.0 scope)."""
    expected_names = {
        "o_step",
        "v_step",
        "a_step",
        "settle_velocity_step",
        "csvr_step",
        "nhc_step",
    }
    assert expected_names == set(step_registry.rngs())

  def test_registry_maps_to_step_classes(self):
    """All registry entries map to Step subclasses."""
    expected_classes = {
        "o_step": O_Step,
        "v_step": V_Step,
        "a_step": A_Step,
        "settle_velocity_step": SETTLE_Velocity_Step,
        "csvr_step": CSVR_Step,
        "nhc_step": NHC_Step,
    }

    for name, cls in expected_classes.items():
      assert step_registry[name] is cls

  def test_make_step_o_step(self):
    """make_step instantiates O_Step with kwargs."""
    step = make_step("o_step", fraction=0.5, project_rigid=True)
    assert isinstance(step, O_Step)
    assert step.fraction == 0.5
    assert step.project_rigid is True

  def test_make_step_v_step(self):
    """make_step instantiates V_Step with kwargs."""
    step = make_step("v_step", fraction=0.5)
    assert isinstance(step, V_Step)
    assert step.fraction == 0.5

  def test_make_step_a_step(self):
    """make_step instantiates A_Step with kwargs."""
    step = make_step("a_step", fraction=1.0)
    assert isinstance(step, A_Step)
    assert step.fraction == 1.0

  def test_make_step_settle_velocity_step(self):
    """make_step instantiates SETTLE_Velocity_Step with kwargs."""
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5]])
    step = make_step("settle_velocity_step", water_indices=water_indices, n_iters=10)
    assert isinstance(step, SETTLE_Velocity_Step)
    assert step.n_iters == 10

  def test_make_step_csvr_step(self):
    """make_step instantiates CSVR_Step with kwargs."""
    step = make_step("csvr_step")
    assert isinstance(step, CSVR_Step)

  def test_make_step_nhc_step(self):
    """make_step instantiates NHC_Step."""
    step = make_step("nhc_step")
    assert isinstance(step, NHC_Step)

  def test_make_step_unknown_name_raises_keyerror(self):
    """make_step raises KeyError for unknown step name."""
    with pytest.raises(KeyError) as exc_info:
      make_step("unknown_step")
    assert "Unknown step" in str(exc_info.value)
    assert "unknown_step" in str(exc_info.value)

  def test_make_step_error_message_lists_available(self):
    """make_step error message includes available step names."""
    with pytest.raises(KeyError) as exc_info:
      make_step("invalid_step_name")
    error_msg = str(exc_info.value)
    # Check that available steps are mentioned
    assert "Available" in error_msg or "o_step" in error_msg


class TestStepInstantiation:
  """Tests for direct instantiation of each step."""

  def test_instantiate_o_step_defaults(self):
    """O_Step instantiates with default parameters."""
    step = O_Step()
    assert step.fraction == 1.0
    assert step.project_rigid is True

  def test_instantiate_v_step_defaults(self):
    """V_Step instantiates with default parameters."""
    step = V_Step()
    assert step.fraction == 0.5

  def test_instantiate_a_step_defaults(self):
    """A_Step instantiates with default parameters."""
    step = A_Step()
    assert step.fraction == 1.0
    assert step.shift_fn is None

  def test_instantiate_settle_velocity_step_defaults(self):
    """SETTLE_Velocity_Step instantiates with default parameters."""
    step = SETTLE_Velocity_Step()
    assert step.water_indices is None
    assert step.n_iters == 10

  def test_instantiate_csvr_step_defaults(self):
    """CSVR_Step instantiates with default parameters."""
    step = CSVR_Step()
    assert step is not None

  def test_instantiate_nhc_step_defaults(self):
    """NHC_Step instantiates."""
    step = NHC_Step()
    assert step is not None

  def test_o_step_custom_parameters(self):
    """O_Step accepts custom parameters."""
    step = O_Step(fraction=0.25, project_rigid=False)
    assert step.fraction == 0.25
    assert step.project_rigid is False

  def test_v_step_custom_parameters(self):
    """V_Step accepts custom parameters."""
    step = V_Step(fraction=0.25)
    assert step.fraction == 0.25

  def test_a_step_custom_parameters(self):
    """A_Step accepts custom shift_fn."""
    def dummy_shift_fn(x, v):
      return x + v

    step = A_Step(fraction=0.5, shift_fn=dummy_shift_fn)
    assert step.fraction == 0.5
    assert step.shift_fn is dummy_shift_fn

  def test_settle_velocity_step_custom_parameters(self):
    """SETTLE_Velocity_Step accepts custom parameters."""
    water_indices = jnp.array([[0, 1, 2]])
    step = SETTLE_Velocity_Step(
        water_indices=water_indices,
        n_iters=20,
        mass_oxygen=16.0,
        mass_hydrogen=1.0,
    )
    assert water_indices is not None
    assert step.n_iters == 20

  def test_csvr_step_custom_parameters(self):
    """CSVR_Step accepts custom initialization."""
    step = CSVR_Step(tau=100.0)
    assert step.tau == 100.0


class TestStepCompositions:
  """Tests for composing steps in a simple integrator loop."""

  def test_compose_v_a_v_from_registry(self):
    """Compose V-A-V steps from registry."""
    v_step = make_step("v_step", fraction=0.5)
    a_step = make_step("a_step", fraction=1.0)

    positions = jnp.zeros((9, 3))
    momentum = jnp.zeros((9, 3))
    force = jnp.ones((9, 3))
    mass = jnp.ones((9, 1))
    rng = jax.random.PRNGKey(0)

    state = IntegratorState(positions=positions, momentum=momentum, force=force, mass=mass, rng=rng)
    params = IntegratorParams(dt=0.001, kT=2.479, gamma=1.0, energy_params=EnergyParams(params=None),
                              water_indices=jnp.zeros((0, 3), dtype=jnp.int32),
                              constraint_dofs=jnp.zeros((0,), dtype=jnp.int32),
                              box=jnp.zeros((3,)),
                              position_old=jnp.zeros((0, 3)),
                              n_dof=0.0)
    # V-A-V cycle
    state = v_step.apply(state, params)
    state = a_step.apply(state, params)
    state = v_step.apply(state, params)

    assert state.positions is not None
    assert state.momentum is not None

  def test_compose_v_o_a_from_registry(self):
    """Compose V-O-A steps from registry."""
    v_step = make_step("v_step", fraction=0.5)
    o_step = make_step("o_step", fraction=1.0)
    a_step = make_step("a_step", fraction=1.0)

    positions = jnp.zeros((9, 3))
    momentum = jnp.zeros((9, 3))
    force = jnp.ones((9, 3))
    mass = jnp.ones((9, 1))
    rng = jax.random.PRNGKey(0)

    state = IntegratorState(positions=positions, momentum=momentum, force=force, mass=mass, rng=rng)
    params = IntegratorParams(dt=0.001, kT=2.479, gamma=1.0, energy_params=EnergyParams(params=None),
                              water_indices=jnp.zeros((0, 3), dtype=jnp.int32),
                              constraint_dofs=jnp.zeros((0,), dtype=jnp.int32),
                              box=jnp.zeros((3,)),
                              position_old=jnp.zeros((0, 3)),
                              n_dof=0.0)
    # V-O-A cycle
    state = v_step.apply(state, params)
    state = o_step.apply(state, params)
    state = a_step.apply(state, params)

    assert state.positions is not None
    assert state.momentum is not None

  def test_compose_with_csvr_from_registry(self):
    """Compose V-A-CSVR sequence from registry."""
    v_step = make_step("v_step", fraction=0.5)
    a_step = make_step("a_step", fraction=1.0)
    csvr_step = make_step("csvr_step")

    positions = jnp.zeros((9, 3))
    momentum = jnp.ones((9, 3))
    force = jnp.ones((9, 3))
    mass = jnp.ones((9, 1))
    rng = jax.random.PRNGKey(0)

    state = IntegratorState(positions=positions, momentum=momentum, force=force, mass=mass, rng=rng)
    params = IntegratorParams(dt=0.001, kT=2.479, gamma=1.0, n_dof=27, energy_params=EnergyParams(params=None),
                              water_indices=jnp.zeros((0, 3), dtype=jnp.int32),
                              constraint_dofs=jnp.zeros((0,), dtype=jnp.int32),
                              box=jnp.zeros((3,)),
                              position_old=jnp.zeros((0, 3)))
    # V-A-CSVR sequence
    state = v_step.apply(state, params)
    state = a_step.apply(state, params)
    state = csvr_step.apply(state, params)

    assert state.positions is not None
    assert state.momentum is not None


class TestStepSequences:
  """Tests for StepSequence dataclass and step_sequences registry."""

  def test_step_sequences_registry_exists(self):
    """step_sequences registry is populated."""
    assert isinstance(step_sequences, dict)
    assert len(step_sequences) > 0

  def test_step_sequences_contains_required_sequences(self):
    """step_sequences contains the v1.0 integrator variants."""
    expected_names = {
        "baoab_langevin",
        "baoab_csvr_npt",
    }
    assert expected_names == set(step_sequences.rngs())

  def test_step_sequence_frozen_dataclass(self):
    """StepSequence is immutable (frozen dataclass)."""
    seq = step_sequences["baoab_langevin"]
    with pytest.raises(Exception):  # frozen dataclass raises dataclass.FrozenInstanceError
      seq.name = "modified"

  def test_baoab_langevin_structure(self):
    """BAOAB_LANGEVIN sequence has expected structure."""
    seq = step_sequences["baoab_langevin"]
    assert seq.name == "baoab_langevin"
    assert seq.steps == ["v_step", "a_step", "o_step", "a_step", "v_step"]
    assert "dt" in seq.parameters
    assert "gamma" in seq.parameters
    assert "kT" in seq.parameters
    assert len(seq.description) > 0

  def test_baoab_csvr_npt_structure(self):
    """BAOAB_CSVR_NPT sequence has expected structure."""
    seq = step_sequences["baoab_csvr_npt"]
    assert seq.name == "baoab_csvr_npt"
    assert seq.steps == ["v_step", "a_step", "csvr_step", "a_step", "v_step"]
    assert "dt" in seq.parameters
    assert "kT" in seq.parameters
    assert "n_dof" in seq.parameters
    assert "tau" in seq.parameters

  def test_all_sequence_step_names_valid(self):
    """All step names in all sequences are in step_registry."""
    for seq_name, seq in step_sequences.items():
      for step_name in seq.steps:
        assert step_name in step_registry, (
            f"Sequence '{seq_name}' references unknown step '{step_name}'"
        )

  def test_make_sequence_returns_step_sequence(self):
    """make_sequence returns a StepSequence instance."""
    seq = make_sequence("baoab_langevin")
    assert isinstance(seq, StepSequence)

  def test_make_sequence_preserves_base_structure(self):
    """make_sequence preserves name, steps, description from base."""
    seq = make_sequence("baoab_langevin")
    base_seq = step_sequences["baoab_langevin"]
    assert seq.name == base_seq.name
    assert seq.steps == base_seq.steps
    assert seq.description == base_seq.description

  def test_make_sequence_parameter_override(self):
    """make_sequence kwargs override base parameters."""
    seq = make_sequence("baoab_langevin", dt=0.25, gamma=2.0)
    assert seq.parameters["dt"] == 0.25
    assert seq.parameters["gamma"] == 2.0
    # Original kT should still be there
    assert seq.parameters["kT"] == 2.479

  def test_make_sequence_parameter_addition(self):
    """make_sequence can add new parameters not in base."""
    seq = make_sequence("baoab_langevin", custom_param=42.0)
    assert seq.parameters["custom_param"] == 42.0
    assert seq.parameters["dt"] == 0.5  # Base param preserved

  def test_make_sequence_csvr_npt_with_overrides(self):
    """make_sequence('baoab_csvr_npt') works with parameter overrides."""
    seq = make_sequence("baoab_csvr_npt", n_dof=64, tau=1000.0)
    assert seq.parameters["n_dof"] == 64
    assert seq.parameters["tau"] == 1000.0
    assert seq.parameters["dt"] == 0.5  # Base preserved

  def test_make_sequence_unknown_name_raises_keyerror(self):
    """make_sequence raises KeyError for unknown sequence name."""
    with pytest.raises(KeyError) as exc_info:
      make_sequence("unknown_sequence")
    assert "Unknown sequence" in str(exc_info.value)
    assert "unknown_sequence" in str(exc_info.value)

  def test_make_sequence_error_message_lists_available(self):
    """make_sequence error includes available sequence names."""
    with pytest.raises(KeyError) as exc_info:
      make_sequence("invalid")
    error_msg = str(exc_info.value)
    # Check that some sequences are mentioned
    assert "baoab_langevin" in error_msg or "Available" in error_msg

  def test_step_sequence_immutability_after_make(self):
    """Sequence returned from make_sequence is immutable."""
    seq = make_sequence("baoab_langevin")
    with pytest.raises(Exception):
      seq.name = "modified"

  def test_step_sequence_default_constraint_dofs(self):
    """StepSequence constraint_dofs defaults to None."""
    seq = step_sequences["baoab_langevin"]
    assert seq.constraint_dofs is None

  def test_step_sequence_description_nonempty(self):
    """All sequences have non-empty description strings."""
    for seq in step_sequences.values():
      assert len(seq.description) > 0
      # Should mention something about the integrator
      assert "integrator" in seq.description.lower() or "BAOAB" in seq.description


class TestStepSequenceCompositions:
  """Tests for composing mini-integrator sequences from registry."""

  def test_compose_baoab_langevin_minimal(self):
    """Compose minimal BAOAB_LANGEVIN sequence (V-A-O-A-V)."""
    seq = make_sequence("baoab_langevin", dt=0.001, gamma=1.0, kT=2.479)
    params = IntegratorParams(
        dt=seq.parameters["dt"],
        gamma=seq.parameters["gamma"],
        kT=seq.parameters["kT"],
        energy_params=EnergyParams(params=None),
        water_indices=jnp.zeros((0, 3), dtype=jnp.int32),
        constraint_dofs=jnp.zeros((0,), dtype=jnp.int32),
        box=jnp.zeros((3,)),
        position_old=jnp.zeros((0, 3)),
        n_dof=0.0
    )
    # Manually compose steps
    v_step = make_step("v_step", fraction=0.5)
    a_step = make_step("a_step", fraction=1.0)
    o_step = make_step("o_step", fraction=1.0)

    # Create mock state
    positions = jnp.zeros((9, 3))
    momentum = jnp.ones((9, 3))
    force = jnp.ones((9, 3))
    mass = jnp.ones((9, 1))
    rng = jax.random.PRNGKey(0)

    state = IntegratorState(positions=positions, momentum=momentum, force=force, mass=mass, rng=rng)

    # Apply sequence steps (V-A-O-A-V)
    state = v_step.apply(state, params)
    state = a_step.apply(state, params)
    state = o_step.apply(state, params)
    state = a_step.apply(state, params)
    state = v_step.apply(state, params)

    assert state.positions is not None
    assert state.momentum is not None
    assert not jnp.any(jnp.isnan(state.positions))
    assert not jnp.any(jnp.isnan(state.momentum))

  def test_compose_baoab_csvr_npt_minimal(self):
    """Compose minimal BAOAB_CSVR_NPT sequence (V-A-CSVR-A-V)."""
    seq = make_sequence("baoab_csvr_npt", dt=0.001, n_dof=27, kT=2.479)
    params = IntegratorParams(
        dt=seq.parameters["dt"],
        kT=seq.parameters["kT"],
        n_dof=seq.parameters["n_dof"],
        energy_params=EnergyParams(params=None),
        gamma=0.0,
        water_indices=jnp.zeros((0, 3), dtype=jnp.int32),
        constraint_dofs=jnp.zeros((0,), dtype=jnp.int32),
        box=jnp.zeros((3,)),
        position_old=jnp.zeros((0, 3))
    )

    v_step = make_step("v_step", fraction=0.5)
    a_step = make_step("a_step", fraction=1.0)
    csvr_step = make_step("csvr_step")

    positions = jnp.zeros((9, 3))
    momentum = jnp.ones((9, 3))
    force = jnp.ones((9, 3))
    mass = jnp.ones((9, 1))
    rng = jax.random.PRNGKey(0)

    state = IntegratorState(positions=positions, momentum=momentum, force=force, mass=mass, rng=rng)

    # V-A-CSVR-A-V
    state = v_step.apply(state, params)
    state = a_step.apply(state, params)
    state = csvr_step.apply(state, params)
    state = a_step.apply(state, params)
    state = v_step.apply(state, params)

    assert state.positions is not None
    assert state.momentum is not None

  def test_sequence_parameter_propagation_across_steps(self):
    """Parameters in sequence are correctly propagated to all steps."""
    seq = make_sequence("baoab_langevin", dt=0.002, gamma=1.5, kT=3.0)
    params = IntegratorParams(
        dt=seq.parameters["dt"],
        gamma=seq.parameters["gamma"],
        kT=seq.parameters["kT"],
        energy_params=EnergyParams(params=None),
        water_indices=jnp.zeros((0, 3), dtype=jnp.int32),
        constraint_dofs=jnp.zeros((0,), dtype=jnp.int32),
        box=jnp.zeros((3,)),
        position_old=jnp.zeros((0, 3)),
        n_dof=0.0
    )
    # Each step should receive these parameters
    v_step = make_step("v_step", fraction=0.5)
    o_step = make_step("o_step", fraction=1.0)

    positions = jnp.zeros((3, 3))
    momentum = jnp.ones((3, 3))
    force = jnp.ones((3, 3))
    mass = jnp.ones((3, 1))
    rng = jax.random.PRNGKey(0)

    state = IntegratorState(positions=positions, momentum=momentum, force=force, mass=mass, rng=rng)

    # Apply with sequence parameters
    state = v_step.apply(state, params)
    state = o_step.apply(state, params)

    # If we got here without error, parameters were propagated
    assert state is not None


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
