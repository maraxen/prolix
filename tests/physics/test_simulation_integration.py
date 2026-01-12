"""Integration tests for simulation loop and analysis."""

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from prolix import analysis, simulate


# Mock ArrayRecord if not installed
@pytest.fixture
def mock_array_record(monkeypatch):
  class MockWriter:
    def __init__(self, path, schema):
      self.path = path
      self.closed = False
      self.records = []

    def write(self, data):
      self.records.append(data)

    def close(self):
      self.closed = True

  monkeypatch.setattr(simulate, "ArrayRecordWriter", MockWriter)
  return MockWriter


def test_simulation_spec_defaults():
  spec = simulate.SimulationSpec(total_time_ns=1.0)
  assert spec.total_time_ns == 1.0
  assert spec.step_size_fs == 2.0
  assert spec.save_interval_ns == 0.001
  assert spec.accumulate_steps == 500


def test_simulation_state_serialization():
  # Create dummy state
  state = simulate.SimulationState(
    positions=jnp.array([[1.0, 2.0, 3.0]]),
    velocities=jnp.array([[0.1, 0.2, 0.3]]),
    forces=jnp.array([[0.0, 0.0, 0.0]]),
    mass=jnp.array([1.0]),
    step=jnp.array(100),
    time_ns=jnp.array(0.1),
    potential_energy=jnp.array(-10.0),
    kinetic_energy=jnp.array(5.0),
  )

  # 1. To Numpy
  np_state = state.numpy()
  assert isinstance(np_state["positions"], np.ndarray)
  assert np_state["step"] == 100

  # 2. To ArrayRecord bytes
  packed = state.to_array_record()
  assert isinstance(packed, bytes)

  # 3. Roundtrip
  restored = simulate.SimulationState.from_array_record(packed)
  chex.assert_trees_all_close(state, restored)


def test_trajectory_writer(tmp_path, mock_array_record):
  path = str(tmp_path / "test.array_record")
  writer = simulate.TrajectoryWriter(path)

  state = simulate.SimulationState(
    positions=jnp.array([[0.0, 0.0, 0.0]]),
    velocities=jnp.array([[0.0, 0.0, 0.0]]),
    step=jnp.array(0),
    time_ns=jnp.array(0.0),
  )

  # Write single
  writer.write(state)
  assert len(writer.writer.records) == 1

  # Write list
  writer.write([state, state])
  assert len(writer.writer.records) == 3

  writer.close()
  assert writer.writer.closed


def test_production_simulation_short_run(tmp_path, mock_array_record):
  # Setup minimal system
  positions = jnp.zeros((3, 3))  # 3 atoms

  # Mock SystemParams
  # We need a system that `system.make_energy_fn` accepts
  # minimal: topology, masses, charges if implicit solvent

  # Minimal SystemParams for testing
  params = {
    "masses": 1.0,
    "charges": jnp.zeros(3),
    "radii": jnp.ones(3),
    # Needed for implicit solvent?
    # make_energy_fn checks for gb_radii if implicit=True
    "gb_radii": jnp.ones(3),
    "gb_screening": jnp.ones(3),
    "atom_types": jnp.array([0, 0, 0]),
    # Bonded terms require arrays, empty if None?
    # system.py implementation assumes keys exist.
    # bonded.py assumes arrays.
    # Let's provide empty arrays with correct shape [0, N_dim]
    "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
    "bond_params": jnp.zeros((0, 2)),
    "angles": jnp.zeros((0, 3), dtype=jnp.int32),
    "angle_params": jnp.zeros((0, 2)),
    "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
    "dihedral_params": jnp.zeros((0, 3)),
    "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
    "improper_params": jnp.zeros((0, 3)),
    "cmap_torsions": jnp.zeros((0, 5), dtype=jnp.int32),
    "cmap_energy_grids": jnp.zeros((0, 24, 24)),
    "cmap_indices": jnp.zeros((0,), dtype=jnp.int32),
    "exclusion_mask": jnp.ones((3, 3)),  # Dense mask
    "scale_matrix_vdw": None,
    "scale_matrix_elec": None,
    "box": None,
  }

  spec = simulate.SimulationSpec(
    total_time_ns=0.000004,  # Very short: 4fs
    step_size_fs=2.0,
    save_interval_ns=0.000002,  # 2fs (1 step per save)
    accumulate_steps=2,  # 2 steps total
  )
  # total_time=4fs. save_interval=2fs. 2 saves. accumulate_steps=2 (1 batch).
  spec.save_path = str(tmp_path / "traj.array_record")

  final_state = simulate.run_simulation(params, positions, spec)

  assert final_state.step == 2  # 2 saves * 1 step/save * 1 (actually steps calculated from time)
  # round(0.000002 * 1e6 / 2) = 1 step per save.
  # total saves = 2.
  # total steps = 2.

  assert final_state.time_ns == 0.000004


def test_analysis_rmsd():
  # Test RMSD
  p1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
  p2 = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # Distance of 2nd atom is 1 diff

  # RMSD = sqrt( (0^2 + 1^2) / 2 ) = sqrt(0.5) = 0.707
  rmsd = analysis.compute_rmsd(p1, p2)
  assert jnp.isclose(rmsd, jnp.sqrt(0.5))


def test_analysis_contact_map():
  p = jnp.array(
    [
      [0.0, 0.0, 0.0],
      [5.0, 0.0, 0.0],  # Dist 5
      [10.0, 0.0, 0.0],  # Dist 10, Dist(0-2)=10
    ]
  )

  # Threshold 8.0
  # 0-1: 5 < 8 -> 1
  # 1-2: 5 < 8 -> 1
  # 0-2: 10 > 8 -> 0
  cmap = analysis.compute_contact_map(p, threshold_angstrom=8.0)

  expected = jnp.array(
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
  )  # Diagonal usually 0 or 1? logic says dist(i,i)=0 < 8 -> 1
  # Code: dist < threshold.

  # Check diagonal
  assert cmap[0, 0] == 1.0  # Self contact
  assert cmap[0, 1] == 1.0
  assert cmap[0, 2] == 0.0
