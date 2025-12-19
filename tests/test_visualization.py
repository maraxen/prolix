"""Tests for the visualization module."""

import numpy as np
import pytest

from prolix.simulate import SimulationState

# Test TrajectoryReader functionality with mocked array_record
# These tests focus on the logic, not the actual file I/O


class MockArrayRecordWriter:
  """Mock ArrayRecordWriter for testing."""

  def __init__(self, path: str, options: str = ""):
    self.path = path
    self.records: list[bytes] = []
    self.closed = False

  def write(self, data: bytes) -> None:
    self.records.append(data)

  def close(self) -> None:
    self.closed = True


class MockArrayRecordReader:
  """Mock ArrayRecordReader for testing."""

  def __init__(self, path: str):
    self.path = path
    # Simulate stored records
    self._records = MockArrayRecordReader._stored_records.get(path, [])
    self.closed = False

  _stored_records: dict[str, list[bytes]] = {}

  def num_records(self) -> int:
    return len(self._records)

  def read(self, indices: list[int]) -> list[bytes]:
    return [self._records[i] for i in indices]

  def close(self) -> None:
    self.closed = True


@pytest.fixture
def mock_array_record(monkeypatch):
  """Fixture that patches array_record with mocks."""
  # Patch at module level after import
  import prolix.visualization as viz_module

  monkeypatch.setattr(viz_module, "ArrayRecordReader", MockArrayRecordReader)

  # Clear stored records
  MockArrayRecordReader._stored_records = {}

  return MockArrayRecordWriter, MockArrayRecordReader


@pytest.fixture
def sample_trajectory(mock_array_record):
  """Create a sample trajectory with a few frames."""
  import msgpack_numpy as m

  m.patch()

  # Create sample states and serialize them
  records = []
  for i in range(5):
    state = SimulationState(
      positions=np.random.randn(10, 3).astype(np.float32) + i * 10,
      velocities=np.random.randn(10, 3).astype(np.float32),
      step=np.array(i * 100),
      time_ns=np.array(i * 0.001),
      potential_energy=np.array(-100.0 + i),
    )
    records.append(state.to_array_record())

  path = "/tmp/test_trajectory.array_record"
  MockArrayRecordReader._stored_records[path] = records

  return path


class TestTrajectoryReader:
  """Tests for the TrajectoryReader class."""

  def test_len(self, sample_trajectory):
    from prolix.visualization import TrajectoryReader

    with TrajectoryReader(sample_trajectory) as reader:
      assert len(reader) == 5

  def test_getitem(self, sample_trajectory):
    from prolix.visualization import TrajectoryReader

    with TrajectoryReader(sample_trajectory) as reader:
      state = reader[0]
      assert isinstance(state, SimulationState)
      assert state.positions.shape == (10, 3)
      assert float(state.step) == 0

      state = reader[2]
      assert float(state.step) == 200

  def test_negative_index(self, sample_trajectory):
    from prolix.visualization import TrajectoryReader

    with TrajectoryReader(sample_trajectory) as reader:
      state = reader[-1]
      assert float(state.step) == 400

  def test_index_out_of_range(self, sample_trajectory):
    from prolix.visualization import TrajectoryReader

    with TrajectoryReader(sample_trajectory) as reader, pytest.raises(IndexError):
      _ = reader[10]

  def test_iter(self, sample_trajectory):
    from prolix.visualization import TrajectoryReader

    with TrajectoryReader(sample_trajectory) as reader:
      states = list(reader)
      assert len(states) == 5
      assert all(isinstance(s, SimulationState) for s in states)

  def test_context_manager(self, sample_trajectory):
    from prolix.visualization import TrajectoryReader

    reader = TrajectoryReader(sample_trajectory)
    assert not reader._closed

    with reader:
      _ = len(reader)

    assert reader._closed


class TestVisualizationFunctions:
  """Tests for visualization functions - import checks only."""

  def test_has_py2dmol_flag(self, mock_array_record):
    from prolix.visualization import HAS_PY2DMOL

    # This will be True or False depending on whether py2Dmol is installed
    assert isinstance(HAS_PY2DMOL, bool)

  def test_visualize_trajectory_import_error_without_py2dmol(
    self, sample_trajectory, monkeypatch
  ):
    import prolix.visualization as viz_module

    # Mock HAS_PY2DMOL to False
    monkeypatch.setattr(viz_module, "HAS_PY2DMOL", False)

    with pytest.raises(ImportError, match="py2Dmol is not installed"):
      viz_module.visualize_trajectory(
        sample_trajectory,
        ca_indices=[0, 1, 2],
      )

  def test_view_frame_import_error_without_py2dmol(self, monkeypatch):
    import prolix.visualization as viz_module

    monkeypatch.setattr(viz_module, "HAS_PY2DMOL", False)

    with pytest.raises(ImportError, match="py2Dmol is not installed"):
      viz_module.view_frame(np.random.randn(10, 3))

  def test_view_pdb_import_error_without_py2dmol(self, monkeypatch):
    import prolix.visualization as viz_module

    monkeypatch.setattr(viz_module, "HAS_PY2DMOL", False)

    with pytest.raises(ImportError, match="py2Dmol is not installed"):
      viz_module.view_pdb("test.pdb")


@pytest.mark.skipif(
  not pytest.importorskip("py2dmol", reason="py2Dmol not installed"),
  reason="py2Dmol not installed",
)
class TestVisualizationWithPy2Dmol:
  """Integration tests that require py2Dmol to be installed."""

  def test_view_frame_basic(self, mock_array_record):
    from prolix.visualization import view_frame

    coords = np.random.randn(10, 3).astype(np.float32)
    viewer = view_frame(
      coords,
      chains=["A"] * 10,
      atom_types=["P"] * 10,
    )
    # Viewer should be created
    assert viewer is not None

  def test_visualize_trajectory(self, sample_trajectory):
    from prolix.visualization import visualize_trajectory

    viewer = visualize_trajectory(
      sample_trajectory,
      ca_indices=[0, 1, 2, 3, 4],
      chains=["A"] * 5,
      atom_types=["P"] * 5,
      frame_stride=2,
    )
    assert viewer is not None
