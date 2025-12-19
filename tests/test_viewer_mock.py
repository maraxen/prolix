
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# Helper to setup mocks before module import
@pytest.fixture
def mock_dependencies():
    """Mock py2Dmol, jax, numpy, and logging to test viewer.py in isolation."""
    # Mock py2Dmol
    mock_py2dmol = types.ModuleType("py2Dmol")
    mock_py2dmol.view = MagicMock()

    # Crucial assertion: view() should not accept width/height in its constructor (as per bug report)
    # We will enforce this via side_effect
    class StrictMockView:
        def __init__(self, *args, **kwargs):
            if "width" in kwargs or "height" in kwargs:
                 raise TypeError("view.__init__() got an unexpected keyword argument 'height'")
        def addModel(self, *args): pass
        def setStyle(self, *args): pass
        def zoomTo(self): pass
        def addModelsAsFrames(self, *args): pass
        def animate(self, *args): pass

    mock_py2dmol.view.side_effect = StrictMockView

    # Mock other deps
    mock_numpy = types.ModuleType("numpy")
    mock_jax = types.ModuleType("jax")
    mock_traj_mod = types.ModuleType("prolix.visualization.trajectory")
    mock_traj_mod.TrajectoryReader = MagicMock()

    # Setup sys.modules
    with patch.dict(sys.modules, {
        "py2Dmol": mock_py2dmol,
        "numpy": mock_numpy,
        "jax": mock_jax,
        "prolix.visualization.trajectory": mock_traj_mod,
        "prolix.visualization": types.ModuleType("prolix.visualization") # Ensure parent exists
    }):
        # We need to make sure prolix.visualization.trajectory is importable as that
        sys.modules["prolix.visualization"].trajectory = mock_traj_mod

        # Now import/reload viewer
        # If it was already imported, we need to reload it to use the new mocks?
        # Since this runs in a new process usually for pytest, or we can use importlib

        # We'll use import_module. If it's already there, we reload it.
        # But since we patched sys.modules, normal import might find the real file but
        # resolve dependencies from our patched sys.modules.

        # Load the viewer module from source file to ensure we get the real code
        # while dependencies are mocked in sys.modules
        import importlib.util
        import os

        # Assume running from repo root
        file_path = os.path.abspath("src/prolix/visualization/viewer.py")
        spec = importlib.util.spec_from_file_location("prolix.visualization.viewer", file_path)
        if spec is None:
             raise ImportError(f"Could not find viewer.py at {file_path}")
        viewer = importlib.util.module_from_spec(spec)

        # Helper to allow relative imports within the loaded module if needed
        # We mocked prolix.visualization in sys.modules, so it should be fine.
        spec.loader.exec_module(viewer)

        yield viewer

def test_view_structure_no_args(mock_dependencies):
    """Test that view_structure calls py2Dmol.view() without width/height."""
    viewer = mock_dependencies

    # Create dummy PDB
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "ATOM..."

        # Should not raise TypeError
        view = viewer.view_structure("dummy.pdb")

        # Verify py2Dmol.view was called
        import py2Dmol
        assert py2Dmol.view.called

def test_view_trajectory_no_args(mock_dependencies):
    """Test that view_trajectory calls py2Dmol.view() without width/height."""
    viewer = mock_dependencies

    # Mock TrajectoryReader behavior
    mock_reader = MagicMock()
    # Mock return value of get_positions() to be a MagicMock (simulating numpy array)
    mock_positions = MagicMock()
    mock_positions.__getitem__.return_value = [[0,0,0]] # One frame
    mock_positions.shape = (10, 1, 3) # (T, N, 3) - fake shape for validation
    mock_reader.get_positions.return_value = mock_positions

    # Also the sliced result needs to have shape? No, code uses positions.shape[1] from the sliced array if it slices first?
    # Code: positions = trajectory.get_positions()[::stride]
    # So __getitem__ returns the positions array.
    # The result of slicing also needs .shape.

    sliced_positions = MagicMock()
    sliced_positions.__iter__.return_value = [[[0,0,0]]] # Iterating yields frames
    sliced_positions.__getitem__.return_value = [[0,0,0]] # Indexing yields frame
    sliced_positions.shape = (1, 1, 3) # 1 frame, 1 atom
    sliced_positions.__len__.return_value = 1

    mock_positions.__getitem__.return_value = sliced_positions

    # Mock file read
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.readlines.return_value = ["ATOM  1..."]

        # Should not raise TypeError
        view = viewer.view_trajectory(mock_reader, "dummy.pdb")

        import py2Dmol
        assert py2Dmol.view.called
