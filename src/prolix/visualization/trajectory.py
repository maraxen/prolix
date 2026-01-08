"""Trajectory reading utilities."""

from __future__ import annotations

import os

import numpy as np

# We need ArrayRecordReader.
# Since array_record might not be installed in all envs (as per simulate.py check),
# we should handle it gracefully or assume it's there if they are reading a trajectory.
try:
  from array_record.python.array_record_module import (
    ArrayRecordReader,  # type: ignore[unresolved-import]
  )
except ImportError:
  ArrayRecordReader = None

from typing import TYPE_CHECKING

from prolix.simulate import SimulationState

if TYPE_CHECKING:
  from collections.abc import Iterator


class TrajectoryReader:
  """Reads simulation states from an ArrayRecord file."""

  def __init__(self, path: str) -> None:
    if ArrayRecordReader is None:
      msg = "array_record not installed. Cannot read trajectory."
      raise ImportError(msg)
    if not os.path.exists(path):
      msg = f"Trajectory file not found: {path}"
      raise FileNotFoundError(msg)

    self.path = path
    self.reader = ArrayRecordReader(path)
    self._num_records = self.reader.num_records()

  def __len__(self) -> int:
    return self._num_records

  def __getitem__(self, index: int) -> SimulationState:
    return self.get_state(index)

  def get_state(self, index: int) -> SimulationState:
    """Get simulation state at specific index."""
    if index < 0 or index >= self._num_records:
      msg = f"Index {index} out of bounds for trajectory with {self._num_records} frames."
      raise IndexError(msg)

    data = self.reader.read([index])[0]
    return SimulationState.from_array_record(data)

  def __iter__(self) -> Iterator[SimulationState]:
    """Iterate over all states."""
    for i in range(self._num_records):
      yield self.get_state(i)

  def get_positions(self) -> np.ndarray:
    """Load all positions into a numpy array (N_frames, N_atoms, 3).

    Warning: This loads the entire trajectory into memory.
    """
    # Read first frame to get shape
    first = self.get_state(0).numpy()
    pos_shape = first["positions"].shape
    n_frames = self._num_records

    all_pos = np.zeros((n_frames, *pos_shape), dtype=np.float32)

    for i in range(n_frames):
      state = self.get_state(i).numpy()
      all_pos[i] = state["positions"]

    return all_pos

  def get_time(self) -> np.ndarray:
    """Load all time points."""
    times = np.zeros(self._num_records)
    for i in range(self._num_records):
      state = self.get_state(i).numpy()
      times[i] = state["time_ns"]
    return times

  def __enter__(self) -> TrajectoryReader:
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    if hasattr(self.reader, "close"):
      self.reader.close()

  def get_energies(self) -> dict[str, np.ndarray | None]:
    """Load potential and kinetic energies."""
    pot = np.zeros(self._num_records)
    kin = np.zeros(self._num_records)
    has_kin = True

    for i in range(self._num_records):
      state = self.get_state(i).numpy()
      if state["potential_energy"] is not None:
        pot[i] = state["potential_energy"]
      if state["kinetic_energy"] is not None:
        kin[i] = state["kinetic_energy"]
      else:
        has_kin = False

    return {"potential": pot, "kinetic": kin if has_kin else None}
