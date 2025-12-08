"""Trajectory reading utilities."""
from __future__ import annotations

import os
from typing import Iterator, Optional, List, Union
import jax.numpy as jnp
import numpy as np

# We need ArrayRecordReader. 
# Since array_record might not be installed in all envs (as per simulate.py check),
# we should handle it gracefully or assume it's there if they are reading a trajectory.
try:
    from array_record.python.array_record_module import ArrayRecordReader
except ImportError:
    ArrayRecordReader = None

from prolix.simulate import SimulationState

class TrajectoryReader:
    """Reads simulation states from an ArrayRecord file."""
    
    def __init__(self, path: str):
        if ArrayRecordReader is None:
            raise ImportError("array_record not installed. Cannot read trajectory.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Trajectory file not found: {path}")
            
        self.path = path
        self.reader = ArrayRecordReader(path)
        self._num_records = self.reader.num_records()
        
    def __len__(self) -> int:
        return self._num_records
        
    def get_state(self, index: int) -> SimulationState:
        """Get simulation state at specific index."""
        if index < 0 or index >= self._num_records:
            raise IndexError(f"Index {index} out of bounds for trajectory with {self._num_records} frames.")
            
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
    
    def get_energies(self) -> dict[str, np.ndarray]:
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

