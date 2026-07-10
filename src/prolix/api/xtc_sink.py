"""MD trajectory XTC sink composed with xtrax Sink protocol (XR-SINK-XTC).

Explicitly **not** ``xtrax.run.ZarrStagingSink`` — Zarr is wrong for MD traj
interchange. Proxide's Rust ``XtcWriter`` is still a stub (returns
``not yet implemented``); this leaf uses mdtraj ``XTCTrajectoryFile`` behind
``XtcWriterBackend`` until proxide exposes a working Python writer. Readback
uses ``proxide.parse_xtc`` (Å).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "ANGSTROM_PER_NM",
    "MdtrajXtcWriter",
    "XtcFrameSink",
    "XtcWriterBackend",
    "write_positions_xtc",
]

ANGSTROM_PER_NM = 10.0


@runtime_checkable
class XtcWriterBackend(Protocol):
    """Minimal frame writer; swap when proxide ships a real PyXtcWriter."""

    def write_frame(self, coords_angstrom: np.ndarray, *, time_ps: float = 0.0) -> None:
        """Append one frame; ``coords_angstrom`` shape ``(n_atoms, 3)``."""

    def close(self) -> None:
        """Flush and close the underlying file."""


@dataclass
class MdtrajXtcWriter:
    """Interim XTC writer via mdtraj (Å in → nm on disk)."""

    path: Path
    _file: Any = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        from mdtraj.formats import XTCTrajectoryFile

        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = XTCTrajectoryFile(str(self.path), "w")

    def write_frame(self, coords_angstrom: np.ndarray, *, time_ps: float = 0.0) -> None:
        if self._closed or self._file is None:
            raise RuntimeError(f"XTC writer already closed: {self.path}")
        coords = np.asarray(coords_angstrom, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"expected (n_atoms, 3), got {coords.shape}")
        xyz_nm = coords[np.newaxis, ...] / ANGSTROM_PER_NM
        self._file.write(xyz_nm, time=np.asarray([time_ps], dtype=np.float32))

    def close(self) -> None:
        if not self._closed and self._file is not None:
            self._file.close()
            self._closed = True
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass
class XtcFrameSink:
    """xtrax ``Sink``-compatible terminal: positions → ``.xtc`` via backend.

    ``ordered=True`` matches the Tap–Sink contract (SafeMap/Scan axes). Host
    ``__call__`` writes immediately; under a JAX trace callers should wrap with
    ``jax.experimental.io_callback`` (same pattern as ``ZarrStagingSink``
    consumers — the sink itself stays host-side).
    """

    path: Path
    ordered: bool = True
    backend: XtcWriterBackend | None = None
    _owns_backend: bool = field(default=False, init=False, repr=False)
    _frame_idx: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.backend is None:
            self.backend = MdtrajXtcWriter(self.path)
            self._owns_backend = True

    def __call__(self, x: Any) -> None:
        """Accept ``(n_atoms, 3)`` or ``(n_frames, n_atoms, 3)`` positions in Å."""
        assert self.backend is not None
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 2:
            frames = arr[np.newaxis, ...]
        elif arr.ndim == 3:
            frames = arr
        else:
            raise ValueError(f"XtcFrameSink expected 2D or 3D positions, got {arr.shape}")
        for frame in frames:
            self.backend.write_frame(frame, time_ps=float(self._frame_idx))
            self._frame_idx += 1

    def close(self) -> None:
        if self._owns_backend and self.backend is not None:
            self.backend.close()

    def __enter__(self) -> XtcFrameSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_positions_xtc(
    path: str | Path,
    positions_angstrom: Any,
    *,
    backend: XtcWriterBackend | None = None,
) -> Path:
    """Write a full trajectory array to ``.xtc`` and return the path.

    ``positions_angstrom``: ``(n_frames, n_atoms, 3)`` or ``(n_atoms, 3)``.
    """
    out = Path(path)
    sink = XtcFrameSink(path=out, backend=backend)
    try:
        sink(positions_angstrom)
    finally:
        sink.close()
    return out
