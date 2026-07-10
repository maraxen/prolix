"""XR-SINK-XTC: XtcFrameSink + mdtraj write / proxide read contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prolix.api.xtc_sink import (
    ANGSTROM_PER_NM,
    MdtrajXtcWriter,
    XtcFrameSink,
    write_positions_xtc,
)


def _tiny_traj() -> np.ndarray:
    return np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0]],
        ],
        dtype=np.float32,
    )


def test_xtc_frame_sink_is_xtrax_sink_protocol():
    from xtrax.stages.boundaries import Sink

    class _Stub:
        def write_frame(self, coords_angstrom, *, time_ps=0.0):
            pass

        def close(self):
            pass

    sink = XtcFrameSink(path=Path("/tmp/unused.xtc"), backend=_Stub())
    assert sink.ordered is True
    assert isinstance(sink, Sink)


def test_write_positions_xtc_roundtrip_proxide(tmp_path: Path):
    proxide = pytest.importorskip("proxide")
    traj = _tiny_traj()
    path = write_positions_xtc(tmp_path / "t.xtc", traj)
    assert path.exists() and path.stat().st_size > 0

    parsed = proxide.parse_xtc(str(path))
    got = np.asarray(parsed["coordinates"], dtype=np.float32)
    # XTC is lossy single-precision; Å-scale coords should be within ~1e-3 Å
    np.testing.assert_allclose(got, traj, atol=1e-3, rtol=0)


def test_xtc_frame_sink_streaming_frames(tmp_path: Path):
    proxide = pytest.importorskip("proxide")
    traj = _tiny_traj()
    path = tmp_path / "stream.xtc"
    with XtcFrameSink(path=path) as sink:
        for frame in traj:
            sink(frame)
    parsed = proxide.parse_xtc(str(path))
    got = np.asarray(parsed["coordinates"], dtype=np.float32)
    np.testing.assert_allclose(got, traj, atol=1e-3, rtol=0)


def test_mdtraj_backend_writes_nm_on_disk(tmp_path: Path):
    from mdtraj.formats import XTCTrajectoryFile

    traj = _tiny_traj()
    path = tmp_path / "nm.xtc"
    w = MdtrajXtcWriter(path)
    try:
        for i, frame in enumerate(traj):
            w.write_frame(frame, time_ps=float(i))
    finally:
        w.close()
    with XTCTrajectoryFile(str(path), "r") as f:
        xyz_nm, *_ = f.read()
    np.testing.assert_allclose(xyz_nm * ANGSTROM_PER_NM, traj, atol=1e-3, rtol=0)


def test_docs_reject_zarr_for_md_traj():
    """AC4: MD path must not recommend ZarrStagingSink."""
    spec = Path(".praxia/docs/specs/260709_xr-sink-xtc.md").read_text()
    assert "ZarrStagingSink" in spec
    assert "not" in spec.lower()
    mod = Path("src/prolix/api/xtc_sink.py").read_text()
    assert "ZarrStagingSink" in mod
    assert "not" in mod.lower()
