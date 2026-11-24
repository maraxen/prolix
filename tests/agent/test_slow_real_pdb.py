"""Task 5 slow group (not in CI's fast set): real PDB round trips, loader parity, cap wall time.

``tests/agent/data/mini.pdb`` is residues 1-3 of data/pdb/1CRN.pdb (20 heavy atoms, no hydrogens).
Each test here loads the real proxide parameterisation, roughly 6 s cold on this machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from prolix.agent import _runtime as rt
from prolix.agent import ops

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
MINI = REPO / "tests" / "agent" / "data" / "mini.pdb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(out: Path, tool: str) -> dict:
    return json.loads((out / f"{tool}.json").read_text())


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_real_pdb_round_trip_all_five_ops(root):
    out = root / "mini"
    loaded = ops.load_system(out_dir=str(out), pdb=str(MINI))
    rec = _record(out, "load_system")
    assert rec["inputs"][0]["sha256"] == _sha(MINI)
    assert "jax_cache" in rec and rec["jax"]["backend"] == "cpu"
    assert loaded["result"]["n_hydrogen"] == 0
    assert loaded["result"]["warnings"] == [ops.NO_H_WARNING]
    assert loaded["result"]["bundle_cache"] == "miss"
    assert loaded["result"]["n_real_atoms"] == 20

    rows = ops.inspect_params(out_dir=str(out / "rows"), pdb=str(MINI), term="bond", limit=5)["result"]
    assert rows["total_rows"] > 0 and rows["truncated"] is (rows["total_rows"] > 5)
    assert rows["bundle_cache"] == "hit"

    eb = ops.energy_breakdown(out_dir=str(out / "eb"), pdb=str(MINI))["result"]
    assert np.isfinite(eb["total"]) and np.isfinite(eb["bonded"])

    mn = ops.minimize(out_dir=str(out / "min"), pdb=str(MINI), sd_steps=30)
    assert mn["result"]["energy_after"] <= mn["result"]["energy_before"]
    min_rec = _record(out / "min", "minimize")
    assert min_rec["outputs"][0]["sha256"] == _sha(out / "min" / "minimized.npz")

    md = ops.md_smoke(out_dir=str(out / "md"), pdb=str(MINI), steps=20, dt_fs=0.5)
    assert md["result"]["finite"] is True
    for d in (out, out / "rows", out / "eb", out / "min", out / "md"):
        assert not [p for p in d.iterdir() if p.name.endswith(".tmp.npz") or p.suffix == ".tmp"]


def test_loader_energy_equals_b1_paramize_script(root):
    from prolix.api.bundle_md import active_positions, energy_fn_from_bundle
    from prolix.agent.system import load_system_bundle
    from scripts.benchmarks._b1_paramize import paramize_pdb_to_bundle

    mine = load_system_bundle(str(MINI))
    script = paramize_pdb_to_bundle(str(MINI))
    x = active_positions(mine)
    e_mine = float(energy_fn_from_bundle(mine)(x))
    e_script = float(energy_fn_from_bundle(script)(active_positions(script)))
    assert e_mine == pytest.approx(e_script, rel=1e-6)


def test_bundle_cache_hit_then_miss_after_file_changes(root):
    pdb = root / "copy.pdb"
    # a marker makes the content (and so the cache key) unique to this test
    pdb.write_text(MINI.read_text() + f"REMARK cache-test {os.getpid()} {time.time_ns()}\n")
    first = ops.load_system(out_dir=str(root / "a"), pdb=str(pdb))
    second = ops.load_system(out_dir=str(root / "b"), pdb=str(pdb), overwrite=True)
    assert (first["result"]["bundle_cache"], second["result"]["bundle_cache"]) == ("miss", "hit")
    pdb.write_text(pdb.read_text() + "REMARK changed\n")
    third = ops.load_system(out_dir=str(root / "c"), pdb=str(pdb))
    assert third["result"]["bundle_cache"] == "miss"


def test_cli_in_process_round_trip_prints_one_json_object(root, capsys):
    from prolix.agent import cli

    cli.app(
        ["load_system", "--pdb", str(MINI), "--out-dir", str(root / "cli"), "--json"],
        result_action="return_value",
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["n_real_atoms"] == 20
    assert _record(root / "cli", "load_system")["result"]["n_real_atoms"] == 20


def test_entry_point_writes_cpu_record_even_with_cuda_requested(root):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XTRAX_JAX_CACHE", "MYXCEL_", "PROLIX_"))}
    env.update(
        HOME=str(root / "home"), XDG_CONFIG_HOME=str(root / "xdg"), JAX_PLATFORMS="cuda",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
    )
    (root / "home").mkdir()
    proc = subprocess.run(
        [str(REPO / ".venv" / "bin" / "prolix"), "load_system", "--pdb", str(MINI),
         "--out-dir", str(root / "ep"), "--json"],
        capture_output=True, text=True, cwd=root, env=env, timeout=900, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert _record(root / "ep", "load_system")["jax"]["backend"] == "cpu"


def test_largest_allowed_minimize_finishes_under_120_seconds(root):
    from prolix.physics.system import make_bundle_from_system
    from scripts.benchmarks.agent_caps_calibration import synthetic_system

    n = rt.MAX_ATOMS
    bundle = make_bundle_from_system(synthetic_system(n), boundary_condition="free")
    t = time.perf_counter()
    res = ops.minimize(out_dir=str(root / "cap"), bundle=bundle, sd_steps=rt.MAX_MIN_STEPS)
    wall = time.perf_counter() - t
    assert wall < 120.0, f"{n} atoms x {rt.MAX_MIN_STEPS} steps took {wall:.1f} s"
    assert res["result"]["sd_steps"] == rt.MAX_MIN_STEPS
