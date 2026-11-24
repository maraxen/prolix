"""Task 5 CI controls: no-fork grep, import lightness, refusal and write controls, isolation.

Fast tests only (no PDB load). Each control that can go red has a positive case beside it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prolix.agent import _runtime as rt
from prolix.agent import ops
from prolix.agent.errors import InputError, RefusalError

REPO = Path(__file__).resolve().parents[2]
REAL_HOME = Path(os.path.expanduser("~"))  # captured at import, before the autouse fixture moves HOME
AGENT_SRC = REPO / "src" / "prolix" / "agent"
HEAVY = ("cisternal", "fastmcp", "cyclopts")


def _tiny_bundle(positions=None):
    from prolix.physics.system import make_bundle_from_system

    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]) if positions is None else positions
    system = SimpleNamespace(
        positions=pos, masses=np.array([12.0, 1.0, 16.0]),
        bonds=np.array([[0, 1]]), bond_params=np.array([[300.0, 1.1]]),
        angles=None, angle_params=None, dihedrals=None, dihedral_params=None,
        impropers=None, improper_params=None,
        charges=np.array([0.4, -0.2, -0.2]), sigmas=np.full(3, 3.15), epsilons=np.full(3, 0.1),
    )
    return make_bundle_from_system(system, boundary_condition="free")


def _grep_forbidden(root: Path, tokens=("multiprocessing", "os.fork")) -> list[str]:
    hits = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        for tok in tokens:
            if tok in text:
                hits.append(f"{path.name}:{tok}")
    return hits


# ---------------------------------------------------------------- no-fork policy


def test_agent_package_uses_no_multiprocessing_or_fork():
    assert _grep_forbidden(AGENT_SRC) == []


def test_positive_control_no_fork_grep_flags_the_token(tmp_path):
    bad = tmp_path / "pkg"
    bad.mkdir()
    (bad / "x.py").write_text("import multiprocessing\n")
    assert _grep_forbidden(bad) == ["x.py:multiprocessing"]


# ---------------------------------------------------------------- import lightness


def _modules_after(code: str) -> set[str]:
    proc = subprocess.run(
        [sys.executable, "-c", code + "\nimport sys\nprint('|'.join(sorted(sys.modules)))"],
        capture_output=True, text=True, cwd=REPO, env={**os.environ, "JAX_PLATFORMS": "cpu"}, timeout=300,
        check=True,
    )
    return set(proc.stdout.strip().splitlines()[-1].split("|"))


def test_ops_and_render_import_no_agent_extra_modules():
    mods = _modules_after("import prolix.agent.ops, prolix.agent.render, prolix.agent.system")
    assert not [m for m in HEAVY if m in mods]


def test_positive_control_tools_and_cli_import_pull_the_agent_extra():
    tools_mods = _modules_after("import prolix.agent.tools")
    assert "cisternal" in tools_mods
    cli_mods = _modules_after("import prolix.agent.cli")
    assert "cisternal" in cli_mods and "cyclopts" in cli_mods


# ---------------------------------------------------------------- refusal and write controls


def test_out_dir_through_symlink_to_outside_root_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(root))
    with pytest.raises(InputError):
        ops.load_system(out_dir=str(root / "link" / "run"), bundle=_tiny_bundle())
    assert list(outside.iterdir()) == []


def test_crash_between_side_file_and_record_leaves_no_record(tmp_path, monkeypatch):
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    out = tmp_path / "crash"

    def boom(path, data):
        raise OSError("simulated kill before the record is written")

    monkeypatch.setattr(rt, "atomic_write_bytes", boom)
    with pytest.raises(OSError):
        ops.minimize(out_dir=str(out), bundle=_tiny_bundle(), sd_steps=5)
    assert not (out / "minimize.json").exists()


def test_non_finite_input_energy_is_input_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    bad = _tiny_bundle(np.array([[np.nan, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]))
    with pytest.raises(InputError, match="non-finite"):
        ops.minimize(out_dir=str(tmp_path / "nan"), bundle=bad, sd_steps=5)


def test_pdb_input_checks_fail_before_any_write(tmp_path, monkeypatch):
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    not_pdb = tmp_path / "x.cif"
    not_pdb.write_text("data_x\n")
    out = tmp_path / "out"
    with pytest.raises(InputError):
        ops.load_system(out_dir=str(out), pdb=str(not_pdb))
    with pytest.raises(InputError):
        ops.load_system(out_dir=str(out), pdb=str(tmp_path / "absent.pdb"))
    pdb = REPO / "tests" / "agent" / "data" / "mini.pdb"
    with pytest.raises(InputError, match="force field"):
        ops.load_system(out_dir=str(out), pdb=str(pdb), ff="no_such_field.xml")
    assert not out.exists() or list(out.iterdir()) == []


def test_all_zero_bond_parameters_are_refused_as_from_pdb_failure(tmp_path, monkeypatch):
    import dataclasses

    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    pdb = REPO / "tests" / "agent" / "data" / "mini.pdb"
    base = _tiny_bundle()
    zero = dataclasses.replace(base, bond_params=np.zeros_like(np.asarray(base.bond_params)))
    monkeypatch.setattr(ops, "load_system_bundle", lambda path, ff: zero)
    fake_ff = tmp_path / "fake_ff.xml"
    fake_ff.write_text("<ForceField/>\n")
    monkeypatch.setattr(ops, "_check_ff", lambda ff: str(fake_ff))
    with pytest.raises(InputError, match="all-zero bond"):
        ops.load_system(out_dir=str(tmp_path / "zero"), pdb=str(pdb))


def test_neighbor_list_is_never_passed_to_the_energy_function(tmp_path, monkeypatch):
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    from prolix.api import bundle_md

    seen = []
    real = bundle_md.energy_fn_from_bundle

    def spy_factory(bundle, **kwargs):
        fn = real(bundle, **kwargs)

        def spy(positions, **call_kwargs):
            seen.append(sorted(call_kwargs))
            return fn(positions, **call_kwargs)

        return spy

    monkeypatch.setattr(bundle_md, "energy_fn_from_bundle", spy_factory)
    ops.minimize(out_dir=str(tmp_path / "nl"), bundle=_tiny_bundle(), sd_steps=3)
    assert seen and all("neighbor" not in keys for keys in seen)


# ---------------------------------------------------------------- host-config isolation (#3527 lesson)


def test_isolated_run_writes_nothing_under_the_real_home(tmp_path, monkeypatch):
    watched = [REAL_HOME / ".cisternal", REAL_HOME / ".config" / "xtrax", REAL_HOME / ".myxcel.toml"]

    def snap():
        return {str(p): (p.stat().st_mtime_ns if p.exists() else None) for p in watched}

    before = snap()
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    ops.load_system(out_dir=str(tmp_path / "iso"), bundle=_tiny_bundle())
    assert snap() == before
    assert Path(os.environ["HOME"]) != REAL_HOME
