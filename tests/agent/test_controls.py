"""Task 5 CI controls: no-fork grep, import lightness, refusal and write controls, isolation.

Fast tests only (no PDB load). Each control that can go red has a positive case beside it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
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
MINI = REPO / "tests" / "agent" / "data" / "mini.pdb"


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


def _spy_energy_factory(monkeypatch):
    """Wrap energy_fn_from_bundle. Returns ``(factory_kwargs, call_kwargs)`` lists that fill as ops run."""
    from prolix.api import bundle_md

    factory_kwargs: list[list[str]] = []
    call_kwargs: list[list[str]] = []
    real = bundle_md.energy_fn_from_bundle

    def spy_factory(bundle, **kwargs):
        factory_kwargs.append(sorted(kwargs))
        fn = real(bundle, **kwargs)

        def spy(positions, **kw):
            call_kwargs.append(sorted(kw))
            return fn(positions, **kw)

        return spy

    monkeypatch.setattr(bundle_md, "energy_fn_from_bundle", spy_factory)
    return factory_kwargs, call_kwargs


def _assert_no_neighbor(*recorded: list[list[str]]) -> None:
    for calls in recorded:
        for keys in calls:
            assert "neighbor" not in keys, f"energy function received neighbor= ({keys})"


def test_neighbor_list_is_never_passed_to_the_energy_function(tmp_path, monkeypatch):
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    factory_kwargs, call_kwargs = _spy_energy_factory(monkeypatch)
    ops.minimize(out_dir=str(tmp_path / "nl"), bundle=_tiny_bundle(), sd_steps=3)
    assert factory_kwargs and call_kwargs, "control is vacuous: the spy saw no energy calls"
    _assert_no_neighbor(factory_kwargs, call_kwargs)


def test_control_a_call_with_neighbor_fails_the_assertion(monkeypatch):
    from prolix.api import bundle_md

    factory_kwargs, call_kwargs = _spy_energy_factory(monkeypatch)
    bundle = _tiny_bundle()
    fn = bundle_md.energy_fn_from_bundle(bundle)
    fn(bundle.positions, neighbor=None)  # a call that passes neighbor= must be visible to the check
    assert call_kwargs == [["neighbor"]]
    with pytest.raises(AssertionError, match="neighbor"):
        _assert_no_neighbor(factory_kwargs, call_kwargs)
    with pytest.raises(AssertionError, match="neighbor"):
        _assert_no_neighbor([["neighbor"]])


# ---------------------------------------------------------------- validation of loaded bundles


def test_bundle_with_atoms_and_no_bonds_is_refused():
    base = _tiny_bundle()
    no_bonds = dataclasses.replace(base, bond_mask=np.zeros_like(np.asarray(base.bond_mask)))
    with pytest.raises(InputError, match="no bonds"):
        ops._validate_loaded(no_bonds)


def test_control_bond_free_single_atom_and_bonded_bundle_pass_validation():
    base = _tiny_bundle()
    ops._validate_loaded(base)  # bonded three-atom system passes
    single = dataclasses.replace(
        base, bond_mask=np.zeros_like(np.asarray(base.bond_mask)), n_atoms=1
    )
    ops._validate_loaded(single)  # one atom has no bond to lose


# ---------------------------------------------------------------- energy breakdown convention


def _assert_cmap_convention(result: dict) -> None:
    assert result.get("bonded_excludes") == ["cmap"], result.get("bonded_excludes")
    assert "cmap" in result.get("nonbonded_by_difference_includes", []), result


def test_energy_breakdown_names_what_nonbonded_by_difference_includes(tmp_path, monkeypatch):
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    res = ops.energy_breakdown(out_dir=str(tmp_path / "eb"), bundle=_tiny_bundle())["result"]
    _assert_cmap_convention(res)
    assert res["nonbonded_by_difference_includes"] == list(ops.NONBONDED_BY_DIFFERENCE_INCLUDES)


def test_control_result_without_the_convention_fields_fails_the_check():
    with pytest.raises(AssertionError):
        _assert_cmap_convention({"total": 1.0, "bonded": 0.5, "nonbonded_by_difference": 0.5})


# ---------------------------------------------------------------- record content


def test_record_args_and_inputs_hold_resolved_pdb_and_ff_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    monkeypatch.setattr(ops, "_BUNDLE_CACHE", OrderedDict())  # no entries from other tests
    monkeypatch.setattr(ops, "load_system_bundle", lambda path, ff_name: _tiny_bundle())
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    pdb = real_dir / "mini.pdb"
    shutil.copy(MINI, pdb)
    pdb_link = tmp_path / "link.pdb"
    pdb_link.symlink_to(pdb)
    ff = real_dir / "ff.xml"
    ff.write_text("<ForceField/>\n")
    ff_link = tmp_path / "ff_link.xml"
    ff_link.symlink_to(ff)

    ops.load_system(out_dir=str(tmp_path / "linked"), pdb=str(pdb_link), ff=str(ff_link))
    rec = json.loads((tmp_path / "linked" / "load_system.json").read_text())
    assert rec["args"]["pdb"] == str(pdb.resolve())
    assert rec["args"]["ff"] == str(ff.resolve())
    assert [i["path"] for i in rec["inputs"]] == [str(pdb.resolve()), str(ff.resolve())]

    # control: a path that is already resolved is recorded unchanged
    ops.load_system(out_dir=str(tmp_path / "direct"), pdb=str(pdb), ff=str(ff))
    direct = json.loads((tmp_path / "direct" / "load_system.json").read_text())
    assert direct["args"]["pdb"] == rec["args"]["pdb"]
    assert direct["inputs"] == rec["inputs"]


# ---------------------------------------------------------------- JAX cache state and strict JSON


def test_jax_cache_state_records_the_error_instead_of_swallowing_it(monkeypatch):
    from xtrax import jax_cache

    def boom():
        raise RuntimeError("bad cache setting")

    monkeypatch.setattr(jax_cache, "current", boom)
    state = ops._jax_cache_state()
    assert state["configured"] is None
    assert "bad cache setting" in state["error"]


def test_control_healthy_jax_cache_state_has_no_error():
    state = ops._jax_cache_state()
    assert "error" not in state


def _reject_constant(token: str):
    raise ValueError(f"non-strict JSON constant {token}")


def test_nan_energy_is_recorded_as_null_in_strict_json():
    text = rt.dumps_record({"result": {"energy_end": float("nan"), "ok": 1.5}}).decode()
    assert "NaN" not in text
    parsed = json.loads(text, parse_constant=_reject_constant)  # strict: no NaN/Infinity tokens
    assert parsed["result"] == {"energy_end": None, "ok": 1.5}


def test_control_infinite_energy_is_null_and_finite_values_survive():
    text = rt.dumps_record({"a": float("inf"), "b": -2.0, "c": [0.0, float("-inf")]}).decode()
    parsed = json.loads(text, parse_constant=_reject_constant)
    assert parsed == {"a": None, "b": -2.0, "c": [0.0, None]}


# ---------------------------------------------------------------- isolation (#3527 lesson)


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, int] | None]:
    """(mtime_ns, size) per path, or None when absent. A write to a file or a new entry changes it."""
    out: dict[str, tuple[int, int] | None] = {}
    for p in paths:
        if p.exists():
            st = p.stat()
            out[str(p)] = (st.st_mtime_ns, st.st_size)
        else:
            out[str(p)] = None
    return out


def test_positive_control_isolation_snapshot_sees_writes(tmp_path):
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    cfg = watched_dir / "config.toml"
    cfg.write_text("a\n")
    watched = [watched_dir, cfg, watched_dir / "later.txt"]
    before = _snapshot(watched)
    time.sleep(0.01)
    cfg.write_text("a much longer line\n")
    (watched_dir / "later.txt").write_text("new\n")
    assert _snapshot(watched) != before


def test_isolated_run_writes_nothing_under_the_real_home(tmp_path, monkeypatch):
    watched = [REAL_HOME / ".cisternal", REAL_HOME / ".config" / "xtrax", REAL_HOME / ".myxcel.toml"]
    before = _snapshot(watched)
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    ops.load_system(out_dir=str(tmp_path / "iso"), bundle=_tiny_bundle())
    assert _snapshot(watched) == before
    assert Path(os.environ["HOME"]) != REAL_HOME


# ---------------------------------------------------------------- OS errors map to input errors (exit 4)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file; the test needs a normal user")
def test_unreadable_pdb_is_input_error_and_readable_control_passes(tmp_path, monkeypatch):
    from prolix.agent.system import count_pdb_atoms

    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    pdb = tmp_path / "locked.pdb"
    shutil.copy(MINI, pdb)
    pdb.chmod(0)
    try:
        with pytest.raises(InputError, match="cannot read"):
            ops.load_system(out_dir=str(tmp_path / "locked"), pdb=str(pdb))
        assert not (tmp_path / "locked").exists()
    finally:
        pdb.chmod(0o644)
    # control: the same file, readable again, is counted
    assert count_pdb_atoms(pdb) == (20, 0)


def test_out_dir_that_cannot_be_resolved_is_input_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    os.rmdir(gone)  # the working directory is now deleted: a relative out_dir cannot be resolved
    with pytest.raises(InputError, match="cannot resolve out_dir"):
        ops._resolve_out_dir("run")
    # control: an absolute out_dir under the root resolves without consulting the working directory
    assert ops._resolve_out_dir(tmp_path / "run") == (tmp_path / "run").resolve()
