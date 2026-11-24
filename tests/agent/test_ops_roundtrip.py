"""Task 2 gates: operation round trips on an injected tiny bundle (CI group, no proxide load).

Each op writes side files then ``<tool>.json``; the tests check the record, golden agreement with
the library, the dt/mass unit handling, and that refusals write nothing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import prolix.physics.simulate as sim
from prolix.agent import _runtime as rt
from prolix.agent import ops
from prolix.agent.errors import InputError, RefusalError
from prolix.simulate import AKMA_TIME_UNIT_FS


def _tiny_bundle():
    from prolix.physics.system import make_bundle_from_system

    system = SimpleNamespace(
        positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]),
        masses=np.array([12.0, 1.0, 16.0]),
        bonds=np.array([[0, 1]]),
        bond_params=np.array([[300.0, 1.1]]),
        angles=None,
        angle_params=None,
        dihedrals=None,
        dihedral_params=None,
        impropers=None,
        improper_params=None,
        charges=np.array([0.4, -0.2, -0.2]),
        sigmas=np.full(3, 3.15),
        epsilons=np.full(3, 0.1),
    )
    return make_bundle_from_system(system, boundary_condition="free")


@pytest.fixture(scope="module")
def bundle():
    return _tiny_bundle()


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    monkeypatch.delenv("PROLIX_USE_PALLAS", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _out(tmp_path, name="out"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    return str(d)


def _record(out, tool):
    with open(f"{out}/{tool}.json") as fh:
        return json.load(fh)


def _no_tmp_left(out):
    import os

    return not [f for f in os.listdir(out) if f.endswith(".tmp") or ".tmp." in f]


def test_load_system_record_schema(bundle, _root):
    out = _out(_root)
    res = ops.load_system(out_dir=out, bundle=bundle)
    rec = _record(out, "load_system")
    assert rec["schema"] == "prolix.agent.run/1"
    assert rec["tool"] == "load_system"
    assert res["result"]["n_real_atoms"] == 3
    assert res["result"]["term_counts"]["bonds"] == 1
    assert res["result"]["bundle_cache"] == "injected"
    assert rec["jax"]["backend"] == "cpu"
    assert set(rec["versions"]) >= {"prolix", "jax", "python"}
    assert "jax_cache" in rec
    assert _no_tmp_left(out)


def test_inspect_params_rows_truncation_and_input_checks(bundle, _root):
    out = _out(_root)
    res = ops.inspect_params(out_dir=out, bundle=bundle, term="bond", limit=5)
    assert res["result"]["total_rows"] == 1
    assert res["result"]["truncated"] is False
    assert res["result"]["rows"][0]["atoms"] == [0, 1]
    with pytest.raises(InputError):
        ops.inspect_params(out_dir=out, bundle=bundle, term="torsion")
    with pytest.raises(InputError):
        ops.inspect_params(out_dir=out, bundle=bundle, term="charge", limit=0)
    with pytest.raises(InputError):
        ops.inspect_params(out_dir=out, bundle=bundle, term="charge", limit=201)


def test_energy_breakdown_matches_library(bundle, _root):
    from prolix.api.bundle_md import (
        active_positions,
        bonded_energy_fn_from_bundle,
        energy_fn_from_bundle,
    )

    out = _out(_root)
    res = ops.energy_breakdown(out_dir=out, bundle=bundle)["result"]
    x = active_positions(bundle)
    e_total = float(energy_fn_from_bundle(bundle)(x))
    e_bonded = float(bonded_energy_fn_from_bundle(bundle)(x))
    assert res["total"] == pytest.approx(e_total, rel=1e-6)
    assert res["bonded"] == pytest.approx(e_bonded, rel=1e-6)
    assert res["nonbonded_by_difference"] == pytest.approx(e_total - e_bonded, rel=1e-6)
    assert res["median_force"] > 0
    # control: moving one atom 0.1 A changes the total beyond tolerance
    moved = x.at[1, 0].add(0.1)
    assert abs(float(energy_fn_from_bundle(bundle)(moved)) - e_total) > 1e-3 * abs(e_total)


def test_minimize_matches_library_and_lowers_energy(bundle, _root):
    from prolix.api.bundle_md import active_positions, energy_fn_from_bundle

    out = _out(_root)
    res = ops.minimize(out_dir=out, bundle=bundle, sd_steps=20)
    x0 = active_positions(bundle)
    ef = energy_fn_from_bundle(bundle)
    expected = sim.run_minimization(ef, x0, sd_steps=20)
    with np.load(f"{out}/minimized.npz") as z:
        np.testing.assert_allclose(z["positions"], np.asarray(expected), rtol=1e-6, atol=1e-8)
    assert res["result"]["energy_after"] <= res["result"]["energy_before"]
    rec = _record(out, "minimize")
    assert rec["outputs"][0]["sha256"] == rt.sha256_file(f"{out}/minimized.npz")
    assert _no_tmp_left(out)


def test_md_nve_uses_bundle_masses_and_akma_dt(bundle, _root, monkeypatch):
    seen = {}

    def fake_nve(energy_fn, x0, steps, dt, mass=1.0, **kw):
        seen.update(steps=steps, dt=dt, mass=np.asarray(mass))
        return x0

    monkeypatch.setattr(sim, "run_nve", fake_nve)
    out = _out(_root)
    res = ops.md_smoke(out_dir=out, bundle=bundle, steps=7, dt_fs=2.0)["result"]
    assert seen["steps"] == 7
    assert seen["dt"] == pytest.approx(2.0 / AKMA_TIME_UNIT_FS)
    np.testing.assert_allclose(seen["mass"], np.asarray(bundle.masses[:3]))
    assert res["init_velocity_temperature_k"] == 300.0
    assert res["seed"] == "fixed 0"
    assert res["conservation"] == "not claimed"


def test_md_nve_rejects_temperature(bundle, _root):
    with pytest.raises(InputError):
        ops.md_smoke(out_dir=_out(_root), bundle=bundle, ensemble="nve", temperature_k=310.0)


def test_md_nvt_forwards_temperature_and_mass(bundle, _root, monkeypatch):
    seen = {}

    def fake_nvt(energy_fn, x0, steps, dt, temperature=300.0, mass=1.0, **kw):
        seen.update(temperature=temperature, mass=np.asarray(mass))
        return x0

    monkeypatch.setattr(sim, "run_nvt_nose_hoover", fake_nvt)
    res = ops.md_smoke(
        out_dir=_out(_root), bundle=bundle, steps=3, dt_fs=1.0, ensemble="nvt", temperature_k=310.0
    )["result"]
    assert seen["temperature"] == 310.0
    np.testing.assert_allclose(seen["mass"], np.asarray(bundle.masses[:3]))
    assert res["temperature_k"] == 310.0


def test_md_smoke_real_integrator_is_finite(bundle, _root):
    out = _out(_root)
    res = ops.md_smoke(out_dir=out, bundle=bundle, steps=10, dt_fs=0.5)["result"]
    assert res["finite"] is True
    with np.load(f"{out}/final.npz") as z:
        assert z["positions"].shape == (3, 3)
    assert jax.default_backend() == "cpu"


def test_refusals_write_nothing(bundle, _root, monkeypatch):
    out = _out(_root, "refused")
    monkeypatch.setattr(rt, "MAX_ATOMS", 2)
    with pytest.raises(RefusalError):
        ops.load_system(out_dir=out, bundle=bundle)
    monkeypatch.setattr(rt, "MAX_ATOMS", 1000)
    with pytest.raises(RefusalError):
        ops.minimize(out_dir=out, bundle=bundle, sd_steps=rt.MAX_MIN_STEPS + 1)
    with pytest.raises(RefusalError):
        ops.md_smoke(out_dir=out, bundle=bundle, steps=rt.MAX_MD_STEPS + 1)
    with pytest.raises(InputError):
        ops.md_smoke(out_dir=out, bundle=bundle, dt_fs=rt.MAX_DT_FS + 0.1)
    with pytest.raises(InputError):
        ops.md_smoke(out_dir=out, bundle=bundle, dt_fs=0.0)
    assert _no_tmp_left(out)
    import os

    assert os.listdir(out) == []


def test_existing_record_needs_overwrite_and_out_root_is_enforced(bundle, _root, tmp_path):
    out = _out(_root)
    ops.load_system(out_dir=out, bundle=bundle)
    with pytest.raises(InputError):
        ops.load_system(out_dir=out, bundle=bundle)
    ops.load_system(out_dir=out, bundle=bundle, overwrite=True)
    outside = tmp_path.parent / "elsewhere_for_prolix_tests"
    with pytest.raises(InputError):
        ops.load_system(out_dir=str(outside), bundle=bundle)
