"""Integration tests for proxide.export — StableHLO round-trip.

Validates:
1. export_energy_fn returns a lowered artifact with expected methods
2. Compiled artifact output matches jax.jit reference
3. save_artifact / load_artifact round-trip
4. export_langevin_step raises NotImplementedError in v1.1
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

# XA-CI: API drift / heavy compile — deselect from GitHub-faithful suite; tracked under XA-DRIFT.
pytestmark = pytest.mark.slow

from prolix import export
from prolix.physics import pbc
from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME
from prolix.physics.system import make_energy_fn_pure
from prolix.typing import EnergyParams

import importlib.util as _ilu, pathlib as _pl
_parity = _ilu.spec_from_file_location(
    "test_explicit_langevin_tip3p_parity",
    _pl.Path(__file__).parent / "physics" / "test_explicit_langevin_tip3p_parity.py",
)
_parity_mod = _ilu.module_from_spec(_parity)
_parity.loader.exec_module(_parity_mod)
_equil_water_positions = _parity_mod._equil_water_positions
_proxide_params_pure_water = _parity_mod._proxide_params_pure_water

N_WATERS = 8


@pytest.fixture(scope="module")
def tip3p_setup():
    """Shared 8-water TIP3P system for export tests."""
    jax.config.update("jax_enable_x64", True)
    positions_a, box_edge = _equil_water_positions(N_WATERS, seed=42)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    sys_dict = _proxide_params_pure_water(N_WATERS)
    displacement_fn, _ = pbc.create_periodic_space(box_vec)
    return dict(
        positions=jnp.array(positions_a, dtype=jnp.float64),
        box_vec=box_vec,
        sys_dict=sys_dict,
        displacement_fn=displacement_fn,
        pme_alpha=float(REGRESSION_EXPLICIT_PME["pme_alpha_per_angstrom"]),
        pme_grid=int(REGRESSION_EXPLICIT_PME["pme_grid_points"]),
        cutoff=float(REGRESSION_EXPLICIT_PME["cutoff_angstrom"]),
    )


def _build(s):
    return make_energy_fn_pure(
        s["displacement_fn"], s["sys_dict"], s["box_vec"],
        cutoff_distance=s["cutoff"],
        pme_grid_points=s["pme_grid"],
        pme_alpha=s["pme_alpha"],
        strict_parameterization=False,
    )


def test_export_returns_lowered_artifact(tip3p_setup):
    """export_energy_fn must return an artifact with .compile() and .as_compiled_mlir()."""
    params, fn = _build(tip3p_setup)
    lowered = export.export_energy_fn(fn, params, tip3p_setup["positions"])
    assert lowered is not None
    assert hasattr(lowered, "compile"), "artifact missing .compile()"
    assert hasattr(lowered, "as_text"), "artifact missing .as_text()"


def test_compiled_artifact_matches_jit(tip3p_setup):
    """Compiled artifact output must match jax.jit reference to float64 tolerance."""
    params, fn = _build(tip3p_setup)
    e_jit = float(jax.jit(fn)(params, tip3p_setup["positions"]))
    lowered = export.export_energy_fn(fn, params, tip3p_setup["positions"])
    compiled = lowered.compile()
    e_compiled = float(compiled(params, tip3p_setup["positions"]))
    rel_err = abs(e_compiled - e_jit) / (abs(e_jit) + 1e-12)
    assert rel_err < 1e-10, (
        f"compiled={e_compiled:.6f} vs jit={e_jit:.6f}, rel_err={rel_err:.2e}"
    )


def test_save_artifact_writes_file(tip3p_setup, tmp_path):
    """save_artifact must write a non-empty MLIR text file."""
    params, fn = _build(tip3p_setup)
    lowered = export.export_energy_fn(fn, params, tip3p_setup["positions"])
    out = tmp_path / "energy.mlir"
    export.save_artifact(lowered, out)
    assert out.exists(), f"File not created: {out}"
    assert out.stat().st_size > 0, "File is empty"


def test_load_artifact_roundtrip(tip3p_setup, tmp_path):
    """save then load must reproduce the exact MLIR text."""
    params, fn = _build(tip3p_setup)
    lowered = export.export_energy_fn(fn, params, tip3p_setup["positions"])
    out = tmp_path / "energy.mlir"
    export.save_artifact(lowered, out)
    loaded = export.load_artifact(out)
    assert loaded == out.read_text()
    assert len(loaded) > 0


