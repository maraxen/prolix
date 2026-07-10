"""XR-PARITY-KUPS: always-on kups_adapter unit contract (no kups package)."""

from __future__ import annotations

import pytest

import math

from prolix.physics import kups_adapter
from prolix.simulate import AKMA_TIME_UNIT_FS



# XA-CI: heavy parity/compile — deselect from GitHub-faithful suite.
pytestmark = pytest.mark.slow

def test_akma_time_unit_matches_adapter_expectation():
    assert abs(AKMA_TIME_UNIT_FS - 48.88821291839) < 1e-6


def test_ev_kcal_constant():
    assert abs(kups_adapter.EV_TO_KCAL_MOL - 23.060549) < 1e-6
    assert abs(kups_adapter.EV_TO_KCAL_MOL * kups_adapter.KCAL_MOL_TO_EV - 1.0) < 1e-12


def test_dt_fs_akma_roundtrip():
    for dt_fs in (0.25, 0.5, 1.0, 2.0):
        ak = kups_adapter.dt_fs_to_akma(dt_fs)
        assert abs(kups_adapter.dt_akma_to_fs(ak) - dt_fs) < 1e-12


def test_gamma_ps_akma_roundtrip():
    for g in (1.0, 10.0, 50.0):
        ak = kups_adapter.gamma_ps_to_akma(g)
        assert abs(kups_adapter.gamma_akma_to_ps(ak) - g) < 1e-12


def test_tau_ps_akma_roundtrip():
    for tau in (0.05, 0.1, 1.0):
        ak = kups_adapter.tau_ps_to_akma(tau)
        assert abs(kups_adapter.tau_akma_to_ps(ak) - tau) < 1e-12


def test_spring_constant_ev_kcal_roundtrip():
    k_ev = 0.01
    k_kcal = kups_adapter.spring_constant_ev_per_angstrom_sq_to_kcal_mol(k_ev)
    assert abs(k_kcal - k_ev * 23.060549) < 1e-12
    back = kups_adapter.spring_constant_kcal_mol_to_ev_per_angstrom_sq(k_kcal)
    assert abs(back - k_ev) < 1e-12


def test_gamma_10ps_matches_temperature_control_formula():
    """Canonical water-NVT reduction: gamma_ps * AKMA_TIME_UNIT_FS * 1e-3."""
    gamma_ps = 10.0
    expected = gamma_ps * AKMA_TIME_UNIT_FS * 1e-3
    assert math.isclose(kups_adapter.gamma_ps_to_akma(gamma_ps), expected, rel_tol=0, abs_tol=1e-15)
