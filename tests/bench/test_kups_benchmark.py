"""Unit and smoke tests for the kUPS benchmark and Phase 3 memory/flash harness."""

from __future__ import annotations

import math

from prolix.physics import kups_adapter
from prolix.simulate import AKMA_TIME_UNIT_FS
from scripts.benchmarks.kups_benchmark import (
    run_tier1_parity,
    run_tier2_thermostat,
    run_tier3_throughput,
)
from scripts.benchmarks.pallas_vs_baseline import (
    _run_isolated_worker,
    build_system,
    make_flash_fns,
)


def test_kups_adapter_extended_unit_identities():
    """Verify exact dynamical unit equivalence between kUPS (eV/A/amu) and Prolix (AKMA)."""
    # Fundamental identity: (t_akma / t_kups) = sqrt(EV_TO_KCAL_MOL)
    ratio = AKMA_TIME_UNIT_FS * kups_adapter.KUPS_FEMTOSECOND
    assert math.isclose(ratio, math.sqrt(kups_adapter.EV_TO_KCAL_MOL), rel_tol=1e-14)

    for dt_fs in (0.5, 1.0, 2.0):
        dt_k = kups_adapter.dt_fs_to_kups(dt_fs)
        assert math.isclose(kups_adapter.dt_kups_to_fs(dt_k), dt_fs, rel_tol=1e-14)
        # Dimensionless product gamma * dt must be identical in both unit systems
        for gamma_ps in (1.0, 10.0):
            g_k = kups_adapter.gamma_ps_to_kups(gamma_ps)
            g_a = kups_adapter.gamma_ps_to_akma(gamma_ps)
            dt_a = kups_adapter.dt_fs_to_akma(dt_fs)
            assert math.isclose(g_k * dt_k, g_a * dt_a, rel_tol=1e-14)

    for e_ev in (0.1, 1.5, -12.3):
        e_kcal = kups_adapter.energy_ev_to_kcal_mol(e_ev)
        assert math.isclose(kups_adapter.energy_kcal_mol_to_ev(e_kcal), e_ev, rel_tol=1e-14)

    for p_kups in (0.25, -3.5):
        p_akma = kups_adapter.momentum_kups_to_akma(p_kups)
        assert math.isclose(kups_adapter.momentum_akma_to_kups(p_akma), p_kups, rel_tol=1e-14)


def test_tier1_single_step_and_nonbonded_parity_smoke():
    """Tier 1A/1B parity at small N confirms <1e-12 float64 and <1e-4 float32 agreement."""
    res = run_tier1_parity(sizes=[64], interpret=True)
    for row in res["harmonic_single_step_f64"]:
        assert row["rel_dE_f64"] < 1e-12
        assert row["max_abs_dF_kcal_mol_A_f64"] < 1e-12
        assert row["baoab_max_pos_err_A_f64"] < 1e-12
        assert row["baoab_max_mom_err_akma_f64"] < 1e-12
        assert row["csvr_max_pos_err_A_f64"] < 1e-12
        assert row["csvr_max_mom_err_akma_f64"] < 1e-12

    nb = res["nonbonded_parity"][0]
    assert nb["xla_vs_kups_rel_dE_f64"] < 1e-11
    assert nb["xla_vs_kups_max_abs_dF_f64"] < 1e-10
    assert nb["pallas_vs_kups_rel_dE_f32"] < 1e-5
    assert nb["pallas_vs_kups_max_abs_dF_f32"] < 1e-3


def test_tier2_thermostat_ensemble_trajectory_identity():
    """With matched PRNG and float64, kUPS and Prolix trajectories have <1e-10 K stepwise diff."""
    rows = run_tier2_thermostat(n_particles=32, n_equil=50, n_sample=100)
    for r in rows:
        assert r["max_stepwise_temp_diff_k"] < 1e-9
        assert r["delta_t_k"] < 1e-9


def test_tier3_propagator_throughput_and_flash_harness_smoke():
    """Smoke test Tier 3 propagator scan and Phase 3 flash/worker harness at N=64."""
    rows, _ = run_tier3_throughput(sizes=[64], n_steps=3, repeats=1, interpret=True)
    assert len(rows) == 1
    assert rows[0]["trajectory_rel_dE_prolix_xla_vs_kups"] < 1e-5
    assert rows[0]["trajectory_rel_dE_prolix_pallas_vs_kups"] < 1e-5

    s = build_system(64, include_nl=False)
    xla_vg, pal_vg = make_flash_fns(s, interpret=True)
    e_x, g_x = xla_vg(s["pos"])
    e_p, g_p = pal_vg(s["pos"])
    assert abs(float(e_x - e_p)) / max(abs(float(e_x)), 1.0) < 1e-5
    assert float( abs(g_x - g_p).max() ) < 1e-3

    w = _run_isolated_worker("flash_xla:64", repeats=1, interpret=True)
    assert w["n_atoms"] == 64
    assert math.isfinite(w["energy"])
