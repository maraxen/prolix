"""XR-VACUUM-DT: EnsemblePlan dt is femtoseconds; gamma is ps⁻¹."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prolix.physics.kups_adapter import dt_fs_to_akma

PDB = Path("data/pdb/2GB1.pdb")


@pytest.mark.slow
def test_2gb1_ensemble_plan_dt05fs_gamma50_1000_steps_finite():
    """AC1: vacuum 2GB1 holds 1000 steps at dt=0.5 fs with gamma≥50 ps⁻¹."""
    if not PDB.exists():
        pytest.skip(f"missing fixture {PDB}")

    import importlib.util

    from prolix.api import EnsemblePlan

    spec = importlib.util.spec_from_file_location(
        "b1_paramize", Path("scripts/benchmarks/_b1_paramize.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bundle = mod.paramize_pdb_to_bundle(str(PDB))
    traj = EnsemblePlan.from_bundles([bundle]).run(
        n_steps=1000, dt=0.5, kT=0.596, seed=0, gamma=50.0
    )
    final = traj[0] if isinstance(traj, list) else traj
    pos = np.asarray(final.positions)
    assert np.all(np.isfinite(pos)), (
        f"non-finite trajectory frac={np.mean(np.isfinite(pos)):.4f}"
    )


def test_dt_unit_akma_escape_hatch_matches_fs():
    """AC3: dt_unit='akma' with converted dt matches default fs path (toy)."""
    from prolix.api import EnsemblePlan

    # Import helper without package path assumptions
    import importlib.util

    ep_path = Path("tests/api/test_ensemble_plan.py")
    spec = importlib.util.spec_from_file_location("test_ensemble_plan_helpers", ep_path)
    assert spec is not None and spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    bundle = helpers._make_minimal_bundle(n_atoms=3)
    plan = EnsemblePlan.from_bundles([bundle])
    traj_fs = plan.run(n_steps=5, dt=0.5, kT=0.596, seed=0, dt_unit="fs")
    traj_ak = plan.run(
        n_steps=5, dt=dt_fs_to_akma(0.5), kT=0.596, seed=0, dt_unit="akma"
    )
    pos_fs = np.asarray(
        (traj_fs[0] if isinstance(traj_fs, list) else traj_fs).positions
    )
    pos_ak = np.asarray(
        (traj_ak[0] if isinstance(traj_ak, list) else traj_ak).positions
    )
    np.testing.assert_allclose(pos_fs, pos_ak, rtol=0, atol=0)
