"""XR-A2A3 A2: 2GB1 force-scale gate after ExclusionSpec wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PDB = Path("data/pdb/2GB1.pdb")
_PARAMIZE = Path("scripts/benchmarks/_b1_paramize.py")


def _load_paramize():
    spec = importlib.util.spec_from_file_location("b1_paramize", _PARAMIZE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_2gb1_paramize_median_grad_under_1e3():
    """C1: exclusions drop median |grad| from ~1e5 to < 1e3 kcal/mol/Å."""
    if not PDB.exists():
        pytest.skip(f"missing fixture {PDB}")

    mod = _load_paramize()
    bundle = mod.paramize_pdb_to_bundle(str(PDB))
    assert int(bundle.excl_mask.sum()) > 0
    stats = mod.assert_force_scale_ok(bundle)
    assert stats["median_abs_grad"] < 1e3
    assert stats["n_real"] > 0
