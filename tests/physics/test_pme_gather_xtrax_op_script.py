"""The pme_gather_xtrax_op experiment's error metric must fail on non-finite output."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/experiments/pme_gather_xtrax_op.py"


def _load():
    spec = importlib.util.spec_from_file_location("pme_gather_xtrax_op", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_finite_error_is_scale_relative():
    mod = _load()
    want = (np.array([200.0, -100.0]), np.array([0.5]))
    got = (np.array([200.0, -98.0]), np.array([0.5]))
    assert mod.scale_relative_error(got, want) == 2.0 / 200.0


def test_nan_after_first_leaf_is_infinite():
    # Python max([3e-7, nan]) == 3e-7: a NaN in the second leaf must not be dropped.
    mod = _load()
    want = (np.ones(4), np.ones(4))
    got = (np.ones(4), np.array([1.0, np.nan, 1.0, 1.0]))
    err = mod.scale_relative_error(got, want)
    assert math.isinf(err)
    assert not err <= 1e-4


def test_nan_in_reference_is_infinite():
    mod = _load()
    want = (np.array([1.0, np.nan]),)
    got = (np.array([1.0, 1.0]),)
    assert math.isinf(mod.scale_relative_error(got, want))
