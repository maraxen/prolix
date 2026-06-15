"""Smoke tests verifying HP1 deprecated symbols emit DeprecationWarning on access."""
import importlib
import warnings

import pytest

pytestmark = pytest.mark.unit


def _access_with_warnings(module_name: str, attr: str) -> list:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = importlib.import_module(module_name)
        _ = getattr(m, attr)
    return [x for x in w if issubclass(x.category, DeprecationWarning)]


def test_langevin_state_deprecated():
    """LangevinState accessed from batched_simulate must emit DeprecationWarning."""
    caught = _access_with_warnings("prolix.batched_simulate", "LangevinState")
    assert caught, "Expected DeprecationWarning for prolix.batched_simulate.LangevinState"
    assert "EnsemblePlan" in str(caught[0].message) or "prolix.types" in str(caught[0].message)


def test_padded_system_deprecated():
    """PaddedSystem accessed from prolix must emit DeprecationWarning."""
    caught = _access_with_warnings("prolix", "PaddedSystem")
    assert caught, "Expected DeprecationWarning for prolix.PaddedSystem"


def test_batched_produce_is_callable():
    """batched_produce still exists and is callable (warns on call, tested elsewhere)."""
    import prolix.batched_simulate as bs

    assert callable(bs.batched_produce)


def test_pad_protein_is_callable():
    """pad_protein still exists and is callable (warns on call, tested elsewhere)."""
    import prolix.padding as p

    assert callable(p.pad_protein)
