"""Tests for the base xtrax floor parsed by scripts/ci/check_xtrax_pin.py (spec section 8).

Tests the pure parser only. ``main()`` also checks for a path install and uv.lock, so it fails
by design in a local dev checkout that points xtrax at a local path.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("check_xtrax_pin", REPO / "scripts" / "ci" / "check_xtrax_pin.py")
assert _SPEC is not None and _SPEC.loader is not None
pin = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pin)


def _pyproject_text(deps: list[str], agent: list[str] | None = None) -> str:
    lines = ["[project]", 'name = "x"', "dependencies = ["]
    lines += [f'  "{d}",' for d in deps]
    lines += ["]"]
    if agent is not None:
        lines += ["[project.optional-dependencies]", "agent = ["]
        lines += [f'  "{d}",' for d in agent]
        lines += ["]"]
    return "\n".join(lines) + "\n"


def test_real_pyproject_base_floor_matches_checker_constants():
    text = (REPO / "pyproject.toml").read_text()
    floor, upper = pin.base_xtrax_bounds(text)
    assert (floor, upper) == (pin.FLOOR, pin.UPPER)
    assert Version(floor) == Version("0.4.1b2")
    assert upper == "0.5"


def test_real_agent_extra_floor_is_at_least_base_floor():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    agent = data["project"]["optional-dependencies"]["agent"]
    extra = [Requirement(a) for a in agent if Requirement(a).name == "xtrax"]
    assert len(extra) == 1
    assert "plugin" in extra[0].extras
    floor = next(s.version for s in extra[0].specifier if s.operator == ">=")
    base_floor, _ = pin.base_xtrax_bounds((REPO / "pyproject.toml").read_text())
    assert Version(floor) >= Version(base_floor)


def test_extra_entry_with_extras_is_not_taken_as_base():
    text = _pyproject_text(["xtrax>=0.4.1b2,<0.5"], agent=["xtrax[plugin]>=0.4.1b10,<0.5"])
    assert pin.base_xtrax_bounds(text) == ("0.4.1b2", "0.5")


def test_only_extra_entry_fails_closed():
    text = _pyproject_text(["numpy"], agent=["xtrax[plugin]>=0.4.1b10,<0.5"])
    with pytest.raises(ValueError, match="expected exactly one base xtrax"):
        pin.base_xtrax_bounds(text)


def test_missing_upper_bound_fails_closed():
    text = _pyproject_text(["xtrax>=0.4.1b2"])
    with pytest.raises(ValueError, match="one >= floor and one < upper"):
        pin.base_xtrax_bounds(text)
