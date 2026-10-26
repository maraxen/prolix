"""Locate ``proxide`` assets/test-data without requiring a sibling ``../proxide`` checkout.

Resolution order for packaged assets (force fields etc.):

1. the installed ``proxide`` package (``proxide/assets``) -- present on every remote/CI venv;
2. a sibling development checkout ``<prolix>/../proxide/src/proxide/assets``.

Proxide's *test data* (``proxide/tests/data``) is not shipped in the wheel; those resolve
only from a sibling checkout and ``pytest.skip`` with a clear reason otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SIBLING = _REPO_ROOT.parent / "proxide"


def _installed_assets_dir() -> Path | None:
    try:
        import proxide
    except ImportError:
        return None
    p = Path(proxide.__file__).resolve().parent / "assets"
    return p if p.is_dir() else None


def proxide_asset(*parts: str) -> Path:
    """Path to a packaged proxide asset, e.g. ``proxide_asset("protein.ff19SB.xml")``.

    Returns the installed-package path when it exists, else the sibling-checkout path (which may
    not exist; callers get a clear FileNotFoundError at use rather than a silent skip).
    """
    installed = _installed_assets_dir()
    if installed is not None and installed.joinpath(*parts).exists():
        return installed.joinpath(*parts)
    return _SIBLING / "src" / "proxide" / "assets" / Path(*parts)


def proxide_test_data(*parts: str) -> Path:
    """Path to a file under ``proxide/tests/data`` (sibling checkout only); skips if absent."""
    p = _SIBLING / "tests" / "data" / Path(*parts)
    if not p.exists():
        pytest.skip(f"requires proxide test data {p} (sibling ../proxide checkout not present)")
    return p
