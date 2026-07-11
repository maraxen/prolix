#!/usr/bin/env python3
"""CI gate for XR-PIN: xtrax must be PyPI-resolved in [FLOOR, UPPER).

FLOOR/UPPER must stay identical to pyproject.toml ``xtrax>=FLOOR,<UPPER``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Keep in sync with pyproject.toml: xtrax>=0.4.0a5,<0.5
FLOOR = "0.4.0a5"
UPPER = "0.5"


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _check_not_path_install() -> str | None:
    """Return error message if xtrax looks like a path/editable install."""
    try:
        from importlib.metadata import distribution
    except ImportError:
        return None

    try:
        dist = distribution("xtrax")
    except Exception:
        return None

    direct = dist.read_text("direct_url.json")
    if not direct:
        return None
    try:
        data = json.loads(direct)
    except json.JSONDecodeError:
        return f"unreadable direct_url.json for xtrax: {direct[:200]!r}"

    url = str(data.get("url", ""))
    if url.startswith("file:") or data.get("dir_info") is not None:
        return (
            f"xtrax is a path/editable install ({url or 'dir_info'}), "
            "not PyPI — XR-PIN requires registry source"
        )
    return None


def _check_uv_lock_registry(repo_root: Path) -> str | None:
    """Return error if uv.lock pins xtrax from a non-registry source."""
    lock = repo_root / "uv.lock"
    if not lock.is_file():
        return None
    text = lock.read_text()
    # Find the [[package]] block for name = "xtrax"
    marker = 'name = "xtrax"'
    idx = text.find(marker)
    if idx < 0:
        return "uv.lock has no xtrax package entry"
    # Prefer the standalone package stanza (not dependency list lines)
    # Scan forward from each occurrence until we find version+source nearby
    search_from = 0
    while True:
        idx = text.find(marker, search_from)
        if idx < 0:
            return "uv.lock xtrax entry missing version/source stanza"
        window = text[idx : idx + 400]
        if "version =" in window and "source =" in window:
            if 'source = { registry =' in window or 'source = {registry =' in window:
                return None
            if "source = { path" in window or "source = { directory" in window:
                return f"uv.lock xtrax source is not PyPI registry:\n{window.split(chr(10))[0:6]}"
            if "source = { git" in window:
                return "uv.lock xtrax source is git, not PyPI registry"
            return f"uv.lock xtrax source is not registry:\n{window[:200]}"
        search_from = idx + len(marker)


def main() -> int:
    try:
        import xtrax
    except ImportError as e:
        return _fail(f"xtrax not importable: {e}")

    ver = getattr(xtrax, "__version__", None)
    if not ver:
        return _fail("xtrax has no __version__ (fail closed)")

    try:
        from packaging.version import Version
    except ImportError:
        return _fail("packaging is required for PEP 440 compare")

    v, lo, hi = Version(ver), Version(FLOOR), Version(UPPER)
    if v < lo or v >= hi:
        return _fail(f"xtrax {ver} outside allowed range [{FLOOR}, {UPPER})")

    path_err = _check_not_path_install()
    if path_err:
        return _fail(path_err)

    repo_root = Path(__file__).resolve().parents[2]
    lock_err = _check_uv_lock_registry(repo_root)
    if lock_err:
        return _fail(lock_err)

    print(f"OK: xtrax {ver} satisfies >={FLOOR},<{UPPER} (PyPI/registry)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
