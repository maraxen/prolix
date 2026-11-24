"""Shared fixtures for tests/agent.

The CPU guard runs at session start: ``import jax`` happens early in pytest, so ``JAX_PLATFORMS``
set here would come too late. A non-CPU default backend stops the session with a message that
names the command to run (spec section 7). Every test runs with HOME and the XDG config dirs
pointed at temporary directories and the xtrax/myxcel/prolix environment variables cleared.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REAL_HOME = Path(os.path.expanduser("~"))


def pytest_sessionstart(session):
    import jax

    backend = jax.default_backend()
    if backend != "cpu":
        pytest.exit(
            f"tests/agent run on the CPU backend only (got {backend!r}); run: JAX_PLATFORMS=cpu pytest tests/agent",
            returncode=3,
        )


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Private HOME and XDG dirs; clear the env that changes caches, Pallas or myxcel config."""
    home = tmp_path / "_home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "_xdg"))
    for var in list(os.environ):
        if var.startswith(("XTRAX_JAX_CACHE", "MYXCEL_", "PROLIX_")) or var == "JAX_COMPILATION_CACHE_DIR":
            monkeypatch.delenv(var, raising=False)
    yield
