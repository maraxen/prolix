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


JAX_CACHE_FLAGS = (
    "jax_compilation_cache_dir",
    "jax_persistent_cache_min_compile_time_secs",
    "jax_persistent_cache_min_entry_size_bytes",
    "jax_compilation_cache_max_size",
)


def _snapshot_jax_cache_state() -> tuple[dict[str, object], object]:
    """The JAX config flags and xtrax's applied-setting marker that a test may change."""
    import jax
    from xtrax import jax_cache

    flags = {name: getattr(jax.config, name) for name in JAX_CACHE_FLAGS}
    return flags, jax_cache._applied


def _restore_jax_cache_state(snapshot: tuple[dict[str, object], object]) -> None:
    import jax
    from xtrax import jax_cache

    flags, applied = snapshot
    for name, value in flags.items():
        if getattr(jax.config, name) != value:
            jax.config.update(name, value)
    jax_cache._applied = applied


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Private HOME and XDG dirs; cisternal logs to a temp dir; JAX cache state restored after each test."""
    home = tmp_path / "_home"
    home.mkdir()
    xdg = tmp_path / "_xdg"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    for var in list(os.environ):
        if var.startswith(("XTRAX_JAX_CACHE", "MYXCEL_", "PROLIX_")) or var == "JAX_COMPILATION_CACHE_DIR":
            monkeypatch.delenv(var, raising=False)
    # cisternal reads [telemetry] log_dir from $XDG_CONFIG_HOME/cisternal/config.toml
    cfg = xdg / "cisternal"
    cfg.mkdir(parents=True)
    log_dir = tmp_path / "_cisternal_logs"
    (cfg / "config.toml").write_text(f'[telemetry]\nlog_dir = "{log_dir.as_posix()}"\n')
    snapshot = _snapshot_jax_cache_state()
    try:
        yield
    finally:
        _restore_jax_cache_state(snapshot)
