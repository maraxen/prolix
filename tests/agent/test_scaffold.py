"""Task 1 scaffold gates: import lightness, entry-package isolation, missing-extra hint, allowlist file.

Subprocess tests keep the parent pytest process free of the heavy modules they check.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HEAVY = ("cisternal", "fastmcp", "cyclopts")


def _run(code: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "JAX_PLATFORMS": "cpu", **(env_extra or {})}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=300,
    )


def test_import_prolix_agent_loads_no_agent_extra_modules():
    proc = _run(
        f"""
        import sys
        import prolix.agent, prolix.agent.errors, prolix.agent._runtime
        leaked = [m for m in {HEAVY!r} if m in sys.modules]
        print("LEAKED", leaked)
        raise SystemExit(0 if not leaked else 3)
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_positive_control_heavy_modules_are_detected():
    proc = _run(
        f"""
        import sys
        import cisternal, fastmcp, cyclopts
        present = [m for m in {HEAVY!r} if m in sys.modules]
        raise SystemExit(0 if len(present) == 3 else 3)
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_import_entry_package_leaves_prolix_unimported():
    proc = _run(
        """
        import sys
        import prolix_agent_entry
        raise SystemExit(0 if "prolix" not in sys.modules else 3)
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_prepare_overrides_preset_jax_platforms():
    proc = _run(
        """
        import os
        import prolix_agent_entry
        prolix_agent_entry.prepare()
        print(os.environ["JAX_PLATFORMS"])
        """,
        env_extra={"JAX_PLATFORMS": "gpu"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "cpu"


def test_cli_entry_exits_1_with_hint_when_cisternal_missing():
    proc = _run(
        """
        import sys
        sys.modules["cisternal"] = None  # import of cisternal now fails
        import prolix_agent_entry
        prolix_agent_entry.cli_main()
        """
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "prolix[agent]" in proc.stderr
    assert "missing: cisternal" in proc.stderr


def test_require_agent_extra_raises_import_error_with_hint(monkeypatch):
    import importlib.util

    from prolix.agent import AgentExtraMissing, require_agent_extra

    real = importlib.util.find_spec

    def fake(name, *a, **k):
        return None if name == "fastmcp" else real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    with pytest.raises(AgentExtraMissing) as info:
        require_agent_extra()
    assert isinstance(info.value, ImportError)
    assert "prolix[agent]" in str(info.value)


def test_job_allowlist_file_is_exactly_one_path_line():
    lines = (REPO / "src" / "prolix" / "agent" / "job_scripts.txt").read_text().splitlines()
    assert lines == ["scripts/benchmarks/prolix_vs_openmm_speed.py"]
    assert (REPO / lines[0]).is_file()
