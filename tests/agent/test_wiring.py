"""Task 4 gates: registry parity, exit-code table, entry-point subprocess and device policy.

CLI tests run the installed ``prolix`` console script in a child process with HOME and XDG config
pointed at temporary directories, so cisternal and xtrax write nothing under the real home.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import cisternal
import cyclopts
import pytest
from cisternal.registration import registry as creg

from prolix.agent import tools
from prolix.agent.errors import InputError, RefusalError

REPO = Path(__file__).resolve().parents[2]
SCRIPT = "scripts/benchmarks/prolix_vs_openmm_speed.py"
ENTRY = "from prolix_agent_entry import cli_main; cli_main()"
EXPECTED = {
    "load_system", "inspect_params", "energy_breakdown", "minimize", "md_smoke", "render_job", "submit_job",
}


def _snapshot_names(name: str) -> set[str]:
    return set(creg.snapshot(name).keys())


# ---------------------------------------------------------------- registry parity


def test_expected_names_match_tool_names_and_both_registries():
    from prolix.agent import cli, mcp

    assert set(tools.TOOL_NAMES) == EXPECTED
    assert _snapshot_names(tools.CLI_REGISTRY) >= EXPECTED
    assert _snapshot_names(tools.MCP_REGISTRY) >= EXPECTED
    cli_names = set(cli.app.resolved_commands().keys()) - {"--help", "--version", "-h"}
    assert cli_names == EXPECTED
    server = mcp.build_server()
    mcp_names = {t.name for t in asyncio.run(server.list_tools())}
    assert mcp_names == EXPECTED


def test_cli_tools_are_sync_and_mcp_twins_are_async():
    for name in EXPECTED:
        cli_fn = creg.snapshot(tools.CLI_REGISTRY)[name].fn
        mcp_fn = creg.snapshot(tools.MCP_REGISTRY)[name].fn
        assert not inspect.iscoroutinefunction(cli_fn), name
        assert inspect.iscoroutinefunction(mcp_fn), name


def test_positive_control_missing_tool_makes_wire_raise():
    # ty 0.0.29 cannot resolve these lazy re-exports, so they are reached through getattr
    tool_decorator = getattr(cisternal, "tool")
    wire_fn = getattr(cisternal, "wire")
    app_cls = getattr(cyclopts, "App")

    @tool_decorator(registry="prolix_test_parity_control")
    def only_one(x: int = 1) -> dict:
        """One tool."""
        return {"x": x}

    with pytest.raises(Exception, match="never_registered"):
        wire_fn(
            None,
            app_cls(name="control"),
            registry="prolix_test_parity_control",
            expected=["only_one", "never_registered"],
        )


# ---------------------------------------------------------------- subprocess helpers


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XTRAX_JAX_CACHE", "MYXCEL_", "PROLIX_"))}
    env.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(tmp_path / "xdg"),
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PROLIX_AGENT_OUT_ROOT=str(tmp_path),
    )
    env.update(extra)
    return env


def _prolix(args: list[str], tmp_path: Path, **extra: str) -> subprocess.CompletedProcess[str]:
    # The running interpreter, not a .venv path: the tests work in any environment with the extra.
    return subprocess.run(
        [sys.executable, "-c", ENTRY, *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=_env(tmp_path, **extra),
        timeout=600,
        check=False,
    )


# ---------------------------------------------------------------- exit-code table (spec section 3)


def _checkout(root: Path) -> Path:
    """A minimal prolix checkout (the pyproject and uv helper that the root check requires)."""
    (root / "scripts" / "slurm").mkdir(parents=True)
    (root / "scripts" / "benchmarks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "prolix"\n')
    shutil.copy(REPO / "scripts" / "slurm" / "_uv_env.sh", root / "scripts" / "slurm" / "_uv_env.sh")
    shutil.copy(REPO / SCRIPT, root / SCRIPT)
    return root


def test_exit_0_success_prints_one_json_object(tmp_path):
    checkout = _checkout(tmp_path / "checkout")
    proc = _prolix(
        ["render_job", "--script", SCRIPT, "--name", "ok_job",
         "--preset", "cpu_quick", "--repo-root", str(checkout), "--json"],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)  # exactly one JSON object, nothing else on stdout
    assert payload["artifact"].endswith("scripts/slurm/agent_jobs/ok_job/job.slurm")
    assert Path(payload["artifact"]).is_file()


@pytest.mark.slow
def test_exit_2_usage_error(tmp_path):
    proc = _prolix(["load_system", "--bogus"], tmp_path)
    assert proc.returncode == 2, proc.stderr


@pytest.mark.slow
def test_exit_2_jax_cache_configuration_error_has_prefix(tmp_path):
    cfg = tmp_path / "xdg" / "xtrax"
    cfg.mkdir(parents=True)
    (cfg / "config.toml").write_text("this is = = not toml\n")
    proc = _prolix(["inspect_params", "--pdb", "x.pdb", "--term", "bond", "--out-dir", str(tmp_path)], tmp_path)
    assert proc.returncode == 2, proc.stderr
    assert "prolix: jax cache configuration error:" in proc.stderr


def test_exit_3_refusal_names_the_cap(tmp_path):
    proc = _prolix(
        ["minimize", "--pdb", str(tmp_path / "missing.pdb"), "--out-dir", str(tmp_path), "--sd-steps", "999999"],
        tmp_path,
    )
    assert proc.returncode == 3, proc.stderr
    assert "cap" in proc.stderr.lower() or "render_job" in proc.stderr


def test_exit_4_missing_file(tmp_path):
    proc = _prolix(["load_system", "--pdb", str(tmp_path / "nope.pdb"), "--out-dir", str(tmp_path)], tmp_path)
    assert proc.returncode == 4, proc.stderr


@pytest.mark.slow
def test_exit_1_missing_extra_prints_hint(tmp_path):
    code = textwrap.dedent(
        """
        import sys
        sys.modules["cisternal"] = None
        import prolix_agent_entry
        prolix_agent_entry.cli_main()
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path,
                          env=_env(tmp_path), timeout=300, check=False)
    assert proc.returncode == 1
    assert "prolix[agent]" in proc.stderr


# ---------------------------------------------------------------- device policy (spec section 11)


_CHILD = textwrap.dedent(
    """
    import atexit, json, sys
    def _report():
        import jax
        from jax._src import xla_bridge as xb
        sys.stdout.write("REPORT " + json.dumps({
            "initialised": bool(xb.backends_are_initialized()),
            "backends": sorted(jax.extend.backend.backends().keys()),
            "default": jax.default_backend(),
        }) + "\\n")
    atexit.register(_report)
    sys.argv = ["prolix", "--help"]
    __ENTRY__
    """
)


def _report(stdout: str) -> dict:
    line = next(x for x in stdout.splitlines() if x.startswith("REPORT "))
    return json.loads(line[len("REPORT "):])


@pytest.mark.slow
def test_entry_point_pins_cpu_even_when_env_asks_for_cuda(tmp_path):
    code = _CHILD.replace("__ENTRY__", "import prolix_agent_entry; prolix_agent_entry.cli_main()")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path,
                          env=_env(tmp_path, JAX_PLATFORMS="cuda"), timeout=600, check=False)
    assert proc.returncode == 0, proc.stderr
    rep = _report(proc.stdout)
    assert rep["backends"] == ["cpu"] and rep["default"] == "cpu"


@pytest.mark.slow
def test_control_bypassing_the_entry_package_does_not_pin_cpu(tmp_path):
    code = _CHILD.replace("__ENTRY__", "from prolix.agent.cli import main; main()")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path,
                          env=_env(tmp_path, JAX_PLATFORMS="cuda"), timeout=600, check=False)
    ok_cpu_only = proc.returncode == 0 and "REPORT" in proc.stdout and _report(proc.stdout)["backends"] == ["cpu"]
    assert not ok_cpu_only, "control is vacuous: the poisoned env did not change the backend"


def test_in_process_op_refuses_when_backend_is_not_cpu(monkeypatch, tmp_path):
    import jax

    from prolix.agent import ops

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("PROLIX_AGENT_OUT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RefusalError):
        ops.load_system(out_dir=str(tmp_path / "x"), bundle=object())


def test_async_mcp_busy_refuses_at_once_and_loop_stays_responsive(monkeypatch, tmp_path):
    import threading

    from prolix.agent import _runtime as rt

    release = threading.Event()
    started = threading.Event()

    def blocking_op(**kwargs):
        started.set()
        release.wait(timeout=30)
        return {"ok": True}

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(50):
                ticks += 1
                await asyncio.sleep(0.01)

        first = asyncio.create_task(tools._locked(blocking_op))
        while not started.is_set():
            await asyncio.sleep(0.005)
        tick_task = asyncio.create_task(ticker())
        with pytest.raises(RefusalError, match="busy"):
            await tools._locked(blocking_op)
        release.set()
        result = await first
        await tick_task
        return result, ticks

    result, ticks = asyncio.run(scenario())
    assert result == {"ok": True}
    assert ticks > 5  # the event loop kept running while the op was blocked
    assert not rt.RUN_LOCK.locked()
