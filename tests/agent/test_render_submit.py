"""Task 3 gates: golden rendered script, allowlist, refusals, pushed set, stub-myxcel submit.

Renders happen in a temporary copy of the minimal repo layout, never in the checkout. Real
``myxcel`` and ``sbatch`` are never invoked: submit tests put a stub ``myxcel`` first on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from prolix.agent import render
from prolix.agent.errors import InputError, RefusalError

REPO = Path(__file__).resolve().parents[2]
SCRIPT = "scripts/benchmarks/prolix_vs_openmm_speed.py"
GOLDEN = REPO / "tests" / "agent" / "golden" / "job_gpu_l40s.slurm"


@pytest.fixture
def repo(tmp_path):
    """A minimal checkout: pyproject naming prolix, the uv helper, the allowlisted script, a .gitignore."""
    root = tmp_path / "checkout"
    (root / "scripts" / "slurm").mkdir(parents=True)
    (root / "scripts" / "benchmarks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "prolix"\n')
    shutil.copy(REPO / "scripts" / "slurm" / "_uv_env.sh", root / "scripts" / "slurm" / "_uv_env.sh")
    shutil.copy(REPO / SCRIPT, root / SCRIPT)
    shutil.copy(REPO / ".gitignore", root / ".gitignore")
    return root


def _artifact(root: Path, name: str) -> Path:
    return root / "scripts" / "slurm" / "agent_jobs" / name


def _no_artifact(root: Path, name: str) -> bool:
    return not _artifact(root, name).exists()


def test_golden_rendered_script_matches_byte_for_byte(repo):
    res = render.render_job(
        repo_root=str(repo), script=SCRIPT, name="golden_l40s", preset="gpu_l40s",
        time="2:00:00", args=["--sweep", "1000; rm -rf /"],
    )
    rendered = Path(res["artifact"]).read_bytes()
    assert rendered == GOLDEN.read_bytes()
    assert b"--sweep '1000; rm -rf /'" in rendered
    assert b"--output=scripts/slurm/agent_jobs/golden_l40s/%j.golden_l40s.out" in rendered
    assert subprocess.run(["bash", "-n", res["artifact"]], check=False).returncode == 0


def test_rendered_record_pins_hashes_and_argvs(repo):
    res = render.render_job(repo_root=str(repo), script=SCRIPT, name="pin_one", preset="cpu_quick")
    rec = json.loads((_artifact(repo, "pin_one") / "render_job.json").read_text())
    assert rec["artifact_sha256"] == res["artifact_sha256"]
    assert rec["source_sha256"] == render.rt.sha256_file(repo / SCRIPT)
    assert res["submit_argvs"][0][:4] == ["myxcel", "push", "engaging", "prolix"]
    assert res["submit_argvs"][0][-1] == "--dry-run"
    assert res["submit_argvs"][1][-1] == "--dry-run"
    assert "--respect-directives" in res["submit_argvs"][1]


def test_stub_uv_failure_prints_status_line_and_exits_nonzero(repo, tmp_path):
    res = render.render_job(repo_root=str(repo), script=SCRIPT, name="fail_stub", preset="cpu_quick")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "uv").write_text("#!/bin/sh\nexit 3\n")
    (bindir / "uv").chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "SLURM_SUBMIT_DIR": str(repo)}
    proc = subprocess.run(["bash", res["artifact"]], cwd=repo, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "=== fail_stub exit=3 ===" in proc.stdout


def test_allowlist_contents_and_shape():
    entries = render.load_allowlist()
    assert entries == [SCRIPT]
    for entry in entries:
        assert (REPO / entry).is_file()
        assert entry.startswith("scripts/") and entry.endswith(".py")


def test_empty_allowlist_refuses(repo, monkeypatch):
    monkeypatch.setattr(render, "load_allowlist", lambda: [])
    with pytest.raises(InputError):
        render.render_job(repo_root=str(repo), script=SCRIPT, name="nope", preset="cpu_quick")
    assert _no_artifact(repo, "nope")


@pytest.mark.parametrize(
    "script",
    ["/etc/passwd.py", "scripts/../pyproject.toml", "scripts/benchmarks/notes.txt",
     "scripts/benchmarks/_b1_paramize.py", "src/prolix/agent/ops.py"],
)
def test_script_refusals_write_nothing(repo, script):
    with pytest.raises(InputError):
        render.render_job(repo_root=str(repo), script=script, name="refused", preset="cpu_quick")
    assert _no_artifact(repo, "refused")


def test_time_name_preset_and_rerender_refusals(repo):
    with pytest.raises(InputError):
        render.render_job(repo_root=str(repo), script=SCRIPT, name="t1", preset="gpu_l40s", time="12:00:01")
    with pytest.raises(InputError):
        render.render_job(repo_root=str(repo), script=SCRIPT, name="bad name!", preset="cpu_quick")
    with pytest.raises(InputError):
        render.render_job(repo_root=str(repo), script=SCRIPT, name="p1", preset="gpu_h200_preempt")
    assert _no_artifact(repo, "t1") and _no_artifact(repo, "p1")
    render.render_job(repo_root=str(repo), script=SCRIPT, name="once", preset="cpu_quick")
    with pytest.raises(InputError):
        render.render_job(repo_root=str(repo), script=SCRIPT, name="once", preset="cpu_quick")


def test_pushed_set_contains_artifact_and_control_outputs_is_dropped(repo):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not installed")
    subprocess.run([git, "init", "-q"], cwd=repo, check=True)
    res = render.render_job(repo_root=str(repo), script=SCRIPT, name="ship_me", preset="cpu_quick")
    rel = Path(res["artifact"]).relative_to(repo).as_posix()
    # the exact command myxcel push uses (rsync.py:182)
    listed = subprocess.run(
        [git, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=repo, capture_output=True, check=True,
    ).stdout.decode().split("\0")
    assert rel in listed
    # control: the same kind of file under the git-ignored outputs/ tree is not shipped
    ignored = repo / "outputs" / "agent_jobs" / "x"
    ignored.mkdir(parents=True)
    (ignored / "job.slurm").write_text("#!/bin/bash\n")
    listed = subprocess.run(
        [git, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=repo, capture_output=True, check=True,
    ).stdout.decode().split("\0")
    assert "outputs/agent_jobs/x/job.slurm" not in listed


STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "$MYXCEL_STUB_LOG"
if [ "$1" = "push" ]; then exit "${MYXCEL_STUB_PUSH_EXIT:-0}"; fi
if [ "$1" = "submit" ]; then printf '%s\\n' "${MYXCEL_STUB_SUBMIT_OUT:-slurm_id: 4242}"; exit 0; fi
exit 9
"""


@pytest.fixture
def stub_myxcel(tmp_path, monkeypatch):
    bindir = tmp_path / "stubbin"
    bindir.mkdir()
    stub = bindir / "myxcel"
    stub.write_text(STUB)
    stub.chmod(0o755)
    log = tmp_path / "stub.log"
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("MYXCEL_STUB_LOG", str(log))
    monkeypatch.delenv("MYXCEL_STUB_PUSH_EXIT", raising=False)
    monkeypatch.delenv("MYXCEL_STUB_SUBMIT_OUT", raising=False)
    monkeypatch.delenv("PROLIX_AGENT_ALLOW_SUBMIT", raising=False)
    return log


def _calls(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def test_submit_dry_run_is_the_default(repo, stub_myxcel):
    render.render_job(repo_root=str(repo), script=SCRIPT, name="dry", preset="cpu_quick")
    res = render.submit_job(name="dry", repo_root=str(repo))
    assert res["dry_run"] is True
    assert res["slurm_id"] is None
    assert len(res["steps"]) == 2
    calls = _calls(stub_myxcel)
    assert calls[0].startswith("push engaging prolix --dry-run")
    assert calls[1].endswith("--respect-directives --dry-run")


def test_confirm_without_env_gate_is_refused(repo, stub_myxcel):
    render.render_job(repo_root=str(repo), script=SCRIPT, name="gate", preset="cpu_quick")
    with pytest.raises(RefusalError):
        render.submit_job(name="gate", repo_root=str(repo), confirm=True)
    assert _calls(stub_myxcel) == []


def test_confirm_with_env_gate_runs_push_then_submit(repo, stub_myxcel, monkeypatch):
    render.render_job(repo_root=str(repo), script=SCRIPT, name="live", preset="cpu_quick")
    monkeypatch.setenv("PROLIX_AGENT_ALLOW_SUBMIT", "1")
    res = render.submit_job(name="live", repo_root=str(repo), confirm=True)
    calls = _calls(stub_myxcel)
    assert calls[0] == "push engaging prolix --apply --yes"
    assert "--dry-run" not in calls[1]
    assert res["slurm_id"] == 4242


def test_failed_push_stops_before_submit(repo, stub_myxcel, monkeypatch):
    render.render_job(repo_root=str(repo), script=SCRIPT, name="pushfail", preset="cpu_quick")
    monkeypatch.setenv("PROLIX_AGENT_ALLOW_SUBMIT", "1")
    monkeypatch.setenv("MYXCEL_STUB_PUSH_EXIT", "5")
    res = render.submit_job(name="pushfail", repo_root=str(repo), confirm=True)
    assert len(res["steps"]) == 1
    assert res["steps"][0]["exit"] == 5
    assert len(_calls(stub_myxcel)) == 1


def test_slurm_id_parse_yields_null_on_other_text(repo, stub_myxcel, monkeypatch):
    render.render_job(repo_root=str(repo), script=SCRIPT, name="noid", preset="cpu_quick")
    monkeypatch.setenv("PROLIX_AGENT_ALLOW_SUBMIT", "1")
    monkeypatch.setenv("MYXCEL_STUB_SUBMIT_OUT", "submitted ok, no id here")
    res = render.submit_job(name="noid", repo_root=str(repo), confirm=True)
    assert res["slurm_id"] is None


def test_missing_myxcel_raises_with_both_argvs(repo, tmp_path, monkeypatch):
    render.render_job(repo_root=str(repo), script=SCRIPT, name="nomyx", preset="cpu_quick")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(InputError) as info:
        render.submit_job(name="nomyx", repo_root=str(repo))
    assert "myxcel push" in str(info.value) and "myxcel submit" in str(info.value)


def test_submit_refusals(repo, stub_myxcel):
    with pytest.raises(RefusalError):
        render.submit_job(name="never_rendered", repo_root=str(repo))
    render.render_job(repo_root=str(repo), script=SCRIPT, name="tamper", preset="cpu_quick")
    (_artifact(repo, "tamper") / "job.slurm").write_text("#!/bin/bash\necho evil\n")
    with pytest.raises(RefusalError):
        render.submit_job(name="tamper", repo_root=str(repo))
    assert _calls(stub_myxcel) == []
