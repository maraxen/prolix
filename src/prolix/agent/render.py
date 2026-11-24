"""Job rendering and submission for long runs (spec section 6). No cisternal import.

``render_job`` writes an sbatch script under ``scripts/slurm/agent_jobs/<name>/`` inside the
checkout (not git-ignored, so ``myxcel push`` ships it) and returns the argv of the two myxcel
commands that would ship and submit it. ``submit_job`` runs those commands, dry-run unless
``confirm`` is set and ``PROLIX_AGENT_ALLOW_SUBMIT=1`` is in the process environment (D8).
"""

from __future__ import annotations

import importlib.resources
import json
import os
import re
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from prolix.agent import _runtime as rt
from prolix.agent.errors import InputError, RefusalError

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
TIME_RE = re.compile(r"^(\d{1,2}):([0-5]\d):([0-5]\d)$")
SLURM_ID_RE = re.compile(r"^slurm_id:\s*(\d+)\s*$", re.MULTILINE)
SUBMIT_ENV_GATE = "PROLIX_AGENT_ALLOW_SUBMIT"
ARTIFACT_ROOT = Path("scripts") / "slurm" / "agent_jobs"
PROJECT = "prolix"
DEFAULT_REMOTE = "engaging"

# Non-preemptible presets only (spec section 6): a preempted job restarts from step 0 because the
# target scripts have no checkpoint API.
PRESETS: dict[str, dict[str, Any]] = {
  "cpu_quick": {
    "lines": ["--partition=mit_quicktest", "--cpus-per-task=2", "--mem=4G"],
    "default_time": "0:10:00",
    "max_time": "0:10:00",
  },
  "gpu_l40s": {
    "lines": ["--partition=mit_normal_gpu", "--gres=gpu:l40s:1", "--cpus-per-task=16", "--mem=64G"],
    "default_time": "2:00:00",
    "max_time": "2:00:00",
  },
}


def _seconds(hms: str) -> int:
  m = TIME_RE.match(hms)
  if m is None:
    raise InputError(f"time must be HH:MM:SS, got {hms!r}")
  h, mi, s = (int(x) for x in m.groups())
  return h * 3600 + mi * 60 + s


def load_allowlist() -> list[str]:
  """Repo-relative script paths that ``render_job`` may render (``job_scripts.txt``, package data)."""
  text = (
    importlib.resources.files("prolix.agent")
    .joinpath("job_scripts.txt")
    .read_text(encoding="utf-8")
  )
  entries: list[str] = []
  for raw in text.splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
      continue
    entries.append(line)
  return entries


def find_repo_root(start: Path | None = None) -> Path:
  """Nearest ancestor of ``start`` (default cwd) that is the prolix checkout."""
  here = (start or Path.cwd()).resolve()
  for cand in (here, *here.parents):
    pyproject = cand / "pyproject.toml"
    if pyproject.is_file() and (cand / "scripts" / "slurm" / "_uv_env.sh").is_file():
      try:
        name = tomllib.loads(pyproject.read_text()).get("project", {}).get("name")
      except tomllib.TOMLDecodeError:
        continue
      if name == PROJECT:
        return cand
  raise InputError("no prolix checkout found above the working directory; pass repo_root")


def _resolve_root(repo_root: str | None) -> Path:
  if repo_root is None:
    return find_repo_root()
  root = Path(repo_root).expanduser().resolve()
  if not root.is_dir():
    raise InputError(f"repo_root {repo_root!r} is not a directory")
  return root


def _check_script(script: str, root: Path) -> None:
  if not isinstance(script, str) or not script:
    raise InputError("script must be a non-empty string")
  if script.startswith("/") or "\\" in script:
    raise InputError("script must be a repo-relative POSIX path")
  parts = script.split("/")
  if any(p in ("", ".", "..") for p in parts):
    raise InputError("script must not contain empty, '.' or '..' segments")
  if not script.startswith("scripts/") or not script.endswith(".py"):
    raise InputError("script must be a .py file under scripts/")
  path = root / script
  if not path.is_file():
    raise InputError(f"script {script!r} does not exist in {root}")
  if not path.resolve().is_relative_to(root):
    raise InputError("script resolves outside repo_root (symlink escape)")
  if script not in load_allowlist():
    raise InputError(f"script {script!r} is not in the render allowlist (job_scripts.txt)")


def _check_name(name: str) -> None:
  if not isinstance(name, str) or not NAME_RE.match(name):
    raise InputError("name must match [A-Za-z0-9_-]{1,40}")


def _artifact_dir(root: Path, name: str) -> Path:
  return root / ARTIFACT_ROOT / name


def render_job_script(*, script: str, args: list[str], preset: str, time: str, name: str) -> str:
  """Return the sbatch script text. Pure: no filesystem access."""
  spec = PRESETS[preset]
  folder = f"scripts/slurm/agent_jobs/{name}"
  quoted = " ".join(shlex.quote(a) for a in args)
  cmd = f"uv run python {script}" + (f" {quoted}" if quoted else "")
  header = ["#!/bin/bash", *(f"#SBATCH {line}" for line in spec["lines"])]
  header += [
    f"#SBATCH --time={time}",
    f"#SBATCH --job-name=prolix_{name}",
    f"#SBATCH --output={folder}/%j.{name}.out",
    f"#SBATCH --error={folder}/%j.{name}.err",
  ]
  body = [
    "set -euo pipefail",
    'cd "${SLURM_SUBMIT_DIR:?}"',
    f"mkdir -p {folder}",
    "source scripts/slurm/_uv_env.sh && prolix_uv_sync",
    "rc=0",
    f"{cmd} || rc=$?",
    f'echo "=== {name} exit=${{rc}} ==="',
    'exit "${rc}"',
  ]
  return "\n".join([*header, *body]) + "\n"


def submit_argvs(
  *, remote: str, artifact: str | Path, project: str = PROJECT, confirm: bool = False
) -> list[list[str]]:
  """``[push, submit]`` argvs. Dry-run unless ``confirm``."""
  push = ["myxcel", "push", remote, project] + (["--apply", "--yes"] if confirm else ["--dry-run"])
  submit = ["myxcel", "submit", remote, project, str(artifact), "--respect-directives"]
  if not confirm:
    submit.append("--dry-run")
  return [push, submit]


def _git_rev(root: Path) -> str | None:
  git = shutil.which("git")
  if git is None or not (root / ".git").exists():
    return None
  try:
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
      [git, "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=20, check=True
    )
  except (OSError, subprocess.SubprocessError):
    return None
  return out.stdout.strip()


def render_job(
  *,
  script: str,
  name: str,
  preset: str,
  args: list[str] | None = None,
  time: str | None = None,
  repo_root: str | None = None,
) -> dict[str, Any]:
  """Write ``scripts/slurm/agent_jobs/<name>/job.slurm`` and ``render_job.json``; return argvs and notes."""
  args = list(args or [])
  if not all(isinstance(a, str) for a in args):
    raise InputError("args must be a list of strings")
  _check_name(name)
  if preset not in PRESETS:
    raise InputError(f"preset must be one of {', '.join(PRESETS)}, got {preset!r}")
  spec = PRESETS[preset]
  time = spec["default_time"] if time is None else time
  if _seconds(time) > _seconds(spec["max_time"]):
    raise InputError(f"time {time} is over the {preset} maximum {spec['max_time']}")
  if _seconds(time) == 0:
    raise InputError("time must be positive")
  root = _resolve_root(repo_root)
  _check_script(script, root)
  folder = _artifact_dir(root, name)
  if folder.exists():
    raise InputError(f"{folder} exists; rendered jobs are immutable, choose another name")

  text = render_job_script(script=script, args=args, preset=preset, time=time, name=name)
  job_bytes = text.encode()
  folder.mkdir(parents=True)
  job_path = folder / "job.slurm"
  rt.atomic_write_bytes(job_path, job_bytes)
  record = {
    "schema": "prolix.agent.render_job/1",
    "name": name,
    "script": script,
    "source_sha256": rt.sha256_file(root / script),
    "artifact_sha256": rt.sha256_file(job_path),
    "args": args,
    "preset": preset,
    "time": time,
    "git_rev": _git_rev(root),
    "repo_root": str(root),
  }

  rt.atomic_write_bytes(
    folder / "render_job.json", (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
  )
  argvs = submit_argvs(remote=DEFAULT_REMOTE, artifact=job_path, confirm=False)
  notes = [
    "repo_root must equal the myxcel project root: myxcel maps the script path only under the project root",
    "myxcel push is additive and ships the whole non-ignored tree, not just this script",
    "rendered directories stay untracked; a human deletes them",
    "a dry run proves the artifact pins, the profile mapping and the sbatch line; it does not prove the remote has the file, "
    "that Slurm accepts the partition, gres or time, that the uv env builds, or that the script runs",
  ]
  return {
    "artifact": str(job_path),
    "artifact_sha256": record["artifact_sha256"],
    "render_record": str(folder / "render_job.json"),
    "submit_argvs": argvs,
    "resources": {"preset": preset, "time": time, "lines": spec["lines"]},
    "notes": notes,
  }


def submit_job(
  *,
  name: str,
  remote: str = DEFAULT_REMOTE,
  confirm: bool = False,
  repo_root: str | None = None,
) -> dict[str, Any]:
  """Run ``myxcel push`` then (only on push exit 0) ``myxcel submit`` for a rendered job."""

  _check_name(name)
  root = _resolve_root(repo_root)
  folder = _artifact_dir(root, name)
  record_path = folder / "render_job.json"
  if not record_path.is_file():
    raise RefusalError(f"no render_job.json for {name!r}; render_job first")
  record = json.loads(record_path.read_text())
  job_path = folder / "job.slurm"
  if not job_path.is_file() or rt.sha256_file(job_path) != record["artifact_sha256"]:
    raise RefusalError(
      "rendered job.slurm is missing or its sha256 no longer matches render_job.json"
    )
  if (
    not (root / record["script"]).is_file()
    or rt.sha256_file(root / record["script"]) != record["source_sha256"]
  ):
    raise RefusalError("source script changed since render; re-render under a new name")
  if confirm and os.environ.get(SUBMIT_ENV_GATE) != "1":
    raise RefusalError(
      f"submission is disabled: set {SUBMIT_ENV_GATE}=1 in the server or CLI environment"
    )

  push, submit = submit_argvs(remote=remote, artifact=job_path, confirm=confirm)
  if shutil.which("myxcel") is None:
    raise InputError(
      "myxcel is not on PATH; run these by hand from the repo root:\n  "
      + " ".join(push)
      + "\n  "
      + " ".join(submit)
    )
  env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
  steps: list[dict[str, Any]] = []
  slurm_id: int | None = None
  for argv in (push, submit):
    try:
      proc = subprocess.run(  # noqa: S603 - argv list, no shell
        argv,
        shell=False,
        timeout=120,
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        check=False,
      )
      code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
      code, out, err = -1, exc.stdout or "", f"timed out after 120 s: {exc}"
    steps.append({"argv": argv, "exit": code, "stdout": out, "stderr": err})
    if code != 0:
      break
    if argv is submit:
      match = SLURM_ID_RE.search(out)
      slurm_id = int(match.group(1)) if match and confirm else None
  return {
    "dry_run": not confirm,
    "steps": steps,
    "slurm_id": slurm_id,
    "follow": "use the myxcel MCP tool job_status(remote, slurm_id); prolix adds no status tool",
  }


__all__ = [
  "PRESETS",
  "load_allowlist",
  "render_job",
  "render_job_script",
  "submit_argvs",
  "submit_job",
]
